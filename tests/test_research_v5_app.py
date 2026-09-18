"""Opt-in research workflow contracts using synthetic CT and verified-shape results."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import zipfile

import nibabel as nib
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ethmoid_pipeline.counting.protocol import ADAPTIVE_PROTOCOL_VERSION, load_counting_protocol
from ethmoid_pipeline.counting.research_v5_policy import METHOD_VERSION, PROTOCOL_VERSION
from ethmoid_pipeline.util import repo_root, sha256_canonical_json, sha256_file
from ethmoid_pipeline.webapp.counting_routes import counting_router
from ethmoid_pipeline.webapp.counting_service import ARTIFACTS, CountingService, RESEARCH_STATUS
from ethmoid_pipeline.webapp.service import WebAppError
from tests.test_adaptive_result_provenance import adaptive_executor
from tests.test_count_retry_lineage import assisted_executor, technical_failure
from tests.test_counting_revisions import finish, setup_service, synthetic_executor
from tests.test_webapp_service import write_nifti


def research_metadata(source, protocol):
    return {"schema_version": 1, "source_sha256": sha256_file(source), "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION, "analysis_mode": "automatic", "assistance": None,
        "research_status": RESEARCH_STATUS,
        "model_identity": {"localization": "research_geometric_baseline", "direct_ethmoid_model": "not_used"},
        "config": protocol["algorithm"], "acquisition": protocol["acquisition"], "quality": protocol["source_quality"],
        "scan_profile": protocol["scan_profile"], "processing_plan": protocol["processing_plan"]}


def research_executor(source, output, config):
    synthetic_executor(source, output, config)
    protocol = json.loads(config.read_text())
    region = nib.load(output / ARTIFACTS["region"])
    nib.save(nib.Nifti1Image(np.asarray(region.dataobj).astype(np.uint8), region.affine), output / ARTIFACTS["region"])
    path = output / "result.json"
    result = json.loads(path.read_text())
    result.update(research_metadata(source, protocol), outcome="automatic_candidate_count",
        artifact_verification={"status": "verified", "source_sha256": sha256_file(source),
            "sha256": {key: sha256_file(output / name) for key, name in ARTIFACTS.items()}})
    path.write_text(json.dumps(result))


def research_failure(source, output, config, *, code="LOCALIZER_FAILED", outcome="technical_error"):
    protocol = json.loads(config.read_text())
    output.mkdir()
    (output / "result.json").write_text(json.dumps({**research_metadata(source, protocol),
        "status": "failed", "failure_code": code, "outcome": outcome,
        "counts": {"left": None, "right": None, "total": None}, "cells": [], "artifacts": {}}))


@pytest.fixture
def research_service(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHMOID_AUTOMATIC_COUNT_METHOD", "research-v5")
    app, case, old = setup_service(tmp_path)
    old.close()
    service = CountingService(app, executor=research_executor)
    try:
        yield app, case, service
    finally:
        service.close()


def test_default_is_v3_and_opt_in_is_operator_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ETHMOID_AUTOMATIC_COUNT_METHOD", raising=False)
    app, case, old = setup_service(tmp_path)
    old.close()
    service = CountingService(app, executor=adaptive_executor)
    try:
        assert service.protocol()["protocol_version"] == ADAPTIVE_PROTOCOL_VERSION
        api = FastAPI()
        api.include_router(counting_router(service))
        with TestClient(api) as client:
            response = client.post(f"/api/cases/{case['case_id']}/count-runs", json={"method": "research-v5"})
            assert response.status_code == 422
        assert service.list_runs(case["case_id"])["runs"] == []
        service._automatic_method = "research-v4"
        with pytest.raises(WebAppError, match="unavailable"):
            service.protocol()
        with pytest.raises(ValueError, match="Unsupported"):
            load_counting_protocol(PROTOCOL_VERSION)
    finally:
        service.close()


def test_opt_in_publishes_native_research_result_and_export(research_service):
    app, case, service = research_service
    run = service.create_run(case["case_id"])
    done = finish(service, case["case_id"], run["run_id"])
    assert done["status"] == "completed", done
    assert done["counts"] == {"left": 1, "right": 1, "total": 2}
    assert done["method_version"] == METHOD_VERSION and done["protocol_version"] == PROTOCOL_VERSION
    assert done["research_status"] == RESEARCH_STATUS and done["training_eligibility"] == "NOT_FOR_TRAINING"
    assert done["model_identity"]["direct_ethmoid_model"] == "not_used"
    assert done["processing_plan"]["version"] == "research-localization-plan-2"
    source = nib.load(app.volume_path(case["case_id"]))
    mask = nib.load(service.artifact(case["case_id"], run["run_id"], "cavities"))
    assert mask.shape == source.shape and np.array_equal(mask.affine, source.affine)
    assert set(np.unique(mask.dataobj)) == {0, 1, 300}
    with zipfile.ZipFile(service.export(case["case_id"], run["run_id"])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["research_status"] == RESEARCH_STATUS
        assert manifest["training_eligibility"] == "NOT_FOR_TRAINING"
        assert manifest["clinical_accuracy"] == "UNMEASURED"
        assert manifest["processing_plan"] == done["processing_plan"]
        assert manifest["model_identity"] == done["model_identity"]
        exported_protocol = json.loads(archive.read("protocol.json"))
        assert exported_protocol["protocol_version"] == PROTOCOL_VERSION
        assert done["protocol_sha256"] == manifest["protocol_sha256"] == exported_protocol["sha256"]
        assert done["protocol_snapshot_sha256"] == manifest["protocol_snapshot_sha256"] == sha256_canonical_json(exported_protocol)
        assert manifest["exported_protocol_sha256"] == hashlib.sha256(archive.read("protocol.json")).hexdigest()
        assert manifest["exported_protocol_hash_scope"] == "exact_protocol_json_bytes_in_archive"
        assert b"Research v5 automatic candidate result" in archive.read("summary.pdf")
        for record in csv.DictReader(io.StringIO(archive.read("counts.csv").decode())):
            assert record["research_status"] == RESEARCH_STATUS and record["method_version"] == METHOD_VERSION
        assert str(app.root).encode() not in archive.read("manifest.json")


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "0" * 64), ("protocol_version", ADAPTIVE_PROTOCOL_VERSION),
    ("method_version", "research-native-0.4.0"), ("config", {"air_hu": -499}),
    ("scan_profile", {}), ("processing_plan", {}), ("acquisition", {}),
    ("model_identity", {"localization": "promoted-model"}), ("research_status", "VALIDATED"),
    ("assistance", {"revision_id": "unexpected"}), ("outcome", "assisted_candidate_count"),
    ("artifact_verification", {"status": "verified"}),
])
def test_worker_cannot_change_research_authority(research_service, field, value):
    _, case, service = research_service
    def tampered(source, output, config):
        research_executor(source, output, config)
        path = output / "result.json"
        result = json.loads(path.read_text())
        result[field] = value
        path.write_text(json.dumps(result))
    service._executor = tampered
    run = service.create_run(case["case_id"])
    rejected = finish(service, case["case_id"], run["run_id"])
    assert rejected["status"] == "failed" and rejected["counts"] is None and rejected["cells"] == []
    assert rejected["outcome"] == "technical_error" and rejected["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"
    with pytest.raises(WebAppError, match="no completed result"):
        service.artifact(case["case_id"], run["run_id"], "cavities")


@pytest.mark.parametrize("field", ["algorithm", "sha256", "scan_profile", "processing_plan", "acquisition", "assistance"])
def test_queued_research_snapshot_must_match_frozen_source_and_policy(research_service, field):
    app, case, service = research_service
    run = service.create_run(case["case_id"])
    assert finish(service, case["case_id"], run["run_id"])["status"] == "completed"
    protocol = json.loads(service._run(case["case_id"], run["run_id"])["protocol_json"])
    if field == "algorithm":
        protocol[field]["air_hu"] = -499
    elif field == "sha256":
        protocol[field] = "0" * 64
    elif field == "acquisition":
        protocol[field]["source_sha256"] = "0" * 64
    else:
        protocol[field] = {}
    with pytest.raises(WebAppError):
        service._validate_research_snapshot(protocol, app.volume_path(case["case_id"]))


@pytest.mark.parametrize("code,outcome", [("ORBIT_WALL_UNRESOLVED", "needs_review"), ("RESOURCE_LIMIT", "technical_error")])
def test_research_failure_withholds_counts_but_retains_method_context(research_service, code, outcome):
    _, case, service = research_service
    service._executor = lambda *args: research_failure(*args, code=code, outcome=outcome)
    run = service.create_run(case["case_id"])
    failed = finish(service, case["case_id"], run["run_id"])
    assert failed["status"] == "failed" and failed["outcome"] == outcome and failed["failure_code"] == code
    assert failed["counts"] is None and failed["cells"] == []
    assert failed["research_status"] == RESEARCH_STATUS and failed["method_version"] == METHOD_VERSION
    assert failed["scan_profile"] and failed["processing_plan"]
    with pytest.raises(WebAppError, match="no completed result"):
        service.export(case["case_id"], run["run_id"])


def test_pre_localization_hu_failure_retains_queued_profile_and_reason(research_service):
    _, case, service = research_service
    def early_failure(source, output, config):
        research_failure(source, output, config, code="HU_RANGE_INVALID")
        path = output / "result.json"
        result = json.loads(path.read_text())
        result.update(scan_profile={}, processing_plan={})
        path.write_text(json.dumps(result))
    service._executor = early_failure
    run = service.create_run(case["case_id"])
    failed = finish(service, case["case_id"], run["run_id"])
    assert failed["failure_code"] == "HU_RANGE_INVALID"
    assert failed["status"] == "failed" and failed["outcome"] == "technical_error"
    assert failed["counts"] is None and failed["cells"] == []
    assert failed["scan_profile"] == run["scan_profile"] and failed["scan_profile"]
    assert failed["processing_plan"] == run["processing_plan"] and failed["processing_plan"]


@pytest.mark.parametrize("failure_code", [None, "LOCALIZER_FAILED", "RESOURCE_LIMIT", "SPHENOID_UNRESOLVED", "ENGINE_ERROR"])
def test_only_specific_early_failure_can_have_empty_profile(research_service, failure_code):
    _, case, service = research_service
    def tampered(source, output, config):
        if failure_code is None:
            research_executor(source, output, config)
        else:
            research_failure(source, output, config, code=failure_code,
                outcome="needs_review" if failure_code == "SPHENOID_UNRESOLVED" else "technical_error")
        path = output / "result.json"
        result = json.loads(path.read_text())
        result.update(scan_profile={}, processing_plan={})
        path.write_text(json.dumps(result))
    service._executor = tampered
    run = service.create_run(case["case_id"])
    rejected = finish(service, case["case_id"], run["run_id"])
    assert rejected["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"
    assert rejected["counts"] is None and rejected["cells"] == []


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64), ("config", {}), ("acquisition", {}),
    ("protocol_version", ADAPTIVE_PROTOCOL_VERSION), ("quality", {}), ("counts", {"left": 0, "right": 0, "total": 0}),
    ("cells", [{"id": 1}]), ("artifacts", ARTIFACTS), ("scan_profile", None)])
def test_early_hu_failure_still_requires_bindings_and_empty_outputs(research_service, field, value):
    _, case, service = research_service
    def tampered(source, output, config):
        research_failure(source, output, config, code="HU_RANGE_INVALID")
        path = output / "result.json"
        result = json.loads(path.read_text())
        result.update(scan_profile={}, processing_plan={})
        result[field] = value
        path.write_text(json.dumps(result))
    service._executor = tampered
    run = service.create_run(case["case_id"])
    rejected = finish(service, case["case_id"], run["run_id"])
    assert rejected["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"
    assert rejected["counts"] is None and rejected["cells"] == []


@pytest.mark.integration
def test_actual_cpu_worker_preserves_pre_localization_hu_failure(research_service, tmp_path):
    python = os.environ.get("ETHMOID_ENGINE_TEST_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("Set ETHMOID_ENGINE_TEST_PYTHON to the isolated CPU counting runtime")
    app, _, service = research_service
    source = write_nifti(tmp_path / "uniform.nii.gz", np.zeros((8, 9, 6), dtype=np.int16))
    case = app.import_nifti(source, case_label="Case902")
    observed = []
    def execute(source, output, config):
        snapshot = json.loads(config.read_text())
        acquisition, plan = config.with_name("acquisition.json"), config.with_name("plan.json")
        acquisition.write_text(json.dumps({**snapshot["acquisition"], **snapshot["profile_characteristics"]}))
        plan.write_text(json.dumps(snapshot["processing_plan"]))
        process = subprocess.run([python, "-m", "ethmoid_pipeline.counting.worker", "--input", str(source),
            "--output", str(output), "--method", "research-v5", "--config", str(config),
            "--acquisition", str(acquisition), "--processing-plan", str(plan)],
            env={**os.environ, "PYTHONPATH": str(repo_root() / "src"), "CUDA_VISIBLE_DEVICES": "", "ETHMOID_INFERENCE_OFFLINE": "1"},
            capture_output=True, timeout=30)
        assert process.returncode == 2, process.stdout.decode() + process.stderr.decode()
        observed.append(json.loads((output / "result.json").read_text()))
        assert not (output / "localizers").exists()
    service._executor = execute
    run = service.create_run(case["case_id"])
    failed = finish(service, case["case_id"], run["run_id"])
    assert len(observed) == 1
    assert observed[0]["failure_code"] == "HU_RANGE_INVALID"
    assert observed[0]["scan_profile"] == observed[0]["processing_plan"] == {}
    assert failed["failure_code"] == "HU_RANGE_INVALID", failed
    assert failed["outcome"] == "technical_error" and failed["counts"] is None and failed["cells"] == []
    assert failed["scan_profile"] == run["scan_profile"] and failed["processing_plan"] == run["processing_plan"]


@pytest.mark.parametrize("field,value", [("counts", {"left": 0, "right": 0, "total": 0}), ("cells", [{"id": 1}]), ("artifacts", ARTIFACTS), ("source_sha256", "0" * 64)])
def test_unverified_failure_cannot_publish_partial_outputs(research_service, field, value):
    _, case, service = research_service
    def tampered(source, output, config):
        research_failure(source, output, config)
        path = output / "result.json"
        result = json.loads(path.read_text())
        result[field] = value
        path.write_text(json.dumps(result))
    service._executor = tampered
    run = service.create_run(case["case_id"])
    rejected = finish(service, case["case_id"], run["run_id"])
    assert rejected["counts"] is None and rejected["cells"] == []
    assert rejected["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"


@pytest.mark.parametrize("initial_method,new_default", [("research-v5", "adaptive-v3"), ("adaptive-v3", "research-v5")])
def test_retry_preserves_method_snapshot_and_history_when_default_changes(research_service, initial_method, new_default):
    app, case, service = research_service
    service._automatic_method = initial_method
    service._executor = research_failure if initial_method == "research-v5" else technical_failure
    first = service.create_run(case["case_id"])
    assert finish(service, case["case_id"], first["run_id"])["status"] == "failed"
    original = dict(service._run(case["case_id"], first["run_id"]))
    service._automatic_method = new_default
    service._executor = research_executor if initial_method == "research-v5" else adaptive_executor
    retried = service.retry(case["case_id"], first["run_id"], "same-method-retry")
    done = finish(service, case["case_id"], retried["run_id"])
    assert done["status"] == "completed", done
    assert done["protocol_version"] == first["protocol_version"]
    assert service._run(case["case_id"], retried["run_id"])["protocol_json"] == original["protocol_json"]
    assert dict(service._run(case["case_id"], first["run_id"])) == original


@pytest.mark.parametrize("initial_method,new_method", [("research-v5", "adaptive-v3"), ("adaptive-v3", "research-v5")])
def test_idempotency_key_cannot_silently_switch_method(research_service, initial_method, new_method):
    _, case, service = research_service
    service._automatic_method = initial_method
    service._executor = research_executor if initial_method == "research-v5" else adaptive_executor
    first = service.create_run(case["case_id"], "fixed-method-key")
    assert finish(service, case["case_id"], first["run_id"])["status"] == "completed"
    service._automatic_method = new_method
    with pytest.raises(WebAppError) as error:
        service.create_run(case["case_id"], "fixed-method-key")
    assert error.value.status_code == 409
    assert len(service.list_runs(case["case_id"])["runs"]) == 1


def test_v5_parent_can_receive_v3_assisted_count_without_rewriting_history(research_service, tmp_path):
    app, case, service = research_service
    service._executor = research_failure
    parent = service.create_run(case["case_id"])
    assert finish(service, case["case_id"], parent["run_id"])["status"] == "failed"
    original = dict(service._run(case["case_id"], parent["run_id"]))
    source = nib.load(app.volume_path(case["case_id"]))
    region = np.zeros(source.shape, dtype=np.uint8)
    region[1:3, 2:5, 1:4] = 1
    region[5:7, 2:4, 2:4] = 2
    path = tmp_path / "regions.nii.gz"
    nib.save(nib.Nifti1Image(region, source.affine), path)
    boundary = service.save_boundary(case["case_id"], parent["run_id"], path)
    service._executor = assisted_executor
    assisted = service.create_run(case["case_id"], boundary_id=boundary["boundary_id"], parent_run_id=parent["run_id"])
    done = finish(service, case["case_id"], assisted["run_id"])
    assert done["status"] == "completed", done
    assert done["protocol_version"] == ADAPTIVE_PROTOCOL_VERSION
    assert done["outcome"] == "assisted_candidate_count" and done["research_status"] is None
    assert done["assistance"]["parent_run_id"] == parent["run_id"]
    assert dict(service._run(case["case_id"], parent["run_id"])) == original
    assisted_row = dict(service._run(case["case_id"], done["run_id"]))
    private_snapshot = json.loads(assisted_row["protocol_json"])
    with zipfile.ZipFile(service.export(case["case_id"], done["run_id"])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        protocol_bytes = archive.read("protocol.json")
        exported = json.loads(protocol_bytes)
        assert manifest["exported_protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
        assert manifest["protocol_snapshot_sha256"] == sha256_canonical_json(private_snapshot)
        assert manifest["protocol_snapshot_hash_scope"] == "canonical_private_queued_snapshot_before_path_redaction"
        assert sha256_canonical_json(exported) != manifest["protocol_snapshot_sha256"]
        assert "region_path" not in exported["assistance"] and "exclusion_path" not in exported["assistance"]
        assert str(tmp_path).encode() not in protocol_bytes
    assert dict(service._run(case["case_id"], done["run_id"])) == assisted_row


def test_worker_dispatch_uses_frozen_method_and_offline_runtime(research_service, tmp_path, monkeypatch):
    import ethmoid_pipeline.webapp.counting_service as module
    app, case, service = research_service
    run = service.create_run(case["case_id"])
    assert finish(service, case["case_id"], run["run_id"])["status"] == "completed"
    snapshot = json.loads(service._run(case["case_id"], run["run_id"])["protocol_json"])
    directory = tmp_path / "dispatch"
    directory.mkdir()
    config = directory / "protocol.json"
    config.write_text(json.dumps(snapshot))
    runtime_python = tmp_path / "EthmoidComplexity/tools/counting-runtime/Scripts/python.exe"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    captured = []
    class FinishedProcess:
        returncode = 0
        def poll(self): return 0
        def close(self): pass
    def launch(command, **kwargs):
        captured.append((command, kwargs))
        return FinishedProcess()
    monkeypatch.setattr(module, "launch_owned_process", launch)
    service._automatic_method = "adaptive-v3"
    service._execute_local(app.volume_path(case["case_id"]), directory / "output", config)
    command, kwargs = captured[0]
    assert command[command.index("--method") + 1] == "research-v5"
    assert "--processing-plan" in command and "--assistance" not in command
    assert kwargs["env"]["ETHMOID_INFERENCE_OFFLINE"] == "1"
    assert json.loads((directory / "processing-plan.json").read_text()) == snapshot["processing_plan"]


def test_restart_retains_v5_checkpoint_when_operator_returns_to_v3(research_service, monkeypatch):
    app, case, first = research_service
    started, release = threading.Event(), threading.Event()
    def execute(source, output, config):
        started.set()
        assert release.wait(5)
        research_executor(source, output, config)
    first._executor = execute
    run = first.create_run(case["case_id"])
    assert started.wait(5)
    closing = threading.Thread(target=first.close)
    closing.start()
    assert first._closed.wait(2)
    release.set()
    closing.join(5)
    assert not closing.is_alive()
    monkeypatch.delenv("ETHMOID_AUTOMATIC_COUNT_METHOD")
    def never(*args):
        pytest.fail("A verified research checkpoint must not be recomputed")
    second = CountingService(app, executor=never)
    try:
        assert second.protocol()["protocol_version"] == ADAPTIVE_PROTOCOL_VERSION
        done = finish(second, case["case_id"], run["run_id"])
        assert done["status"] == "completed" and done["protocol_version"] == PROTOCOL_VERSION
    finally:
        second.close()
