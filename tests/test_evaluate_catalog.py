"""Catalog evaluation isolation and accounting, using only synthetic volumes."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from tests.test_evaluate_counting_cohort import cohort as legacy_cohort, FAKE_WORKER, git

spec = importlib.util.spec_from_file_location("evaluate_catalog", Path(__file__).resolve().parents[1] / "scripts/evaluate_catalog.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture
def manifest(tmp_path):
    entries = []
    for label, split in [("Case003", "regression"), ("Case007", "acceptance")]:
        path = tmp_path / f"{label}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.int16), np.eye(4)), path)
        digest = evaluation.checks.sha256(path)
        entries.append(dict(study_label=label, series_label="Series001", split=split,
            nifti=path.name, source_sha256=digest, automated_qc="PASS",
            acquisition={"source_sha256": digest, "slice_thickness_mm": 2.5}))
    path = tmp_path / "manifest.local.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    return path, entries


def test_regression_does_not_open_acceptance_volume(manifest, tmp_path):
    path, entries = manifest
    (tmp_path / entries[1]["nifti"]).unlink()
    assert evaluation.load_manifest(path, tmp_path, "regression") == entries[:1]
    with pytest.raises(evaluation.checks.EvaluationError):
        evaluation.load_manifest(path, tmp_path, "acceptance")


@pytest.mark.parametrize("fault", ["duplicate", "escape", "split", "acquisition", "qc", "source_hash"])
def test_manifest_rejects_identity_source_and_split_faults(manifest, tmp_path, fault):
    path, entries = manifest
    if fault == "duplicate":
        entries.append(deepcopy(entries[0]))
    elif fault == "escape":
        entries[0]["nifti"] = "../outside.nii.gz"
    elif fault == "split":
        entries[1]["split"] = "regression"
    elif fault == "acquisition":
        entries[0]["acquisition"]["source_sha256"] = "0" * 64
    elif fault == "qc":
        entries[0]["automated_qc"] = "FAIL"
    else:
        (tmp_path / entries[0]["nifti"]).write_bytes(b"changed")
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    with pytest.raises(evaluation.checks.EvaluationError):
        evaluation.load_manifest(path, tmp_path, "all")


def test_resolution_summary_does_not_count_failures_or_claim_accuracy():
    summary = evaluation.summarize([
        dict(study_label="Case003", slice_thickness_mm=0.625, status="completed", elapsed_seconds=3,
             diagnostics={"uncertain_candidates": 7}),
        dict(study_label="Case003", slice_thickness_mm=2.5, status="failed", elapsed_seconds=2,
             failure_code="SPHENOID_UNRESOLVED"),
        dict(study_label="Case007", slice_thickness_mm=None, status="failed", elapsed_seconds=1,
             failure_code="WORKER_TIMEOUT")])
    assert summary["attempted"] == 3
    assert summary["studies"] == 2
    assert summary["completed"] == 1 and summary["withheld"] == 2
    assert summary["by_thickness_mm"]["2.5"]["completed"] == 0
    assert summary["uncertain_candidates"] == 7
    assert summary["accuracy_measured"] is False


def test_failed_worker_timeout_terminates_process_tree(tmp_path, monkeypatch):
    import subprocess
    class Process:
        pid = 123456
        waits = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1: raise subprocess.TimeoutExpired("synthetic", timeout)
            return -1
    terminated = []
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(evaluation.subprocess, "run", lambda command, **k: terminated.append(command))
    monkeypatch.setattr(evaluation.os, "killpg", lambda *a: terminated.append(a), raising=False)
    with pytest.raises(evaluation.checks.EvaluationError, match="WORKER_TIMEOUT"):
        evaluation.worker(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, None, tmp_path / "worker.log", 1)
    assert terminated


@pytest.fixture
def runnable(tmp_path, monkeypatch):
    options, old_manifest, algorithm = legacy_cohort.__wrapped__(tmp_path, monkeypatch)
    pipeline, work = options["pipeline_root"], options["work_root"]
    package = pipeline / "src/ethmoid_pipeline/counting"
    worker = FAKE_WORKER.replace("'config','landmarks'", "'config','landmarks','acquisition'")
    worker = worker.replace("ethmoid-count-prototype-v1", "ethmoid-count-prototype-v2")
    worker = worker.replace("fault=os.environ", "from ethmoid_pipeline.counting.quality import acquisition_snapshot,resolution_quality\nresult['acquisition']=acquisition_snapshot(json.loads(Path(a.acquisition).read_text()),result['source_sha256'])\nresult['quality']=resolution_quality(result['acquisition'],image.affine)\nfault=os.environ")
    worker = worker.replace("(destination/'result.json')", """if fault=='acquisition_spacing': result['acquisition']['spacing_mm']=[0.4,0.4,0.625]
if fault=='acquisition_class': result['acquisition']['resolution_class']='within_protocol'
if fault=='acquisition_identity': result['acquisition']['study_label']='Case999'
if fault=='acquisition_source': result['acquisition']['source_sha256']='0'*64
if fault=='quality_missing': result.pop('quality')
if fault=='quality_class': result['quality']['resolution_class']='within_protocol'
if fault=='quality_flags': result['quality']['flags']=[]
if fault=='quality_spacing': result['quality']['native_spacing_mm']=[1.,1.,1.]
(destination/'result.json')""")
    (package / "worker.py").write_text(worker)
    (package / "quality.py").write_bytes((Path(__file__).resolve().parents[1] / "src/ethmoid_pipeline/counting/quality.py").read_bytes())
    (package / "protocol.json").write_text(json.dumps({"protocol_version": "ethmoid-count-prototype-v2", "algorithm": algorithm}))
    git(pipeline, "add", ".")
    git(pipeline, "commit", "-q", "-m", "Synthetic catalog worker")
    entries = []
    for entry in old_manifest["entries"][:2]:
        entries.append({"study_label": entry["case"], "series_label": entry["series"],
            "split": "regression", "nifti": entry["nifti"], "source_sha256": entry["nifti_sha256"],
            "automated_qc": "PASS", "acquisition": {"source_sha256": entry["nifti_sha256"],
                "study_label": entry["case"], "series_label": entry["series"],
                "slice_thickness_mm": 2.5, "spacing_mm": [0.6, 0.7, 0.8], "resolution_class": "outside_protocol"}})
    path = work / "counting/catalog-manifest.local.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: {"synthetic_test_only": True})
    return {**options, "split": "regression", "manifest": path,
        "expected_code_sha": git(pipeline, "rev-parse", "HEAD"),
        "expected_manifest_sha256": evaluation.checks.sha256(path),
        "expected_protocol_sha256": evaluation.checks.sha256(package / "protocol.json")}


def test_real_subprocess_attempts_record_success_and_abstention(runnable, monkeypatch):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    result = evaluation.evaluate(**runnable)
    assert result["attempted"] == 2 and result["completed"] == 1 and result["withheld"] == 1
    assert result["failure_codes"] == {"TARGET_CLIPPED": 1}
    failed = evaluation.checks.read_json(runnable["output"] / "Case002_Series001/attempt.local.json")
    assert failed["counts"] == evaluation.checks.NULL_COUNTS
    with pytest.raises(evaluation.checks.EvaluationError, match="IMMUTABLE"):
        evaluation.evaluate(**runnable)


@pytest.mark.parametrize("fault", ["false_count_failure", "source_mutation", "code_mutation", "config_mutation", "missing_result"])
def test_subprocess_corruption_never_becomes_a_completed_count(runnable, monkeypatch, fault):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    result = evaluation.evaluate(**runnable)
    assert result["completed"] == 0 and result["withheld"] == 2
    assert result["accuracy_measured"] is False


@pytest.mark.parametrize("field", ["expected_code_sha", "expected_manifest_sha256", "expected_protocol_sha256"])
def test_freeze_mismatch_cannot_start_worker(runnable, field):
    runnable[field] = "0" * 64
    with pytest.raises(evaluation.checks.EvaluationError, match="FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)
    assert not runnable["output"].exists()


@pytest.mark.parametrize("fault", ["acquisition_spacing", "acquisition_class", "acquisition_identity",
    "acquisition_source", "quality_missing", "quality_class", "quality_flags", "quality_spacing"])
def test_subprocess_acquisition_and_quality_must_match_frozen_source(runnable, monkeypatch, fault):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    result = evaluation.evaluate(**runnable)
    expected = "WORKER_ACQUISITION_MISMATCH" if fault.startswith("acquisition") else "WORKER_QUALITY_MISMATCH"
    assert result["attempted"] == result["withheld"] == 2
    assert result["completed"] == 0
    assert result["failure_codes"] == {expected: 2}


def test_later_missing_source_is_recorded_without_losing_summary(runnable, monkeypatch):
    original = evaluation.worker
    entries = evaluation.checks.read_json(runnable["manifest"])["entries"]
    later = runnable["work_root"] / entries[1]["nifti"]
    def remove_later(*args, **kwargs):
        result = original(*args, **kwargs)
        later.unlink()
        return result
    monkeypatch.setattr(evaluation, "worker", remove_later)
    result = evaluation.evaluate(**runnable)
    assert result["attempted"] == 2 and result["completed"] == 1 and result["withheld"] == 1
    failed = evaluation.checks.read_json(runnable["output"] / "Case002_Series001/attempt.local.json")
    assert failed["failure_code"] == "PROTECTED_PATH_REQUIRED"
    assert failed["counts"] == evaluation.checks.NULL_COUNTS
    assert failed["quality"]["resolution_class"] == "outside_protocol"
    assert failed["quality"]["native_spacing_mm"] is None
    assert failed["acquisition"]["slice_thickness_mm"] == 2.5
    assert evaluation.checks.read_json(runnable["output"] / "summary.local.json") == result
