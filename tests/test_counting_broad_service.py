"""Synthetic upgrade and source-quality integration, without model inference."""
import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.webapp.counting_service import CountingService, PROTOCOL_VERSION
from ethmoid_pipeline.webapp.service import WebAppError
from tests.test_counting_revisions import setup_service, finish, synthetic_executor


def test_catalog_thickness_is_frozen_separately_from_interval_and_exported(tmp_path, monkeypatch):
    app, case, service = setup_service(tmp_path)
    context = {"study_id": "study-test", "study_label": "Case001", "series_id": "series-test", "series_label": "Series004",
               "slice_thickness_mm": 2.5, "spacing_mm": [0.5, 0.5, 0.625], "resolution_class": "outside_protocol", "warnings": ["Do not export arbitrary text"]}
    monkeypatch.setattr(app, "catalog_context", lambda _: context, raising=False)
    try:
        run = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], run["run_id"])
        assert result["status"] == "completed"
        assert result["quality"]["slice_thickness_mm"] == 2.5
        assert result["quality"]["slice_interval_mm"] == 0.625
        assert result["quality"]["out_of_protocol_resolution"] is True
        assert all("out_of_protocol_resolution" in cell["reasons"] for cell in result["cells"])
        context["slice_thickness_mm"] = 0.625
        assert service.get_run(case["case_id"], run["run_id"])["quality"]["slice_thickness_mm"] == 2.5
        with zipfile.ZipFile(service.export(case["case_id"], run["run_id"])) as archive:
            assert json.loads(archive.read("manifest.json"))["quality"] == result["quality"]
            assert b"outside_protocol,2.5,0.625,out_of_protocol_resolution" in archive.read("counts.csv")
            assert b"OUT-OF-PROTOCOL EXPERIMENTAL ATTEMPT" in archive.read("summary.pdf")
            assert b"Do not export arbitrary text" not in archive.read("summary.pdf")
    finally:
        service.close()


def test_failed_attempt_retains_resolution_warnings_without_count(tmp_path, monkeypatch):
    def failed(source, output, config):
        output.mkdir()
        (output / "result.json").write_text(json.dumps({"status": "failed", "failure_code": "SPHENOID_UNRESOLVED"}))
    app, case, service = setup_service(tmp_path, failed)
    monkeypatch.setattr(app, "catalog_context", lambda _: {"slice_thickness_mm": 2.5, "spacing_mm": [0.5, 0.5, 2.5], "resolution_class": "outside_protocol"}, raising=False)
    try:
        run = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], run["run_id"])
        assert result["status"] == "failed" and result["failure_code"] == "SPHENOID_UNRESOLVED"
        assert result["counts"] is None and result["cells"] == []
        assert result["quality"]["resolution_class"] == "outside_protocol"
        assert "Out-of-protocol" in result["warnings"][0]
    finally:
        service.close()


def test_worker_cannot_override_server_acquisition_facts(tmp_path):
    def contradictory(source, output, config):
        synthetic_executor(source, output, config)
        path = output / "result.json"
        result = json.loads(path.read_text())
        result["quality"] = {"resolution_class": "within_protocol"}
        path.write_text(json.dumps(result))
    app, case, service = setup_service(tmp_path, contradictory)
    try:
        run = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], run["run_id"])
        assert result["status"] == "failed"
        assert result["quality"]["slice_thickness_mm"] is None
        assert result["quality"]["resolution_class"] == "unknown"
    finally:
        service.close()


def test_v1_result_remains_readable_correctable_and_exportable_after_v2_upgrade(tmp_path):
    app, case, service = setup_service(tmp_path)
    run = service.create_run(case["case_id"])
    assert finish(service, case["case_id"], run["run_id"])["status"] == "completed"
    row = service._run(case["case_id"], run["run_id"])
    result, directory, _ = service._snapshot(row, 0)
    # Construct an old persisted fixture. No production migration changes any
    # historical file; this test checks compatibility with that earlier schema.
    legacy_protocol = {"protocol_version": "ethmoid-count-prototype-v1", "algorithm": {"air_hu": -700.0}}
    result["protocol_version"] = legacy_protocol["protocol_version"]
    result["config"] = legacy_protocol["algorithm"]
    result.pop("quality", None)
    result.pop("acquisition", None)
    (directory / "result.json").write_text(json.dumps(result))
    legacy_hashes = service._hashes(directory)
    with app.store.connect() as connection:
        connection.execute("UPDATE count_runs SET protocol_version=?,protocol_json=?,original_result_json=?,original_hashes_json=? WHERE id=?", (
            legacy_protocol["protocol_version"], json.dumps(legacy_protocol), json.dumps(result), json.dumps(legacy_hashes), run["run_id"]))
    service.close()
    upgraded = CountingService(app, executor=synthetic_executor)
    try:
        assert upgraded.protocol()["protocol_version"] == PROTOCOL_VERSION == "ethmoid-count-prototype-v2"
        read = upgraded.get_run(case["case_id"], run["run_id"])
        assert read["protocol_version"] == "ethmoid-count-prototype-v1"
        source = nib.load(app.volume_path(case["case_id"]))
        selection = np.zeros(source.shape, np.uint8)
        selection[0, 2, 2] = 1  # Native HU=-600: air in v2 defaults, not this saved v1 run.
        mask = tmp_path / "manual.nii.gz"
        nib.save(nib.Nifti1Image(selection, source.affine), mask)
        corrected = upgraded.revise(case["case_id"], run["run_id"], 0, {"type": "add", "side": "left"}, mask)
        added = max(corrected["cells"], key=lambda cell: cell["id"])
        assert added["air_voxel_count"] == 0 and added["voxel_count"] == 1
        assert upgraded._hashes(directory) == legacy_hashes
        with zipfile.ZipFile(upgraded.export(case["case_id"], run["run_id"])) as archive:
            assert json.loads(archive.read("protocol.json")) == legacy_protocol
            assert json.loads(archive.read("manifest.json"))["protocol_version"] == "ethmoid-count-prototype-v1"
            assert json.loads(archive.read("reviewed/result.json"))["config"]["air_hu"] == -700.0
    finally:
        upgraded.close()


def test_stale_catalog_blocks_new_count_but_preserves_historical_read_and_export(tmp_path, monkeypatch):
    app, case, service = setup_service(tmp_path)
    try:
        run = service.create_run(case["case_id"])
        assert finish(service, case["case_id"], run["run_id"])["status"] == "completed"
        def stale(_):
            raise WebAppError("The selected catalog source is no longer current.", status_code=409)
        monkeypatch.setattr(app, "assert_catalog_countable", stale, raising=False)
        with pytest.raises(WebAppError) as failure:
            service.create_run(case["case_id"])
        assert failure.value.status_code == 409
        assert service.get_run(case["case_id"], run["run_id"])["status"] == "completed"
        assert service.export(case["case_id"], run["run_id"]).is_file()
    finally:
        service.close()


def test_catalog_is_rechecked_when_a_queued_attempt_starts(tmp_path, monkeypatch):
    executed = []
    app, case, service = setup_service(tmp_path, lambda *args: executed.append(True))
    checks = []
    def current_only_at_enqueue(_):
        checks.append(True)
        if len(checks) > 1:
            raise WebAppError("The selected catalog source changed.", status_code=409)
    monkeypatch.setattr(app, "assert_catalog_countable", current_only_at_enqueue, raising=False)
    try:
        run = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], run["run_id"])
        assert result["status"] == "failed" and result["counts"] is None
        assert len(checks) == 2 and executed == []
    finally:
        service.close()
