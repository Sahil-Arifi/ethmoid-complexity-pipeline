from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.dicom import PreflightFailure
from ethmoid_pipeline.webapp.catalog_queue import PreparationQueue
from ethmoid_pipeline.webapp.service import WebAppError
from tests.test_catalog import catalog, write_verified_case


def selected(service):
    return service.list_studies()["studies"][0]["series"][0]


def drain(queue):
    while (job := queue._claim()) is not None:
        queue.process(job)


def test_explicit_request_promotes_queued_without_duplicate_or_running_reset(catalog):
    service, _, _ = catalog
    service.sync()
    queue = PreparationQueue(service)
    queue.enqueue_all()
    sid = service.list_studies()["studies"][1]["series"][0]["series_id"]
    queue.enqueue(sid, explicit=True)
    claimed = queue._claim()
    assert claimed["series_id"] == sid and claimed["priority"] == -100
    queue.enqueue(sid, explicit=True)
    with service.store.connect() as connection:
        rows = connection.execute("SELECT * FROM preparation_jobs WHERE series_id=?", (sid,)).fetchall()
    assert len(rows) == 1 and rows[0]["status"] == "preparing" and rows[0]["attempt"] == 1


def test_restart_keeps_unchanged_ready_without_hu_rerun(catalog, tmp_path, monkeypatch):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    queue = PreparationQueue(service)
    monkeypatch.setattr(queue, "_existing", lambda *a: pytest.fail("Ready volume must not re-run HU validation"))
    queue.enqueue_all()
    service.sync()
    queue.enqueue_all()
    assert selected(service)["case_id"] == case["case_id"]
    assert selected(service)["preparation_status"] == "ready"
    with service.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM preparation_jobs WHERE series_id=?", (selected(service)["series_id"],)).fetchone()[0] == 0


def test_changed_cached_volume_is_invalidated_and_queued(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    volume = service.app.volume_path(case["case_id"])
    volume.write_bytes(volume.read_bytes() + b"changed")
    PreparationQueue(service).enqueue(selected(service)["series_id"], startup=True)
    assert selected(service)["preparation_status"] == "queued"
    assert selected(service)["case_id"] is None
    assert service.app.get_case(case["case_id"])["case_id"] == case["case_id"]


def test_opening_corrupted_volume_exposes_retry_and_updates_recommendation(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    volume = service.app.volume_path(case["case_id"])
    volume.write_bytes(volume.read_bytes() + b"changed")
    with pytest.raises(WebAppError, match="must be prepared again"):
        service.app.volume_path(case["case_id"])
    study = service.list_studies()["studies"][0]
    assert selected(service)["preparation_status"] == "unavailable"
    assert selected(service)["retryable"] is True
    assert study["recommended_series_id"] is None


def test_new_counts_rehash_raw_source_but_history_remains_accessible(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    service.app.assert_catalog_countable(case["case_id"])
    native[0].write_bytes(native[0].read_bytes() + b"changed")
    with pytest.raises(WebAppError, match="source files changed"):
        service.app.assert_catalog_countable(case["case_id"])
    assert service.app.get_case(case["case_id"])["case_id"] == case["case_id"]
    assert service.app.volume_path(case["case_id"]).is_file()
    assert selected(service)["preparation_status"] == "unavailable"
    assert selected(service)["retryable"] is True
    assert service.list_studies()["studies"][0]["recommended_series_id"] is None
    with pytest.raises(WebAppError, match="retained volume"):
        service.app.assert_catalog_countable(case["case_id"])


def test_disk_failure_is_durable_and_only_retried_explicitly(catalog, monkeypatch):
    service, _, _ = catalog
    service.sync()
    queue = PreparationQueue(service)
    sid = selected(service)["series_id"]
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 100, 0))
    queue.enqueue(sid)
    queue.process(queue._claim())
    summary = selected(service)
    assert summary["preparation_status"] == "unavailable" and summary["retryable"] is True
    assert "insufficient free space" in summary["reason"] and summary["counts"] is None
    queue.enqueue(sid, startup=True)
    assert queue._claim() is None
    queue.enqueue(sid, explicit=True)
    retry = queue._claim()
    assert retry["attempt"] == 2


@pytest.mark.parametrize("issue,code", [
    ("multiple AcquisitionNumber groups are present", "MIXED_ACQUISITION"),
    ("ImageOrientationPatient changes within the series", "ORIENTATION_INVALID"),
    ("duplicate or near-duplicate projected slice positions are present", "DUPLICATE_PLANES"),
    ("a likely missing-plane gap is present", "MISSING_PLANES"),
    ("RescaleType is not explicitly HU", "HU_UNCALIBRATED"),
    ("could not parse SYNTHETIC_PRIVATE_VALUE", "METADATA_INVALID"),
    ("quantized MPR physical corner residual exceeds the 0.001 mm limit", "GEOMETRY_RECONCILIATION_UNRESOLVED"),
])
def test_preflight_failure_retains_fixed_structured_limitations(catalog, monkeypatch, issue, code):
    service, _, _ = catalog
    service.sync()
    import ethmoid_pipeline.webapp.catalog_queue as module
    def fail(*args, **kwargs):
        raise PreflightFailure([issue])
    monkeypatch.setattr(module, "preflight_series", fail)
    queue = PreparationQueue(service)
    queue.enqueue(selected(service)["series_id"])
    job = queue._claim()
    queue.process(job)
    report = service.app.root / "preparation" / job["id"] / "attempt-0001/preflight-failure.local.json"
    value = json.loads(report.read_text())
    assert value["issues"][0]["code"] == code
    assert "SYNTHETIC_PRIVATE_VALUE" not in report.read_text()
    assert selected(service)["reason"] == value["issues"][0]["message"]
    assert selected(service)["retryable"] is False and selected(service)["counts"] is None


def test_unresolved_serialized_geometry_has_specific_nonretryable_diagnostics(catalog,monkeypatch):
    service,_,_=catalog;service.sync()
    import ethmoid_pipeline.webapp.catalog_queue as module
    def fail(*args,**kwargs):
        raise module.GeometryReconciliationFailure("Synthetic geometry rejection",diagnostics={"stage":"reloaded_geometry","form":"qform","max_world_residual_mm":.0011,"physical_residual_limit_mm":.001})
    monkeypatch.setattr(module,"convert_series",fail)
    queue=PreparationQueue(service);queue.enqueue(selected(service)["series_id"]);job=queue._claim();queue.process(job)
    result=json.loads((service.app.root/'preparation'/job['id']/'attempt-0001/geometry-failure.local.json').read_text())
    assert result['geometry_diagnostics']['max_world_residual_mm']==.0011
    assert selected(service)['retryable'] is False and selected(service)['preparation_status']=='unavailable'


def test_single_process_owner_and_restart_recovers_interrupted_job(catalog):
    service, _, _ = catalog
    service.sync()
    first, second = PreparationQueue(service), PreparationQueue(service)
    assert first._acquire() and not second._acquire()
    first._lock_file.close()
    first._lock_file = None
    second.enqueue(selected(service)["series_id"])
    job = second._claim()
    # Freeze the loop; startup must recover the previous process's active job.
    second._stop.set()
    second.start(enqueue_all=False)
    second.close()
    with service.store.connect() as connection:
        row = connection.execute("SELECT * FROM preparation_jobs WHERE id=?", (job["id"],)).fetchone()
    assert row["status"] == "queued" and row["error_code"] == "INTERRUPTED"
    assert selected(service)["preparation_status"] == "queued"


def test_catalog_cli_indexes_isolated_inventory_with_aggregate_only_output(catalog):
    service, _, _ = catalog
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_local_catalog.py"
    completed = subprocess.run([sys.executable, str(script), "--work-root", str(service.config.work_root), "--catalog-only"], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0
    value = json.loads(completed.stdout)
    assert value["catalog"]["total_studies"] == 2 and value["catalog"]["total_series"] == 3
    assert value["inference_started"] is False
    assert str(service.config.source_root) not in completed.stdout and "SYNTHETIC" not in completed.stdout


def test_exact_verified_cohort_is_copied_without_converter(catalog, tmp_path, monkeypatch):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    volume = service.app.volume_path(case["case_id"])
    retained = service.config.work_root / "fixture-cohort"
    shutil.copytree(volume.parent, retained)
    source = retained / volume.name
    with service.store.connect() as connection:
        connection.execute("DELETE FROM cases WHERE id=?", (case["case_id"],))
    cohort = service.config.work_root / "counting/cohort.v1.local.json"
    cohort.parent.mkdir()
    cohort.write_text(json.dumps({"entries": [{"case": "Case099", "series": "Series001", "nifti": str(source.relative_to(service.config.work_root)), "nifti_sha256": sha256_file(source)}]}))
    service.sync()
    import ethmoid_pipeline.webapp.catalog_queue as module
    monkeypatch.setattr(module, "convert_series", lambda *a, **k: pytest.fail("Verified cohort must be reused"))
    queue = PreparationQueue(service)
    queue.enqueue(selected(service)["series_id"])
    queue.process(queue._claim())
    result = selected(service)
    assert result["preparation_status"] == "ready" and result["case_id"] != case["case_id"]
    assert sha256_file(service.app.volume_path(result["case_id"])) == sha256_file(source)
    service.app.assert_catalog_countable(result["case_id"])


@pytest.mark.integration
def test_preparation_runs_real_converter_and_all_voxel_verification_without_inference(catalog):
    executable = default_dcm2niix()
    if not executable.is_file():
        pytest.skip("Pinned dcm2niix is not installed")
    service, native, _ = catalog
    service.app.dcm2niix = executable
    before = [sha256_file(path) for path in native]
    service.sync()
    queue = PreparationQueue(service)
    queue.enqueue_all()
    drain(queue)
    studies = service.list_studies()["studies"]
    assert [s["preparation_status"] for t in studies for s in t["series"]] == ["ready", "unavailable", "ready"]
    assert [sha256_file(path) for path in native] == before
    for study in studies:
        for series in study["series"]:
            assert series["count_status"] == "not_run" and series["counts"] is None
            if series["case_id"]:
                case = service.app.get_case(series["case_id"])
                assert case["validation_status"] == "VERIFIED"
                assert case["validation_details"]["metrics"]["max_absolute_hu_error"] == 0
    with service.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM count_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 2
    queue.enqueue_all()
    assert queue._claim() is None
