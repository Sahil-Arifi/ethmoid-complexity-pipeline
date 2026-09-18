"""Retry provenance is distinct from source-bound anatomical assistance."""
import json
import threading

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.counting.profiles import sealed
from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.counting_service import CountingService
from ethmoid_pipeline.webapp.service import WebAppError
from tests.test_adaptive_result_provenance import adaptive_executor
from tests.test_counting_revisions import finish, setup_service


def technical_failure(source, output, config):
    output.mkdir()
    (output / "result.json").write_text(json.dumps({
        "status": "failed", "failure_code": "LOCALIZER_FAILED",
        "counts": {"left": None, "right": None, "total": None},
        "cells": [], "artifacts": {},
    }))


@pytest.fixture
def adaptive_service(tmp_path):
    app, case, old = setup_service(tmp_path)
    old.close()
    service = CountingService(app, executor=technical_failure)
    try:
        yield app, case, service
    finally:
        service.close()


def failed_run(service, case):
    run = service.create_run(case["case_id"])
    failed = finish(service, case["case_id"], run["run_id"])
    assert failed["outcome"] == "technical_error"
    return failed


def test_automatic_technical_failure_can_retry_to_verified_count(tmp_path):
    calls = []

    def execute(source, output, config):
        calls.append(1)
        if len(calls) == 1:
            technical_failure(source, output, config)
        else:
            adaptive_executor(source, output, config)

    app, case, old = setup_service(tmp_path)
    old.close()
    service = CountingService(app, executor=execute)
    try:
        first = service.create_run(case["case_id"])
        failed = finish(service, case["case_id"], first["run_id"])
        assert failed["outcome"] == "technical_error"
        assert failed["counts"] is None
        original_row = dict(service._run(case["case_id"], first["run_id"]))
        parent_directory = app._path(original_row["output_relpath"])
        original_files = {file: sha256_file(file) for file in parent_directory.rglob("*") if file.is_file()}
        retried = service.retry(case["case_id"], first["run_id"], "retry-synthetic")
        completed = finish(service, case["case_id"], retried["run_id"])
        assert completed["status"] == "completed", completed
        assert completed["counts"] == {"left": 1, "right": 1, "total": 2}
        assert completed["outcome"] == "automatic_candidate_count"
        assert completed["analysis_mode"] == "automatic" and completed["assistance"] is None
        assert retried["run_id"] != first["run_id"]
        assert len(calls) == 2
        with app.store.connect() as db:
            job = db.execute("SELECT parent_run_id,assistance_id FROM count_jobs WHERE run_id=?", (retried["run_id"],)).fetchone()
        assert dict(job) == {"parent_run_id": first["run_id"], "assistance_id": None}
        assert dict(service._run(case["case_id"], first["run_id"])) == original_row
        assert {file: sha256_file(file) for file in parent_directory.rglob("*") if file.is_file()} == original_files
        assert service.retry(case["case_id"], first["run_id"], "retry-synthetic")["run_id"] == retried["run_id"]
        assert len(service.list_runs(case["case_id"])["runs"]) == 2
    finally:
        service.close()


@pytest.mark.parametrize("invalid", ["missing", "wrong_case", "wrong_source", "completed", "assisted_job", "assisted_protocol"])
def test_invalid_automatic_parent_is_rejected_before_queuing(adaptive_service, invalid):
    app, case, service = adaptive_service
    parent = failed_run(service, case)
    parent_id = parent["run_id"]
    if invalid == "missing":
        parent_id = "missing-parent"
    elif invalid == "wrong_case":
        case = app.import_nifti(app.volume_path(case["case_id"]), case_label="Case902")
    else:
        with app.store.connect() as db:
            if invalid == "wrong_source":
                db.execute("UPDATE count_runs SET source_sha256=? WHERE id=?", ("0" * 64, parent_id))
            elif invalid == "completed":
                db.execute("UPDATE count_runs SET status='completed' WHERE id=?", (parent_id,))
            elif invalid == "assisted_job":
                db.execute("UPDATE count_jobs SET assistance_id='unexpected-boundary' WHERE run_id=?", (parent_id,))
            else:
                protocol = json.loads(service._run(case["case_id"], parent_id, db)["protocol_json"])
                protocol["assistance"] = {"revision_id": "unexpected-boundary"}
                db.execute("UPDATE count_runs SET protocol_json=? WHERE id=?", (json.dumps(protocol), parent_id))
    with pytest.raises(WebAppError):
        service.create_run(case["case_id"], parent_run_id=parent_id)
    with app.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM count_runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM count_jobs WHERE state!='done'").fetchone()[0] == 0


def test_retry_endpoint_membership_and_changed_source_are_rejected(adaptive_service):
    app, case, service = adaptive_service
    parent = failed_run(service, case)
    other = app.import_nifti(app.volume_path(case["case_id"]), case_label="Case902")
    with pytest.raises(WebAppError, match="not found"):
        service.retry(other["case_id"], parent["run_id"])
    with app.store.connect() as db:
        db.execute("UPDATE count_runs SET source_sha256=? WHERE id=?", ("0" * 64, parent["run_id"]))
    with pytest.raises(WebAppError, match="same source"):
        service.retry(case["case_id"], parent["run_id"])
    assert len(service.list_runs(other["case_id"])["runs"]) == 0


@pytest.mark.parametrize("request_kind", ["different_retry_parent", "ordinary_automatic"])
def test_retry_idempotency_key_cannot_be_reused_for_different_parent(adaptive_service, request_kind):
    app, case, service = adaptive_service
    first_parent = failed_run(service, case)
    second_parent = failed_run(service, case)
    retry = service.retry(case["case_id"], first_parent["run_id"], "bound-retry-key")
    finish(service, case["case_id"], retry["run_id"])
    assert service.retry(case["case_id"], first_parent["run_id"], "bound-retry-key")["run_id"] == retry["run_id"]
    with pytest.raises(WebAppError) as conflict:
        if request_kind == "different_retry_parent":
            service.retry(case["case_id"], second_parent["run_id"], "bound-retry-key")
        else:
            service.create_run(case["case_id"], "bound-retry-key")
    assert conflict.value.status_code == 409
    with app.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM count_runs").fetchone()[0] == 3
        job = db.execute("SELECT parent_run_id,assistance_id FROM count_jobs WHERE run_id=?", (retry["run_id"],)).fetchone()
    assert dict(job) == {"parent_run_id": first_parent["run_id"], "assistance_id": None}


@pytest.mark.parametrize("invalid", ["wrong_case", "wrong_source", "assisted_parent", "unexpected_assistance"])
def test_publication_rechecks_retry_parent_and_assistance(adaptive_service, invalid):
    app, case, service = adaptive_service
    parent = failed_run(service, case)
    other = app.import_nifti(app.volume_path(case["case_id"]), case_label="Case902")
    other_parent = failed_run(service, other)
    started, release = threading.Event(), threading.Event()

    def execute(source, output, config):
        adaptive_executor(source, output, config)
        if invalid == "unexpected_assistance":
            path = output / "result.json"
            result = json.loads(path.read_text())
            result["assistance"] = {"parent_run_id": parent["run_id"]}
            path.write_text(json.dumps(result))
        started.set()
        assert release.wait(5)

    service._executor = execute
    retry = service.retry(case["case_id"], parent["run_id"])
    try:
        assert started.wait(5)
        with app.store.connect() as db:
            if invalid == "wrong_case":
                db.execute("UPDATE count_jobs SET parent_run_id=? WHERE run_id=?", (other_parent["run_id"], retry["run_id"]))
            elif invalid == "wrong_source":
                db.execute("UPDATE count_runs SET source_sha256=? WHERE id=?", ("0" * 64, parent["run_id"]))
            elif invalid == "assisted_parent":
                db.execute("UPDATE count_jobs SET assistance_id='unexpected-boundary' WHERE run_id=?", (parent["run_id"],))
    finally:
        release.set()
    rejected = finish(service, case["case_id"], retry["run_id"])
    assert rejected["status"] == "failed" and rejected["counts"] is None
    assert rejected["cells"] == [] and rejected["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"
    assert rejected["outcome"] == "technical_error"
    with pytest.raises(WebAppError, match="no completed result"):
        service.artifact(case["case_id"], retry["run_id"], "cavities")


def assisted_executor(source, output, config):
    adaptive_executor(source, output, config)
    protocol = json.loads(config.read_text())
    assistance = protocol["assistance"]
    fixed = {key: assistance[key] for key in ("schema_version", "source_sha256", "revision_id", "parent_run_id", "region_sha256")}
    path = output / "result.json"
    result = json.loads(path.read_text())
    result.update(analysis_mode="assisted", outcome="assisted_candidate_count",
        model_identity={"localization": "reviewer_native_region", "direct_ethmoid_model": "unavailable_without_promoted_weights"},
        assistance=sealed({**fixed, "exclusion_sha256": None, "midline_ras_x": -10.25}))
    path.write_text(json.dumps(result))


@pytest.mark.parametrize("tampered", [None, "parent_run_id", "source_sha256", "revision_id"])
def test_assisted_retry_preserves_original_boundary_authority(adaptive_service, tmp_path, tampered):
    app, case, service = adaptive_service
    parent = failed_run(service, case)
    image = nib.load(app.volume_path(case["case_id"]))
    region = np.zeros(image.shape, dtype=np.uint8)
    region[1:3, 2:5, 1:4] = 1
    region[5:7, 2:4, 2:4] = 2
    path = tmp_path / "boundary.nii.gz"
    nib.save(nib.Nifti1Image(region, image.affine), path)
    boundary = service.save_boundary(case["case_id"], parent["run_id"], path)
    initial = service.create_run(case["case_id"], boundary_id=boundary["boundary_id"], parent_run_id=parent["run_id"])
    initial = finish(service, case["case_id"], initial["run_id"])
    assert initial["status"] == "failed"
    original_row = dict(service._run(case["case_id"], initial["run_id"]))

    def execute(source, output, config):
        assisted_executor(source, output, config)
        if tampered:
            path = output / "result.json"
            result = json.loads(path.read_text())
            result["assistance"][tampered] = "wrong-value"
            result["assistance"] = sealed(result["assistance"])
            path.write_text(json.dumps(result))

    service._executor = execute
    retry = service.retry(case["case_id"], initial["run_id"])
    finished = finish(service, case["case_id"], retry["run_id"])
    assert finished["assistance"] == {"boundary_id": boundary["boundary_id"], "parent_run_id": parent["run_id"]}
    if tampered:
        assert finished["status"] == "failed" and finished["counts"] is None
        assert finished["failure_code"] == "INVALID_OR_UNAVAILABLE_RESULT"
    else:
        assert finished["status"] == "completed", finished
        assert finished["outcome"] == "assisted_candidate_count"
        assert finished["counts"] == {"left": 1, "right": 1, "total": 2}
        assert finished["processing_plan"]["localization_backend"] == "reviewer_native_region"
    assert dict(service._run(case["case_id"], initial["run_id"])) == original_row
