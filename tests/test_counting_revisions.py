from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.counting_service import ARTIFACTS, PROTOCOL_VERSION, CountingService
from ethmoid_pipeline.webapp.service import AppService, WebAppError
from ethmoid_pipeline.webapp.store import Store
from tests.test_webapp_service import write_nifti


def test_presentation_exports_match_current_revision(completed):
    _, case, service, run = completed
    corrected = service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 300})
    assert corrected["counts"]["total"] == 1
    with zipfile.ZipFile(service.export(case["case_id"], run["run_id"], 1)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read("summary.pdf").startswith(b"%PDF-")
        assert archive.read("representative.png").startswith(b"\x89PNG\r\n\x1a\n")
        assert b"original,0,1,1,2" in archive.read("counts.csv")
        assert b"reviewed,1,1,0,1" in archive.read("counts.csv")
        assert manifest["report_sha256"] == {name: hashlib.sha256(archive.read(name)).hexdigest() for name in ("summary.pdf", "representative.png")}


PROTOCOL = {"protocol_version": PROTOCOL_VERSION, "algorithm": {"air_hu": -500.0}}


def synthetic_executor(source_path: Path, output: Path, config_path: Path) -> None:
    """Deterministic synthetic instances; never a model/clinical accuracy fixture."""
    source = nib.load(source_path)
    output.mkdir(parents=True)
    cavities = np.zeros(source.shape, dtype=np.uint16)
    cavities[1:3, 2:5, 1:4] = 1
    cavities[5:7, 2:4, 2:4] = 300
    air = np.where(np.asarray(source.dataobj) < -500, cavities, 0).astype(np.uint16)
    region = np.where(cavities == 1, 1, np.where(cavities == 300, 2, 0)).astype(np.uint16)
    for key, data in (("cavities", cavities), ("air", air), ("region", region)):
        image = nib.Nifti1Image(data, source.affine)
        image.set_qform(source.affine, code=1)
        image.set_sform(source.affine, code=1)
        nib.save(image, output / ARTIFACTS[key])
    cells = []
    for identity, side in ((1, "left"), (300, "right")):
        cells.append({
            "id": identity, "side": side,
            "centroid_ras_mm": nib.affines.apply_affine(source.affine, np.argwhere(cavities == identity).mean(axis=0)).tolist(),
            "origin": "automatic", "included": True, "uncertain": True,
            "reasons": ["Synthetic fixture only"],
            "voxel_count": int(np.count_nonzero(cavities == identity)),
            "air_voxel_count": int(np.count_nonzero(air == identity)),
        })
    result = {
        "schema_version": 1, "status": "completed", "method_version": "synthetic-fixture",
        "source_sha256": sha256_file(source_path), "protocol_version": PROTOCOL_VERSION,
        "config": json.loads(config_path.read_text())["algorithm"], "counts": {"left": 1, "right": 1, "total": 2},
        "cells": cells, "warnings": ["Synthetic output; no accuracy claim."], "artifacts": ARTIFACTS,
    }
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")


def setup_service(tmp_path: Path, executor=synthetic_executor):
    app = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused")
    hu = np.full((8, 9, 6), 800, dtype=np.int16)
    hu[1:3, 2:5, 1:4] = -900
    hu[5:7, 2:4, 2:4] = 35
    # Unassigned air/opaque patches permit testing real-HU correction behavior.
    hu[3, 2:5, 1:4] = -600
    hu[0, 2, 2] = -600
    hu[0, 3, 2] = 35
    source = write_nifti(tmp_path / "source.nii.gz", hu)
    case = app.import_nifti(source, case_label="Case901")
    service = CountingService(app, executor=executor, protocol=PROTOCOL)
    return app, case, service


def finish(service: CountingService, case_id: str, run_id: str):
    service._pool.submit(lambda: None).result(timeout=10)
    return service.get_run(case_id, run_id)


@pytest.fixture
def completed(tmp_path: Path):
    app, case, service = setup_service(tmp_path)
    run = service.create_run(case["case_id"], "test-request")
    run = finish(service, case["case_id"], run["run_id"])
    assert run["status"] == "completed"
    yield app, case, service, run
    service.close()


def test_schema_one_migration_preserves_cases_and_legacy_annotations(tmp_path: Path):
    app, case, service = setup_service(tmp_path)
    service.close()
    legacy = write_nifti(tmp_path / "legacy.nii.gz", np.ones((8, 9, 6), dtype=np.uint8))
    app.save_annotation(case["case_id"], legacy)
    with app.store.connect() as connection:
        connection.execute("DROP TABLE cell_revisions")
        connection.execute("DROP TABLE count_runs")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
    Store(app.root)
    assert app.get_case(case["case_id"])["latest_revision"] == 1
    assert nib.load(app.latest_annotation_path(case["case_id"])).get_data_dtype() == np.dtype("uint8")
    backups = list(app.root.glob("schema-v1-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "1"
        assert backup.execute("SELECT COUNT(*) FROM annotation_revisions").fetchone()[0] == 1
    Store(app.root)
    assert len(list(app.root.glob("schema-v1-*.sqlite3"))) == 1


def test_no_landmark_or_legacy_annotation_needed_and_ids_above_255_survive(completed):
    app, case, service, run = completed
    assert app.get_case(case["case_id"])["latest_revision"] is None
    assert run["counts"] == {"left": 1, "right": 1, "total": 2}
    assert run["latest_revision"] == 0
    path = service.artifact(case["case_id"], run["run_id"], "cavities")
    image = nib.load(path)
    assert image.get_data_dtype() == np.dtype("uint16")
    assert 300 in np.unique(np.asarray(image.dataobj))
    opaque = next(c for c in run["cells"] if c["id"] == 300)
    assert opaque["air_voxel_count"] == 0 and opaque["included"]


def test_empty_automatic_output_is_unavailable_not_a_zero_count(tmp_path: Path):
    def empty_executor(source, output, config):
        synthetic_executor(source, output, config)
        path = output / "result.json"
        result = json.loads(path.read_text())
        result.update(cells=[], counts={"left": 0, "right": 0, "total": 0})
        path.write_text(json.dumps(result))
        for name in ARTIFACTS.values():
            image = nib.load(output / name)
            nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint16), image.affine), output / name)

    _, case, service = setup_service(tmp_path, empty_executor)
    try:
        run = service.create_run(case["case_id"])
        finished = finish(service, case["case_id"], run["run_id"])
        assert finished["status"] == "failed"
        assert finished["counts"] is None and finished["original_counts"] is None
    finally:
        service.close()


@pytest.mark.parametrize("correct_physical_sides", [True, False])
def test_laterality_uses_ras_direction_not_voxel_index_or_scanner_zero(tmp_path: Path, correct_physical_sides: bool):
    def reversed_axis_executor(source, output, config):
        synthetic_executor(source, output, config)
        if correct_physical_sides:
            path = output / "result.json"
            result = json.loads(path.read_text())
            for cell in result["cells"]:
                cell["side"] = "right" if cell["id"] == 1 else "left"
            path.write_text(json.dumps(result))
            region_path = output / "region.nii.gz"
            image = nib.load(region_path)
            values = np.asarray(image.dataobj)
            corrected = np.where(values == 1, 2, np.where(values == 2, 1, 0)).astype(np.uint16)
            nib.save(nib.Nifti1Image(corrected, image.affine), region_path)
    app = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused")
    volume = write_nifti(tmp_path / "reversed.nii.gz")
    source = nib.load(volume)
    affine = source.affine.copy()
    affine[0, 0] = -0.5
    affine[0, 3] = 50.0  # Both sides have positive X; zero is not the midline.
    image = nib.Nifti1Image(np.asarray(source.dataobj), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, volume)
    case = app.import_nifti(volume, case_label="Case904")
    service = CountingService(app, executor=reversed_axis_executor, protocol=PROTOCOL)
    try:
        run = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], run["run_id"])
        assert result["status"] == ("completed" if correct_physical_sides else "failed")
        if correct_physical_sides:
            left = next(c for c in result["cells"] if c["side"] == "left")
            right = next(c for c in result["cells"] if c["side"] == "right")
            assert 0 < left["centroid_ras_mm"][0] < right["centroid_ras_mm"][0]
    finally:
        service.close()


def test_original_prediction_immutable_after_manual_opaque_add_and_remove(completed):
    _, case, service, run = completed
    original = service.artifact(case["case_id"], run["run_id"], "cavities")
    original_hash = sha256_file(original)
    point = run["cells"][0]["centroid_ras_mm"]
    edited = service.revise(case["case_id"], run["run_id"], 0, {
        "type": "add", "side": "left", "centroid_ras_mm": point,
        "uncertain": True, "reasons": ["Manual opaque-cell hypothesis"],
    })
    added = next(c for c in edited["cells"] if c["id"] == 301)
    assert added["origin"] == "manual" and added["voxel_count"] == added["air_voxel_count"] == 0
    assert edited["counts"]["total"] == 3 and edited["original_counts"]["total"] == 2
    edited = service.revise(case["case_id"], run["run_id"], 1, {"type": "remove", "id": 301})
    assert edited["counts"]["total"] == 2 and edited["original_counts"]["total"] == 2
    assert sha256_file(original) == original_hash
    assert service.get_run(case["case_id"], run["run_id"], revision=0)["counts"]["total"] == 2
    edited = service.revise(case["case_id"], run["run_id"], 2, {"type": "add", "side": "left", "centroid_ras_mm": point})
    assert any(c["id"] == 302 for c in edited["cells"])


def test_split_merge_lineage_counts_and_no_id_reuse(completed, tmp_path: Path):
    app, case, service, run = completed
    selected = np.zeros((8, 9, 6), dtype=np.uint8)
    selected[1, 2:5, 1:4] = 1
    mask = write_nifti(tmp_path / "selection.nii.gz", selected)
    split = service.revise(case["case_id"], run["run_id"], 0, {"type": "split", "id": 1}, mask)
    assert split["counts"] == {"left": 2, "right": 1, "total": 3}
    assert split["lineage"][-1]["created_ids"] == [301, 302]
    assert next(c for c in split["cells"] if c["id"] == 1)["retired"]
    merged = service.revise(case["case_id"], run["run_id"], 1, {"type": "merge", "ids": [301, 302]})
    assert merged["counts"]["total"] == 2
    assert merged["lineage"][-1]["created_ids"] == [303]
    path = service.artifact(case["case_id"], run["run_id"], "air", None)
    assert set(np.unique(np.asarray(nib.load(path).dataobj))) == {0, 303}
    with pytest.raises(WebAppError, match="non-retired"):
        service.revise(case["case_id"], run["run_id"], 2, {"type": "update", "id": 1, "included": True})
    app.get_case(case["case_id"])


def test_stale_edit_does_not_overwrite_existing_revision(completed):
    _, case, service, run = completed
    service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 300})
    path = service.artifact(case["case_id"], run["run_id"], "cavities", None)
    digest = sha256_file(path)
    with pytest.raises(WebAppError) as exc:
        service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 1})
    assert exc.value.status_code == 409 and sha256_file(path) == digest


def test_removing_a_cell_clears_current_overlays_but_preserves_original(completed):
    _, case, service, run = completed
    originals = {name: sha256_file(service.artifact(case["case_id"], run["run_id"], name)) for name in ("cavities", "air")}
    edited = service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 1})
    assert edited["counts"]["total"] == 1
    assert next(c for c in edited["cells"] if c["id"] == 1)["voxel_count"] == 0
    for name in ("cavities", "air"):
        current = nib.load(service.artifact(case["case_id"], run["run_id"], name, None))
        assert 1 not in np.unique(np.asarray(current.dataobj))
        assert sha256_file(service.artifact(case["case_id"], run["run_id"], name)) == originals[name]


def test_painted_additions_and_expansion_use_frozen_threshold_and_keep_opaque_content(completed, tmp_path: Path):
    _, case, service, run = completed
    added = np.zeros((8, 9, 6), dtype=np.uint8)
    added[0, 2:4, 2] = 1  # -600 HU air and +35 HU opaque content.
    mask = write_nifti(tmp_path / "added.nii.gz", added)
    edited = service.revise(case["case_id"], run["run_id"], 0, {"type": "add", "side": "left"}, mask)
    added_cell = next(c for c in edited["cells"] if c["id"] == 301)
    assert added_cell["voxel_count"] == 2 and added_cell["air_voxel_count"] == 1
    air = np.asarray(nib.load(service.artifact(case["case_id"], run["run_id"], "air", None)).dataobj)
    assert air[0, 2, 2] == 301 and air[0, 3, 2] == 0
    expanded = np.zeros((8, 9, 6), dtype=np.uint8)
    expanded[1:4, 2:5, 1:4] = 1
    mask = write_nifti(tmp_path / "expanded.nii.gz", expanded)
    edited = service.revise(case["case_id"], run["run_id"], 1, {"type": "update", "id": 1}, mask)
    first = next(c for c in edited["cells"] if c["id"] == 1)
    assert first["voxel_count"] == first["air_voxel_count"] == 27
    assert next(c for c in edited["cells"] if c["id"] == 300)["air_voxel_count"] == 0


def test_missing_frozen_air_threshold_refuses_paint_instead_of_using_a_new_default(tmp_path: Path):
    def missing_threshold(source, output, config):
        synthetic_executor(source, output, config)
        path = output / "result.json"
        result = json.loads(path.read_text())
        result["config"] = {}
        path.write_text(json.dumps(result))
    _, case, service = setup_service(tmp_path, missing_threshold)
    try:
        created = service.create_run(case["case_id"])
        run = finish(service, case["case_id"], created["run_id"])
        added = np.zeros((8, 9, 6), dtype=np.uint8)
        added[0, 2, 2] = 1
        mask = write_nifti(tmp_path / "added.nii.gz", added)
        with pytest.raises(WebAppError, match="frozen air-content threshold"):
            service.revise(case["case_id"], run["run_id"], 0, {"type": "add", "side": "left"}, mask)
        assert service.get_run(case["case_id"], run["run_id"])["revision"] == 0
    finally:
        service.close()


def test_approved_case_cannot_start_or_edit_count_runs(completed):
    app, case, service, run = completed
    with app.store.connect() as connection:
        connection.execute("UPDATE cases SET state='APPROVED', review_status='Approved' WHERE id=?", (case["case_id"],))
    with pytest.raises(WebAppError, match="Approved cases"):
        service.create_run(case["case_id"])
    with pytest.raises(WebAppError, match="Approved cases"):
        service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 1})
    assert service.get_run(case["case_id"], run["run_id"])["original_counts"]["total"] == 2


def test_boundary_edit_preserves_other_ids_and_rejects_overlap(completed, tmp_path: Path):
    _, case, service, run = completed
    selected = np.zeros((8, 9, 6), dtype=np.uint8)
    selected[1:3, 2:5, 1:3] = 1
    mask = write_nifti(tmp_path / "selection.nii.gz", selected)
    edited = service.revise(case["case_id"], run["run_id"], 0, {"type": "update", "id": 1}, mask)
    assert edited["counts"]["total"] == 2
    assert next(c for c in edited["cells"] if c["id"] == 1)["voxel_count"] == 12
    selected[5, 2, 2] = 1
    write_nifti(mask, selected)
    with pytest.raises(WebAppError, match="overwrite"):
        service.revise(case["case_id"], run["run_id"], 1, {"type": "update", "id": 1}, mask)


def test_selection_geometry_and_binary_values_are_checked(completed, tmp_path: Path):
    _, case, service, run = completed
    for index, values in enumerate((np.ones((2, 2, 2), dtype=np.uint8), np.full((8, 9, 6), 2, dtype=np.uint8))):
        mask = write_nifti(tmp_path / f"invalid_{index}.nii.gz", values)
        with pytest.raises(WebAppError, match="source geometry"):
            service.revise(case["case_id"], run["run_id"], 0, {"type": "split", "id": 1}, mask)
    assert service.get_run(case["case_id"], run["run_id"])["revision"] == 0


def test_export_has_distinct_original_reviewed_counts_and_hash_bound_masks(completed):
    _, case, service, run = completed
    service.revise(case["case_id"], run["run_id"], 0, {"type": "remove", "id": 300})
    path = service.export(case["case_id"], run["run_id"])
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        original = json.loads(archive.read("original/result.json"))
        corrected = json.loads(archive.read("reviewed/result.json"))
        assert manifest["original_counts"] == original["counts"] == {"left": 1, "right": 1, "total": 2}
        assert manifest["corrected_counts"] == corrected["counts"] == {"left": 1, "right": 0, "total": 1}
        assert manifest["clinical_accuracy"] == "UNMEASURED"
        assert b"300,right,False,automatic" in archive.read("reviewed/cells.csv")
        assert all(f"{side}/{filename}" in archive.namelist() for side in ("original", "reviewed") for filename in ARTIFACTS.values())


def test_repeated_export_does_not_replace_an_active_download(completed):
    _, case, service, run = completed
    first = service.export(case["case_id"], run["run_id"])
    before = first.read_bytes()
    # A browser may still hold the first ZIP open while another request starts.
    # Windows refuses to replace such a file; either platform must keep it intact.
    with first.open("rb") as active_download:
        second = service.export(case["case_id"], run["run_id"])
        assert first != second
        assert first.name == second.name == "count_revision_000000.zip"
        assert active_download.read() == before
    assert first.read_bytes() == before
    with zipfile.ZipFile(second) as archive:
        assert json.loads(archive.read("manifest.json"))["revision"] == 0


def test_corrupted_mask_or_source_blocks_reads_and_export(completed):
    app, case, service, run = completed
    artifact = service.artifact(case["case_id"], run["run_id"], "cavities")
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(WebAppError, match="integrity"):
        service.export(case["case_id"], run["run_id"])
    source = app.volume_path(case["case_id"])
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(WebAppError, match="source volume"):
        service.get_run(case["case_id"], run["run_id"])


def test_completed_count_artifacts_follow_existing_case_deletion(completed):
    app, case, service, run = completed
    artifact = service.artifact(case["case_id"], run["run_id"], "cavities")
    assert artifact.is_relative_to(app.root / "cases" / case["case_id"])
    app.delete_draft(case["case_id"])
    assert not artifact.exists()
    with app.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM count_runs").fetchone()[0] == 0


def test_outside_artifact_directory_is_rejected_without_exposing_path(completed, tmp_path: Path):
    _, _, service, _ = completed
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "result.json").write_text("{}")
    with pytest.raises(WebAppError, match="unavailable or invalid") as exc:
        service._safe_file(outside, "result.json")
    assert str(outside) not in str(exc.value)


@pytest.mark.parametrize("kind", ["traversal", "wrong_source", "wrong_count", "uint8", "air_mismatch"])
def test_invalid_engine_outputs_never_become_counts(tmp_path: Path, kind: str):
    def invalid(source, output, config):
        synthetic_executor(source, output, config)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        if kind == "traversal":
            result["artifacts"]["cavities"] = "../source.nii.gz"
        elif kind == "wrong_source":
            result["source_sha256"] = "0" * 64
        elif kind == "wrong_count":
            result["counts"]["total"] = 999
        else:
            path = output / ("cavities.nii.gz" if kind == "uint8" else "air.nii.gz")
            image = nib.load(path)
            values = np.asarray(image.dataobj).copy()
            if kind == "air_mismatch":
                values[0, 0, 0] = 999
            nib.save(nib.Nifti1Image(values.astype(np.uint8 if kind == "uint8" else np.uint16), image.affine), path)
        result_path.write_text(json.dumps(result))
    _, case, service = setup_service(tmp_path, invalid)
    try:
        created = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], created["run_id"])
        assert result["status"] == "failed" and result["counts"] is None and result["original_counts"] is None
        assert result["cells"] == []
    finally:
        service.close()


def test_double_click_idempotency_and_interrupted_restart(tmp_path: Path):
    started, release = threading.Event(), threading.Event()
    def blocked(source, output, config):
        started.set()
        assert release.wait(timeout=10)
        synthetic_executor(source, output, config)
    app, case, service = setup_service(tmp_path, blocked)
    try:
        first = service.create_run(case["case_id"], "same-request")
        assert started.wait(timeout=5)
        second = service.create_run(case["case_id"], "other-request")
        assert first["run_id"] == second["run_id"]
        with pytest.raises(WebAppError, match="active count run"):
            app.delete_draft(case["case_id"])
        release.set()
        final = finish(service, case["case_id"], first["run_id"])
        assert final["status"] == "completed"
        assert service.create_run(case["case_id"], "same-request")["run_id"] == first["run_id"]
        with app.store.connect() as connection:
            connection.execute("UPDATE count_runs SET status='running',original_result_json=NULL,latest_revision=NULL WHERE id=?", (first["run_id"],))
        restarted = CountingService(app, executor=synthetic_executor, protocol=PROTOCOL)
        try:
            interrupted = restarted.get_run(case["case_id"], first["run_id"])
            assert interrupted["status"] == "failed" and interrupted["failure_code"] == "INTERRUPTED"
            assert interrupted["counts"] is None
        finally:
            restarted.close()
    finally:
        release.set()
        service.close()


def test_engine_failure_is_not_zero_and_only_allowlisted_code_is_exposed(tmp_path: Path):
    def failed(_source, output, _config):
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps({
            "status": "failed", "failure_code": "SPHENOID_UNRESOLVED",
            "message": "Do not relay potentially identifying library output.",
            "counts": {"left": None, "right": None, "total": None},
        }))
    _, case, service = setup_service(tmp_path, failed)
    try:
        created = service.create_run(case["case_id"])
        result = finish(service, case["case_id"], created["run_id"])
        assert result["failure_code"] == "SPHENOID_UNRESOLVED"
        assert result["counts"] is None and result["original_counts"] is None
        assert "potentially identifying" not in json.dumps(result)
    finally:
        service.close()


def test_production_worker_matches_offline_isolated_evaluation_environment(tmp_path: Path, monkeypatch):
    local = tmp_path / "local"
    runtime = local / "EthmoidComplexity" / "tools" / "counting-runtime"
    python = runtime / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic executable; never launched")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    monkeypatch.setenv("nnUNet_compile", "true")
    monkeypatch.setenv("ETHMOID_INFERENCE_OFFLINE", "0")
    captured = {}
    def capture(command, **options):
        captured.update(command=command, **options)
        return SimpleNamespace(returncode=0, wait=lambda timeout: 0)
    monkeypatch.setattr("ethmoid_pipeline.webapp.counting_service.subprocess.Popen", capture)
    app, case, service = setup_service(tmp_path)
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(PROTOCOL))
    try:
        service._execute_local(app.volume_path(case["case_id"]), tmp_path / "output", config)
        assert captured["command"][0] == str(python)
        assert captured["env"]["PYTHONNOUSERSITE"] == "1"
        assert captured["env"]["nnUNet_compile"] == "false"
        assert captured["env"]["ETHMOID_INFERENCE_OFFLINE"] == "1"
        assert captured["env"]["TOTALSEG_HOME_DIR"] == str(runtime / "totalsegmentator")
        assert captured["cwd"] == runtime
    finally:
        service.close()


@pytest.mark.parametrize("operation", [
    {"type": "add", "side": "left", "centroid_ras_mm": [99999, 0, 0]},
    {"type": "merge", "ids": [1, 300]},
    {"type": "merge", "ids": [1, {}]},
    {"type": "update", "id": 1, "uncertain": "no"},
    {"type": "update", "id": 1, "included": 0},
    {"type": "remove", "id": True},
])
def test_invalid_corrections_preserve_original(completed, operation):
    _, case, service, run = completed
    with pytest.raises(WebAppError):
        service.revise(case["case_id"], run["run_id"], 0, operation)
    assert service.get_run(case["case_id"], run["run_id"])["latest_revision"] == 0
