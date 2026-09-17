from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import generate_uid
import pytest

from ethmoid_pipeline.inventory import build_inventory
from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.app import create_app
from ethmoid_pipeline.webapp.catalog import CatalogService
from ethmoid_pipeline.webapp.catalog_config import CatalogConfig
from ethmoid_pipeline.webapp.service import AppService, WebAppError
from ethmoid_pipeline.webapp.store import Store, encode_json
from tests.synthetic import write_ct_series
from tests.test_webapp_service import write_nifti, write_matching_qc


def write_verified_case(service, paths, label, directory):
    records = sorted((pydicom.dcmread(path) for path in paths), key=lambda d: float(d.ImagePositionPatient[2]))
    data = np.stack([d.pixel_array.T.astype(np.int16) - 1024 for d in records], axis=2)
    affine = np.diag([-0.5, -0.5, 0.625, 1.0])
    source = directory / (label + ".nii")
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, source)
    report = write_matching_qc(directory / (label + ".json"), source, label)
    case = service.app.import_nifti(source, case_label=label, qc_report=report)
    target = service.app.volume_path(case["case_id"]).parent
    (target / "qc_report.json").write_bytes(report.read_bytes())
    (target / "source_hashes.json").write_text(json.dumps({"slices": [{"sha256": sha256_file(path)} for path in paths]}))
    return case


@pytest.fixture
def catalog(tmp_path):
    source, work = tmp_path / "source", tmp_path / "work"
    work.mkdir()
    study = generate_uid()
    native = write_ct_series(source / "a", study_uid=study)
    scout = write_ct_series(source / "b", study_uid=study, image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"])
    write_ct_series(source / "c")
    build_inventory(source, work)
    app = AppService(work_root=work / "webapp", dcm2niix=tmp_path / "unused")
    service = CatalogService(app, CatalogConfig(work, source, auto_prepare=False))
    return service, native, scout


def test_schema2_migration_backs_up_and_preserves_legacy_rows(catalog, tmp_path):
    service, _, _ = catalog
    old = service.app.import_nifti(write_nifti(tmp_path / "old.nii"), case_label="Case099")
    with service.store.connect() as connection:
        connection.execute("UPDATE cases SET reviewer_notes='Synthetic retained notes',state='ANNOTATING' WHERE id=?", (old["case_id"],))
        connection.execute("UPDATE metadata SET value='2' WHERE key='schema_version'")
    Store(service.app.root)
    backups = list(service.app.root.glob("schema-v2-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "2"
        assert backup.execute("SELECT id FROM cases").fetchone()[0] == old["case_id"]
    restored = service.app.get_case(old["case_id"])
    assert restored["reviewer_notes"] == "Synthetic retained notes" and restored["state"] == "ANNOTATING"
    Store(service.app.root)
    assert len(list(service.app.root.glob("schema-v2-*.sqlite3"))) == 1


def test_catalog_groups_all_series_with_safe_responses(catalog):
    service, _, _ = catalog
    service.sync()
    value = service.list_studies()
    assert len(value["studies"]) == 2
    assert value["catalog"]["total_series"] == 3
    first = value["studies"][0]
    assert first["series_count"] == 2
    assert first["recommended_series_id"] == first["series"][0]["series_id"]
    scout = first["series"][1]
    assert scout["source_kind"] == "scout" and scout["preparation_status"] == "unavailable"
    assert scout["retryable"] is False and scout["counts"] is None
    serialized = json.dumps(value)
    assert "study_uid" not in serialized and "series_uid" not in serialized
    assert str(service.config.source_root) not in serialized and "SYNTHETIC" not in serialized
    assert "relative_files" not in serialized


def test_identity_match_uses_exact_dicom_hashes_not_case_label(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    context = service.app.get_case(case["case_id"])["catalog"]
    assert context["study_label"] == "Case001" and context["series_label"] == "Series001"
    assert service.app.get_case(case["case_id"])["case_label"] == "Case099"
    first = service.list_studies()["studies"][0]
    assert first["series"][0]["case_id"] == case["case_id"]
    assert service.list_studies()["unlinked_cases"] == []
    service.sync()
    assert service.list_studies()["studies"][0]["series"][0]["case_id"] == case["case_id"]


def test_same_label_does_not_link_unrelated_source(catalog, tmp_path):
    service, _, _ = catalog
    case = service.app.import_nifti(write_nifti(tmp_path / "unrelated.nii"), case_label="Case001")
    service.sync()
    assert service.app.catalog_context(case["case_id"]) is None
    assert service.list_studies()["unlinked_cases"][0]["case_id"] == case["case_id"]


def test_sync_is_idempotent_preserves_selection_and_stable_labels(catalog):
    service, _, _ = catalog
    service.sync()
    before = service.list_studies()["studies"]
    first = before[0]
    service.select(first["study_id"], first["series"][1]["series_id"])
    # Discovery labels may change when folders are reordered. Identity is stable.
    mapping_path = service.config.work_root / "inventory/uid_map.local.json"
    inventory_path = service.config.work_root / "inventory/inventory.redacted.json"
    mapping = json.loads(mapping_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    for study in mapping["cases"]:
        study["case"] = "Case002" if study["case"] == "Case001" else "Case001"
    for item in inventory["series"]:
        item["case"] = "Case002" if item["case"] == "Case001" else "Case001"
    mapping_path.write_text(json.dumps(mapping))
    inventory_path.write_text(json.dumps(inventory))
    service.sync()
    after = service.list_studies()["studies"]
    assert [item["study_id"] for item in after] == [item["study_id"] for item in before]
    assert [item["study_label"] for item in after] == ["Case001", "Case002"]
    assert after[0]["selected_series_id"] == first["series"][1]["series_id"]


def test_changed_source_preserves_history_but_invalidates_active_reuse(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    native[0].write_bytes(native[0].read_bytes() + b"changed")
    service.sync()
    series = service.list_studies()["studies"][0]["series"][0]
    assert series["case_id"] is None and series["preparation_status"] == "not_prepared"
    assert service.app.get_case(case["case_id"])["case_id"] == case["case_id"]
    assert service.app.catalog_context(case["case_id"]) is not None


def test_selection_rejects_cross_study_series(catalog):
    service, _, _ = catalog
    service.sync()
    studies = service.list_studies()["studies"]
    with pytest.raises(WebAppError, match="not found"):
        service.select(studies[0]["study_id"], studies[1]["series"][0]["series_id"])


def test_api_preserves_legacy_routes_and_adds_catalog(catalog, tmp_path):
    service, _, _ = catalog
    app = create_app(work_root=service.app.root, dcm2niix=tmp_path / "unused")
    app.state.catalog_service.config = service.config
    with TestClient(app) as client:
        app.state.catalog_service._thread.join(timeout=5)
        response = client.get("/api/studies")
        assert response.status_code == 200
        first = response.json()["studies"][0]
        series = client.get(f"/api/studies/{first['study_id']}/series").json()["series"]
        selected = series[-1]["series_id"]
        response = client.patch(f"/api/studies/{first['study_id']}", json={"selected_series_id": selected})
        assert response.json()["study"]["selected_series_id"] == selected
        assert client.get("/api/catalog/status").json()["total_studies"] == 2
        assert client.get("/api/cases").json() == {"cases": []}


def test_copied_source_hashes_cannot_link_unrelated_voxels(catalog, tmp_path):
    service, native, _ = catalog
    volume = write_nifti(tmp_path / "unrelated.nii")
    report = write_matching_qc(tmp_path / "unrelated.json", volume, "Case099")
    case = service.app.import_nifti(volume, case_label="Case099", qc_report=report)
    target = service.app.volume_path(case["case_id"]).parent
    (target / "qc_report.json").write_bytes(report.read_bytes())
    (target / "source_hashes.json").write_text(json.dumps({"slices": [{"sha256": sha256_file(path)} for path in native]}))
    service.sync()
    assert service.app.catalog_context(case["case_id"]) is None
    assert all(item["case_id"] is None for study in service.list_studies()["studies"] for item in study["series"])


def test_reuse_rejects_integer_pixel_shift_even_when_hu_values_and_qc_claim_match(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    target = service.app.volume_path(case["case_id"])
    image = nib.load(target)
    affine = image.affine.copy()
    affine[0, 3] += .5
    shifted = nib.Nifti1Image(np.asarray(image.dataobj).copy(), affine)
    shifted.set_qform(affine, code=1)
    shifted.set_sform(affine, code=1)
    nib.save(shifted, target)
    report_path = write_matching_qc(target.parent / "qc_report.json", target, "Case099")
    with service.store.connect() as connection:
        connection.execute("UPDATE cases SET source_sha256=?,affine_json=?,validation_details_json=? WHERE id=?", (sha256_file(target), encode_json(affine.tolist()), report_path.read_text(), case["case_id"]))
    service.sync()
    assert service.app.catalog_context(case["case_id"]) is None


def test_auto_selection_follows_usable_recommendation_but_explicit_does_not(catalog):
    service, _, _ = catalog
    service.sync()
    first = service.list_studies()["studies"][0]
    selected = first["selected_series_id"]
    with service.store.connect() as connection:
        connection.execute("UPDATE catalog_series SET preparation_status='unavailable',reason_code='GEOMETRY_INVALID' WHERE id=?", (selected,))
    service.refresh_recommendations()
    assert service.study(first["study_id"])["recommended_series_id"] is None
    service.select(first["study_id"], first["series"][1]["series_id"])
    service.refresh_recommendations()
    assert service.study(first["study_id"])["selected_series_id"] == first["series"][1]["series_id"]


def test_recommendation_tiers_and_spacing_coverage_do_not_depend_on_ready_state():
    def row(label, *, kernel="BONEPLUS", kind="native", thin=True, interval=0.625, count=100, ready=False):
        return {"preparation_status": "ready" if ready else "queued", "series_label": label,
                "descriptor_json": encode_json({"resolution_class": "within_protocol" if thin else "outside_protocol", "source_kind": kind,
                    "kernel": kernel, "spacing_mm": [.5, .5, interval], "image_count": count})}
    values = [row("Series007", kind="mpr", thin=False, interval=.3),
              row("Series006", thin=False, interval=2.5, ready=True),
              row("Series005", kernel="STANDARD", interval=.3),
              row("Series004", interval=.625, ready=True),
              row("Series003", interval=.5, count=150),
              row("Series002", interval=.5, count=200),
              row("Series001", interval=.5, count=200)]
    assert [item["series_label"] for item in sorted(values, key=CatalogService.rank)] == [
        "Series001", "Series002", "Series003", "Series004", "Series005", "Series007", "Series006"]
