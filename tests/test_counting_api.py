from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient

from ethmoid_pipeline.webapp.app import create_app
from tests.test_counting_revisions import PROTOCOL, finish, synthetic_executor
from tests.test_webapp_service import write_nifti
from scripts.create_e2e_fixture import build_directional_phantom
from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.counting_service import PROTOCOL_VERSION


def test_count_api_full_workflow_and_multipart_revision(tmp_path: Path):
    app = create_app(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused")
    counting = app.state.counting_service
    counting._executor = synthetic_executor
    counting._protocol = PROTOCOL
    case = app.state.ethmoid_service.import_nifti(write_nifti(tmp_path / "source.nii.gz"), case_label="Case902")
    base = f"/api/cases/{case['case_id']}/count-runs"
    with TestClient(app) as client:
        response = client.post(base, json={"idempotency_key": "workflow-request"})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        finish(counting, case["case_id"], run_id)
        run_url = f"{base}/{run_id}"
        assert client.get(base).json()["runs"][0]["run_id"] == run_id
        assert client.get(run_url).json()["counts"]["total"] == 2
        assert client.get(f"{run_url}/artifacts/cavities").status_code == 200
        assert client.get(f"{run_url}/artifacts/private").status_code == 404
        selection = np.zeros((8, 9, 6), dtype=np.uint8)
        selection[1, 2:5, 1:4] = 1
        mask = write_nifti(tmp_path / "split.nii.gz", selection)
        response = client.post(f"{run_url}/revisions", data={"metadata": json.dumps({"parent_revision": 0, "operation": {"type": "split", "id": 1}})}, files={"mask": ("split.nii.gz", mask.read_bytes(), "application/gzip")})
        assert response.status_code == 200, response.text
        assert response.json()["counts"]["total"] == 3
        assert response.json()["original_counts"]["total"] == 2
        assert client.get(f"{run_url}/revisions/latest").json()["revision"] == 1
        assert client.get(f"{run_url}/artifacts/cavities?revision=latest").status_code == 200
        assert client.get(f"{run_url}/export").headers["content-type"] == "application/zip"
        stale = client.post(f"{run_url}/revisions", json={"parent_revision": 0, "operation": {"type": "remove", "id": 300}})
        assert stale.status_code == 409
        assert client.post(base, json={"threshold": -500}).status_code == 422
        assert client.get(f"{run_url}/artifacts/cavities?revision=../original").status_code == 422
        assert client.post(f"{run_url}/revisions", json={"parent_revision": 1, "operation": {"type": "update", "id": 300, "side": "left"}, "secret": True}).status_code == 422


def test_failed_source_and_wrong_case_cannot_count_or_read_results(tmp_path: Path):
    app = create_app(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused")
    counting = app.state.counting_service
    counting._executor = synthetic_executor
    counting._protocol = PROTOCOL
    case = app.state.ethmoid_service.import_nifti(write_nifti(tmp_path / "source.nii.gz"), case_label="Case903")
    with TestClient(app) as client:
        with app.state.ethmoid_service.store.connect() as connection:
            connection.execute("UPDATE cases SET validation_status='FAILED' WHERE id=?", (case["case_id"],))
        response = client.post(f"/api/cases/{case['case_id']}/count-runs")
        assert response.status_code == 409
        assert client.post("/api/cases/missing/count-runs").status_code == 404


def test_export_revision_two_after_remove_point_add_and_reload(tmp_path: Path):
    def execute(source_path, output, _config):
        source = nib.load(source_path)
        x, y, z = np.indices(source.shape)
        left = ((x - 15) / 5) ** 2 + ((y - 30) / 7) ** 2 + ((z - 24) / 8) ** 2 <= 1
        right = ((x - 33) / 7) ** 2 + ((y - 27) / 5) ** 2 + ((z - 20) / 6) ** 2 <= 1
        cavities = np.where(left, 1, np.where(right, 300, 0)).astype(np.uint16)
        air = np.where(np.asarray(source.dataobj) < -500, cavities, 0).astype(np.uint16)
        region = np.where(left, 1, np.where(right, 2, 0)).astype(np.uint16)
        output.mkdir(parents=True)
        artifacts = {name: f"{name}.nii.gz" for name in ("cavities", "air", "region")}
        for name, data in (("cavities", cavities), ("air", air), ("region", region)):
            image = nib.Nifti1Image(data, source.affine)
            image.set_qform(source.affine, 1)
            image.set_sform(source.affine, 1)
            nib.save(image, output / artifacts[name])
        cells = [{"id": identity, "side": side,
                  "centroid_ras_mm": nib.affines.apply_affine(source.affine, np.argwhere(cavities == identity).mean(axis=0)).tolist(),
                  "included": True, "uncertain": True, "origin": "automatic", "reasons": [],
                  "voxel_count": int(np.count_nonzero(cavities == identity)),
                  "air_voxel_count": int(np.count_nonzero(air == identity))}
                 for identity, side in ((1, "left"), (300, "right"))]
        result = {"schema_version": 1, "status": "completed", "method_version": "synthetic-browser-regression",
                  "source_sha256": sha256_file(source_path), "protocol_version": PROTOCOL_VERSION,
                  "counts": {"left": 1, "right": 1, "total": 2}, "cells": cells,
                  "warnings": [], "config": {"air_hu": -500}, "artifacts": artifacts}
        (output / "result.json").write_text(json.dumps(result))

    fixture = tmp_path / "directional.nii"
    build_directional_phantom(fixture)
    work = tmp_path / "work"
    app = create_app(work_root=work, dcm2niix=tmp_path / "unused")
    app.state.counting_service._protocol = PROTOCOL
    app.state.counting_service._executor = execute
    case = app.state.ethmoid_service.import_nifti(fixture, case_label="Case904")
    with TestClient(app) as client:
        url = f"/api/cases/{case['case_id']}/count-runs"
        run = client.post(url).json()
        result = finish(app.state.counting_service, case["case_id"], run["run_id"])
        assert result["status"] == "completed"
        url += f"/{run['run_id']}"
        point = result["cells"][1]["centroid_ras_mm"]
        assert client.post(url + "/revisions", json={"parent_revision": 0, "operation": {"type": "remove", "id": 300}}).status_code == 200
        added = client.post(url + "/revisions", json={"parent_revision": 1, "operation": {"type": "add", "side": "right", "centroid_ras_mm": point}})
        assert added.status_code == 200
        assert added.json()["revision"] == 2
    reloaded = create_app(work_root=work, dcm2niix=tmp_path / "unused")
    with TestClient(reloaded) as client:
        assert client.get(url).json()["counts"]["total"] == 2
        response = client.get(url + "/export?revision=2")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/zip"
        assert 'filename="count_revision_000002.zip"' in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert json.loads(archive.read("manifest.json"))["revision"] == 2
            assert b"301,right,True,manual" in archive.read("reviewed/cells.csv")
        partial = client.get(url + "/export?revision=2", headers={"Range": "bytes=0-3"})
        assert partial.status_code == 206
        assert partial.content == b"PK\x03\x04"
        assert client.get(url + "/export?revision=2").status_code == 200
        assert not list(work.glob("cases/*/count-runs/*/export_*"))
