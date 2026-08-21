from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp import create_app
from ethmoid_pipeline.webapp.service import GENERIC_QC_CHECK_NAMES


def make_nifti(path: Path, *, mask: bool = False) -> Path:
    data = np.zeros((5, 6, 7), dtype=np.uint8 if mask else np.int16)
    if mask:
        data[1:3, 2:4, 3] = 3
    affine = np.diag([0.7, 0.7, 1.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def test_http_nifti_annotation_review_and_delete_routes(tmp_path: Path) -> None:
    source = make_nifti(tmp_path / "source.nii.gz")
    report = tmp_path / "qc.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "AUTOMATED_PASS_PENDING_MANUAL",
                "case": "Case201",
                "series": "Series001",
                "checks": [
                    {"name": name, "passed": True}
                    for name in sorted(GENERIC_QC_CHECK_NAMES)
                ],
                "issues": [],
                "warnings": [],
                "manual_slicer_qc": "PENDING",
                "clinical_series_and_coverage_review": "PENDING",
                "metrics": {
                    "nifti_sha256": sha256_file(source),
                    "nifti_shape": [5, 6, 7],
                    "hu_voxel_count": 5 * 6 * 7,
                    "hu_nonfinite_count": 0,
                    "max_absolute_hu_error": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    app = create_app(work_root=tmp_path / "app", dcm2niix=tmp_path / "unused.exe")
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Ethmoid" in index.text
        response = client.post(
            "/api/imports/nifti",
            data={"case_label": "Case201"},
            files={
                "file": ("source.nii.gz", source.read_bytes(), "application/gzip"),
                "qc_report": ("qc.json", report.read_bytes(), "application/json"),
            },
        )
        assert response.status_code == 200, response.text
        case = response.json()["case"]
        case_id = case["case_id"]
        assert case["validation_status"] == "VERIFIED"
        assert client.get("/api/cases").json()["cases"][0]["case_id"] == case_id
        volume_response = client.get(f"/api/cases/{case_id}/volume")
        assert volume_response.status_code == 200
        assert volume_response.headers["content-disposition"].startswith("inline;")
        viewer_state = {
            "crosshair": [0.25, 0.5, 0.75],
            "view_mode": "render",
            "selected_label": 2,
        }
        persisted = client.patch(
            f"/api/cases/{case_id}/viewer-state", json=viewer_state
        )
        assert persisted.status_code == 200
        assert persisted.json()["viewer_state"] == viewer_state

        mask = make_nifti(tmp_path / "annotation.nii", mask=True)
        saved = client.put(
            f"/api/cases/{case_id}/annotations",
            data={"viewer_state": json.dumps({"selected_label": 3, "brush_size": 2})},
            files={"mask": ("annotation.nii", mask.read_bytes(), "application/octet-stream")},
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["revision"] == 1
        assert saved.json()["voxel_counts"]["3"] == 4
        annotation_response = client.get(
            f"/api/cases/{case_id}/annotations/latest"
        )
        assert annotation_response.status_code == 200
        assert annotation_response.headers["content-disposition"].startswith("inline;")
        assert client.get(f"/api/cases/{case_id}/annotations/latest.nii.gz").headers[
            "content-disposition"
        ].startswith("inline;")

        update = {
            "status": "Draft",
            "qc_checklist": {
                "coverage": True,
                "orientation": True,
                "image_quality": True,
                "slice_continuity": True,
            },
            "reviewer_notes": "Reviewed locally.",
        }
        assert client.patch(f"/api/cases/{case_id}/review", json=update).status_code == 200
        finalized = client.post(f"/api/cases/{case_id}/finalize")
        assert finalized.status_code == 200
        assert finalized.json()["state"] == "READY_FOR_REVIEW"
        export_response = client.get(f"/api/cases/{case_id}/export")
        assert export_response.status_code == 200
        assert export_response.headers["content-disposition"].startswith("attachment;")
        assert client.delete(f"/api/cases/{case_id}").status_code == 409

        second = client.post(
            "/api/imports/nifti",
            data={"case_label": "Case202"},
            files={"file": ("source.nii.gz", source.read_bytes(), "application/gzip")},
        ).json()["case"]
        assert client.delete(f"/api/cases/{second['case_id']}").status_code == 204
        assert client.get(f"/api/cases/{second['case_id']}").status_code == 404


def test_http_rejects_wrong_mask_geometry(tmp_path: Path) -> None:
    source = make_nifti(tmp_path / "source.nii.gz")
    app = create_app(work_root=tmp_path / "app", dcm2niix=tmp_path / "unused.exe")
    with TestClient(app) as client:
        case = client.post(
            "/api/imports/nifti",
            data={"case_label": "Case203"},
            files={"file": ("source.nii.gz", source.read_bytes(), "application/gzip")},
        ).json()["case"]
        wrong = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
        wrong_path = tmp_path / "wrong.nii"
        nib.save(wrong, wrong_path)
        response = client.put(
            f"/api/cases/{case['case_id']}/annotations",
            data={"viewer_state": "{}"},
            files={"mask": ("wrong.nii", wrong_path.read_bytes())},
        )
        assert response.status_code == 422
        assert "shape" in response.json()["detail"]
