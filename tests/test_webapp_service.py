from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.webapp.service import (
    GENERIC_QC_CHECK_NAMES,
    AppService,
    WebAppError,
)
from tests.synthetic import rewrite, write_ct_series


def write_nifti(path: Path, data: np.ndarray | None = None) -> Path:
    values = (
        np.arange(8 * 9 * 6, dtype=np.int16).reshape(8, 9, 6)
        if data is None
        else data
    )
    affine = np.asarray(
        [[0.5, 0, 0, -12], [0, 0.5, 0, -24], [0, 0, 0.625, 3], [0, 0, 0, 1]],
        dtype=float,
    )
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def write_matching_qc(path: Path, volume: Path, case_label: str) -> Path:
    image = nib.load(volume)
    shape = [int(item) for item in image.shape]
    voxel_count = int(np.prod(shape))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "AUTOMATED_PASS_PENDING_MANUAL",
                "case": case_label,
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
                    "nifti_sha256": sha256_file(volume),
                    "nifti_shape": shape,
                    "hu_voxel_count": voxel_count,
                    "hu_nonfinite_count": 0,
                    "max_absolute_hu_error": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_nifti_annotation_revision_review_and_export(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    volume = write_nifti(tmp_path / "input.nii.gz")
    report = write_matching_qc(tmp_path / "qc.json", volume, "Case101")

    case = service.import_nifti(volume, case_label="Case101", qc_report=report)
    assert case["case_id"]
    assert case["validation_status"] == "VERIFIED"
    assert case["state"] == "READY"
    assert case["annotation_disposition"] == "PROVISIONAL_NOT_FOR_TRAINING"
    assert case["qc_checklist"] == {
        "coverage": False,
        "orientation": False,
        "image_quality": False,
        "slice_continuity": False,
    }
    updated_view = service.update_viewer_state(
        case["case_id"],
        {"crosshair": [0.2, 0.4, 0.6], "view_mode": "render", "selected_label": 4},
    )
    assert updated_view["viewer_state"] == {
        "crosshair": [0.2, 0.4, 0.6],
        "view_mode": "render",
        "selected_label": 4,
    }

    mask_values = np.zeros((8, 9, 6), dtype=np.int16)
    mask_values[1:3, 2:5, 1] = 1
    mask_values[4:6, 2:5, 1] = 2
    mask = write_nifti(tmp_path / "mask.nii.gz", mask_values)
    revision = service.save_annotation(
        case["case_id"], mask, {"selected_label": 2, "view_mode": "multiplanar"}
    )
    assert revision["revision"] == 1
    assert revision["metrics"]["labels"][0]["voxel_count"] == 6

    stored_mask = nib.load(service.latest_annotation_path(case["case_id"]))
    assert stored_mask.get_data_dtype() == np.dtype("uint8")
    assert np.array_equal(stored_mask.affine, nib.load(volume).affine)
    assert set(np.unique(np.asarray(stored_mask.dataobj))) == {0, 1, 2}

    reviewed = service.update_review(
        case["case_id"],
        qc_checklist={
            "coverage": True,
            "orientation": True,
            "image_quality": True,
            "slice_continuity": True,
        },
        reviewer_notes="Geometry and coverage reviewed.",
    )
    assert reviewed["reviewer_notes"] == "Geometry and coverage reviewed."
    ready = service.finalize_case(case["case_id"])
    assert ready["state"] == "READY_FOR_REVIEW"
    assert ready["review_status"] == "Ready for review"
    assert ready["metrics"][0]["label_value"] == 1
    approved = service.update_review(case["case_id"], status="Approved")
    assert approved["state"] == "APPROVED"
    with pytest.raises(WebAppError, match="cannot be finalized again"):
        service.finalize_case(case["case_id"])
    assert service.get_case(case["case_id"])["state"] == "APPROVED"

    archive_path = service.export_case(case["case_id"])
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "annotation.nii.gz",
            "manifest.json",
            "metrics.json",
            "protocol.json",
            "review.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        protocol = json.loads(archive.read("protocol.json"))
    assert manifest["case_label"] == "Case101"
    assert manifest["training_eligibility"] == "PROVISIONAL_NOT_FOR_TRAINING"
    assert [label["value"] for label in protocol["labels"]] == [0, 1, 2, 3, 4]


def test_unverified_nifti_can_be_viewed_but_not_approved(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    volume = write_nifti(tmp_path / "input.nii")
    case = service.import_nifti(volume, case_label="Case102")
    assert case["validation_status"] == "UNVERIFIED"
    assert service.volume_path(case["case_id"]).read_bytes() == volume.read_bytes()

    mask = write_nifti(tmp_path / "mask.nii.gz", np.zeros((8, 9, 6), dtype=np.uint8))
    service.save_annotation(case["case_id"], mask)
    service.update_review(
        case["case_id"],
        qc_checklist={key: True for key in ("coverage", "orientation", "image_quality", "slice_continuity")},
    )
    service.finalize_case(case["case_id"])
    with pytest.raises(WebAppError, match="verified source"):
        service.update_review(case["case_id"], status="Approved")


def test_case_labels_match_the_three_digit_converter_contract(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    volume = write_nifti(tmp_path / "input.nii")
    with pytest.raises(WebAppError, match="Case###"):
        service.import_nifti(volume, case_label="Case1000")


def test_minimal_forged_qc_report_does_not_mark_nifti_verified(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    volume = write_nifti(tmp_path / "input.nii")
    report = tmp_path / "forged.json"
    report.write_text(
        json.dumps(
            {
                "status": "AUTOMATED_PASS_PENDING_MANUAL",
                "checks": [{"name": "voxel count", "passed": True}],
                "metrics": {"nifti_sha256": sha256_file(volume)},
            }
        ),
        encoding="utf-8",
    )
    case = service.import_nifti(
        volume, case_label="Case108", qc_report=report
    )
    assert case["validation_status"] == "UNVERIFIED"
    assert case["validation_details"]["report_contract_matches"] is False


@pytest.mark.parametrize("invalid_value", [5, -1, 1.5])
def test_annotation_rejects_values_outside_protocol(tmp_path: Path, invalid_value: float) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    case = service.import_nifti(
        write_nifti(tmp_path / "input.nii.gz"), case_label="Case103"
    )
    values = np.zeros((8, 9, 6), dtype=np.float32)
    values[0, 0, 0] = invalid_value
    mask = write_nifti(tmp_path / "mask.nii.gz", values)
    with pytest.raises(WebAppError, match="0 through 4"):
        service.save_annotation(case["case_id"], mask)


def test_finalization_detects_source_mutation(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    case = service.import_nifti(
        write_nifti(tmp_path / "input.nii.gz"), case_label="Case104"
    )
    mask = write_nifti(tmp_path / "mask.nii.gz", np.zeros((8, 9, 6), dtype=np.uint8))
    service.save_annotation(case["case_id"], mask)
    with service.volume_path(case["case_id"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(WebAppError, match="source volume changed"):
        service.finalize_case(case["case_id"])


def test_dicom_import_lists_clean_and_failed_series_without_uids(tmp_path: Path) -> None:
    service = AppService(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused.exe")
    clean = write_ct_series(tmp_path / "clean")
    imported = service.import_dicom(clean, case_label="Case105")
    assert imported["status"] == "AWAITING_SERIES"
    assert imported["series"][0]["preflight_status"] == "PASS"
    assert imported["series"][0]["spacing_mm"] == [0.5, 0.5, 0.625]
    assert not ({"study_uid", "series_uid", "files"} & imported["series"][0].keys())

    mixed = write_ct_series(
        tmp_path / "mixed", acquisition_numbers=[1, 1, 1, 2, 2, 2]
    )
    failed = service.import_dicom(mixed, case_label="Case106")
    assert failed["series"][0]["preflight_status"] == "FAIL"
    assert any("acquisition" in issue.lower() for issue in failed["series"][0]["issues"])
    failed_staging = service.root / "imports" / failed["import_id"]
    assert failed_staging.is_dir()
    service.delete_import(failed["import_id"])
    assert not failed_staging.exists()
    with pytest.raises(WebAppError, match="Import not found"):
        service.get_import(failed["import_id"])


@pytest.mark.integration
def test_dicom_import_conversion_uses_generic_audited_pipeline(tmp_path: Path) -> None:
    executable = Path(os.environ.get("ETHMOID_DCM2NIIX", default_dcm2niix()))
    if not executable.is_file():
        pytest.skip("pinned dcm2niix is not installed")
    service = AppService(work_root=tmp_path / "work", dcm2niix=executable)
    paths = write_ct_series(
        tmp_path / "source",
        positions=[0.0, 0.8, 1.6, 2.4, 3.2],
        instance_numbers=[5, 4, 3, 2, 1],
    )
    for path in paths:
        rewrite(path, SliceThickness=0.8, SpacingBetweenSlices=0.8)
    imported = service.import_dicom(paths, case_label="Case107")
    converted = service.convert_dicom(imported["import_id"], "Series001")
    assert converted["source_type"] == "DICOM"
    assert converted["validation_status"] == "VERIFIED"
    assert converted["validation_details"]["status"] == "AUTOMATED_PASS_PENDING_MANUAL"
    assert service.volume_path(converted["case_id"]).is_file()
    assert not (service.root / "imports" / imported["import_id"]).exists()
