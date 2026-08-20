from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pydicom

from ..conversion import convert_series
from ..dicom import PreflightFailure, preflight_series
from ..util import PipelineError
from ..verification import verify_conversion
from .service import WebAppError


def inspect_dicom_upload(paths: Iterable[Path], staging_root: Path) -> list[dict[str, Any]]:
    """Group a neutral staged upload and run the existing strict preflight."""

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            study_uid = str(dataset.StudyInstanceUID)
            series_uid = str(dataset.SeriesInstanceUID)
        except Exception as exc:
            raise WebAppError(
                "At least one selected file is not a readable DICOM image.", status_code=422
            ) from exc
        groups.setdefault((study_uid, series_uid), []).append(path)
    if not groups:
        raise WebAppError("No DICOM series were found.", status_code=422)

    values: list[dict[str, Any]] = []
    ordered_groups = sorted(groups.items(), key=lambda item: min(path.name for path in item[1]))
    for index, ((study_uid, series_uid), series_paths) in enumerate(ordered_groups, start=1):
        label = f"Series{index:03d}"
        try:
            result = preflight_series(series_paths, study_uid, series_uid)
            summary: dict[str, Any] = {
                "series_label": label,
                "preflight_status": "PASS",
                "image_count": len(result.records),
                "rows": result.rows,
                "columns": result.columns,
                "pixel_spacing_mm": list(result.pixel_spacing),
                "slice_spacing_mm": result.slice_spacing,
                "spacing_mm": [
                    result.pixel_spacing[0],
                    result.pixel_spacing[1],
                    result.slice_spacing,
                ],
                "slice_thickness_mm": result.slice_thickness,
                "kernel": result.kernel,
                "warnings": list(result.warnings),
                "issues": [],
            }
        except PreflightFailure as exc:
            summary = {
                "series_label": label,
                "preflight_status": "FAIL",
                "image_count": len(series_paths),
                "rows": 0,
                "columns": 0,
                "pixel_spacing_mm": None,
                "slice_spacing_mm": None,
                "spacing_mm": [],
                "slice_thickness_mm": None,
                "kernel": None,
                "warnings": [],
                "issues": list(exc.issues),
            }
        summary.update(
            {
                # These values stay in the PHI-sensitive work root. The service
                # removes them from every HTTP response.
                "study_uid": study_uid,
                "series_uid": series_uid,
                "files": [
                    path.resolve().relative_to(staging_root.resolve()).as_posix()
                    for path in sorted(series_paths, key=lambda item: item.name)
                ],
            }
        )
        values.append(summary)
    return values


def convert_uploaded_series(
    paths: Iterable[Path],
    *,
    output_dir: Path,
    dcm2niix: Path,
    expected_study_uid: str,
    expected_series_uid: str,
    case_label: str,
    series_label: str,
) -> dict[str, Any]:
    """Use the reusable audited converter and verifier for one staged series."""

    selected = [Path(item).resolve() for item in paths]
    if not selected:
        raise WebAppError("No DICOM files were selected.", status_code=422)
    dicom_root = selected[0].parent
    if any(path.parent != dicom_root for path in selected):
        raise WebAppError("The staged DICOM selection is inconsistent.", status_code=409)
    work_root = output_dir.parent
    try:
        converted = convert_series(
            dicom_root,
            work_root,
            selected,
            case_label,
            series_label,
            dcm2niix,
            expected_study_uid=expected_study_uid,
            expected_series_uid=expected_series_uid,
            final_dir=output_dir,
            require_clean_provenance=False,
        )
        verification = verify_conversion(
            dicom_root,
            work_root,
            selected,
            case_label,
            series_label,
            expected_study_uid=expected_study_uid,
            expected_series_uid=expected_series_uid,
            artifact_dir=output_dir,
            require_clean_provenance=False,
        )
    except PreflightFailure as exc:
        raise WebAppError(str(exc), status_code=422) from exc
    except PipelineError as exc:
        raise WebAppError(str(exc), status_code=500) from exc
    if verification.get("status") == "FAIL":
        raise WebAppError("The converted DICOM series failed automated verification.", status_code=422)
    return {"nifti": converted["nifti"], "validation": verification}
