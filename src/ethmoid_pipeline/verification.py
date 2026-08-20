from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pydicom

from .conversion import _selection_identity_sha256, _validate_selected_paths
from .dicom import PreflightResult, assert_pilot_fingerprint, preflight_series
from .inventory import resolve_selection
from .privacy import NEUTRAL_NIFTI_DESCRIPTION, verify_sanitized_sidecar
from .provenance import current_code_provenance
from .util import (
    PipelineError,
    atomic_write_json,
    atomic_write_text,
    is_within,
    read_json,
    repo_root,
    sha256_canonical_json,
    sha256_file,
    utc_now,
    validate_labels,
    validate_roots,
)


def _dicom_pixel_ras(record: Any, row: int, column: int) -> np.ndarray:
    row_direction = np.asarray(record.iop[:3], dtype=float)
    column_direction = np.asarray(record.iop[3:], dtype=float)
    ipp = np.asarray(record.ipp, dtype=float)
    point_lps = (
        ipp
        + row_direction * (column * record.pixel_spacing[1])
        + column_direction * (row * record.pixel_spacing[0])
    )
    return np.asarray([-point_lps[0], -point_lps[1], point_lps[2]], dtype=float)


def _map_to_nifti(point_ras: np.ndarray, inverse_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    floating = (inverse_affine @ np.append(point_ras, 1.0))[:3]
    rounded = np.rint(floating).astype(int)
    return floating, rounded


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _full_volume_hu_metrics(
    image: nib.spatialimages.SpatialImage,
    records: tuple[Any, ...],
    inverse_affine: np.ndarray,
) -> dict[str, Any]:
    voxel_count = 0
    finite_error_count = 0
    nonfinite_count = 0
    error_sum = 0.0
    max_error = 0.0

    for record in records:
        origin_float, origin = _map_to_nifti(
            _dicom_pixel_ras(record, 0, 0), inverse_affine
        )
        row_float, row_index = _map_to_nifti(
            _dicom_pixel_ras(record, 1, 0), inverse_affine
        )
        column_float, column_index = _map_to_nifti(
            _dicom_pixel_ras(record, 0, 1), inverse_affine
        )
        if max(
            float(np.max(np.abs(origin_float - origin))),
            float(np.max(np.abs(row_float - row_index))),
            float(np.max(np.abs(column_float - column_index))),
        ) > 1e-3:
            raise PipelineError("Full-volume HU mapping is not on the NIfTI voxel grid.")
        row_step = row_index - origin
        column_step = column_index - origin
        if (
            int(np.sum(np.abs(row_step))) != 1
            or int(np.sum(np.abs(column_step))) != 1
            or int(np.argmax(np.abs(row_step))) == int(np.argmax(np.abs(column_step)))
        ):
            raise PipelineError("Full-volume HU mapping is not an axis-aligned lossless grid.")
        row_axis = int(np.argmax(np.abs(row_step)))
        column_axis = int(np.argmax(np.abs(column_step)))
        slice_axes = {0, 1, 2} - {row_axis, column_axis}
        if len(slice_axes) != 1:
            raise PipelineError("Full-volume HU mapping has an invalid plane axis.")
        slice_axis = slice_axes.pop()
        fixed_index = int(origin[slice_axis])
        if fixed_index < 0 or fixed_index >= image.shape[slice_axis]:
            raise PipelineError("Full-volume HU mapping is outside the NIfTI extent.")

        slicer: list[int | slice] = [slice(None), slice(None), slice(None)]
        slicer[slice_axis] = fixed_index
        native_plane = np.asarray(image.dataobj[tuple(slicer)])
        remaining_axes = [axis for axis in range(3) if axis != slice_axis]
        if remaining_axes == [row_axis, column_axis]:
            observed = native_plane
        elif remaining_axes == [column_axis, row_axis]:
            observed = native_plane.T
        else:
            raise PipelineError("Full-volume HU plane axes cannot be resolved.")
        if int(row_step[row_axis]) < 0:
            observed = np.flip(observed, axis=0)
        if int(column_step[column_axis]) < 0:
            observed = np.flip(observed, axis=1)

        dataset = pydicom.dcmread(record.path)
        expected = (
            dataset.pixel_array.astype(np.float64) * record.slope + record.intercept
        )
        if observed.shape != expected.shape:
            raise PipelineError("Full-volume HU plane shape does not match DICOM pixels.")
        observed_float = np.asarray(observed, dtype=np.float64)
        finite = np.isfinite(expected) & np.isfinite(observed_float)
        nonfinite_count += int(finite.size - np.count_nonzero(finite))
        differences = np.abs(observed_float[finite] - expected[finite])
        voxel_count += int(expected.size)
        finite_error_count += int(differences.size)
        if differences.size:
            error_sum += float(np.sum(differences, dtype=np.float64))
            max_error = max(max_error, float(np.max(differences)))

    return {
        "voxel_count": voxel_count,
        "finite_error_count": finite_error_count,
        "nonfinite_count": nonfinite_count,
        "mean_absolute_error": (
            error_sum / finite_error_count if finite_error_count else None
        ),
        "max_absolute_error": max_error if finite_error_count else None,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['summary']}"
        for item in report["checks"]
    )
    issues = "\n".join(f"- {issue}" for issue in report["issues"]) or "- None"
    return f"""# Automated QC: {report['case']}/{report['series']}

Status: **{report['status']}**
Generated: {report['generated_at_utc']}

## Checks

{checks}

## Blocking issues

{issues}

Manual 3D Slicer and clinical review remain **PENDING** after automated checks pass.
"""


def _verify_preflighted_conversion(
    *,
    work_root: Path,
    preflight: PreflightResult,
    case_label: str,
    series_label: str,
    artifact_dir: Path,
    inventory_map_path: Path | None,
    pipeline_provenance: dict[str, Any] | None,
    require_clean_provenance: bool,
    strict_storage_encoding: tuple[str, float, float] | None,
) -> dict[str, Any]:
    neutral_name = f"{case_label}_{series_label}"
    pilot_dir = artifact_dir.resolve()
    work = work_root.resolve()
    if pilot_dir == work or not is_within(pilot_dir, work):
        raise PipelineError("The conversion artifact directory must be inside the work root.")
    nifti_path = pilot_dir / f"{neutral_name}.nii"
    sidecar_path = pilot_dir / f"{neutral_name}.json"
    if not pilot_dir.is_dir() or not nifti_path.is_file() or not sidecar_path.is_file():
        raise PipelineError("The converted output is incomplete; run conversion first.")

    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    current_pipeline = (
        dict(pipeline_provenance)
        if pipeline_provenance is not None
        else current_code_provenance(require_clean=require_clean_provenance)
    )

    def add_check(name: str, passed: bool, summary: str, failure: str | None = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "summary": summary})
        if not passed:
            issues.append(failure or name)

    image = nib.load(nifti_path, mmap=True)
    shape = tuple(int(item) for item in image.shape)
    shape_ok = len(shape) == 3 and int(np.prod(shape)) == (
        len(preflight.records) * preflight.rows * preflight.columns
    )
    add_check(
        "voxel count",
        shape_ok,
        f"shape={shape}; expected voxel count={len(preflight.records) * preflight.rows * preflight.columns}",
        "NIfTI shape or voxel count does not match the selected DICOM series",
    )

    sform, sform_code = image.get_sform(coded=True)
    qform, qform_code = image.get_qform(coded=True)
    forms_valid = int(sform_code) > 0 and int(qform_code) > 0
    add_check(
        "qform and sform present",
        forms_valid,
        f"qform_code={int(qform_code)}; sform_code={int(sform_code)}",
        "NIfTI qform or sform is not coded",
    )
    forms_match = forms_valid and bool(np.allclose(sform, qform, atol=1e-4, rtol=0.0))
    maximum_form_delta = (
        float(np.max(np.abs(np.asarray(sform) - np.asarray(qform)))) if forms_valid else None
    )
    add_check(
        "qform and sform match",
        forms_match,
        f"maximum matrix delta={maximum_form_delta}",
        "NIfTI qform and sform differ beyond 0.0001",
    )

    affine = np.asarray(sform if int(sform_code) > 0 else qform, dtype=float)
    try:
        inverse_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise PipelineError("The NIfTI affine is singular.") from exc

    expected_zooms = sorted(
        [preflight.pixel_spacing[0], preflight.pixel_spacing[1], preflight.slice_spacing]
    )
    actual_zooms = sorted(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))
    spacing_ok = bool(np.allclose(expected_zooms, actual_zooms, atol=0.01, rtol=0.0))
    add_check(
        "physical spacing",
        spacing_ok,
        f"DICOM={expected_zooms}; NIfTI={actual_zooms}",
        "NIfTI affine spacing does not match DICOM geometry",
    )

    max_index_residual = 0.0
    max_world_residual = 0.0
    all_inside = True
    rounded_points: list[np.ndarray] = []
    center_indices: list[np.ndarray] = []
    for record in preflight.records:
        sample_pixels = (
            (0, 0),
            (0, record.columns - 1),
            (record.rows - 1, 0),
            (record.rows - 1, record.columns - 1),
            (record.rows // 2, record.columns // 2),
        )
        for row, column in sample_pixels:
            point_ras = _dicom_pixel_ras(record, row, column)
            floating, rounded = _map_to_nifti(point_ras, inverse_affine)
            max_index_residual = max(
                max_index_residual, float(np.max(np.abs(floating - rounded)))
            )
            reconstructed = (affine @ np.append(rounded, 1.0))[:3]
            max_world_residual = max(
                max_world_residual, float(np.linalg.norm(reconstructed - point_ras))
            )
            if any(index < 0 or index >= shape[axis] for axis, index in enumerate(rounded)):
                all_inside = False
            rounded_points.append(rounded)
            if row == record.rows // 2 and column == record.columns // 2:
                center_indices.append(rounded)

    geometry_ok = all_inside and max_index_residual <= 1e-3 and max_world_residual <= 0.05
    add_check(
        "DICOM-to-NIfTI physical coordinates",
        geometry_ok,
        f"max index residual={max_index_residual:.6g} voxel; max world residual={max_world_residual:.6g} mm",
        "DICOM pixel centers do not map exactly into the NIfTI grid",
    )

    point_array = np.asarray(rounded_points, dtype=int)
    extent_ok = shape_ok and all(
        int(point_array[:, axis].min()) == 0
        and int(point_array[:, axis].max()) == shape[axis] - 1
        for axis in range(3)
    )
    add_check(
        "full field-of-view extent",
        extent_ok,
        "DICOM corners span the full NIfTI index extent" if extent_ok else "extent mismatch",
        "NIfTI coverage is cropped, padded, or incomplete",
    )

    center_array = np.asarray(center_indices, dtype=int)
    steps = np.diff(center_array, axis=0)
    unique_steps = {tuple(int(item) for item in step) for step in steps}
    slice_order_ok = (
        len(center_indices) == len(preflight.records)
        and len({tuple(item) for item in center_indices}) == len(center_indices)
        and len(unique_steps) == 1
        and sum(abs(item) for item in next(iter(unique_steps), ())) == 1
    )
    add_check(
        "slice sequence",
        slice_order_ok,
        f"unique voxel step={sorted(unique_steps)}",
        "DICOM planes do not map to one complete consecutive NIfTI axis",
    )

    header_description = bytes(image.header["descrip"]).rstrip(b"\x00")
    header_data_type = bytes(image.header["data_type"]).rstrip(b"\x00")
    header_database_name = bytes(image.header["db_name"]).rstrip(b"\x00")
    header_auxiliary = bytes(image.header["aux_file"]).rstrip(b"\x00")
    header_intent = bytes(image.header["intent_name"]).rstrip(b"\x00")
    header_safe = (
        header_description == NEUTRAL_NIFTI_DESCRIPTION
        and not header_data_type
        and not header_database_name
        and not header_auxiliary
        and not header_intent
        and len(image.header.extensions) == 0
    )
    add_check(
        "NIfTI header privacy allowlist",
        header_safe,
        "neutral descrip; empty legacy/free-text fields and extensions" if header_safe else "unexpected header content",
        "NIfTI header privacy fields are not sanitized",
    )
    sidecar_issues = verify_sanitized_sidecar(sidecar_path)
    add_check(
        "sidecar privacy allowlist",
        not sidecar_issues,
        "sanitized key allowlist" if not sidecar_issues else "; ".join(sidecar_issues),
        "conversion sidecar contains unexpected metadata",
    )

    source_manifest = read_json(pilot_dir / "source_hashes.json")
    stored_hashes = source_manifest.get("slices", [])
    current_hashes = [
        {"slice_index": index, "sha256": sha256_file(record.path)}
        for index, record in enumerate(preflight.records)
    ]
    hashes_ok = stored_hashes == current_hashes
    add_check(
        "source immutability",
        hashes_ok,
        f"{len(current_hashes)} source hashes unchanged" if hashes_ok else "hash mismatch",
        "source DICOM hashes changed after conversion",
    )

    conversion_report = read_json(pilot_dir / "conversion_report.json")
    tool_provenance = read_json(pilot_dir / "tool_provenance.json")
    current_manifest_digest = sha256_canonical_json(current_hashes)
    selection_identity = _selection_identity_sha256(preflight.records)
    selection_binding_ok = (
        source_manifest.get("ordered_manifest_sha256") == current_manifest_digest
        and conversion_report.get("ordered_source_manifest_sha256")
        == current_manifest_digest
        and conversion_report.get("selection_identity_sha256") == selection_identity
    )
    if inventory_map_path is not None:
        selection_binding_ok = (
            selection_binding_ok
            and conversion_report.get("inventory_uid_map_sha256")
            == sha256_file(inventory_map_path)
        )
    binding_summary = (
        "ordered source manifest and local UID identity digest match"
        if inventory_map_path is None
        else "inventory, ordered source manifest, and local UID identity digests match"
    )
    add_check(
        "approved selection binding",
        selection_binding_ok,
        binding_summary if selection_binding_ok else "selection binding mismatch",
        "conversion output is not bound to the current ordered source manifest",
    )
    provenance_ok = tool_provenance.get("pipeline") == current_pipeline
    if current_pipeline.get("git_dirty") is True:
        provenance_summary = "pipeline and lock hashes match; uncommitted changes were recorded"
    elif current_pipeline.get("git_commit"):
        provenance_summary = "clean committed pipeline and lock hashes match conversion provenance"
    else:
        provenance_summary = "supplied pipeline provenance matches conversion provenance"
    add_check(
        "pipeline provenance",
        provenance_ok,
        provenance_summary if provenance_ok else "pipeline provenance mismatch",
        "the current pipeline revision or dependency locks differ from conversion provenance",
    )

    proxy_slope = _safe_float(getattr(image.dataobj, "slope", None))
    proxy_intercept = _safe_float(getattr(image.dataobj, "inter", None))
    if strict_storage_encoding is not None:
        expected_dtype, expected_slope, expected_intercept = strict_storage_encoding
        dtype_scaling_ok = (
            str(image.get_data_dtype()) == expected_dtype
            and proxy_slope == expected_slope
            and proxy_intercept == expected_intercept
        )
        add_check(
            "NIfTI dtype and scaling",
            dtype_scaling_ok,
            f"dtype={image.get_data_dtype()}; slope={proxy_slope}; intercept={proxy_intercept}",
            "NIfTI storage dtype or scaling differs from the audited CT encoding",
        )

    hu_metrics = _full_volume_hu_metrics(image, preflight.records, inverse_affine)
    expected_voxels = len(preflight.records) * preflight.rows * preflight.columns
    hu_ok = (
        hu_metrics["voxel_count"] == expected_voxels
        and hu_metrics["finite_error_count"] == expected_voxels
        and hu_metrics["nonfinite_count"] == 0
        and hu_metrics["max_absolute_error"] is not None
        and hu_metrics["max_absolute_error"] <= 1.0
    )
    add_check(
        "full-volume HU preservation",
        hu_ok,
        f"voxels={hu_metrics['voxel_count']}; nonfinite={hu_metrics['nonfinite_count']}; mean error={hu_metrics['mean_absolute_error']}; max error={hu_metrics['max_absolute_error']} HU",
        "Every NIfTI voxel was not finite and equal to the corresponding DICOM HU within 1 HU",
    )

    report = {
        "schema_version": 1,
        "status": "AUTOMATED_PASS_PENDING_MANUAL" if not issues else "FAIL",
        "generated_at_utc": utc_now(),
        "case": case_label,
        "series": series_label,
        "checks": checks,
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(
            dict.fromkeys(
                [
                    *preflight.warnings,
                    *(
                        ["pipeline provenance records uncommitted source changes"]
                        if current_pipeline.get("git_dirty") is True
                        else []
                    ),
                ]
            )
        ),
        "metrics": {
            "nifti_shape": list(shape),
            "nifti_dtype": str(image.get_data_dtype()),
            "nifti_proxy_slope": proxy_slope,
            "nifti_proxy_intercept": proxy_intercept,
            "max_index_residual_voxel": max_index_residual,
            "max_world_residual_mm": max_world_residual,
            "hu_voxel_count": hu_metrics["voxel_count"],
            "hu_nonfinite_count": hu_metrics["nonfinite_count"],
            "mean_absolute_hu_error": hu_metrics["mean_absolute_error"],
            "max_absolute_hu_error": hu_metrics["max_absolute_error"],
            "nifti_sha256": sha256_file(nifti_path),
        },
        "manual_slicer_qc": "PENDING",
        "clinical_series_and_coverage_review": "PENDING",
    }
    atomic_write_json(pilot_dir / "qc_report.json", report)
    atomic_write_text(pilot_dir / "qc_report.md", _report_markdown(report))
    return report


def verify_conversion(
    dicom_root: Path,
    work_root: Path,
    paths: Iterable[Path],
    case_label: str,
    series_label: str,
    *,
    expected_study_uid: str | None = None,
    expected_series_uid: str | None = None,
    artifact_dir: Path | None = None,
    pipeline_provenance: dict[str, Any] | None = None,
    require_clean_provenance: bool = True,
) -> dict[str, Any]:
    """Verify one generic conversion against its selected source series."""
    validate_labels(case_label, series_label)
    validate_roots(dicom_root, work_root, repo_root())
    selected = _validate_selected_paths(dicom_root, paths)
    preflight = preflight_series(selected, expected_study_uid, expected_series_uid)
    neutral_name = f"{case_label}_{series_label}"
    destination = artifact_dir or (work_root / "conversions" / neutral_name)
    return _verify_preflighted_conversion(
        work_root=work_root,
        preflight=preflight,
        case_label=case_label,
        series_label=series_label,
        artifact_dir=destination,
        inventory_map_path=None,
        pipeline_provenance=pipeline_provenance,
        require_clean_provenance=require_clean_provenance,
        strict_storage_encoding=None,
    )


def verify_pilot(
    dicom_root: Path,
    work_root: Path,
    case_label: str,
    series_label: str,
) -> dict[str, Any]:
    validate_roots(dicom_root, work_root, repo_root())
    paths, study_uid, series_uid = resolve_selection(
        dicom_root, work_root, case_label, series_label
    )
    preflight = preflight_series(paths, study_uid, series_uid)
    assert_pilot_fingerprint(preflight, case_label, series_label)
    neutral_name = f"{case_label}_{series_label}"
    return _verify_preflighted_conversion(
        work_root=work_root,
        preflight=preflight,
        case_label=case_label,
        series_label=series_label,
        artifact_dir=work_root / "pilot" / neutral_name,
        inventory_map_path=work_root / "inventory" / "uid_map.local.json",
        pipeline_provenance=None,
        require_clean_provenance=True,
        strict_storage_encoding=("int16", 1.0, -1024.0),
    )
