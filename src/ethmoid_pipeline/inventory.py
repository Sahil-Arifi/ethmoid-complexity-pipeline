from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from .util import PipelineError, atomic_write_json, read_json, utc_now, validate_labels


INVENTORY_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "Modality",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "ImageType",
    "ConvolutionKernel",
]


def _has_dicom_preamble(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def discover_dicom_paths(root: Path) -> list[Path]:
    paths = [path for path in root.rglob("*") if path.is_file() and _has_dicom_preamble(path)]
    paths.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    return paths


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            return "\\".join(str(item) for item in value)
        except TypeError:
            pass
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _series_safe_summary(case_label: str, series_label: str, series: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(series.pop("_positions"), dtype=float)
    orientation = series.pop("_orientation")
    spacing_median: float | None = None
    spacing_regular: bool | None = None
    if len(positions) > 1 and orientation is not None:
        row = np.asarray(orientation[:3], dtype=float)
        column = np.asarray(orientation[3:], dtype=float)
        normal = np.cross(row, column)
        coordinate = np.sort(positions @ normal)
        gaps = np.diff(coordinate)
        spacing_median = float(np.median(gaps))
        tolerance = max(0.01, abs(spacing_median) * 0.01)
        spacing_regular = bool(np.all(np.abs(gaps - spacing_median) <= tolerance))
    return {
        "case": case_label,
        "series": series_label,
        "image_count": series["image_count"],
        "modality": series["modality"],
        "rows": series["rows"],
        "columns": series["columns"],
        "pixel_spacing_mm": series["pixel_spacing_mm"],
        "slice_thickness_mm": series["slice_thickness_mm"],
        "spacing_between_slices_mm": series["spacing_between_slices_mm"],
        "ipp_spacing_median_mm": spacing_median,
        "ipp_spacing_regular": spacing_regular,
        "kernel": series["kernel"],
        "image_type": series["image_type"],
        "transfer_syntax": series["transfer_syntax"],
    }


def build_inventory(dicom_root: Path, work_root: Path) -> dict[str, Any]:
    root = dicom_root.resolve(strict=True)
    paths = discover_dicom_paths(root)
    if not paths:
        raise PipelineError("No Part 10 DICOM files were found below the supplied root.")

    studies: OrderedDict[str, dict[str, Any]] = OrderedDict()
    unreadable = 0
    for path in paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=INVENTORY_TAGS)
        except Exception:
            unreadable += 1
            continue
        study_uid = str(getattr(dataset, "StudyInstanceUID", ""))
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not study_uid or not series_uid:
            raise PipelineError("A DICOM file is missing its study or series UID.")
        study = studies.setdefault(study_uid, {"series": OrderedDict()})
        series = study["series"].setdefault(
            series_uid,
            {
                "files": [],
                "first": dataset,
                "positions": [],
                "orientation": _float_list(getattr(dataset, "ImageOrientationPatient", None)),
            },
        )
        series["files"].append(path.relative_to(root).as_posix())
        position = _float_list(getattr(dataset, "ImagePositionPatient", None))
        if position is not None and len(position) == 3:
            series["positions"].append(position)

    if unreadable:
        raise PipelineError("At least one DICOM file could not be parsed; inventory was not promoted.")

    uid_cases: list[dict[str, Any]] = []
    redacted_series: list[dict[str, Any]] = []
    for case_index, (study_uid, study) in enumerate(studies.items(), start=1):
        case_label = f"Case{case_index:03d}"
        uid_series: list[dict[str, Any]] = []
        for series_index, (series_uid, series) in enumerate(study["series"].items(), start=1):
            series_label = f"Series{series_index:03d}"
            first = series["first"]
            transfer_syntax = str(getattr(first.file_meta, "TransferSyntaxUID", ""))
            uid_series.append(
                {
                    "series": series_label,
                    "series_uid": series_uid,
                    "relative_files": series["files"],
                }
            )
            safe = {
                "image_count": len(series["files"]),
                "modality": _safe_text(getattr(first, "Modality", None)),
                "rows": _optional_int(getattr(first, "Rows", None)),
                "columns": _optional_int(getattr(first, "Columns", None)),
                "pixel_spacing_mm": _float_list(getattr(first, "PixelSpacing", None)),
                "slice_thickness_mm": _optional_float(getattr(first, "SliceThickness", None)),
                "spacing_between_slices_mm": _optional_float(
                    getattr(first, "SpacingBetweenSlices", None)
                ),
                "kernel": _safe_text(getattr(first, "ConvolutionKernel", None)),
                "image_type": _safe_text(getattr(first, "ImageType", None)),
                "transfer_syntax": transfer_syntax,
                "_positions": series["positions"],
                "_orientation": series["orientation"],
            }
            redacted_series.append(_series_safe_summary(case_label, series_label, safe))
        uid_cases.append({"case": case_label, "study_uid": study_uid, "series": uid_series})

    inventory_dir = work_root / "inventory"
    uid_map = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "source_root": str(root),
        "dicom_file_count": len(paths),
        "cases": uid_cases,
    }
    redacted = {
        "schema_version": 1,
        "generated_at_utc": uid_map["generated_at_utc"],
        "dicom_file_count": len(paths),
        "study_count": len(uid_cases),
        "series_count": len(redacted_series),
        "series": redacted_series,
    }
    atomic_write_json(inventory_dir / "uid_map.local.json", uid_map)
    atomic_write_json(inventory_dir / "inventory.redacted.json", redacted)
    return {
        "dicom_file_count": len(paths),
        "study_count": len(uid_cases),
        "series_count": len(redacted_series),
        "uid_map": inventory_dir / "uid_map.local.json",
        "redacted_inventory": inventory_dir / "inventory.redacted.json",
    }


def resolve_selection(
    dicom_root: Path, work_root: Path, case_label: str, series_label: str
) -> tuple[list[Path], str, str]:
    validate_labels(case_label, series_label)
    mapping = read_json(work_root / "inventory" / "uid_map.local.json")
    root = dicom_root.resolve(strict=True)
    if Path(mapping.get("source_root", "")).resolve() != root:
        raise PipelineError("The local inventory belongs to a different DICOM root; rerun inventory.")
    for case in mapping.get("cases", []):
        if case.get("case") != case_label:
            continue
        for series in case.get("series", []):
            if series.get("series") != series_label:
                continue
            paths: list[Path] = []
            for relative in series.get("relative_files", []):
                candidate = (root / Path(relative)).resolve(strict=True)
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise PipelineError("The local inventory contains an unsafe source path.") from exc
                paths.append(candidate)
            if not paths:
                raise PipelineError("The selected neutral series has no files.")
            return paths, str(case["study_uid"]), str(series["series_uid"])
        raise PipelineError("The requested neutral series is absent from the local inventory.")
    raise PipelineError("The requested neutral case is absent from the local inventory.")
