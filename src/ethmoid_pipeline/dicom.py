from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from .util import PipelineError


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    study_uid: str
    series_uid: str
    sop_uid: str
    frame_uid: str
    acquisition_number: str | None
    irradiation_event_uid: str | None
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float
    spacing_between_slices: float | None
    iop: tuple[float, float, float, float, float, float]
    ipp: tuple[float, float, float]
    slope: float
    intercept: float
    instance_number: int | None
    coordinate: float


@dataclass(frozen=True)
class PreflightResult:
    records: tuple[SliceRecord, ...]
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_spacing: float
    slice_thickness: float
    kernel: str
    image_type: tuple[str, ...]
    iop: tuple[float, float, float, float, float, float]
    transfer_syntax: str
    warnings: tuple[str, ...]
    private_tags_present: bool
    source_kind: str = "native"
    profile: str = "strict"

    def safe_report(self, case_label: str, series_label: str) -> dict[str, Any]:
        return {
            "status": "PASS",
            "case": case_label,
            "series": series_label,
            "image_count": len(self.records),
            "rows": self.rows,
            "columns": self.columns,
            "pixel_spacing_mm": list(self.pixel_spacing),
            "slice_spacing_mm": self.slice_spacing,
            "slice_thickness_mm": self.slice_thickness,
            "kernel": self.kernel,
            "image_type": list(self.image_type),
            "image_orientation_patient": list(self.iop),
            "intensity_units": "HU",
            "transfer_syntax": self.transfer_syntax,
            "private_tags_present": self.private_tags_present,
            "warnings": list(self.warnings),
            "source_kind": self.source_kind,
            "preflight_profile": self.profile,
        }


class PreflightFailure(PipelineError):
    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(dict.fromkeys(issues))
        super().__init__("Series preflight failed: " + "; ".join(self.issues))


def _as_float_tuple(value: Any, expected: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is missing or non-numeric") from exc
    if len(result) != expected or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} has an invalid shape or value")
    return result


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is missing or non-numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is non-finite")
    return result


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip().upper() for part in value.split("\\") if part.strip())
    try:
        return tuple(str(part).strip().upper() for part in value if str(part).strip())
    except TypeError:
        return (str(value).strip().upper(),)


def _string_or_none(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _optional_int(value: Any, field: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is non-numeric") from exc


def _angle_degrees(
    left: np.ndarray, right: np.ndarray, *, undirected: bool = False
) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 180.0
    dot = float(np.dot(left, right))
    if undirected:
        dot = abs(dot)
    cosine = float(np.clip(dot / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _constant_numeric(
    values: list[tuple[float, ...]], tolerance: float, field: str, issues: list[str]
) -> None:
    baseline = np.asarray(values[0], dtype=float)
    if any(np.max(np.abs(np.asarray(value, dtype=float) - baseline)) > tolerance for value in values[1:]):
        issues.append(f"{field} changes within the series")


def _constant_exact(values: list[Any], field: str, issues: list[str]) -> None:
    if len(set(values)) != 1:
        issues.append(f"{field} changes within the series")


def preflight_series(
    paths: Iterable[Path], expected_study_uid: str | None = None, expected_series_uid: str | None = None,
    *, profile: str = "strict",
) -> PreflightResult:
    if profile not in {"strict", "exploratory"}:
        raise PreflightFailure(["unknown preflight profile"])
    input_paths = list(paths)
    if not input_paths:
        raise PreflightFailure(["no source files were selected"])

    issues: list[str] = []
    warnings: list[str] = []
    pending: list[dict[str, Any]] = []
    private_tags_present = False
    burned_values: set[str | None] = set()
    recognizable_values: set[str | None] = set()
    identity_removed_values: set[str | None] = set()

    for path in input_paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            issues.append("a selected DICOM could not be parsed")
            continue
        try:
            transfer_syntax = str(dataset.file_meta.TransferSyntaxUID)
            study_uid = str(dataset.StudyInstanceUID)
            series_uid = str(dataset.SeriesInstanceUID)
            sop_uid = str(dataset.SOPInstanceUID)
            frame_uid = str(dataset.FrameOfReferenceUID)
            rows = int(dataset.Rows)
            columns = int(dataset.Columns)
            pixel_spacing = _as_float_tuple(dataset.PixelSpacing, 2, "PixelSpacing")
            slice_thickness = _as_float(dataset.SliceThickness, "SliceThickness")
            spacing_between = (
                _as_float(dataset.SpacingBetweenSlices, "SpacingBetweenSlices")
                if hasattr(dataset, "SpacingBetweenSlices")
                else None
            )
            iop = _as_float_tuple(dataset.ImageOrientationPatient, 6, "ImageOrientationPatient")
            ipp = _as_float_tuple(dataset.ImagePositionPatient, 3, "ImagePositionPatient")
            slope = _as_float(dataset.RescaleSlope, "RescaleSlope")
            intercept = _as_float(dataset.RescaleIntercept, "RescaleIntercept")
            number_of_frames = _optional_int(
                getattr(dataset, "NumberOfFrames", 1), "NumberOfFrames"
            )
            instance_number = _optional_int(
                getattr(dataset, "InstanceNumber", None), "InstanceNumber"
            )
            if slope == 0:
                raise ValueError("RescaleSlope is zero")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            issues.append(str(exc) if str(exc) else "required CT geometry metadata is missing")
            continue

        image_type = _text_tuple(getattr(dataset, "ImageType", None))
        description = str(getattr(dataset, "SeriesDescription", "")).upper()
        kernel = str(getattr(dataset, "ConvolutionKernel", "")).upper()
        rescale_type = str(getattr(dataset, "RescaleType", "")).strip().upper()
        multienergy = str(
            getattr(dataset, "MultienergyCTAcquisition", "")
        ).strip().upper()
        modality = str(getattr(dataset, "Modality", "")).upper()
        sop_class = str(getattr(dataset, "SOPClassUID", ""))
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
        bits = (
            int(getattr(dataset, "BitsAllocated", -1)),
            int(getattr(dataset, "BitsStored", -1)),
            int(getattr(dataset, "HighBit", -1)),
            int(getattr(dataset, "PixelRepresentation", -1)),
            int(getattr(dataset, "SamplesPerPixel", -1)),
        )
        localizer_terms = {"LOCALIZER", "SCOUT", "TOPOGRAM"}
        derived_terms = {"DERIVED", "SECONDARY", "REFORMATTED", "MPR", "MIP", "MINIP"}

        if modality != "CT" or sop_class != str(CTImageStorage):
            issues.append("the selection contains a non-classic-CT image")
        if transfer_syntax != str(ExplicitVRLittleEndian):
            issues.append("the selection uses an unsupported transfer syntax")
        if number_of_frames != 1:
            issues.append("multi-frame CT is outside milestone-one scope")
        if photometric != "MONOCHROME2":
            issues.append("photometric interpretation is not MONOCHROME2")
        if rescale_type != "HU":
            issues.append("RescaleType is not explicitly HU")
        if multienergy not in {"", "NO"} and profile == "strict":
            issues.append("multienergy CT is outside milestone-one HU validation scope")
        if profile == "strict" and not {"ORIGINAL", "PRIMARY", "AXIAL"}.issubset(set(image_type)):
            issues.append("images are not ORIGINAL/PRIMARY/AXIAL")
        combined_terms = set(image_type) | set(description.replace("/", " ").split())
        combined_terms.update(term for item in image_type for term in item.split())
        if combined_terms & localizer_terms:
            issues.append("a localizer/scout image is present")
        if profile == "strict" and combined_terms & derived_terms:
            issues.append("a derived/reformatted image is present")
        if profile == "exploratory":
            if combined_terms & {"MIP", "MINIP", "AVERAGE", "PROJECTION"}:
                issues.append("a projection or slab-average image is not a native 3D volume")
            if not ({"ORIGINAL", "PRIMARY"}.issubset(set(image_type)) or
                    {"DERIVED", "SECONDARY"}.issubset(set(image_type)) and
                    bool(combined_terms & {"REFORMATTED", "MPR"})):
                issues.append("the image is neither native CT nor an identified volumetric reformat")
            if multienergy not in {"", "NO"} or "VMI" in combined_terms:
                warnings.append("calibrated multienergy reconstruction is exploratory")
            if combined_terms & {"DERIVED", "REFORMATTED", "MPR"}:
                warnings.append("derived volumetric reconstruction is exploratory")
            if rows <= 0 or columns <= 0 or min(pixel_spacing) <= 0 or slice_thickness <= 0:
                issues.append("image dimensions, spacing and thickness must be positive")
        if any(item < 0 for item in bits):
            issues.append("pixel encoding metadata is incomplete")

        burned = _string_or_none(getattr(dataset, "BurnedInAnnotation", None))
        recognizable = _string_or_none(getattr(dataset, "RecognizableVisualFeatures", None))
        identity_removed = _string_or_none(getattr(dataset, "PatientIdentityRemoved", None))
        burned_values.add(burned.upper() if burned else None)
        recognizable_values.add(recognizable.upper() if recognizable else None)
        identity_removed_values.add(identity_removed.upper() if identity_removed else None)
        if burned and burned.upper() == "YES":
            issues.append("BurnedInAnnotation is YES")
        if recognizable and recognizable.upper() == "YES":
            issues.append("RecognizableVisualFeatures is YES")
        if any(0x6000 <= element.tag.group <= 0x60FF for element in dataset.iterall()):
            issues.append("overlay data is present")
        if hasattr(dataset, "GraphicAnnotationSequence"):
            issues.append("graphic annotation data is present")
        private_tags_present = private_tags_present or any(
            element.tag.is_private for element in dataset.iterall()
        )

        row = np.asarray(iop[:3], dtype=float)
        column = np.asarray(iop[3:], dtype=float)
        normal = np.cross(row, column)
        coordinate = float(np.dot(np.asarray(ipp, dtype=float), normal))
        pending.append(
            {
                "path": path,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_uid": sop_uid,
                "frame_uid": frame_uid,
                "acquisition_number": _string_or_none(getattr(dataset, "AcquisitionNumber", None)),
                "irradiation_event_uid": _string_or_none(
                    getattr(dataset, "IrradiationEventUID", None)
                ),
                "rows": rows,
                "columns": columns,
                "pixel_spacing": pixel_spacing,
                "slice_thickness": slice_thickness,
                "spacing_between_slices": spacing_between,
                "iop": iop,
                "ipp": ipp,
                "slope": slope,
                "intercept": intercept,
                "instance_number": instance_number,
                "coordinate": coordinate,
                "image_type": image_type,
                "source_kind": "mpr" if combined_terms & {"DERIVED", "REFORMATTED", "MPR"} else "native",
                "kernel": kernel,
                "rescale_type": rescale_type,
                "multienergy": multienergy,
                "transfer_syntax": transfer_syntax,
                "photometric": photometric,
                "bits": bits,
            }
        )

    if not pending:
        raise PreflightFailure(issues or ["no readable DICOM images remain"])

    study_uids = [item["study_uid"] for item in pending]
    series_uids = [item["series_uid"] for item in pending]
    _constant_exact(study_uids, "StudyInstanceUID", issues)
    _constant_exact(series_uids, "SeriesInstanceUID", issues)
    if expected_study_uid and set(study_uids) != {expected_study_uid}:
        issues.append("the selection no longer matches the inventoried study")
    if expected_series_uid and set(series_uids) != {expected_series_uid}:
        issues.append("the selection no longer matches the inventoried series")

    _constant_exact([item["frame_uid"] for item in pending], "FrameOfReferenceUID", issues)
    _constant_exact([item["rows"] for item in pending], "Rows", issues)
    _constant_exact([item["columns"] for item in pending], "Columns", issues)
    _constant_exact([item["kernel"] for item in pending], "ConvolutionKernel", issues)
    _constant_exact([item["rescale_type"] for item in pending], "RescaleType", issues)
    _constant_exact([item["multienergy"] for item in pending], "MultienergyCTAcquisition", issues)
    _constant_exact([item["transfer_syntax"] for item in pending], "TransferSyntaxUID", issues)
    _constant_exact([item["photometric"] for item in pending], "PhotometricInterpretation", issues)
    _constant_exact([item["bits"] for item in pending], "pixel encoding", issues)
    if profile == "exploratory":
        _constant_exact([item["image_type"] for item in pending], "ImageType", issues)
    _constant_numeric([item["pixel_spacing"] for item in pending], 1e-4, "PixelSpacing", issues)
    _constant_numeric([(item["slice_thickness"],) for item in pending], 0.01, "SliceThickness", issues)
    _constant_numeric([(item["slope"], item["intercept"]) for item in pending], 1e-8, "HU rescale", issues)

    spacing_values = [item["spacing_between_slices"] for item in pending]
    if all(value is None for value in spacing_values):
        warnings.append("SpacingBetweenSlices is absent; IPP geometry is authoritative")
    elif any(value is None for value in spacing_values):
        issues.append("SpacingBetweenSlices is inconsistently present")
    else:
        _constant_numeric([(float(value),) for value in spacing_values], 0.01, "SpacingBetweenSlices", issues)

    baseline_iop = np.asarray(pending[0]["iop"], dtype=float)
    baseline_row = baseline_iop[:3]
    baseline_column = baseline_iop[3:]
    if abs(np.linalg.norm(baseline_row) - 1.0) > 1e-3:
        issues.append("IOP row direction is not unit length")
    if abs(np.linalg.norm(baseline_column) - 1.0) > 1e-3:
        issues.append("IOP column direction is not unit length")
    if abs(float(np.dot(baseline_row, baseline_column))) > 1e-3:
        issues.append("IOP row and column directions are not orthogonal")
    baseline_normal = np.cross(baseline_row, baseline_column)
    axial_angle = _angle_degrees(
        baseline_normal, np.asarray([0.0, 0.0, 1.0]), undirected=True
    )
    if axial_angle > 15.0 and profile == "strict":
        issues.append("series orientation is more than 15 degrees from axial")
    elif axial_angle > 5.0:
        warnings.append("series is oblique or non-axial" if profile == "exploratory" else "series is axial-oblique by more than 5 degrees")
    for item in pending[1:]:
        iop = np.asarray(item["iop"], dtype=float)
        if max(_angle_degrees(iop[:3], baseline_row), _angle_degrees(iop[3:], baseline_column)) > 0.1:
            issues.append("ImageOrientationPatient changes within the series")
            break

    # All positions must use the one validated baseline normal. Computing each
    # coordinate from its own IOP can hide a direction reversal.
    for item in pending:
        item["coordinate"] = float(
            np.dot(np.asarray(item["ipp"], dtype=float), baseline_normal)
        )

    sop_uids = [item["sop_uid"] for item in pending]
    if len(set(sop_uids)) != len(sop_uids):
        issues.append("duplicate SOPInstanceUID values are present")
    for field, title in (
        ("acquisition_number", "AcquisitionNumber"),
        ("irradiation_event_uid", "IrradiationEventUID"),
    ):
        groups = {item[field] if item[field] is not None else "<MISSING>" for item in pending}
        if len(groups) > 1:
            issues.append(f"multiple {title} groups are present")
        elif groups == {"<MISSING>"}:
            warnings.append(f"{title} is absent")

    pending.sort(key=lambda item: item["coordinate"])
    if profile == "exploratory":
        origin = np.asarray(pending[0]["ipp"], dtype=float)
        normal_unit = baseline_normal / np.linalg.norm(baseline_normal) if np.linalg.norm(baseline_normal) else baseline_normal
        for item in pending[1:]:
            offset = np.asarray(item["ipp"], dtype=float) - origin
            if np.linalg.norm(offset - np.dot(offset, normal_unit) * normal_unit) > 0.01:
                issues.append("slice positions contain in-plane drift or shear")
                break
    coordinates = np.asarray([item["coordinate"] for item in pending], dtype=float)
    if len(coordinates) < 2:
        issues.append("fewer than two spatial planes are present")
        median_spacing = float("nan")
    else:
        gaps = np.diff(coordinates)
        if np.any(gaps <= 0.01):
            issues.append("duplicate or near-duplicate projected slice positions are present")
        median_spacing = float(np.median(gaps))
        tolerance = max(0.01, abs(median_spacing) * 0.01)
        if np.any(np.abs(gaps - median_spacing) > tolerance):
            issues.append("projected slice spacing is nonuniform")
        if np.any(gaps > 1.5 * median_spacing):
            issues.append("a likely missing-plane gap is present")

    if burned_values == {None}:
        warnings.append("BurnedInAnnotation is absent; visual pixel review remains mandatory")
    if recognizable_values == {None}:
        warnings.append("RecognizableVisualFeatures is absent")
    if identity_removed_values != {"YES"}:
        warnings.append("source is not documented as deidentified")
    if private_tags_present:
        warnings.append("private DICOM tags are present and are not propagated intentionally")

    if issues:
        raise PreflightFailure(issues)

    records = tuple(
        SliceRecord(
            path=item["path"],
            study_uid=item["study_uid"],
            series_uid=item["series_uid"],
            sop_uid=item["sop_uid"],
            frame_uid=item["frame_uid"],
            acquisition_number=item["acquisition_number"],
            irradiation_event_uid=item["irradiation_event_uid"],
            rows=item["rows"],
            columns=item["columns"],
            pixel_spacing=item["pixel_spacing"],
            slice_thickness=item["slice_thickness"],
            spacing_between_slices=item["spacing_between_slices"],
            iop=item["iop"],
            ipp=item["ipp"],
            slope=item["slope"],
            intercept=item["intercept"],
            instance_number=item["instance_number"],
            coordinate=item["coordinate"],
        )
        for item in pending
    )
    first = pending[0]
    return PreflightResult(
        records=records,
        rows=first["rows"],
        columns=first["columns"],
        pixel_spacing=first["pixel_spacing"],
        slice_spacing=median_spacing,
        slice_thickness=first["slice_thickness"],
        kernel=first["kernel"],
        image_type=first["image_type"],
        iop=first["iop"],
        transfer_syntax=first["transfer_syntax"],
        warnings=tuple(dict.fromkeys(warnings)),
        private_tags_present=private_tags_present,
        source_kind=first["source_kind"],
        profile=profile,
    )


def assert_pilot_fingerprint(result: PreflightResult, case_label: str, series_label: str) -> None:
    if (case_label, series_label) != ("Case003", "Series007"):
        raise PipelineError("Milestone one is restricted to audited Case003/Series007.")
    mismatches: list[str] = []
    if len(result.records) != 224:
        mismatches.append("slice count")
    if (result.rows, result.columns) != (512, 512):
        mismatches.append("matrix")
    if abs(result.slice_spacing - 0.625) > 0.001:
        mismatches.append("IPP spacing")
    if abs(result.slice_thickness - 0.625) > 0.001:
        mismatches.append("slice thickness")
    if not np.allclose(result.pixel_spacing, (0.488281, 0.488281), atol=1e-6, rtol=0.0):
        mismatches.append("in-plane pixel spacing")
    if not np.allclose(result.iop, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), atol=1e-6, rtol=0.0):
        mismatches.append("image orientation")
    if result.kernel != "BONEPLUS":
        mismatches.append("reconstruction kernel")
    if mismatches:
        raise PipelineError(
            "The neutral pilot label no longer matches the audited fingerprint: "
            + ", ".join(mismatches)
            + "."
        )


def sensitive_values(paths: Iterable[Path]) -> set[str]:
    values: set[str] = set()
    keywords = (
        "PatientName",
        "PatientID",
        "IssuerOfPatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "IrradiationEventUID",
    )
    first_path = next(iter(paths), None)
    if first_path is None:
        return values
    dataset = pydicom.dcmread(first_path, stop_before_pixels=True)
    for keyword in keywords:
        value = getattr(dataset, keyword, None)
        if value is not None and str(value).strip():
            values.add(str(value))
    return values
