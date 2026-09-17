"""Acquisition facts and resolution warnings, independent of inference libraries.

Thickness is a DICOM acquisition fact, never inferred from a NIfTI voxel step.
Only a server-created, source-bound acquisition snapshot is accepted by a worker.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def acquisition_snapshot(value: dict[str, Any] | None, source_sha256: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Acquisition metadata must be an object")
    if value.get("source_sha256", source_sha256) != source_sha256:
        raise ValueError("Acquisition metadata belongs to another source")
    result: dict[str, Any] = {"source_sha256": source_sha256}
    for name, pattern in (("study_id", r"[A-Za-z0-9_-]{1,100}"), ("series_id", r"[A-Za-z0-9_-]{1,100}"),
                          ("study_label", r"Case[0-9]{3,}"), ("series_label", r"Series[0-9]{3,}")):
        item = value.get(name)
        if item is not None and (not isinstance(item, str) or not re.fullmatch(pattern, item)):
            raise ValueError("Acquisition identity is not neutral")
        result[name] = item
    thickness = value.get("slice_thickness_mm")
    if thickness is not None and (type(thickness) not in (float, int) or not math.isfinite(thickness) or thickness <= 0):
        raise ValueError("Acquisition thickness is invalid")
    result["slice_thickness_mm"] = float(thickness) if thickness is not None else None
    spacing = value.get("spacing_mm")
    if spacing is not None and (not isinstance(spacing, (list, tuple)) or len(spacing) != 3 or any(
            type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in spacing)):
        raise ValueError("Acquisition spacing is invalid")
    result["spacing_mm"] = list(map(float, spacing)) if spacing is not None else None
    classification = value.get("resolution_class", "unknown")
    if classification not in {"within_protocol", "outside_protocol", "unknown"}:
        raise ValueError("Acquisition resolution classification is invalid")
    result["resolution_class"] = classification
    # Free-form catalog warnings and arbitrary identifiers are deliberately not
    # copied into reports; quality warnings below use fixed neutral wording.
    return result


def is_oblique(affine: np.ndarray) -> bool:
    matrix = np.asarray(affine, dtype=float)[:3, :3]
    direction = np.abs(matrix / np.linalg.norm(matrix, axis=0))
    axes = np.argmax(direction, axis=0)
    expected = np.zeros((3, 3))
    expected[axes, np.arange(3)] = 1
    return not np.allclose(direction, expected, atol=1e-4, rtol=0)


def resolution_quality(acquisition: dict[str, Any], affine: np.ndarray | None = None) -> dict[str, Any]:
    thickness = acquisition.get("slice_thickness_mm")
    acquired_spacing = acquisition.get("spacing_mm")
    native_spacing = [round(float(v), 9) for v in np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)] if affine is not None else None
    larger = any(v > 1.0 + 1e-6 for values in (native_spacing, acquired_spacing) if values for v in values)
    outside_resolution = larger or (thickness is not None and thickness > 1.0 + 1e-6)
    outside = outside_resolution or acquisition.get("resolution_class") == "outside_protocol"
    classification = "outside_protocol" if outside else "unknown" if thickness is None or native_spacing is None else "within_protocol"
    flags, warnings = [], []
    if outside_resolution:
        flags.append("out_of_protocol_resolution")
        warnings.append("Out-of-protocol experimental attempt: the source exceeds the clinician's 1 mm-or-less resolution criterion. Thin walls and small cells may remain unresolved.")
    elif outside:
        flags.append("outside_protocol_reconstruction")
        warnings.append("The selected reconstruction is outside the original protocol; this is an experimental attempt.")
    if thickness is None:
        flags.append("acquisition_thickness_unknown")
        warnings.append("Source slice thickness is unknown; voxel spacing is not a substitute for acquisition thickness.")
    oblique = bool(affine is not None and is_oblique(affine))
    if oblique:
        flags.append("oblique_localization_proxy")
        warnings.append("An anatomical localization proxy is resampled for this oblique source; cavity and air instances remain on the original source grid.")
    return {
        "resolution_class": classification,
        "out_of_protocol_resolution": outside_resolution if thickness is not None or larger else None,
        "slice_thickness_mm": thickness,
        "acquisition_spacing_mm": acquired_spacing,
        "slice_interval_mm": acquired_spacing[2] if acquired_spacing else None,
        "native_spacing_mm": native_spacing,
        "oblique_source": oblique,
        "flags": flags, "warnings": warnings,
        "clinical_accuracy": "unmeasured", "count_completeness": "unvalidated",
    }


def quality_note(quality: dict[str, Any] | None) -> str:
    """Fixed wording for presentation exports, with no user-supplied text."""
    classification = (quality or {}).get("resolution_class")
    if classification == "outside_protocol":
        return "OUT-OF-PROTOCOL EXPERIMENTAL ATTEMPT. Source resolution/reconstruction may hide cells and septa."
    if classification == "within_protocol":
        return "Source resolution meets the 1 mm criterion; anatomical correctness and count completeness remain unvalidated."
    return "Source acquisition thickness is unknown. Protocol resolution eligibility cannot be confirmed."
