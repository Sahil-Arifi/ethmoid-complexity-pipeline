"""Source-bound scan characterization; safe to import in the application runtime.

These measurements are engineering diagnostics, not diagnoses or validated noise
estimates. No policy depends on patient identity, case labels, or free-form text.
"""
from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .quality import acquisition_snapshot, resolution_quality

PROFILE_VERSION = "scan-profile-1"
PLAN_VERSION = "scan-adaptive-plan-1"
PROXY_MAX_VOXELS = 128_000_000


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(value: dict) -> dict:
    """Fresh JSON-compatible snapshot with a digest over every substantive field."""
    payload = {key: item for key, item in value.items() if key != "sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**json.loads(raw), "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def build_scan_profile(source: str | Path, acquisition: dict | None = None) -> dict:
    source = Path(source)
    digest = file_sha256(source)
    facts = acquisition_snapshot(acquisition, digest)
    image = nib.load(source)
    affine = np.asarray(image.affine, dtype=float)
    if len(image.shape) != 3 or not np.isfinite(affine).all():
        raise ValueError("Scan profile requires a finite 3D affine")
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3] / spacing
    if np.any(spacing <= 0) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Scan profile requires an orthogonal calibrated grid")
    # Sample at most 64^3 evenly distributed source voxels, with no CT resampling.
    steps = tuple(max(1, int(np.ceil(size / 64))) for size in image.shape)
    sample = np.asarray(image.dataobj[tuple(slice(None, None, step) for step in steps)], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("Scan profile contains nonfinite calibrated values")
    air = sample[sample < -800]
    air_mad = float(np.median(np.abs(air - np.median(air)))) if air.size >= 32 else None
    corners = nib.affines.apply_affine(affine, list(product(*[(-0.5, size - 0.5) for size in image.shape])))
    kind = (acquisition or {}).get("reconstruction_kind", "unknown")
    kernel = (acquisition or {}).get("kernel_class", "unknown")
    if kind not in {"native", "mpr", "unknown"} or kernel not in {"bone", "soft", "unknown"}:
        raise ValueError("Only normalized reconstruction categories may enter a scan profile")
    return sealed({
        "version": PROFILE_VERSION, "source_sha256": digest,
        "shape": list(image.shape), "affine": affine.tolist(),
        "native_spacing_mm": spacing.tolist(), "slice_thickness_mm": facts["slice_thickness_mm"],
        "slice_interval_mm": facts["spacing_mm"][2] if facts["spacing_mm"] else None,
        "axis_codes": list(nib.aff2axcodes(affine)),
        "physical_bounds_ras_mm": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
        "reconstruction_kind": kind, "kernel_class": kernel,
        "quality": resolution_quality(facts, affine),
        "image_diagnostics": {"sample_voxels": int(sample.size), "sampling_steps": list(steps),
            "hu_percentiles": [float(v) for v in np.percentile(sample, [0.1, 1, 50, 99, 99.9])],
            "air_sample_mad_hu": air_mad, "extreme_hu_fraction": float(np.mean((sample < -1100) | (sample > 4000))),
            "interpretation": "descriptive_sample_only_not_validated_noise_or_artifact_classification"},
        "anatomical_coverage": "requires_localization", "clinical_accuracy": "unmeasured",
    })


def build_processing_plan(profile: dict, *, assistance: dict | None = None, model_identity: dict | None = None) -> dict:
    if profile.get("version") != PROFILE_VERSION or sealed(profile) != profile:
        raise ValueError("The scan profile digest or version is invalid")
    quality = profile["quality"]
    category = "unknown_resolution" if quality["resolution_class"] == "unknown" else (
        "experimental_resolution" if quality["resolution_class"] == "outside_protocol" else "thin_resolution")
    if assistance is not None and model_identity is not None:
        raise ValueError("A processing plan cannot select two anatomical membership authorities")
    backend = "reviewer_native_region" if assistance is not None else "nnunet_v2_direct_anatomy" if model_identity is not None else "v2_geometric_baseline"
    lineage = {key: assistance.get(key) for key in ("revision_id", "parent_run_id", "region_sha256", "exclusion_sha256", "midline_ras_x")} if assistance is not None else None
    return sealed({
        "version": PLAN_VERSION, "source_sha256": profile["source_sha256"],
        "scan_profile_sha256": profile["sha256"], "acquisition_policy": category,
        "localization_policy": "bounded_oblique_anatomy" if quality["oblique_source"] else "native_axis_aligned",
        "localization_backend": backend, "direct_model": model_identity or "unavailable_without_promoted_weights",
        "assistance": lineage,
        "proxy_max_voxels": PROXY_MAX_VOXELS, "crop_margin_mm": [12.0, 24.0],
        "crop_policy": "complete_localized_orbits_nasal_nerves_and_major_sinuses_with_margin",
        "hu_interpolation": "linear_anatomy_proxy_only", "mask_interpolation": "nearest_anatomy_only",
        "counting_grid": "original_native", "cell_separation_parameters": "frozen_v2_unchanged",
        "bone_support_max_mm": 1.01, "processing_workers": 1,
        "uncertainty_policy": "preserve_all_acquisition_and_geometric_warnings",
        "resolution_recovery_claim": False,
    })


def validate_processing_plan(supplied: dict | None, expected: dict) -> dict:
    if supplied is not None and supplied != expected:
        raise ValueError("Processing plan differs from the exact source-bound policy snapshot")
    return expected
