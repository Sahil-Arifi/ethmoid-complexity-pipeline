"""Versioned scan-adaptive and assisted localization with native cell counting.

The v2 cavity algorithm and thresholds are intentionally unchanged. This is an
engineering improvement to orchestration and memory use, not a trained cell model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .engine import CountingError, analyze_volume, geometry
from .grids import check_proxy_limit, localization_grid
from .localization import Localization
from .profiles import build_scan_profile, build_processing_plan, file_sha256, sealed, validate_processing_plan

METHOD_VERSION = "scan-adaptive-native-0.3.0"
PROTOCOL_VERSION = "ethmoid-count-prototype-v3"


def _read_bound_mask(path: str, digest: str, image, *, values: tuple[int, ...]) -> np.ndarray:
    target = Path(path)
    if not target.is_file() or file_sha256(target) != digest:
        raise CountingError("ASSISTANCE_INTEGRITY", "A boundary artifact differs from its recorded source-bound revision.")
    mask = nib.load(target)
    if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-5, rtol=0):
        raise CountingError("ASSISTANCE_GEOMETRY", "Boundary corrections must use the exact original source grid.")
    data = np.asanyarray(mask.dataobj)
    if mask.get_data_dtype() != np.dtype("uint8") or not np.isfinite(data).all() or not np.isin(data, values).all():
        raise CountingError("ASSISTANCE_VALUES", "Boundary masks require uint8 labels on the original source grid.")
    return data


def assisted_localization(assistance: dict, source_sha256: str, image, canonical) -> tuple[Localization, dict, np.ndarray]:
    if not isinstance(assistance, dict) or assistance.get("schema_version") != 1 or assistance.get("source_sha256") != source_sha256:
        raise CountingError("ASSISTANCE_SOURCE", "Boundary correction provenance does not match this exact source.")
    for key in ("revision_id", "parent_run_id"):
        if not isinstance(assistance.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", assistance[key]):
            raise CountingError("ASSISTANCE_LINEAGE", "A boundary correction must identify its immutable revision and parent run.")
    try:
        data = _read_bound_mask(assistance["region_path"], assistance["region_sha256"], image, values=(0, 1, 2))
        excluded = np.zeros(image.shape, dtype=bool)
        if assistance.get("exclusion_path"):
            excluded = _read_bound_mask(assistance["exclusion_path"], assistance["exclusion_sha256"], image, values=(0, 1)).astype(bool)
    except (KeyError, TypeError):
        raise CountingError("ASSISTANCE_INTEGRITY", "The boundary correction is missing its local mask or digest.") from None
    if not np.any(data) or np.any((data > 0) & excluded):
        raise CountingError("ASSISTANCE_VALUES", "Boundary membership must be nonempty and separate from excluded anatomy.")
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine), nib.orientations.io_orientation(canonical.affine))
    labels = nib.orientations.apply_orientation(data, orientation)
    excluded = nib.orientations.apply_orientation(excluded, orientation)
    region = labels > 0
    positions = np.where(region)
    if any(int(axis.min()) == 0 or int(axis.max()) == region.shape[i] - 1 for i, axis in enumerate(positions)):
        raise CountingError("TARGET_CLIPPED", "The corrected region reaches the acquired field-of-view boundary.")
    midline = assistance.get("midline_ras_x")
    if midline is None:
        centers = [nib.affines.apply_affine(canonical.affine, ndi.center_of_mass(labels == label))[0] for label in (1, 2) if np.any(labels == label)]
        midline = float(np.mean(centers)) if len(centers) == 2 else float(nib.affines.apply_affine(canonical.affine, (np.asarray(canonical.shape) - 1) / 2)[0])
    if isinstance(midline, bool) or not isinstance(midline, (float, int)) or not np.isfinite(midline):
        raise CountingError("ASSISTANCE_LATERALITY", "The corrected anatomical midline must be finite.")
    lineage = {key: assistance[key] for key in ("schema_version", "source_sha256", "revision_id", "parent_run_id", "region_sha256")}
    lineage.update(midline_ras_x=float(midline), exclusion_sha256=assistance.get("exclusion_sha256"))
    return Localization(region, excluded, {}, float(midline), [
        "Clinician-assisted anatomical boundaries; candidate separation remains automatic and unvalidated.",
        "The reviewer-defined membership must include protocol variants and exclude neighboring parent sinuses; membership is not certified by software checks.",
    ]), sealed(lineage), labels


def _crop_bounds(shape: tuple, affine: np.ndarray, masks: dict, margin: float) -> tuple[slice, ...]:
    names = ("eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right", "nasal_cavity_left", "nasal_cavity_right", "sinus_maxillary", "sinus_frontal")
    lower, upper = np.asarray(shape), np.zeros(3, dtype=int)
    seen = False
    for name in names:
        if name not in masks or not masks[name].any():
            if name != "sinus_frontal":
                raise CountingError("LANDMARK_EMPTY", "A required landmark is missing for bounded anatomical processing.")
            continue
        where = np.where(masks[name])
        lower = np.minimum(lower, [int(axis.min()) for axis in where])
        upper = np.maximum(upper, [int(axis.max()) + 1 for axis in where])
        seen = True
    if not seen:
        raise CountingError("LANDMARK_EMPTY", "Bounded processing requires source-bound anatomical landmarks.")
    pad = np.ceil(margin / geometry(affine)).astype(int)
    return tuple(slice(max(0, int(lower[i] - pad[i])), min(shape[i], int(upper[i] + pad[i]))) for i in range(3))


def bounded_localize(hu, affine, masks, cfg, plan, diagnostics) -> Localization:
    """Crop anatomy if a full native or oblique anatomy grid exceeds its budget.

    All localized structures used to determine target coverage are retained.
    A target that touches an artificial crop face is retried with more context;
    it is never accepted merely because cropping made the allocation fit.
    """
    from .localization import localize_native
    def check(grid):
        check_proxy_limit(grid)
        if math.prod(grid["shape"]) > plan.get("proxy_max_voxels", 128_000_000):
            raise CountingError("RESOURCE_LIMIT", "The anatomical processing grid exceeds its fixed voxel budget.")
    full = localization_grid(hu.shape, affine)
    try:
        check(full)
    except CountingError as error:
        if error.code != "RESOURCE_LIMIT":
            raise
    else:
        diagnostics.update(mode="full_anatomical_extent", geometry=full)
        return localize_native(hu, affine, masks, cfg, full)
    diagnostics.update(mode="bounded_anatomical_crop", attempts=[])
    last = None
    for margin in plan["crop_margin_mm"]:
        crop = _crop_bounds(hu.shape, affine, masks, margin)
        cropped_affine = np.asarray(affine).copy()
        cropped_affine[:3, 3] = nib.affines.apply_affine(affine, [sl.start for sl in crop])
        shape = tuple(sl.stop - sl.start for sl in crop)
        grid = localization_grid(shape, cropped_affine)
        record = {"margin_mm": margin, "native_bounds": [[sl.start, sl.stop] for sl in crop], "geometry": grid}
        diagnostics["attempts"].append(record)
        try:
            check(grid)
            local = localize_native(hu[crop], cropped_affine, {name: mask[crop] for name, mask in masks.items()}, cfg, grid)
            # Preserve a margin greater than the reconstruction-growth distance.
            points = np.where(local.region)
            pad = np.ceil((cfg.boundary_growth_mm + 1) / geometry(cropped_affine)).astype(int)
            if any((crop[i].start > 0 and points[i].min() <= pad[i]) or
                   (crop[i].stop < hu.shape[i] and points[i].max() >= shape[i] - pad[i] - 1) for i in range(3)):
                raise CountingError("TARGET_CLIPPED", "The target needs more context than the bounded anatomical crop.")
        except CountingError as error:
            record["outcome"] = error.code
            if error.code not in {"TARGET_CLIPPED", "RESOURCE_LIMIT"}:
                raise
            last = error
            continue
        region = np.zeros(hu.shape, bool)
        excluded = np.ones(hu.shape, bool)
        region[crop], excluded[crop] = local.region, local.excluded
        variants = {}
        for name, mask in local.variants.items():
            variants[name] = np.zeros(hu.shape, bool)
            variants[name][crop] = mask
        record["outcome"] = "localized"
        return Localization(region, excluded, variants, local.midline_ras_x, local.warnings + [
            "An oversized anatomy grid was replaced with a source-landmark crop at unchanged spacing. Cell IDs and HU decisions remain native.",
        ])
    raise last or CountingError("RESOURCE_LIMIT", "The bounded anatomical proxy cannot fit the local budget.")


@dataclass
class AdaptiveStrategy:
    acquisition: dict | None = None
    assistance: dict | None = None
    processing_plan: dict | None = None
    progress_path: Path | None = None
    model_manifest: Path | None = None
    side_labels: Any = None

    def stage(self, stage: str) -> None:
        if self.progress_path is not None:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"schema_version": 1, "stage": stage}), encoding="utf-8")
            temporary.replace(self.progress_path)

    def initial_metadata(self) -> dict:
        self.stage("profiling")
        return {"method_version": METHOD_VERSION, "protocol_version": PROTOCOL_VERSION,
            "analysis_mode": "assisted" if self.assistance is not None else "automatic",
            "model_identity": {"localization": "reviewer_native_region" if self.assistance is not None else "v2_geometric_baseline", "direct_ethmoid_model": "unavailable_without_promoted_weights"},
            "outcome": "technical_error", "scan_profile": {}, "processing_plan": {}, "assistance": None}

    def localize(self, source, image, canonical, hu, destination, landmarks_dir, cfg, result):
        from .localization import load_landmarks, run_localizers, cache_producer_summary, cache_producer_warnings
        profile = build_scan_profile(source, self.acquisition)
        result["scan_profile"] = profile
        assisted = None
        if self.assistance is not None:
            assisted, lineage, self.side_labels = assisted_localization(self.assistance, result["source_sha256"], image, canonical)
            result["assistance"] = lineage
        model_identity = None
        if self.model_manifest is not None:
            from .models import load_model_manifest
            manifest, _ = load_model_manifest(self.model_manifest)
            model_identity = {"model_id": manifest["model_id"], "model_manifest_sha256": file_sha256(self.model_manifest)}
        try:
            plan = validate_processing_plan(self.processing_plan, build_processing_plan(profile, assistance=self.assistance, model_identity=model_identity))
        except ValueError:
            raise CountingError("PROCESSING_PLAN_MISMATCH", "The requested processing plan does not match this source and versioned policy.") from None
        result["processing_plan"] = plan
        self.stage("localizing")
        if self.assistance is not None:
            result["localization_geometry"] = {"mode": "source_bound_assisted_region", "counting_grid": "original_native"}
            return assisted
        if self.model_manifest is not None:
            from .models import run_direct_model
            local, identity, self.side_labels = run_direct_model(self.model_manifest, source, canonical, destination / "direct-anatomy")
            result["model_identity"] = identity
            result["localization_geometry"] = {"mode": "source_bound_direct_model", "counting_grid": "original_native"}
            return local
        directory = Path(landmarks_dir) if landmarks_dir else destination / "localizers"
        if landmarks_dir is None:
            run_localizers(source, directory)
        for full_extent in (False, True):
            if full_extent:
                directory = destination / "localizers-full-extent"
                run_localizers(source, directory, full_extent=True)
            try:
                masks = load_landmarks(directory, canonical, source_sha256=result["source_sha256"])
                if (directory / "localizers.local.json").is_file():
                    result["localizer_provenance"] = cache_producer_summary(directory)
                local = bounded_localize(hu, canonical.affine, masks, cfg, plan, result["localization_geometry"])
            except CountingError as error:
                if not full_extent and error.code in {"TARGET_CLIPPED", "LANDMARK_EMPTY"}:
                    continue
                raise
            if (directory / "localizers.local.json").is_file():
                local.warnings.extend(cache_producer_warnings(directory))
            if full_extent:
                local.warnings.append("Source-bound anatomical localization required a full-field localizer retry.")
            return local
        raise CountingError("LOCALIZER_FAILED", "The anatomical localization did not complete.")

    def assign_sides(self, cavities, cells):
        if self.side_labels is None:
            return
        for cell in cells:
            labels = set(np.unique(self.side_labels[cavities == cell["id"]])) - {0}
            if len(labels) != 1:
                code = "ASSISTANCE_LATERALITY" if self.assistance is not None else "DIRECT_MODEL_LATERALITY"
                raise CountingError(code, "A candidate crosses conflicting anatomical side labels; the membership requires review.")
            cell["side"] = "left" if labels == {1} else "right"
            cell["reasons"].append("reviewer_defined_anatomical_membership" if self.assistance is not None else "direct_model_anatomical_membership")
            cell["uncertain"] = True

    def verify_artifacts(self, source, destination, result):
        self.stage("verifying")
        if file_sha256(source) != result["source_sha256"]:
            raise CountingError("SOURCE_CHANGED", "The original source changed during analysis.")
        image = nib.load(source)
        loaded = {}
        for name, filename in result["artifacts"].items():
            mask = nib.load(destination / filename)
            dtype = np.dtype("uint8" if name == "region" else "uint16")
            if mask.shape != image.shape or mask.get_data_dtype() != dtype or not np.allclose(mask.affine, image.affine, atol=1e-5, rtol=0):
                raise CountingError("ARTIFACT_VERIFICATION", "A generated mask does not preserve the original source grid and label type.")
            loaded[name] = np.asanyarray(mask.dataobj)
        cells, cavities = result["cells"], loaded["cavities"]
        if set(np.unique(cavities)) - {0} != {cell["id"] for cell in cells}:
            raise CountingError("ARTIFACT_VERIFICATION", "Instance labels and the cell registry disagree.")
        expected_air = np.where(image.get_fdata(dtype=np.float32) < result["config"]["air_hu"], cavities, 0)
        if not np.array_equal(expected_air, loaded["air"]):
            raise CountingError("ARTIFACT_VERIFICATION", "Air labels do not match calibrated native HU samples.")
        for cell in cells:
            positions = np.where(cavities == cell["id"])
            centroid = nib.affines.apply_affine(image.affine, [np.mean(axis) for axis in positions])
            if (len(positions[0]) != cell["voxel_count"] or
                np.count_nonzero(loaded["air"] == cell["id"]) != cell["air_voxel_count"] or
                not np.allclose(centroid, cell["centroid_ras_mm"], atol=1e-5, rtol=0)):
                raise CountingError("ARTIFACT_VERIFICATION", "Native instance measurements failed verification.")
        result["artifact_verification"] = {"status": "verified", "source_sha256": result["source_sha256"],
            "sha256": {name: file_sha256(destination / filename) for name, filename in result["artifacts"].items()}}

    def finalize(self, result):
        review = {"SPHENOID_UNRESOLVED", "SPHENOID_AMBIGUOUS", "TARGET_CLIPPED", "LANDMARK_EMPTY", "LANDMARK_SCALE", "LANDMARK_LATERALITY", "EMPTY_REGION", "NO_COMPARTMENTS", "NO_SUPPORTED_CELLS", "INSUFFICIENT_COVERAGE", "ASSISTANCE_LATERALITY", "DIRECT_MODEL_LATERALITY"}
        if result.get("failure_code"):
            result.update(status="failed", counts={"left": None, "right": None, "total": None}, cells=[], artifacts={})
            result["outcome"] = "needs_review" if result["failure_code"] in review else "technical_error"
        else:
            result["outcome"] = "assisted_candidate_count" if self.assistance is not None else "automatic_candidate_count"
        self.stage("completed" if result["status"] == "completed" else result["outcome"])


def analyze_adaptive_volume(volume_path, output_dir, landmarks_dir=None, config=None, *, acquisition=None,
                            assistance=None, processing_plan=None, progress_path=None, model_manifest=None):
    strategy = AdaptiveStrategy(acquisition, assistance, processing_plan,
        Path(progress_path) if progress_path else None, Path(model_manifest) if model_manifest else None)
    return analyze_volume(volume_path, output_dir, landmarks_dir, config, acquisition=acquisition, _strategy=strategy)
