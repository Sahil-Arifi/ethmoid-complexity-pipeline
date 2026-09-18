"""Opt-in research method. No trained weights or new reference labels are implied.

V1/V2/V3 default behavior is preserved. This experiment changes anatomical
localization only; native cavity acceptance, air classification and IDs retain
the existing contract. Every changed policy is bound to a new protocol/plan.
"""
from __future__ import annotations

from itertools import product
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from .adaptive import AdaptiveStrategy, _crop_bounds
from .engine import CountingError, analyze_volume, geometry
from .grids import check_proxy_limit, localization_grid
from .localization import Localization
from .profiles import build_scan_profile
from .research_policy import (ANATOMY_PROXY_POLICY, LOCALIZATION_POLICY,
    METHOD_VERSION, PROTOCOL_VERSION, build_research_plan)


def research_anatomy_grid(shape, affine) -> dict:
    """Bound a complete RAS anatomy view; never use it to assign final cell IDs."""
    affine = np.asarray(affine, dtype=float)
    corners = nib.affines.apply_affine(affine, list(product(*[(-0.5, n - 0.5) for n in shape])))
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    extent = upper - lower
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise CountingError("RESOURCE_LIMIT", "The anatomy extent cannot fit the research proxy policy.")
    minimum = max(float(nib.affines.voxel_sizes(affine).min()), ANATOMY_PROXY_POLICY["minimum_spacing_mm"])
    increment = ANATOMY_PROXY_POLICY["spacing_increment_mm"]
    step = math.ceil(minimum / increment) * increment
    while step <= ANATOMY_PROXY_POLICY["maximum_spacing_mm"]:
        target_shape = [max(1, math.ceil(float(width) / step)) for width in extent]
        if math.prod(target_shape) <= ANATOMY_PROXY_POLICY["maximum_voxels"]:
            target = np.diag([step, step, step, 1.0])
            target[:3, 3] = lower + step / 2
            return {"mode": "anatomy_only_ras_proxy", "shape": target_shape,
                "affine": target.tolist(), "spacing_mm": [step] * 3,
                "hu_interpolation": "linear", "mask_interpolation": "nearest",
                "counting_grid": "original_native", "proxy_max_voxels": ANATOMY_PROXY_POLICY["maximum_voxels"],
                "research_resolution_policy": dict(ANATOMY_PROXY_POLICY),
                "native_min_spacing_mm": float(nib.affines.voxel_sizes(affine).min()),
                "resolution_recovery_claim": False}
        step += increment
    raise CountingError("RESOURCE_LIMIT", "The complete anatomy view exceeds the research budget at the maximum permitted spacing.")


def research_localize(hu, affine, masks, cfg, plan, diagnostics) -> Localization:
    from .localization import localize_native

    def fits(grid):
        check_proxy_limit(grid)
        if math.prod(grid["shape"]) > plan["proxy_max_voxels"]:
            raise CountingError("RESOURCE_LIMIT", "The native-spacing anatomy grid exceeds its fixed voxel budget.")

    def infer(values, transform, landmarks, grid, record):
        return localize_native(values, transform, landmarks, cfg, grid,
            localization_policy=LOCALIZATION_POLICY, diagnostics=record)

    full = localization_grid(hu.shape, affine)
    diagnostics.update(mode="full_anatomical_extent", geometry=full, localization={})
    try:
        fits(full)
    except CountingError as error:
        if error.code != "RESOURCE_LIMIT":
            raise
    else:
        return infer(hu, affine, masks, full, diagnostics["localization"])

    diagnostics.update(mode="bounded_anatomical_crop", attempts=[])
    last = None
    for margin in plan["crop_margin_mm"]:
        crop = _crop_bounds(hu.shape, affine, masks, margin)
        transform = np.asarray(affine).copy()
        transform[:3, 3] = nib.affines.apply_affine(affine, [s.start for s in crop])
        shape = tuple(s.stop - s.start for s in crop)
        grid = localization_grid(shape, transform)
        record = {"margin_mm": margin, "native_bounds": [[s.start, s.stop] for s in crop],
            "geometry": grid, "localization": {}}
        diagnostics["attempts"].append(record)
        try:
            fits(grid)
            local = infer(hu[crop], transform, {name: mask[crop] for name, mask in masks.items()}, grid, record["localization"])
            points = np.where(local.region)
            if not len(points[0]):
                raise CountingError("EMPTY_REGION", "The localized research region is empty.")
            pad = np.ceil((cfg.boundary_growth_mm + 1) / geometry(transform)).astype(int)
            if any((crop[i].start > 0 and points[i].min() <= pad[i]) or
                   (crop[i].stop < hu.shape[i] and points[i].max() >= shape[i] - pad[i] - 1) for i in range(3)):
                raise CountingError("TARGET_CLIPPED", "The research target needs more context than its anatomy crop.")
        except CountingError as error:
            record["outcome"] = error.code
            if error.code not in {"TARGET_CLIPPED", "RESOURCE_LIMIT"}:
                raise
            last = error
            continue
        region, excluded = np.zeros(hu.shape, bool), np.ones(hu.shape, bool)
        region[crop], excluded[crop] = local.region, local.excluded
        variants = {}
        for name, mask in local.variants.items():
            variants[name] = np.zeros(hu.shape, bool)
            variants[name][crop] = mask
        record["outcome"] = "localized"
        return Localization(region, excluded, variants, local.midline_ras_x, local.warnings + [
            "Research localization used a bounded native-spacing anatomy crop; final cell decisions remain native."])

    # An unresolved anatomical boundary must not trigger a spacing search. This
    # fallback is exclusively for a verified allocation constraint, preserving
    # the complete acquired physical extent and explicit padding exclusion.
    if last is not None and last.code != "RESOURCE_LIMIT":
        raise last
    proxy = research_anatomy_grid(hu.shape, affine)
    diagnostics.update(mode="research_bounded_resolution_anatomy", geometry=proxy, localization={})
    local = infer(hu, affine, masks, proxy, diagnostics["localization"])
    local.warnings.append(
        "Experimental anatomy-only view uses a coarser recorded spacing to bound memory. "
        "Its anatomical accuracy is unmeasured; final cavity evidence, air, IDs and measurements use the original CT voxels.")
    return local


class ResearchStrategy(AdaptiveStrategy):
    def finalize(self, result):
        super().finalize(result)
        if result.get("failure_code") in {"ORBIT_FLOOR_UNRESOLVED", "ORBIT_WALL_UNRESOLVED", "ROOF_UNRESOLVED"}:
            result["outcome"] = "needs_review"
            self.stage("needs_review")

    def initial_metadata(self):
        result = super().initial_metadata()
        result.update(method_version=METHOD_VERSION, protocol_version=PROTOCOL_VERSION,
            model_identity={"localization": "research_geometric_baseline", "direct_ethmoid_model": "not_used"},
            research_status="NOT_ACCURACY_VALIDATED")
        return result

    def localize(self, source, image, canonical, hu, destination, landmarks_dir, cfg, result):
        from .localization import load_landmarks, run_localizers, cache_producer_summary, cache_producer_warnings
        profile = build_scan_profile(source, self.acquisition)
        result["scan_profile"] = profile
        plan = build_research_plan(profile)
        if self.processing_plan is not None and self.processing_plan != plan:
            raise CountingError("PROCESSING_PLAN_MISMATCH", "The research plan does not match its source-bound policy.")
        result["processing_plan"] = plan
        self.stage("localizing")
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
                local = research_localize(hu, canonical.affine, masks, cfg, plan, result["localization_geometry"])
            except CountingError as error:
                if not full_extent and error.code in {"TARGET_CLIPPED", "LANDMARK_EMPTY"}:
                    continue
                raise
            if (directory / "localizers.local.json").is_file():
                local.warnings.extend(cache_producer_warnings(directory))
            local.warnings.append("Research localization policy; anatomical accuracy and complete cell detection are unmeasured.")
            return local
        raise CountingError("LOCALIZER_FAILED", "The research anatomical localization did not complete.")


def analyze_research_volume(volume_path, output_dir, landmarks_dir=None, config=None, *, acquisition=None,
                            processing_plan=None, progress_path=None):
    strategy = ResearchStrategy(acquisition=acquisition, processing_plan=processing_plan,
        progress_path=Path(progress_path) if progress_path else None)
    return analyze_volume(volume_path, output_dir, landmarks_dir, config, acquisition=acquisition, _strategy=strategy)
