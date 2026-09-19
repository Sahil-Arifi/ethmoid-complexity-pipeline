"""Scene-level synthetic wall metrics with fixed engineering criteria.

No patient/reference data, model fitting, threshold tuning or clinical claims are
performed here. The first-hit comparison is a boundary-only proxy supplied with
the same procedural seed/search extent; it does not run the complete v5 anatomy
localizer. Its observed input rays and interpolated fitted surface are measured
separately. Only the former can receive observed-target precision.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy import ndimage as ndi

from .engine import CountingError
from .localization import _continuous_wall
from .synthetic_walls import BONE_HU, SyntheticWallScene, candidate_targets
from .wall_proposals import WallCandidates, WallProposalError, decode_surface


BENCHMARK_VERSION = "procedural-observed-wall-benchmark-2"
METHODS = ("image_only", "learned", "first_hit_quadratic")
CRITERIA = {
    "maximum_false_acceptances": 0,
    "minimum_accepted_observed_target_precision": 0.98,
    "maximum_accepted_mean_error_mm": "max(1.0, native_spacing_y_mm)",
    "minimum_accepted_observable_scenes": 1,
    "scope": "predetermined_engineering_controls_not_clinical_acceptance",
}


def _first_hit_quadratic(scene: SyntheticWallScene, limit_y: int):
    """Original v5-like first-hit/orientation checks plus the exact curve fitter."""
    hu, seed = scene.hu, scene.seed
    spacing = np.asarray(scene.spacing_mm)
    bone = hu >= BONE_HU
    wall = np.zeros((hu.shape[0], hu.shape[2]), dtype=np.int32)
    valid = np.zeros(wall.shape, bool)
    normal = ndi.gaussian_filter(hu, sigma=0.5 / spacing)
    # This loop intentionally preserves the production boundary-ray checks.
    for xx, zz in zip(*np.where(seed.any(axis=1))):
        start = int(np.flatnonzero(seed[xx, :, zz]).max())
        hits = np.flatnonzero(bone[xx, start:limit_y + 1, zz])
        if not hits.size:
            continue
        hit = start + int(hits[0])
        if xx < 1 or zz < 1 or xx >= hu.shape[0] - 1 or zz >= hu.shape[2] - 1 or hit < 2:
            continue
        gy = (normal[xx, hit, zz] - normal[xx, hit - 2, zz]) / (2 * spacing[1])
        gx = (normal[xx + 1, hit - 1, zz] - normal[xx - 1, hit - 1, zz]) / (2 * spacing[0])
        gz = (normal[xx, hit - 1, zz + 1] - normal[xx, hit - 1, zz - 1]) / (2 * spacing[2])
        if gy <= 0 or gy < max(abs(gx), abs(gz)):
            continue
        wall[xx, zz], valid[xx, zz] = hit, True
    if valid.sum() < max(12, seed.any(axis=1).sum() * 0.5):
        raise CountingError("SPHENOID_UNRESOLVED", "The boundary-only proxy lacks sufficient first-hit wall support.")
    fitted = _continuous_wall(wall, valid, spacing)
    size = (max(1, int(np.ceil(2 / spacing[0]))), max(1, int(np.ceil(2 / spacing[2]))))
    fitted = ndi.median_filter(fitted, size=size)
    xx, zz = np.where(valid)
    observed = np.column_stack((xx, wall[valid], zz)).astype(np.int32)
    return observed, fitted


def _empty(code: str) -> dict:
    return {"accepted": False, "abstention_code": code, "false_acceptance": False,
            "selected_point_count": 0, "target_labeled_point_count": 0,
            "observed_target_precision": None, "mean_physical_error_mm": None,
            "maximum_physical_error_mm": None, "fitted_surface_mean_error_mm": None,
            "fitted_surface_maximum_error_mm": None, "fitted_points_are_observed": False,
            "engineering_acceptance_passed": False}


def _measure(scene: SyntheticWallScene, points: np.ndarray, *, fitted=None) -> dict:
    targets = candidate_targets(scene, points)
    measured = targets >= 0
    precision = float(np.mean(targets[measured] == 1)) if np.any(measured) else None
    physical_y = scene.origin_mm[1] + points[:, 1] * scene.spacing_mm[1]
    target_y = scene.target_leading_surface_y_mm[points[:, 0], points[:, 2]]
    errors = np.abs(physical_y - target_y)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("An accepted surface must contain finite observed source points")
    false_acceptance = bool(not scene.observable or scene.ambiguous or scene.cropped
                            or scene.wall_state in {"missing", "averaged_away"})
    result = {"accepted": True, "abstention_code": None, "false_acceptance": false_acceptance,
              "selected_point_count": len(points), "target_labeled_point_count": int(measured.sum()),
              "observed_target_precision": precision, "mean_physical_error_mm": float(errors.mean()),
              "maximum_physical_error_mm": float(errors.max()),
              "fitted_surface_mean_error_mm": None, "fitted_surface_maximum_error_mm": None,
              "fitted_points_are_observed": False,
              "engineering_acceptance_passed": bool(not false_acceptance and precision is not None
                  and precision >= 0.98 and float(errors.mean()) <= max(1.0, scene.spacing_mm[1]))}
    if fitted is not None:
        columns = np.any(scene.seed, axis=1)
        predicted = scene.origin_mm[1] + fitted[columns] * scene.spacing_mm[1]
        fitted_errors = np.abs(predicted - scene.target_leading_surface_y_mm[columns])
        result.update(fitted_surface_mean_error_mm=float(fitted_errors.mean()),
                      fitted_surface_maximum_error_mm=float(fitted_errors.max()),
                      observed_points_scope="gradient_accepted_inputs_to_fit_not_interpolated_output",
                      interpolated_surface_scope="all_seeded_columns_descriptive_only")
    else:
        result["observed_points_scope"] = "selected_observed_candidate_faces"
    return result


def _validate_scene_candidates(scene: SyntheticWallScene, candidates: WallCandidates, scores) -> None:
    if not isinstance(scene, SyntheticWallScene) or not isinstance(candidates, WallCandidates):
        raise ValueError("Synthetic scene and observed candidate objects are required")
    points = candidates.positions_ijk
    candidate_targets(scene, points)  # Shape/integer/bounds checks, not selection.
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(points),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Benchmark scores must be finite candidate probabilities")
    if (scene.hu.shape != candidates.source_shape or scene.spacing_mm != candidates.spacing_mm
            or candidates.bone_hu != BONE_HU or not np.array_equal(scene.seed.any(axis=1), candidates.seed_column_mask)
            or not np.array_equal(np.argwhere(scene.seed.any(axis=1)), candidates.columns_xz)
            or not np.array_equal(scene.hu[tuple(points.T)], candidates.candidate_hu)
            or not 0 <= candidates.limit_y < scene.hu.shape[1]):
        raise ValueError("Observed candidates do not match this phantom source and sampling")
    fronts = np.array([np.flatnonzero(scene.seed[x, :, z])[-1] for x, z in candidates.columns_xz])
    if not np.array_equal(fronts, candidates.seed_front_y):
        raise ValueError("Observed candidates do not match the phantom's posterior seed")


def _scene_row(scene: SyntheticWallScene, methods: dict, *, extraction_failed=False) -> dict:
    return {"schema_version": 1, "benchmark_version": BENCHMARK_VERSION,
            "geometry_id": scene.geometry_id, "augmentation_id": scene.augmentation_id,
            "family": scene.family, "wall_state": scene.wall_state,
            "observable": scene.observable, "ambiguous": scene.ambiguous, "cropped": scene.cropped,
            "support_fraction": scene.support_fraction, "spacing_mm": list(scene.spacing_mm),
            "error_definition": "absolute_Y_error_at_same_XZ_to_analytic_seed_facing_slab_boundary_mm",
            "coordinate_origin_included": True, "accuracy_scope": scene.provenance,
            "baseline_scope": "boundary_only_first_hit_gradient_and_exact_v5_curve_fit_not_full_anatomical_inference",
            "extraction_failed": extraction_failed,
            "engineering_criteria": dict(CRITERIA), "methods": methods}


def extraction_failure_row(scene: SyntheticWallScene, code: str) -> dict:
    """Retain an unavailable input without silently omitting a benchmark scene."""
    if not isinstance(scene, SyntheticWallScene) or not isinstance(code, str) or not code.startswith("WALL_") or len(code) > 100:
        raise ValueError("An explicit wall extraction outcome is required")
    methods = {name: {**_empty(code), "execution_scope": "not_run_extraction_failed"} for name in METHODS}
    return _scene_row(scene, methods, extraction_failed=True)


def evaluate_scene(scene: SyntheticWallScene, candidates: WallCandidates, scores, *, evidence: dict | None = None) -> dict:
    """Compare fixed methods without passing ground truth into either decoder."""
    _validate_scene_candidates(scene, candidates, scores)
    if evidence is not None:
        if not isinstance(evidence, dict) or evidence:
            raise ValueError("An empty evidence dictionary is required")
        evidence.update(candidate_positions_ijk=candidates.positions_ijk.copy(),
            candidate_hu=candidates.candidate_hu.copy(), candidate_scores=np.asarray(scores).copy(),
            candidate_targets=candidate_targets(scene, candidates.positions_ijk),
            source_spacing_mm=np.asarray(scene.spacing_mm), source_origin_mm=np.asarray(scene.origin_mm))
    methods = {}
    for name, probabilities in (("image_only", np.ones(len(candidates.positions_ijk))), ("learned", scores)):
        try:
            decoded = decode_surface(candidates, probabilities)
            if not np.array_equal(decoded.positions_ijk, candidates.positions_ijk[decoded.selected_candidate_indices]):
                raise ValueError("Decoder points are not its selected observed candidates")
            methods[name] = _measure(scene, decoded.positions_ijk)
            if evidence is not None:
                evidence[name + "_selected_candidate_indices"] = decoded.selected_candidate_indices.copy()
                evidence[name + "_observed_points_ijk"] = decoded.positions_ijk.copy()
        except WallProposalError as exc:
            methods[name] = _empty(exc.code)
            if evidence is not None:
                evidence[name + "_selected_candidate_indices"] = np.empty(0, dtype=np.int32)
                evidence[name + "_observed_points_ijk"] = np.empty((0, 3), dtype=np.int32)
    try:
        observed, fitted = _first_hit_quadratic(scene, candidates.limit_y)
        methods["first_hit_quadratic"] = _measure(scene, observed, fitted=fitted)
        if evidence is not None:
            evidence["first_hit_quadratic_observed_points_ijk"] = observed.copy()
            evidence["first_hit_quadratic_fitted_surface_ijk"] = fitted.copy()
    except CountingError as exc:
        methods["first_hit_quadratic"] = _empty(exc.code)
        if evidence is not None:
            evidence["first_hit_quadratic_observed_points_ijk"] = np.empty((0, 3), dtype=np.int32)
            evidence["first_hit_quadratic_fitted_surface_ijk"] = np.empty((0, 0), dtype=np.int32)
    return _scene_row(scene, methods)


def summarize(rows) -> dict:
    """Aggregate fixed metrics; all-abstention results cannot pass vacuously."""
    values = list(rows)
    ids = [(r["geometry_id"], r["augmentation_id"]) for r in values]
    if len(ids) != len(set(ids)):
        raise ValueError("A scene augmentation cannot be counted twice in a benchmark")
    if any(r.get("benchmark_version") != BENCHMARK_VERSION or r.get("engineering_criteria") != CRITERIA
           or set(r.get("methods", {})) != set(METHODS) for r in values):
        raise ValueError("Benchmark rows must use the same predetermined method and criteria")
    result = {}
    for method in METHODS:
        accepted = [(r, r["methods"][method]) for r in values if r["methods"][method]["accepted"]]
        abstained = [r["methods"][method] for r in values if not r["methods"][method]["accepted"]]
        false = sum(bool(not r["observable"] or r["ambiguous"] or r["cropped"]
                         or r["wall_state"] in {"missing", "averaged_away"}) for r, _ in accepted)
        observed_accepts = sum(bool(r["observable"] and not r["ambiguous"] and not r["cropped"]) for r, _ in accepted)
        errors = [m["mean_physical_error_mm"] for _, m in accepted]
        precision = [m["observed_target_precision"] for _, m in accepted]
        if any(e is None or not math.isfinite(e) or e < 0 for e in errors):
            raise ValueError("Accepted point errors must be finite nonnegative measurements")
        precision_pass = all(p is not None and math.isfinite(p) and 0.98 <= p <= 1 for p in precision)
        error_pass = all(m["mean_physical_error_mm"] <= max(1.0, r["spacing_mm"][1]) for r, m in accepted)
        enough = observed_accepts >= 1
        result[method] = {
            "accepted": len(accepted), "abstained": len(abstained),
            "abstention_reasons": dict(Counter(m["abstention_code"] for m in abstained)),
            "false_acceptances": false, "accepted_observable_scenes": observed_accepts,
            "mean_scene_physical_error_mm": float(np.mean(errors)) if errors else None,
            "worst_scene_mean_physical_error_mm": max(errors) if errors else None,
            "minimum_observed_target_precision": min(p for p in precision if p is not None) if any(p is not None for p in precision) else None,
            "engineering_gate_passed": bool(enough and false == 0 and precision_pass and error_pass),
            "engineering_gate_status": "insufficient_accepted_evidence" if not enough else "passed" if false == 0 and precision_pass and error_pass else "failed",
            "criterion_checks": {"accepted_evidence_present": enough, "zero_false_acceptances": false == 0,
                                 "all_accepted_precision_at_least_0_98": precision_pass,
                                 "all_accepted_mean_errors_within_fixed_spacing_limit": error_pass},
        }
    return {"schema_version": 1, "benchmark_version": BENCHMARK_VERSION,
            "scenes": len(values), "geometries": len({r["geometry_id"] for r in values}),
            "families": sorted({r["family"] for r in values}),
            "observable_scenes": sum(bool(r["observable"]) for r in values),
            "unobservable_scenes": sum(not r["observable"] for r in values),
            "extraction_failed_scenes": sum(bool(r.get("extraction_failed")) for r in values),
            "engineering_criteria": dict(CRITERIA), "clinical_accuracy_measured": False,
            "methods": result}
