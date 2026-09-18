"""CPU-only research-proxy contracts; synthetic checks are not clinical evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from importlib.resources import files
from itertools import product
import json
import math

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting import adaptive, grids, localization, research
from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, compartment_stage
from ethmoid_pipeline.counting.profiles import build_processing_plan, build_scan_profile, file_sha256, sealed
from ethmoid_pipeline.counting.protocol import load_counting_protocol
from ethmoid_pipeline.counting.research_policy import (
    ANATOMY_PROXY_POLICY, LOCALIZATION_POLICY, METHOD_VERSION, PROTOCOL_VERSION,
    build_research_plan,
)


def affine_for(spacing, angle=0.0, *, flipped=False):
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(spacing)
    if flipped:
        affine[:3, 0] *= -1
    affine[:3, 3] = [-37.25, 11.125, -8.5]
    return affine


def landmarks(shape):
    # Source-edge landmarks force the crop helper to preserve full coverage.
    result = {}
    for name in ("brain", "eye_left", "eye_right", "optic_nerve_left",
                 "optic_nerve_right", "nasal_cavity_left", "nasal_cavity_right",
                 "sinus_maxillary", "sinus_frontal"):
        mask = np.zeros(shape, bool)
        mask[0, 0, 0] = mask[-1, -1, -1] = True
        result[name] = mask
    return result


@pytest.mark.parametrize("spacing,angle,flipped", [
    ((0.2, 0.2, 2.5), 0, False),
    ((0.2, 0.65, 2.5), 31, False),
    ((0.45, 0.8, 1.25), -23, True),
    ((0.7, 0.7, 0.7), 0, False),
])
def test_grid_covers_every_source_half_voxel_corner(spacing, angle, flipped):
    shape = (51, 39, 25)
    affine = affine_for(spacing, angle, flipped=flipped)
    unchanged = affine.copy()
    plan = research.research_anatomy_grid(shape, affine)
    assert plan == research.research_anatomy_grid(shape, affine.copy())
    assert np.array_equal(affine, unchanged)
    step = plan["spacing_mm"][0]
    assert 0.5 <= step <= 1.0
    assert plan["spacing_mm"] == [step] * 3
    assert math.prod(plan["shape"]) <= 64_000_000
    assert plan["counting_grid"] == "original_native"
    assert plan["resolution_recovery_claim"] is False
    assert np.allclose(np.asarray(plan["affine"])[:3, :3], np.eye(3) * step)
    corners = nib.affines.apply_affine(affine, list(product(*[(-0.5, n - 0.5) for n in shape])))
    proxy_coordinates = nib.affines.apply_affine(np.linalg.inv(plan["affine"]), corners)
    assert np.all(proxy_coordinates >= -0.5 - 1e-10)
    assert np.all(proxy_coordinates <= np.asarray(plan["shape"]) - 0.5 + 1e-10)
    # Complete coverage adds less than one proxy voxel at the upper boundary.
    lower = np.asarray(plan["affine"])[:3, 3] - step / 2
    upper = lower + np.asarray(plan["shape"]) * step
    assert np.allclose(lower, corners.min(axis=0))
    assert np.all(upper - corners.max(axis=0) < step + 1e-10)


@pytest.mark.parametrize("shape,expected_spacing", [
    ((400, 400, 400), 0.5),  # Exactly the fixed 64 million voxel boundary.
    ((401, 401, 401), 0.625),
    ((600, 600, 600), 0.75),
    ((800, 800, 800), 1.0),
])
def test_budget_selects_first_permitted_spacing_without_allocating_volume(shape, expected_spacing):
    plan = research.research_anatomy_grid(shape, np.diag([0.5, 0.5, 0.5, 1]))
    assert plan["spacing_mm"] == [expected_spacing] * 3
    assert math.prod(plan["shape"]) <= 64_000_000
    assert plan["research_resolution_policy"] == ANATOMY_PROXY_POLICY


def test_resource_limit_is_reported_before_any_resampling(monkeypatch):
    hu = np.zeros((4, 4, 4), np.float32)
    masks = landmarks(hu.shape)
    affine = np.diag([1000.0, 1000.0, 1000.0, 1])
    monkeypatch.setattr(grids.ndi, "affine_transform", lambda *a, **kw: pytest.fail("Allocated an ineligible proxy"))
    monkeypatch.setattr(localization, "localize_native", lambda *a, **kw: pytest.fail("Inferred an ineligible proxy"))
    with pytest.raises(CountingError) as failure:
        research.research_localize(hu, affine, masks, CountingConfig(),
                                  {"proxy_max_voxels": 1, "crop_margin_mm": [12, 24]}, {})
    assert failure.value.code == "RESOURCE_LIMIT"


@pytest.mark.parametrize("angle", [0, 29])
def test_proxy_uses_linear_hu_nearest_masks_and_excludes_physical_padding(angle):
    shape = (18, 14, 10)
    affine = affine_for((0.4, 0.7, 0.9), angle)
    indices = np.indices(shape)
    hu = (13 * indices[0] + 7 * indices[1] - 3 * indices[2] - 700).astype(np.float32)
    mask = (indices[0] >= 5) & (indices[0] < 11) & (indices[1] > 2)
    original_hu, original_mask, original_affine = hu.copy(), mask.copy(), affine.copy()
    plan = research.research_anatomy_grid(shape, affine)
    values, proxy_affine, masks, valid = grids.anatomy_proxy(hu, affine, {"test": mask}, plan)
    coordinates = np.moveaxis(np.indices(values.shape), 0, -1)
    source_positions = nib.affines.apply_affine(np.linalg.inv(affine) @ proxy_affine, coordinates)
    expected_valid = np.all((source_positions >= -0.5) & (source_positions <= np.asarray(shape) - 0.5), axis=-1)
    assert np.array_equal(valid, expected_valid)
    clamped = np.clip(source_positions, 0, np.asarray(shape) - 1)
    expected_hu = clamped[..., 0] * 13 + clamped[..., 1] * 7 - clamped[..., 2] * 3 - 700
    assert np.allclose(values[valid], expected_hu[valid], atol=1e-4)
    nearest = np.floor(clamped + 0.5).astype(int)
    expected_mask = mask[tuple(np.moveaxis(nearest, -1, 0))] & expected_valid
    assert np.array_equal(masks["test"], expected_mask)
    assert masks["test"].dtype == bool
    assert np.all(values[~valid] == -1024)
    assert not masks["test"][~valid].any()
    if angle:
        assert (~valid).any(), "Oblique bounding boxes must contain padding outside acquired anatomy"
    assert np.array_equal(hu, original_hu)
    assert np.array_equal(mask, original_mask)
    assert np.array_equal(affine, original_affine)


def test_padding_cannot_become_accepted_anatomical_membership(monkeypatch):
    hu = np.zeros((18, 14, 10), np.float32)
    affine = affine_for((0.4, 0.7, 0.9), 29)
    masks = landmarks(hu.shape)
    plan = research.research_anatomy_grid(hu.shape, affine)

    def padding_region(values, transform, source_masks, cfg, **options):
        return localization.Localization(values == -1024, np.zeros_like(values, bool), {}, 0, [])

    monkeypatch.setattr(localization, "automatic_region", padding_region)
    with pytest.raises(CountingError) as failure:
        localization.localize_native(hu, affine, masks, CountingConfig(), plan,
                                     localization_policy=LOCALIZATION_POLICY)
    assert failure.value.code == "TARGET_CLIPPED"


@pytest.mark.parametrize("fallback", [False, True])
def test_research_dispatch_forwards_policy_without_mutating_native_arrays(monkeypatch, fallback):
    hu = np.zeros((24, 20, 16), np.float32)
    affine = affine_for((0.4, 0.7, 0.9), 17)
    masks = landmarks(hu.shape)
    originals = {name: mask.copy() for name, mask in masks.items()}
    calls = []
    expected = localization.Localization(np.zeros_like(hu, bool), np.zeros_like(hu, bool), {}, 0, [])
    expected.region[6:18, 5:15, 4:12] = True

    def infer(values, transform, source_masks, cfg, grid, *, localization_policy, diagnostics):
        assert values is hu
        assert np.array_equal(transform, affine)
        assert localization_policy == LOCALIZATION_POLICY
        assert cfg == CountingConfig()
        diagnostics["synthetic_check"] = True
        calls.append(grid)
        return expected

    monkeypatch.setattr(localization, "localize_native", infer)
    diagnostic = {}
    result = research.research_localize(hu, affine, masks, CountingConfig(),
        {"proxy_max_voxels": 1 if fallback else 128_000_000, "crop_margin_mm": [12, 24]}, diagnostic)
    assert result is expected
    assert len(calls) == 1
    assert diagnostic["localization"]["synthetic_check"] is True
    assert diagnostic["mode"] == ("research_bounded_resolution_anatomy" if fallback else "full_anatomical_extent")
    if fallback:
        assert [item["outcome"] for item in diagnostic["attempts"]] == ["RESOURCE_LIMIT", "RESOURCE_LIMIT"]
        assert calls[0]["spacing_mm"][0] >= 0.5
    assert not hu.any()
    for name in masks:
        assert np.array_equal(masks[name], originals[name])


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("reason", ["SPHENOID_UNRESOLVED", "SPHENOID_AMBIGUOUS", "EMPTY_REGION"])
def test_anatomical_uncertainty_does_not_search_successively_coarser_grids(monkeypatch, fallback, reason):
    hu = np.zeros((12, 12, 12), np.float32)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args[4])
        raise CountingError(reason, "Synthetic anatomical uncertainty")

    monkeypatch.setattr(localization, "localize_native", fail)
    with pytest.raises(CountingError) as failure:
        research.research_localize(hu, affine_for((0.4, 0.6, 0.9), 17), landmarks(hu.shape),
            CountingConfig(), {"proxy_max_voxels": 1 if fallback else 128_000_000,
                               "crop_margin_mm": [12, 24]}, {})
    assert failure.value.code == reason
    assert len(calls) == 1


def synthetic_source(tmp_path, *, flipped=False):
    hu = np.full((48, 40, 32), 800, np.float32)
    hu[8:20, 10:30, 8:24] = -900
    hu[28:40, 10:30, 8:24] = -900
    # A partial opacity boundary checks that air labels still use native HU.
    hu[8:12, 10:30, 8:24] = 35
    affine = affine_for((0.5, 0.7, 1.0), 19, flipped=flipped)
    affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(hu.shape) - 1) / 2)
    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), path)
    return path, hu, nib.load(path).affine


def test_research_plan_is_sealed_independent_and_preserves_frozen_cell_policy(tmp_path):
    source, _, _ = synthetic_source(tmp_path)
    profile = build_scan_profile(source)
    baseline = build_processing_plan(profile)
    baseline_copy = deepcopy(baseline)
    plan = build_research_plan(profile)
    assert sealed(plan) == plan
    assert plan["source_sha256"] == profile["source_sha256"]
    assert plan["scan_profile_sha256"] == profile["sha256"]
    assert plan["counting_grid"] == "original_native"
    assert plan["cell_separation_parameters"] == baseline["cell_separation_parameters"] == "frozen_v2_unchanged"
    assert plan["bone_support_max_mm"] == baseline["bone_support_max_mm"] == 1.01
    assert plan["automatic_completion_guaranteed"] is False
    plan["localization_parameters"]["sphenoid_seed_clearance_mm"] = 99
    plan["anatomy_proxy_fallback"]["maximum_spacing_mm"] = 99
    assert LOCALIZATION_POLICY["sphenoid_seed_clearance_mm"] == 1
    assert ANATOMY_PROXY_POLICY["maximum_spacing_mm"] == 1
    assert build_processing_plan(profile) == baseline_copy
    assert build_research_plan(profile) != plan
    protocol = json.loads(files("ethmoid_pipeline.counting").joinpath("protocol.research.v4.json").read_text())
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    assert protocol["algorithm"] == load_counting_protocol()["algorithm"]
    assert protocol["algorithm"] == load_counting_protocol(adaptive.PROTOCOL_VERSION)["algorithm"]
    assert protocol["algorithm"] == asdict(CountingConfig())
    assert load_counting_protocol()["protocol_version"] == "ethmoid-count-prototype-v2"
    with pytest.raises(ValueError, match="Unsupported"):
        load_counting_protocol(PROTOCOL_VERSION)


@pytest.mark.parametrize("flipped", [False, True])
def test_actual_engine_publishes_native_ids_hu_and_coordinates_after_research_proxy(tmp_path, monkeypatch, flipped):
    source, hu, affine = synthetic_source(tmp_path, flipped=flipped)
    source_digest = file_sha256(source)
    source_masks = landmarks(hu.shape)
    canonical = nib.as_closest_canonical(nib.load(source))
    mask_orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                                       nib.orientations.io_orientation(canonical.affine))
    canonical_masks = {name: nib.orientations.apply_orientation(mask, mask_orientation)
                       for name, mask in source_masks.items()}
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **kw: canonical_masks)
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **kw: pytest.fail("Unexpected GPU/localizer call"))
    seen = []

    def synthetic_localization(values, transform, masks, config, *, localization_policy, diagnostics):
        assert localization_policy == LOCALIZATION_POLICY
        positions = np.moveaxis(np.indices(values.shape), 0, -1)
        native_positions = nib.affines.apply_affine(np.linalg.inv(affine) @ transform, positions)
        region = np.all((native_positions > [4, 6, 4]) & (native_positions < [43, 33, 27]), axis=-1)
        seen.append((values.shape, transform.copy(), asdict(config)))
        return localization.Localization(region, np.zeros(values.shape, bool), {}, 0, ["Synthetic anatomy stub"])

    monkeypatch.setattr(localization, "automatic_region", synthetic_localization)
    # Force the research resource-fallback path on a small phantom. The actual
    # proxy interpolation, region mapping, counting and artifact checks run.
    real_grid = research.localization_grid

    def unavailable_native_grid(shape, transform):
        grid = real_grid(shape, transform)
        grid["shape"] = [128_000_001, 1, 1]
        return grid

    monkeypatch.setattr(research, "localization_grid", unavailable_native_grid)
    output = tmp_path / "research-output"
    result = research.analyze_research_volume(source, output, tmp_path / "synthetic-cache")
    assert result["status"] == "completed", result
    assert result["method_version"] == METHOD_VERSION
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["outcome"] == "automatic_candidate_count"
    assert result["research_status"] == "NOT_ACCURACY_VALIDATED"
    assert result["artifact_verification"]["status"] == "verified"
    assert result["localization_geometry"]["mode"] == "research_bounded_resolution_anatomy"
    assert result["processing_plan"] == build_research_plan(result["scan_profile"])
    assert len(seen) == 1 and seen[0][0] != hu.shape
    assert seen[0][2] == asdict(CountingConfig())
    assert file_sha256(source) == source_digest
    assert np.array_equal(nib.load(source).get_fdata(dtype=np.float32), hu)
    cavities = nib.load(output / result["artifacts"]["cavities"])
    air = nib.load(output / result["artifacts"]["air"])
    assert cavities.shape == air.shape == hu.shape
    assert np.array_equal(cavities.affine, affine)
    assert np.array_equal(air.affine, affine)
    assert cavities.get_data_dtype() == air.get_data_dtype() == np.dtype("uint16")
    expected_ids, expected_air, expected_cells, _ = compartment_stage(hu, np.ones_like(hu, bool), affine)
    assert np.array_equal(np.asanyarray(cavities.dataobj), expected_ids)
    assert np.array_equal(np.asanyarray(air.dataobj), expected_air)
    assert result["counts"] == {"left": 1, "right": 1, "total": 2}
    for actual, expected in zip(result["cells"], expected_cells):
        for field in ("id", "side", "voxel_count", "air_voxel_count"):
            assert actual[field] == expected[field]
        assert actual["centroid_ras_mm"] == pytest.approx(expected["centroid_ras_mm"])


@pytest.mark.parametrize("reason", ["SPHENOID_UNRESOLVED", "ORBIT_FLOOR_UNRESOLVED",
                                    "ORBIT_WALL_UNRESOLVED", "ROOF_UNRESOLVED"])
def test_actual_research_engine_keeps_failed_anatomy_unavailable_without_retry(tmp_path, monkeypatch, reason):
    source, hu, _ = synthetic_source(tmp_path)
    calls = []
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **kw: landmarks(hu.shape))
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **kw: pytest.fail("Retried anatomical uncertainty"))

    def fail(*args, **kwargs):
        calls.append(kwargs["localization_policy"])
        raise CountingError(reason, "Synthetic anatomy remains unresolved")

    monkeypatch.setattr(localization, "automatic_region", fail)
    output = tmp_path / "failed-output"
    result = research.analyze_research_volume(source, output, tmp_path / "synthetic-cache")
    assert calls == [LOCALIZATION_POLICY]
    assert result["status"] == "failed"
    assert result["outcome"] == "needs_review"
    assert result["failure_code"] == reason
    assert result["counts"] == {"left": None, "right": None, "total": None}
    assert result["cells"] == [] and result["artifacts"] == {}
    assert not list(output.glob("*.nii.gz"))
    assert json.loads((output / "result.json").read_text())["counts"]["total"] is None


@pytest.mark.parametrize("reason,expected_outcome", [
    ("ORBIT_FLOOR_UNRESOLVED", "needs_review"),
    ("ORBIT_WALL_UNRESOLVED", "needs_review"),
    ("ROOF_UNRESOLVED", "needs_review"),
    ("RESOURCE_LIMIT", "technical_error"),
    ("ENGINE_ERROR", "technical_error"),
    ("SOURCE_CHANGED", "technical_error"),
    ("ARTIFACT_VERIFICATION", "technical_error"),
    ("PROCESSING_PLAN_MISMATCH", "technical_error"),
])
def test_research_finalization_clears_partial_results_and_distinguishes_failure_outcomes(
    tmp_path, reason, expected_outcome,
):
    progress_path = tmp_path / "progress.json"
    strategy = research.ResearchStrategy(progress_path=progress_path)
    result = {
        **strategy.initial_metadata(),
        "status": "completed", "failure_code": reason,
        "counts": {"left": 1, "right": 1, "total": 2},
        "cells": [{"id": 1}, {"id": 2}],
        "artifacts": {"cavities": "partial-cavities.nii.gz"},
    }
    frozen_v3_result = deepcopy(result)
    strategy.finalize(result)
    assert result["status"] == "failed"
    assert result["failure_code"] == reason
    assert result["outcome"] == expected_outcome
    assert result["counts"] == {"left": None, "right": None, "total": None}
    assert result["cells"] == []
    assert result["artifacts"] == {}
    assert result["method_version"] == METHOD_VERSION
    assert json.loads(progress_path.read_text())["stage"] == expected_outcome
    # The additional review reasons belong to the research strategy only.
    adaptive.AdaptiveStrategy().finalize(frozen_v3_result)
    assert frozen_v3_result["outcome"] == "technical_error"


def test_research_plan_cannot_be_substituted_for_frozen_v3_or_another_source(tmp_path, monkeypatch):
    source, _, _ = synthetic_source(tmp_path)
    profile = build_scan_profile(source)
    research_plan = build_research_plan(profile)
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **kw: pytest.fail("Inferred before validating the plan"))
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **kw: pytest.fail("Loaded masks before validating the plan"))
    legacy = adaptive.analyze_adaptive_volume(source, tmp_path / "v3-plan", processing_plan=research_plan)
    assert legacy["failure_code"] == "PROCESSING_PLAN_MISMATCH"
    assert legacy["method_version"] == adaptive.METHOD_VERSION
    wrong_source_plan = sealed({**research_plan, "source_sha256": "f" * 64})
    invalid = research.analyze_research_volume(source, tmp_path / "wrong-source", processing_plan=wrong_source_plan)
    assert invalid["failure_code"] == "PROCESSING_PLAN_MISMATCH"
    frozen = load_counting_protocol(adaptive.PROTOCOL_VERSION)
    crossed = research.analyze_research_volume(source, tmp_path / "wrong-protocol", config=frozen)
    assert crossed["failure_code"] == "INVALID_CONFIG"
    for result in (legacy, invalid, crossed):
        assert result["counts"]["total"] is None
        assert result["artifacts"] == {}
