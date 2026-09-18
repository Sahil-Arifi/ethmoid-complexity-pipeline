"""Research localization phantoms; these are not clinical accuracy claims."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, compartment_stage
from ethmoid_pipeline.counting.localization import _sphenoid_boundary, localize_native


RESEARCH_POLICY = {"sphenoid_seed_clearance_mm": 1.0, "physical_ray_extent": True}


def sinus_phantom(spacing=(0.5, 0.5, 0.625), *, anterior_mm=20.0):
    """Sample the same bone-supported box and landmark locations in mm."""
    spacing = np.asarray(spacing, dtype=float)
    shape = np.ceil(np.array([60, 40, 32]) / spacing).astype(int)
    x, y, z = np.ogrid[tuple(slice(0, int(n)) for n in shape)]
    x, y, z = x * spacing[0], y * spacing[1], z * spacing[2]
    inside = (x >= 28) & (x < 32) & (y >= 10) & (y < anterior_mm) & (z >= 8) & (z < 24)
    hu = np.where(inside, -900, 800).astype(np.float32)
    nerves = (((x >= 12) & (x < 14)) | ((x >= 46) & (x < 48))) & (y >= 20) & (y < 20 + spacing[1]) & (z >= 8) & (z < 28)
    eye_bounds = (np.array([10 / spacing[0], 0, 0]), np.array([50 / spacing[0], 0, 0]))
    args = (hu, hu >= 200, inside, inside.copy(), nerves, eye_bounds, int(3 / spacing[2]), spacing, 30 / spacing[0])
    return args


@pytest.mark.parametrize("interval", [0.625, 1.25, 2.0, 2.5])
def test_same_physical_sinus_gets_bone_supported_wall_across_slice_intervals(interval):
    args = sinus_phantom((0.5, 0.5, interval))
    original_hu = args[0].copy()
    diagnostic = {}
    parent, wall = _sphenoid_boundary(*args, localization_policy=RESEARCH_POLICY, diagnostics=diagnostic)
    assert np.array_equal(args[0], original_hu)
    assert parent.any() and not np.any(parent & args[1])
    assert np.all(wall == round(20 / args[7][1]))
    assert diagnostic["sphenoid"]["seed_clearance_mm"] == 1.0
    assert diagnostic["sphenoid"]["interior_seed_voxels"] > 0
    assert diagnostic["sphenoid"]["supported_wall_rays"] >= diagnostic["sphenoid"]["required_wall_rays"]


@pytest.mark.parametrize("interval", [1.25, 2.0, 2.5])
def test_default_keeps_legacy_spacing_dependent_clearance_and_abstention(interval):
    args = sinus_phantom((0.5, 0.5, interval))
    for options in ({}, {"localization_policy": None}):
        diagnostic = {}
        with pytest.raises(CountingError, match="no spatially resolved interior") as failure:
            _sphenoid_boundary(*args, diagnostics=diagnostic, **options)
        assert failure.value.code == "SPHENOID_UNRESOLVED"
        assert diagnostic["sphenoid"]["seed_clearance_mm"] == 2 * interval
        assert diagnostic["sphenoid"]["interior_seed_voxels"] == 0


def test_default_and_explicit_none_produce_identical_thin_wall_and_parent():
    args = sinus_phantom()
    ordinary = _sphenoid_boundary(*args)
    diagnostic = {}
    explicit = _sphenoid_boundary(*args, localization_policy=None, diagnostics=diagnostic)
    assert all(np.array_equal(a, b) for a, b in zip(ordinary, explicit))
    assert diagnostic["sphenoid"]["seed_clearance_mm"] == 1.25


@pytest.mark.parametrize("xy", [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0)])
def test_ray_search_extent_is_physical_under_unequal_xy_sampling(xy):
    args = sinus_phantom((*xy, 0.625), anterior_mm=28)
    diagnostic = {}
    _, wall = _sphenoid_boundary(*args, localization_policy=RESEARCH_POLICY, diagnostics=diagnostic)
    assert diagnostic["sphenoid"]["ray_search_extent_mm"] == pytest.approx(10)
    assert np.all(wall * xy[1] == 28)
    # Previously X voxel width was directly added to a Y index. On this grid
    # that searches only 5 mm instead of 10 mm, missing a wall 8 mm away.
    if xy == (1.0, 0.5):
        legacy_diagnostic = {}
        with pytest.raises(CountingError, match="supported anterior bony wall"):
            _sphenoid_boundary(*args, diagnostics=legacy_diagnostic)
        assert legacy_diagnostic["sphenoid"]["ray_search_extent_mm"] == pytest.approx(5)


def test_research_seed_does_not_manufacture_missing_anterior_bone():
    args = list(sinus_phantom((0.5, 0.5, 2.5)))
    # Retain the side/posterior support while removing every anterior ray hit.
    args[0][:, 40:, :] = -900
    args[1] = args[0] >= 200
    diagnostic = {}
    with pytest.raises(CountingError, match="supported anterior bony wall") as failure:
        _sphenoid_boundary(*args, localization_policy=RESEARCH_POLICY, diagnostics=diagnostic)
    assert failure.value.code == "SPHENOID_UNRESOLVED"
    assert diagnostic["sphenoid"]["interior_seed_voxels"] > 0
    assert diagnostic["sphenoid"]["supported_wall_rays"] == 0


@pytest.mark.parametrize("interval", [1.25, 2.0, 2.5])
def test_research_localization_does_not_resolve_subslice_cell(interval):
    args = sinus_phantom((0.5, 0.5, interval))
    _sphenoid_boundary(*args, localization_policy=RESEARCH_POLICY)
    hu = np.full((24, 24, 12), 800, np.float32)
    hu[6:18, 6:18, 6] = -900  # One acquired slice; no second resolved plane.
    original = hu.copy()
    with pytest.raises(CountingError) as failure:
        compartment_stage(hu, np.ones_like(hu, bool), np.diag([0.5, 0.5, interval, 1]))
    assert failure.value.code == "NO_SUPPORTED_CELLS"
    assert np.array_equal(hu, original)


@pytest.mark.parametrize("policy", [{}, {**RESEARCH_POLICY, "sphenoid_seed_clearance_mm": 0},
    {**RESEARCH_POLICY, "sphenoid_seed_clearance_mm": True}, {**RESEARCH_POLICY, "physical_ray_extent": False}])
def test_unversioned_research_settings_are_rejected(policy):
    with pytest.raises(CountingError) as failure:
        _sphenoid_boundary(*sinus_phantom(), localization_policy=policy)
    assert failure.value.code == "INVALID_CONFIG"


@pytest.mark.parametrize("oblique", [False, True])
def test_policy_and_diagnostics_reach_automatic_region_on_native_and_proxy_grids(monkeypatch, oblique):
    import ethmoid_pipeline.counting.grids as grids
    import ethmoid_pipeline.counting.localization as localization
    hu = np.zeros((12, 12, 12), np.float32)
    region = np.zeros_like(hu, bool)
    region[4:8, 4:8, 4:8] = True
    masks = {name: np.zeros_like(region) for name in ("brain", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right")}
    diagnostic = {}
    def automatic(values, affine, landmarks, config, *, localization_policy, diagnostics):
        assert localization_policy is RESEARCH_POLICY and diagnostics is diagnostic
        diagnostics["sphenoid"] = {"seed_clearance_mm": 1.0}
        return localization.Localization(region, np.zeros_like(region), {}, 0.0, [])
    monkeypatch.setattr(localization, "automatic_region", automatic)
    if oblique:
        monkeypatch.setattr(grids, "anatomy_proxy", lambda *args: (hu, np.eye(4), masks, np.ones_like(region)))
    result = localize_native(hu, np.eye(4), masks, CountingConfig(),
        {"mode": "anatomy_only_ras_proxy" if oblique else "native_axis_aligned"},
        localization_policy=RESEARCH_POLICY, diagnostics=diagnostic)
    assert np.array_equal(result.region, region)
    assert diagnostic["sphenoid"]["seed_clearance_mm"] == 1.0
