"""Procedural engineering controls, never evidence of anatomical accuracy."""
from dataclasses import FrozenInstanceError, replace
from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting.synthetic_walls import (
    BONE_HU, GEOMETRY_FAMILIES, SceneAugmentation, candidate_targets,
    make_geometry, render_scene, split_geometry_families,
)


def native_points(scene):
    bone = scene.hu >= BONE_HU
    first = bone.copy()
    first[:, 1:, :] &= ~bone[:, :-1, :]
    first[:, 0, :] = False
    y = np.arange(scene.hu.shape[1])[None, :, None]
    front = np.max(np.where(scene.seed, y, -1), axis=1)
    return np.argwhere(first & (front[:, None, :] >= 0) & (y > front[:, None, :]))


def world_points(scene, indices):
    return np.asarray(scene.origin_mm) + indices * np.asarray(scene.spacing_mm)


@pytest.mark.parametrize("family", GEOMETRY_FAMILIES)
def test_equation_families_produce_bounded_observed_targets_and_distractors(family):
    scene = render_scene(make_geometry(0, family))
    assert scene.hu.dtype == np.float32 and scene.hu.ndim == 3
    assert np.isfinite(scene.hu).all() and scene.hu.size <= 2_000_000
    assert scene.seed.dtype == scene.outer_wall.dtype == scene.wall_support.dtype == np.dtype(bool)
    assert scene.observable and not scene.ambiguous and scene.support_fraction >= 0.8
    assert np.all(scene.outer_wall[scene.wall_support])
    assert np.all(scene.hu[scene.wall_support] >= BONE_HU)
    assert np.all(scene.hu[scene.seed] < BONE_HU)
    points = native_points(scene)
    labels = candidate_targets(scene, points)
    assert set(labels) == {0, 1}
    assert np.count_nonzero(labels == 1) == np.count_nonzero(scene.wall_support)
    assert scene.provenance == "procedural_equations_not_expert_or_patient_anatomy"


def test_planar_target_comes_from_known_outer_wall_not_first_bone_hit():
    geometry = replace(make_geometry(4, "plane"), wall_offset_mm=4.0,
                       tilt_x=0.0, tilt_z=0.0, wall_thickness_mm=1.6)
    scene = render_scene(geometry, SceneAugmentation(spacing_mm=(0.5, 0.5, 0.5)))
    points = native_points(scene)
    labels = candidate_targets(scene, points)
    y = world_points(scene, points)[:, 1]
    # Physical slab [3.2,4.8] mm intersects these sampled target voxels.
    assert np.all((y[labels == 1] + 0.25 >= 3.2) & (y[labels == 1] - 0.25 <= 4.8))
    assert np.any((labels == 0) & (y < 3.2))
    assert np.any((labels == 0) & (y > 4.8))
    columns = {}
    for point, label in zip(points, labels):
        columns.setdefault((int(point[0]), int(point[2])), []).append((point[1], label))
    assert any(sorted(items)[0][1] == 0 and any(label == 1 for _, label in items)
               for items in columns.values())
    permutation = np.arange(len(points))[::-1]
    assert np.array_equal(candidate_targets(scene, points[permutation]), labels[permutation])


def test_ids_and_rendering_are_deterministic_without_global_random_state():
    geometry = make_geometry(17, "asymmetric", seed=8)
    assert geometry == make_geometry(17, "asymmetric", seed=8)
    assert geometry.geometry_id != make_geometry(18, "asymmetric", seed=8).geometry_id
    assert geometry.geometry_id != make_geometry(17, "saddle", seed=8).geometry_id
    augmentation = SceneAugmentation(noise_std_hu=75, blur_sigma_mm=0.4, seed=123)
    before = np.random.get_state()
    first, second = render_scene(geometry, augmentation), render_scene(geometry, augmentation)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert np.array_equal(first.hu, second.hu) and np.array_equal(first.wall_support, second.wall_support)
    assert first.augmentation_id == second.augmentation_id
    clean = render_scene(geometry)
    assert clean.geometry_id == first.geometry_id and clean.augmentation_id != first.augmentation_id
    assert not np.array_equal(clean.hu, first.hu)
    assert np.array_equal(clean.outer_wall, first.outer_wall)


def test_geometry_and_all_returned_arrays_are_immutable():
    geometry = make_geometry(1)
    with pytest.raises(FrozenInstanceError):
        geometry.index = 3
    scene = render_scene(geometry)
    for array in (scene.hu, scene.seed, scene.analytic_seed, scene.outer_wall, scene.wall_support,
                  scene.target_surface_y_mm, scene.target_leading_surface_y_mm, scene.support_columns):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 1
    render_scene(geometry, SceneAugmentation(noise_std_hu=30))
    assert geometry == make_geometry(1)


@pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (0.4, 0.8, 2.5), (1.25, 0.625, 2.0)])
def test_anisotropic_sampling_uses_physical_slab_geometry(spacing):
    geometry = replace(make_geometry(2, "plane"), wall_offset_mm=4.0,
                       tilt_x=0.0, tilt_z=0.0, wall_thickness_mm=1.8)
    scene = render_scene(geometry, SceneAugmentation(spacing_mm=spacing))
    assert scene.spacing_mm == spacing and scene.geometry_id == geometry.geometry_id
    expected_shape = tuple(int(np.ceil(mm / step)) for mm, step in zip(geometry.extent_mm, spacing))
    assert scene.hu.shape == expected_shape
    points = np.argwhere(scene.outer_wall)
    ys = world_points(scene, points)[:, 1]
    assert np.all(ys + spacing[1] / 2 >= 3.1 - 1e-5)
    assert np.all(ys - spacing[1] / 2 <= 4.9 + 1e-5)
    assert np.allclose(scene.target_surface_y_mm, 4.0)
    assert np.allclose(scene.target_leading_surface_y_mm, 3.1)
    assert scene.observable


@pytest.mark.parametrize("opacity", ["air", "full", "partial"])
def test_opacity_changes_chamber_signal_without_creating_expert_anatomy(opacity):
    geometry = make_geometry(1)
    scene = render_scene(geometry, SceneAugmentation(opacity=opacity))
    expected = {"air": {-900.0}, "full": {scene.soft_tissue_hu}, "partial": {-900.0, scene.soft_tissue_hu}}[opacity]
    assert set(np.unique(scene.hu[scene.seed])) == expected
    assert scene.observable and scene.geometry_id == geometry.geometry_id


@pytest.mark.parametrize("state", ["missing", "averaged_away"])
@pytest.mark.parametrize("noise", [0, 250])
def test_missing_or_erased_wall_cannot_receive_recovered_support(state, noise):
    scene = render_scene(make_geometry(1), SceneAugmentation(wall_state=state, noise_std_hu=noise, blur_sigma_mm=0.4))
    assert not scene.observable and not scene.wall_support.any() and scene.support_fraction == 0
    assert not candidate_targets(scene, native_points(scene)).any()
    assert bool(scene.outer_wall.any()) == (state == "averaged_away")
    assert np.any(scene.hu >= BONE_HU)  # Distractors remain observable.


def test_ambiguous_bone_sheets_do_not_become_forced_choice_targets():
    scene = render_scene(make_geometry(5), SceneAugmentation(wall_state="ambiguous"))
    points = native_points(scene)
    assert len(points) > 0 and scene.outer_wall.any()
    assert scene.ambiguous and not scene.observable and not scene.wall_support.any()
    assert np.all(candidate_targets(scene, points) == -1)


def test_cropping_preserves_observed_partial_support_without_claiming_complete_wall():
    geometry = make_geometry(2, "tilted")
    complete = render_scene(geometry)
    cropped = render_scene(geometry, SceneAugmentation(wall_state="cropped"))
    assert cropped.cropped and not cropped.observable
    assert cropped.geometry_id == complete.geometry_id and cropped.hu.shape[1] < complete.hu.shape[1]
    assert cropped.wall_support.any() and cropped.support_fraction < 1
    assert np.all(cropped.outer_wall[cropped.wall_support])
    assert np.all(cropped.hu[cropped.wall_support] >= BONE_HU)


def test_family_splits_are_complete_disjoint_and_order_independent_before_augmentation():
    geometries = [make_geometry(i, family) for family in GEOMETRY_FAMILIES for i in range(3)]
    split = split_geometry_families(geometries, seed=23)
    assert split == split_geometry_families(reversed(geometries), seed=23)
    assert all(split.values())
    assert {g.geometry_id for values in split.values() for g in values} == {g.geometry_id for g in geometries}
    for left, right in combinations(split.values(), 2):
        assert not {g.geometry_id for g in left} & {g.geometry_id for g in right}
        assert not {g.family for g in left} & {g.family for g in right}
    assigned = {g.geometry_id: name for name, values in split.items() for g in values}
    for name, values in split.items():
        geometry = values[0]
        for augmentation in (SceneAugmentation(), SceneAugmentation(spacing_mm=(0.5, 1.25, 2.0), noise_std_hu=40, seed=3)):
            scene = render_scene(geometry, augmentation)
            assert assigned[scene.geometry_id] == name


def test_split_rejects_duplicate_geometries_too_few_families_and_invalid_ratios():
    geometries = [make_geometry(0, family) for family in GEOMETRY_FAMILIES]
    with pytest.raises(ValueError, match="duplicated"):
        split_geometry_families([*geometries, geometries[0]])
    with pytest.raises(ValueError, match="three independent"):
        split_geometry_families(geometries[:2])
    for fractions in ((1, 0, 0), (0.2, 0.2, 0.2), (float("nan"), 0.5, 0.5)):
        with pytest.raises(ValueError, match="fractions"):
            split_geometry_families(geometries, fractions=fractions)


def test_resource_limit_fails_before_any_coordinate_or_volume_allocation(monkeypatch):
    geometry = make_geometry(0)
    monkeypatch.setattr(np, "arange", lambda *a, **kw: pytest.fail("Coordinates allocated before budget rejection"))
    with pytest.raises(ValueError, match="voxel budget"):
        render_scene(geometry, SceneAugmentation(spacing_mm=(0.01, 0.01, 0.01)))
    with pytest.raises(ValueError, match="voxel budget"):
        render_scene(geometry, max_voxels=100)
    with pytest.raises(ValueError, match="voxel budget"):
        render_scene(geometry, SceneAugmentation(spacing_mm=(1e-320, 1e-320, 1e-320)))


def test_leading_surface_projects_normal_thickness_in_a_tilted_wall():
    geometry = replace(make_geometry(2, "plane"), tilt_x=0.4, tilt_z=0.3, wall_thickness_mm=2)
    scene = render_scene(geometry)
    assert np.allclose(scene.target_surface_y_mm - scene.target_leading_surface_y_mm, np.sqrt(1.25))


def test_noisy_opaque_seed_is_bone_excluded_without_changing_analytic_geometry():
    from ethmoid_pipeline.counting.wall_proposals import extract_candidates
    geometry = make_geometry(3, "smooth_bowl", seed=34)
    augmentation = SceneAugmentation(opacity="partial", noise_std_hu=60, blur_sigma_mm=0.2, seed=66)
    scene = render_scene(geometry, augmentation)
    assert np.array_equal(scene.seed, scene.analytic_seed & (scene.hu < BONE_HU))
    assert not np.any(scene.hu[scene.seed] >= BONE_HU)
    clean = render_scene(geometry)
    assert np.array_equal(clean.outer_wall, scene.outer_wall)
    assert np.array_equal(clean.target_surface_y_mm, scene.target_surface_y_mm)
    found = extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
    assert len(found.positions_ijk) > 0


def test_wall_and_distractor_thicknesses_overlap_independently():
    geometries = [make_geometry(i, "plane") for i in range(40)]
    assert all(1.2 <= g.wall_thickness_mm <= 2 and 1.2 <= g.distractor_thickness_mm <= 2 for g in geometries)
    assert any(g.wall_thickness_mm < g.distractor_thickness_mm for g in geometries)
    assert any(g.wall_thickness_mm > g.distractor_thickness_mm for g in geometries)
    assert min(g.wall_thickness_mm for g in geometries) < max(g.distractor_thickness_mm for g in geometries)
    assert min(g.distractor_thickness_mm for g in geometries) < max(g.wall_thickness_mm for g in geometries)


def test_full_opacity_has_no_fixed_inside_outside_intensity_shortcut():
    geometry = make_geometry(1)
    samples = [render_scene(geometry, SceneAugmentation(opacity="full", seed=i)) for i in range(3)]
    assert len({s.soft_tissue_hu for s in samples}) == 3
    for scene in samples:
        assert -20 <= scene.soft_tissue_hu <= 80
        assert np.all(scene.hu[scene.seed] == scene.soft_tissue_hu)
        assert scene.hu[0, 0, 0] == scene.soft_tissue_hu


def test_noisy_full_opacity_regressions_retain_analytic_seed_and_exclude_bone():
    from ethmoid_pipeline.counting.wall_proposals import extract_candidates
    removed = 0
    for seed in range(5):
        scene = render_scene(make_geometry(3), SceneAugmentation(opacity="full", noise_std_hu=100, seed=seed))
        removed += np.count_nonzero(scene.analytic_seed & ~scene.seed)
        assert np.array_equal(scene.seed, scene.analytic_seed & (scene.hu < BONE_HU))
        extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
    assert removed > 0


@pytest.mark.parametrize("augmentation", [
    SceneAugmentation(spacing_mm=(0, 1, 1)), SceneAugmentation(spacing_mm=(1, float("nan"), 1)),
    SceneAugmentation(opacity="invalid"), SceneAugmentation(wall_state="recovered"),
    SceneAugmentation(noise_std_hu=-1), SceneAugmentation(blur_sigma_mm=float("inf")),
])
def test_invalid_sampling_is_explicit(augmentation):
    with pytest.raises(ValueError):
        render_scene(make_geometry(0), augmentation)


@pytest.mark.parametrize("points", [np.array([[0.5, 1, 2]]), np.array([[-1, 0, 0]]),
                                    np.array([[9999, 0, 0]]), np.array([1, 2, 3])])
def test_candidate_coordinates_must_be_on_the_native_grid(points):
    with pytest.raises(ValueError, match="native voxel"):
        candidate_targets(render_scene(make_geometry(0)), points)


def test_empty_candidates_are_valid_and_do_not_fabricate_a_target():
    labels = candidate_targets(render_scene(make_geometry(0)), np.empty((0, 3), dtype=np.int64))
    assert labels.dtype == np.int8 and labels.shape == (0,)
