"""Independent procedural controls, not clinical anatomy or counting tests."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting.wall_proposals import (
    WallProposalError, extract_candidates, candidate_patch, decode_surface,
    _IntervalWorkBudget, _interval_domain_extreme, MAX_FEASIBILITY_WORK,
)


def scene(*, wall_y=16, spacing=(0.5, 0.5, 0.5)):
    hu = np.full((20, 32, 20), -900.0, np.float32)
    seed = np.zeros(hu.shape, bool)
    seed[4:16, 4:8, 4:16] = True
    if wall_y is not None:
        hu[4:16, wall_y:wall_y + 2, 4:16] = 800
    return hu, seed, spacing


def extract(hu, seed, spacing=(0.5, 0.5, 0.5), **kwargs):
    return extract_candidates(hu, seed, spacing, 29, **kwargs)


def fails(code, callable_, *args, **kwargs):
    with pytest.raises(WallProposalError) as result:
        callable_(*args, **kwargs)
    assert result.value.code == code


def test_all_runs_are_candidates_not_only_first_hit_and_source_is_unchanged():
    hu, seed, spacing = scene()
    hu[4:16, 11:13, 4:16] = 400
    before, seed_before = hu.copy(), seed.copy()
    found = extract(hu, seed, spacing)
    assert len(found.positions_ijk) == 288
    first = found.positions_ijk[found.column_indices == 0]
    assert first.tolist() == [[4, 11, 4], [4, 16, 4]]
    assert found.run_end_y[:2].tolist() == [12, 17]
    assert np.array_equal(hu, before) and np.array_equal(seed, seed_before)
    assert np.all(hu[tuple(found.positions_ijk.T)] >= 200)
    assert not found.positions_ijk.flags.writeable
    assert not found.seed_column_mask.flags.writeable
    assert found.scan_bounds_ijk == ((0, 20), (0, 32), (0, 20))


def test_threshold_is_inclusive_and_search_limit_does_not_invent_a_run_end():
    hu, seed, spacing = scene(wall_y=None)
    hu[4:16, 25:31, 4:16] = 200
    found = extract_candidates(hu, seed, spacing, 26)
    assert np.all(found.positions_ijk[:, 1] == 25)
    assert np.all(found.run_end_y == 30)
    hu[4:16, 25:31, 4:16] = 199.999
    assert len(extract(hu, seed).positions_ijk) == 0


@pytest.mark.parametrize("bad", [(0, 1, 1), (1, float("nan"), 1), (1, 1)])
def test_invalid_spacing_rejected(bad):
    hu, seed, _ = scene()
    fails("WALL_INPUT_INVALID", extract, hu, seed, bad)


def test_bad_seed_nonfinite_hu_and_explicit_budget_failures():
    hu, seed, _ = scene()
    fails("WALL_INPUT_INVALID", extract, hu, seed.astype(np.uint8))
    hu[5, 5, 5] = 400
    fails("WALL_INPUT_INVALID", extract, hu, seed)
    hu[5, 5, 5] = np.nan
    fails("WALL_INPUT_INVALID", extract, hu, seed)
    hu[5, 5, 5] = -900
    fails("WALL_RESOURCE_LIMIT", extract, hu, seed, max_candidates=10)
    fails("WALL_NO_SUPPORT", extract, hu, np.zeros_like(seed))


def test_patch_physical_coordinates_raw_hu_seed_and_padding_channels():
    hu, seed, spacing = scene(spacing=(0.5, 1.0, 2.0))
    # A globally linear HU field has an exact trilinear interpolation reference.
    x, y, z = np.indices(hu.shape)
    hu = (x + 10 * y + 100 * z).astype(np.float64)
    hu[seed] = -900
    found = extract(hu, seed, spacing)
    index = int(np.flatnonzero(np.all(found.positions_ijk == (10, 8, 10), axis=1))[0])
    patch = candidate_patch(hu, seed, found, index, physical_size_mm=4, output_shape=(5, 5, 5))
    assert patch.shape == (3, 5, 5, 5) and patch.dtype == np.float32
    assert patch[0, 2, 2, 2] == hu[10, 8, 10]
    # Offsets are 0.8 mm per patch sample, converted per source axis.
    assert patch[0, 3, 2, 2] == pytest.approx(hu[10, 8, 10] + 1.6)
    assert patch[0, 2, 2, 3] == pytest.approx(hu[10, 8, 10] + 40)
    assert set(np.unique(patch[1])).issubset({0, 1})
    assert np.all(patch[2] == 1)
    outside = candidate_patch(hu, seed, found, 0, physical_size_mm=32, output_shape=(16, 16, 16))
    assert np.any(outside[2] == 0)
    assert np.all(outside[0][outside[2] == 0] == -1000)
    assert np.all(outside[1][outside[2] == 0] == 0)


def test_patch_does_not_round_unchanged_float64_hu_or_modify_inputs():
    hu, seed, spacing = scene()
    hu = hu.astype(np.float64)
    hu[hu > 200] = 800.123456789
    found = extract(hu, seed, spacing)
    original = hu.copy()
    candidate_patch(hu, seed, found, 0)
    assert np.array_equal(hu, original)
    hu[tuple(found.positions_ijk[0])] += 1
    fails("WALL_INPUT_CHANGED", candidate_patch, hu, seed, found, 0)


def test_patch_resource_limit_and_invalid_extent():
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    fails("WALL_RESOURCE_LIMIT", candidate_patch, hu, seed, found, 0, output_shape=(128, 128, 128))
    fails("WALL_INPUT_INVALID", candidate_patch, hu, seed, found, 0, physical_size_mm=float("nan"))


def test_single_sheet_is_observed_and_unsupported_output_is_nan():
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    wall = decode_surface(found, np.full(len(found.positions_ijk), .8))
    assert len(wall.positions_ijk) == 144
    assert np.all(wall.positions_ijk[:, 1] == 16)
    assert np.all(hu[tuple(wall.positions_ijk.T)] >= found.bone_hu)
    assert np.all(wall.y_surface_mm[wall.supported_mask] == 8)
    assert np.all(np.isnan(wall.y_surface_mm[~wall.supported_mask]))
    assert np.array_equal(found.positions_ijk[wall.selected_candidate_indices], wall.positions_ijk)
    assert wall.diagnostics["global_optimum_claim"] is False
    assert wall.diagnostics["interpolated_support_points"] == 0


def test_curved_sheet_can_be_selected_without_a_global_quadratic_fit():
    hu, seed, spacing = scene(wall_y=None)
    truth = {}
    for x in range(4, 16):
        for z in range(4, 16):
            y = 16 + int(round(1.5 * np.sin(x / 3) + 1.5 * np.sin(z / 3)))
            hu[x, y:y + 2, z] = 800
            truth[(x, z)] = y
    found = extract(hu, seed, spacing)
    wall = decode_surface(found, np.full(len(found.positions_ijk), .8))
    assert all(truth[(x, z)] == y for x, y, z in wall.positions_ijk)


def test_wrong_high_score_cannot_create_missing_or_averaged_away_wall():
    hu, seed, spacing = scene(wall_y=None)
    found = extract(hu, seed, spacing)
    fails("WALL_NO_SUPPORT", decode_surface, found, np.ones(0))
    hu[4:16, 16:18, 4:16] = 180  # Acquisition has no thresholded supporting bone.
    found = extract(hu, seed, spacing)
    fails("WALL_NO_SUPPORT", decode_surface, found, np.ones(0))


@pytest.mark.parametrize("scores", [(.9, .9), (.99, .01), (.01, .99)])
def test_two_complete_sheets_remain_ambiguous_even_with_wrong_confident_scores(scores):
    hu, seed, spacing = scene()
    hu[4:16, 23:25, 4:16] = 800
    found = extract(hu, seed, spacing)
    probability = np.where(found.positions_ijk[:, 1] == 16, scores[0], scores[1])
    fails("WALL_AMBIGUOUS", decode_surface, found, probability)


def test_fragmentary_wrong_first_hit_does_not_replace_coherent_wall():
    hu, seed, spacing = scene()
    hu[10, 10:12, 10] = 800
    found = extract(hu, seed, spacing)
    probability = np.where(found.positions_ijk[:, 1] == 10, .999, .7)
    wall = decode_surface(found, probability)
    assert np.all(wall.positions_ijk[:, 1] == 16)


def test_enclosed_unsupported_hole_is_not_interpolated_into_support():
    hu, seed, spacing = scene()
    hu[10, :, 10] = -900
    found = extract(hu, seed, spacing)
    fails("WALL_UNSUPPORTED_HOLE", decode_surface, found, np.ones(len(found.positions_ijk)))


def test_disconnected_evidence_is_rejected():
    hu, seed, spacing = scene()
    hu[9:11, :, :] = -900
    found = extract(hu, seed, spacing)
    fails("WALL_DISCONNECTED_SUPPORT", decode_surface, found, np.ones(len(found.positions_ijk)))


@pytest.mark.parametrize("few_rays", [True, False])
def test_minimum_twelve_and_half_seeded_columns_both_apply(few_rays):
    hu, seed, spacing = scene(wall_y=None)
    if few_rays:
        seed[:] = False
        seed[4:8, 4:8, 4:8] = True
        hu[4:6, 16:18, 4:8] = 800  # Eight rays is below the absolute minimum.
    else:
        hu[4:8, 16:18, 4:16] = 800  # 48/144 is below the relative minimum.
    found = extract(hu, seed, spacing)
    fails("WALL_INSUFFICIENT_SUPPORT", decode_surface, found, np.ones(len(found.positions_ijk)))


def test_collinear_support_does_not_define_a_surface():
    hu, seed, spacing = scene(wall_y=None)
    seed[:] = False
    for i in range(4, 16):
        seed[i, 4:8, i] = True
        hu[i, 16:18, i] = 800
    found = extract(hu, seed, spacing)
    fails("WALL_SPATIAL_SUPPORT", decode_surface, found, np.ones(len(found.positions_ijk)))


@pytest.mark.parametrize("which", ["seed", "sheet", "search"])
def test_source_or_search_cropping_cannot_supply_enclosure(which):
    hu, seed, spacing = scene()
    limit = 29
    if which == "seed":
        seed[0, 4:8, 4:16] = True
    elif which == "sheet":
        hu[4:16, 16:, 4:16] = 800
    else:
        limit = 7
    found = extract_candidates(hu, seed, spacing, limit)
    fails("WALL_CROPPED", decode_surface, found, np.ones(len(found.positions_ijk)))


def test_bad_scores_low_confidence_and_candidate_complexity_are_explicit():
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    for scores in [np.ones(1), np.full(len(found.positions_ijk), np.nan), np.full(len(found.positions_ijk), 2)]:
        fails("WALL_INPUT_INVALID", decode_surface, found, scores)
    fails("WALL_SCORE_UNSUPPORTED", decode_surface, found, np.full(len(found.positions_ijk), .1))
    hu[4:16, 23:25, 4:16] = 800
    found = extract(hu, seed, spacing)
    fails("WALL_RESOURCE_LIMIT", decode_surface, found, np.ones(len(found.positions_ijk)), max_candidates_per_column=1)


def test_incoherent_sheet_abstains_and_result_is_deterministic():
    hu, seed, spacing = scene(wall_y=None)
    for x in range(4, 16):
        for z in range(4, 16):
            y = 12 if (x + z) % 2 else 24
            hu[x, y:y + 2, z] = 800
    found = extract(hu, seed, spacing)
    fails("WALL_INCOHERENT", decode_surface, found, np.ones(len(found.positions_ijk)))
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    a, b = [decode_surface(found, np.full(len(found.positions_ijk), .8)) for _ in range(2)]
    assert np.array_equal(a.selected_candidate_indices, b.selected_candidate_indices)
    assert a.diagnostics == b.diagnostics


def test_forged_subthreshold_candidate_evidence_is_rejected():
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    forged = replace(found, candidate_hu=np.zeros(len(found.positions_ijk)))
    fails("WALL_INPUT_INVALID", decode_surface, forged, np.ones(len(found.positions_ijk)))


@pytest.mark.parametrize("changes", [
    {"candidate_hu": np.ones(1)},
    {"run_end_y": np.ones(1, np.int32)},
    {"column_indices": np.ones(1, np.int32)},
    {"seed_front_y": np.ones(1, np.int32)},
    {"spacing_mm": (1.0, float("nan"), 1.0)},
    {"bone_hu": float("nan")},
    {"source_shape": (20, -32, 20)},
    {"limit_y": 100},
])
def test_incomplete_or_nonfinite_candidate_metadata_is_rejected(changes):
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    fails("WALL_INPUT_INVALID", decode_surface, replace(found, **changes), np.ones(len(found.positions_ijk)))


def test_duplicate_runs_cannot_multiply_observed_support():
    hu, seed, spacing = scene()
    hu[4:16, 23:25, 4:16] = 800
    found = extract(hu, seed, spacing)
    positions = found.positions_ijk.copy()
    positions[1] = positions[0]
    fails("WALL_INPUT_INVALID", decode_surface, replace(found, positions_ijk=positions), np.ones(len(positions)))


def test_shallow_quantized_anisotropic_wall_retains_its_exact_observed_stair_steps():
    hu, seed, _ = scene(wall_y=None)
    spacing = (.625, 2.5, .625)
    for x in range(4, 16):
        # Sample a known .5 mm/mm wall in a 2.5 mm depth grid. A single observed
        # 2.5 mm stair would wrongly appear to have slope 4 from voxel centers.
        y = int(round(12 + .5 * (x - 4) * spacing[0] / spacing[1]))
        hu[x, y:y + 2, 4:16] = 800
    found = extract(hu, seed, spacing)
    wall = decode_surface(found, np.ones(len(found.positions_ijk)))
    assert len(np.unique(wall.positions_ijk[:, 1])) == 2
    assert np.array_equal(wall.positions_ijk, found.positions_ijk)
    assert np.all(wall.y_surface_mm[wall.supported_mask] == wall.positions_ijk[:, 1] * 2.5)
    assert wall.diagnostics["depth_observation_interval_mm"] == 2.5
    assert wall.diagnostics["interpolated_support_points"] == 0


@pytest.mark.parametrize("steps_per_column", [1, 3])
def test_physically_too_steep_anisotropic_wall_cannot_spend_quantization_slack_repeatedly(steps_per_column):
    hu = np.full((20, 80, 20), -900., np.float32)
    seed = np.zeros_like(hu, bool)
    seed[4:16, 4:8, 4:16] = True
    for x in range(4, 16):
        y = 12 + (x - 4) * steps_per_column
        hu[x, y:y + 2, 4:16] = 800
    found = extract_candidates(hu, seed, (.625, 2.5, .625), 77)
    fails("WALL_INCOHERENT", decode_surface, found, np.ones(len(found.positions_ijk)))


@pytest.mark.parametrize("transform", ["flip_x", "flip_z", "exchange_xz"])
def test_observed_points_follow_surface_plane_coordinate_transforms(transform):
    hu, seed, spacing = scene(wall_y=None, spacing=(0.5, 0.8, 1.2))
    for x in range(4, 16):
        for z in range(4, 16):
            y = 16 + int(round(np.sin(x / 3) + np.sin(z / 4)))
            hu[x, y:y + 2, z] = 800
    found = extract(hu, seed, spacing)
    original = decode_surface(found, np.ones(len(found.positions_ijk)))
    expected = original.positions_ijk.copy()
    if transform == "exchange_xz":
        hu, seed = hu.transpose(2, 1, 0), seed.transpose(2, 1, 0)
        spacing = (spacing[2], spacing[1], spacing[0])
        expected = expected[:, [2, 1, 0]]
    else:
        axis = 0 if transform == "flip_x" else 2
        hu, seed = np.flip(hu, axis), np.flip(seed, axis)
        expected[:, axis] = hu.shape[axis] - 1 - expected[:, axis]
    changed = extract(hu, seed, spacing)
    observed = decode_surface(changed, np.ones(len(changed.positions_ijk)))
    assert set(map(tuple, observed.positions_ijk)) == set(map(tuple, expected))
    assert np.all(hu[tuple(observed.positions_ijk.T)] >= 200)


@pytest.mark.parametrize("preference", ["lower", "upper", "equal"])
def test_complete_rival_walls_with_variable_candidate_ordinals_cannot_hide_from_scores(preference):
    # The two complete walls are at 16 and 26. Three unrelated partial plates
    # change both walls' depth ordinals across columns. No single ordinal seed
    # describes either wall, which defeated the original bounded ICM search.
    hu = np.full((20, 42, 20), -900., np.float32)
    seed = np.zeros_like(hu, bool)
    seed[4:16, 4:8, 4:16] = True
    hu[4:16, 16:18, 4:16] = 800
    hu[4:16, 26:28, 4:16] = 800
    hu[4:8, 10:12, 4:16] = 800
    hu[8:12, 21:23, 4:16] = 800
    hu[8:16, 32:34, 4:16] = 800
    found = extract_candidates(hu, seed, (.8, .8, .8), 39)
    scores = (np.ones(len(found.positions_ijk)) if preference == "equal" else
              np.where(found.positions_ijk[:, 1] == (16 if preference == "lower" else 26), .99, .01))
    fails("WALL_AMBIGUOUS", decode_surface, found, scores)


def test_a_partial_rival_is_not_hidden_by_infeasible_irrelevant_alternatives_elsewhere():
    # 12/16 columns can switch from J10 to J14 while the last row stays J10.
    # That is a valid competing sheet. Forcing EVERY column off J10 would
    # instead select an incompatible J30 plate on the last row and miss it.
    hu = np.full((10, 36, 10), -900., np.float32)
    seed = np.zeros_like(hu, bool)
    seed[3:7, 3:6, 3:7] = True
    hu[3:7, 10:12, 3:7] = 800
    hu[3:6, 14:16, 3:7] = 800
    hu[6, 30:32, 3:7] = 800
    found = extract_candidates(hu, seed, (2., 1., 2.), 33)
    scores = np.where(found.positions_ijk[:, 1] == 10, .99, .01)
    fails("WALL_AMBIGUOUS", decode_surface, found, scores)


def test_unique_feasible_sheet_enters_ranking_even_when_all_ordinal_initializations_are_bad():
    hu = np.full((20, 42, 20), -900., np.float32)
    seed = np.zeros_like(hu, bool)
    seed[4:16, 4:8, 4:16] = True
    hu[4:16, 16:18, 4:16] = 800
    hu[4:8, 10:12, 4:16] = 800
    hu[8:12, 21:23, 4:16] = 800
    hu[12:16, 32:34, 4:16] = 800
    found = extract_candidates(hu, seed, (.8, .8, .8), 39)
    wall = decode_surface(found, np.ones(len(found.positions_ijk)), max_iterations=1)
    assert np.all(wall.positions_ijk[:, 1] == 16)
    assert wall.diagnostics["variable_feasible_columns"] == 0
    assert wall.diagnostics["global_optimum_claim"] is False
    assert wall.diagnostics["feasibility_work_units"] <= wall.diagnostics["feasibility_work_limit"]


def test_scores_can_rank_small_certified_variation_but_cannot_change_observed_voxel_coordinates():
    hu, seed, _ = scene()
    hu[10, 20:22, 10] = 800
    found = extract(hu, seed, (1., .5, 1.))
    scores = np.where(found.positions_ijk[:, 1] == 20, .99, .8)
    wall = decode_surface(found, scores, smoothness=0)
    assert wall.diagnostics["variable_feasible_columns"] == 1
    assert wall.diagnostics["variable_feasible_columns"] < wall.diagnostics["competing_sheet_required_columns"]
    assert np.count_nonzero(wall.positions_ijk[:, 1] == 20) == 1
    np.testing.assert_array_equal(wall.positions_ijk, found.positions_ijk[wall.selected_candidate_indices])
    assert np.all(hu[tuple(wall.positions_ijk.T)] >= 200)


def test_feasibility_work_exhaustion_is_an_abstention_not_infeasibility_or_unique_surface():
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    fails("WALL_RESOURCE_LIMIT", decode_surface, found, np.ones(len(found.positions_ijk)), max_feasibility_work=1)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, MAX_FEASIBILITY_WORK + 1])
def test_feasibility_budget_cannot_be_disabled_or_unbounded(budget):
    hu, seed, spacing = scene()
    found = extract(hu, seed, spacing)
    fails("WALL_INPUT_INVALID", decode_surface, found, np.ones(len(found.positions_ijk)), max_feasibility_work=budget)


def _enumerated_linear_program_extrema(centers, neighbors, half_width, slope):
    """Independent small-domain oracle: enumerate bone choices, solve each LP.

    The oracle knows no propagation, domain-ceiling or decoder implementation.
    It checks continuous interval feasibility and independently minimizes and
    maximizes every coordinate over every candidate combination.
    """
    from itertools import product
    from scipy.optimize import linprog

    n = len(centers)
    rows, limits = [], []
    for i, edges in enumerate(neighbors):
        for j, spacing in edges:
            row = np.zeros(n)
            row[i], row[j] = 1, -1
            rows.append(row)
            limits.append(slope * spacing)
    minimum, maximum = np.full(n, np.inf), np.full(n, -np.inf)
    possible = False
    for choices in product(*(range(len(domain)) for domain in centers)):
        bounds = [(centers[i][choice] - half_width, centers[i][choice] + half_width)
                  for i, choice in enumerate(choices)]
        feasibility = linprog(np.zeros(n), A_ub=rows, b_ub=limits, bounds=bounds, method="highs")
        assert feasibility.success or feasibility.status == 2
        if not feasibility.success:
            continue
        possible = True
        for i in range(n):
            objective = np.zeros(n)
            objective[i] = 1
            low = linprog(objective, A_ub=rows, b_ub=limits, bounds=bounds, method="highs")
            high = linprog(-objective, A_ub=rows, b_ub=limits, bounds=bounds, method="highs")
            assert low.success and high.success
            minimum[i] = min(minimum[i], low.fun)
            maximum[i] = max(maximum[i], -high.fun)
    return (minimum, maximum) if possible else None


@pytest.mark.parametrize("seed", range(12))
def test_interval_domain_extrema_match_exhaustive_independent_linear_programs(seed):
    rng = np.random.default_rng(seed)
    centers = [np.sort(rng.choice(np.arange(-4., 8., 2.), size=2, replace=False)) for _ in range(4)]
    # A small rectangular cycle with anisotropic physical neighbor distances.
    neighbors = [[(1, .75), (2, 1.25)], [(0, .75), (3, 1.25)],
                 [(0, 1.25), (3, .75)], [(1, 1.25), (2, .75)]]
    half_width, slope = .5, 1.3
    oracle = _enumerated_linear_program_extrema(centers, neighbors, half_width, slope)
    least = _interval_domain_extreme(centers, neighbors, half_width, slope, _IntervalWorkBudget(100_000))
    greatest = _interval_domain_extreme(centers, neighbors, half_width, slope,
                                       _IntervalWorkBudget(100_000), greatest=True)
    if oracle is None:
        assert least is None and greatest is None
    else:
        assert least is not None and greatest is not None
        np.testing.assert_allclose(least.values_mm, oracle[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(greatest.values_mm, oracle[1], atol=1e-7, rtol=0)
        for solution in (least, greatest):
            chosen = np.array([domain[index] for domain, index in zip(centers, solution.candidate_choices)])
            assert np.all(np.abs(solution.values_mm - chosen) <= half_width + 1e-7)
