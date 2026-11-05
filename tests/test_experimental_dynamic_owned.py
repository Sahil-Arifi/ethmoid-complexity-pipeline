"""Portable procedural contracts for the opt-in experimental grouping policy."""
from itertools import permutations

import numpy as np
import pytest

# The application environment intentionally omits the numerical inference stack.
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from scipy import ndimage as ndi

from ethmoid_pipeline.counting import engine
from ethmoid_pipeline.counting import experimental_dynamic_owned as policy


def direct_pair_evidence(labels, bone, spacing, config):
    """Independent slow oracle: run an EDT for each current interface pair."""
    result = {}
    for axis in range(3):
        step = max(1, int(np.ceil(config.plate_half_width_mm / spacing[axis])))
        positive = engine._shift(labels, axis, step)
        negative = engine._shift(labels, axis, -step)
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        low, high = tuple(low), tuple(high)
        left, right = labels[low], labels[high]
        boundary = (left != right) & (left > 0) & (right > 0)
        if not boundary.any():
            continue
        pairs = np.sort(np.column_stack((left[boundary], right[boundary])), axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        totals = np.bincount(inverse)
        for index, pair in enumerate(unique):
            first, second = map(int, pair)
            plate = bone & (((positive == first) & (negative == second)) |
                            ((positive == second) & (negative == first)))
            nearby = ndi.distance_transform_edt(~plate, sampling=spacing) <= config.separator_support_mm if plate.any() else np.zeros_like(bone)
            supported = (nearby[low] | nearby[high])[boundary]
            total = int(totals[index])
            count = int(np.count_nonzero(supported[inverse == index]))
            key = first, second
            if key not in result or total > result[key][0]:
                result[key] = total, count, axis
    return result


@pytest.mark.parametrize("spacing", [(0.3, 0.6, 0.9), (0.4, 0.6, 0.9), (0.5, 0.5, 0.5)])
def test_incidence_matches_direct_edt_after_unions_and_relabeling(spacing):
    spacing = np.array(spacing)
    rng = np.random.default_rng(20261104)
    labels = rng.integers(0, 6, (8, 9, 7), dtype=np.int32)
    bone = labels == 0
    for permutation in (np.arange(6), np.array([0, 5, 3, 1, 4, 2])):
        raw = permutation[labels]
        prepared = policy._prepare_incidence(raw, bone, spacing, engine.CountingConfig(), (0, 1, 2))
        for mapping in ([0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 4, 5], [0, 1, 1, 1, 4, 4], [0, 1, 1, 1, 1, 1]):
            mapping = np.array(mapping)
            assert policy._measure_incidence(prepared, mapping, (0, 1, 2)) == direct_pair_evidence(mapping[raw], bone, spacing, engine.CountingConfig())


def witness_fixture(farther_owned=False):
    labels = np.zeros((7, 8, 7), np.int32)
    labels[3, 4, 3], labels[4, 4, 3] = 1, 2
    labels[1, 3, 3], labels[5, 3, 3] = 3, 4
    bone = np.zeros(labels.shape, bool)
    bone[3, 3, 3] = True
    if farther_owned:
        bone[3, 5, 4] = True
        labels[1, 5, 4], labels[5, 5, 4] = 1, 2
    return labels, bone


def test_foreign_raw_flanks_can_acquire_current_pair_ownership():
    labels, bone = witness_fixture()
    prepared = policy._prepare_incidence(labels, bone, np.ones(3), engine.CountingConfig(), (0, 1, 2))
    assert any(tuple(row[1:]) == (3, 4) for row in prepared["incidence"])
    assert policy._measure_incidence(prepared, np.arange(5), (0, 1, 2))[(1, 2)][:2] == (1, 0)
    mapping = np.array([0, 1, 2, 1, 2])
    actual = policy._measure_incidence(prepared, mapping, (0, 1, 2))
    assert actual[(1, 2)][:2] == (1, 1)
    assert actual == direct_pair_evidence(mapping[labels], bone, np.ones(3), engine.CountingConfig())


def test_farther_owned_witness_is_not_discarded_for_nearer_foreign_witness():
    labels, bone = witness_fixture(farther_owned=True)
    prepared = policy._prepare_incidence(labels, bone, np.ones(3), engine.CountingConfig(), (0, 1, 2))
    actual = policy._measure_incidence(prepared, np.arange(5), (0, 1, 2))
    assert actual[(1, 2)][:2] == (1, 1)
    assert actual == direct_pair_evidence(labels, bone, np.ones(3), engine.CountingConfig())


@pytest.mark.parametrize("step,expected", [(np.nextafter(0.3, 0), 1), (0.3, 1), (np.nextafter(0.3, 1), 0)])
def test_anisotropic_edt_radius_boundary_has_no_added_tolerance(step, expected):
    spacing = np.array([step, 0.7, 1.1])
    labels = np.zeros((17, 9, 7), np.int32)
    labels[3, 4, 3], labels[3, 5, 3] = 1, 2
    labels[8, 1, 3], labels[8, 7, 3] = 1, 2
    bone = np.zeros(labels.shape, bool)
    bone[8, 4, 3] = True
    prepared = policy._prepare_incidence(labels, bone, spacing, engine.CountingConfig(), (0, 1, 2))
    actual = policy._measure_incidence(prepared, np.arange(3), (0, 1, 2))
    assert actual == direct_pair_evidence(labels, bone, spacing, engine.CountingConfig())
    assert actual[(1, 2)][:2] == (1, expected)


def fragment_fixture():
    coords = np.indices((41, 31, 21))
    inside = (coords[0] >= 3) & (coords[0] <= 37) & (coords[1] >= 3) & (coords[1] <= 27) & (coords[2] >= 3) & (coords[2] <= 17)
    opening = (coords[1] >= 14) & (coords[1] <= 16) & (coords[2] >= 9) & (coords[2] <= 11)
    bone = inside & (coords[0] == 20) & ~opening
    free = inside & ~bone
    labels = np.zeros(free.shape, np.int32)
    labels[free & (coords[0] < 20)] = 3
    labels[free & (coords[0] > 20)] = 4
    labels[free & (coords[0] >= 18) & (coords[0] <= 20) & opening] = 1
    labels[free & (coords[0] >= 21) & (coords[0] <= 22) & opening] = 2
    return labels, free, bone


def check_direct_edt_lifecycle(labels, bone, spacing, affine, trace):
    mapping = np.arange(int(labels.max()) + 1)
    bounds = ndi.find_objects(labels)
    original_anchors = {label: engine._physical_anchor(labels[region] == label, region, affine)
                        for label, region in enumerate(bounds, 1) if region is not None}
    constraints = set()
    for index, stage in enumerate(trace["dynamic_separator_stages"]):
        measured = direct_pair_evidence(mapping[labels], bone, spacing, engine.CountingConfig())
        constraints.update(pair for pair, (total, backed, _) in measured.items() if backed / total >= engine.CountingConfig().separator_support_fraction)
        protected = {tuple(sorted((int(mapping[a]), int(mapping[b])))) for a, b in constraints}
        assert all(a != b for a, b in protected)
        assert stage["interfaces"] == [{"basins": list(pair), "dominant_axis": axis,
            "interface_faces": total, "supported_faces": backed, "support_fraction": backed / total,
            "preserved": pair in protected} for pair, (total, backed, axis) in sorted(measured.items())]
        options = []
        for (a, b), (total, backed, axis) in measured.items():
            if (a, b) not in protected:
                anchors = tuple(sorted(min(anchor for label, anchor in original_anchors.items() if mapping[label] == root) for root in (a, b)))
                options.append((backed / total, -total * float(np.prod(np.delete(spacing, axis))), anchors, a, b))
        if index == len(trace["merge_decisions"]):
            assert not options
            break
        fraction, minus_area, anchors, a, b = min(options)
        decision = trace["merge_decisions"][index]
        assert decision["basins"] == [a, b]
        assert decision["support_fraction"] == fraction
        assert decision["dominant_interface_area_mm2"] == -minus_area
        assert decision["component_marker_anchor_pair"] == anchors
        mapping[mapping == b] = a
    assert mapping.tolist() == trace["basin_to_merged_label"]
    return {int(mapping[label]) for pair in constraints for label in pair}


@pytest.mark.parametrize("permutation", list(permutations((1, 2, 3, 4))))
def test_fragmented_true_wall_label_permutations_and_trace_none_match(permutation):
    labels, free, bone = fragment_fixture()
    labels = np.array([0, *permutation], np.int32)[labels]
    spacing = np.full(3, 0.5)
    affine = np.diag([*spacing, 1])
    trace = {}
    result, uncertain = policy.candidate(labels, free, bone, spacing, engine.CountingConfig(), affine=affine, markers=labels, trace=trace)
    plain, plain_uncertain = policy.candidate(labels, free, bone, spacing, engine.CountingConfig(), affine=affine, markers=labels)
    assert np.array_equal(result, plain) and uncertain == plain_uncertain
    assert np.array_equal(result > 0, free)
    assert len(np.unique(result[free])) == 2
    left = (labels == permutation[0]) | (labels == permutation[2])
    right = (labels == permutation[1]) | (labels == permutation[3])
    assert len(np.unique(result[left])) == len(np.unique(result[right])) == 1
    assert np.unique(result[left])[0] != np.unique(result[right])[0]
    assert uncertain == check_direct_edt_lifecycle(labels, bone, spacing, affine, trace)


def test_resource_limit_raises_instead_of_truncating_support(monkeypatch):
    labels, bone = witness_fixture(farther_owned=True)
    monkeypatch.setattr(policy, "MAX_GENERATED_INCIDENCES", 0)
    with pytest.raises(policy.ResourceLimitError, match="GENERATED_INCIDENCE_BUDGET_EXCEEDED"):
        policy._prepare_incidence(labels, bone, np.ones(3), engine.CountingConfig(), (0, 1, 2))


def test_accumulated_constraint_survives_later_support_weakening(monkeypatch):
    # A policy-unit graph, not a geometric or cell-count oracle.
    labels = np.arange(1, 5, dtype=np.int32).reshape(4, 1, 1)
    free = np.ones(labels.shape, bool)
    bone = np.zeros(labels.shape, bool)
    seen = []

    def measured(prepared, mapping, axes):
        seen.append(mapping.tolist())
        if mapping[2] == 2:
            return {(1, 3): (1, 1, 0), (1, 2): (2, 0, 0), (3, 4): (1, 0, 0)}
        if mapping[4] == 4:
            return {(1, 3): (1, 0, 0), (3, 4): (1, 0, 0)}
        return {(1, 3): (1, 0, 0)}

    monkeypatch.setattr(policy, "_prepare_incidence", lambda *args: {"preflight": {}})
    monkeypatch.setattr(policy, "_measure_incidence", measured)
    trace = {}
    result, uncertain = policy.candidate(labels, free, bone, np.ones(3), engine.CountingConfig(),
        affine=np.eye(4), markers=labels, trace=trace)
    assert seen == [[0, 1, 2, 3, 4], [0, 1, 1, 3, 4], [0, 1, 1, 3, 3]]
    assert result.ravel().tolist() == [1, 1, 3, 3]
    assert uncertain == {1, 3}
    assert [row["basins"] for row in trace["merge_decisions"]] == [[1, 2], [3, 4]]
    for stage in trace["dynamic_separator_stages"][1:]:
        protected = next(row for row in stage["interfaces"] if row["basins"] == [1, 3])
        assert protected["support_fraction"] == 0 and protected["preserved"]
