import numpy as np
import pytest

from ethmoid_pipeline.counting.partition_audit import audit_partition, exact_partition


def test_core_ledger_separates_missing_extra_split_and_mixed_basins():
    labels = np.array([1, 1, 2, 3, 0, 4])
    cores = np.array([1, 2, 3, 3, 4, 0])
    result = audit_partition(labels, cores, np.array([1, 1, 1, 1, 1, 0], bool))
    assert result['mixed_core_basins'] == [1]
    assert result['split_cores'] == [3]
    assert result['missing_core_voxels'] == 1
    assert result['missed_foreground_voxels'] == 1
    assert result['extra_foreground_voxels'] == 1
    assert result['merge_only_core_feasible'] is False


def test_unknown_voxels_do_not_invent_core_relations():
    result = audit_partition(np.array([7, 7, 90, 90]), np.array([1, 0, 2, 0]), np.ones(4, bool))
    assert result['basin_cores'] == {7: [1], 90: [2]}
    assert result['merge_only_core_feasible'] is True
    assert result['missed_foreground_voxels'] == 0


def test_missing_noncore_foreground_does_not_disappear_from_audit():
    result = audit_partition(np.array([1, 0]), np.array([1, 0]), np.ones(2, bool))
    assert result['merge_only_core_feasible'] is True
    assert result['missed_foreground_voxels'] == 1


@pytest.mark.parametrize('labels,cores,foreground', [
    ([1, -1], [1, 0], [True, True]),
    ([1.0, 2.0], [1, 2], [True, True]),
    ([1, 2], [1], [True, True]),
    ([1, 2], [1, 2], [True, False]),
])
def test_invalid_oracle_inputs_are_rejected(labels, cores, foreground):
    with pytest.raises(ValueError):
        audit_partition(np.asarray(labels), np.asarray(cores), np.asarray(foreground))


def test_exact_objective_distinguishes_costs_from_hard_relations():
    # Positive costs favor joining; negative costs favor a cut.
    result = exact_partition(3, [(0, 1, 2), (1, 2, 2), (0, 2, -5)])
    assert result['objective'] == -3
    assert result['optimal_partitions'] == 2
    constrained = exact_partition(3, [(0, 1, 2), (1, 2, 2), (0, 2, -5)], same=[(0, 1)], different=[(1, 2)])
    assert constrained['labels'] == [0, 0, 1]
    assert constrained['objective'] == -3


def test_unknown_edges_remain_unconstrained_and_all_optima_counted():
    assert exact_partition(3, [])['optimal_partitions'] == 5
    assert exact_partition(8, [])['optimal_partitions'] == 4140


def test_contradictory_transitive_relations_are_infeasible():
    result = exact_partition(3, [], same=[(0, 1), (1, 2)], different=[(0, 2)])
    assert result == {'feasible': False, 'labels': None, 'objective': None, 'optimal_partitions': 0}


def test_false_hard_separator_cannot_be_repaired_by_exact_optimization():
    costs = [(0, 1, 5), (0, 2, -3), (1, 2, -3)]
    observed = exact_partition(3, costs)
    assert observed['labels'] == [0, 0, 1]
    assert observed['objective'] == -6
    assert observed['optimal_partitions'] == 1
    constrained = exact_partition(3, costs, different=[(0, 1)])
    assert constrained['labels'] == [0, 1, 2]
    assert constrained['objective'] == -1
    assert not exact_partition(3, costs, same=[(0, 1)], different=[(0, 1)])['feasible']


@pytest.mark.parametrize('n,costs', [(9, []), (0, []), (True, []), (2, [(0, 2, 1)]),
    (2, [(0, 1, np.nan)]), (2, [(0, 1, np.inf)]), (2, [(0, 1, 10**1000)]),
    (2, [(0, 1, 1), (1, 0, 2)])])
def test_exact_reference_is_bounded_and_validates_costs(n, costs):
    with pytest.raises(ValueError):
        exact_partition(n, costs)
