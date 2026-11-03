"""Synthetic checks for the research-only full-field connectivity selector."""
import numpy as np
import pytest

from ethmoid_pipeline.counting import instance_connectivity as connectivity
from ethmoid_pipeline.counting import instance_learning as learning


def test_all_terminal_positive_and_any_endpoint_negative():
    values = np.zeros((7, 1, 1, 5), np.float64)
    values[0] = 1
    values[3, 0, 0, :4] = [.9, .8, .3, .7]
    values[6, 0, 0, 0] = .2
    ids = np.array([[[1, 2, 2, -1, 1]]], np.int32)
    records = connectivity.select_constraints(values, ids, [(2, 1)])
    assert [(r['kind'], r['ids']) for r in records] == [
        ('same', [1]), ('same', [2]), ('different', [1, 2])]
    assert [r['capacity'] for r in records] == pytest.approx([.3, .8, .9])
    assert records[0]['cache_flat_index'] == np.ravel_multi_index((3, 0, 0, 2), values.shape)


def test_every_known_terminal_is_protected_not_only_farthest_pair():
    values = np.zeros((7, 1, 1, 5), np.float64)
    values[0] = 1
    values[3, 0, 0, :4] = [.9, .2, .8, .8]
    values[6, 0, 0, 0] = .95
    ids = np.array([[[1, -1, 1, -1, 1]]], np.int32)
    record, = connectivity.select_constraints(values, ids, [])
    assert record['capacity'] == pytest.approx(.8)


def test_bypass_refresh_changes_selected_capacity():
    values = np.zeros((7, 2, 1, 2), np.float64)
    values[0] = 1
    values[3, 0, 0, 0] = .8
    values[1, 0, 0, :] = .9
    values[3, 1, 0, 0] = .7
    ids = np.full((2, 1, 2), -1, np.int32)
    ids[0, 0, :] = [1, 2]
    first, = connectivity.select_constraints(values, ids, [(1, 2)])
    assert first['capacity'] == .8
    values.reshape(-1)[first['cache_flat_index']] = .6
    second, = connectivity.select_constraints(values, ids, [(1, 2)])
    assert second['capacity'] == .7
    assert second['cache_flat_index'] != first['cache_flat_index']


@pytest.mark.parametrize('barrier', ['source', 'target', 'affinity', 'all_tied'])
def test_capacity_selects_original_foreground_or_affinity_scalar(barrier):
    values = np.ones((7, 2, 1, 1), np.float64)
    ids = np.array([[[1]], [[2]]], np.int32)
    positions = [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)]
    chosen = {'source': 0, 'target': 1, 'affinity': 2, 'all_tied': 0}[barrier]
    for position in positions if barrier == 'all_tied' else [positions[chosen]]:
        values[position] = .3
    record, = connectivity.select_constraints(values, ids, [(1, 2)])
    assert record['capacity'] == .3
    assert record['cache_flat_index'] == np.ravel_multi_index(positions[chosen], values.shape)


def test_threshold_components_exactly_match_existing_six_head_decoder():
    rng = np.random.default_rng(813)
    values = rng.choice([.2, np.nextafter(.5, 0), .5, .8], size=(7, 2, 3, 5))
    ids = np.arange(1, 31, dtype=np.int32).reshape(2, 3, 5)
    pairs = [(a, b) for a in range(1, 31) for b in range(a + 1, 31)]
    decoded = learning.decode_probabilities(values).reshape(-1)
    for record in connectivity.select_constraints(values, ids, pairs):
        a, b = (value - 1 for value in record['ids'])
        expected = decoded[a] > 0 and decoded[b] > 0 and decoded[a] == decoded[b]
        assert (record['capacity'] >= .5) == expected


def test_long_edges_require_only_endpoints_not_intervening_foreground():
    values = np.zeros((7, 1, 1, 5), np.float64)
    values[0, 0, 0, [0, 4]] = .9
    values[6, 0, 0, 0] = .8
    ids = np.array([[[1, -1, -1, -1, 2]]], np.int32)
    record, = connectivity.select_constraints(values, ids, [(1, 2)])
    assert record['capacity'] == .8


def test_tied_bypass_is_deterministic_but_one_cut_need_not_disconnect():
    values = np.ones((7, 2, 1, 2), np.float64)
    values[1:] = .8
    ids = np.array([[[1, 2]], [[-1, -1]]], np.int32)
    first = connectivity.select_constraints(values, ids, [(1, 2)])
    assert first == connectivity.select_constraints(values.copy(), ids.copy(), [(2, 1)])
    changed = values.copy()
    changed.reshape(-1)[first[0]['cache_flat_index']] -= .01
    assert connectivity.select_constraints(changed, ids, [(1, 2)])[0]['capacity'] == .8


@pytest.mark.parametrize('kind', ['same', 'different'])
@pytest.mark.parametrize('bottleneck', ['affinity', 'foreground'])
def test_unique_selected_scalar_matches_finite_difference_and_autograd(kind, bottleneck):
    torch = pytest.importorskip('torch')
    values = np.ones((7, 1, 1, 3), np.float64)
    values[3, 0, 0, :2] = [.7, .8]
    if bottleneck == 'foreground':
        values[0, 0, 0, 2] = .6
    ids = np.array([[[1, -1, 1 if kind == 'same' else 2]]], np.int32)
    pairs = [] if kind == 'same' else [(1, 2)]
    record, = connectivity.select_constraints(values, ids, pairs)
    leaf = torch.tensor(values, requires_grad=True)
    scalar = leaf.reshape(-1)[record['cache_flat_index']]
    loss = (scalar - (.9 if kind == 'same' else .5)).square()
    loss.backward()
    derivatives = []
    for sign in [-1, 1]:
        perturbed = values.copy()
        perturbed.reshape(-1)[record['cache_flat_index']] += sign * 1e-6
        updated, = connectivity.select_constraints(perturbed, ids, pairs)
        derivatives.append((updated['capacity'] - (.9 if kind == 'same' else .5)) ** 2)
    assert (derivatives[1] - derivatives[0]) / 2e-6 == pytest.approx(
        leaf.grad.reshape(-1)[record['cache_flat_index']].item(), abs=1e-9)
    assert torch.count_nonzero(leaf.grad) == 1


@pytest.mark.parametrize('seed', [3, 17, 81])
def test_continuous_capacities_match_independent_all_pairs_floyd_oracle(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(size=(7, 5, 5, 5))
    ids = np.full((5, 5, 5), -1, np.int32)
    selected = rng.choice(ids.size, 10, replace=False)
    ids.reshape(-1)[selected] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 10]
    pairs = [(1, 2), (2, 3), (1, 10)]
    matrix = np.zeros((ids.size, ids.size))
    np.fill_diagonal(matrix, 1)
    for point in np.ndindex(ids.shape):
        a = np.ravel_multi_index(point, ids.shape)
        for head, offset in enumerate(learning.OFFSETS, 1):
            target = tuple(x + step for x, step in zip(point, offset))
            if all(x < size for x, size in zip(target, ids.shape)):
                b = np.ravel_multi_index(target, ids.shape)
                matrix[a, b] = matrix[b, a] = min(values[head][point],
                    values[0][point], values[0][target])
    for node in range(ids.size):
        matrix = np.maximum(matrix, np.minimum(matrix[:, node, None], matrix[None, node, :]))
    for record in connectivity.select_constraints(values, ids, pairs):
        a = np.flatnonzero(ids.reshape(-1) == record['ids'][0])
        b = a if record['kind'] == 'same' else np.flatnonzero(ids.reshape(-1) == record['ids'][1])
        expected = matrix[np.ix_(a, b)].min() if record['kind'] == 'same' else matrix[np.ix_(a, b)].max()
        assert record['capacity'] == expected
        assert values.reshape(-1)[record['cache_flat_index']] == expected


def test_inputs_unchanged_noncontiguous_and_review_id_not_invented():
    values = np.full((7, 2, 3, 5), .8, np.float32).transpose(0, 3, 2, 1)
    ids = np.full(values.shape[1:], -1, np.int32)
    ids[0, 0, 0], ids[-1, -1, -1] = 1, 8
    before_values, before_ids = values.copy(), ids.copy()
    records = connectivity.select_constraints(values, ids, [(1, 8)])
    assert records[0]['ids'] == [1, 8]
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(ids, before_ids)


@pytest.mark.parametrize('pairs', [[(1, 1)], [(1, 3)], [(0, 2)], [(1, 2, 3)],
                                  [(1, 2), (2, 1)], [(True, 2)], [(1., 2)]])
def test_invalid_or_duplicate_pairs_rejected(pairs):
    with pytest.raises(ValueError):
        connectivity.select_constraints(np.ones((7, 2, 1, 1)), np.array([[[1]], [[2]]]), pairs)


@pytest.mark.parametrize('values,ids', [
    (np.ones((6, 2, 1, 1)), np.ones((2, 1, 1), np.int32)),
    (np.full((7, 2, 1, 1), np.nan), np.ones((2, 1, 1), np.int32)),
    (np.full((7, 2, 1, 1), 1.1), np.ones((2, 1, 1), np.int32)),
    (np.ones((7, 2, 1, 1)), np.ones((2, 1, 1))),
    (np.ones((7, 2, 1, 1)), np.full((2, 1, 1), -2)),
    (np.ones((7, 2, 1, 1)), np.ones((1, 2, 1), np.int32)),
    (np.ones((7, 0, 1, 1)), np.ones((0, 1, 1), np.int32)),
])
def test_invalid_fields_rejected(values, ids):
    with pytest.raises(ValueError):
        connectivity.select_constraints(values, ids, [])


def test_empty_constraints_and_singleton_core_return_no_records():
    assert connectivity.select_constraints(np.ones((7, 1, 1, 1)), np.ones((1, 1, 1), np.int32), []) == []
