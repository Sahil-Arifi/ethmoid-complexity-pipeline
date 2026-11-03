"""Research-only maximin constraints; never used by automatic inference.

Unknown voxels participate in paths without becoming anatomical labels. A
selected scalar supplies one deterministic branch derivative; ties and bypasses
require graph refresh, and do not imply a unique smooth derivative.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .instance_learning import OFFSETS, edge_slices


def _validate(probabilities, instance_ids, distinct_pairs):
    values, ids = np.asarray(probabilities), np.asarray(instance_ids)
    if (values.ndim != 4 or values.shape[0] != 7 or 0 in values.shape[1:]
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError('Probabilities must be finite [0,1] values with shape 7DHW')
    if (ids.shape != values.shape[1:] or ids.dtype.kind not in 'iu'
            or np.any(ids < -1)):
        raise ValueError('Instance IDs must be matching 3D integers: -1, 0 or positive')
    groups, totals = np.unique(ids[ids > 0], return_counts=True)
    totals = dict(zip(map(int, groups), map(int, totals)))
    pairs = set()
    try:
        for supplied in distinct_pairs:
            pair = tuple(supplied)
            if (len(pair) != 2 or any(isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, np.integer)) or v <= 0 for v in pair)
                    or pair[0] == pair[1]):
                raise ValueError('Each distinct pair needs two different positive integer IDs')
            pair = tuple(sorted(map(int, pair)))
            if pair in pairs or any(value not in totals for value in pair):
                raise ValueError('Distinct pairs must be unique and reference present groups')
            pairs.add(pair)
    except TypeError as error:
        raise ValueError('Distinct pairs must be an iterable of two-ID observations') from error
    return values, ids, totals, sorted(pairs)


def _terminal_tree(values, terminals):
    """Build the full-grid ranked MST, then retain its terminal-connecting subtree."""
    shape = values.shape[1:]
    count = int(np.prod(shape))
    if count > np.iinfo(np.int32).max:
        raise ValueError('Graph node indexing exceeds int32')
    nodes = np.arange(count, dtype=np.int32).reshape(shape)
    starts, ends, capacities, affinity_indices = [], [], [], []
    for head, offset in enumerate(OFFSETS, 1):
        source, target = edge_slices(shape, offset)
        a, b = nodes[source].ravel(), nodes[target].ravel()
        starts.append(a)
        ends.append(b)
        capacities.append(np.minimum(np.minimum(values[0][source], values[0][target]),
                                     values[head][source]).ravel())
        affinity_indices.append(head * count + a.astype(np.int64))
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    capacities, affinity_indices = np.concatenate(capacities), np.concatenate(affinity_indices)
    order = np.lexsort((affinity_indices, -capacities.astype(np.float64)))
    # Positive unique ranks preserve zero-capacity edges and the exact tie policy.
    ranks = np.empty(order.size, np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    graph = coo_matrix((ranks, (starts, ends)), shape=(count, count)).tocsr()
    tree = minimum_spanning_tree(graph, overwrite=True).tocoo()
    forest = coo_matrix((np.ones(2 * tree.nnz, np.uint8),
        (np.r_[tree.row, tree.col], np.r_[tree.col, tree.row])), shape=(count, count)).tocsr()
    _, predecessors = breadth_first_order(forest, int(terminals[0]), directed=False,
                                         return_predecessors=True)
    # Pruning happens only AFTER the complete graph determines the maximum tree.
    visited = {int(terminals[0])}
    edges = []
    for terminal in terminals[1:]:
        child = int(terminal)
        while child not in visited:
            parent = int(predecessors[child])
            if parent < 0:
                raise RuntimeError('Full grid unexpectedly disconnected')
            edges.append((min(child, parent), max(child, parent)))
            visited.add(child)
            child = parent
    a, b = np.asarray(edges, np.int64).T
    xyz_a, xyz_b = np.array(np.unravel_index(a, shape)), np.array(np.unravel_index(b, shape))
    difference = (xyz_b - xyz_a).T
    heads = np.empty(a.size, np.int64)
    for head, offset in enumerate(OFFSETS, 1):
        heads[np.all(difference == offset, axis=1)] = head
    affinity_indices = heads * count + a
    flat = values.reshape(-1)
    capacities = np.minimum(np.minimum(flat[a], flat[b]), flat[affinity_indices])
    # Spatial FG indices precede all affinity indices; a < b resolves FG ties.
    selected_scalars = np.where(flat[a] == capacities, a,
                               np.where(flat[b] == capacities, b, affinity_indices))
    order = np.lexsort((affinity_indices, -capacities.astype(np.float64)))
    return a[order], b[order], capacities[order], selected_scalars[order], visited


def select_constraints(probabilities: np.ndarray, instance_ids: np.ndarray,
                       distinct_pairs=()) -> list[dict]:
    """Select all-core minimum and different-group maximum maximin constraints.

    Every grid voxel and all six original offsets participate. Edge capacity is
    min(affinity, foreground at either endpoint), so capacity >= .5 exactly
    reproduces the decoder's foreground/affinity graph. For each positive ID,
    the completion edge protects ALL supplied terminals, not one sampled pair.
    For each distinct pair, the first cross-group edge detects ANY merged pair.
    Records gather the original 7DHW scalar; inputs are never modified.
    """
    values, ids, totals, pairs = _validate(probabilities, instance_ids, distinct_pairs)
    same_pending = {group for group, total in totals.items() if total > 1}
    different_pending = set(pairs)
    if not same_pending and not different_pending:
        return []
    terminals = np.flatnonzero(ids > 0)
    a, b, capacities, scalar_indices, visited = _terminal_tree(values, terminals)
    parent = {node: node for node in visited}
    size = {node: 1 for node in visited}
    flat_ids = ids.reshape(-1)
    marks = {int(node): {int(flat_ids[node]): 1} for node in terminals}
    same, different = {}, {}

    def find(node):
        node = int(node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right, capacity, scalar_index in zip(a, b, capacities, scalar_indices):
        left, right = find(left), find(right)
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
        first, second = marks.pop(left, {}), marks.pop(right, {})
        for group, total in second.items():
            first[group] = first.get(group, 0) + total
        if first:
            marks[left] = first
        record = {'capacity': float(capacity), 'cache_flat_index': int(scalar_index)}
        for group in same_pending.intersection(first):
            if first[group] == totals[group]:
                same[group] = {'kind': 'same', 'ids': [group], **record}
                same_pending.remove(group)
        for pair in tuple(different_pending):
            if pair[0] in first and pair[1] in first:
                different[pair] = {'kind': 'different', 'ids': list(pair), **record}
                different_pending.remove(pair)
        if not same_pending and not different_pending:
            break
    if same_pending or different_pending:
        raise RuntimeError('Terminal tree did not resolve every supplied constraint')
    return [same[group] for group in sorted(same)] + [different[pair] for pair in pairs]
