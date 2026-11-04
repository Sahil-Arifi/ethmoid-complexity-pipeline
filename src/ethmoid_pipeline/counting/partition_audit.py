"""Bounded synthetic diagnostics; never used by automatic counting.

Core IDs declare incompatible reference groups. Zero means unknown, not
background. Graph optima concern only the supplied objective and constraints;
they do not establish anatomical correctness or spatial connectivity.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def audit_partition(labels: np.ndarray, cores: np.ndarray, foreground: np.ndarray) -> dict:
    """Report partial-core relations and foreground errors independently.

    Merge-only core feasibility is a necessary condition for a spatial repair,
    not proof that adjacency-respecting merges or full reference recovery exist.
    """
    labels, cores, foreground = map(np.asarray, (labels, cores, foreground))
    if labels.shape != cores.shape or labels.shape != foreground.shape:
        raise ValueError("labels, cores and foreground must have equal shapes")
    for name, array in (("labels", labels), ("cores", cores)):
        if array.dtype.kind not in "iu" or np.any(array < 0):
            raise ValueError(f"{name} must contain nonnegative integers")
    if foreground.dtype.kind != "b":
        raise ValueError("foreground must be boolean")
    if np.any((cores > 0) & ~foreground):
        raise ValueError("reference cores must lie inside foreground")

    basin_cores = {int(label): [] for label in np.unique(labels[labels > 0])}
    core_basins = {int(core): [] for core in np.unique(cores[cores > 0])}
    known = (labels > 0) & (cores > 0)
    # Pair IDs through Python integers to preserve mixed uint64/int64 values.
    pairs = {(int(label), int(core)) for label, core in zip(labels[known], cores[known])}
    for label, core in sorted(pairs):
        basin_cores[label].append(core)
        core_basins[core].append(label)
    mixed = [label for label, ids in basin_cores.items() if len(ids) > 1]
    missing = int(np.count_nonzero((cores > 0) & (labels == 0)))
    return {
        "basin_cores": basin_cores,
        "core_basins": core_basins,
        "mixed_core_basins": mixed,
        "split_cores": [core for core, ids in core_basins.items() if len(ids) > 1],
        "missing_core_voxels": missing,
        "missed_foreground_voxels": int(np.count_nonzero(foreground & (labels == 0))),
        "extra_foreground_voxels": int(np.count_nonzero(~foreground & (labels > 0))),
        "merge_only_core_feasible": not mixed and missing == 0,
    }


def exact_partition(n: int, costs, *, same=(), different=()) -> dict:
    """Enumerate partitions of 1..8 nodes, minimizing the sum of cut costs.

    Positive costs favor joining; negative costs favor cutting. Missing edges
    impose no preference. All ties under math.fsum are counted; the returned
    labels are the first canonical optimum, not a claim of unique evidence.
    Hard relations are checked independently of costs, including transitivity.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= 8:
        raise ValueError("exact reference requires 1..8 nodes")

    def pair(i, j):
        if any(isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k < n for k in (i, j)):
            raise ValueError("relation endpoints must be valid integer node IDs")
        if i == j:
            raise ValueError("relation endpoints must be distinct")
        return min(i, j), max(i, j)

    edges = []
    seen = set()
    for i, j, weight in costs:
        edge = pair(i, j)
        if edge in seen:
            raise ValueError("duplicate undirected edge")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("costs must be finite real numbers")
        try:
            weight = float(weight)
        except OverflowError as exc:
            raise ValueError("cost magnitude exceeds finite arithmetic") from exc
        if not math.isfinite(weight):
            raise ValueError("costs must be finite real numbers")
        seen.add(edge)
        edges.append((*edge, weight))
    try:
        math.fsum(abs(weight) for _, _, weight in edges)
    except OverflowError as exc:
        raise ValueError("total cost magnitude exceeds finite arithmetic") from exc
    must_join = [pair(i, j) for i, j in same]
    must_cut = [pair(i, j) for i, j in different]
    result = {"feasible": False, "labels": None, "objective": None, "optimal_partitions": 0}

    def visit(labels, highest):
        if len(labels) < n:
            for group in range(highest + 2):
                visit([*labels, group], max(highest, group))
            return
        if any(labels[i] != labels[j] for i, j in must_join):
            return
        if any(labels[i] == labels[j] for i, j in must_cut):
            return
        objective = math.fsum(weight for i, j, weight in edges if labels[i] != labels[j])
        if not result["feasible"] or objective < result["objective"]:
            result.update(feasible=True, labels=labels, objective=objective, optimal_partitions=1)
        elif objective == result["objective"]:
            result["optimal_partitions"] += 1

    visit([0], 0)
    return result
