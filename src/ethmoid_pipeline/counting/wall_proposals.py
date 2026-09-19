"""Quarantined observed-bone wall proposals for synthetic research.

This module does not localize a sphenoid, assign clinical labels, or count cells.
Inputs must already be a verified axis-aligned anatomical view with +J anterior.
The caller owns source/affine provenance. Patch interpolation is model input only.
The decoder is bounded deterministic multi-start local optimization, NOT a global
optimum. It may abstain despite a feasible surface. Scores cannot create bone or
erase a second independently coherent sheet. Unsupported output stays NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy import ndimage as ndi


MAX_SOURCE_VOXELS = 128_000_000
MAX_COLUMN_GRID = 4_000_000
MAX_SEED_COLUMNS = 16_384
MAX_CANDIDATES = 65_536
MAX_PATCH_VOXELS = 64 ** 3
SOLVER_VERSION = "bounded_multistart_icm_observed_wall_v1"


class WallProposalError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fail(code, message):
    raise WallProposalError(code, message)


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WallCandidates:
    positions_ijk: np.ndarray
    candidate_hu: np.ndarray
    run_end_y: np.ndarray
    column_indices: np.ndarray
    columns_xz: np.ndarray
    seed_front_y: np.ndarray
    seed_column_mask: np.ndarray
    source_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    limit_y: int
    bone_hu: float
    cropped_seed: bool

    @property
    def scan_bounds_ijk(self):
        """Half-open voxel bounds; no world-coordinate or anatomical claim."""
        return tuple((0, size) for size in self.source_shape)


@dataclass(frozen=True)
class WallSurface:
    selected_candidate_indices: np.ndarray
    positions_ijk: np.ndarray
    supported_mask: np.ndarray
    y_surface_mm: np.ndarray
    diagnostics: dict


def _source_arrays(hu, seed):
    values, mask = np.asarray(hu), np.asarray(seed)
    if (values.ndim != 3 or any(size < 3 for size in values.shape)
            or mask.shape != values.shape or mask.dtype != np.bool_
            or values.dtype.kind not in "fiu"):
        _fail("WALL_INPUT_INVALID", "Finite 3D HU and a matching boolean seed are required.")
    if values.size > MAX_SOURCE_VOXELS or values.shape[0] * values.shape[2] > MAX_COLUMN_GRID:
        _fail("WALL_RESOURCE_LIMIT", "The source exceeds the fixed research input budget.")
    return values, mask


def extract_candidates(hu, seed, spacing_mm, limit_y, *, bone_hu=200.0,
                       max_candidates=MAX_CANDIDATES) -> WallCandidates:
    """Find the minimum-J voxel of EVERY observed bone run after a seed front.

    Coordinates are input voxel indices, sorted by (I,K), then increasing J.
    A run extending to the acquired J edge is retained as evidence, but cannot
    be selected by the decoder. Runs are not truncated at the search limit.
    """
    values, mask = _source_arrays(hu, seed)
    spacing = np.asarray(spacing_mm, dtype=float)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or type(limit_y) not in (int, np.int32, np.int64)
            or not 0 <= int(limit_y) < values.shape[1]
            or type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES
            or type(bone_hu) not in (int, float) or not math.isfinite(bone_hu)):
        _fail("WALL_INPUT_INVALID", "Spacing, search bounds, threshold or candidate budget is invalid.")
    if not np.isfinite(values).all():
        _fail("WALL_INPUT_INVALID", "The source contains nonfinite HU values.")
    columns_mask = mask.any(axis=1)
    columns = np.argwhere(columns_mask)
    if not len(columns):
        _fail("WALL_NO_SUPPORT", "No seeded columns are available.")
    if len(columns) > MAX_SEED_COLUMNS:
        _fail("WALL_RESOURCE_LIMIT", "The seed exceeds the fixed column budget.")
    if np.any(values[mask] >= bone_hu):
        _fail("WALL_INPUT_INVALID", "A lumen seed cannot contain thresholded bone.")
    cropped = any(np.any(np.take(mask, edge, axis=axis))
                  for axis in range(3) for edge in (0, mask.shape[axis] - 1))
    fronts, positions, ends, owners, observed = [], [], [], [], []
    for column, (xx, zz) in enumerate(columns):
        front = int(np.flatnonzero(mask[xx, :, zz])[-1])
        fronts.append(front)
        if front >= limit_y:
            cropped = True
            continue
        bone_line = values[xx, front + 1:, zz] >= bone_hu
        transitions = np.diff(np.pad(bone_line.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1) + front + 1
        stops = np.flatnonzero(transitions == -1) + front
        for start, end in zip(starts, stops):
            if start > limit_y:
                break
            if len(positions) >= max_candidates:
                _fail("WALL_RESOURCE_LIMIT", "The observed candidate budget was exceeded; no hits were discarded.")
            positions.append((int(xx), int(start), int(zz)))
            owners.append(column)
            ends.append(int(end))
            observed.append(float(values[xx, start, zz]))
    return WallCandidates(
        _readonly(np.asarray(positions).reshape(-1, 3), np.int32),
        _readonly(observed, np.float64), _readonly(ends, np.int32),
        _readonly(owners, np.int32), _readonly(columns, np.int32),
        _readonly(fronts, np.int32), _readonly(columns_mask, bool),
        tuple(values.shape), tuple(float(v) for v in spacing), int(limit_y),
        float(bone_hu), bool(cropped))


def candidate_patch(hu, seed, candidates: WallCandidates, index: int, *,
                    physical_size_mm=16.0, output_shape=(32, 32, 32)) -> np.ndarray:
    """Return [HU, seed, validity] float32 channels on an I/J/K model grid.

    Each dimension samples voxel centers across the requested physical extent,
    centered at the candidate. HU is linear, seed nearest-neighbor; outside HU
    is -1000 and validity zero. Channel zero is raw HU (not normalized). No source
    or finished labels are resampled in place. The caller must bind the source.
    """
    values, mask = _source_arrays(hu, seed)
    if values.shape != candidates.source_shape or type(index) is not int or not 0 <= index < len(candidates.positions_ijk):
        _fail("WALL_INPUT_INVALID", "The patch source or candidate index is invalid.")
    shape = tuple(output_shape)
    if len(shape) != 3 or any(type(n) is not int or n < 3 for n in shape):
        _fail("WALL_INPUT_INVALID", "Patch dimensions must be three positive integer sampling sizes.")
    if math.prod(shape) > MAX_PATCH_VOXELS:
        _fail("WALL_RESOURCE_LIMIT", "The model patch exceeds its voxel budget.")
    extent = np.asarray(physical_size_mm, dtype=float)
    if extent.ndim == 0:
        extent = np.repeat(extent, 3)
    if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0) or np.any(extent > 64):
        _fail("WALL_INPUT_INVALID", "Patch physical extent must be finite and at most 64 mm per axis.")
    center = candidates.positions_ijk[index]
    if float(values[tuple(center)]) != float(candidates.candidate_hu[index]):
        _fail("WALL_INPUT_CHANGED", "Observed candidate HU differs from extraction.")
    axes = [center[i] + ((np.arange(n) + 0.5) / n - 0.5) * extent[i] / candidates.spacing_mm[i]
            for i, n in enumerate(shape)]
    coordinates = np.array(np.meshgrid(*axes, indexing="ij"))
    valid = np.ones(shape, bool)
    for axis in range(3):
        valid &= (coordinates[axis] >= 0) & (coordinates[axis] <= values.shape[axis] - 1)
    sampled_hu = ndi.map_coordinates(values, coordinates, output=np.float32,
                                     order=1, mode="constant", cval=-1000.0, prefilter=False)
    sampled_seed = ndi.map_coordinates(mask, coordinates, output=np.float32,
                                       order=0, mode="constant", cval=0, prefilter=False)
    if not np.isfinite(sampled_hu).all():
        _fail("WALL_INPUT_INVALID", "The model patch contains nonfinite HU values.")
    return np.stack((sampled_hu, sampled_seed, valid)).astype(np.float32)


def _support_geometry(candidates, columns):
    points = candidates.columns_xz[columns]
    if len(points) < max(12, len(candidates.columns_xz) * 0.5):
        _fail("WALL_INSUFFICIENT_SUPPORT", "At least 12 and half of seeded columns require observed bone support.")
    if (np.ptp(points[:, 0]) < 2 or np.ptp(points[:, 1]) < 2
            or np.linalg.matrix_rank((points - points.mean(axis=0)) * np.asarray(candidates.spacing_mm)[[0, 2]]) < 2):
        _fail("WALL_SPATIAL_SUPPORT", "Wall support is sparse or collinear in the surface plane.")
    supported = np.zeros(candidates.seed_column_mask.shape, bool)
    supported[tuple(points.T)] = True
    if ndi.label(supported, ndi.generate_binary_structure(2, 1))[1] != 1:
        _fail("WALL_DISCONNECTED_SUPPORT", "Observed wall support is disconnected.")
    if np.any(ndi.binary_fill_holes(supported) & ~supported):
        _fail("WALL_UNSUPPORTED_HOLE", "The supported wall encloses an unsupported hole.")
    return supported


def _validate_candidate_metadata(candidates):
    """Reject incomplete evidence records before geometry or support arithmetic.

    This validates structure, not source authenticity. The caller still binds
    observations to the verified source from which extraction was performed.
    """
    if not isinstance(candidates, WallCandidates):
        _fail("WALL_INPUT_INVALID", "Observed candidate metadata is required.")
    shape, spacing = candidates.source_shape, np.asarray(candidates.spacing_mm)
    if (not isinstance(shape, tuple) or len(shape) != 3
            or any(type(n) is not int or n < 3 for n in shape)
            or spacing.shape != (3,) or spacing.dtype.kind not in "fiu"
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or type(candidates.limit_y) is not int or not 0 <= candidates.limit_y < shape[1]
            or type(candidates.bone_hu) not in (int, float) or not math.isfinite(candidates.bone_hu)
            or type(candidates.cropped_seed) is not bool):
        _fail("WALL_INPUT_INVALID", "Candidate source geometry is malformed.")
    if math.prod(shape) > MAX_SOURCE_VOXELS or shape[0] * shape[2] > MAX_COLUMN_GRID:
        _fail("WALL_RESOURCE_LIMIT", "Candidate source metadata exceeds the fixed research budget.")
    positions, columns = candidates.positions_ijk, candidates.columns_xz
    if (not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1] != 3
            or positions.dtype.kind not in "iu" or not isinstance(columns, np.ndarray)
            or columns.ndim != 2 or columns.shape[1] != 2 or columns.dtype.kind not in "iu"):
        _fail("WALL_INPUT_INVALID", "Candidate and seed-column coordinates must be integer arrays.")
    count, column_count = len(positions), len(columns)
    for values, expected, kinds in ((candidates.candidate_hu, count, "fiu"),
                                    (candidates.run_end_y, count, "iu"),
                                    (candidates.column_indices, count, "iu"),
                                    (candidates.seed_front_y, column_count, "iu")):
        if not isinstance(values, np.ndarray) or values.shape != (expected,) or values.dtype.kind not in kinds:
            _fail("WALL_INPUT_INVALID", "Candidate evidence arrays are incomplete or mismatched.")
    mask = candidates.seed_column_mask
    if (not isinstance(mask, np.ndarray) or mask.shape != (shape[0], shape[2]) or mask.dtype != np.bool_
            or not column_count or np.any(columns < 0) or np.any(columns >= np.asarray(shape)[[0, 2]])
            or not np.array_equal(np.argwhere(mask), columns)
            or np.any(candidates.seed_front_y < 0) or np.any(candidates.seed_front_y >= shape[1])):
        _fail("WALL_INPUT_INVALID", "Seed-column support differs from the source metadata.")
    if count > MAX_CANDIDATES or column_count > MAX_SEED_COLUMNS:
        _fail("WALL_RESOURCE_LIMIT", "Candidate metadata exceeds the fixed decoder budget.")


def decode_surface(candidates: WallCandidates, scores, *, minimum_score=0.5,
                   maximum_slope=2.0, smoothness=0.05, max_iterations=12,
                   max_candidates_per_column=8) -> WallSurface:
    """Approximate observed-candidate surface selection with independent vetoes.

    Scores must be probabilities in [0,1]; they rank candidates, not truth.
    Multi-start ICM uses each depth ordinal and the score maximum. In addition,
    every complete ordinal sheet is checked without considering its score, so a
    model cannot hide a plausible parallel wall by scoring it poorly. Any found
    coherent alternative differing on >= half of common columns is ambiguous.
    This conservative check is NOT an exhaustive proof that no alternative exists.
    Depth observations occupy intervals of +/- half the native J spacing; a
    graph constraint checks whether those intervals admit consistent slopes.
    Interval feasibility does not supply interpolated or extra support points.
    """
    _validate_candidate_metadata(candidates)
    probability = np.asarray(scores, dtype=float)
    count = len(candidates.positions_ijk)
    if (probability.shape != (count,) or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or type(max_iterations) is not int or not 1 <= max_iterations <= 32
            or type(max_candidates_per_column) is not int or not 1 <= max_candidates_per_column <= 8
            or type(minimum_score) not in (int, float) or not 0 <= minimum_score <= 1
            or type(maximum_slope) not in (int, float) or not 0 < maximum_slope <= 4
            or type(smoothness) not in (int, float) or not 0 <= smoothness <= 1):
        _fail("WALL_INPUT_INVALID", "Candidate scores or decoder resource parameters are invalid.")
    if candidates.cropped_seed:
        _fail("WALL_CROPPED", "The seed touches acquired or search bounds.")
    if not count:
        _fail("WALL_NO_SUPPORT", "No observed bone faces exist; model scores cannot supply them.")
    if np.any(candidates.candidate_hu < candidates.bone_hu) or not np.isfinite(candidates.candidate_hu).all():
        _fail("WALL_INPUT_INVALID", "A wall candidate lacks observed thresholded bone.")
    groups = [[] for _ in candidates.columns_xz]
    cropped_count = 0
    for index, (position, end, owner) in enumerate(zip(candidates.positions_ijk, candidates.run_end_y, candidates.column_indices)):
        xx, yy, zz = position
        if (not 0 <= owner < len(groups) or not 0 <= xx < candidates.source_shape[0]
                or not 0 <= zz < candidates.source_shape[2]
                or tuple(position[[0, 2]]) != tuple(candidates.columns_xz[owner])
                or not candidates.seed_front_y[owner] < yy <= candidates.limit_y or not yy <= end < candidates.source_shape[1]):
            _fail("WALL_INPUT_INVALID", "Candidate geometry differs from its seeded source column.")
        if xx in (0, candidates.source_shape[0] - 1) or zz in (0, candidates.source_shape[2] - 1) or end == candidates.source_shape[1] - 1:
            cropped_count += 1
            continue
        groups[owner].append(index)
    active = np.array([i for i, group in enumerate(groups) if group], dtype=np.int32)
    if not len(active) and cropped_count:
        _fail("WALL_CROPPED", "All observed sheets terminate at the acquired source edge.")
    supported = _support_geometry(candidates, active)
    groups = [np.asarray(sorted(groups[i], key=lambda n: candidates.positions_ijk[n, 1]), dtype=np.int32) for i in active]
    if max(map(len, groups)) > max_candidates_per_column:
        _fail("WALL_RESOURCE_LIMIT", "A column has too many candidate sheets for bounded alternative checking.")
    if any(np.any(candidates.run_end_y[group[:-1]] + 1 >= candidates.positions_ijk[group[1:], 1])
           for group in groups):
        _fail("WALL_INPUT_INVALID", "Candidate runs overlap or duplicate the same observed bone face.")
    points = candidates.columns_xz[active]
    lookup = {tuple(point): i for i, point in enumerate(points)}
    neighbors = [[] for _ in active]
    edges = []
    for i, (xx, zz) in enumerate(points):
        for delta, step in (((1, 0), candidates.spacing_mm[0]), ((0, 1), candidates.spacing_mm[2])):
            j = lookup.get((xx + delta[0], zz + delta[1]))
            if j is not None:
                neighbors[i].append((j, step)); neighbors[j].append((i, step))
                edges.append((i, j, step))
    depth = candidates.positions_ijk[:, 1] * candidates.spacing_mm[1]
    unary = -np.log(np.maximum(probability, 1e-8))
    # Coordinate updates optimize the same mean-unary/mean-edge objective used
    # to rank the found solutions. Multiplying the objective by N is constant.
    edge_weight = smoothness * len(groups) / len(edges) if edges else 0.0

    def required_slope(a, b, step):
        # A one-voxel stair is compatible with a shallow subvoxel surface. Model
        # this observation quantization explicitly instead of penalizing the
        # distance between centers as if it were exact wall displacement.
        return np.maximum(0.0, np.abs(depth[a] - depth[b]) - candidates.spacing_mm[1]) / step

    def coherent(selected):
        if not all(required_slope(selected[a], selected[b], step) <= maximum_slope + 1e-7
                   for a, b, step in edges):
            return False
        # Pairwise interval overlap alone is insufficient: a long, steep ramp
        # could repeatedly spend the same half-voxel uncertainty. Propagate the
        # tightest graph lower bounds to check simultaneous interval feasibility.
        # These latent values are never returned as observed wall support.
        lower = depth[selected].copy() - candidates.spacing_mm[1] * 0.5
        upper = depth[selected] + candidates.spacing_mm[1] * 0.5
        pending = [(-float(value), i) for i, value in enumerate(lower)]
        heapq.heapify(pending)
        while pending:
            negative, i = heapq.heappop(pending)
            value = -negative
            if value < lower[i] - 1e-7:
                continue
            for j, step in neighbors[i]:
                required = value - maximum_slope * step
                if required > upper[j] + 1e-7:
                    return False
                if required > lower[j] + 1e-7:
                    lower[j] = required
                    heapq.heappush(pending, (-float(required), j))
        return True

    def energy(selected):
        penalty = [required_slope(selected[a], selected[b], step) ** 2 for a, b, step in edges]
        return float(unary[selected].mean() + smoothness * (np.mean(penalty) if penalty else 0))

    solutions = {}
    def retain(selected):
        if coherent(selected):
            solutions[tuple(int(v) for v in selected)] = selected.copy()

    seeds = [np.array([g[min(rank, len(g) - 1)] for g in groups], dtype=np.int32)
             for rank in range(max(map(len, groups)))]
    seeds.append(np.array([g[np.argmax(probability[g])] for g in groups], dtype=np.int32))
    for initial in seeds:
        retain(initial)  # A low model score cannot suppress a coherent alternative.
        selected = initial.copy()
        for _ in range(max_iterations):
            changed = False
            for i, group in enumerate(groups):
                costs = unary[group].copy()
                for j, step in neighbors[i]:
                    slope = required_slope(group, selected[j], step)
                    costs += edge_weight * slope ** 2
                    costs[slope > maximum_slope + 1e-7] = np.inf
                best = int(np.argmin(costs))
                if np.isfinite(costs[best]) and selected[i] != group[best]:
                    selected[i] = group[best]
                    changed = True
            if not changed:
                break
        retain(selected)
    if not solutions:
        _fail("WALL_INCOHERENT", "No coherent observed sheet was found by bounded local search.")
    ranked = sorted(solutions.values(), key=lambda selected: (energy(selected), tuple(selected)))
    chosen = ranked[0]
    for alternative in ranked[1:]:
        separated = np.abs(depth[chosen] - depth[alternative]) > candidates.spacing_mm[1] * 0.5
        if np.count_nonzero(separated) >= max(12, len(chosen) * 0.5):
            _fail("WALL_AMBIGUOUS", "Multiple coherent observed sheets remain plausible regardless of model ranking.")
    if float(probability[chosen].mean()) < minimum_score:
        _fail("WALL_SCORE_UNSUPPORTED", "Model support for the observed sheet is below the research threshold.")
    output = np.full(supported.shape, np.nan, dtype=np.float64)
    output[tuple(points.T)] = depth[chosen]
    return WallSurface(_readonly(chosen, np.int32), _readonly(candidates.positions_ijk[chosen], np.int32),
                       _readonly(supported, bool), _readonly(output), {
                           "solver": SOLVER_VERSION, "global_optimum_claim": False,
                           "complete_alternative_search_claim": False,
                           "seed_columns": len(candidates.columns_xz), "observed_support_columns": len(chosen),
                           "required_support_columns": max(12, len(candidates.columns_xz) * 0.5),
                           "cropped_candidates_excluded": cropped_count, "starts": len(seeds),
                           "coherent_solutions_found": len(solutions), "mean_candidate_score": float(probability[chosen].mean()),
                           "energy": energy(chosen), "interpolated_support_points": 0,
                           "depth_observation_interval_mm": candidates.spacing_mm[1],
                           "quantization_constraint": "simultaneous_graph_intervals_not_independent_edge_slack",
                           "anatomical_accuracy_validated": False})
