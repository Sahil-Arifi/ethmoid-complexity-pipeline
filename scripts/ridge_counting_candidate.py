"""Synthetic-only continuous-HU proposals; not imported by production callers."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_minima
from skimage.segmentation import watershed

from ethmoid_pipeline.counting.engine import (
    CountingConfig, CountingError, _merge_unsupported_separators, geometry,
)

METHOD = 'continuous-ridge-withholding-v1'
REASON = 'experimental_continuous_ridge_requires_review'
PROMINENCE_HU = 100.0
MAX_BASINS = 128
FACE = ndi.generate_binary_structure(3, 1)


def _pairs(labels):
    pairs = set()
    for axis in range(3):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(low)], labels[tuple(high)]
        touching = (a > 0) & (b > 0) & (a != b)
        pairs.update(map(tuple, np.unique(np.sort(np.column_stack((a[touching], b[touching])), axis=1), axis=0)))
    return sorted(pairs)


def _distance(mask, spacing):
    return ndi.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]


def propose_partitions(hu, roi, affine):
    values, region = np.asarray(hu), np.asarray(roi)
    if (values.ndim != 3 or values.shape != region.shape or region.dtype != bool
            or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise CountingError('INVALID_VOLUME', 'Finite real 3D HU and a matching boolean ROI are required.')
    spacing = geometry(affine)
    if not region.any():
        raise CountingError('EMPTY_REGION', 'A nonempty synthetic region is required.')
    free = region & (values < 200)
    if not free.any():
        raise CountingError('NO_COMPARTMENTS', 'No non-bone space is available.')
    distance = _distance(free, spacing)
    sigma = 0.5 * float(spacing.min())
    smoothed = ndi.gaussian_filter(values.astype(np.float64), sigma=sigma / spacing, mode='reflect')
    peak = float(smoothed.max()) + PROMINENCE_HU
    if not np.isfinite(peak):
        raise CountingError('INVALID_VOLUME', 'Intensity range exceeds finite arithmetic.')
    landscape = np.where(free, smoothed, peak)
    markers, n = ndi.label(h_minima(landscape, PROMINENCE_HU, footprint=FACE) & free, FACE)
    components, total = ndi.label(free, FACE)
    seeded = set(np.unique(components[markers > 0]))
    for component in range(1, total + 1):
        if component not in seeded:
            positions = np.flatnonzero(components == component)
            position = positions[np.argmin(smoothed.ravel()[positions])]
            n += 1
            markers.ravel()[position] = n
    if n > MAX_BASINS:
        raise CountingError('RIDGE_SEED_LIMIT', 'Synthetic candidate exceeds its fixed proposal budget.')
    labels = watershed(smoothed - 0.001 * distance, markers, mask=free, connectivity=FACE).astype(np.int32)
    initial = int(n)
    merges = []
    required_radius = float(spacing.max()) * 1.01
    # Freeze evidence before contraction so label traversal cannot change thresholds.
    parents = np.arange(initial + 1)
    for a, b in _pairs(labels):
        left, right = labels == a, labels == b
        levels = [float(np.percentile(smoothed[m], 20)) for m in (left, right)]
        level = min(levels) + PROMINENCE_HU
        passage = (left | right) & (smoothed <= level)
        clearance = _distance(passage, spacing)
        connected, _ = ndi.label(passage & (clearance >= required_radius), FACE)
        interiors = [m & (distance > required_radius) & (smoothed <= value)
                     for m, value in zip((left, right), levels)]
        shared = (set(np.unique(connected[interiors[0]])) & set(np.unique(connected[interiors[1]]))) - {0}
        if shared:
            merges.append({'basins': [int(a), int(b)], 'level_hu': level,
                           'required_radius_mm': required_radius,
                           'verified_radius_floor_mm': float(clearance[np.isin(connected, list(shared))].min())})
            parents[parents == parents[b]] = parents[a]
    labels = parents[labels].astype(np.int32)
    ids = np.unique(labels[labels > 0])
    compact = np.zeros_like(labels)
    for identifier, value in enumerate(ids, 1):
        compact[labels == value] = identifier
    return compact, {'initial_basin_count': initial, 'final_basin_count': len(ids),
                     'passage_merges': merges, 'smoothing_sigma_mm': sigma}


def count_with_ridges(hu, roi, affine):
    partitions, diagnostics = propose_partitions(hu, roi, affine)
    values, region, cfg = np.asarray(hu), np.asarray(roi), CountingConfig()
    spacing = geometry(affine)
    bone = values >= cfg.bone_hu
    free = region & ~bone
    supported, _ = _merge_unsupported_separators(partitions, free, bone, spacing, cfg)
    withheld = []
    if len(np.unique(supported[supported > 0])) < diagnostics['final_basin_count']:
        withheld.append('material_ambiguity')
    connected, _ = ndi.label(free, FACE)
    if any(len(np.unique(partitions[connected == value])) > 1
           for value in np.unique(connected[connected > 0])):
        withheld.append('unresolved_communication')
    # Proposal boundaries never contribute to original-HU strong-bone enclosure.
    bone_near = ndi.distance_transform_edt(~bone, sampling=spacing) <= min(max(float(spacing.max()) * 1.01, .8), 1.01)
    roi_edge = region & ~ndi.binary_erosion(region, structure=FACE)
    proposals, rejected = [], []
    for identifier in range(1, diagnostics['final_basin_count'] + 1):
        mask = partitions == identifier
        size = int(mask.sum())
        surface = mask & ~ndi.binary_erosion(mask, structure=FACE)
        surface_size = max(1, int(surface.sum()))
        enclosure = float(np.count_nonzero(surface & bone_near)) / surface_size
        edge = float(np.count_nonzero(surface & roi_edge)) / surface_size
        unsupported_edge = float(np.count_nonzero(surface & roi_edge & ~bone_near)) / surface_size
        if (size < cfg.minimum_voxels or _distance(mask, spacing).max() <= float(spacing.max()) * 1.01
                or enclosure < cfg.minimum_enclosure_fraction or unsupported_edge > cfg.maximum_roi_boundary_fraction):
            rejected.append(identifier)
            continue
        coordinates = np.argwhere(mask)
        physical = coordinates @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
        centroid = physical.mean(axis=0)
        air_count = int(np.count_nonzero(mask & (values < cfg.air_hu)))
        reasons = [REASON]
        if air_count == 0:
            reasons.append('fully_opacified_candidate_requires_review')
        elif air_count < size * .8:
            reasons.append('partly_opacified_candidate')
        if enclosure < .85:
            reasons.append('incomplete_bony_enclosure')
        if edge > 0:
            reasons.append('touches_inferred_anatomical_boundary')
        if size < 20:
            reasons.append('small_resolution_limited_candidate')
        if abs(float(centroid[0])) <= float(np.abs(np.asarray(affine)[0, :3]).sum()):
            reasons.append('midline_laterality_requires_review')
        if physical[:, 0].min() < 0 < physical[:, 0].max():
            reasons.append('crosses_midline_centroid_laterality_requires_review')
        cell = {'id': 0, 'side': 'left' if centroid[0] <= 0 else 'right',
                'centroid_ras_mm': centroid.tolist(), 'origin': 'automatic', 'included': True,
                'uncertain': True, 'reasons': reasons, 'voxel_count': size, 'air_voxel_count': air_count}
        proposals.append((tuple(round(v, 6) for v in centroid), mask, cell))
    if rejected:
        withheld.append('rejected_geometric_proposal')
    cavities, cells = np.zeros(values.shape, np.uint16), []
    for identifier, (_, mask, cell) in enumerate(sorted(proposals, key=lambda item: item[0]), 1):
        cavities[mask] = identifier
        cell['id'] = identifier
        cells.append(cell)
    counts = {'left': sum(c['side'] == 'left' for c in cells),
              'right': sum(c['side'] == 'right' for c in cells), 'total': len(cells)}
    diagnostics.update(method=METHOD, status='withheld' if withheld else 'accepted',
        counts={key: None for key in counts} if withheld else counts,
        withheld_reasons=withheld, experimental_count_available=not withheld,
        rejected_partition_ids=rejected, clinical_accuracy='NOT_MEASURED')
    return {'cavities': cavities, 'air': np.where(values < cfg.air_hu, cavities, 0).astype(np.uint16),
            'partitions': partitions, 'cells': cells, 'diagnostics': diagnostics,
            'warnings': ['Synthetic continuous-ridge proposals; clinical accuracy is unmeasured.']}
