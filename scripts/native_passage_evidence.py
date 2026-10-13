"""Image-derived passage measurements, never a clinical count or merge rule."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ethmoid_pipeline.counting.engine import CountingError, geometry
from ridge_counting_candidate import FACE, MAX_BASINS, _distance, _pairs


def _shared(mask, terminals):
    components, _ = ndi.label(mask, FACE)
    shared = (set(np.unique(components[terminals[0]]))
              & set(np.unique(components[terminals[1]]))) - {0}
    return np.isin(components, list(shared)) if shared else None


def _first_connected(levels, mask_at, terminals):
    """Binary search a monotone sequence; return None if even its last mask fails."""
    if len(levels) == 0 or _shared(mask_at(levels[-1]), terminals) is None:
        return None
    low, high = 0, len(levels) - 1
    while low < high:
        middle = (low + high) // 2
        if _shared(mask_at(levels[middle]), terminals) is None:
            low = middle + 1
        else:
            high = middle
    return float(levels[low])


def passage_evidence(hu, roi, affine, partitions):
    """Return minimax native HU and widest voxel-center clearance for each contact.

    Only measured HU, region, affine and detector proposals enter this function.
    A connected low-HU path can also be a blurred closed wall. Its existence never
    removes either of the detector's material or communication abstentions.
    """
    values, region, labels = np.asarray(hu), np.asarray(roi), np.asarray(partitions)
    if (values.ndim != 3 or values.shape != region.shape or values.shape != labels.shape
            or region.dtype != bool or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.isfinite(values).all()
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise CountingError('INVALID_PASSAGE_INPUT', 'Finite HU, boolean ROI and integer proposals must share a 3D grid.')
    spacing = geometry(affine)
    ids = np.unique(labels[labels > 0])
    if len(ids) > MAX_BASINS:
        raise CountingError('RIDGE_SEED_LIMIT', 'Passage diagnostics exceed the proposal budget.')
    free = region & (values < 200)
    if np.any((labels > 0) & ~free):
        raise CountingError('INVALID_PASSAGE_INPUT', 'Proposals must stay inside original non-bone region.')
    if any(ndi.label(labels == identifier, FACE)[1] != 1 for identifier in ids):
        raise CountingError('INVALID_PASSAGE_INPUT', 'Each proposal must be face connected.')
    distance = _distance(free, spacing)
    interior_radius = float(spacing.max()) * 1.01
    bone_near = (ndi.distance_transform_edt(values < 200, sampling=spacing)
                 <= min(max(interior_radius, .8), 1.01)) if np.any(values >= 200) else np.zeros(values.shape, bool)
    records, witnesses = [], {}
    for a, b in _pairs(labels):
        masks = (labels == a, labels == b)
        centers = [np.asarray(affine)[:3, :3] @ np.argwhere(mask).mean(axis=0)
                   + np.asarray(affine)[:3, 3] for mask in masks]
        order = sorted(range(2), key=lambda i: tuple(np.round(centers[i], 9)))
        pair = [int((a, b)[i]) for i in order]
        masks = tuple(masks[i] for i in order)
        floors = [float(np.percentile(values[mask], 20)) for mask in masks]
        terminals = tuple(mask & (distance > interior_radius) & (values <= floor)
                          for mask, floor in zip(masks, floors))
        record = {'basins': pair, 'centroids_ras_mm': [centers[i].tolist() for i in order],
                  'terminal_hu_limits': floors, 'terminal_sizes': [int(t.sum()) for t in terminals],
                  'native_bottleneck_hu': None, 'voxel_center_clearance_mm': None,
                  'witness_key': None, 'status': 'no_interior_terminals'}
        union = masks[0] | masks[1]
        contact = ((masks[0] & ndi.binary_dilation(masks[1], FACE))
                   | (masks[1] & ndi.binary_dilation(masks[0], FACE)))
        record['contact_hu_quantiles'] = np.percentile(values[contact], [0, 20, 50, 80, 100]).tolist()
        record['contact_near_strong_bone_fraction'] = float(bone_near[contact].mean())
        if all(terminal.any() for terminal in terminals):
            levels = np.unique(values[union])
            bottleneck = _first_connected(levels, lambda level: union & (values <= level), terminals)
            if bottleneck is None:
                record['status'] = 'no_native_path'
            else:
                passage = union & (values <= bottleneck)
                clearance = _distance(passage, spacing)
                # Ascending negative radii make the eligible masks grow monotonically.
                radius = _first_connected(-np.unique(clearance[passage])[::-1],
                    lambda negative: passage & (clearance >= -negative), terminals)
                witness = _shared(passage & (clearance >= -radius), terminals)
                key = f'pair_{pair[0]}_{pair[1]}'
                witnesses[key] = witness.astype(np.uint8)
                record.update(status='measured', native_bottleneck_hu=bottleneck,
                              voxel_center_clearance_mm=-radius, witness_key=key)
        records.append(record)
    records.sort(key=lambda row: tuple(np.asarray(row['centroids_ras_mm']).ravel()))
    return {'method': 'native-passage-evidence-v1', 'counts_changed': False,
            'clinical_accuracy': 'NOT_MEASURED', 'pairs': records}, witnesses
