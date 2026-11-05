"""Quarantined changed numerical policy; no repository installation or clinical claim."""
import math

import numpy as np
from scipy import ndimage as ndi

from ethmoid_pipeline.counting import engine

VERSION = "dynamic-owned-physical-incidence-research-0.1"
EXPECTED_ENGINE_SHA256 = "9f1cd9ac64a47d8a94933e7824a428b92017074e2c2b22485dde695414c69eb9"
MAX_GENERATED_INCIDENCES = 8_000_000
MAX_STENCIL_VOXELS = 2_000_000
MAX_STAGE_INTERFACES = 2_000_000


class ResourceLimitError(RuntimeError):
    pass


def _radius_offsets(shape, spacing, radius):
    extent = [min(int(size) - 1, math.ceil(radius / float(step))) for size, step in zip(shape, spacing)]
    stencil_shape = tuple(2 * value + 1 for value in extent)
    if math.prod(stencil_shape) > MAX_STENCIL_VOXELS:
        raise ResourceLimitError("STENCIL_VOXEL_BUDGET_EXCEEDED")
    stencil = np.ones(stencil_shape, bool)
    stencil[tuple(extent)] = False
    # A single-point EDT preserves the legacy distance arithmetic at the radius boundary.
    eligible = ndi.distance_transform_edt(stencil, sampling=spacing) <= radius
    return np.argwhere(eligible).astype(np.int32) - np.array(extent, np.int32)


def _prepare_incidence(labels, bone, spacing, config, axes):
    offsets = _radius_offsets(labels.shape, spacing, config.separator_support_mm)
    face_a, face_b, face_axis, incidence_parts = [], [], [], []
    face_count = generated = 0
    axis_plate_counts = {}
    shape = np.array(labels.shape)
    for axis in axes:
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        left, right = labels[tuple(low)], labels[tuple(high)]
        boundary = (left != right) & (left > 0) & (right > 0)
        points = np.argwhere(boundary).astype(np.int32)
        count = len(points)
        face_a.append(left[boundary])
        face_b.append(right[boundary])
        face_axis.append(np.full(count, axis, np.int8))
        step = max(1, int(np.ceil(config.plate_half_width_mm / spacing[axis])))
        positive = engine._shift(labels, axis, step)
        negative = engine._shift(labels, axis, -step)
        plate = bone & (positive > 0) & (negative > 0) & (positive != negative)
        axis_plate_counts[str(axis)] = int(plate.sum())
        parts = []
        if count and plate.any():
            for endpoint in (0, 1):
                base = points.copy()
                base[:, axis] += endpoint
                for offset in offsets:
                    nearby = base + offset
                    valid = np.all((nearby >= 0) & (nearby < shape), axis=1)
                    ids = np.flatnonzero(valid)
                    coordinates = nearby[valid]
                    owned = plate[tuple(coordinates.T)]
                    ids, coordinates = ids[owned], coordinates[owned]
                    if not len(ids):
                        continue
                    generated += len(ids)
                    if generated > MAX_GENERATED_INCIDENCES:
                        raise ResourceLimitError("GENERATED_INCIDENCE_BUDGET_EXCEEDED")
                    first, second = positive[tuple(coordinates.T)], negative[tuple(coordinates.T)]
                    parts.append(np.column_stack((ids + face_count, np.minimum(first, second), np.maximum(first, second))).astype(np.int32))
        if parts:
            # Multiple bone origins with the same raw flank pair have identical future ownership.
            incidence_parts.append(np.unique(np.concatenate(parts), axis=0))
        face_count += count
    incidence = np.concatenate(incidence_parts) if incidence_parts else np.empty((0, 3), np.int32)
    return {"face_a": np.concatenate(face_a), "face_b": np.concatenate(face_b),
            "face_axis": np.concatenate(face_axis), "incidence": incidence,
            "preflight": {"face_count": face_count, "generated_incidence_rows": generated,
                          "unique_incidence_rows": len(incidence), "radius_offset_count": len(offsets),
                          "potential_plate_voxels_by_axis": axis_plate_counts,
                          "generated_incidence_budget": MAX_GENERATED_INCIDENCES}}


def _measure_incidence(prepared, mapping, axes):
    first = mapping[prepared["face_a"]]
    second = mapping[prepared["face_b"]]
    low, high = np.minimum(first, second), np.maximum(first, second)
    active = low != high
    support = np.zeros(len(low), bool)
    incidence = prepared["incidence"]
    if len(incidence):
        face, a, b = incidence.T
        one, two = mapping[a], mapping[b]
        matched = active[face] & (np.minimum(one, two) == low[face]) & (np.maximum(one, two) == high[face])
        support[face[matched]] = True
    measured = {}
    for axis in axes:
        selected = active & (prepared["face_axis"] == axis)
        pairs = np.column_stack((low[selected], high[selected]))
        if not len(pairs):
            continue
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        totals = np.bincount(inverse)
        backed = np.bincount(inverse, weights=support[selected].astype(np.uint8))
        for pair, total, count in zip(unique, totals, backed):
            key = tuple(map(int, pair))
            if key not in measured or total > measured[key][0]:
                measured[key] = (int(total), int(count), axis)
    return measured


def candidate(labels, free, bone, spacing, config, *, affine=None, markers=None, trace=None):
    if affine is None or markers is None:
        raise ValueError("PHYSICAL_AFFINE_AND_ORIGINAL_MARKERS_REQUIRED")
    labels, free, bone, markers = map(np.asarray, (labels, free, bone, markers))
    spacing = np.asarray(spacing, dtype=float)
    if labels.ndim != 3 or not labels.shape == free.shape == bone.shape == markers.shape:
        raise ValueError("ALIGNED_THREE_DIMENSIONAL_INPUTS_REQUIRED")
    if labels.dtype.kind not in "iu" or markers.dtype.kind not in "iu" or np.any(labels < 0) or np.any(markers < 0):
        raise ValueError("NONNEGATIVE_INTEGER_LABELS_REQUIRED")
    if not np.array_equal(labels > 0, free) or np.any((markers > 0) & (markers != labels)):
        raise ValueError("MARKERS_AND_RAW_FOREGROUND_MUST_MATCH")
    if not np.array_equal(spacing, engine.geometry(affine)):
        raise ValueError("SPACING_MUST_MATCH_PHYSICAL_AFFINE")
    axes = engine._execution_axes(affine, spacing)[0]
    n = int(labels.max())
    bounds = ndi.find_objects(markers)
    anchors = {}
    for label in np.unique(labels[labels > 0]):
        label = int(label)
        region = bounds[label - 1] if label <= len(bounds) else None
        if region is None:
            raise ValueError("EVERY_RAW_BASIN_REQUIRES_ORIGINAL_MARKER")
        anchors[label] = engine._physical_anchor(markers[region] == label, region, affine)
    if len(set(anchors.values())) != len(anchors):
        raise ValueError("ORIGINAL_PHYSICAL_MARKER_ANCHOR_COLLISION")
    prepared = _prepare_incidence(labels, bone.astype(bool, copy=False), spacing, config, axes)
    mapping = np.arange(n + 1, dtype=np.int32)
    constraints, decisions, stages = set(), [], []
    stage_interfaces = unions = 0
    while True:
        measured = _measure_incidence(prepared, mapping, axes)
        for pair, (total, backed, _) in measured.items():
            if backed / total >= config.separator_support_fraction:
                constraints.add(pair)
        mapped_constraints = {tuple(sorted((int(mapping[a]), int(mapping[b])))) for a, b in constraints}
        assert all(a != b for a, b in mapped_constraints)
        options = []
        for (a, b), (total, backed, axis) in measured.items():
            if (a, b) in mapped_constraints:
                continue
            area = total * float(np.prod(np.delete(spacing, axis)))
            key = tuple(sorted((anchors[a], anchors[b])))
            options.append((backed / total, -area, key, a, b))
        if trace is not None:
            stage_interfaces += len(measured)
            if stage_interfaces > MAX_STAGE_INTERFACES:
                raise ResourceLimitError("RECORDED_STAGE_INTERFACE_BUDGET_EXCEEDED")
            stages.append({"regions": len(np.unique(mapping[labels])) - int(np.any(labels == 0)),
                "interfaces": [{"basins": list(pair), "dominant_axis": axis,
                    "interface_faces": total, "supported_faces": backed, "support_fraction": backed / total,
                    "preserved": pair in mapped_constraints}
                    for pair, (total, backed, axis) in sorted(measured.items())]})
        if not options:
            break
        fraction, minus_area, key, a, b = min(options)
        mapping[mapping == b] = a
        anchors[a] = min(anchors[a], anchors[b])
        unions += 1
        if trace is not None:
            decisions.append({"basins": [a, b], "merged": True,
                "reason": "dynamic_pair_owned_insufficient_separator_support", "support_fraction": fraction,
                "dominant_interface_area_mm2": -minus_area, "component_marker_anchor_pair": key})
        assert unions <= n - 1
    uncertain = {int(mapping[value]) for pair in constraints for value in pair}
    if trace is not None:
        trace["interfaces"] = stages[0]["interfaces"]
        trace["dynamic_separator_stages"] = stages
        trace["merge_decisions"] = decisions
        trace["basin_to_merged_label"] = mapping.tolist()
        trace["incidence_preflight"] = prepared["preflight"]
        trace["candidate_version"] = VERSION
        trace["tie_policy"] = "physical_axis_order_then_fraction_area_component_min_all_original_marker_anchors"
    return mapping[labels], uncertain
