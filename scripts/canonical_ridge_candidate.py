"""Isolated successor to the frozen ridge candidate; synthetic inputs only."""
import numpy as np
from scipy import ndimage as ndi

from ethmoid_pipeline.counting.engine import CountingError, geometry
from native_passage_evidence import _first_connected, _shared
from ridge_counting_candidate import FACE, _distance, count_with_ridges

COMMUNICATION_REASON = 'unresolved_subvoxel_communication'


def _inputs(hu, roi, affine):
    values, region = np.asarray(hu), np.asarray(roi)
    if (values.ndim != 3 or values.shape != region.shape or region.dtype != bool
            or any(size == 0 for size in values.shape)
            or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise CountingError('INVALID_VOLUME', 'Finite real 3D HU and a matching boolean ROI are required.')
    try:
        if np.iscomplexobj(affine):
            raise CountingError('INVALID_GEOMETRY', 'A real voxel-center affine is required.')
        matrix = np.asarray(affine, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise CountingError('INVALID_GEOMETRY', 'A real voxel-center affine is required.') from None
    spacing = geometry(matrix)
    if not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise CountingError('INVALID_GEOMETRY', 'A voxel-center affine must have a homogeneous final row.')
    if not region.any():
        raise CountingError('EMPTY_REGION', 'A nonempty synthetic region is required.')
    return values, region, matrix, spacing


def _canonical_grid(shape, matrix, spacing):
    directions = matrix[:3, :3] / spacing
    signs = np.array([1 if directions[np.argmax(np.abs(directions[:, axis])), axis] > 0 else -1
                      for axis in range(3)])
    oriented = directions * signs
    # Signed zero is immaterial; physical column values, never native axis IDs, break ties.
    def key(axis):
        values = tuple(-np.abs(oriented[:, axis])) + tuple(-oriented[:, axis])
        return tuple(0. if value == 0 else float(value) for value in values)
    order = tuple(sorted(range(3), key=key))
    flipped = tuple(axis for axis, original in enumerate(order) if signs[original] < 0)
    mapping = np.zeros((4, 4))
    mapping[3, 3] = 1
    for axis, original in enumerate(order):
        mapping[original, axis] = signs[original]
        if signs[original] < 0:
            mapping[original, 3] = shape[original] - 1
    return order, flipped, mapping


def component_bridge_evidence(hu, roi, affine):
    """Observe connectivity through all ROI intensities; never infer an opening."""
    values, region, _, spacing = _inputs(hu, roi, affine)
    free = region & (values < 200)
    components, number = ndi.label(free, FACE)
    record = {'component_count': int(number), 'status': 'not_two_components',
              'native_bottleneck_hu': None, 'touches_roi_boundary': None, 'terminal_sizes': []}
    witness = np.zeros(values.shape, np.uint8)
    # shortcut: pair evidence is limited to exactly two components; all larger sets remain withheld.
    if number != 2:
        return record, witness
    distance = _distance(free, spacing)
    masks = (components == 1, components == 2)
    terminals = tuple(mask & (distance > 1.01 * float(spacing.max()))
                      & (values <= np.percentile(values[mask], 20)) for mask in masks)
    record['terminal_sizes'] = [int(terminal.sum()) for terminal in terminals]
    record['status'] = 'no_interior_terminals'
    if not all(terminal.any() for terminal in terminals):
        return record, witness
    level = _first_connected(np.unique(values[region]), lambda value: region & (values <= value), terminals)
    record['status'] = 'no_roi_path'
    if level is not None:
        connected = _shared(region & (values <= level), terminals)
        edge = region & ~ndi.binary_erosion(region, structure=FACE)
        witness = connected.astype(np.uint8)
        record.update(status='measured', native_bottleneck_hu=level,
                      touches_roi_boundary=bool(np.any(connected & edge)))
    return record, witness


def count_with_canonical_ridges(hu, roi, affine):
    values, region, matrix, spacing = _inputs(hu, roi, affine)
    order, flipped, mapping = _canonical_grid(values.shape, matrix, spacing)
    def forward(array):
        return np.ascontiguousarray(np.flip(array.transpose(order), axis=flipped))
    canonical_hu, canonical_roi, canonical_affine = forward(values), forward(region), matrix @ mapping
    output = count_with_ridges(canonical_hu, canonical_roi, canonical_affine)
    evidence, output['component_bridge'] = component_bridge_evidence(canonical_hu, canonical_roi, canonical_affine)
    diagnostics = output['diagnostics']
    diagnostics['component_bridge_evidence'] = evidence
    if evidence['component_count'] > 1:
        diagnostics['withheld_reasons'].append(COMMUNICATION_REASON)
        diagnostics.update(status='withheld', experimental_count_available=False,
                           counts={'left': None, 'right': None, 'total': None})
    for key in ('partitions', 'cavities', 'air', 'component_bridge'):
        output[key] = np.flip(output[key], axis=flipped).transpose(np.argsort(order)).copy()
    diagnostics['method'] = 'canonical-component-withholding-v1'
    diagnostics['native_index_from_canonical'] = mapping.tolist()
    output['warnings'] = ['Synthetic orientation and withholding repair; clinical accuracy is unmeasured.']
    return output
