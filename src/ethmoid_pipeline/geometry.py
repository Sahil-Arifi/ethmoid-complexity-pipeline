"""Discrete voxel assignment proof for the bounded exploratory MPR policy."""
from __future__ import annotations

import numpy as np

from .util import PipelineError


class PhysicalGridLimitExceeded(PipelineError):
    def __init__(self, maximum: float, limit: float):
        self.diagnostics = {"max_world_residual_mm": maximum, "physical_residual_limit_mm": limit}
        super().__init__("Quantized MPR source corner exceeds the 0.001 mm physical limit.")


def dicom_point_ras(record, row: int, column: int) -> np.ndarray:
    iop = np.asarray(record.iop, dtype=float)
    point = np.asarray(record.ipp) + row * record.pixel_spacing[0] * iop[3:] + column * record.pixel_spacing[1] * iop[:3]
    return point * np.asarray([-1.0, -1.0, 1.0])


def prove_native_voxel_bijection(shape, affine, records, *, physical_limit_mm: float | None = None) -> dict:
    """Prove a complete signed-axis voxel permutation, without a visited mask.

    Fixed unit row/column steps, complete endpoints and unique consecutive planes
    imply every voxel is assigned once. Checking physical error at all corners
    bounds every pixel because the per-plane position maps are affine.
    """
    affine = np.asarray(affine, dtype=float)
    if len(shape) != 3 or not np.all(np.isfinite(affine)):
        raise PipelineError("Native-grid geometry is invalid.")
    inverse = np.linalg.inv(affine)
    signature, origins, maximum_mm, maximum_index = None, [], 0.0, 0.0
    for record in records:
        samples = ((0, 0), (1, 0), (0, 1), (0, record.columns - 1), (record.rows - 1, 0), (record.rows - 1, record.columns - 1))
        mapped = []
        for row, column in samples:
            point = dicom_point_ras(record, row, column)
            floating = (inverse @ np.append(point, 1.0))[:3]
            index = np.rint(floating).astype(int)
            index_error = float(np.max(np.abs(floating - index)))
            error = float(np.linalg.norm((affine @ np.append(index, 1.0))[:3] - point))
            maximum_mm, maximum_index = max(maximum_mm, error), max(maximum_index, index_error)
            if index_error >= 0.5 or any(v < 0 or v >= shape[k] for k, v in enumerate(index)):
                raise PipelineError("Source pixels do not have an unambiguous in-bounds native voxel assignment.")
            mapped.append(index)
        origin, row_step, column_step = mapped[0], mapped[1] - mapped[0], mapped[2] - mapped[0]
        if np.sum(np.abs(row_step)) != 1 or np.sum(np.abs(column_step)) != 1:
            raise PipelineError("Source pixels do not map to native unit voxel steps.")
        row_axis, column_axis = int(np.argmax(np.abs(row_step))), int(np.argmax(np.abs(column_step)))
        if row_axis == column_axis:
            raise PipelineError("Source in-plane axes overlap.")
        slice_axis = ({0, 1, 2} - {row_axis, column_axis}).pop()
        current = (tuple(row_step), tuple(column_step), slice_axis)
        if signature is None:
            signature = current
        if current != signature or shape[row_axis] != record.rows or shape[column_axis] != record.columns:
            raise PipelineError("Source axes or dimensions change within the native voxel assignment.")
        if origin[row_axis] != (0 if row_step[row_axis] > 0 else shape[row_axis] - 1) or origin[column_axis] != (0 if column_step[column_axis] > 0 else shape[column_axis] - 1):
            raise PipelineError("Source in-plane endpoints do not span the complete native image.")
        for (row, column), index in zip(samples, mapped):
            if not np.array_equal(index, origin + row * row_step + column * column_step):
                raise PipelineError("Source corners do not share one exact native pixel mapping.")
        origins.append(tuple(int(v) for v in origin))
    if not origins or signature is None:
        raise PipelineError("No native planes were assigned.")
    plane_indices = np.asarray(origins)[:, signature[2]]
    steps = np.diff(plane_indices)
    if len(plane_indices) != shape[signature[2]] or len(set(plane_indices)) != len(plane_indices) or set(plane_indices) != set(range(shape[signature[2]])) or (len(steps) and (not np.all(steps == steps[0]) or abs(steps[0]) != 1)):
        raise PipelineError("Source planes do not map once to every consecutive native plane.")
    if physical_limit_mm is not None and maximum_mm > physical_limit_mm:
        raise PhysicalGridLimitExceeded(maximum_mm, physical_limit_mm)
    return {"assignment": (signature, tuple(origins)), "max_world_residual_mm": maximum_mm,
            "max_index_residual_voxel": maximum_index, "one_to_one_native_voxels": True}
