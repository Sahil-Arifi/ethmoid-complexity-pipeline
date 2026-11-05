"""Isolated measurements from the sealed wall-aligned ridge prototype.

Recipe: opposite free air flanks at ceil(1.5 mm / native spacing), a bright
subthreshold center, half the bone-to-air contrast, and 6/9 tangential support.
Keep only six-connected ridge components touching six-neighbor bone dilation.
Measure padded physical EDT on ``free & ~ridge``; do not change ``free``.

The prototype used engine f6c37bbc at commit 1e041f3. Its private stage used this
distance for maxima/watershed, original free space for watershed/output, and
bone | ridge for merge support. It left acceptance unchanged. Current engine
0.2.3 differs, so this module deliberately provides no compartment adapter.
These are measured candidate ridges, not anatomical walls or closing planes.
The separate enclosure-support measurement follows the later pinned research
recipe. It makes no acceptance decision and is never enabled by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

PINNED_RECIPE = {
    "source_commit": "1e041f3823470f91eb45a07dd499d4187bd69f6c",
    "engine_sha256": "f6c37bbcc77fe8d4c6126c21f30717e78126a3a3fc1380b3f0ef9d083227c875",
    "ridge_trial_sha256": "d4f433278e46d7c992533a33c815399bb3a85ca6ab4978a4c4189b9d7ef7d373",
    "ridge_aligned_sha256": "26c5c80f04ada41c42c31373ed6df17215e76626e3f5a8d1a6ec6862ca90fd72",
    "acceptance_trial_sha256": "c2d10db15cbf62ff2def25a8b55a7b1c01b5bd4d6c5362d86ecb07783ba0e568",
}


class RidgeError(ValueError):
    """Fixed error codes do not expose source values."""


@dataclass(frozen=True)
class RidgeMeasurement:
    candidate_ridge: np.ndarray
    anchored_ridge: np.ndarray
    distance_mm: np.ndarray


@dataclass(frozen=True)
class RidgeEnclosureSupport:
    eligible_ridge: np.ndarray
    non_roi_support: np.ndarray
    near_radius_mm: float


def _interface_inputs(merged_labels, anchored_ridge):
    try:
        labels, ridge = np.asarray(merged_labels), np.asarray(anchored_ridge)
        if (labels.ndim != 3 or any(n == 0 for n in labels.shape)
                or labels.dtype.kind not in "iu" or np.any(labels < 0)
                or ridge.shape != labels.shape or ridge.dtype != np.dtype(bool)):
            raise RidgeError("RIDGE_INPUT_INVALID")
    except (TypeError, ValueError, OverflowError):
        raise RidgeError("RIDGE_INPUT_INVALID") from None
    return labels, ridge


def ridge_interface_endpoints(merged_labels, anchored_ridge):
    """Select measured ridge endpoints at faces of distinct positive labels.

    Inputs are final merged proposal labels and an already measured, anchored
    ridge mask on the same grid. Background, same-label faces, diagonal contact
    and unmeasured parts of an interface supply no endpoints.
    """
    labels, ridge = _interface_inputs(merged_labels, anchored_ridge)
    eligible = np.zeros_like(ridge)
    for axis in range(3):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lower), tuple(upper)
        a, b = labels[lo], labels[hi]
        distinct = (a > 0) & (b > 0) & (a != b)
        eligible[lo] |= distinct & ridge[lo]
        eligible[hi] |= distinct & ridge[hi]
    return eligible


def measure_ridge_enclosure_support(merged_labels, anchored_ridge, search, affine):
    """Measure the pinned, optional extra enclosure support; change no labels.

    Radius is ``min(max(max_spacing * 1.01, 0.8), 1.01)`` mm. The unpadded
    physical distance transform is guarded when there are no eligible endpoints.
    Exclude ``search & ~binary_erosion(search)`` from the resulting neighborhood.
    As in the pinned producer, the neighborhood is otherwise not clipped to
    search or labels: a caller would intersect it with each proposal's surface
    before combining it with existing bone support. This is not an ROI closure.
    """
    labels, ridge = _interface_inputs(merged_labels, anchored_ridge)
    try:
        domain = np.asarray(search)
        if domain.shape != labels.shape or domain.dtype != np.dtype(bool):
            raise RidgeError("RIDGE_INPUT_INVALID")
        spacing = _spacing(affine)
    except RidgeError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise RidgeError("RIDGE_INPUT_INVALID") from None
    eligible = ridge_interface_endpoints(labels, ridge)
    radius = min(max(float(spacing.max()) * 1.01, 0.8), 1.01)
    near = (ndi.distance_transform_edt(~eligible, sampling=spacing) <= radius
            if eligible.any() else np.zeros_like(ridge))
    roi_edge = domain & ~ndi.binary_erosion(domain)
    return RidgeEnclosureSupport(eligible, near & ~roi_edge, radius)


def _spacing(affine):
    matrix = np.asarray(affine)
    if matrix.shape != (4, 4) or matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise RidgeError("RIDGE_GEOMETRY_INVALID")
    matrix = matrix.astype(np.float64)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    if (not np.array_equal(matrix[3], [0, 0, 0, 1]) or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise RidgeError("RIDGE_GEOMETRY_INVALID")
    directions = matrix[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise RidgeError("RIDGE_GEOMETRY_INVALID")
    return spacing


def measure_ridges(hu, free, affine, *, air_hu=-500.0, bone_hu=200.0,
                   plate_half_width_mm=1.5):
    """Return native-grid ridge masks and ridge-aware distance, without labels.

    ``free`` must be the caller's existing boolean non-bone search domain.
    It may contain anatomical exclusions; this function never fills its holes.
    Parameters default to the sealed prototype and are not selected from counts.
    No current engine, source file, model, annotation or reference is opened.
    """
    try:
        values, domain = np.asarray(hu), np.asarray(free)
        if (values.ndim != 3 or any(n == 0 for n in values.shape)
                or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                or domain.shape != values.shape or domain.dtype != np.dtype(bool)):
            raise RidgeError("RIDGE_INPUT_INVALID")
        for value in (air_hu, bone_hu, plate_half_width_mm):
            scalar = np.asarray(value)
            if scalar.ndim != 0 or scalar.dtype.kind not in "iuf" or not np.isfinite(scalar):
                raise RidgeError("RIDGE_CONFIG_INVALID")
        if not (-1000 < air_hu < 0 and 50 <= bone_hu <= 1000 and 0 < plate_half_width_mm <= 5):
            raise RidgeError("RIDGE_CONFIG_INVALID")
        with np.errstate(over="ignore"):
            values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all() or np.any(domain & (values >= bone_hu)):
            raise RidgeError("RIDGE_INPUT_INVALID")
        spacing = _spacing(affine)
    except RidgeError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise RidgeError("RIDGE_INPUT_INVALID") from None

    ridge = np.zeros(domain.shape, bool)
    for axis in range(3):
        step = max(1, int(np.ceil(plate_half_width_mm / spacing[axis])))
        if 2 * step >= values.shape[axis]:
            continue
        middle = [slice(None)] * 3
        lower, upper = middle.copy(), middle.copy()
        middle[axis] = slice(step, -step)
        lower[axis], upper[axis] = slice(None, -2 * step), slice(2 * step, None)
        m, lo, hi = tuple(middle), tuple(lower), tuple(upper)
        flank = np.maximum(values[lo], values[hi])
        candidate = np.zeros_like(domain)
        candidate[m] = (domain[m] & domain[lo] & domain[hi]
                        & (values[m] >= air_hu) & (flank < air_hu)
                        & ((values[m] - flank) >= (bone_hu - air_hu) / 2))
        tangential = [3, 3, 3]
        tangential[axis] = 1
        support = ndi.convolve(candidate.astype(np.uint8), np.ones(tangential, np.uint8),
                               mode="constant", cval=0)
        ridge |= candidate & (support >= 6)

    components, _ = ndi.label(ridge, structure=ndi.generate_binary_structure(3, 1))
    bone_adjacent = ndi.binary_dilation(values >= bone_hu,
                                       structure=ndi.generate_binary_structure(3, 1))
    anchored_ids = np.unique(components[bone_adjacent & ridge])
    anchored = np.isin(components, anchored_ids) if anchored_ids.size else np.zeros_like(ridge)
    distance = ndi.distance_transform_edt(np.pad(domain & ~anchored, 1), sampling=spacing)
    distance = distance[1:-1, 1:-1, 1:-1].astype(np.float32)
    return RidgeMeasurement(ridge, anchored, distance)
