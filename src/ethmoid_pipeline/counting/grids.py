"""Anatomy-only RAS proxies. Native CT voxels remain the counting authority."""
from __future__ import annotations

from itertools import product
import math

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .quality import is_oblique

PROXY_MAX_VOXELS = 128_000_000


def localization_grid(shape, affine) -> dict:
    spacing = nib.affines.voxel_sizes(affine)
    if not is_oblique(affine):
        return {"mode": "native_axis_aligned", "shape": list(shape), "affine": np.asarray(affine).tolist(),
                "spacing_mm": spacing.tolist(), "counting_grid": "original_native", "proxy_max_voxels": PROXY_MAX_VOXELS}
    step = float(spacing.min())
    corners = nib.affines.apply_affine(affine, list(product(*[(-0.5, size - 0.5) for size in shape])))
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    extent = (upper - lower) / step
    if not np.isfinite(extent).all():
        from .engine import CountingError
        raise CountingError("RESOURCE_LIMIT", "The anatomical proxy extent is outside the supported local budget.")
    # Python integers avoid overflow before the allocation guard examines an
    # extreme but finite affine or highly anisotropic source.
    target_shape = [max(1, math.ceil(float(width))) for width in extent]
    target = np.diag([step, step, step, 1.0])
    target[:3, 3] = lower + step / 2
    return {"mode": "anatomy_only_ras_proxy", "shape": target_shape, "affine": target.tolist(),
            "spacing_mm": [step] * 3, "hu_interpolation": "linear", "mask_interpolation": "nearest",
            "counting_grid": "original_native", "proxy_max_voxels": PROXY_MAX_VOXELS}


def check_proxy_limit(plan: dict) -> None:
    from .engine import CountingError
    if plan["mode"] == "anatomy_only_ras_proxy" and int(np.prod(plan["shape"], dtype=object)) > PROXY_MAX_VOXELS:
        raise CountingError("RESOURCE_LIMIT", "The anatomical proxy exceeds the fixed local voxel budget; the source was not coarsened.")


def resample(values, source_affine, target_shape, target_affine, *, order, outside=0):
    mapping = np.linalg.inv(source_affine) @ target_affine
    return ndi.affine_transform(values, mapping[:3, :3], mapping[:3, 3], output_shape=tuple(target_shape),
                                order=order, mode="constant", cval=outside, prefilter=False)


def anatomy_proxy(hu, affine, landmarks, plan):
    check_proxy_limit(plan)
    target, shape = np.asarray(plan["affine"]), tuple(plan["shape"])
    mapping = np.linalg.inv(affine) @ target
    # Use true half-voxel physical extent for validity, rather than treating
    # interpolation padding as an acquired air/bone compartment.
    valid = np.ones(shape, dtype=bool)
    x = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    for start in range(0, shape[2], 16):
        z = np.arange(start, min(start + 16, shape[2]))[None, None, :]
        block = np.ones((shape[0], shape[1], z.shape[2]), dtype=bool)
        for axis in range(3):
            position = mapping[axis, 0] * x + mapping[axis, 1] * y + mapping[axis, 2] * z + mapping[axis, 3]
            block &= (position >= -0.5) & (position <= hu.shape[axis] - 0.5)
        valid[:, :, start:start + z.shape[2]] = block
    proxy_hu = ndi.affine_transform(hu, mapping[:3, :3], mapping[:3, 3], output_shape=shape,
                                    order=1, mode="nearest", prefilter=False)
    proxy_hu[~valid] = -1024
    masks = {name: ndi.affine_transform(mask, mapping[:3, :3], mapping[:3, 3], output_shape=shape,
                                       order=0, mode="nearest", prefilter=False).astype(bool) & valid
             for name, mask in landmarks.items()}
    return proxy_hu, target, masks, valid
