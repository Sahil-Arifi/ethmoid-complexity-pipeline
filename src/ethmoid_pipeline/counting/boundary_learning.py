"""Native-grid two-head boundary model; no reference counts enter learning."""
from __future__ import annotations

import time
import numpy as np

from . import instance_learning as legacy
from .instance_experiment import require, starts

VERSION = "boundary-instance-v1"
MODEL_VERSION = "native-membership-separator-unet-2"
CONFIG = {
    "patch_size": 64, "batch_size": 2, "steps": 2000,
    "patch_sampling": "native_valid_containing_anchor-v2",
    "patch_sampling_rng": "independent_generator_same_seed",
    "maximum_gpu_seconds": 3600, "validation_interval": 100,
    "seed": 20261105, "learning_rate": 0.001, "weight_decay": 0.0001,
    "channels": [8, 16, 32, 64], "inference_stride": 32,
    "precision": "float32", "grid": "native_canonical_no_interpolation",
    "heads": ["membership", "separator"], "threshold": 0.5,
    "minimum_native_voxels": 4, "maximum_native_voxels": 64000000,
}


def sample_patch_center(shape, anchor, rng, size=64):
    """Sample a native valid window containing the anchor, matching short-axis inference padding."""
    try:
        dimensions, selected = np.asarray(shape), np.asarray(anchor)
    except (TypeError, ValueError):
        require(False, "BOUNDARY_PATCH_SAMPLER_INPUT_INVALID")
    require(type(size) is int and size >= 8 and size % 8 == 0
            and isinstance(rng, np.random.Generator)
            and dimensions.shape == selected.shape == (3,)
            and dimensions.dtype.kind in "iu" and selected.dtype.kind in "iu"
            and np.all(dimensions > 0) and np.all(dimensions <= np.iinfo(np.intp).max - size)
            and np.all(selected >= 0) and np.all(selected < dimensions), "BOUNDARY_PATCH_SAMPLER_INPUT_INVALID")
    low = []
    for length, position in zip(dimensions, selected):
        length, position = int(length), int(position)
        if length < size:
            low.append(0)
        else:
            lower, upper = max(0, position - size + 1), min(position, length - size)
            low.append(int(rng.integers(lower, upper + 1)))
    return np.asarray(low, dtype=np.intp) + size // 2


def build_model():
    from torch import nn
    model = legacy.build_model()
    model.head = nn.Conv3d(8, 2, 1)
    return model


def targets(arrays):
    values = np.stack([arrays["foreground_labels"], arrays["separator_labels"]])
    require(values.ndim == 4 and values.dtype.kind in "iu"
            and np.isin(values, [-1, 0, 1]).all(), "BOUNDARY_TARGETS_INVALID")
    known = values >= 0
    return np.maximum(values, 0).astype(np.float32), known


def masked_loss(logits, target, known):
    import torch
    from torch.nn import functional as F
    require(logits.ndim == 5 and logits.shape[1] == 2
            and logits.shape == target.shape == known.shape
            and known.dtype == torch.bool and logits.device == target.device == known.device,
            "BOUNDARY_LOSS_SHAPE_INVALID")
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(target).all())
            and not bool(((target != 0) & (target != 1) & known).any()),
            "BOUNDARY_LOSS_VALUES_INVALID")
    heads = []
    for head in range(2):
        classes = []
        for value in (0, 1):
            selected = known[:, head] & (target[:, head] == value)
            if bool(selected.any()):
                classes.append(F.binary_cross_entropy_with_logits(
                    logits[:, head][selected], target[:, head][selected]))
        if classes:
            heads.append(torch.stack(classes).mean())
    return torch.stack(heads).mean() if heads else logits.sum() * 0


def probabilities(hu, model, torch, *, device="cuda", patch_size=64, stride=32, deadline=None):
    """Overlap-average scalar heads on their original grid, without resampling."""
    require(hu.ndim == 3 and hu.size <= CONFIG["maximum_native_voxels"]
            and np.isfinite(hu).all(), "BOUNDARY_INPUT_INVALID")
    require(type(patch_size) is int and patch_size >= 8 and patch_size % 8 == 0
            and type(stride) is int and 0 < stride <= patch_size, "BOUNDARY_TILING_INVALID")
    sums = np.zeros((2, *hu.shape), np.float32)
    weights = np.zeros(hu.shape, np.uint16)
    model.eval()
    with torch.inference_mode():
        for a in starts(hu.shape[0], patch_size, stride):
            for b in starts(hu.shape[1], patch_size, stride):
                for c in starts(hu.shape[2], patch_size, stride):
                    require(deadline is None or time.monotonic() < deadline, "INFERENCE_TIME_BUDGET_EXCEEDED")
                    low = np.array([a, b, c]); high = np.minimum(low + patch_size, hu.shape)
                    box = tuple(slice(x, y) for x, y in zip(low, high))
                    sub = tuple(slice(0, y-x) for x, y in zip(low, high))
                    patch = np.full((patch_size,)*3, -1000, np.float32)
                    patch[sub] = hu[box]
                    x = torch.from_numpy(legacy.normalize_hu(patch)[None]).to(device)
                    result = torch.sigmoid(model(x))[0].cpu().numpy()
                    require(result.shape == (2, patch_size, patch_size, patch_size)
                            and np.isfinite(result).all(), "BOUNDARY_PREDICTIONS_INVALID")
                    sums[(slice(None), *box)] += result[(slice(None), *sub)]
                    weights[box] += 1
                    require(deadline is None or time.monotonic() < deadline, "INFERENCE_TIME_BUDGET_EXCEEDED")
    require(bool(np.all(weights > 0)), "BOUNDARY_TILING_COVERAGE_FAILED")
    sums /= weights[None]
    return sums
