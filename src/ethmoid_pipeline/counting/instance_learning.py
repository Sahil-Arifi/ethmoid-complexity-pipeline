"""Isolated partial-instance learning; no clinical labels or production method.

PyTorch is loaded only when a model or loss is requested. Label values are -1
unknown, 0 explicitly non-cell, and positive IDs for supported instance regions.
An ID is an annotation assertion, never an inferred watershed identity.
"""
from __future__ import annotations

import numpy as np


MODEL_VERSION = "video-guided-instance-affinity-1"
OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
           (4, 0, 0), (0, 4, 0), (0, 0, 4))
THRESHOLD = 0.5


def normalize_hu(hu: np.ndarray) -> np.ndarray:
    """Two display ranges in [-1, 1]; source intensities remain unchanged."""
    hu = np.asarray(hu)
    if hu.ndim != 3 or 0 in hu.shape or not np.isfinite(hu).all():
        raise ValueError("A finite nonempty 3D HU array is required")
    return np.stack((np.clip(hu, -1000, 1000) / 1000,
                     (np.clip(hu, -1000, 200) + 400) / 600)).astype(np.float32)


def edge_slices(shape, offset):
    """Affinity at voxel p describes the edge p -> p + offset."""
    if len(shape) != 3 or any(int(size) <= 0 for size in shape):
        raise ValueError("A nonempty 3D shape is required")
    if tuple(offset) not in OFFSETS:
        raise ValueError("Unsupported affinity offset")
    source = tuple(slice(0, max(0, size-step)) for size, step in zip(shape, offset))
    target = tuple(slice(min(step, size), size) for size, step in zip(shape, offset))
    return source, target


def build_targets(instance_ids: np.ndarray, *, distinct_pairs=(),
                  foreground_labels: np.ndarray | None = None):
    """Return float32 targets and boolean supervision mask, each (7, D, H, W).

    Foreground labels optionally add explicit -1/0/1 observations where an
    instance ID is unavailable. They may not contradict known instance labels.
    Affinities require two positive IDs: equal IDs mean same-cell; unequal IDs
    mean different-cell only when that pair has explicit annotation support.
    Background and unknown endpoints never provide affinity supervision.
    """
    ids = np.asarray(instance_ids)
    if (ids.ndim != 3 or 0 in ids.shape or ids.dtype.kind not in "iu"
            or np.any(ids < -1) or np.any(ids > np.iinfo(np.int32).max)):
        raise ValueError("Instance IDs must be nonempty 3D integers: -1, 0 or positive")
    pairs = set()
    for pair in distinct_pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(v) is not int or v <= 0 for v in pair) or pair[0] == pair[1]):
            raise ValueError("Different-cell observations require two distinct positive integer IDs")
        pairs.add(tuple(sorted(pair)))
    targets = np.zeros((7, *ids.shape), dtype=np.float32)
    known = np.zeros(targets.shape, dtype=bool)
    targets[0] = ids > 0
    known[0] = ids >= 0
    if foreground_labels is not None:
        labels = np.asarray(foreground_labels)
        if (labels.shape != ids.shape or labels.dtype.kind not in "iu"
                or not np.isin(labels, (-1, 0, 1)).all()):
            raise ValueError("Foreground labels must be matching integer -1/0/1 observations")
        supplied = labels >= 0
        if np.any(supplied & known[0] & (labels != targets[0])):
            raise ValueError("Foreground and instance observations conflict")
        targets[0, supplied] = labels[supplied]
        known[0] |= supplied
    for channel, offset in enumerate(OFFSETS, start=1):
        source, target = edge_slices(ids.shape, offset)
        a, b = ids[source], ids[target]
        both = (a > 0) & (b > 0)
        same = both & (a == b)
        different = np.zeros(a.shape, dtype=bool)
        for left, right in pairs:
            different |= ((a == left) & (b == right)) | ((a == right) & (b == left))
        targets[(channel, *source)] = same
        known[(channel, *source)] = same | (both & different)
    return targets, known


def deterministic_max_pool3d(inputs):
    """Nonoverlapping 2-cubed max pooling with first-linear-index tie gradients.

    The installed CUDA native max-pool backward has no deterministic kernel.
    Disjoint groups and max over their final dimension preserve the native
    operation and tie policy without an overlapping gradient scatter.
    """
    if inputs.ndim != 5 or any(size < 2 or size % 2 for size in inputs.shape[2:]):
        raise ValueError("Pooling requires NCDHW input with even spatial sizes")
    n, c, d, h, w = inputs.shape
    groups = inputs.reshape(n, c, d//2, 2, h//2, 2, w//2, 2)
    groups = groups.permute(0, 1, 2, 4, 6, 3, 5, 7)
    return groups.reshape(n, c, d//2, h//2, w//2, 8).max(dim=-1).values


def build_model():
    """Four-level 2-input/7-logit 3D U-Net, with no pretrained weights."""
    import torch
    from torch import nn

    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels), nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels), nn.SiLU(),
        )

    class NonOverlappingMaxPool(nn.Module):
        def forward(self, inputs):
            return deterministic_max_pool3d(inputs)

    class InstanceUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([block(2, 8), block(8, 16),
                                           block(16, 32), block(32, 64)])
            self.pool = NonOverlappingMaxPool()
            self.ups = nn.ModuleList([nn.ConvTranspose3d(64, 32, 2, stride=2),
                                     nn.ConvTranspose3d(32, 16, 2, stride=2),
                                     nn.ConvTranspose3d(16, 8, 2, stride=2)])
            self.decoders = nn.ModuleList([block(64, 32), block(32, 16), block(16, 8)])
            self.head = nn.Conv3d(8, 7, 1)

        def forward(self, inputs):
            if (inputs.ndim != 5 or inputs.shape[1] != 2
                    or any(size < 8 or size % 8 for size in inputs.shape[2:])):
                raise ValueError("U-Net inputs require two channels and spatial sizes divisible by eight")
            skips = []
            x = inputs
            for index, encoder in enumerate(self.encoders):
                x = encoder(x)
                if index != len(self.encoders)-1:
                    skips.append(x)
                    x = self.pool(x)
            for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = decoder(torch.cat((up(x), skip), dim=1))
            return self.head(x)

    return InstanceUNet()


def masked_balanced_bce(logits, targets, known):
    """Mean per-head class means; unknown targets have exactly zero gradient.

    A missing class does not become a synthetic example. A fully unknown batch
    returns differentiable zero; readiness must separately require both classes.
    """
    import torch
    from torch.nn import functional as functional

    if (logits.ndim != 5 or logits.shape[1] != 7
            or targets.shape != logits.shape or known.shape != logits.shape
            or known.dtype != torch.bool or targets.device != logits.device
            or known.device != logits.device or not logits.is_floating_point()
            or not targets.is_floating_point()):
        raise ValueError("Loss requires matching floating N7DHW logits/targets and a boolean mask")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets[known]).all():
        raise ValueError("Known targets and logits must be finite")
    if not torch.all((targets[known] == 0) | (targets[known] == 1)):
        raise ValueError("Known targets must be binary")
    safe_targets = torch.where(known, targets, torch.zeros_like(targets))
    values = functional.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    heads = []
    for channel in range(7):
        classes = []
        for label in (0, 1):
            selected = known[:, channel] & (safe_targets[:, channel] == label)
            if selected.any():
                classes.append(values[:, channel][selected].mean())
        if classes:
            heads.append(torch.stack(classes).mean())
    return torch.stack(heads).mean() if heads else logits.sum() * 0


def decode_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Fixed foreground/affinity threshold graph components on the model grid.

    Long edges require foreground at both endpoints; intervening voxels do not
    assert membership. Component size/laterality/native projection are runner
    responsibilities. Empty foreground returns an all-zero int32 label volume.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    values = np.asarray(probabilities)
    if (values.ndim != 4 or values.shape[0] != 7 or 0 in values.shape[1:]
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Decoder requires finite probabilities in [0,1] with shape 7DHW")
    foreground = values[0] >= THRESHOLD
    result = np.zeros(foreground.shape, dtype=np.int32)
    count = int(foreground.sum())
    if count == 0:
        return result
    if count > np.iinfo(np.int32).max:
        raise ValueError("Decoder component indexing exceeds int32")
    indices = np.full(foreground.shape, -1, dtype=np.int32)
    indices[foreground] = np.arange(count, dtype=np.int32)
    rows, columns = [], []
    for channel, offset in enumerate(OFFSETS, start=1):
        source, target = edge_slices(foreground.shape, offset)
        selected = foreground[source] & foreground[target] & (values[(channel, *source)] >= THRESHOLD)
        rows.append(indices[source][selected])
        columns.append(indices[target][selected])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)), shape=(count, count)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    result[foreground] = labels.astype(np.int32) + 1
    return result
