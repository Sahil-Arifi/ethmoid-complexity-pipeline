"""Isolated marker filtering, without an application or compartment adapter."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class SeedError(ValueError):
    """Fixed diagnostic without source values."""


def discard_ridge_markers(maxima, ridge):
    """Remove whole six-connected maxima components touching measured ridge.

    Both masks must be boolean; callers must explicitly convert uint8 h-maxima.
    This filters markers only and makes no anatomical or count claim.
    """
    try:
        maxima, ridge = np.asarray(maxima), np.asarray(ridge)
        if (maxima.ndim != 3 or any(n == 0 for n in maxima.shape)
                or maxima.dtype != np.dtype(bool) or ridge.shape != maxima.shape
                or ridge.dtype != np.dtype(bool)):
            raise SeedError("SEED_INPUT_INVALID")
    except (TypeError, ValueError, OverflowError):
        raise SeedError("SEED_INPUT_INVALID") from None
    labels, count = ndi.label(maxima, structure=ndi.generate_binary_structure(3, 1))
    keep = np.ones(count + 1, bool)
    keep[0] = False
    keep[labels[ridge]] = False
    return keep[labels]
