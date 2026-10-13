"""A resolved bone plate does not prove that its narrow opening is closed."""
from pathlib import Path
import sys

import numpy as np
import pytest

ndi = pytest.importorskip("scipy.ndimage")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ridge_counting_candidate import count_with_ridges


@pytest.mark.parametrize("reflect", [False, True])
def test_supported_plate_with_narrow_real_opening_cannot_be_counted_as_two(reflect):
    hu = np.full((30, 24, 20), 800, dtype=np.float32)
    hu[4:26, 4:20, 4:16] = -900
    hu[14:16, 4:20, 4:16] = 800
    hu[14:16, 12, 10] = -900
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = [-7.25, -5.75, -4.75]
    if reflect:
        hu = hu[::-1].copy()
        affine[:3, 3] += affine[:3, 0] * (hu.shape[0] - 1)
        affine[:3, 0] *= -1
    assert ndi.label(hu < 200, ndi.generate_binary_structure(3, 1))[1] == 1
    roi = np.ones(hu.shape, bool)
    for array in (hu, roi, affine):
        array.setflags(write=False)
    result = count_with_ridges(hu, roi, affine)
    # Strong-bone support passes; the opening check alone prevents a false count.
    assert len(result["cells"]) == 2
    assert result["diagnostics"]["withheld_reasons"] == ["unresolved_communication"]
    assert result["diagnostics"]["status"] == "withheld"
    assert result["diagnostics"]["counts"] == {"left": None, "right": None, "total": None}
