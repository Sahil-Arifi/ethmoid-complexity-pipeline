from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ethmoid_pipeline.counting.engine import CountingError, compartment_stage
from synthetic_faint_wall_candidate import REASON, count_with_faint_wall, infer_exclusion


def test_reaches_only_face_connected_weak_voxels_and_preserves_thresholds():
    hu = np.full((7, 7, 7), -900.0)
    hu[1, 1, 1] = 200
    hu[2:4, 1, 1] = 50
    hu[4, 1, 1] = np.nextafter(50.0, 0)
    hu[4, 2, 1] = 100  # Diagonal contact cannot bridge the subthreshold voxel.
    hu[5, 5, 5] = 100
    before = hu.copy()
    mask = infer_exclusion(hu, np.ones_like(hu, dtype=bool))
    expected = np.zeros_like(mask)
    expected[2:4, 1, 1] = True
    assert np.array_equal(mask, expected)
    assert np.array_equal(hu, before)


def unit_volume(content=-900):
    hu = np.full((20, 18, 16), 800, dtype=np.float32)
    hu[3:17, 3:15, 3:13] = content
    affine = np.diag([0.5, 0.5, 0.5, 1])
    affine[:3, 3] = (-4.75, -4.25, -3.75)
    return hu, np.ones_like(hu, dtype=bool), affine


def test_roi_cannot_connect_weak_voxels_through_excluded_seed_or_gap():
    hu = np.full((5, 5, 5), -900.0)
    hu[:, 2, 2] = 100
    hu[0, 2, 2] = 800
    roi = np.ones_like(hu, dtype=bool)
    roi[2, 2, 2] = False
    mask = infer_exclusion(hu, roi)
    assert mask[1, 2, 2] and not mask[2:, 2, 2].any()
    roi[0, 2, 2] = False
    assert not infer_exclusion(hu, roi).any()


@pytest.mark.parametrize("content", [-900, 35])
def test_empty_mask_retains_complete_baseline_output(content):
    hu, roi, affine = unit_volume(content)
    original = compartment_stage(hu, roi, affine)
    actual = count_with_faint_wall(hu, roi, affine)
    assert not actual[4].any()
    assert np.array_equal(original[0], actual[0])
    assert np.array_equal(original[1], actual[1])
    assert original[2:] == actual[2:4]


def test_analytic_weak_sheet_changes_proposals_with_original_hu_bookkeeping():
    hu, roi, affine = unit_volume()
    hu[9, 3:15, 3:13] = 75
    before = [array.copy() for array in (hu, roi, affine)]
    for array in (hu, roi, affine):
        array.setflags(write=False)
    assert len(compartment_stage(hu, roi, affine)[2]) == 1
    cavities, air, cells, warnings, mask = count_with_faint_wall(hu, roi, affine)
    assert len(cells) == 2
    assert not cavities[mask].any()
    assert np.array_equal(air, np.where(hu < -500, cavities, 0))
    assert all(cell["uncertain"] and REASON in cell["reasons"] for cell in cells)
    assert any("Experimental weak-barrier" in warning for warning in warnings)
    for cell in cells:
        assert cell["voxel_count"] == np.count_nonzero(cavities == cell["id"])
        assert cell["air_voxel_count"] == np.count_nonzero(air == cell["id"])
    assert all(np.array_equal(array, saved) for array, saved in zip((hu, roi, affine), before))


def test_true_aperture_is_not_filled_by_propagation():
    hu, roi, affine = unit_volume()
    hu[9, 3:15, 3:13] = 75
    hu[9, 7:11, 6:10] = -900
    actual = count_with_faint_wall(hu, roi, affine)
    assert not actual[4][9, 7:11, 6:10].any()
    assert len(actual[2]) == 1


def test_weak_sheet_cannot_supply_missing_strong_enclosure():
    hu = np.full((12, 12, 12), 75.0)
    hu[0, 0, 0] = 800
    hu[3:9, 3:9, 3:9] = -900
    with pytest.raises(CountingError, match="not a zero-cell count"):
        count_with_faint_wall(hu, np.ones_like(hu, dtype=bool), np.eye(4))


@pytest.mark.parametrize("transform_kind", ["swap", "flip"])
def test_physical_masks_and_centroids_survive_array_reorientation(transform_kind):
    hu, roi, affine = unit_volume()
    hu[9, 3:15, 3:13] = 75
    original = count_with_faint_wall(hu, roi, affine)
    transform = np.eye(4)
    if transform_kind == "swap":
        transform[:3, :3] = np.eye(3)[[1, 0, 2]]
        altered_hu, altered_roi = hu.swapaxes(0, 1), roi.swapaxes(0, 1)
        undo = lambda array: array.swapaxes(0, 1)
    else:
        transform[0, 0], transform[0, 3] = -1, hu.shape[0] - 1
        altered_hu, altered_roi = hu[::-1], roi[::-1]
        undo = lambda array: array[::-1]
    actual = count_with_faint_wall(altered_hu, altered_roi, affine @ transform)
    for i in (0, 1, 4):
        assert np.array_equal(original[i], undo(actual[i]))
    assert [cell["side"] for cell in original[2]] == [cell["side"] for cell in actual[2]]
    assert np.allclose([cell["centroid_ras_mm"] for cell in original[2]],
                       [cell["centroid_ras_mm"] for cell in actual[2]], rtol=0, atol=1e-10)


@pytest.mark.parametrize("bad", ["shape", "nan", "complex", "roi_type", "empty"])
def test_invalid_input_is_rejected_before_propagation(bad):
    hu, roi, _ = unit_volume()
    if bad == "shape":
        roi = roi[:-1]
    elif bad == "nan":
        hu[0, 0, 0] = np.nan
    elif bad == "complex":
        hu = hu.astype(complex)
    elif bad == "roi_type":
        roi = roi.astype(np.uint8)
    else:
        roi[:] = False
    with pytest.raises(CountingError):
        infer_exclusion(hu, roi)
