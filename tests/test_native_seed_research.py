"""Marker-field controls only; these do not validate native cell counts."""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
from scipy import ndimage as ndi
from skimage.morphology import h_maxima

from ethmoid_pipeline.counting.native_seed_research import SeedError, discard_ridge_markers


def test_intersection_removes_the_entire_face_connected_plateau():
    maxima = np.zeros((7, 7, 7), bool)
    maxima[1:6, 3, 3] = True
    maxima[3, 3:6, 3] = True
    ridge = np.zeros_like(maxima)
    ridge[3, 3, 3] = True
    assert (maxima & ~ridge).any()
    assert not discard_ridge_markers(maxima, ridge).any()


def test_diagonal_markers_are_separate_and_only_intersected_components_go():
    maxima = np.zeros((7, 7, 7), bool)
    maxima[1:4, 2, 2] = True
    maxima[3:5, 3, 3] = True  # Edge contact with the first plateau is not a face.
    maxima[6, 0, 0] = True
    ridge = np.zeros_like(maxima)
    ridge[2, 2, 2] = True
    expected = maxima.copy()
    expected[1:4, 2, 2] = False
    assert np.array_equal(discard_ridge_markers(maxima, ridge), expected)


@pytest.mark.parametrize("location", ["absent", "background", "all-background"])
def test_background_label_never_becomes_a_marker(location):
    maxima = np.zeros((5, 5, 5), bool)
    if location != "all-background":
        maxima[2, 1:4, 2] = True
    ridge = np.zeros_like(maxima)
    if location != "absent":
        ridge[0] = True
    result = discard_ridge_markers(maxima, ridge)
    assert np.array_equal(result, maxima)
    assert result.dtype == np.dtype(bool)
    assert not np.shares_memory(result, maxima)
    assert not np.shares_memory(result, ridge)


def test_readonly_noncontiguous_inputs_are_unchanged():
    maxima = np.zeros((7, 6, 5), bool)
    maxima[1:6, 3, 2] = True
    ridge = np.zeros_like(maxima)
    ridge[3, 3, 2] = True
    maxima, ridge = maxima[::-1], ridge[::-1]
    copies = maxima.copy(), ridge.copy()
    maxima.flags.writeable = ridge.flags.writeable = False
    result = discard_ridge_markers(maxima, ridge)
    assert not result.any()
    assert all(np.array_equal(before, after) for before, after in zip(copies, (maxima, ridge)))


@pytest.mark.parametrize("which", ["maxima", "ridge"])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.complex64, object])
def test_non_boolean_masks_are_rejected_without_coercion(which, dtype):
    maxima = np.ones((3, 4, 5), bool)
    ridge = np.zeros_like(maxima)
    if which == "maxima":
        maxima = maxima.astype(dtype)
    else:
        ridge = ridge.astype(dtype)
    with pytest.raises(SeedError, match="^SEED_INPUT_INVALID$"):
        discard_ridge_markers(maxima, ridge)


@pytest.mark.parametrize("shape", [(), (3,), (3, 4), (3, 4, 5, 1), (0, 4, 5), (3, 0, 5), (3, 4, 0)])
def test_rank_and_empty_axes_are_rejected(shape):
    maxima = np.zeros(shape, bool)
    with pytest.raises(SeedError, match="^SEED_INPUT_INVALID$"):
        discard_ridge_markers(maxima, maxima.copy())


@pytest.mark.parametrize("ridge", [np.zeros((3, 4, 6), bool), np.zeros((3, 4), bool), True])
def test_ridge_shape_must_match(ridge):
    with pytest.raises(SeedError, match="^SEED_INPUT_INVALID$"):
        discard_ridge_markers(np.zeros((3, 4, 5), bool), ridge)


def test_ragged_input_has_a_fixed_error():
    with pytest.raises(SeedError, match="^SEED_INPUT_INVALID$"):
        discard_ridge_markers([[True], [True, False]], np.zeros((3, 4, 5), bool))


def test_weak_wall_edt_plateau_is_discarded_whole_after_explicit_bool_cast():
    hu = np.full((17, 13, 13), 800, np.float32)
    hu[1:-1, 1:-1, 1:-1] = -900
    hu[8, 1:-1, 1:-1] = 0
    free, ridge = hu < 200, hu == 0
    distance = ndi.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1, 1:-1]
    raw = h_maxima(distance, 1)
    assert raw.dtype == np.uint8
    with pytest.raises(SeedError, match="^SEED_INPUT_INVALID$"):
        discard_ridge_markers(raw, ridge)
    maxima = raw.astype(bool)
    assert maxima.sum() == 5 and np.count_nonzero(maxima & ridge) == 1
    assert ndi.label(maxima)[1] == 1
    assert ndi.label(maxima & ~ridge)[1] == 2  # Voxel clipping invents two markers.
    assert not discard_ridge_markers(maxima, ridge).any()
