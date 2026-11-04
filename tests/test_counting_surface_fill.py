"""Physical surface interpolation controls; no anatomical accuracy claims."""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from scipy import ndimage as ndi

from ethmoid_pipeline.counting import localization, research_v5
from ethmoid_pipeline.counting.engine import CountingError, analyze_volume
from ethmoid_pipeline.counting.localization import _fill_surfaces


def test_surface_fill_uses_physical_nearest_support():
    surface = np.array([[0, 0, 22], [11, 0, 0]], dtype=np.int32)
    valid = surface > 0
    spacing = np.array([3.0, 1.0])
    supports = np.argwhere(valid)
    positions = np.moveaxis(np.indices(surface.shape), 0, -1)
    squared_distance = np.sum(((positions[..., None, :] - supports) * spacing) ** 2, axis=-1)
    assert np.all(np.diff(np.sort(squared_distance, axis=-1), axis=-1) > 0)
    expected = surface[valid][np.argmin(squared_distance, axis=-1)]

    actual = _fill_surfaces(surface, valid, spacing)

    assert actual[0, 0] == 22  # 2 mm away; the index-nearest donor is 3 mm away.
    assert np.array_equal(actual, expected)
    assert actual.dtype == surface.dtype
    assert np.array_equal(actual[valid], surface[valid])


@pytest.mark.parametrize("step", [0.6, 1.0])
def test_isotropic_surface_fill_preserves_existing_donors(step):
    surface = np.array([[0, 0, 22], [11, 0, 0]], dtype=np.int32)
    valid = surface > 0
    legacy_indices = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    assert np.array_equal(_fill_surfaces(surface, valid, np.array([step, step])), surface[tuple(legacy_indices)])


@pytest.mark.parametrize("fine_step", [1.0, 0.5])
def test_surface_fill_matches_refinement_at_common_physical_positions(fine_step):
    coarse = np.array([[0, 0, 22], [11, 0, 0]], dtype=np.int32)
    stride = np.array([3 / fine_step, 1 / fine_step], dtype=int)
    fine = np.zeros(tuple((np.array(coarse.shape) - 1) * stride + 1), dtype=np.int32)
    fine[tuple(np.argwhere(coarse).T * stride[:, None])] = coarse[coarse > 0]

    coarse_fill = _fill_surfaces(coarse, coarse > 0, np.array([3.0, 1.0]))
    fine_fill = _fill_surfaces(fine, fine > 0, np.array([fine_step, fine_step]))

    assert np.array_equal(coarse_fill, fine_fill[::stride[0], ::stride[1]])


def test_surface_fill_preserves_no_support_error():
    with pytest.raises(CountingError) as failure:
        _fill_surfaces(np.zeros((2, 3), np.int32), np.zeros((2, 3), bool), np.ones(2))
    assert failure.value.code == "ORBIT_WALL_UNRESOLVED"


@pytest.mark.parametrize("spacing", [[1], [1, 2, 3], [0, 1], [-1, 1], [np.nan, 1], [1, np.inf]])
def test_surface_fill_rejects_invalid_plane_spacing(spacing):
    with pytest.raises(CountingError) as failure:
        _fill_surfaces(np.ones((2, 3), np.int32), np.ones((2, 3), bool), np.asarray(spacing))
    assert failure.value.code == "INVALID_GEOMETRY"


@pytest.mark.parametrize("caller, expected_spacing", [
    ("orbit", [3.0, 5.0]), ("roof", [2.0, 3.0]), ("posterior", [2.0, 5.0]),
])
def test_surface_callers_forward_their_plane_axes(monkeypatch, caller, expected_spacing):
    spacing = np.array([2.0, 3.0, 5.0])
    calls = []

    def fill_sentinel(surface, valid, plane_spacing):
        assert surface.shape == valid.shape == (5, 4)
        assert valid.all()
        assert np.array_equal(plane_spacing, expected_spacing)
        calls.append(True)
        return np.full(surface.shape, 37, dtype=surface.dtype)

    monkeypatch.setattr(localization, "_fill_surfaces", fill_sentinel)
    if caller == "orbit":
        seed = np.zeros((12, 5, 4), bool)
        seed[0] = True
        bone = np.zeros_like(seed)
        bone[3] = True
        result = localization._medial_orbital_surface(seed, bone, 6.0, "left", spacing)
    elif caller == "roof":
        hu = np.zeros((5, 4, 8), np.float32)
        brain = np.zeros(hu.shape, bool)
        brain[:, :, 4:] = True
        bone = np.zeros_like(brain)
        bone[:, :, 3] = True
        nasal = np.zeros_like(brain)
        nasal[:, :, 0] = True
        result, _ = localization._intracranial_roof(hu, bone, brain, nasal, spacing)
    else:
        wall = np.full((5, 4), 7, np.int32)
        result = localization._continuous_wall(wall, np.ones(wall.shape, bool), spacing)
    assert calls == [True]
    assert np.all(result == 37)


@pytest.mark.parametrize("method", ["native", "research-v5"])
def test_surface_fill_component_version_survives_method_metadata(tmp_path, method):
    runner = analyze_volume if method == "native" else research_v5.analyze_research_volume
    result = runner(tmp_path / "missing-synthetic.nii", tmp_path / method)
    assert result["failure_code"] == "SOURCE_UNAVAILABLE"
    assert result["counting_engine_version"] == "native-bone-watershed-0.2.2"
    expected_method = "native-bone-watershed-0.2.2" if method == "native" else "research-native-0.5.0"
    assert result["method_version"] == expected_method
