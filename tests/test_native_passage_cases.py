"""Independent physical and array-contract checks; no counting detector calls."""
from pathlib import Path
import sys

import numpy as np
import pytest

ndi = pytest.importorskip("scipy.ndimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import native_passage_cases as passage


@pytest.fixture(scope="module")
def items():
    return passage.cases()


def _points(mask, affine):
    return np.argwhere(mask) @ affine[:3, :3].T + affine[:3, 3]


def test_exact_fixed_design_and_metadata(items):
    expected = {(wall, 0.0, sigma, 0.0, 0.0) for wall in (0.125, 0.5, 1.0) for sigma in (0.0, 0.3)}
    expected |= {(1.0, opening, sigma, 0.0, 0.0) for opening in (0.25, 0.5, 1.5) for sigma in (0.0, 0.3)}
    expected |= {(wall, opening, 0.3, phase, noise) for wall, opening in ((0.5, 0.0), (1.0, 0.5))
                 for phase, noise in ((0.125, 0.0), (0.0, 25.0))}
    assert len(items) == len({case.fixture.name for case in items}) == 32
    for transform in ("base", "swap_xz_flip_y"):
        selected = [case for case in items if case.transform == transform]
        assert len(selected) == 16
        assert {(c.wall_mm, c.aperture_mm, c.sigma_mm, c.phase_mm, c.noise_hu) for c in selected} == expected
    for case in items:
        record = passage.metadata(case)
        assert record["name"] == case.fixture.name
        assert record["expected_count"] == record["physical_count"] == (1 if case.aperture_mm else 2)
        assert record["transform"] == case.transform
        assert record["reference_core_inset_mm"] == 1.0
        assert record["fine_spacing_mm"] == 0.125 and record["native_spacing_mm"] == 0.5
        assert record["clinical_accuracy"] == "NOT_MEASURED"
        assert "not clinical" in record["observation_model"]
        assert record["noise_seed"] == (1729 if case.noise_hu else None)


def test_geometry_topology_bounds_and_read_only_contract(items):
    for case in items:
        f = case.fixture
        shape = (32, 28, 24) if case.transform == "base" else (24, 28, 32)
        assert f.hu.shape == f.labels.shape == f.roi.shape == case.cores.shape == shape
        assert case.highres_labels.shape == tuple(size * 4 for size in shape)
        assert f.hu.dtype == np.float32 and f.labels.dtype == np.uint16
        assert case.highres_labels.dtype == np.uint8 and case.cores.dtype == np.uint16
        assert f.roi.dtype == bool and f.roi.all() and np.isfinite(f.hu).all()
        assert ndi.label(case.highres_labels > 0)[1] == f.components == (1 if case.aperture_mm else 2)
        assert set(np.unique(f.labels)) == set(range(f.components + 1))
        points = _points(f.labels > 0, f.affine)
        assert np.all(points >= [-5, -4, -3]) and np.all(points < [5, 4, 3])
        core_points = _points(case.cores > 0, f.affine)
        assert np.all(core_points >= [-4, -3, -2]) and np.all(core_points < [4, 3, 2])
        # Every chosen aperture is narrower than the 2 mm core inset diameter.
        assert np.all(np.abs(core_points[:, 0]) - case.wall_mm / 2 >= 1.0)
        for identifier in range(1, f.components + 1):
            assert np.any(case.cores == identifier)
            assert np.all(f.labels[case.cores == identifier] == identifier)
        for array in (f.hu, f.labels, f.roi, f.affine, case.highres_labels, case.cores):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 0


def test_reference_labels_follow_independent_physical_predicates(items):
    for case in items[:16]:
        for factor, labels in ((1, case.fixture.labels), (4, case.highres_labels)):
            edges = np.array([-8.0 + case.phase_mm, -7.0, -6.0])
            axes = [edge + (np.arange(size * factor) + 0.5) * 0.5 / factor
                    for edge, size in zip(edges, (32, 28, 24))]
            x, y, z = axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
            inside = ((x >= -5) & (x < 5) & (y >= -4) & (y < 4) & (z >= -3) & (z < 3))
            plate = (x >= -case.wall_mm / 2) & (x < case.wall_mm / 2)
            hole = ((y >= -case.aperture_mm / 2) & (y < case.aperture_mm / 2)
                    & (z >= -case.aperture_mm / 2) & (z < case.aperture_mm / 2))
            expected_free = inside & (~plate | hole)
            np.testing.assert_array_equal(labels > 0, expected_free)
            if not case.aperture_mm:
                np.testing.assert_array_equal(labels == 1, expected_free & (x < 0))
                np.testing.assert_array_equal(labels == 2, expected_free & (x >= 0))
        # Fine-grid volume agrees with the exact half-open physical construction.
        air_volume = 10 * 8 * 6 - case.wall_mm * (8 * 6 - case.aperture_mm ** 2)
        assert np.count_nonzero(case.highres_labels) * 0.125 ** 3 == air_volume
        if case.aperture_mm == 0.25:
            in_plate = (axes[0] >= -0.5) & (axes[0] < 0.5)
            assert np.all(np.count_nonzero(case.highres_labels[in_plate], axis=(1, 2)) == 4)


def test_observed_hu_uses_declared_blur_averaging_and_noise(items):
    for case in items[:16]:
        observed = np.where(case.highres_labels > 0, -900.0, 800.0)
        if case.sigma_mm:
            observed = ndi.gaussian_filter(observed, case.sigma_mm / 0.125,
                                           mode="constant", cval=800.0, truncate=4.0)
        expected = observed.reshape(32, 4, 28, 4, 24, 4).mean(axis=(1, 3, 5))
        if case.noise_hu:
            expected += np.random.default_rng(1729).normal(0, 25, (32, 28, 24))
        np.testing.assert_array_equal(case.fixture.hu, expected.astype(np.float32))
        if not case.noise_hu:
            assert case.fixture.hu.min() >= -900 and case.fixture.hu.max() <= 800


def test_reindexing_preserves_all_arrays_and_world_positions(items):
    for base, changed in zip(items[:16], items[16:]):
        for name in ("hu", "labels", "roi"):
            np.testing.assert_array_equal(getattr(changed.fixture, name),
                                          getattr(base.fixture, name).transpose(2, 1, 0)[:, ::-1, :])
        for name in ("highres_labels", "cores"):
            np.testing.assert_array_equal(getattr(changed, name),
                                          getattr(base, name).transpose(2, 1, 0)[:, ::-1, :])
        indices = np.array([[0, 0, 0], [31, 27, 23], [5, 11, 19], [16, 14, 12]])
        reindexed = np.column_stack((indices[:, 2], 27 - indices[:, 1], indices[:, 0]))
        np.testing.assert_allclose(indices @ base.fixture.affine[:3, :3].T + base.fixture.affine[:3, 3],
                                   reindexed @ changed.fixture.affine[:3, :3].T + changed.fixture.affine[:3, 3], atol=0, rtol=0)
        fine_indices = np.array([[0, 0, 0], [127, 111, 95], [25, 44, 61]])
        fine_reindexed = np.column_stack((fine_indices[:, 2], 111 - fine_indices[:, 1], fine_indices[:, 0]))
        fine_world = []
        for case, index in ((base, fine_indices), (changed, fine_reindexed)):
            # A fine voxel's center lies within its native voxel, not at its corner.
            affine = case.fixture.affine
            native_coordinates = (index + 0.5) / 4 - 0.5
            fine_world.append(native_coordinates @ affine[:3, :3].T + affine[:3, 3])
        np.testing.assert_array_equal(fine_world[0], fine_world[1])
        for identifier in range(1, base.fixture.components + 1):
            original = _points(base.fixture.labels == identifier, base.fixture.affine)
            transformed = _points(changed.fixture.labels == identifier, changed.fixture.affine)
            np.testing.assert_array_equal(np.array(sorted(map(tuple, original))), np.array(sorted(map(tuple, transformed))))


def test_repeatability_and_noise_do_not_change_truth(items):
    again = passage.cases()
    for first, repeated in zip(items, again):
        assert passage.metadata(first) == passage.metadata(repeated)
        for name in ("hu", "labels", "roi", "affine"):
            np.testing.assert_array_equal(getattr(first.fixture, name), getattr(repeated.fixture, name))
        np.testing.assert_array_equal(first.highres_labels, repeated.highres_labels)
        np.testing.assert_array_equal(first.cores, repeated.cores)
    for noisy in (c for c in items[:16] if c.noise_hu):
        noiseless = next(c for c in items[:16] if (c.wall_mm, c.aperture_mm, c.sigma_mm, c.phase_mm, c.noise_hu)
                         == (noisy.wall_mm, noisy.aperture_mm, 0.3, 0.0, 0.0))
        np.testing.assert_array_equal(noisy.fixture.labels, noiseless.fixture.labels)
        np.testing.assert_array_equal(noisy.highres_labels, noiseless.highres_labels)
        np.testing.assert_array_equal(noisy.cores, noiseless.cores)
        assert not np.array_equal(noisy.fixture.hu, noiseless.fixture.hu)
        difference = noisy.fixture.hu.astype(float) - noiseless.fixture.hu
        assert abs(difference.mean()) < 1 and 24 < difference.std() < 26
