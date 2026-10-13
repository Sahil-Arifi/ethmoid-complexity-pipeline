from pathlib import Path
import sys

import numpy as np
import pytest

ndi = pytest.importorskip("scipy.ndimage")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ridge_stress_cases as stress
from synthetic_counting_experiment import array_hash, construction_counts


@pytest.fixture(scope="module")
def items():
    return stress.cases()


def physical_points(mask, affine):
    return np.argwhere(mask) @ affine[:3, :3].T + affine[:3, 3]


def test_fixed_design_topology_and_read_only_contract(items):
    assert len(items) == len({c.fixture.name for c in items}) == 12
    assert sum(c.expected_abstention for c in items) == 8
    for case in items:
        f = case.fixture
        assert f.hu.dtype == np.float32 and f.labels.dtype == np.uint16 and f.roi.dtype == bool
        assert f.hu.shape == f.roi.shape == f.labels.shape == case.cores.shape
        assert np.isfinite(f.hu).all() and f.roi.all()
        assert ndi.label(f.labels > 0)[1] == f.components
        assert ndi.label(case.highres_labels > 0)[1] == f.components
        assert construction_counts(f)["total"] == f.components
        for identifier in range(1, f.components + 1):
            assert np.any(case.cores == identifier)
            assert ndi.label(case.cores == identifier)[1] == 1
            assert np.all(f.labels[case.cores == identifier] == identifier)
        for array in (f.hu, f.labels, f.roi, f.affine, case.highres_labels, case.cores):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 0


def test_twins_have_identical_detector_inputs_but_different_truth(items):
    for transform in ("base", "swap_xz_flip_y"):
        for hu in (35, 100):
            bone, content = [c for c in items if c.transform == transform and c.observed_sheet_hu == hu]
            for name in ("hu", "roi", "affine"):
                assert array_hash(getattr(bone.fixture, name)) == array_hash(getattr(content.fixture, name))
                np.testing.assert_array_equal(getattr(bone.fixture, name), getattr(content.fixture, name))
            assert array_hash(bone.fixture.labels) != array_hash(content.fixture.labels)
            assert array_hash(bone.cores) != array_hash(content.cores)
            assert (bone.fixture.components, content.fixture.components) == (2, 1)
            assert bone.equivalence_group == content.equivalence_group
            assert stress.metadata(bone)["expected_abstention"] and stress.metadata(content)["expected_abstention"]


def test_exact_central_voxel_occupancy_and_independent_center_truth(items):
    for case in items[:4]:
        f = case.fixture
        fraction = (case.observed_sheet_hu + 900) / 1700
        assert fraction * 800 + (1 - fraction) * -900 == pytest.approx(case.observed_sheet_hu)
        sheet = f.hu == case.observed_sheet_hu
        points = physical_points(sheet, f.affine)
        np.testing.assert_allclose(points[:, 0], 0, atol=1e-12)
        assert len(np.unique(np.argwhere(sheet)[:, 0])) == 1
        assert fraction * stress.SPACING_MM[0] < stress.SPACING_MM[0]
        if case.construction == "bone":
            assert case.wall_mm == pytest.approx(fraction * stress.SPACING_MM[0])
            assert not f.labels[sheet].any()
            assert ndi.label(f.hu < 200)[1] == 1
        else:
            assert np.all(f.labels[sheet] == 1)


def test_reindexing_preserves_world_voxels_cores_counts_and_centroids(items):
    for base, changed in zip(items[:6], items[6:]):
        for name in ("hu", "labels", "roi"):
            np.testing.assert_array_equal(getattr(changed.fixture, name),
                                          getattr(base.fixture, name).transpose(2, 1, 0)[:, ::-1, :])
        np.testing.assert_array_equal(changed.cores, base.cores.transpose(2, 1, 0)[:, ::-1, :])
        np.testing.assert_array_equal(changed.highres_labels, base.highres_labels.transpose(2, 1, 0)[:, ::-1, :])
        assert construction_counts(base.fixture) == construction_counts(changed.fixture)
        for identifier in range(1, base.fixture.components + 1):
            original = physical_points(base.fixture.labels == identifier, base.fixture.affine)
            transformed = physical_points(changed.fixture.labels == identifier, changed.fixture.affine)
            np.testing.assert_allclose(original.mean(axis=0), transformed.mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(np.array(sorted(map(tuple, original))),
                                       np.array(sorted(map(tuple, transformed))), atol=1e-12)


def test_cores_have_at_least_one_mm_physical_margin(items):
    for case in items:
        points = physical_points(case.cores > 0, case.fixture.affine)
        assert np.all(stress.HALF_EXTENT_MM - np.abs(points) >= 1 - 1e-12)
        if case.construction == "bone":
            assert np.all(np.abs(points[:, 0]) - case.wall_mm / 2 >= 1 - 1e-12)
        elif case.construction == "fragment":
            lower, upper = np.array(stress.FRAGMENT_BOUNDS_MM).T
            distance = np.linalg.norm(np.maximum(np.maximum(lower - points, points - upper), 0), axis=1)
            assert np.all(distance >= 1 - 1e-12)
        elif case.construction == "opening":
            near_plane = np.abs(points[:, 0]) < 1.25
            aperture_margin = np.minimum(1.2 - np.abs(points[near_plane, 1] - 0.15),
                                         1.2 - np.abs(points[near_plane, 2]))
            assert np.all(aperture_margin >= 1 - 1e-12)


def test_control_constructions_and_metadata(items):
    opening, fragment = items[4:6]
    x, y, z = stress._coordinates()
    center = (np.argmin(np.abs(x)), np.argmin(np.abs(y)), np.argmin(np.abs(z)))
    assert opening.fixture.hu[center] == -900
    assert opening.fixture.labels[center] == 1
    assert not opening.expected_abstention and not fragment.expected_abstention
    for case in items:
        record = stress.metadata(case)
        assert record["name"] == case.fixture.name
        assert record["physical_count"] == case.fixture.components
        assert record["transform"] == case.transform
        assert record["physical_truth"] and record["reference_core_inset_mm"] == 1
    # The fragment cannot divide either the coarse or independently sampled interior.
    assert ndi.label(fragment.fixture.labels > 0)[1] == ndi.label(fragment.highres_labels > 0)[1] == 1
