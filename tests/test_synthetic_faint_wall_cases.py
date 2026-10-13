"""Physical challenge construction checks; no detector is invoked here."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def cases():
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    return experiment.challenge_cases()


def test_challenge_has_exactly_predeclared_physical_pairs_and_controls(cases):
    assert len(cases) == len({case.fixture.name for case in cases}) == 32
    assert sum(case.wall_mm > 0 for case in cases) == 16
    assert sum(case.wall_mm == 0 for case in cases) == 16
    for case in cases:
        expected = 2 if case.wall_mm else 1
        assert ndi.label(case.highres_labels > 0)[1] == expected
        for identifier in range(1, expected + 1):
            assert ndi.label(case.highres_labels == identifier)[1] == 1
            assert np.any(case.cores == identifier)
            assert ndi.label(case.cores == identifier)[1] == 1
            assert np.all(case.fixture.labels[case.cores == identifier] == identifier)


def physical_coordinates(case):
    grid = np.indices(case.fixture.labels.shape)
    return np.einsum("ij,jxyz->ixyz", case.fixture.affine[:3, :3], grid) + case.fixture.affine[:3, 3, None, None, None]


def test_rows_materials_and_physical_volumes_match_frozen_table(cases):
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    pair_rows = [(0.375, 0.125, 0.125), (0.375, 0.375, 0.125),
                 (0.75, 0.375, 0.375), (0.75, 0.625, 0.375)]
    for construction in ("off_center_flat", "oblique", "curved", "axis_permutation"):
        selected = [case for case in cases if case.construction == construction]
        assert [(case.wall_mm, case.sigma_mm, case.phase_mm) for case in selected] == pair_rows
    for construction in ("no_divider", "perforated_divider", "attached_ridge", "isolated_fragment",
                         "content_ramp", "gaussian_noise", "soft_sheet_35", "soft_sheet_100"):
        selected = [case for case in cases if case.construction == construction]
        assert [(case.sigma_mm, case.phase_mm) for case in selected] == [(0.125, 0.125), (0.375, 0.375)]
    for case in cases:
        recorded = experiment.challenge_record(case)
        volume = 1536 - case.wall_mm * 96 if case.wall_mm else {
            "perforated_divider": 1536 - 0.75 * (12 * 8 - 3 * 3),
            "attached_ridge": 1536 - 0.75 * 4 * 8,
            "isolated_fragment": 1536 - 0.75 * 4 * 4,
        }.get(case.construction, 1536)
        assert recorded["physical_cavity_volume_mm3"] == volume
        assert recorded["highres_cavity_volume_mm3"] == volume
        assert recorded["highres_physical_components"] == (2 if case.wall_mm else 1)
        assert "expected_air_volume_mm3" not in recorded


def test_coarse_truth_is_direct_physical_predicate_and_cores_are_conservative(cases):
    for case in cases:
        x, y, z = physical_coordinates(case)
        inside = (x >= -8) & (x < 8) & (y >= -6) & (y < 6) & (z >= -4) & (z < 4)
        interior = (x >= -7) & (x < 7) & (y >= -5) & (y < 5) & (z >= -3) & (z < 3)
        expected = np.zeros(case.fixture.labels.shape, dtype=np.uint16)
        core = np.zeros_like(expected)
        if case.wall_mm:
            if case.construction == "oblique":
                distance, slope = x + 0.75 - y / 4, 0.25
            elif case.construction == "curved":
                distance, slope = x - 0.5 - (y ** 2 - 12) / 25, 0.48
            else:
                distance, slope = x - 1, 0
            half_width = case.wall_mm / 2
            expected[inside & (distance < -half_width)] = 1
            expected[inside & (distance >= half_width)] = 2
            margin = np.sqrt(1 + slope ** 2)
            core[interior & (distance < -half_width - margin)] = 1
            core[interior & (distance >= half_width + margin)] = 2
        else:
            if case.construction == "perforated_divider":
                bone = (x >= -0.375) & (x < 0.375) & ((np.abs(y) >= 1.5) | (np.abs(z) >= 1.5))
                neighborhood = (np.abs(x) <= 1.375) & ((np.abs(y) >= 0.5) | (np.abs(z) >= 0.5))
            elif case.construction == "attached_ridge":
                bone = (x >= -0.375) & (x < 0.375) & (y < -2)
                neighborhood = (np.abs(x) <= 1.375) & (y <= -1)
            elif case.construction == "isolated_fragment":
                bone = (x >= -0.375) & (x < 0.375) & (np.abs(y) < 2) & (np.abs(z) < 2)
                neighborhood = (np.abs(x) <= 1.375) & (np.abs(y) <= 3) & (np.abs(z) <= 3)
            else:
                bone = neighborhood = np.zeros_like(inside)
            expected[inside & ~bone] = 1
            core[interior & ~neighborhood] = 1
        np.testing.assert_array_equal(case.fixture.labels, expected)
        np.testing.assert_array_equal(case.cores, core)
        for identifier in range(1, 3 if case.wall_mm else 2):
            assert ndi.label(case.fixture.labels == identifier)[1] == 1


@pytest.mark.parametrize("construction", ["off_center_flat", "oblique", "curved", "no_divider",
                                         "perforated_divider", "attached_ridge", "isolated_fragment",
                                         "soft_sheet_35", "soft_sheet_100"])
def test_fine_truth_independently_obeys_physical_predicates(cases, construction):
    case = next(case for case in cases if case.construction == construction)
    x = (-12 + case.phase_mm + (np.arange(192) + 0.5) * 0.125)[:, None, None]
    y = (-10 + (np.arange(160) + 0.5) * 0.125)[None, :, None]
    z = (-8 + (np.arange(128) + 0.5) * 0.125)[None, None, :]
    inside = (x >= -8) & (x < 8) & (y >= -6) & (y < 6) & (z >= -4) & (z < 4)
    reference = np.zeros((192, 160, 128), dtype=np.uint8)
    if case.wall_mm:
        distance = (x - 1 if construction == "off_center_flat" else
                    x + 0.75 - y / 4 if construction == "oblique" else
                    x - 0.5 - (y ** 2 - 12) / 25)
        reference[inside & (distance < -case.wall_mm / 2)] = 1
        reference[inside & (distance >= case.wall_mm / 2)] = 2
    else:
        bone = (x >= -0.375) & (x < 0.375)
        if construction == "perforated_divider":
            bone = bone & ((np.abs(y) >= 1.5) | (np.abs(z) >= 1.5))
        elif construction == "attached_ridge":
            bone = bone & (y < -2)
        elif construction == "isolated_fragment":
            bone = bone & (np.abs(y) < 2) & (np.abs(z) < 2)
        else:
            bone = np.zeros_like(inside)
        reference[inside & ~bone] = 1
    np.testing.assert_array_equal(case.highres_labels, reference)


def convolution_oracle(material, sigma):
    radius = int(4 * sigma / 0.125 + 0.5)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets * 0.125 / sigma) ** 2)
    weights /= weights.sum()
    blurred = np.convolve(np.pad(material, radius, constant_values=800), weights, mode="valid")
    return blurred.reshape(48, 4).mean(axis=1)


@pytest.mark.parametrize("construction", ["off_center_flat", "content_ramp", "soft_sheet_35", "soft_sheet_100"])
def test_image_formation_matches_independent_convolution_then_block_mean(cases, construction):
    for case in (case for case in cases if case.construction == construction):
        fine_x = -12 + case.phase_mm + (np.arange(192) + 0.5) * 0.125
        inside = (fine_x >= -8) & (fine_x < 8)
        material = np.full(192, 800.0)
        material[inside] = -900
        if case.wall_mm:
            material[inside & (fine_x >= 1 - case.wall_mm / 2) & (fine_x < 1 + case.wall_mm / 2)] = 800
        elif construction == "content_ramp":
            material[inside] = -900 + 300 * (fine_x[inside] + 8) / 16
        else:
            material[inside & (fine_x >= -0.375) & (fine_x < 0.375)] = 35 if construction == "soft_sheet_35" else 100
        np.testing.assert_allclose(case.fixture.hu[:, 20, 16], convolution_oracle(material, case.sigma_mm), rtol=0, atol=1e-3)
        np.testing.assert_array_equal(case.fixture.hu[:, 0, 0], np.full(48, 800, dtype=np.float32))


def test_declared_whole_field_noise_is_added_after_image_averaging(cases):
    for row, seed in enumerate((2026101301, 2026101302)):
        noisy = [case for case in cases if case.construction == "gaussian_noise"][row]
        unnoised = [case for case in cases if case.construction == "no_divider"][row]
        noise = np.random.Generator(np.random.PCG64(seed)).normal(0, 75, size=(48, 40, 32))
        np.testing.assert_allclose(noisy.fixture.hu, unnoised.fixture.hu.astype(np.float64) + noise, rtol=0, atol=1e-4)
        assert noisy.noise_seed == seed
        assert np.any(noisy.fixture.hu[:, 0, 0] != 800)
        for name in ("labels", "roi", "affine"):
            np.testing.assert_array_equal(getattr(noisy.fixture, name), getattr(unnoised.fixture, name))
        np.testing.assert_array_equal(noisy.highres_labels, unnoised.highres_labels)
        np.testing.assert_array_equal(noisy.cores, unnoised.cores)


def test_axis_permutation_preserves_physics_with_matching_affine(cases):
    flat = [case for case in cases if case.construction == "off_center_flat"]
    permuted = [case for case in cases if case.construction == "axis_permutation"]
    for first, second in zip(flat, permuted):
        assert second.fixture.hu.shape == (40, 48, 32)
        assert second.highres_labels.shape == (160, 192, 128)
        for name in ("hu", "labels", "roi"):
            np.testing.assert_array_equal(getattr(second.fixture, name), np.swapaxes(getattr(first.fixture, name), 0, 1))
        for name in ("highres_labels", "cores"):
            np.testing.assert_array_equal(getattr(second, name), np.swapaxes(getattr(first, name), 0, 1))
        expected_affine = first.fixture.affine.copy()
        expected_affine[:, [0, 1]] = expected_affine[:, [1, 0]]
        np.testing.assert_array_equal(second.fixture.affine, expected_affine)
        np.testing.assert_array_equal(physical_coordinates(second)[:, 10, 20, 16], physical_coordinates(first)[:, 20, 10, 16])


def test_half_open_enclosure_divider_and_strict_aperture_boundaries(cases):
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    coordinates = (np.array([-8, 0.8125, 1.1875, 8]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_array_equal(experiment._reference(coordinates, "off_center_flat", 0.375)[:, 0, 0], [1, 0, 2, 0])
    aperture = (np.array([0.0]), np.array([-1.5, -1.499, 0, 1.499, 1.5]), np.array([0.0]))
    np.testing.assert_array_equal(experiment._reference(aperture, "perforated_divider", 0)[0, :, 0], [0, 1, 1, 1, 0])
    fragment = (np.array([0.0]), np.array([-2.0, -1.999, 1.999, 2.0]), np.array([0.0]))
    np.testing.assert_array_equal(experiment._reference(fragment, "isolated_fragment", 0)[0, :, 0], [1, 0, 0, 1])
    flat = next(case for case in cases if case.construction == "off_center_flat")
    assert not np.array_equal(flat.fixture.labels, flat.highres_labels[::4, ::4, ::4])


def test_seeded_challenge_reproduction_preserves_all_six_array_hashes(cases):
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    for row, original in enumerate(case for case in cases if case.construction == "gaussian_noise"):
        repeated = experiment.make_challenge_case("gaussian_noise", row)
        assert experiment.challenge_record(repeated) == experiment.challenge_record(original)


def test_records_have_real_shapes_read_only_inputs_six_hashes_and_json_roundtrip(cases):
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    from synthetic_counting_experiment import array_hash
    records = [experiment.challenge_record(case) for case in cases]
    assert records == json.loads(json.dumps(records, allow_nan=False))
    for case, recorded in zip(cases, records):
        assert recorded["shape"] == list(case.fixture.hu.shape)
        assert recorded["highres_shape"] == list(case.highres_labels.shape)
        assert recorded["construction_counts"]["total"] == (2 if case.wall_mm else 1)
        assert set(recorded["hashes"]) == {"hu", "labels", "roi", "affine", "highres_labels", "cores"}
        assert case.fixture.hu.dtype == np.float32
        assert case.fixture.labels.dtype == case.cores.dtype == np.uint16
        assert case.highres_labels.dtype == np.uint8
        assert case.fixture.roi.dtype == bool
        for name in ("hu", "labels", "roi", "affine", "highres_labels", "cores"):
            array = getattr(case, name) if name in ("highres_labels", "cores") else getattr(case.fixture, name)
            assert not array.flags.writeable
            assert recorded["hashes"][name] == array_hash(array)


@pytest.mark.parametrize("construction,row", [("unknown", 0), ("oblique", 4), ("no_divider", 2),
                                               ("curved", -1), ("no_divider", True), ("oblique", "0")])
def test_generator_rejects_parameters_outside_fixed_challenge(construction, row):
    experiment = importlib.import_module("synthetic_faint_wall_cases")
    with pytest.raises(ValueError):
        experiment.make_challenge_case(construction, row)
