"""Measurement controls from the sealed ridge recipe, not cell-count validation."""
from itertools import permutations, product

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage as ndi
from ethmoid_pipeline.counting.native_ridge_research import (
    RidgeError, measure_ridges, measure_ridge_enclosure_support,
    ridge_interface_endpoints,
)


def scene(kind, spacing=(.6, .6, .6)):
    hu = np.full((48, 48, 48), 800, np.float32)
    hu[5:43, 7:41, 7:41] = -900
    if kind == "weak-wall": hu[24, 7:41, 7:41] = 0
    elif kind == "open-fold": hu[24, 7:17, 7:41] = 0
    elif kind == "floating-disc": hu[24, 12:37, 12:37] = 0
    elif kind == "inclusion": hu[23:26, 22:25, 22:25] = 0
    elif kind == "opaque": hu[5:43, 7:41, 7:41] = 35
    elif kind == "bone-wall": hu[24, 7:41, 7:41] = 800
    return hu, hu < 200, np.diag([*spacing, 1.])


def ordinary_distance(free, affine):
    return ndi.distance_transform_edt(np.pad(free, 1), sampling=np.linalg.norm(affine[:3, :3], axis=0))[1:-1, 1:-1, 1:-1].astype(np.float32)


def test_weak_wall_measures_only_observed_material_and_preserves_inputs():
    hu, free, affine = scene("weak-wall")
    copies = [a.copy() for a in (hu, free, affine)]
    result = measure_ridges(hu, free, affine)
    expected = np.zeros_like(free)
    expected[24, 7:41, 7:41] = True
    for y, z in product((7, 40), repeat=2): expected[24, y, z] = False
    assert np.array_equal(result.candidate_ridge, expected)
    assert np.array_equal(result.anchored_ridge, expected)
    assert result.anchored_ridge.sum() == 1152
    assert result.distance_mm.dtype == np.float32
    assert result.distance_mm[24, 24, 24] == 0 < ordinary_distance(free, affine)[24, 24, 24]
    assert np.all(result.distance_mm[~free] == 0)
    assert all(np.array_equal(a, b) for a, b in zip((hu, free, affine), copies))


@pytest.mark.parametrize("kind", ["empty", "opaque", "bone-wall"])
def test_no_air_flanked_subthreshold_material_leaves_distance_unchanged(kind):
    hu, free, affine = scene(kind)
    result = measure_ridges(hu, free, affine)
    assert not result.candidate_ridge.any()
    assert not result.anchored_ridge.any()
    assert np.array_equal(result.distance_mm, ordinary_distance(free, affine))


@pytest.mark.parametrize("kind", ["floating-disc", "inclusion"])
def test_unattached_bright_objects_do_not_become_distance_barriers(kind):
    hu, free, affine = scene(kind)
    result = measure_ridges(hu, free, affine)
    assert result.candidate_ridge.any()
    assert not result.anchored_ridge.any()
    assert np.array_equal(result.distance_mm, ordinary_distance(free, affine))


def test_attached_open_fold_does_not_gain_a_closing_plane():
    hu, free, affine = scene("open-fold")
    result = measure_ridges(hu, free, affine)
    assert result.anchored_ridge.sum() == 336
    assert np.all(hu[result.anchored_ridge] == 0)
    assert not result.anchored_ridge[:, 17:, :].any()
    assert ndi.label(free & ~result.anchored_ridge)[1] == 1
    assert result.distance_mm[24, 30, 24] > 0


def test_bone_anchor_requires_a_face_neighbor_not_diagonal_contact():
    hu, free, affine = scene("floating-disc")
    hu[23, 11, 13] = 800; free[23, 11, 13] = False
    assert not measure_ridges(hu, free, affine).anchored_ridge.any()
    hu[24, 11, 13] = 800; free[24, 11, 13] = False
    result = measure_ridges(hu, free, affine)
    assert np.array_equal(result.anchored_ridge, result.candidate_ridge)


def test_native_crop_padding_keeps_edge_distance_finite():
    hu = np.full((5, 5, 5), -900, np.float32)
    result = measure_ridges(hu, np.ones_like(hu, bool), np.diag([2., 3., 4., 1.]))
    assert not result.anchored_ridge.any()
    assert result.distance_mm[0, 2, 2] == 2
    assert result.distance_mm[2, 2, 2] == 6


def test_anisotropic_signed_axis_geometry_and_domain_holes():
    hu, free, affine = scene("weak-wall", (.48, .65, .8))
    free[24, 15:19, 15:19] = False
    expected = measure_ridges(hu, free, affine)
    assert not expected.anchored_ridge[~free].any()
    for perm in permutations(range(3)):
        for flipped in product((False, True), repeat=3):
            transformed = hu.transpose(perm)
            domain = free.transpose(perm)
            candidate = expected.candidate_ridge.transpose(perm)
            anchored = expected.anchored_ridge.transpose(perm)
            distance = expected.distance_mm.transpose(perm)
            matrix = affine.copy(); matrix[:3, :3] = affine[:3, list(perm)]
            for axis, flip in enumerate(flipped):
                if flip:
                    transformed, domain, candidate, anchored, distance = [np.flip(a, axis) for a in (transformed, domain, candidate, anchored, distance)]
                    matrix[:3, 3] += matrix[:3, axis] * (transformed.shape[axis]-1)
                    matrix[:3, axis] *= -1
            actual = measure_ridges(transformed, domain, matrix)
            assert np.array_equal(actual.candidate_ridge, candidate)
            assert np.array_equal(actual.anchored_ridge, anchored)
            assert np.allclose(actual.distance_mm, distance, rtol=0, atol=1e-6)


@pytest.mark.parametrize("change,code", [
    ("shape", "RIDGE_INPUT_INVALID"), ("nan", "RIDGE_INPUT_INVALID"),
    ("numeric-mask", "RIDGE_INPUT_INVALID"), ("bone-in-domain", "RIDGE_INPUT_INVALID"),
    ("shear", "RIDGE_GEOMETRY_INVALID"), ("zero-spacing", "RIDGE_GEOMETRY_INVALID"),
    ("bad-width", "RIDGE_CONFIG_INVALID"), ("boolean-threshold", "RIDGE_CONFIG_INVALID"),
    ("ragged", "RIDGE_INPUT_INVALID"),
])
def test_invalid_inputs_have_fixed_errors(change, code):
    hu, free, affine = scene("weak-wall"); options = {}
    if change == "shape": free = free[:-1]
    elif change == "nan": hu[20, 20, 20] = np.nan
    elif change == "numeric-mask": free = free.astype(np.int8)
    elif change == "bone-in-domain": free[0, 0, 0] = True
    elif change == "shear": affine[0, 1] = .2
    elif change == "zero-spacing": affine[0, 0] = 0
    elif change == "bad-width": options["plate_half_width_mm"] = 0
    elif change == "boolean-threshold": options["air_hu"] = True
    elif change == "ragged": hu = [[1], [1, 2]]
    with pytest.raises(RidgeError, match="^"+code+"$"):
        measure_ridges(hu, free, affine, **options)


def test_interface_background_same_label_and_empty_ridge_supply_no_support():
    labels = np.ones((7, 7, 7), np.int32)
    ridge = np.ones_like(labels, bool)
    search = np.ones_like(ridge)
    for values, measured in ((labels * 0, ridge), (labels, ridge),
                             (labels * 0 + np.indices(labels.shape)[0], ridge * False)):
        result = measure_ridge_enclosure_support(values, measured, search, np.eye(4))
        assert not result.eligible_ridge.any()
        assert not result.non_roi_support.any()
    labels[:3] = 0
    assert not ridge_interface_endpoints(labels, ridge).any()


def test_only_measured_positive_face_endpoints_are_eligible_without_wrap():
    labels = np.zeros((6, 5, 4), np.int32)
    labels[1:3, 1:4, 1:3] = 1
    labels[3:5, 1:4, 1:3] = 2
    ridge = np.zeros_like(labels, bool)
    ridge[2, 2, 1] = True
    ridge[3, 2, 2] = True
    ridge[0, 0, 0] = ridge[-1, 0, 0] = True
    labels[0, 0, 0], labels[-1, 0, 0] = 3, 4
    expected = np.zeros_like(ridge)
    expected[2, 2, 1] = expected[3, 2, 2] = True
    assert np.array_equal(ridge_interface_endpoints(labels, ridge), expected)
    # Distinct diagonal IDs cannot make an interface endpoint eligible.
    labels.fill(0); ridge.fill(False)
    labels[2, 2, 2], labels[3, 3, 2] = 1, 2
    ridge[2, 2, 2] = ridge[3, 3, 2] = True
    assert not ridge_interface_endpoints(labels, ridge).any()


def test_open_fold_in_one_merged_proposal_cannot_supply_enclosure_support():
    hu, free, affine = scene("open-fold")
    ridge = measure_ridges(hu, free, affine).anchored_ridge
    assert ridge.any()
    labels = free.astype(np.int32)
    result = measure_ridge_enclosure_support(labels, ridge, free, affine)
    assert not result.eligible_ridge.any()
    assert not result.non_roi_support.any()


def test_support_excludes_exact_roi_edge_but_does_not_invent_domain_clipping():
    labels = np.zeros((9, 9, 9), np.int32)
    labels[3, 4, 4], labels[4, 4, 4] = 1, 2
    ridge = np.zeros_like(labels, bool); ridge[3, 4, 4] = True
    search = np.zeros_like(ridge); search[3:8, 1:8, 1:8] = True
    copies = [a.copy() for a in (labels, ridge, search)]
    result = measure_ridge_enclosure_support(labels, ridge, search,
                                             np.diag([.6, .6, .6, 1.]))
    roi_edge = search & ~ndi.binary_erosion(search)
    assert result.eligible_ridge[3, 4, 4]
    assert not result.non_roi_support[roi_edge].any()
    assert result.non_roi_support[4, 4, 4]  # Interior proposal surface may use support.
    assert result.non_roi_support[2, 4, 4]  # Exact recipe is not clipped to search.
    assert not result.non_roi_support[5, 4, 4]  # 1.2 mm exceeds the 0.8 mm radius.
    assert all(np.array_equal(a, b) for a, b in zip((labels, ridge, search), copies))


@pytest.mark.parametrize("spacing,radius", [
    ((.2, .3, .4), .8), ((.6, .6, .9), .9 * 1.01), ((2., .6, .6), 1.01),
])
def test_support_uses_physical_spacing_and_pinned_radius(spacing, radius):
    labels = np.zeros((7, 7, 7), np.int32)
    labels[3, 3, 3], labels[4, 3, 3] = 1, 2
    ridge = np.zeros_like(labels, bool); ridge[3, 3, 3] = True
    # Empty search has no ROI edge. This also distinguishes the EDT empty-input
    # guard from clipping the support to the domain.
    search = np.zeros_like(ridge)
    actual = measure_ridge_enclosure_support(labels, ridge, search,
                                            np.diag([*spacing, 1.]))
    coordinates = np.indices(labels.shape).transpose(1, 2, 3, 0)
    distances = np.linalg.norm((coordinates - [3, 3, 3]) * spacing, axis=-1)
    assert actual.near_radius_mm == radius
    assert np.array_equal(actual.non_roi_support, distances <= radius)


def test_support_matches_face_oracle_under_all_signed_axis_transforms():
    labels = np.zeros((7, 8, 9), np.int32)
    labels[1:4, 1:7, 1:8] = 1
    labels[4:6, 1:7, 1:8] = 2
    ridge = np.zeros_like(labels, bool); ridge[3, 2:6, 2:7] = True
    ridge[4, 4, 4] = True
    search = labels > 0; search[2, 4, 4] = False
    spacing = np.array([.48, .65, .8]); affine = np.diag([*spacing, 1.])
    expected_endpoints = np.zeros_like(ridge)
    for point in np.ndindex(labels.shape):
        if not ridge[point] or labels[point] == 0:
            continue
        for axis, offset in product(range(3), (-1, 1)):
            neighbor = list(point); neighbor[axis] += offset
            if (0 <= neighbor[axis] < labels.shape[axis]
                    and 0 < labels[tuple(neighbor)] != labels[point]):
                expected_endpoints[point] = True
    radius = min(max(float(spacing.max()) * 1.01, .8), 1.01)
    points = np.argwhere(expected_endpoints)
    coordinates = np.indices(labels.shape).transpose(1, 2, 3, 0)
    distances = np.linalg.norm((coordinates[..., None, :] - points) * spacing,
                               axis=-1).min(axis=-1)
    expected_support = ((distances <= radius)
                        & ~(search & ~ndi.binary_erosion(search)))
    for perm in permutations(range(3)):
        for flipped in product((False, True), repeat=3):
            arrays = [a.transpose(perm) for a in
                      (labels, ridge, search, expected_endpoints, expected_support)]
            matrix = affine.copy(); matrix[:3, :3] = affine[:3, list(perm)]
            for axis, flip in enumerate(flipped):
                if flip:
                    arrays = [np.flip(a, axis) for a in arrays]
                    matrix[:3, 3] += matrix[:3, axis] * (arrays[0].shape[axis] - 1)
                    matrix[:3, axis] *= -1
            actual = measure_ridge_enclosure_support(*arrays[:3], matrix)
            assert np.array_equal(actual.eligible_ridge, arrays[3])
            assert np.array_equal(actual.non_roi_support, arrays[4])
            assert actual.near_radius_mm == radius


@pytest.mark.parametrize("change,code", [
    ("negative-label", "RIDGE_INPUT_INVALID"), ("float-label", "RIDGE_INPUT_INVALID"),
    ("numeric-ridge", "RIDGE_INPUT_INVALID"), ("shape", "RIDGE_INPUT_INVALID"),
    ("numeric-search", "RIDGE_INPUT_INVALID"), ("empty-axis", "RIDGE_INPUT_INVALID"),
    ("shear", "RIDGE_GEOMETRY_INVALID"), ("ragged", "RIDGE_INPUT_INVALID"),
])
def test_support_rejects_invalid_labels_masks_and_geometry(change, code):
    labels = np.zeros((3, 4, 5), np.int32)
    ridge, search = np.zeros_like(labels, bool), np.ones_like(labels, bool)
    affine = np.eye(4)
    if change == "negative-label": labels[0, 0, 0] = -1
    elif change == "float-label": labels = labels.astype(np.float32)
    elif change == "numeric-ridge": ridge = ridge.astype(np.uint8)
    elif change == "shape": search = search[:-1]
    elif change == "numeric-search": search = search.astype(np.uint8)
    elif change == "empty-axis": labels, ridge, search = [a[:0] for a in (labels, ridge, search)]
    elif change == "shear": affine[0, 1] = .1
    elif change == "ragged": labels = [[1], [1, 2]]
    with pytest.raises(RidgeError, match="^" + code + "$"):
        measure_ridge_enclosure_support(labels, ridge, search, affine)
