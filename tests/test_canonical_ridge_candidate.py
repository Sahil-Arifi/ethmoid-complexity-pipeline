"""Exact reindexing regression for the synthetic ridge candidate."""
from inspect import signature
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from canonical_ridge_candidate import count_with_canonical_ridges
from native_passage_cases import _base
from ethmoid_pipeline.counting.engine import CountingError


REINDEXINGS = list(product(permutations(range(3)), product((1, -1), repeat=3)))


def _transform(array, axes, signs):
    return array.transpose(axes)[tuple(slice(None, None, sign) for sign in signs)].copy()


def _reindexed_affine(affine, shape, axes, signs):
    mapping = np.zeros((4, 4), dtype=float)
    mapping[3, 3] = 1
    for new_axis, (old_axis, sign) in enumerate(zip(axes, signs)):
        mapping[old_axis, new_axis] = sign
        if sign < 0:
            mapping[old_axis, 3] = shape[old_axis] - 1
    return affine @ mapping


@pytest.fixture(scope='module', params=('axis_aligned', 'anisotropic_oblique',
                                     'anisotropic_oblique_strong_wall'))
def plateau(request):
    fixture = _base(1 if request.param.endswith('strong_wall') else .125, 0, 0).fixture
    hu, roi, affine = fixture.hu.copy(), fixture.roi.copy(), fixture.affine.copy()
    if request.param.startswith('anisotropic_oblique'):
        c = np.sqrt(.5)
        rotation = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1.]])
        affine[:3, :3] = rotation @ np.diag([.4, .7, 1.1])
        affine[:3, 3] = np.array([2.3, -1.7, .6]) - affine[:3, :3] @ ((np.array(hu.shape) - 1) / 2)
    for array in (hu, roi, affine):
        array.setflags(write=False)
    expected = count_with_canonical_ridges(hu, roi, affine)
    if request.param != 'anisotropic_oblique':
        assert len(expected['cells']) > 0
    assert expected['diagnostics']['final_basin_count'] > 1
    return hu, roi, affine, expected


@pytest.mark.parametrize('axes,signs', REINDEXINGS,
                         ids=[f'axes-{a}-signs-{s}' for a, s in REINDEXINGS])
def test_all_signed_reindexings_preserve_exact_proposals_and_physical_registry(plateau, axes, signs):
    hu, roi, affine, expected = plateau
    changed_inputs = (_transform(hu, axes, signs), _transform(roi, axes, signs),
                      _reindexed_affine(affine, hu.shape, axes, signs))
    before = [array.copy() for array in changed_inputs]
    for array in changed_inputs:
        array.setflags(write=False)
    actual = count_with_canonical_ridges(*changed_inputs)
    for array, original in zip(changed_inputs, before):
        np.testing.assert_array_equal(array, original)
    for key in ('partitions', 'cavities', 'air', 'component_bridge'):
        np.testing.assert_array_equal(actual[key], _transform(expected[key], axes, signs), err_msg=key)
    for key in ('counts', 'status', 'withheld_reasons', 'experimental_count_available',
                'initial_basin_count', 'final_basin_count', 'clinical_accuracy', 'component_bridge_evidence'):
        assert actual['diagnostics'][key] == expected['diagnostics'][key], key
    assert len(actual['cells']) == len(expected['cells'])
    for original, changed in zip(expected['cells'], actual['cells']):
        np.testing.assert_allclose(changed['centroid_ras_mm'], original['centroid_ras_mm'],
                                   atol=1e-8, rtol=0)
        assert {k: v for k, v in changed.items() if k != 'centroid_ras_mm'} == {
            k: v for k, v in original.items() if k != 'centroid_ras_mm'}


@pytest.mark.parametrize('defect', ('shape', 'nan', 'inf', 'zero_spacing',
                                   'singular', 'shear', 'non_affine_bottom_row'))
def test_invalid_geometry_is_rejected_before_counting(defect):
    fixture = _base(.125, 0, 0).fixture
    affine = fixture.affine.copy()
    if defect == 'shape':
        affine = affine[:3]
    elif defect == 'nan':
        affine[0, 3] = np.nan
    elif defect == 'inf':
        affine[1, 1] = np.inf
    elif defect == 'zero_spacing':
        affine[:3, 0] = 0
    elif defect == 'singular':
        affine[:3, 1] = affine[:3, 0]
    elif defect == 'shear':
        affine[0, 1] = .2
    else:
        affine[3, 0] = .1
    with pytest.raises(CountingError) as error:
        count_with_canonical_ridges(fixture.hu, fixture.roi, affine)
    assert error.value.code == 'INVALID_GEOMETRY'


def test_entry_point_accepts_observation_only():
    assert tuple(signature(count_with_canonical_ridges).parameters) == ('hu', 'roi', 'affine')


@pytest.mark.parametrize('defect', ('nonnumeric', 'ragged', 'nonconvertible', 'overflow'))
def test_affine_conversion_errors_are_reported_as_invalid_geometry(defect, monkeypatch):
    fixture = _base(.125, 0, 0).fixture
    affine = fixture.affine.astype(object)
    if defect == 'nonnumeric':
        affine[0, 0] = 'bad'
    elif defect == 'ragged':
        affine = [[1, 0], [0]]
    elif defect == 'nonconvertible':
        affine[0, 0] = object()
    else:
        affine[0, 0] = 10 ** 1000
    def unexpected_proposal(*args):
        pytest.fail('Invalid geometry reached proposal generation.')
    monkeypatch.setattr('canonical_ridge_candidate.count_with_ridges', unexpected_proposal)
    with pytest.raises(CountingError) as error:
        count_with_canonical_ridges(fixture.hu, fixture.roi, affine)
    assert error.value.code == 'INVALID_GEOMETRY'


@pytest.mark.parametrize('defect', ('nan_hu', 'complex_hu', 'integer_roi', 'empty_roi', 'empty_grid', 'complex_affine'))
def test_invalid_observations_are_rejected(defect):
    fixture = _base(.125, 0, 0).fixture
    hu, roi, affine = fixture.hu.copy(), fixture.roi.copy(), fixture.affine.copy()
    if defect == 'nan_hu':
        hu.flat[0] = np.nan
    elif defect == 'complex_hu':
        hu = hu.astype(complex)
    elif defect == 'integer_roi':
        roi = roi.astype(int)
    elif defect == 'empty_roi':
        roi[:] = False
    elif defect == 'empty_grid':
        hu, roi = hu[:0], roi[:0]
    else:
        affine = affine.astype(complex)
        affine[0, 0] += 1j
    with pytest.raises(CountingError):
        count_with_canonical_ridges(hu, roi, affine)


@pytest.mark.parametrize('reference_key', ('labels', 'cores', 'expected_count', 'physical_count', 'metadata'))
def test_reference_arguments_are_rejected(reference_key):
    fixture = _base(.125, 0, 0).fixture
    with pytest.raises(TypeError):
        count_with_canonical_ridges(fixture.hu, fixture.roi, fixture.affine, **{reference_key: None})
