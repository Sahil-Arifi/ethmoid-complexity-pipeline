"""Conditional synthetic inference, including its known float32 falsifier."""
from decimal import Decimal, localcontext
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
from saturated_barrier_candidate import count_with_saturated_barrier


def test_sealed_strong_wall_recovers_two_only_under_explicit_model():
    f = _base(1.0, 0.0, 0.0).fixture
    baseline = count_with_canonical_ridges(f.hu, f.roi, f.affine)
    assert baseline['diagnostics']['status'] == 'withheld'
    actual = count_with_saturated_barrier(f.hu, f.roi, f.affine)
    assert actual['diagnostics']['counts'] == {'left': 1, 'right': 1, 'total': 2}
    assert actual['diagnostics']['status'] == 'accepted'
    assert actual['diagnostics']['conditional_model'] == 'CONDITIONAL_EXACT_TWO_MATERIAL_MODEL'
    assert actual['diagnostics']['clinical_accuracy'] == 'NOT_MEASURED'
    for key in ('partitions', 'cavities', 'air', 'component_bridge'):
        np.testing.assert_array_equal(actual[key], baseline[key])
    assert actual['cells'] == baseline['cells']
    assert any('unquantized' in text and 'not inferred' in text for text in actual['warnings'])


def test_baseline_accepted_single_space_is_preserved():
    hu = np.full((20, 18, 16), 800, np.float32)
    hu[3:-3, 3:-3, 3:-3] = -900
    roi, affine = np.ones(hu.shape, bool), np.diag([.5, .5, .5, 1])
    baseline = count_with_canonical_ridges(hu, roi, affine)
    assert baseline['diagnostics']['status'] == 'accepted'
    actual = count_with_saturated_barrier(hu, roi, affine)
    assert actual['diagnostics']['counts'] == baseline['diagnostics']['counts']
    assert actual['diagnostics']['status'] == baseline['diagnostics']['status']
    assert actual['cells'] == baseline['cells']
    for key in ('partitions', 'cavities', 'air', 'component_bridge'):
        np.testing.assert_array_equal(actual[key], baseline[key])


@pytest.mark.parametrize('kind', ('partial_hu', 'soft_sheet', 'roi_edge', 'partial_roi'))
def test_candidate_does_not_remove_veto_without_all_model_requirements(kind):
    f = _base(1.0, 0.0, 0.0).fixture
    hu, roi, affine = f.hu.copy(), f.roi.copy(), f.affine.copy()
    if kind == 'partial_hu':
        hu[16, 14, 12] = 799
    elif kind == 'soft_sheet':
        hu[(hu == 800) & (np.indices(hu.shape)[0] == 16)] = 35
    elif kind == 'roi_edge':
        # Open one existing free component to the outer image boundary.
        hu[:8, 14, 12] = -900
    else:
        roi[0, 0, 0] = False
    baseline = count_with_canonical_ridges(hu, roi, affine)
    actual = count_with_saturated_barrier(hu, roi, affine)
    assert actual['diagnostics']['status'] == baseline['diagnostics']['status'] == 'withheld'
    assert actual['diagnostics']['counts'] == baseline['diagnostics']['counts']
    assert not actual['diagnostics']['conditional_recovery']


@pytest.mark.parametrize('axes,signs', list(product(permutations(range(3)), product((1, -1), repeat=3))))
def test_all_signed_reindexings_preserve_conditional_count_and_masks(axes, signs):
    f = _base(1.0, 0.0, 0.0).fixture
    expected = count_with_saturated_barrier(f.hu, f.roi, f.affine)
    slices = tuple(slice(None, None, sign) for sign in signs)
    def transform(array):
        return array.transpose(axes)[slices].copy()
    mapping = np.zeros((4, 4))
    mapping[3, 3] = 1
    for new_axis, (old_axis, sign) in enumerate(zip(axes, signs)):
        mapping[old_axis, new_axis] = sign
        if sign < 0:
            mapping[old_axis, 3] = f.hu.shape[old_axis] - 1
    actual = count_with_saturated_barrier(transform(f.hu), transform(f.roi), f.affine @ mapping)
    assert actual['diagnostics']['counts'] == expected['diagnostics']['counts']
    assert actual['diagnostics']['conditional_recovery']
    for key in ('partitions', 'cavities', 'air', 'component_bridge'):
        np.testing.assert_array_equal(actual[key], transform(expected[key]))


def test_positive_aperture_rounds_to_identical_float32_and_falsifies_promotion():
    """A positive-area channel connects the spaces, yet all observed bytes match."""
    f = _base(1.0, 0.0, 0.0).fixture
    opened_observation = f.hu.copy()
    with localcontext() as ctx:
        ctx.prec = 100
        voxel_side = Decimal('0.5')
        aperture_side = Decimal('1e-20')
        air_fraction = (aperture_side / voxel_side) ** 2
        exact_hu = Decimal(800) - Decimal(1700) * air_fraction
        assert aperture_side > 0 and air_fraction > 0 and exact_hu < Decimal(800)
        # A channel at y=z=+0.25 mm spans the complete 1 mm wall (two X voxels).
        for x in (15, 16):
            assert opened_observation[x, 14, 12] == 800
            opened_observation[x, 14, 12] = np.float32(str(exact_hu))
    assert opened_observation.dtype == np.float32
    assert opened_observation.tobytes() == f.hu.tobytes()
    physical_connected_space_count = 1  # The stipulated channel is continuous and strictly positive.
    actual = count_with_saturated_barrier(opened_observation, f.roi, f.affine)
    assert actual['diagnostics']['counts']['total'] == 2 != physical_connected_space_count
    assert actual['diagnostics']['conditional_recovery']
    # This passing regression documents rejection of the inference rule, not clinical success.
