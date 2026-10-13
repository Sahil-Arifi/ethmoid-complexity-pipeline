"""Analytic unit constructions, separate from frozen evaluation fixtures."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ridge_counting_candidate import count_with_ridges, propose_partitions
from ethmoid_pipeline.counting.engine import CountingError


def unit_volume(wall=None, opening=False):
    hu = np.full((30, 24, 20), 800, dtype=np.float32)
    hu[4:26, 4:20, 4:16] = -900
    if wall is not None:
        hu[14:16, 4:20, 4:16] = wall
    if opening:
        hu[14:16, 9:15, 7:13] = -900
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = [-7.25, -5.75, -4.75]
    return hu, np.ones(hu.shape, bool), affine


@pytest.mark.parametrize('wall', [800, -300])
def test_continuous_proposals_detect_strong_and_faint_dividers(wall):
    labels, details = propose_partitions(*unit_volume(wall))
    assert labels.max() == details['final_basin_count'] == 2
    assert labels[8, 12, 10] != labels[22, 12, 10]


@pytest.mark.parametrize('wall,opening', [(None, False), (800, True)])
def test_one_space_and_resolved_aperture_remain_one(wall, opening):
    labels, details = propose_partitions(*unit_volume(wall, opening))
    assert labels.max() == details['final_basin_count'] == 1
    assert all(m['verified_radius_floor_mm'] >= m['required_radius_mm'] for m in details['passage_merges'])


@pytest.mark.parametrize('defect,code', [('nan', 'INVALID_VOLUME'), ('empty', 'EMPTY_REGION'),
    ('roi_type', 'INVALID_VOLUME'), ('shape', 'INVALID_VOLUME'), ('complex', 'INVALID_VOLUME'),
    ('shear', 'INVALID_GEOMETRY'), ('bone', 'NO_COMPARTMENTS')])
def test_bad_inputs_reject(defect, code):
    hu, roi, affine = unit_volume()
    if defect == 'nan': hu[0, 0, 0] = np.nan
    if defect == 'empty': roi[:] = False
    if defect == 'roi_type': roi = roi.astype(np.uint8)
    if defect == 'shape': roi = roi[:-1]
    if defect == 'complex': hu = hu.astype(complex)
    if defect == 'shear': affine[0, 1] = .2
    if defect == 'bone': hu[:] = 800
    with pytest.raises(CountingError) as error:
        propose_partitions(hu, roi, affine)
    assert error.value.code == code


@pytest.mark.parametrize('wall,opening,expected', [(None, False, 1), (800, False, 2), (800, True, 1)])
def test_accepted_strong_geometry_and_actual_opening(wall, opening, expected):
    hu, roi, affine = unit_volume(wall, opening)
    for array in (hu, roi, affine): array.setflags(write=False)
    result = count_with_ridges(hu, roi, affine)
    assert result['diagnostics']['status'] == 'accepted'
    assert result['diagnostics']['counts']['total'] == expected
    assert np.array_equal(result['air'], np.where(hu < -500, result['cavities'], 0))
    for cell in result['cells']:
        mask = result['cavities'] == cell['id']
        assert cell['voxel_count'] == mask.sum() and cell['uncertain']
        assert np.allclose(cell['centroid_ras_mm'], np.argwhere(mask).mean(axis=0) @ affine[:3, :3].T + affine[:3, 3])


@pytest.mark.parametrize('wall', [-300, 35, 100])
def test_weak_ridge_is_a_proposal_withheld_as_a_count(wall):
    result = count_with_ridges(*unit_volume(wall))
    assert result['diagnostics']['status'] == 'withheld'
    assert result['diagnostics']['counts'] == {'left': None, 'right': None, 'total': None}
    assert 'material_ambiguity' in result['diagnostics']['withheld_reasons']
    assert len(result['cells']) == 2


def test_geometry_rejection_does_not_become_zero_count():
    hu, roi, affine = unit_volume()
    hu[:] = -900
    result = count_with_ridges(hu, roi, affine)
    assert result['diagnostics']['status'] == 'withheld'
    assert result['diagnostics']['counts']['total'] is None
    assert 'rejected_geometric_proposal' in result['diagnostics']['withheld_reasons']


def test_axis_swap_and_reflection_preserve_native_physical_proposals():
    hu, roi, affine = unit_volume(-300)
    original = count_with_ridges(hu, roi, affine)
    transformed_affine = affine[:, [2, 1, 0, 3]].copy()
    transformed_affine[:3, 3] += transformed_affine[:3, 1] * (hu.shape[1] - 1)
    transformed_affine[:3, 1] *= -1
    transformed = count_with_ridges(hu.swapaxes(0, 2)[:, ::-1, :], roi.swapaxes(0, 2)[:, ::-1, :], transformed_affine)
    assert np.array_equal(original['cavities'], transformed['cavities'][:, ::-1, :].swapaxes(0, 2))
    assert original['diagnostics']['counts'] == transformed['diagnostics']['counts']
    for first, second in zip(original['cells'], transformed['cells']):
        assert first['side'] == second['side']
        assert np.allclose(first['centroid_ras_mm'], second['centroid_ras_mm'])


def test_three_basin_passage_contraction_is_independent_of_reflection():
    hu = np.full((64, 28, 28), 800, dtype=np.float32)
    hu[4:24, 4:24, 4:24], hu[24:40, 4:24, 4:24], hu[40:60, 4:24, 4:24] = -900, -850, -800
    hu[23:25, 4:24, 4:24], hu[39:41, 4:24, 4:24] = -825, -775
    for x in (12, 32, 50): hu[x-2:x+3, 12:17, 12:17] = -1200
    roi = np.ones(hu.shape, bool)
    affine = np.diag([.5, .5, .5, 1.])
    reflected = affine.copy()
    reflected[0, 3], reflected[0, 0] = .5 * (hu.shape[0] - 1), -.5
    original = count_with_ridges(hu, roi, affine)
    changed = count_with_ridges(hu[::-1], roi, reflected)
    assert original['diagnostics']['counts'] == changed['diagnostics']['counts']
    assert original['diagnostics']['status'] == changed['diagnostics']['status']
    assert original['diagnostics']['final_basin_count'] == changed['diagnostics']['final_basin_count']
    assert np.array_equal(original['cavities'], changed['cavities'][::-1])
