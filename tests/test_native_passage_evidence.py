"""Analytic checks for image evidence without changing count acceptance."""
from pathlib import Path
import sys

import numpy as np
import pytest

ndi = pytest.importorskip('scipy.ndimage')
pytest.importorskip('skimage')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_passage_evidence import passage_evidence
from ridge_counting_candidate import count_with_ridges
from ethmoid_pipeline.counting.engine import CountingError


def example(bridge=-900):
    hu = np.full((24, 20, 18), 800., np.float32)
    hu[3:21, 3:17, 3:15] = -900
    hu[11:13, 3:17, 3:15] = 800
    hu[11:13, 10, 9] = bridge
    labels = np.zeros(hu.shape, np.int32)
    labels[(hu < 200) & (np.indices(hu.shape)[0] < 12)] = 1
    labels[(hu < 200) & (np.indices(hu.shape)[0] >= 12)] = 2
    affine = np.diag([.5, .6, .7, 1.])
    affine[:3, 3] = [-5.75, -5.7, -5.95]
    return hu, np.ones(hu.shape, bool), affine, labels


@pytest.mark.parametrize('level', [-900, -650, -100, 100])
def test_known_bottleneck_and_independent_witness(level):
    arrays = example(level)
    for array in arrays: array.setflags(write=False)
    report, witnesses = passage_evidence(*arrays)
    row = report['pairs'][0]
    assert row['native_bottleneck_hu'] == level
    assert row['voxel_center_clearance_mm'] == pytest.approx(.6)
    witness = witnesses[row['witness_key']].astype(bool)
    assert ndi.label(witness)[1] == 1
    assert set(np.unique(arrays[3][witness])) == {1, 2}
    assert arrays[0][witness].max() == level
    assert report['counts_changed'] is False


def test_widest_path_uses_longer_wider_route_at_same_intensity():
    hu, roi, affine, labels = example()
    hu[11:13, 5:8, 5:8] = -900
    labels[11, 5:8, 5:8], labels[12, 5:8, 5:8] = 1, 2
    report, _ = passage_evidence(hu, roi, affine, labels)
    assert report['pairs'][0]['native_bottleneck_hu'] == -900
    assert report['pairs'][0]['voxel_center_clearance_mm'] == pytest.approx(1.2)


def test_witness_retains_each_disjoint_qualifying_corridor():
    hu, roi, affine, labels = example()
    hu[hu < 200] = -650
    for y in (5, 13):
        hu[3:21, y, 9] = -900
        labels[11, y, 9], labels[12, y, 9] = 1, 2
    report, witnesses = passage_evidence(hu, roi, affine, labels)
    row = report['pairs'][0]
    assert row['native_bottleneck_hu'] == -900
    components, count = ndi.label(witnesses[row['witness_key']])
    assert count == 2
    for identifier in range(1, count + 1):
        assert set(np.unique(labels[components == identifier])) == {1, 2}


def test_sealed_wall_has_no_adjacent_pair():
    report, witnesses = passage_evidence(*example(800))
    assert report['pairs'] == [] and witnesses == {}


def test_absent_bone_does_not_invent_boundary_support():
    hu = np.full((8, 8, 8), -900., np.float32)
    labels = np.ones(hu.shape, np.int32)
    labels[1:] = 2
    report, _ = passage_evidence(hu, np.ones(hu.shape, bool), np.diag([.5, .5, .5, 1.]), labels)
    assert report['pairs'][0]['contact_near_strong_bone_fraction'] == 0


def test_physical_reindexing_preserves_evidence_and_witness():
    hu, roi, affine, labels = example(-650)
    expected, masks = passage_evidence(hu, roi, affine, labels)
    def transform(a): return a.transpose(2, 1, 0)[:, ::-1, :].copy()
    mapping = np.array([[0, 0, 1, 0], [0, -1, 0, hu.shape[1]-1], [1, 0, 0, 0], [0, 0, 0, 1.]])
    actual, changed = passage_evidence(transform(hu), transform(roi), affine @ mapping, transform(labels))
    for key in ('native_bottleneck_hu', 'voxel_center_clearance_mm', 'terminal_hu_limits', 'contact_hu_quantiles'):
        assert actual['pairs'][0][key] == expected['pairs'][0][key]
    assert np.allclose(actual['pairs'][0]['centroids_ras_mm'], expected['pairs'][0]['centroids_ras_mm'])
    for key in masks: assert np.array_equal(transform(masks[key]), changed[key])


@pytest.mark.parametrize('defect', ['nan', 'complex', 'float_labels', 'outside', 'disconnected', 'shape'])
def test_invalid_input_rejects(defect):
    hu, roi, affine, labels = example()
    if defect == 'nan': hu[0, 0, 0] = np.nan
    if defect == 'complex': hu = hu.astype(complex)
    if defect == 'float_labels': labels = labels.astype(float)
    if defect == 'outside': labels[0, 0, 0] = 1
    if defect == 'disconnected': labels[20, 16, 14] = 1
    if defect == 'shape': labels = labels[:-1]
    with pytest.raises(CountingError, match='.'):
        passage_evidence(hu, roi, affine, labels)


def test_evidence_does_not_remove_existing_narrow_opening_abstention():
    hu, roi, affine, _ = example()
    result = count_with_ridges(hu, roi, affine)
    assert result['diagnostics']['status'] == 'withheld'
    report, _ = passage_evidence(hu, roi, affine, result['partitions'])
    assert report['pairs'][0]['native_bottleneck_hu'] == -900
    assert result['diagnostics']['counts']['total'] is None


def test_missing_interior_is_explicit_not_a_failed_zero_measurement():
    hu, roi, affine, labels = example()
    labels[:] = 0
    labels[3:5, 3, 3:5] = 1
    labels[5:7, 3, 3:5] = 2
    report, witnesses = passage_evidence(hu, roi, affine, labels)
    assert report['pairs'][0]['status'] == 'no_interior_terminals'
    assert report['pairs'][0]['native_bottleneck_hu'] is None
    assert witnesses == {}
