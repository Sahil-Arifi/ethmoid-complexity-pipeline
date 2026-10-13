"""Independent analytic walls test abstention and image-only bridge evidence."""
from inspect import signature
from pathlib import Path
import sys

import numpy as np
import pytest

ndi = pytest.importorskip("scipy.ndimage")
pytest.importorskip("skimage")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from canonical_ridge_candidate import count_with_canonical_ridges
from ridge_counting_candidate import count_with_ridges


FACE = ndi.generate_binary_structure(3, 1)
REASON = "unresolved_subvoxel_communication"


def divided_box(bridge=800.0):
    hu = np.full((30, 24, 20), 800.0, np.float32)
    hu[4:26, 4:20, 4:16] = -900.0
    hu[14:16, 4:20, 4:16] = 800.0
    hu[14:16, 12, 10] = bridge
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [-7.25, -5.75, -4.75]
    return hu, np.ones(hu.shape, bool), affine


def assert_withheld(result):
    details = result["diagnostics"]
    assert details["status"] == "withheld"
    assert details["counts"] == {"left": None, "right": None, "total": None}
    assert details["experimental_count_available"] is False
    assert details["withheld_reasons"].count(REASON) == 1


@pytest.mark.parametrize("bridge", [800.0, 375.0, 693.75])
def test_above_cutoff_wall_is_measured_without_accepting_or_merging(bridge):
    hu, roi, affine = divided_box(bridge)
    previous = count_with_ridges(hu, roi, affine)
    assert previous["diagnostics"]["counts"]["total"] == 2
    result = count_with_canonical_ridges(hu, roi, affine)
    assert_withheld(result)
    assert result["cells"] == previous["cells"]
    for key in ("partitions", "cavities", "air"):
        assert np.array_equal(result[key], previous[key])

    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["component_count"] == 2
    assert record["status"] == "measured"
    assert record["native_bottleneck_hu"] == bridge
    assert record["terminal_sizes"] == [1120, 1120]
    assert record["touches_roi_boundary"] is (bridge == 800.0)
    witness = result["component_bridge"]
    assert witness.dtype == np.uint8 and witness.shape == hu.shape
    # These boxes have one connected sublevel set at the analytic bridge HU.
    assert np.array_equal(witness.astype(bool), roi & (hu <= bridge))
    assert ndi.label(witness, FACE)[1] == 1
    assert ndi.label(roi & (hu < bridge), FACE)[1] == 2


def test_disconnected_roi_cannot_borrow_a_path_from_excluded_bone():
    hu, roi, affine = divided_box(375.0)
    roi[14:16] = False
    result = count_with_canonical_ridges(hu, roi, affine)
    assert_withheld(result)
    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["component_count"] == 2
    assert record["status"] == "no_roi_path"
    assert record["native_bottleneck_hu"] is None
    assert record["touches_roi_boundary"] is None
    assert all(size > 0 for size in record["terminal_sizes"])
    assert not result["component_bridge"].any()


def test_witness_excludes_unrelated_eligible_roi_island():
    hu, roi, affine = divided_box(375.0)
    roi[:3, :3, :3] = False
    roi[:2, :2, :2] = True
    hu[:2, :2, :2] = 300.0
    result = count_with_canonical_ridges(hu, roi, affine)
    assert_withheld(result)
    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["status"] == "measured"
    assert record["native_bottleneck_hu"] == 375.0
    witness = result["component_bridge"].astype(bool)
    expected = roi & (hu <= 375.0)
    expected[:2, :2, :2] = False
    assert np.array_equal(witness, expected)
    assert not witness[:2, :2, :2].any()
    assert record["touches_roi_boundary"] is False


def test_more_than_two_components_withhold_even_when_bridge_is_unavailable():
    hu = np.full((42, 24, 20), 800.0, np.float32)
    hu[4:38, 4:20, 4:16] = -900.0
    hu[14:16, 4:20, 4:16] = 800.0
    hu[26:28, 4:20, 4:16] = 800.0
    roi = np.ones(hu.shape, bool)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    result = count_with_canonical_ridges(hu, roi, affine)
    assert_withheld(result)
    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["component_count"] == 3
    assert record["status"] == "not_two_components"
    assert record["native_bottleneck_hu"] is None
    assert record["touches_roi_boundary"] is None
    assert not result["component_bridge"].any()


def test_tiny_component_does_not_invent_interior_terminals():
    hu, roi, affine = divided_box()
    hu[16:26, 4:20, 4:16] = 800.0
    hu[20, 12, 10] = -900.0
    result = count_with_canonical_ridges(hu, roi, affine)
    assert_withheld(result)
    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["component_count"] == 2
    assert record["status"] == "no_interior_terminals"
    assert sorted(record["terminal_sizes"]) == [0, 1120]
    assert record["native_bottleneck_hu"] is None
    assert record["touches_roi_boundary"] is None
    assert not result["component_bridge"].any()


@pytest.mark.parametrize("opening", [False, True])
def test_previously_accepted_single_component_keeps_its_count(opening):
    hu, roi, affine = divided_box()
    if opening:
        hu[14:16, 9:15, 7:13] = -900.0
    else:
        hu[14:16, 4:20, 4:16] = -900.0
    previous = count_with_ridges(hu, roi, affine)
    result = count_with_canonical_ridges(hu, roi, affine)
    assert result["diagnostics"]["status"] == "accepted"
    assert result["diagnostics"]["counts"] == previous["diagnostics"]["counts"]
    assert result["diagnostics"]["counts"]["total"] == 1
    assert REASON not in result["diagnostics"]["withheld_reasons"]
    assert result["cells"] == previous["cells"]
    for key in ("partitions", "cavities", "air"):
        assert np.array_equal(result[key], previous[key])
    record = result["diagnostics"]["component_bridge_evidence"]
    assert record["component_count"] == 1
    assert record["status"] == "not_two_components"
    assert record["native_bottleneck_hu"] is None
    assert not result["component_bridge"].any()


def test_guard_preserves_existing_resolved_narrow_passage_abstention():
    arrays = divided_box(-900.0)
    previous = count_with_ridges(*arrays)
    result = count_with_canonical_ridges(*arrays)
    assert previous["diagnostics"]["status"] == "withheld"
    assert "unresolved_communication" in result["diagnostics"]["withheld_reasons"]
    assert REASON not in result["diagnostics"]["withheld_reasons"]
    assert result["diagnostics"]["counts"]["total"] is None


def test_detector_accepts_only_observations_and_does_not_mutate_them():
    assert list(signature(count_with_canonical_ridges).parameters) == ["hu", "roi", "affine"]
    arrays = divided_box(375.0)
    saved = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    result = count_with_canonical_ridges(*arrays)
    assert_withheld(result)
    for array, expected in zip(arrays, saved):
        assert np.array_equal(array, expected)
    with pytest.raises(TypeError):
        count_with_canonical_ridges(*arrays, expected_count=1)


def test_bridge_extractor_exposes_only_observation_inputs():
    from canonical_ridge_candidate import component_bridge_evidence

    assert list(signature(component_bridge_evidence).parameters) == ["hu", "roi", "affine"]
    arrays = divided_box(375.0)
    for array in arrays:
        array.setflags(write=False)
    record, witness = component_bridge_evidence(*arrays)
    assert record["native_bottleneck_hu"] == 375.0
    assert witness.dtype == np.uint8
    with pytest.raises(TypeError):
        component_bridge_evidence(*arrays, expected_count=1)
