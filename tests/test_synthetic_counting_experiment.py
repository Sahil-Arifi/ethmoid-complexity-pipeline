"""Experiment boundaries and independent scoring, using only synthetic arrays."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

spec = importlib.util.spec_from_file_location(
    "synthetic_counting_experiment",
    Path(__file__).resolve().parents[1] / "scripts/synthetic_counting_experiment.py",
)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def test_spatial_matching_rejects_compensating_merge_and_split():
    truth = np.zeros((8, 8, 8), dtype=np.uint16)
    truth[1:3, 1:3, 1:3] = 1
    truth[5:7, 5:7, 5:7] = 2
    detected = truth.copy()
    detected[truth == 2] = 1
    detected[1, 1:3, 1:3] = 2
    assert len(np.unique(truth)) == len(np.unique(detected))
    assert experiment.exact_correspondence(truth, detected) is None


def test_spatial_matching_accepts_relabeling_but_rejects_missing_voxels():
    truth = np.zeros((8, 8, 8), dtype=np.uint16)
    truth[1:3, 1:3, 1:3] = 1
    truth[5:7, 5:7, 5:7] = 2
    detected = np.where(truth == 1, 9, np.where(truth == 2, 4, 0)).astype(np.uint16)
    assert experiment.exact_correspondence(truth, detected) == {1: 9, 2: 4}
    detected[1, 1, 1] = 0
    assert experiment.exact_correspondence(truth, detected) is None


def test_fixture_generation_is_frozen_and_independently_checked():
    first = experiment.fixtures()
    second = experiment.fixtures()
    assert len(first) == 20
    assert len({case.name for case in first}) == 20
    assert [experiment.fixture_record(case) for case in first] == [experiment.fixture_record(case) for case in second]
    for case in first:
        experiment.check_fixture(case)
        assert not case.hu.flags.writeable
        assert not case.labels.flags.writeable


def test_axis_reindexing_preserves_the_physical_construction():
    cases = {case.name: case for case in experiment.fixtures()}
    original = cases["two_sealed"]
    flipped = cases["axis_flip"]
    permuted = cases["axis_permutation"]
    assert np.array_equal(flipped.labels[::-1], original.labels)
    assert np.array_equal(permuted.labels.transpose(1, 2, 0), original.labels)
    assert experiment.construction_counts(original) == experiment.construction_counts(flipped) == experiment.construction_counts(permuted)
    for label in (1, 2):
        centroids = []
        for case in (original, flipped, permuted):
            index = np.argwhere(case.labels == label).mean(axis=0)
            centroids.append(case.affine[:3, :3] @ index + case.affine[:3, 3])
        assert np.allclose(centroids, centroids[0])


def test_ambiguity_does_not_become_an_exact_count_pass():
    case = next(case for case in experiment.fixtures() if case.name == "small_wall_opening")
    result, _, _ = experiment.measure(case)
    assert result["construction_counts"] is None
    assert result["possible_totals"] == [1, 2]
    assert result["category"] == "ambiguous"
    assert result["exact_count_pass"] is None


def test_failed_detection_has_null_counts_and_no_instances():
    case = next(case for case in experiment.fixtures() if case.name == "unbounded_soft_tissue")
    result, labels, air = experiment.measure(case)
    assert result["failure_code"] == "NO_SUPPORTED_CELLS"
    assert result["candidate_counts"] == {"left": None, "right": None, "total": None}
    assert result["cells"] == []
    assert not labels.any() and not air.any()


def test_frozen_suite_meets_every_predeclared_requirement():
    results = [experiment.measure(case)[0] for case in experiment.fixtures()]
    assert len(results) == 20
    assert sum(row["category"] == "count" for row in results) == 14
    assert all(row["passed"] for row in results), [row["name"] for row in results if not row["passed"]]
    assert all(row["exact_count_pass"] is None for row in results if row["category"] != "count")


def test_runner_refuses_existing_destinations_and_repository_outputs(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.run(existing)
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    with pytest.raises(ValueError, match="outside"):
        experiment.run(experiment.REPO / ".local" / "forbidden-output")
