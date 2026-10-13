"""Independent phantom and scoring checks; no detector is invoked here."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
from scipy import ndimage as ndi

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synthetic_blur_experiment.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("synthetic_blur_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def registry(case, detected, *, flag_control=True):
    cells = []
    for identifier in sorted(set(np.unique(detected)) - {0}):
        center = np.argwhere(detected == identifier).mean(axis=0)
        x = float(case.fixture.affine[0, :3] @ center + case.fixture.affine[0, 3])
        reasons = ["crosses_midline_centroid_laterality_requires_review"] if case.wall_mm == 0 and flag_control else []
        cells.append({"id": int(identifier), "side": "left" if x <= 0 else "right",
                      "voxel_count": int(np.count_nonzero(detected == identifier)), "reasons": reasons})
    return cells


@pytest.fixture(scope="module")
def pair():
    return experiment.make_case(0.5, 0, 0)


@pytest.mark.parametrize("phase", [0, 0.25])
@pytest.mark.parametrize("wall, peak", [(0.25, [-475, -50]), (0.5, [-50, 800])])
def test_unblurred_wall_matches_analytic_voxel_occupancy(wall, peak, phase):
    case = experiment.make_case(wall, 0, phase)
    x = case.fixture.affine[0, 3] + np.arange(48) * 0.5
    bone_fraction = np.maximum(0, np.minimum(x + 0.25, wall / 2) - np.maximum(x - 0.25, -wall / 2)) / 0.5
    central = np.abs(x) <= 1.5
    profile = case.fixture.hu[:, 20, 16]
    np.testing.assert_array_equal(profile[central], (-900 + 1700 * bone_fraction)[central])
    assert float(profile[central].max()) == peak[[0, 0.25].index(phase)]


@pytest.mark.parametrize("sigma", [0.25, 0.5])
@pytest.mark.parametrize("phase", [0, 0.25])
def test_blur_matches_independent_normalized_discrete_convolution(sigma, phase):
    case = experiment.make_case(0.25, sigma, phase)
    fine_x = -12 + phase + (np.arange(192) + 0.5) * 0.125
    air = ((fine_x >= -8) & (fine_x < -0.125)) | ((fine_x >= 0.125) & (fine_x < 8))
    material = np.where(air, -900.0, 800.0)
    sigma_cells = sigma / 0.125
    radius = int(4 * sigma_cells + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    reference = np.convolve(np.pad(material, radius, constant_values=800), kernel, mode="valid")
    reference = reference.reshape(48, 4).mean(axis=1)
    # Y/Z centers are farther than the 2 mm maximum kernel radius from outer faces.
    np.testing.assert_allclose(case.fixture.hu[:, 20, 16], reference, rtol=0, atol=1e-3)
    np.testing.assert_array_equal(case.fixture.hu[:, 0, 0], np.full(48, 800, dtype=np.float32))


def test_fixed_cases_preserve_independent_volume_topology_identity_and_hashes():
    first = experiment.fixtures()
    assert len(first) == 24 and len({case.fixture.name for case in first}) == 24
    assert sum(case.wall_mm > 0 for case in first) == 18
    assert sum(case.wall_mm == 0 for case in first) == 6
    first_records = [experiment.record(case) for case in first]
    for case, recorded in zip(first, first_records):
        expected = 2 if case.wall_mm else 1
        assert case.fixture.hu.shape == (48, 40, 32)
        assert case.highres_labels.shape == (192, 160, 128)
        assert case.highres_labels.dtype == np.uint8
        assert case.fixture.labels.dtype == case.cores.dtype == np.uint16
        assert set(np.unique(case.fixture.labels)) == set(range(expected + 1))
        assert ndi.label(case.highres_labels > 0)[1] == expected
        assert np.count_nonzero(case.highres_labels) * 0.125 ** 3 == (16 - case.wall_mm) * 12 * 8
        assert recorded["highres_air_volume_mm3"] == recorded["expected_air_volume_mm3"]
        assert recorded["construction_counts"]["total"] == expected
        np.testing.assert_array_equal(case.fixture.affine[:3, 3], [-11.75 + case.phase_mm, -9.75, -7.75])
        centers = np.indices(case.fixture.labels.shape) * 0.5 + case.fixture.affine[:3, 3, None, None, None]
        x, y, z = centers
        boxes = [(-8, 8)] if expected == 1 else [(-8, -case.wall_mm / 2), (case.wall_mm / 2, 8)]
        for identifier in range(1, expected + 1):
            lower, upper = boxes[identifier - 1]
            reference = (x >= lower) & (x < upper) & (y >= -6) & (y < 6) & (z >= -4) & (z < 4)
            core_reference = (x >= lower + 1) & (x < upper - 1) & (y >= -5) & (y < 5) & (z >= -3) & (z < 3)
            np.testing.assert_array_equal(case.fixture.labels == identifier, reference)
            np.testing.assert_array_equal(case.cores == identifier, core_reference)
            assert np.count_nonzero(case.cores == identifier) > 0
            assert np.all(case.fixture.labels[case.cores == identifier] == identifier)
            assert ndi.label(case.highres_labels == identifier)[1] == 1
        for array in (case.fixture.hu, case.fixture.labels, case.fixture.roi, case.fixture.affine,
                      case.highres_labels, case.cores):
            assert not array.flags.writeable
        assert set(recorded["hashes"]) == {"hu", "labels", "roi", "affine", "highres_labels", "cores"}
        json.dumps(recorded, allow_nan=False)
    assert first_records == [experiment.record(case) for case in experiment.fixtures()]
    touching = experiment.make_case(0.25, 0, 0)
    assert ndi.label(touching.fixture.labels > 0)[1] == 1
    assert set(np.unique(touching.fixture.labels)) == {0, 1, 2}


@pytest.mark.parametrize("parameters", [(-0.25, 0, 0), (0.25, 0.1, 0), (0.25, 0, 0.1), (np.nan, 0, 0)])
def test_generator_rejects_parameters_outside_fixed_design(parameters):
    with pytest.raises(ValueError):
        experiment.make_case(*parameters)


def test_perfect_and_relabelled_predictions_pass(pair):
    for detected in (pair.fixture.labels, np.where(pair.fixture.labels == 1, 9, np.where(pair.fixture.labels == 2, 4, 0)).astype(np.uint16)):
        scored = experiment.score(pair, detected, registry(pair, detected))
        assert scored["passed"]
        assert scored["candidate_counts"] == scored["construction_counts"] == {"left": 1, "right": 1, "total": 2}
        assert scored["unmatched_prediction_ids"] == []
        assert len(scored["matches"]) == 2
        assert all(row["dice"] == row["core_coverage"] == 1 for row in scored["matches"])
        assert all(row["predicted_components"] == 1 for row in scored["matches"])
        json.dumps(scored, allow_nan=False)


def test_empty_prediction_retains_every_truth_row_with_zero_metrics(pair):
    detected = np.zeros_like(pair.fixture.labels)
    scored = experiment.score(pair, detected, [])
    assert not scored["passed"] and not scored["exact_count_pass"] and not scored["bijection_pass"]
    assert [row["truth_id"] for row in scored["matches"]] == [1, 2]
    assert all(row["matched_id"] is None and row["dice"] == row["core_coverage"] == 0 for row in scored["matches"])
    assert scored["candidate_counts"]["total"] == 0


@pytest.mark.parametrize("error", ["merge", "split", "extra", "compensating_merge_split"])
def test_incorrect_partitions_fail_without_hiding_metrics(pair, error):
    detected = pair.fixture.labels.copy()
    if error in {"merge", "compensating_merge_split"}:
        detected[detected == 2] = 1
    if error in {"split", "compensating_merge_split"}:
        mask = (pair.fixture.labels == 1) & (np.indices(detected.shape)[1] >= 20)
        detected[mask] = 3 if error == "split" else 2
    if error == "extra":
        detected[1:3, 1:3, 1:3] = 99
    scored = experiment.score(pair, detected, registry(pair, detected))
    assert not scored["passed"] and len(scored["matches"]) == 2
    assert all("dice" in row and "core_coverage" in row for row in scored["matches"])
    if error == "compensating_merge_split":
        assert scored["exact_count_pass"] and not scored["dice_pass"]
    if error in {"extra", "split"}:
        assert scored["unmatched_prediction_ids"]
    if error == "merge":
        assert sum(row["matched_id"] is None for row in scored["matches"]) == 1


def test_zero_overlap_is_unmatched_rather_than_a_fictitious_correspondence(pair):
    detected = np.zeros_like(pair.fixture.labels)
    detected[1:3, 1:3, 1:3] = 77
    scored = experiment.score(pair, detected, registry(pair, detected))
    assert all(row["matched_id"] is None and row["dice"] == 0 for row in scored["matches"])
    assert scored["unmatched_prediction_ids"] == [77]


def test_side_swapped_registry_fails_despite_perfect_masks_and_totals(pair):
    cells = registry(pair, pair.fixture.labels)
    for cell in cells:
        cell["side"] = "right" if cell["side"] == "left" else "left"
    scored = experiment.score(pair, pair.fixture.labels, cells)
    assert scored["exact_count_pass"] and scored["dice_pass"] and scored["core_coverage_pass"]
    assert not scored["side_assignment_pass"] and not scored["passed"]
    assert all(row["truth_side"] == row["predicted_centroid_side"] != row["predicted_side"] for row in scored["matches"])


def test_detached_island_fails_face_connectivity_despite_good_overlap(pair):
    detected = pair.fixture.labels.copy()
    detected[0, 0, 0] = 1
    scored = experiment.score(pair, detected, registry(pair, detected))
    assert scored["dice_pass"] and scored["core_coverage_pass"] and not scored["connected_pass"]
    assert scored["matches"][0]["predicted_components"] == 2
    assert not scored["passed"]


def test_core_coverage_is_separate_from_high_dice(pair):
    detected = pair.fixture.labels.copy()
    core = np.argwhere(pair.cores == 1)
    removed = core[:int(np.ceil(len(core) * 0.06))]
    detected[tuple(removed.T)] = 0
    scored = experiment.score(pair, detected, registry(pair, detected))
    assert scored["dice_pass"] and not scored["core_coverage_pass"] and not scored["passed"]
    assert scored["matches"][0]["dice"] >= 0.90
    assert scored["matches"][0]["core_coverage"] < 0.95


@pytest.mark.parametrize("phase", [0, 0.25])
def test_single_space_controls_require_the_exact_engine_flag_and_ignore_side(phase):
    case = experiment.make_case(0, 0, phase)
    cells = registry(case, case.fixture.labels)
    cells[0]["side"] = "right" if cells[0]["side"] == "left" else "left"
    scored = experiment.score(case, case.fixture.labels, cells)
    assert scored["passed"] and scored["side_assignment_pass"] and scored["flags_pass"]
    cells[0]["reasons"] = []
    missing = experiment.score(case, case.fixture.labels, cells)
    assert missing["exact_count_pass"] and missing["dice_pass"]
    assert not missing["flags_pass"] and not missing["passed"]


def test_hungarian_assignment_optimizes_global_dice(pair):
    truth = np.zeros_like(pair.fixture.labels)
    truth.reshape(-1)[:10] = 1
    truth.reshape(-1)[10:20] = 2
    detected = np.zeros_like(truth)
    detected.reshape(-1)[:6] = 7
    detected.reshape(-1)[6:10] = 8
    detected.reshape(-1)[10:14] = 7
    detected.reshape(-1)[20:26] = 8
    case = replace(pair, fixture=replace(pair.fixture, labels=truth), cores=truth)
    scored = experiment.score(case, detected, registry(case, detected))
    assert {row["truth_id"]: row["matched_id"] for row in scored["matches"]} == {1: 8, 2: 7}
    assert [row["dice"] for row in scored["matches"]] == [0.4, 0.4]


def test_visibility_is_threshold_specific_and_never_changes_truth_score():
    visible = experiment.make_case(1, 0, 0)
    invisible = experiment.make_case(0.25, 0, 0)
    for case, expected in ((visible, True), (invisible, False)):
        result = experiment.visibility(case)
        assert result["wall_visible"] is expected
        assert result["nonbone_components"] == (2 if expected else 1)
        assert experiment.score(case, case.fixture.labels, registry(case, case.fixture.labels))["passed"]
        json.dumps(result, allow_nan=False)
    control = experiment.make_case(0, 0.5, 0.25)
    assert experiment.visibility(control)["wall_visible"] is None
    assert experiment.visibility(invisible)["central_profile_max_hu"] == -475


def test_visibility_requires_each_entire_core_in_one_nonzero_component():
    case = experiment.make_case(1, 0, 0)
    hu = case.fixture.hu.copy()
    location = tuple(np.argwhere(case.cores == 1)[0])
    hu[location] = 200
    modified = replace(case, fixture=replace(case.fixture, hu=hu))
    result = experiment.visibility(modified)
    assert not result["wall_visible"]
    assert 0 in result["core_nonbone_component_ids"]["1"]
    assert experiment.score(modified, modified.fixture.labels, registry(modified, modified.fixture.labels))["passed"]


def test_score_rejects_shape_and_registry_inconsistencies(pair):
    with pytest.raises(ValueError, match="shape"):
        experiment.score(pair, pair.fixture.labels[:-1], [])
    with pytest.raises(ValueError, match="registry"):
        experiment.score(pair, pair.fixture.labels, [])


@pytest.mark.parametrize("bad_id", [1.9, "1", True])
def test_score_rejects_registry_ids_that_would_change_under_integer_coercion(pair, bad_id):
    cells = registry(pair, pair.fixture.labels)
    cells[0]["id"] = bad_id
    with pytest.raises(ValueError, match="registry"):
        experiment.score(pair, pair.fixture.labels, cells)


def test_real_protocol_survives_json_roundtrip_without_changing_frozen_inputs(pair):
    snapshot = experiment.protocol([experiment.make_case(0, 0, 0), pair])
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot


def test_runner_preserves_existing_outputs_and_rejects_repository_artifacts(tmp_path):
    import synthetic_counting_experiment as baseline

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.run(existing)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    with pytest.raises(ValueError, match="outside"):
        experiment.run(baseline.REPO / ".local" / "forbidden-output")


def test_source_change_is_rejected_before_output_or_detector_calls(tmp_path, monkeypatch):
    import synthetic_counting_experiment as baseline

    repo = tmp_path / "source"
    (repo / ".local").mkdir(parents=True)
    frozen = {"source_hashes": {"engine.py": "original"}}
    (repo / ".local/frozen-blur-plan.json").write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(baseline, "REPO", repo)
    monkeypatch.setattr(experiment, "fixtures", lambda: [])
    monkeypatch.setattr(experiment, "protocol", lambda cases: {"source_hashes": {"engine.py": "changed"}})
    monkeypatch.setattr(experiment, "measure", lambda case: pytest.fail("Detector reached after source drift"))
    destination = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="source differ"):
        experiment.run(destination)
    assert not destination.exists()


def test_freezing_is_measurement_free_and_cannot_overwrite_plan(tmp_path, monkeypatch):
    import synthetic_counting_experiment as baseline

    repo = tmp_path / "source"
    repo.mkdir()
    monkeypatch.setattr(baseline, "REPO", repo)
    monkeypatch.setattr(experiment, "fixtures", lambda: [])
    monkeypatch.setattr(experiment, "protocol", lambda cases: {"source_hashes": {"engine.py": "frozen"}})
    monkeypatch.setattr(experiment, "measure", lambda case: pytest.fail("Detector reached during freezing"))
    plan = experiment.freeze_plan()
    assert json.loads(plan.read_text(encoding="utf-8")) == {"source_hashes": {"engine.py": "frozen"}}
    with pytest.raises(FileExistsError):
        experiment.freeze_plan()
    assert json.loads(plan.read_text(encoding="utf-8"))["source_hashes"]["engine.py"] == "frozen"


def malformed_detector_output(pair):
    detected = pair.fixture.labels.copy()
    air = np.where(pair.fixture.hu < -500, detected, 0).astype(np.uint16)
    cells = registry(pair, detected)
    for cell in cells:
        mask = detected == cell["id"]
        center = pair.fixture.affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + pair.fixture.affine[:3, 3]
        cell.update(air_voxel_count=int(np.count_nonzero(air == cell["id"])),
                    centroid_ras_mm=center.tolist(), uncertain=bool(cell["reasons"]))
    return detected, air, cells + [cells[0].copy()], []


def test_malformed_detector_registry_is_retained_as_execution_error(pair, monkeypatch):
    from ethmoid_pipeline.counting import engine

    monkeypatch.setattr(engine, "compartment_stage", lambda *args: malformed_detector_output(pair))
    row, detected, air = experiment.measure(pair)
    assert row["failure_code"] == "UNEXPECTED_ValueError"
    assert row["candidate_counts"] == {"left": None, "right": None, "total": None}
    assert row["cells"] == [] and not row["passed"]
    assert not detected.any() and not air.any()
    assert all(match["dice"] == match["core_coverage"] == 0 for match in row["matches"])


def test_runner_keeps_execution_errors_separate_from_recovery_failures(pair, tmp_path, monkeypatch):
    from ethmoid_pipeline.counting import engine
    import synthetic_counting_experiment as baseline

    repo = tmp_path / "source"
    (repo / ".local").mkdir(parents=True)
    snapshot = {"test_only": "no real detector calls"}
    (repo / ".local/frozen-blur-plan.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(baseline, "REPO", repo)
    monkeypatch.setattr(experiment, "fixtures", lambda: [pair])
    monkeypatch.setattr(experiment, "protocol", lambda cases: snapshot)
    monkeypatch.setattr(engine, "compartment_stage", lambda *args: malformed_detector_output(pair))
    destination = tmp_path / "artifacts"
    summary = experiment.run(destination)
    assert summary["execution_errors"] == [pair.fixture.name]
    assert summary["physical_recovery_failures"] == []
    saved = json.loads((destination / pair.fixture.name / "result.json").read_text(encoding="utf-8"))
    assert saved["failure_code"] == "UNEXPECTED_ValueError"
    assert saved["candidate_counts"]["total"] is None

