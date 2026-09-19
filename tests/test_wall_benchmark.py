"""Small actual-phantom benchmark controls; no training or clinical evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting import wall_benchmark as benchmark
from ethmoid_pipeline.counting.synthetic_walls import (
    SceneAugmentation, candidate_targets, make_geometry, render_scene,
)
from ethmoid_pipeline.counting.wall_proposals import extract_candidates


def control(*, state="present", index=2, spacing=(0.8, 0.8, 0.8), distractors=False):
    geometry = replace(make_geometry(index, "plane"), tilt_x=0.0, tilt_z=0.0,
                       wall_thickness_mm=1.2, distractor_thickness_mm=0.9 if distractors else 0.01)
    scene = render_scene(geometry, SceneAugmentation(spacing_mm=spacing, wall_state=state))
    candidates = extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
    # Independent oracle scores are a unit-test control, never a trained result.
    labels = candidate_targets(scene, candidates.positions_ijk)
    scores = np.where(labels == 1, 0.95, 0.05)
    return scene, candidates, scores


def test_actual_planar_control_measures_observed_points_in_world_coordinates():
    scene, candidates, scores = control()
    row = benchmark.evaluate_scene(scene, candidates, scores)
    assert row["coordinate_origin_included"]
    assert row["geometry_id"] == scene.geometry_id and row["augmentation_id"] == scene.augmentation_id
    assert row["observable"] and row["accuracy_scope"] == "procedural_equations_not_expert_or_patient_anatomy"
    for method in row["methods"].values():
        assert method["accepted"] and method["abstention_code"] is None
        assert method["observed_target_precision"] == 1
        assert method["mean_physical_error_mm"] <= scene.spacing_mm[1]
        assert not method["false_acceptance"]
    expected_error = np.mean(np.abs(scene.origin_mm[1] + candidates.positions_ijk[:, 1] * scene.spacing_mm[1]
        - scene.target_leading_surface_y_mm[candidates.positions_ijk[:, 0], candidates.positions_ijk[:, 2]]))
    assert row["methods"]["learned"]["mean_physical_error_mm"] == pytest.approx(expected_error)
    json.dumps(row, allow_nan=False)


def test_world_translation_changes_neither_errors_nor_decisions():
    scene, candidates, scores = control()
    shifted = replace(scene, origin_mm=(scene.origin_mm[0] + 53, scene.origin_mm[1] - 61.25, scene.origin_mm[2] + 9),
                      target_surface_y_mm=scene.target_surface_y_mm - 61.25,
                      target_leading_surface_y_mm=scene.target_leading_surface_y_mm - 61.25)
    before, after = benchmark.evaluate_scene(scene, candidates, scores), benchmark.evaluate_scene(shifted, candidates, scores)
    for name in benchmark.METHODS:
        assert before["methods"][name]["accepted"] == after["methods"][name]["accepted"]
        assert before["methods"][name]["mean_physical_error_mm"] == pytest.approx(after["methods"][name]["mean_physical_error_mm"], abs=1e-5)


def test_exact_continuous_wall_is_reused_and_fit_is_never_labeled_observed(monkeypatch):
    scene, candidates, scores = control()
    actual = benchmark._continuous_wall
    calls = []
    def recorded(wall, valid, spacing):
        calls.append((wall.copy(), valid.copy(), spacing.copy()))
        return actual(wall, valid, spacing)
    monkeypatch.setattr(benchmark, "_continuous_wall", recorded)
    row = benchmark.evaluate_scene(scene, candidates, scores)
    baseline = row["methods"]["first_hit_quadratic"]
    assert len(calls) == 1 and baseline["accepted"]
    assert baseline["selected_point_count"] == np.count_nonzero(calls[0][1])
    assert baseline["observed_points_scope"] == "gradient_accepted_inputs_to_fit_not_interpolated_output"
    assert baseline["fitted_points_are_observed"] is False
    assert baseline["fitted_surface_mean_error_mm"] is not None
    assert "not_full_anatomical_inference" in row["baseline_scope"]


@pytest.mark.parametrize("state", ["missing", "averaged_away"])
def test_absent_support_abstains_without_zero_error_or_vacuous_pass(state):
    scene, candidates, scores = control(state=state)
    row = benchmark.evaluate_scene(scene, candidates, scores)
    for method in row["methods"].values():
        assert not method["accepted"] and method["abstention_code"]
        assert method["observed_target_precision"] is None and method["mean_physical_error_mm"] is None
    for summary in benchmark.summarize([row])["methods"].values():
        assert summary["accepted"] == 0 and summary["abstained"] == 1
        assert not summary["engineering_gate_passed"]
        assert summary["engineering_gate_status"] == "insufficient_accepted_evidence"


@pytest.mark.parametrize("state", ["cropped", "ambiguous"])
def test_actual_unobservable_control_counts_baseline_false_acceptance(state):
    scene, candidates, scores = control(state=state)
    if state == "ambiguous":
        # This control tests two separately observed walls, not details that
        # disappeared into a single sampled bone run.
        assert np.all(np.bincount(candidates.column_indices) >= 2)
    row = benchmark.evaluate_scene(scene, candidates, scores)
    assert not scene.observable
    for name in ("image_only", "learned"):
        assert not row["methods"][name]["accepted"]
        assert row["methods"][name]["abstention_code"] == ("WALL_CROPPED" if state == "cropped" else "WALL_AMBIGUOUS")
    baseline = row["methods"]["first_hit_quadratic"]
    assert baseline["accepted"] and baseline["false_acceptance"]
    if state == "ambiguous":
        assert baseline["observed_target_precision"] is None
    summary = benchmark.summarize([row])["methods"]["first_hit_quadratic"]
    assert summary["false_acceptances"] == 1 and not summary["engineering_gate_passed"]


def test_distractor_septa_are_not_correct_merely_because_first_hit_is_bone():
    scene, candidates, scores = control(distractors=True)
    row = benchmark.evaluate_scene(scene, candidates, scores)
    baseline = row["methods"]["first_hit_quadratic"]
    assert baseline["accepted"]
    assert baseline["observed_target_precision"] < 0.98
    assert not benchmark.summarize([row])["methods"]["first_hit_quadratic"]["engineering_gate_passed"]


def test_fixed_precision_and_physical_error_gates_apply_per_accepted_scene():
    scene, candidates, scores = control()
    row = benchmark.evaluate_scene(scene, candidates, scores)
    assert benchmark.summarize([row])["methods"]["learned"]["engineering_gate_passed"]
    for key, value in (("observed_target_precision", 0.979), ("mean_physical_error_mm", 1.001)):
        changed = deepcopy(row)
        changed["methods"]["learned"][key] = value
        changed["methods"]["learned"]["engineering_acceptance_passed"] = True
        assert not benchmark.summarize([changed])["methods"]["learned"]["engineering_gate_passed"]
    thick, found, score = control(index=4, spacing=(0.6, 2.0, 0.6))
    coarse = benchmark.evaluate_scene(thick, found, score)
    assert coarse["methods"]["learned"]["accepted"]
    coarse["methods"]["learned"]["mean_physical_error_mm"] = 1.5
    assert benchmark.summarize([coarse])["methods"]["learned"]["engineering_gate_passed"]
    coarse["methods"]["learned"]["mean_physical_error_mm"] = 2.001
    assert not benchmark.summarize([coarse])["methods"]["learned"]["engineering_gate_passed"]


def test_extraction_failure_is_accounted_with_no_fake_method_execution():
    scene, _, _ = control()
    row = benchmark.extraction_failure_row(scene, "WALL_RESOURCE_LIMIT")
    assert row["extraction_failed"]
    for method in row["methods"].values():
        assert not method["accepted"] and method["execution_scope"] == "not_run_extraction_failed"
        assert method["abstention_code"] == "WALL_RESOURCE_LIMIT"
        assert method["mean_physical_error_mm"] is None
    summary = benchmark.summarize([row])
    assert summary["scenes"] == summary["extraction_failed_scenes"] == 1
    assert all(not value["engineering_gate_passed"] for value in summary["methods"].values())


def test_summary_aggregates_all_outcomes_and_rejects_duplicate_scenes():
    accepted = benchmark.evaluate_scene(*control())
    unavailable = benchmark.extraction_failure_row(control(index=3)[0], "WALL_NO_SUPPORT")
    summary = benchmark.summarize([accepted, unavailable])
    assert summary["scenes"] == summary["geometries"] == 2
    assert summary["clinical_accuracy_measured"] is False
    assert summary["methods"]["learned"]["accepted"] == summary["methods"]["learned"]["abstained"] == 1
    assert summary["methods"]["learned"]["abstention_reasons"] == {"WALL_NO_SUPPORT": 1}
    with pytest.raises(ValueError, match="twice"):
        benchmark.summarize([accepted, accepted])
    empty = benchmark.summarize([])
    assert empty["scenes"] == 0
    assert all(not value["engineering_gate_passed"] for value in empty["methods"].values())
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("scores", [np.array([float("nan")]), np.array([-0.1]), np.array([1.1])])
def test_invalid_scores_are_input_errors_not_successful_abstention(scores):
    scene, candidates, _ = control()
    with pytest.raises(ValueError, match="probabilities"):
        benchmark.evaluate_scene(scene, candidates, np.full(len(candidates.positions_ijk), scores[0]))


def test_source_or_seed_mismatch_is_rejected_and_inputs_are_not_mutated():
    scene, candidates, scores = control()
    original = scene.hu.copy(), scene.seed.copy(), candidates.positions_ijk.copy(), scores.copy()
    benchmark.evaluate_scene(scene, candidates, scores)
    for before, after in zip(original, (scene.hu, scene.seed, candidates.positions_ijk, scores)):
        assert np.array_equal(before, after)
    with pytest.raises(ValueError, match="source"):
        benchmark.evaluate_scene(replace(scene, spacing_mm=(1, 1, 1)), candidates, scores)
    altered = scene.seed.copy()
    altered[:, np.max(np.where(altered)[1]), :] = False
    with pytest.raises(ValueError, match="posterior seed"):
        benchmark.evaluate_scene(replace(scene, seed=altered), candidates, scores)


def test_summary_does_not_trust_a_false_acceptance_flag_or_changed_criteria():
    row = benchmark.evaluate_scene(*control(state="cropped"))
    row["methods"]["first_hit_quadratic"]["false_acceptance"] = False
    assert benchmark.summarize([row])["methods"]["first_hit_quadratic"]["false_acceptances"] == 1
    row["engineering_criteria"]["minimum_accepted_observed_target_precision"] = 0.2
    with pytest.raises(ValueError, match="predetermined"):
        benchmark.summarize([row])


def test_retained_evidence_remeasures_metrics_without_model_inference():
    scene, candidates, scores = control()
    evidence = {}
    result = benchmark.evaluate_scene(scene, candidates, scores, evidence=evidence)
    np.testing.assert_array_equal(evidence["candidate_positions_ijk"], candidates.positions_ijk)
    np.testing.assert_array_equal(evidence["candidate_scores"], scores)
    for method in ("image_only", "learned"):
        selected = evidence[method + "_selected_candidate_indices"]
        points = evidence[method + "_observed_points_ijk"]
        np.testing.assert_array_equal(points, candidates.positions_ijk[selected])
        expected = candidate_targets(scene, points)
        assert np.mean(expected == 1) == result["methods"][method]["observed_target_precision"]
        measured = np.abs(scene.origin_mm[1] + points[:, 1] * scene.spacing_mm[1]
            - scene.target_leading_surface_y_mm[points[:, 0], points[:, 2]])
        assert measured.mean() == result["methods"][method]["mean_physical_error_mm"]
    assert evidence["first_hit_quadratic_fitted_surface_ijk"].shape == scene.seed.any(axis=1).shape
    assert not result["methods"]["first_hit_quadratic"]["fitted_points_are_observed"]
    with pytest.raises(ValueError, match="empty evidence"):
        benchmark.evaluate_scene(scene, candidates, scores, evidence=evidence)


def test_abstention_evidence_has_empty_selected_points_without_fake_zero_error():
    scene, candidates, scores = control(state="missing")
    evidence = {}
    row = benchmark.evaluate_scene(scene, candidates, scores, evidence=evidence)
    for method in benchmark.METHODS:
        assert not row["methods"][method]["accepted"]
        assert row["methods"][method]["mean_physical_error_mm"] is None
        assert evidence[method + "_observed_points_ijk"].shape == (0, 3)


def test_merged_wall_truth_is_not_invented_as_a_second_observed_face():
    geometry = replace(make_geometry(2, "plane"), tilt_x=0, tilt_z=0,
                       wall_thickness_mm=1.9569, distractor_thickness_mm=0.01)
    scene = render_scene(geometry, SceneAugmentation(wall_state="ambiguous"))
    candidates = extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
    assert np.all(np.bincount(candidates.column_indices) == 1)
    row = benchmark.evaluate_scene(scene, candidates, np.ones(len(candidates.positions_ijk)))
    # The geometric decoder sees only one run. The independent phantom knows
    # it came from merged walls, and must record failure, never correct anatomy.
    assert row["methods"]["image_only"]["accepted"]
    assert row["methods"]["image_only"]["false_acceptance"]
    summary = benchmark.summarize([row])
    assert not summary["methods"]["image_only"]["engineering_gate_passed"]
    assert summary["methods"]["image_only"]["false_acceptances"] == 1


def test_point_abstention_retains_reason_without_publishing_partial_evidence():
    scene, candidates, scores = control()
    evidence = {}
    original = benchmark.evaluate_scene(scene, candidates, scores, evidence=evidence)
    assert original["methods"]["learned"]["accepted"]
    selected = evidence["learned_selected_candidate_indices"]
    scores = scores.copy()
    scores[selected[len(selected) // 2]] = 0.000001
    evidence = {}
    changed = benchmark.evaluate_scene(scene, candidates, scores, evidence=evidence)
    method = changed["methods"]["learned"]
    assert not method["accepted"]
    assert method["abstention_code"] == "WALL_POINT_UNSUPPORTED"
    assert method["mean_physical_error_mm"] is None
    assert method["observed_target_precision"] is None
    assert method["support_diagnostics"]["score_unsupported_points"] == 1
    assert method["support_diagnostics"]["points_removed"] == 0
    assert method["support_diagnostics"]["whole_proposal_withheld"] is True
    assert evidence["learned_selected_candidate_indices"].size == 0
    assert evidence["learned_observed_points_ijk"].shape == (0, 3)
    assert changed["methods"]["image_only"] == original["methods"]["image_only"]
    json.dumps(changed, allow_nan=False)
