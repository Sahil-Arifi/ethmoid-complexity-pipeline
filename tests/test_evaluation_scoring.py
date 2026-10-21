"""Invented records and arrays test scoring, without inference or clinical files."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation_scoring.py"
spec = importlib.util.spec_from_file_location("evaluation_scoring", SCRIPT)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def census(n=3):
    return {
        "schema_version": 1,
        "evidence_type": "synthetic",
        "evaluation_freeze_id": "synthetic-check-1",
        "method_sha256": "a" * 64,
        "cases": [{"case_id": f"synthetic-{i}", "participant_group_id": f"group-{i}",
                   "source_sha256": str(i + 1) * 64} for i in range(n)],
    }


def record(i, reference=4, prediction=4, endpoint="left", state="available"):
    return {
        "evaluation_freeze_id": "synthetic-check-1", "case_id": f"synthetic-{i}",
        "endpoint": endpoint,
        "reference": {"state": "final", "count": reference, "independence": "independent",
                      "source_sha256": str(i + 1) * 64, "scope": "full_3d",
                      "locked_at": "2026-10-20T12:00:00-07:00", "evidence_ref": "invented-reference"},
        "prediction": {"state": state, "count": prediction, "kind": "original_automatic",
                       "source_sha256": str(i + 1) * 64, "method_sha256": "a" * 64,
                       "run_id": f"invented-run-{i}", "uncertainty": [], "reason": "synthetic fixture"},
    }


def test_draft_arithmetic_keeps_withheld_case_and_all_scheduled_endpoints():
    result = scoring.evaluate_counts(census(), [record(0), record(1, 5, None, state="withheld"), record(2, 6, 8)])
    left = result["endpoints"]["left"]
    assert left["denominators"] == {"N": 3, "R": 3, "C": 2, "P": 2, "E": 1}
    assert left["mean_absolute_error"] == 1
    assert left["mean_signed_error"] == 1
    assert left["median_absolute_error"] == 1
    assert left["max_absolute_error"] == 2
    assert left["exact_agreement"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert left["answer_coverage"]["value"] == 2 / 3
    assert left["end_to_end_exact"]["value"] == 1 / 3
    assert result["endpoints"]["right"]["prediction_states"]["unknown"] == 3
    withheld = next(row for row in result["rows"] if row["case_id"] == "synthetic-1" and row["endpoint"] == "left")
    assert withheld["signed_error"] is None
    assert len(result["rows"]) == 9


@pytest.mark.parametrize("field,value", [
    ("state", "provisional"), ("independence", "model_assisted"), ("independence", "unknown"),
    ("source_sha256", "f" * 64), ("scope", "landmark_2d"), ("scope", "partial_volume"),
    ("locked_at", None), ("evidence_ref", None),
])
def test_ineligible_reference_does_not_remove_available_answer(field, value):
    row = record(0)
    row["reference"][field] = value
    result = scoring.evaluate_counts(census(1), [row])
    left = result["endpoints"]["left"]
    assert left["denominators"] == {"N": 1, "R": 0, "C": 1, "P": 0, "E": 0}
    assert left["answer_coverage"]["value"] == 1
    assert left["exact_agreement"]["value"] is None
    assert result["rows"][0]["reference_exclusions"]


@pytest.mark.parametrize("state", ["missing", "conflict"])
def test_missing_or_conflicting_reference_is_not_zero(state):
    row = record(0, None, 0)
    row["reference"]["state"] = state
    left = scoring.evaluate_counts(census(1), [row])["endpoints"]["left"]
    assert left["denominators"] == {"N": 1, "R": 0, "C": 1, "P": 0, "E": 0}


def test_explicit_zero_is_a_valid_count_and_empty_denominators_are_null():
    result = scoring.evaluate_counts(census(1), [record(0, 0, 0)])
    assert result["endpoints"]["left"]["denominators"]["E"] == 1
    empty = scoring.evaluate_counts(census(0), [])["endpoints"]["left"]
    assert empty["exact_agreement"] == {"numerator": 0, "denominator": 0, "value": None}
    assert empty["mean_absolute_error"] is None


def test_every_prediction_state_partitions_census_and_flags_survive():
    states = ["available", "withheld", "technical_failure", "not_run", "unknown"]
    rows = [record(i, 4, 4 if state == "available" else None, state=state) for i, state in enumerate(states)]
    rows[0]["prediction"]["uncertainty"] = ["synthetic ambiguity"]
    result = scoring.evaluate_counts(census(5), rows)
    left = result["endpoints"]["left"]
    assert left["prediction_states"] == dict.fromkeys(states, 1)
    assert left["denominators"] == {"N": 5, "R": 5, "C": 1, "P": 1, "E": 1}
    assert left["uncertain_available"] == 1
    assert result["rows"][0]["prediction_uncertainty"] == ["synthetic ambiguity"]


def test_opposite_side_errors_do_not_hide_behind_exact_total():
    rows = [record(0, 4, 5), record(0, 6, 5, "right"), record(0, 10, 10, "total")]
    result = scoring.evaluate_counts(census(1), rows)
    assert [result["endpoints"][side]["mean_signed_error"] for side in ("left", "right", "total")] == [1, -1, 0]
    assert result["total_consistency"][0] == {"case_id": "synthetic-0", "reference": "consistent", "prediction": "consistent"}


@pytest.mark.parametrize("which", ["reference", "prediction"])
def test_inconsistent_stated_totals_are_rejected(which):
    rows = [record(0, 4, 4), record(0, 6, 6, "right"), record(0, 10, 10, "total")]
    rows[-1][which]["count"] = 11
    with pytest.raises(ValueError, match="total"):
        scoring.evaluate_counts(census(1), rows)


@pytest.mark.parametrize("bad", [True, False, -1, 2.5, "4", float("nan"), float("inf"), None])
@pytest.mark.parametrize("which", ["reference", "prediction"])
def test_counts_reject_coercion_and_nonfinite_values(which, bad):
    row = record(0)
    row[which]["count"] = bad
    with pytest.raises(ValueError, match="count"):
        scoring.evaluate_counts(census(1), [row])


@pytest.mark.parametrize("state", ["withheld", "technical_failure", "not_run", "unknown"])
def test_unavailable_prediction_cannot_carry_a_number(state):
    with pytest.raises(ValueError, match="count"):
        scoring.evaluate_counts(census(1), [record(0, 4, 4, state=state)])


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "f" * 64), ("method_sha256", "f" * 64),
    ("kind", "corrected"), ("run_id", ""), ("uncertainty", "unstructured flag"),
])
def test_invalid_output_provenance_rejects_batch_instead_of_partial_report(field, value):
    row = record(0)
    row["prediction"][field] = value
    with pytest.raises(ValueError):
        scoring.evaluate_counts(census(1), [row])


def test_duplicate_outside_and_wrong_freeze_rows_reject():
    with pytest.raises(ValueError, match="duplicate"):
        scoring.evaluate_counts(census(1), [record(0), record(0)])
    with pytest.raises(ValueError, match="census"):
        scoring.evaluate_counts(census(1), [record(1)])
    row = record(0)
    row["evaluation_freeze_id"] = "other-freeze"
    with pytest.raises(ValueError, match="freeze"):
        scoring.evaluate_counts(census(1), [row])


def test_related_reconstructions_cannot_inflate_participant_denominator():
    plan = census(2)
    plan["cases"][1]["participant_group_id"] = plan["cases"][0]["participant_group_id"]
    with pytest.raises(ValueError, match="participant"):
        scoring.evaluate_counts(plan, [])


def test_partial_endpoint_set_cannot_mix_prediction_runs():
    rows = [record(0), record(0, 6, 6, "right")]
    rows[1]["prediction"]["run_id"] = "different-original-run"
    with pytest.raises(ValueError, match="same original run"):
        scoring.evaluate_counts(census(1), rows)


@pytest.mark.parametrize("kind", ["reference", "prediction"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_total_cannot_be_smaller_than_a_known_side_when_other_side_is_missing(kind, side):
    rows = [record(0, 4, 4, side), record(0, 4, 4, "total")]
    rows[1][kind]["count"] = 3
    with pytest.raises(ValueError, match="total"):
        scoring.evaluate_counts(census(1), rows)


def test_feasible_partial_total_is_not_claimed_to_be_verified():
    rows = [record(0, 4, 4), record(0, 7, 7, "total")]
    result = scoring.evaluate_counts(census(1), rows)
    assert result["total_consistency"][0]["reference"] == "not_checkable"
    assert result["total_consistency"][0]["prediction"] == "not_checkable"


@pytest.mark.parametrize("malformation", ["missing_reference", "missing_count", "bad_case_id", "bad_state", "nonlist_records"])
def test_malformed_input_has_explicit_validation_failure(malformation):
    row = record(0)
    rows = [row]
    if malformation == "missing_reference":
        del row["reference"]
    elif malformation == "missing_count":
        del row["prediction"]["count"]
    elif malformation == "bad_case_id":
        row["case_id"] = []
    elif malformation == "bad_state":
        row["prediction"]["state"] = {}
    else:
        rows = {"not": "a record list"}
    with pytest.raises(ValueError):
        scoring.evaluate_counts(census(1), rows)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", 2), ("evidence_type", "clinical"), ("cases", None)])
def test_invalid_or_clinical_census_is_rejected(field, value):
    plan = census(1)
    plan[field] = value
    with pytest.raises(ValueError):
        scoring.evaluate_counts(plan, [])


def test_scoring_does_not_mutate_records_and_is_order_invariant():
    plan, rows = census(2), [record(0), record(1, 5, 6)]
    before = deepcopy((plan, rows))
    first = scoring.evaluate_counts(plan, rows)
    assert (plan, rows) == before
    assert first == scoring.evaluate_counts(plan, list(reversed(rows)))


def labels(values):
    return np.array(values, dtype=np.uint16).reshape(-1, 1, 1)


def spatial(reference, prediction, ref_sides=None, pred_sides=None):
    ref_sides = ref_sides if ref_sides is not None else {int(i): "left" for i in np.unique(reference) if i}
    pred_sides = pred_sides if pred_sides is not None else {int(i): "left" for i in np.unique(prediction) if i}
    return scoring.score_synthetic_instances(reference, prediction, ref_sides, pred_sides)


@pytest.mark.parametrize("truth,prediction,relationships,exact", [
    ([1, 1, 0, 2, 2], [7, 7, 0, 9, 9], ["one_to_one", "one_to_one"], True),
    ([1, 1, 0, 2, 2], [7, 7, 7, 7, 7], ["merge"], False),
    ([1, 1, 1, 1, 0], [7, 7, 9, 9, 0], ["split"], False),
    ([1, 1, 0, 2, 2], [7, 7, 0, 0, 0], ["one_to_one", "missed"], False),
    ([1, 1, 0, 0, 0], [7, 7, 0, 9, 9], ["one_to_one", "extra"], False),
    ([1, 1, 2, 2], [7, 9, 7, 9], ["complex"], False),
    ([1, 1, 0, 0], [0, 7, 7, 0], ["one_to_one"], False),
    ([1, 1, 0, 0], [0, 0, 7, 7], ["missed", "extra"], False),
])
def test_complete_synthetic_overlap_groups(truth, prediction, relationships, exact):
    result = spatial(labels(truth), labels(prediction))
    assert [group["relationship"] for group in result["groups"]] == relationships
    assert result["exact_instance_match"] is exact


def test_compensating_merge_and_split_cannot_pass_equal_total():
    result = spatial(labels([1, 1, 0, 2, 2, 0, 3, 3]), labels([5, 5, 0, 5, 5, 0, 6, 7]))
    assert result["reference_count"] == result["prediction_count"] == 3
    assert [g["relationship"] for g in result["groups"]] == ["merge", "split"]
    assert not result["exact_instance_match"]


def test_side_swap_fails_even_when_geometry_and_total_match():
    result = spatial(labels([1, 0, 2]), labels([7, 0, 9]), {1: "left", 2: "right"}, {7: "right", 9: "left"})
    assert result["exact_geometry_match"]
    assert not result["side_assignment_match"]
    assert not result["exact_instance_match"]


def test_spatial_invalid_arrays_and_registries_reject():
    with pytest.raises(ValueError, match="shape"):
        spatial(labels([1, 0]), labels([1]))
    for bad in (np.ones((2, 1, 1), dtype=float), np.array([-1, 0]).reshape(2, 1, 1), np.ones((2, 1), dtype=int)):
        with pytest.raises(ValueError):
            spatial(bad, bad)
    with pytest.raises(ValueError, match="registry"):
        spatial(labels([1, 0]), labels([1, 0]), {1: "unknown"}, {1: "left"})


def test_mixed_integer_dtypes_preserve_sparse_large_ids():
    reference = np.array([2**63 + 1, 0, 2**63 + 2], dtype=np.uint64).reshape(3, 1, 1)
    prediction = np.array([7, 0, 9], dtype=np.int64).reshape(3, 1, 1)
    result = spatial(reference, prediction)
    assert result["exact_instance_match"]
    assert [group["reference_ids"] for group in result["groups"]] == [[2**63 + 1], [2**63 + 2]]


def test_shared_axis_reindexing_does_not_change_spatial_result():
    reference = np.zeros((3, 4, 5), dtype=np.uint16)
    reference[0, 1:3, 1:3] = 1
    reference[2, 1:3, 1:3] = 2
    prediction = reference * 3
    assert spatial(reference, prediction) == spatial(reference[::-1].transpose(2, 1, 0), prediction[::-1].transpose(2, 1, 0))


def test_demo_runs_offline_and_refuses_to_overwrite_output(tmp_path):
    output = tmp_path / "demo.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--demo", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    first = output.read_bytes()
    report = json.loads(first)
    assert report["evidence_type"] == "synthetic_engineering_only"
    assert report["clinical_accuracy"] == "NOT_MEASURED"
    assert report["inputs"]["census"]["evidence_type"] == "synthetic"
    assert len(report["scorer_sha256"]) == 64
    assert report["counts"]["endpoints"]["left"]["denominators"] == {"N": 3, "R": 3, "C": 2, "P": 2, "E": 1}
    assert report["spatial"]["compensating_merge_split"]["exact_instance_match"] is False
    repeated = subprocess.run([sys.executable, str(SCRIPT), "--demo", "--output", str(output)], capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == first
    second = tmp_path / "repeat.json"
    assert subprocess.run([sys.executable, str(SCRIPT), "--demo", "--output", str(second)], capture_output=True).returncode == 0
    assert second.read_bytes() == first
