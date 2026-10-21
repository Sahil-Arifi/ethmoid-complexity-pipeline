"""Synthetic evaluation arithmetic and geometry checks; no inference or clinical importer."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np


ENDPOINTS = ("left", "right", "total")
PREDICTION_STATES = ("available", "withheld", "technical_failure", "not_run", "unknown")
REFERENCE_STATES = ("final", "provisional", "missing", "conflict")


def _fields(value, required, name):
    if not isinstance(value, dict) or not set(required).issubset(value):
        raise ValueError(f"{name} must be an object with fields: {', '.join(required)}.")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")
    return value


def _count(value, name, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} count must be a nonnegative integer, excluding booleans.")


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Invalid {name}.")


def _reference(reference, source):
    _fields(reference, ("state", "count", "independence", "source_sha256", "scope", "locked_at", "evidence_ref"), "Reference")
    state = reference["state"]
    _choice(state, REFERENCE_STATES, "reference state")
    _count(reference["count"], "Reference", nullable=state != "final")
    if state in ("missing", "conflict") and reference["count"] is not None:
        raise ValueError("Missing/conflicting reference count must be null.")
    _choice(reference["independence"], ("independent", "model_assisted", "unknown"), "reference independence")
    _choice(reference["scope"], ("full_3d", "landmark_2d", "partial_volume", "unknown"), "reference scope")
    if reference["source_sha256"] is not None:
        _hash(reference["source_sha256"], "Reference source")
    if reference["locked_at"] is not None:
        _text(reference["locked_at"], "Reference lock")
        try:
            locked = datetime.fromisoformat(reference["locked_at"])
        except ValueError as error:
            raise ValueError("Reference lock must be an ISO timestamp.") from error
        if locked.tzinfo is None:
            raise ValueError("Reference lock timestamp must include a timezone.")
    if reference["evidence_ref"] is not None:
        _text(reference["evidence_ref"], "Reference evidence")
    requirements = {
        "reference_not_final": state == "final",
        "reference_not_independent": reference["independence"] == "independent",
        "reference_source_unmatched": reference["source_sha256"] == source,
        "reference_scope_unmatched": reference["scope"] == "full_3d",
        "reference_unlocked": reference["locked_at"] is not None,
        "reference_evidence_missing": reference["evidence_ref"] is not None,
    }
    return [reason for reason, satisfied in requirements.items() if not satisfied]


def _prediction(prediction, source, method):
    _fields(prediction, ("state", "count", "uncertainty"), "Prediction")
    state = prediction["state"]
    _choice(state, PREDICTION_STATES, "prediction state")
    _count(prediction["count"], "Prediction", nullable=state != "available")
    if state != "available" and prediction["count"] is not None:
        raise ValueError("Unavailable prediction count must be null.")
    flags = prediction["uncertainty"]
    if not isinstance(flags, list) or any(not isinstance(flag, str) or not flag.strip() for flag in flags):
        raise ValueError("Prediction uncertainty must be a list of nonempty strings.")
    if state == "available":
        _fields(prediction, ("kind", "source_sha256", "method_sha256", "run_id"), "Available prediction")
        if prediction["kind"] != "original_automatic":
            raise ValueError("Available prediction must be original automatic output.")
        if prediction["source_sha256"] != source or prediction["method_sha256"] != method:
            raise ValueError("Available prediction source/method does not match census.")
        _text(prediction["run_id"], "Prediction run")
    else:
        _fields(prediction, ("reason",), "Unavailable prediction")
        _text(prediction["reason"], "Unavailable prediction reason")


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _totals(case_id, records):
    result = {"case_id": case_id, "reference": "not_checkable", "prediction": "not_checkable"}
    available = [records[case_id, endpoint]["prediction"] for endpoint in ENDPOINTS
                 if (case_id, endpoint) in records and records[case_id, endpoint]["prediction"]["state"] == "available"]
    if len({entry["run_id"] for entry in available}) > 1:
        raise ValueError("Prediction endpoints must come from the same original run.")
    for kind, state in (("reference", "final"), ("prediction", "available")):
        entries = {endpoint: records[case_id, endpoint][kind] for endpoint in ENDPOINTS
                   if (case_id, endpoint) in records and records[case_id, endpoint][kind]["state"] == state}
        total = entries.get("total")
        if total is None:
            continue
        sides = [entries[side] for side in ("left", "right") if side in entries]
        if kind == "reference":
            sides = [side for side in sides if side["source_sha256"] is not None and
                     side["source_sha256"] == total["source_sha256"] and side["scope"] == total["scope"] == "full_3d"]
        if any(side["count"] > total["count"] for side in sides):
            raise ValueError(f"{kind.capitalize()} total is smaller than a known side count.")
        if len(sides) != 2:
            continue
        if sum(side["count"] for side in sides) != total["count"]:
            raise ValueError(f"{kind.capitalize()} total conflicts with left plus right; resolve the record conflict.")
        result[kind] = "consistent"
    return result


def evaluate_counts(census: dict, records: list[dict]) -> dict:
    """Score normalized synthetic snapshots; supplied metadata is not authenticated evidence."""
    _fields(census, ("schema_version", "evidence_type", "evaluation_freeze_id", "method_sha256", "cases"), "Census")
    if type(census["schema_version"]) is not int or census["schema_version"] != 1 or census["evidence_type"] != "synthetic":
        raise ValueError("Only the version 1 synthetic scoring schema is supported.")
    freeze = _text(census["evaluation_freeze_id"], "Evaluation freeze")
    method = _hash(census["method_sha256"], "Method")
    if not isinstance(census["cases"], list) or not isinstance(records, list):
        raise ValueError("Census cases and records must be lists.")
    cases, participants, sources = {}, set(), set()
    for case in census["cases"]:
        _fields(case, ("case_id", "participant_group_id", "source_sha256"), "Census case")
        identifier = _text(case["case_id"], "Case ID")
        participant = _text(case["participant_group_id"], "Participant group")
        source = _hash(case["source_sha256"], "Source")
        if identifier in cases or participant in participants or source in sources:
            raise ValueError("Census requires unique cases, sources and one case per participant.")
        cases[identifier] = case
        participants.add(participant)
        sources.add(source)
    current = {}
    for record in records:
        _fields(record, ("evaluation_freeze_id", "case_id", "endpoint", "reference", "prediction"), "Record")
        if record["evaluation_freeze_id"] != freeze:
            raise ValueError("Record freeze does not match census.")
        key = (_text(record["case_id"], "Record case ID"), _text(record["endpoint"], "Record endpoint"))
        if key[0] not in cases or key[1] not in ENDPOINTS:
            raise ValueError("Record endpoint is outside the scheduled census.")
        if key in current:
            raise ValueError("Reject duplicate current records; resolve revisions before scoring.")
        source = cases[key[0]]["source_sha256"]
        _reference(record["reference"], source)
        _prediction(record["prediction"], source, method)
        current[key] = record
    totals = [_totals(identifier, current) for identifier in sorted(cases)]
    rows = []
    for identifier, case in sorted(cases.items()):
        for endpoint in ENDPOINTS:
            record = current.get((identifier, endpoint))
            reference = record["reference"] if record else {"state": "missing", "count": None}
            prediction = record["prediction"] if record else {"state": "unknown", "count": None, "uncertainty": [], "reason": "scheduled_unit_has_no_record"}
            excluded = _reference(reference, case["source_sha256"]) if record else ["scheduled_unit_has_no_record"]
            available = prediction["state"] == "available"
            error = prediction["count"] - reference["count"] if available and not excluded else None
            rows.append({"case_id": identifier, "participant_group_id": case["participant_group_id"],
                         "endpoint": endpoint, "reference_state": reference["state"],
                         "reference_count": reference["count"], "reference_exclusions": excluded,
                         "prediction_state": prediction["state"], "predicted_count": prediction["count"],
                         "prediction_uncertainty": list(prediction["uncertainty"]),
                         "prediction_reason": prediction.get("reason", ""),
                         "signed_error": error, "absolute_error": abs(error) if error is not None else None})
    endpoints = {}
    for endpoint in ENDPOINTS:
        selected = [row for row in rows if row["endpoint"] == endpoint]
        errors = [row["signed_error"] for row in selected if row["signed_error"] is not None]
        absolute = [abs(error) for error in errors]
        n, r, c = len(selected), sum(not row["reference_exclusions"] for row in selected), sum(row["prediction_state"] == "available" for row in selected)
        p, e = len(errors), errors.count(0)
        endpoints[endpoint] = {
            "denominators": {"N": n, "R": r, "C": c, "P": p, "E": e},
            "prediction_states": {state: sum(row["prediction_state"] == state for row in selected) for state in PREDICTION_STATES},
            "reference_states": {state: sum(row["reference_state"] == state for row in selected) for state in REFERENCE_STATES},
            "reference_exclusions": dict(sorted(Counter(reason for row in selected for reason in row["reference_exclusions"]).items())),
            "uncertain_available": sum(row["prediction_state"] == "available" and bool(row["prediction_uncertainty"]) for row in selected),
            "mean_signed_error": statistics.mean(errors) if errors else None,
            "mean_absolute_error": statistics.mean(absolute) if absolute else None,
            "median_absolute_error": statistics.median(absolute) if absolute else None,
            "max_absolute_error": max(absolute) if absolute else None,
            "exact_agreement": _ratio(e, p), "answer_coverage": _ratio(c, n),
            "reference_coverage": _ratio(r, n), "end_to_end_exact": _ratio(e, r),
            "census_exact": _ratio(e, n),
        }
    return {"schema_version": 1, "evidence_type": "synthetic", "evaluation_freeze_id": freeze,
            "method_sha256": method, "endpoints": endpoints, "rows": rows, "total_consistency": totals}


def score_synthetic_instances(reference, prediction, reference_sides, prediction_sides):
    """Describe any-overlap groups on a complete, known shared synthetic grid.

    This is an exact construction diagnostic, not clinical correspondence or a tolerance rule.
    """
    if not isinstance(reference, np.ndarray) or not isinstance(prediction, np.ndarray):
        raise ValueError("Synthetic labels must be numpy arrays.")
    if reference.shape != prediction.shape:
        raise ValueError("Synthetic label arrays must have the same shape.")
    for array, registry in ((reference, reference_sides), (prediction, prediction_sides)):
        if array.ndim != 3 or not array.size or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError("Labels must be nonempty 3D arrays of nonnegative integer IDs.")
        ids = {int(i) for i in np.unique(array) if i}
        if not isinstance(registry, dict) or set(registry) != ids or any(type(i) is not int or side not in ("left", "right") for i, side in registry.items()):
            raise ValueError("Synthetic side registry must cover each nonzero label exactly.")
    graph = {("r", i): set() for i in sorted(reference_sides)}
    graph.update({("p", i): set() for i in sorted(prediction_sides)})
    both = (reference != 0) & (prediction != 0)
    # Python integer pairs avoid float promotion when signed and unsigned label arrays differ.
    for ref, pred in sorted(set(zip(reference[both].tolist(), prediction[both].tolist()))):
        a, b = ("r", int(ref)), ("p", int(pred))
        graph[a].add(b)
        graph[b].add(a)
    groups, visited = [], set()
    for start in graph:
        if start in visited:
            continue
        component, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph[node] - component)
        visited.update(component)
        refs = sorted(i for kind, i in component if kind == "r")
        preds = sorted(i for kind, i in component if kind == "p")
        relationship = ("extra" if not refs else "missed" if not preds else
                        "one_to_one" if len(refs) == len(preds) == 1 else
                        "merge" if len(preds) == 1 else "split" if len(refs) == 1 else "complex")
        exact = relationship == "one_to_one" and np.array_equal(reference == refs[0], prediction == preds[0])
        side = reference_sides[refs[0]] == prediction_sides[preds[0]] if relationship == "one_to_one" else None
        groups.append({"reference_ids": refs, "prediction_ids": preds, "relationship": relationship,
                       "exact_geometry_match": bool(exact), "side_assignment_match": side})
    geometry = all(group["exact_geometry_match"] for group in groups)
    sides = all(group["side_assignment_match"] is True for group in groups)
    return {"reference_count": len(reference_sides), "prediction_count": len(prediction_sides),
            "groups": groups, "exact_geometry_match": geometry, "side_assignment_match": sides,
            "exact_instance_match": geometry and sides}


def synthetic_demo():
    census = {"schema_version": 1, "evidence_type": "synthetic", "evaluation_freeze_id": "synthetic-demo-1",
              "method_sha256": "a" * 64, "cases": []}
    records = []
    for i, (reference, prediction) in enumerate(((4, 4), (5, None), (6, 8))):
        case = {"case_id": f"synthetic-{i}", "participant_group_id": f"group-{i}", "source_sha256": str(i + 1) * 64}
        census["cases"].append(case)
        records.append({"evaluation_freeze_id": census["evaluation_freeze_id"], "case_id": case["case_id"], "endpoint": "left",
                        "reference": {"state": "final", "count": reference, "independence": "independent",
                                      "source_sha256": case["source_sha256"], "scope": "full_3d",
                                      "locked_at": "2026-10-20T00:00:00+00:00", "evidence_ref": "invented reference"},
                        "prediction": {"state": "available" if prediction is not None else "withheld", "count": prediction,
                                       "kind": "original_automatic", "source_sha256": case["source_sha256"],
                                       "method_sha256": census["method_sha256"], "run_id": f"invented-run-{i}",
                                       "uncertainty": [], "reason": "invented example"}})
    examples = {
        "relabelled_exact": ([1, 1, 0, 2, 2], [7, 7, 0, 9, 9]),
        "merge": ([1, 1, 0, 2, 2], [7, 7, 7, 7, 7]),
        "split": ([1, 1, 1, 1, 0], [7, 7, 9, 9, 0]),
        "missed": ([1, 1, 0, 2, 2], [7, 7, 0, 0, 0]),
        "extra": ([1, 1, 0, 0, 0], [7, 7, 0, 9, 9]),
        "complex": ([1, 1, 2, 2], [7, 9, 7, 9]),
        "partial_overlap": ([1, 1, 0, 0], [0, 7, 7, 0]),
        "compensating_merge_split": ([1, 1, 0, 2, 2, 0, 3, 3], [5, 5, 0, 5, 5, 0, 6, 7]),
    }
    spatial = {}
    for name, (truth, detected) in examples.items():
        ref, pred = (np.array(values, dtype=np.uint16).reshape(-1, 1, 1) for values in (truth, detected))
        spatial[name] = score_synthetic_instances(ref, pred, {int(i): "left" for i in np.unique(ref) if i},
                                                 {int(i): "left" for i in np.unique(pred) if i})
    return {"evidence_type": "synthetic_engineering_only", "clinical_accuracy": "NOT_MEASURED",
            "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {"census": census, "records": records, "spatial_label_sequences": examples},
            "counts": evaluate_counts(census, records), "spatial": spatial}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True, help="Run built-in invented records only.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; an existing file is never overwritten.")
    args = parser.parse_args()
    payload = json.dumps(synthetic_demo(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except OSError as error:
        parser.exit(2, f"Could not create a new demo output: {error.strerror}\n")
    print("Synthetic scoring checks written. Clinical accuracy: NOT_MEASURED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
