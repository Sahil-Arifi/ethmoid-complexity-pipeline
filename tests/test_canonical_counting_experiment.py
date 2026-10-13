"""Runner boundaries use tiny observations and injected detector behavior."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import canonical_counting_experiment as experiment
from ridge_counting_experiment import ARRAY_NAMES, NULL_COUNTS, _sha
from synthetic_counting_experiment import array_hash


@pytest.fixture
def arrays():
    return {"hu": np.full((6, 5, 4), -900, np.float32),
            "roi": np.ones((6, 5, 4), bool), "affine": np.eye(4),
            "labels": np.full((6, 5, 4), 7, np.uint16),
            "highres_labels": np.full((12, 10, 8), 8, np.uint8),
            "cores": np.full((6, 5, 4), 9, np.uint16)}


def diagnostics(total=1):
    return {"status": "withheld" if total is None else "accepted",
            "counts": NULL_COUNTS.copy() if total is None else {"left": 0, "right": total, "total": total},
            "withheld_reasons": ["unresolved_subvoxel_communication"] if total is None else [],
            "experimental_count_available": total is not None, "clinical_accuracy": "NOT_MEASURED",
            "component_bridge_evidence": {"component_count": 2 if total is None else 1}}


def prediction(hu):
    shape = hu.shape
    return {"partitions": np.ones(shape, np.int32), "cavities": np.ones(shape, np.uint16),
            "air": np.ones(shape, np.uint16), "component_bridge": np.zeros(shape, np.uint8),
            "cells": [{"id": 1, "side": "right", "centroid_ras_mm": ((np.array(shape) - 1) / 2).tolist(),
                       "voxel_count": int(hu.size), "air_voxel_count": int(hu.size)}],
            "warnings": [], "diagnostics": diagnostics()}


def metadata(name, physical=1, baseline=1, group="known_development"):
    return {"name": name, "group": group, "original_group": group, "physical_count": physical,
            "baseline_detector_diagnostics": diagnostics(baseline), "baseline_failure": None,
            "reference_old_partition_hash": "old-partitions", "equivalence_group": None}


def _frozen(tmp_path, monkeypatch, count=0):
    known = tmp_path / "native" / "run-01"
    known.mkdir(parents=True)
    ridge = tmp_path / "ridge" / "run-01"
    ridge.mkdir(parents=True)
    (ridge / "saved.txt").write_text("old ridge evidence")
    (known.parent / "failed-audit.json").write_text("retain failed audit")
    (known / "protocol.json").write_text(json.dumps({"known_run": str(ridge)}))
    protected = {str(root): {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()}
                 for root in (known.parent, ridge)}
    snapshot = {"known_run": str(known), "case_count": count, "source": "frozen", "protected_files": protected}
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot)
    plan = tmp_path / "plan.json"
    experiment.freeze(plan, known)
    return known, ridge, plan, snapshot


def test_measurement_receives_only_independent_readonly_observations(arrays, monkeypatch):
    def detector(*, hu, roi, affine):
        for key, value in (("hu", hu), ("roi", roi), ("affine", affine)):
            assert not value.flags.writeable and not np.shares_memory(value, arrays[key])
            np.testing.assert_array_equal(value, arrays[key])
        return prediction(hu)
    monkeypatch.setattr(experiment, "count_with_canonical_ridges", detector, raising=False)
    record, masks = experiment.measure(arrays)
    changed = arrays | {key: object() for key in ("labels", "cores", "highres_labels")}
    second, changed_masks = experiment.measure(changed)
    assert record == second
    for key in masks:
        np.testing.assert_array_equal(masks[key], changed_masks[key])


@pytest.mark.parametrize("target", ["hu", "roi", "affine"])
def test_mutation_is_detected_even_after_write_flag_is_enabled(arrays, monkeypatch, target):
    original = {key: value.copy() for key, value in arrays.items()}
    def detector(**inputs):
        value = inputs[target]
        value.setflags(write=True)
        value.flat[0] = not value.flat[0] if value.dtype == bool else value.flat[0] + 1
        return prediction(inputs["hu"])
    monkeypatch.setattr(experiment, "count_with_canonical_ridges", detector, raising=False)
    with pytest.raises(ValueError, match="mutated"):
        experiment.measure(arrays)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original[key])


def test_middle_failure_preserves_complete_census_and_null_counts(arrays, tmp_path, monkeypatch):
    known, _, plan, frozen = _frozen(tmp_path, monkeypatch, count=3)
    items = [(arrays, metadata(name)) for name in ("first", "broken", "last")]
    monkeypatch.setattr(experiment, "_items", lambda path: items)
    calls = []
    def detector(**inputs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected middle failure")
        return prediction(inputs["hu"])
    monkeypatch.setattr(experiment, "count_with_canonical_ridges", detector, raising=False)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, destination)
    summary = json.loads((destination / "results.json").read_text())
    assert len(calls) == summary["case_count"] == len(summary["results"]) == 3
    assert summary["execution_errors"] == 1 and summary["historical_files_preserved"]
    assert [r["metadata"]["name"] for r in summary["results"]] == ["first", "broken", "last"]
    broken = summary["results"][1]
    assert broken["failure"] == "RuntimeError: injected middle failure"
    assert broken["candidate"]["diagnostics"]["counts"] == NULL_COUNTS
    assert summary["results"][2]["failure"] is None
    for row in summary["results"]:
        folder = destination / row["metadata"]["name"]
        assert json.loads((folder / "result.json").read_text()) == row
        for filename, hashes in (("input.npz", row["input_hashes"]), ("candidate.npz", row["output_hashes"])):
            with np.load(folder / filename, allow_pickle=False) as saved:
                assert set(saved.files) == set(hashes)
                assert {key: array_hash(saved[key]) for key in saved.files} == hashes
    assert json.loads((destination / "protocol.json").read_text()) == frozen
    manifest = json.loads((destination / "sha256.json").read_text())
    assert manifest == {p.relative_to(destination).as_posix(): _sha(p) for p in destination.rglob("*")
                        if p.is_file() and p.name != "sha256.json"}
    assert "Withholding is not a correct numerical prediction" in (destination / "report.md").read_text()


def test_frozen_drift_refuses_output_and_measurement(tmp_path, monkeypatch):
    known, _, plan, snapshot = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot | {"source": "changed"})
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("loaded after drift"))
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="Frozen"):
        experiment.execute(plan, destination)
    assert not destination.exists()


def test_plan_and_output_cannot_be_overwritten(tmp_path, monkeypatch):
    known, _, plan, _ = _frozen(tmp_path, monkeypatch)
    before = plan.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.freeze(plan, known)
    assert plan.read_bytes() == before
    destination = tmp_path / "run"
    destination.mkdir()
    sentinel = destination / "results.json"
    sentinel.write_text("existing")
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("loaded for existing output"))
    with pytest.raises(FileExistsError):
        experiment.execute(plan, destination)
    assert sentinel.read_text() == "existing"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("tree", ["native", "ridge"])
@pytest.mark.parametrize("operation", ["freeze", "execute"])
def test_both_protected_trees_reject_plan_and_output(tmp_path, monkeypatch, tree, operation):
    known, ridge, plan, _ = _frozen(tmp_path, monkeypatch)
    root = known.parent if tree == "native" else ridge
    target = root / "new" / ("plan.json" if operation == "freeze" else "run")
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("loaded before protected-path rejection"))
    with pytest.raises(ValueError, match="protected"):
        experiment.freeze(target, known) if operation == "freeze" else experiment.execute(plan, target)
    assert not target.parent.exists()


def _historical_rows(arrays, tmp_path):
    hashes = {key: array_hash(value) for key, value in arrays.items()}
    old = np.ones(arrays["hu"].shape, np.int32)
    rows = [{"metadata": {"name": f"case_{i}", "group": "known_development", "physical_count": 1},
             "input_hashes": hashes, "output_hashes": {"partitions": array_hash(old)},
             "detector_diagnostics": diagnostics(), "failure": None, "evidence": {"pairs": []}}
            for i in range(100)]
    folder = tmp_path / "case_0"
    folder.mkdir()
    np.savez_compressed(folder / "input.npz", **arrays)
    np.savez_compressed(folder / "evidence.npz", partitions=old)
    (folder / "result.json").write_text(json.dumps(rows[0]))
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    return rows, folder


@pytest.mark.parametrize("defect", ["extra_input", "missing_input", "input_hash", "extra_evidence",
                                    "missing_evidence", "evidence_hash", "row", "duplicate", "count"])
def test_saved_case_requires_exact_manifests_hashes_and_census(arrays, tmp_path, defect):
    rows, folder = _historical_rows(arrays, tmp_path)
    if defect in ("extra_input", "missing_input", "input_hash"):
        saved = arrays.copy()
        if defect == "extra_input": saved["unexpected"] = np.ones(1)
        if defect == "missing_input": del saved["cores"]
        if defect == "input_hash": saved["hu"] = arrays["hu"] + 1
        np.savez_compressed(folder / "input.npz", **saved)
    if defect in ("extra_evidence", "missing_evidence", "evidence_hash"):
        saved = {"partitions": np.ones(arrays["hu"].shape, np.int32)}
        if defect == "extra_evidence": saved["unexpected"] = np.ones(1)
        if defect == "missing_evidence": saved = {}
        if defect == "evidence_hash": saved["partitions"] *= 2
        np.savez_compressed(folder / "evidence.npz", **saved)
    if defect == "row": (folder / "result.json").write_text("{}")
    if defect == "duplicate": rows[-1] = rows[0]
    if defect == "count": rows.pop()
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    with pytest.raises(ValueError):
        experiment._items(tmp_path)


def test_report_keeps_withheld_errors_and_lost_correct_counts_distinct():
    rows = []
    for name, physical, baseline, current, failure, group in (
            ("kept", 1, 1, 1, None, "known_development"),
            ("lost", 2, 2, None, None, "known_development"),
            ("wrong", 1, 2, None, None, "new_perturbations"),
            ("broken", 1, 1, None, "error", "new_perturbations")):
        rows.append({"metadata": metadata(name, physical, baseline, group), "failure": failure,
                     "candidate": {"diagnostics": diagnostics(current)}, "input_hashes": {}, "output_hashes": {}})
    summary = experiment.summarize(rows)
    table = {r["group"]: r for r in summary["comparison"]}
    assert table["known_development"]["baseline"] == {"correct_accepted": 2, "wrong_accepted": 0, "withheld": 0, "errors": 0}
    assert table["known_development"]["candidate"] == {"correct_accepted": 1, "wrong_accepted": 0, "withheld": 1, "errors": 0}
    assert table["known_development"]["correct_accepted_lost"] == ["lost"]
    assert table["new_perturbations"]["candidate"] == {"correct_accepted": 0, "wrong_accepted": 0, "withheld": 1, "errors": 1}
    assert summary["gates"]["all_eight_prior_wrong_acceptances_withheld"] is False
    assert summary["gates"]["all_baseline_correct_single_counts_preserved"] is False
    assert summary["spatial_reindexing_audit"] == "NOT_PERFORMED"


def test_all_saved_rows_load_with_original_metadata_and_partition_provenance(arrays, tmp_path):
    rows, first = _historical_rows(arrays, tmp_path)
    for row in rows[1:]:
        folder = tmp_path / row["metadata"]["name"]
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(row))
        for filename in ("input.npz", "evidence.npz"):
            (folder / filename).write_bytes((first / filename).read_bytes())
    items = experiment._items(tmp_path)
    assert len(items) == 100
    for (loaded, info), row in zip(items, rows):
        assert set(loaded) == set(ARRAY_NAMES)
        assert all(not value.flags.writeable for value in loaded.values())
        assert info["name"] == row["metadata"]["name"]
        assert info["original_group"] == row["metadata"]["group"]
        assert info["physical_count"] == row["metadata"]["physical_count"]
        assert info["baseline_detector_diagnostics"] == row["detector_diagnostics"]
        assert info["reference_old_partition_hash"] == row["output_hashes"]["partitions"]


def test_snapshot_tracks_rules_runtime_input_and_both_protected_trees(arrays, tmp_path, monkeypatch):
    # Restore the real snapshot after using the boundary fixture to create history.
    snapshot_function = experiment._snapshot
    known, ridge, _, _ = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "_snapshot", snapshot_function)
    monkeypatch.setattr(experiment, "_items", lambda path: [(arrays, metadata("case"))])
    fake_repo = tmp_path / "source"
    paths = ["src/ethmoid_pipeline/counting/engine.py", "docs/CANONICAL_COUNTING_EXPERIMENT.md"]
    paths += ["scripts/" + name for name in ("canonical_ridge_candidate.py", "canonical_counting_experiment.py",
              "native_passage_evidence.py", "ridge_counting_candidate.py", "ridge_counting_experiment.py",
              "synthetic_counting_experiment.py", "synthetic_blur_experiment.py")]
    for relative in paths:
        path = fake_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source")
    monkeypatch.setattr(experiment, "REPO", fake_repo)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    monkeypatch.setattr(experiment, "version", lambda package: "1.0")
    baseline = experiment._snapshot(known)
    assert set(baseline["sources"]) == set(paths)
    assert set(baseline["protected_files"]) == {str(known.parent), str(ridge)}
    assert baseline["source_commit"] == "commit"
    for relative in ("docs/CANONICAL_COUNTING_EXPERIMENT.md", "scripts/canonical_ridge_candidate.py"):
        path = fake_repo / relative
        path.write_text("changed")
        assert experiment._snapshot(known) != baseline
        path.write_text("frozen source")
    arrays["hu"][0, 0, 0] += 1
    assert experiment._snapshot(known) != baseline
    arrays["hu"][0, 0, 0] -= 1
    monkeypatch.setattr(experiment, "version", lambda package: "2.0")
    assert experiment._snapshot(known) != baseline
    monkeypatch.setattr(experiment, "version", lambda package: "1.0")
    for protected in (known.parent / "failed-audit.json", ridge / "saved.txt"):
        old = protected.read_text()
        protected.write_text("changed")
        assert experiment._snapshot(known) != baseline
        protected.write_text(old)


def test_count_gates_require_all_eight_wrong_acceptances_and_consistent_twins():
    rows = []
    for index in range(8):
        rows.append({"metadata": metadata(f"wrong_{index}", physical=1, baseline=2, group="new_perturbations"),
                     "candidate": {"diagnostics": diagnostics(None)}, "failure": None,
                     "input_hashes": {}, "output_hashes": {}})
    for pair in range(4):
        for physical in (1, 2):
            info = metadata(f"twin_{pair}_{physical}", physical=physical, baseline=None)
            info["equivalence_group"] = f"twins_{pair}"
            rows.append({"metadata": info, "candidate": {"diagnostics": diagnostics(None)}, "failure": None,
                         "input_hashes": {key: "same" for key in ("hu", "roi", "affine")},
                         "output_hashes": {"partitions": "same"}})
    summary = experiment.summarize(rows)
    assert summary["gates"]["all_eight_prior_wrong_acceptances_withheld"]
    assert summary["gates"]["material_twins_consistent_and_withheld"]
    assert not summary["gates"]["complete_unique_census"]
    rows[0]["failure"] = "error"
    rows[-1]["output_hashes"]["partitions"] = "different"
    summary = experiment.summarize(rows)
    assert not summary["gates"]["all_eight_prior_wrong_acceptances_withheld"]
    assert not summary["gates"]["material_twins_consistent_and_withheld"]


def test_new_acceptance_with_wall_bridge_fails_gate():
    row = {"metadata": metadata("bad_wall", baseline=None), "candidate": {"diagnostics": diagnostics()},
           "failure": None, "input_hashes": {}, "output_hashes": {}}
    row["candidate"]["diagnostics"]["component_bridge_evidence"]["component_count"] = 2
    summary = experiment.summarize([row])
    assert summary["acceptance_increases"] == ["bad_wall"]
    assert not summary["gates"]["no_acceptance_increase_from_wall_path"]


def test_mutated_history_is_reported_after_complete_execution(arrays, tmp_path, monkeypatch):
    known, ridge, plan, _ = _frozen(tmp_path, monkeypatch, count=1)
    monkeypatch.setattr(experiment, "_items", lambda path: [(arrays, metadata("case"))])
    def detector(**inputs):
        (ridge / "saved.txt").write_text("injected mutation")
        return prediction(inputs["hu"])
    monkeypatch.setattr(experiment, "count_with_canonical_ridges", detector)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, destination)
    summary = json.loads((destination / "results.json").read_text())
    assert summary["case_count"] == 1 and summary["execution_errors"] == 0
    assert not summary["historical_files_preserved"]
    assert not summary["gates"]["protected_files_preserved"]
