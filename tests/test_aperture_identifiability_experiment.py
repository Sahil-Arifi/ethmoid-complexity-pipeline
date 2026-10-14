"""Tiny synthetic observations exercise the experiment's evidence boundaries."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aperture_identifiability_experiment as experiment
from ridge_counting_experiment import _sha
from synthetic_counting_experiment import array_hash


@pytest.fixture
def arrays():
    sealed = np.full((2, 2, 2), 800.0)
    opened = sealed.copy()
    opened[0, 0, 0] = 790.0
    alternatives = []
    for deficit in (100.0, 200.0, 300.0):
        alternative = sealed.copy()
        alternative[0, 0, 0] -= deficit
        alternatives.append(alternative)
    return {"sealed": sealed, "open": opened, "midpoint": (sealed + opened) / 2,
            **{f"coarse_{i}": value for i, value in enumerate(alternatives)}, "affine": np.eye(4)}


def _compatible(observed, hypotheses, error_bound_hu):
    residuals = [float(np.max(np.abs(observed - hypothesis))) for hypothesis in hypotheses]
    return {"residual_linf": residuals,
            "compatible_indices": [i for i, residual in enumerate(residuals) if residual <= error_bound_hu]}


def _frozen(tmp_path, monkeypatch, count=3):
    roots = tuple(tmp_path / name for name in ("canonical", "native", "ridge"))
    for root in roots:
        root.mkdir()
        (root / "historical.txt").write_text("protected evidence", encoding="utf-8")
    known = roots[0] / "run-01"
    known.mkdir()
    protected = {str(root): {p.relative_to(root).as_posix(): _sha(p)
                            for p in root.rglob("*") if p.is_file()} for root in roots}
    source_scenarios = getattr(experiment, "scenarios", lambda: [{"transform": "base"}] * count)()
    items = [dict(source_scenarios[i % len(source_scenarios)], name=f"tiny_{i}") for i in range(count)]
    snapshot = {"known_run": str(known), "case_count": count, "scenario_count": count,
                "scenarios": items, "source": "frozen", "runtime": {"numpy": "fixed"},
                "protected_files": protected}
    monkeypatch.setattr(experiment, "_protected_roots", lambda path: roots, raising=False)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot, raising=False)
    monkeypatch.setattr(experiment, "scenarios", lambda: items, raising=False)
    plan = tmp_path / "plan.json"
    experiment.freeze(plan, known)
    return known, roots, plan, snapshot


def test_midpoint_overlap_and_open_observation_use_different_bounds(arrays):
    result = experiment.measure(arrays)
    assert result["delta_hu"] == 10.0 and result["changed_voxels"] == 1
    by_error = {row["error_bound_hu"]: row for row in result["by_error"]}
    assert set(by_error) == {0, 1, 5, 25}
    assert by_error[5]["pair_midpoint"]["compatible_indices"] == [0, 1]
    assert by_error[5]["overlap_expected"] is True
    assert by_error[5]["pair_open"]["compatible_indices"] == [1]
    assert by_error[25]["restricted_sealed_only"] is True
    assert by_error[25]["restricted_open"]["compatible_indices"] == [0]
    assert all(row["augmented_has_opening"] for row in by_error.values())
    assert all("counts" not in row for row in by_error.values())


def test_numerical_tolerance_does_not_expand_the_observation_error(arrays):
    arrays["open"][0, 0, 0] = 800.0 - 10.000001
    arrays["midpoint"] = (arrays["sealed"] + arrays["open"]) / 2
    rows = {row["error_bound_hu"]: row for row in experiment.measure(arrays)["by_error"]}
    assert rows[5]["overlap_expected"] is False
    assert rows[5]["pair_midpoint"]["compatible_indices"] == []


def test_compatibility_receives_copied_readonly_arrays_without_metadata(arrays, monkeypatch):
    calls = []
    def compatible(observed, hypotheses, error_bound_hu):
        calls.append(1)
        for value in (observed, *hypotheses):
            assert isinstance(value, np.ndarray) and not value.flags.writeable
            assert all(not np.shares_memory(value, original) for original in arrays.values())
        return _compatible(observed, hypotheses, error_bound_hu)
    monkeypatch.setattr(experiment, "compatibility", compatible, raising=False)
    first = experiment.measure(arrays)
    second = experiment.measure(arrays | {"physical_count": object(), "construction": object()})
    assert calls and first == second


@pytest.mark.parametrize("target", ["observation", "hypothesis"])
def test_mutation_is_detected_even_if_readonly_flag_is_reenabled(arrays, monkeypatch, target):
    originals = {key: value.copy() for key, value in arrays.items()}
    def compatible(observed, hypotheses, error_bound_hu):
        value = observed if target == "observation" else hypotheses[0]
        value.setflags(write=True)
        value.flat[0] += 1
        return _compatible(observed, hypotheses, error_bound_hu)
    monkeypatch.setattr(experiment, "compatibility", compatible, raising=False)
    with pytest.raises(ValueError, match="mutat"):
        experiment.measure(arrays)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], originals[key])


@pytest.mark.parametrize("field", ["source", "runtime"])
def test_frozen_drift_prevents_construction_and_output(tmp_path, monkeypatch, field):
    _, _, plan, snapshot = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot | {field: "changed"})
    monkeypatch.setattr(experiment, "construct", lambda info: pytest.fail("constructed after drift"), raising=False)
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="[Ff]rozen"):
        experiment.execute(plan, destination)
    assert not destination.exists()


def test_existing_plan_and_output_remain_unchanged(tmp_path, monkeypatch):
    known, _, plan, _ = _frozen(tmp_path, monkeypatch)
    before = plan.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.freeze(plan, known)
    assert plan.read_bytes() == before
    destination = tmp_path / "run"
    destination.mkdir()
    sentinel = destination / "results.json"
    sentinel.write_text("existing evidence", encoding="utf-8")
    monkeypatch.setattr(experiment, "construct", lambda info: pytest.fail("constructed for existing output"), raising=False)
    with pytest.raises(FileExistsError):
        experiment.execute(plan, destination)
    assert sentinel.read_text(encoding="utf-8") == "existing evidence"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("tree_index", [0, 1, 2])
def test_protected_trees_reject_new_plan_and_run(tmp_path, monkeypatch, tree_index):
    known, roots, plan, _ = _frozen(tmp_path, monkeypatch)
    target = roots[tree_index] / "new"
    with pytest.raises(ValueError, match="protected"):
        experiment.freeze(target / "plan.json", known)
    with pytest.raises(ValueError, match="protected"):
        experiment.execute(plan, target / "run")
    assert not target.exists()


def test_middle_measurement_failure_retains_arrays_census_and_exact_manifest(arrays, tmp_path, monkeypatch):
    _, _, plan, frozen = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "construct", lambda info: arrays, raising=False)
    original_measure = experiment.measure
    calls = []
    def measured(values):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected middle failure")
        return original_measure(values)
    monkeypatch.setattr(experiment, "measure", measured)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, destination)
    saved = json.loads((destination / "results.json").read_text())
    assert len(calls) == len(saved["results"]) == 3
    assert saved["execution_errors"] == 1 and saved["historical_files_preserved"]
    assert [row["metadata"]["name"] for row in saved["results"]] == ["tiny_0", "tiny_1", "tiny_2"]
    broken = saved["results"][1]
    assert broken["measurement"] is None
    assert broken["failure"] == "RuntimeError: injected middle failure"
    assert saved["results"][0]["failure"] is None and saved["results"][2]["failure"] is None
    for row in saved["results"]:
        folder = destination / row["metadata"]["name"]
        assert json.loads((folder / "result.json").read_text()) == row
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            assert {name: array_hash(archive[name]) for name in archive.files} == row["array_hashes"]
    assert json.loads((destination / "protocol.json").read_text()) == frozen
    manifest = json.loads((destination / "sha256.json").read_text())
    assert manifest == {p.relative_to(destination).as_posix(): _sha(p)
                        for p in destination.rglob("*") if p.is_file() and p.name != "sha256.json"}


def _manifest(root):
    inventory = {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*")
                 if p.is_file() and p != root / "sha256.json"}
    (root / "sha256.json").write_text(json.dumps(inventory), encoding="utf-8")


def _tiny_history(tmp_path):
    known = tmp_path / "history" / "run-01"
    known.mkdir(parents=True)
    protocol = {"source": "historical", "case_count": 100}
    for path in (known / "protocol.json", known.parent / "plan.json"):
        path.write_text(json.dumps(protocol), encoding="utf-8")
    inputs = {"hu": np.full((2, 2, 2), -900., np.float32), "roi": np.ones((2, 2, 2), bool),
              "affine": np.eye(4), "labels": np.ones((2, 2, 2), np.uint16),
              "highres_labels": np.ones((2, 2, 2), np.uint8), "cores": np.ones((2, 2, 2), np.uint16)}
    outputs = {"partitions": np.ones((2, 2, 2), np.int32), "cavities": np.ones((2, 2, 2), np.uint16),
               "air": np.ones((2, 2, 2), np.uint16), "component_bridge": np.zeros((2, 2, 2), np.uint8)}
    rows = []
    for i in range(100):
        name = f"case_{i}"
        folder = known / name
        folder.mkdir()
        for filename, values in (("input.npz", inputs), ("candidate.npz", outputs)):
            if i == 0:
                np.savez_compressed(folder / filename, **values)
            else:
                (folder / filename).write_bytes((known / "case_0" / filename).read_bytes())
        accepted = i < 22
        row = {"metadata": {"name": name, "physical_count": 1}, "failure": None,
               "input_hashes": {key: array_hash(value) for key, value in inputs.items()},
               "output_hashes": {key: array_hash(value) for key, value in outputs.items()},
               "candidate": {"diagnostics": {"status": "accepted" if accepted else "withheld",
                   "counts": {"left": 0 if accepted else None, "right": 1 if accepted else None,
                              "total": 1 if accepted else None}}}}
        (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")
        rows.append(row)
    (known / "results.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
    _manifest(known)
    return known, rows, inputs


def test_historical_reader_validates_all_100_records_without_counting(tmp_path):
    known, rows, _ = _tiny_history(tmp_path)
    result = experiment._history(known)
    assert result["case_count"] == len(result["records"]) == 100
    assert result["detector_rerun"] is False
    assert result["counts"] == {"correct_accepted": 22, "wrong_accepted": 0, "withheld": 78, "errors": 0}
    assert [row["name"] for row in result["records"]] == [row["metadata"]["name"] for row in rows]
    assert result["records"][0]["saved_counts"]["total"] == 1
    assert result["records"][-1]["saved_counts"]["total"] is None


@pytest.mark.parametrize("defect", ["manifest", "protocol", "row", "array_hash", "coherent_hash_edit"])
def test_historical_reader_rejects_changed_evidence(tmp_path, defect):
    known, rows, inputs = _tiny_history(tmp_path)
    if defect == "manifest":
        (known / "unlisted.txt").write_text("new evidence", encoding="utf-8")
    elif defect == "protocol":
        (known / "protocol.json").write_text(json.dumps({"source": "changed"}), encoding="utf-8")
        _manifest(known)
    elif defect == "row":
        (known / "case_0" / "result.json").write_text("{}", encoding="utf-8")
        _manifest(known)
    else:
        inputs["hu"] = inputs["hu"] + 1
        np.savez_compressed(known / "case_0" / "input.npz", **inputs)
        if defect == "coherent_hash_edit":
            rows[0]["input_hashes"]["hu"] = array_hash(inputs["hu"])
            (known / "case_0" / "result.json").write_text(json.dumps(rows[0]), encoding="utf-8")
            (known / "results.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
        else:
            _manifest(known)
    with pytest.raises(ValueError, match="Historical"):
        experiment._history(known)
