"""Runner boundary checks with isolated tiny arrays and injected failures."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import native_passage_experiment as experiment


@pytest.fixture
def arrays():
    return {"hu": np.full((6, 5, 4), -900, np.float32),
            "roi": np.ones((6, 5, 4), bool), "affine": np.eye(4),
            "labels": np.full((6, 5, 4), 7, np.uint16),
            "highres_labels": np.full((12, 10, 8), 8, np.uint8),
            "cores": np.full((6, 5, 4), 9, np.uint16)}


def _partitions(hu):
    result = np.ones(hu.shape, np.int32)
    result[result.shape[0] // 2:] = 2
    return result


def _evidence(*, hu, roi, affine, partitions):
    return {"pairs": [{"status": "measured", "native_bottleneck_hu": float(hu.min())}]}, {"path": partitions == 1}


def _frozen(tmp_path, monkeypatch, count=0):
    known = tmp_path / "known"
    known.mkdir()
    protected = known / "saved.txt"
    protected.write_text("retain historical evidence", encoding="utf-8")
    snapshot = {"known_run": str(known), "case_count": count,
                "source": "frozen", "protected_files": {"saved.txt": experiment._sha(protected)}}
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot)
    plan = tmp_path / "plan.json"
    experiment.freeze(plan, known)
    return known, plan, snapshot


@pytest.mark.parametrize("cached", [False, True])
def test_truth_never_reaches_detector_or_evidence(arrays, monkeypatch, cached):
    image_partitions = _partitions(arrays["hu"])
    calls = []

    def detector(*, hu, roi, affine):
        calls.append("detector")
        for key, value in (("hu", hu), ("roi", roi), ("affine", affine)):
            assert not value.flags.writeable and not np.shares_memory(value, arrays[key])
            np.testing.assert_array_equal(value, arrays[key])
        return {"partitions": image_partitions, "diagnostics": {"status": "withheld"}}

    def evidence(*, hu, roi, affine, partitions):
        calls.append("evidence")
        assert all(not value.flags.writeable for value in (hu, roi, affine, partitions))
        assert not np.shares_memory(partitions, image_partitions)
        np.testing.assert_array_equal(partitions, image_partitions)
        return _evidence(hu=hu, roi=roi, affine=affine, partitions=partitions)

    monkeypatch.setattr(experiment, "count_with_ridges", detector)
    monkeypatch.setattr(experiment, "passage_evidence", evidence)
    first = experiment.measure(arrays, image_partitions if cached else None)
    changed_truth = arrays | {key: object() for key in ("labels", "highres_labels", "cores")}
    second = experiment.measure(changed_truth, image_partitions if cached else None)
    assert first[0] == second[0] and first[2] == second[2]
    for key in first[1]:
        np.testing.assert_array_equal(first[1][key], second[1][key])
    assert calls == (["evidence"] * 2 if cached else ["detector", "evidence"] * 2)


@pytest.mark.parametrize("stage,target", [("detector", "hu"), ("evidence", "hu"),
                                          ("evidence", "roi"), ("evidence", "affine"),
                                          ("evidence", "partitions")])
def test_hashes_detect_mutation_even_when_writer_enables_write_flag(arrays, monkeypatch, stage, target):
    originals = {key: value.copy() for key, value in arrays.items()}
    image_partitions = _partitions(arrays["hu"])
    original_partitions = image_partitions.copy()

    def corrupt(value):
        value.setflags(write=True)
        value.flat[0] = not value.flat[0] if value.dtype == bool else value.flat[0] + 1

    def detector(**inputs):
        if stage == "detector":
            corrupt(inputs[target])
        return {"partitions": image_partitions, "diagnostics": {}}

    def evidence(**inputs):
        if stage == "evidence":
            corrupt(inputs[target])
        return _evidence(**inputs)

    monkeypatch.setattr(experiment, "count_with_ridges", detector)
    monkeypatch.setattr(experiment, "passage_evidence", evidence)
    with pytest.raises(experiment.EvidenceFailure, match="mutated"):
        experiment.measure(arrays)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], originals[key])
    np.testing.assert_array_equal(image_partitions, original_partitions)


def test_failed_case_is_retained_and_later_cases_complete(arrays, tmp_path, monkeypatch):
    known, plan, snapshot = _frozen(tmp_path, monkeypatch, count=3)
    items = [(arrays, {"name": name, "group": group, "physical_count": count}, None)
             for name, group, count in (("first", "known_development", 2),
                                        ("broken", "known_development", 1),
                                        ("last", "new_perturbations", 1))]
    monkeypatch.setattr(experiment, "_items", lambda path: items)
    measured = []

    def measure(values, partitions):
        measured.append(len(measured))
        if len(measured) == 2:
            raise RuntimeError("injected middle failure")
        report, witnesses = _evidence(**{key: values[key] for key in ("hu", "roi", "affine")},
                                      partitions=_partitions(values["hu"]))
        return report, witnesses, {"status": "withheld", "counts": {"total": None}}

    monkeypatch.setattr(experiment, "measure", measure)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, destination)
    result = json.loads((destination / "results.json").read_text())
    assert len(measured) == result["case_count"] == len(result["results"]) == 3
    assert result["execution_errors"] == 1 and result["historical_files_preserved"]
    assert result["counts_changed"] is False and result["clinical_accuracy"] == "NOT_MEASURED"
    assert [row["metadata"]["name"] for row in result["results"]] == ["first", "broken", "last"]
    assert result["results"][1]["failure"] == "RuntimeError: injected middle failure"
    assert result["results"][1]["failure_stage"] == "measurement"
    assert result["results"][1]["evidence"] is None
    assert result["results"][2]["failure"] is None
    for row in result["results"]:
        folder = destination / row["metadata"]["name"]
        assert json.loads((folder / "result.json").read_text()) == row
        with np.load(folder / "input.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key in arrays:
                np.testing.assert_array_equal(saved[key], arrays[key])
        with np.load(folder / "evidence.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(row["output_hashes"])
    assert all(row["eligible_cases"] == 1 for row in result["threshold_diagnostics"])
    assert (destination / "report.md").is_file()
    assert json.loads((destination / "protocol.json").read_text()) == snapshot


@pytest.mark.parametrize("corrupt_partitions", [False, True])
def test_extractor_failure_retains_original_detector_artifacts_and_completes_census(
        arrays, tmp_path, monkeypatch, corrupt_partitions):
    known, plan, _ = _frozen(tmp_path, monkeypatch, count=3)
    items = [(arrays, {"name": name, "group": "new_perturbations", "physical_count": 2}, None)
             for name in ("first", "broken", "last")]
    monkeypatch.setattr(experiment, "_items", lambda path: items)
    detector_partitions = _partitions(arrays["hu"])
    diagnostics = {"status": "withheld", "counts": {"total": None}, "withheld_reasons": ["material_ambiguity"]}
    detected, extracted = [], []

    def detector(*, hu, roi, affine):
        detected.append(1)
        return {"partitions": detector_partitions.copy(), "diagnostics": diagnostics.copy()}

    def evidence(**inputs):
        extracted.append(1)
        if len(extracted) == 2:
            if corrupt_partitions:
                inputs["partitions"].setflags(write=True)
                inputs["partitions"].fill(0)
            raise RuntimeError("injected extractor failure")
        return _evidence(**inputs)

    monkeypatch.setattr(experiment, "count_with_ridges", detector)
    monkeypatch.setattr(experiment, "passage_evidence", evidence)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, destination)
    result = json.loads((destination / "results.json").read_text())
    assert len(detected) == len(extracted) == result["case_count"] == 3
    assert result["execution_errors"] == 1
    failed = result["results"][1]
    assert failed["failure_stage"] == "evidence"
    assert failed["failure"] == "RuntimeError: injected extractor failure"
    assert failed["detector_diagnostics"] == diagnostics
    assert failed["evidence"] is None
    assert result["results"][2]["failure"] is None
    with np.load(destination / "broken/evidence.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"partitions"}
        np.testing.assert_array_equal(saved["partitions"], detector_partitions)
        assert failed["output_hashes"]["partitions"] == experiment.array_hash(saved["partitions"])


def test_frozen_drift_refuses_measurement_and_output(tmp_path, monkeypatch):
    known, plan, snapshot = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot | {"source": "changed"})
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("Cases loaded after drift"))
    monkeypatch.setattr(experiment, "measure", lambda *args: pytest.fail("Measured after drift"))
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="Frozen"):
        experiment.execute(plan, destination)
    assert not destination.exists()


def test_plan_and_existing_output_are_never_overwritten(tmp_path, monkeypatch):
    known, plan, snapshot = _frozen(tmp_path, monkeypatch)
    original_plan = plan.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.freeze(plan, known)
    assert plan.read_bytes() == original_plan
    destination = tmp_path / "run"
    destination.mkdir()
    sentinel = destination / "results.json"
    sentinel.write_bytes(b"existing experiment")
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("Cases loaded for existing output"))
    with pytest.raises(FileExistsError):
        experiment.execute(plan, destination)
    assert sentinel.read_bytes() == b"existing experiment"
    assert set(destination.iterdir()) == {sentinel}


@pytest.mark.parametrize("operation", ["freeze", "execute"])
def test_historical_tree_cannot_be_used_for_new_outputs(tmp_path, monkeypatch, operation):
    known, plan, snapshot = _frozen(tmp_path, monkeypatch)
    original = {p.relative_to(known): p.read_bytes() for p in known.rglob("*") if p.is_file()}
    monkeypatch.setattr(experiment, "_items", lambda path: pytest.fail("Cases reached before history rejection"))
    target = known / "nested" / ("plan.json" if operation == "freeze" else "run")
    with pytest.raises(ValueError):
        if operation == "freeze":
            experiment.freeze(target, known)
        else:
            experiment.execute(plan, target)
    assert {p.relative_to(known): p.read_bytes() for p in known.rglob("*") if p.is_file()} == original
    assert not target.parent.exists()


@pytest.mark.parametrize("defect", ["extra", "missing"])
def test_historical_archive_requires_exact_input_manifest(arrays, tmp_path, defect):
    rows = [{"name": f"case_{index}", "input_hashes": {key: experiment.array_hash(value) for key, value in arrays.items()}}
            for index in range(68)]
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    folder = tmp_path / "case_0"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps(rows[0]))
    saved = arrays.copy()
    if defect == "extra":
        saved["unexpected_truth"] = np.zeros(1)
    else:
        del saved["cores"]
    np.savez_compressed(folder / "input.npz", **saved)
    with pytest.raises(ValueError):
        experiment._items(tmp_path.resolve())
