"""Tiny fixtures test evidence handling without rendering the evaluation census."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sampling_information_experiment as experiment
from ridge_counting_experiment import _sha
from synthetic_counting_experiment import array_hash


def tiny_pair(index=0, split="development"):
    return {"pair_id": f"tiny_{index}", "pair_index": index, "split": split,
            "aperture_mm": .125, "recess_depth_mm": .25, "sigma_mm": 0.,
            "noise_seeds": {"original": 20261019+index*10, "shifted": 20261020+index*10}}


@pytest.fixture
def tiny_renderer(monkeypatch):
    def render(aperture_mm, recess_depth_mm, sigma_mm, shift_x_mm=0.):
        first = np.arange(8, dtype=float).reshape(2, 2, 2)
        second = first.copy()
        if shift_x_mm:
            second[0, 0, 0] += 40.
        return first, second
    monkeypatch.setattr(experiment, "render_pair", render, raising=False)
    return render


def frozen(tmp_path, monkeypatch, count=3):
    roots = tuple(tmp_path / name for name in ("aperture", "canonical", "native", "ridge"))
    for root in roots:
        root.mkdir()
        (root / "historical.txt").write_text("protected evidence", encoding="utf-8")
    known = roots[0] / "run-01"
    known.mkdir()
    items = [tiny_pair(i, "development" if i == 0 else "evaluation") for i in range(count)]
    protected = {str(root): {p.relative_to(root).as_posix(): _sha(p)
                            for p in root.rglob("*") if p.is_file()} for root in roots}
    snapshot = {"known_run": str(known), "scenarios": items, "sources": {"runner": "frozen"},
                "runtime": {"numpy": "fixed"}, "protected_files": protected}
    monkeypatch.setattr(experiment, "_protected_roots", lambda path: roots, raising=False)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot, raising=False)
    monkeypatch.setattr(experiment, "scenarios", lambda: items)
    plan = tmp_path / "plan.json"
    experiment.freeze(plan, known)
    return known, roots, plan, snapshot


def test_fixed_census_is_twenty_unique_pairs_with_both_truths_and_frozen_seeds():
    items = experiment.scenarios()
    assert len(items) == len({item["pair_id"] for item in items}) == 20
    development = [item for item in items if item["split"] == "development"]
    evaluation = [item for item in items if item["split"] == "evaluation"]
    assert len(development) == 4 and len(evaluation) == 16
    assert {(i["aperture_mm"], i["recess_depth_mm"], i["sigma_mm"]) for i in development} == {
        (a, .25, s) for a in (.125, .25) for s in (0., .15)}
    assert {(i["aperture_mm"], i["recess_depth_mm"], i["sigma_mm"]) for i in evaluation} == {
        (a, d, s) for a in (.0625, .09375, .15625, .21875)
        for d in (.125, .375) for s in (0., .15)}
    assert [i["pair_index"] for i in items] == list(range(20))
    assert all(i["noise_seeds"] == {"original": 20261019+10*j, "shifted": 20261020+10*j}
               for j, i in enumerate(items))


def test_construction_reuses_noise_across_truths_but_samples_second_grid(tiny_renderer):
    info = tiny_pair()
    arrays = experiment.construct(info)
    for acquisition in ("original", "shifted"):
        expected_noise = np.random.default_rng(info["noise_seeds"][acquisition]).uniform(-5, 5, (2, 2, 2))
        np.testing.assert_array_equal(arrays[f"{acquisition}_noise"], expected_noise)
        for truth in (0, 1):
            np.testing.assert_array_equal(arrays[f"{acquisition}_observation_{truth}"],
                arrays[f"{acquisition}_hypothesis_{truth}"]+expected_noise)
    np.testing.assert_array_equal(arrays["original_hypothesis_0"], arrays["original_hypothesis_1"])
    assert np.max(np.abs(arrays["shifted_hypothesis_0"]-arrays["shifted_hypothesis_1"])) == 40
    assert arrays["interpolated_observation_0"].shape == (4, 4, 4)
    np.testing.assert_array_equal(arrays["original_plus_shifted_observation_1"],
        np.concatenate([arrays["original_observation_1"], arrays["shifted_observation_1"]], axis=0))
    assert all(not value.flags.writeable for value in arrays.values())


def test_inference_only_receives_isolated_readonly_arrays(monkeypatch):
    source = np.arange(8, dtype=float).reshape(2, 2, 2)
    import sampling_information_model
    def checked(observed, hypotheses, error_bound=5.):
        assert error_bound == 5.
        for value in [observed, *hypotheses]:
            assert not value.flags.writeable and not np.shares_memory(value, source)
        return sampling_information_model.classify(observed, hypotheses, error_bound)
    monkeypatch.setattr(experiment, "classify", checked, raising=False)
    result = experiment.measure(source, [source, source+20])
    assert result["conditional_count"] == 1 and result["compatible_indices"] == [0]


@pytest.mark.parametrize("target", ["observation", "hypothesis"])
def test_mutation_of_isolated_inputs_is_detected_and_originals_survive(monkeypatch, target):
    source = np.zeros((2, 2, 2))
    def changed(observed, hypotheses, error_bound=5.):
        value = observed if target == "observation" else hypotheses[0]
        value.setflags(write=True)
        value.flat[0] += 1
        raise RuntimeError("injected mutation")
    monkeypatch.setattr(experiment, "classify", changed, raising=False)
    with pytest.raises(ValueError, match="mutat"):
        experiment.measure(source, [source, source+20])
    assert not np.any(source)


@pytest.mark.parametrize("field", ["sources", "runtime", "protected_files", "scenarios"])
def test_frozen_drift_refuses_to_construct_or_create_output(tmp_path, monkeypatch, field):
    _, _, plan, snapshot = frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "_snapshot", lambda path: snapshot | {field: "changed"})
    monkeypatch.setattr(experiment, "construct", lambda info: pytest.fail("constructed after drift"))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="[Ff]rozen"):
        experiment.execute(plan, output)
    assert not output.exists()


@pytest.mark.parametrize("tree", range(4))
def test_entire_aperture_folder_and_prior_trees_are_protected(tmp_path, monkeypatch, tree):
    known, roots, plan, _ = frozen(tmp_path, monkeypatch)
    target = roots[tree] / "new"
    with pytest.raises(ValueError, match="protected"):
        experiment.freeze(target / "plan.json", known)
    with pytest.raises(ValueError, match="protected"):
        experiment.execute(plan, target)
    assert not target.exists()


def test_existing_plan_and_output_are_never_overwritten(tmp_path, monkeypatch):
    known, _, plan, _ = frozen(tmp_path, monkeypatch)
    old = plan.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.freeze(plan, known)
    assert plan.read_bytes() == old
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("evidence")
    with pytest.raises(FileExistsError):
        experiment.execute(plan, output)
    assert sentinel.read_text() == "evidence" and list(output.iterdir()) == [sentinel]


def test_middle_failure_preserves_complete_census_arrays_and_manifest(tmp_path, monkeypatch, tiny_renderer):
    _, _, plan, snapshot = frozen(tmp_path, monkeypatch)
    original = experiment.measure
    calls = []
    def failing_once(observation, hypotheses):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected failure")
        return original(observation, hypotheses)
    monkeypatch.setattr(experiment, "measure", failing_once)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, output)
    result = json.loads((output / "results.json").read_text())
    assert len(result["results"]) == 3*2*3
    assert result["execution_errors"] == 1
    assert sum(row["failure"] is not None for row in result["results"]) == 1
    assert result["historical_files_preserved"]
    for item in snapshot["scenarios"]:
        folder = output / item["pair_id"]
        row = json.loads((folder / "result.json").read_text())
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            assert {name: array_hash(archive[name]) for name in archive.files} == row["array_hashes"]
    manifest = json.loads((output / "sha256.json").read_text())
    assert manifest == {p.relative_to(output).as_posix(): _sha(p) for p in output.rglob("*")
                        if p.is_file() and p != output / "sha256.json"}
    errors = sum(row["errors"] for row in result["table"])
    assert errors == 1


def test_two_runs_are_byte_identical_and_controls_excluded_from_denominators(tmp_path, monkeypatch, tiny_renderer):
    _, _, plan, _ = frozen(tmp_path, monkeypatch)
    first = experiment.execute(plan, tmp_path / "first")
    second = experiment.execute(plan, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/sha256.json").read_bytes() == (tmp_path / "second/sha256.json").read_bytes()
    assert len(first["results"]) == 18
    assert sum(row["total"] for row in first["table"]) == 18
    assert first["control_count"] > 18
    assert first["execution_errors"] == 0
    assert first["gates"]["true_hypothesis_compatible"]
    assert first["gates"]["reindex_consistency"]
    assert first["gates"]["midpoint_ambiguity"]
    assert first["gates"]["identical_input_withheld"]
    assert first["gates"]["no_fit_withheld"]


def test_pair_preparation_failure_keeps_both_truths_for_every_method(tmp_path, monkeypatch, tiny_renderer):
    _, _, plan, _ = frozen(tmp_path, monkeypatch)
    original = experiment.evaluate_pair
    def broken(info, arrays, construction_failure=None):
        if info["pair_index"] == 1:
            raise ValueError("injected control preparation failure")
        return original(info, arrays, construction_failure)
    monkeypatch.setattr(experiment, "evaluate_pair", broken)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, output)
    saved = json.loads((output / "results.json").read_text())
    assert len(saved["results"]) == 18
    failed = [row for row in saved["results"] if row["pair_id"] == "tiny_1"]
    assert len(failed) == 6 and all("preparation failure" in row["failure"] for row in failed)
    assert sum(row["errors"] for row in saved["table"]) == 6
    assert not saved["gates"]["complete_controls"]
    assert (output / "sha256.json").exists()


def test_missing_or_duplicate_controls_cannot_pass_software_gates(tiny_renderer, monkeypatch):
    info = tiny_pair()
    monkeypatch.setattr(experiment, "scenarios", lambda: [info])
    pair = experiment.evaluate_pair(info, experiment.construct(info))
    assert experiment.summarize([pair])["gates"]["complete_controls"]
    pair["controls"].pop()
    assert not experiment.summarize([pair])["gates"]["complete_controls"]
    pair["controls"].append(pair["controls"][0])
    assert not experiment.summarize([pair])["gates"]["complete_controls"]


@pytest.mark.parametrize("invalid", [None, {}, {"conditional_count": 0},
    {"compatible_indices": [0], "conditional_count": 1, "status": "unique", "residual_linf": [float("nan"), 20]}])
def test_malformed_classifier_returns_are_rejected(monkeypatch, invalid):
    monkeypatch.setattr(experiment, "classify", lambda *args: invalid)
    with pytest.raises(ValueError, match="Invalid conditional"):
        experiment.measure(np.zeros((2, 2, 2)), [np.zeros((2, 2, 2)), np.ones((2, 2, 2))*20])


def test_changed_protected_tree_during_execution_is_saved_as_failed_gate(tmp_path, monkeypatch, tiny_renderer):
    _, roots, plan, _ = frozen(tmp_path, monkeypatch)
    original = experiment.construct
    def changed(info):
        (roots[0] / "historical.txt").write_text("injected history drift")
        return original(info)
    monkeypatch.setattr(experiment, "construct", changed)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="retained failures"):
        experiment.execute(plan, output)
    saved = json.loads((output / "results.json").read_text())
    assert len(saved["results"]) == 18
    assert not saved["historical_files_preserved"]
    assert not saved["gates"]["protected_files_preserved"]


def test_utility_target_does_not_admit_failed_software_evidence(tiny_renderer, monkeypatch):
    items = [tiny_pair(i, "evaluation") for i in range(16)]
    monkeypatch.setattr(experiment, "scenarios", lambda: items)
    pairs = [experiment.evaluate_pair(info, experiment.construct(info)) for info in items]
    pairs[0]["controls"][0]["failure"] = "injected control failure"
    summary = experiment.summarize(pairs)
    assert summary["utility_target"]["criteria_met"]
    assert not summary["utility_target"]["software_gates_passed"]
    assert not summary["utility_target"]["supported"]


def synthetic_prior(tmp_path, monkeypatch):
    roots = tuple(tmp_path / name for name in ("aperture", "canonical", "native", "ridge"))
    for root in roots:
        root.mkdir()
        (root / "evidence.txt").write_text("original")
    canonical = roots[1] / "run-01"
    canonical.mkdir()
    (canonical / "protocol.json").write_text(json.dumps({"protected_files": {
        str(roots[2]): {}, str(roots[3]): {}}}))
    history = {"counts": {"correct_accepted": 22, "wrong_accepted": 0, "withheld": 78, "errors": 0}}
    monkeypatch.setattr(experiment.prior, "_history", lambda path: history)
    known = roots[0] / "run-01"
    known.mkdir()
    protocol = {"experiment": "aperture-identifiability-v1", "known_run": str(canonical),
        "historical_baseline": history, "protected_files": experiment.prior._protected_files(roots[1:])}
    for path in (known / "protocol.json", known.parent / "plan.json"):
        path.write_text(json.dumps(protocol))
    (known / "sha256.json").write_text(json.dumps({"protocol.json": _sha(known / "protocol.json")}))
    sibling = roots[0] / "run-02"
    sibling.mkdir()
    (sibling / "proof.txt").write_text("protected sibling evidence")
    return known, roots


def test_real_snapshot_freezes_sources_seeds_runtime_and_whole_prior_folder(tmp_path, monkeypatch):
    known, roots = synthetic_prior(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "render_pair", lambda *args: pytest.fail("freeze rendered observations"))
    snapshot = experiment._snapshot(known)
    assert set(snapshot["protected_files"]) == {str(root) for root in roots}
    assert "run-02/proof.txt" in snapshot["protected_files"][str(roots[0])]
    assert set(snapshot["runtime"]) >= {"numpy", "scipy"}
    assert snapshot["sources"]["scripts/sampling_information_model.py"] == _sha(
        experiment.REPO / "scripts/sampling_information_model.py")
    assert "scripts/aperture_observation_model.py" in snapshot["sources"]
    assert len(snapshot["scenarios"]) == 20 and len(snapshot["census_sha256"]) == 64


@pytest.mark.parametrize("defect", ["manifest", "protocol", "canonical", "native", "ridge"])
def test_snapshot_rejects_prior_evidence_drift_before_freeze(tmp_path, monkeypatch, defect):
    known, roots = synthetic_prior(tmp_path, monkeypatch)
    if defect == "manifest":
        (known / "extra.txt").write_text("changed")
    elif defect == "protocol":
        (known.parent / "plan.json").write_text("{}")
    else:
        root = roots[{"canonical": 1, "native": 2, "ridge": 3}[defect]]
        (root / "evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="Historical"):
        experiment._snapshot(known)
