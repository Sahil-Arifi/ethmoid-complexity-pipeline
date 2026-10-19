"""Freeze and compare supplied hypotheses using original and new synthetic samples."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.ndimage import zoom

import aperture_identifiability_experiment as prior
from ridge_counting_experiment import _external_path, _json, _sha
from sampling_information_model import classify, render_pair
from synthetic_counting_experiment import REPO, array_hash

METHODS = ("original", "interpolated", "original_plus_shifted")
ERROR_BOUND = 5.
SHIFT_X_MM = .125


def scenarios():
    items = []
    for split, apertures, depths in (
        ("development", (.125, .25), (.25,)),
        ("evaluation", (.0625, .09375, .15625, .21875), (.125, .375)),
    ):
        for aperture in apertures:
            for depth in depths:
                for sigma in (0., .15):
                    index = len(items)
                    items.append({"pair_id": f"pair_{index:02d}", "pair_index": index,
                        "split": split, "aperture_mm": aperture, "recess_depth_mm": depth,
                        "sigma_mm": sigma, "noise_seeds": {
                            "original": 20261019+index*10, "shifted": 20261020+index*10}})
    return items


def construct(info):
    arrays = {}
    for acquisition, shift in (("original", 0.), ("shifted", SHIFT_X_MM)):
        predictions = render_pair(info["aperture_mm"], info["recess_depth_mm"], info["sigma_mm"], shift)
        noise = np.random.default_rng(info["noise_seeds"][acquisition]).uniform(
            -ERROR_BOUND, ERROR_BOUND, predictions[0].shape)
        arrays[f"{acquisition}_noise"] = noise
        for truth, prediction in enumerate(predictions):
            arrays[f"{acquisition}_hypothesis_{truth}"] = prediction
            arrays[f"{acquisition}_observation_{truth}"] = prediction+noise
    for kind in ("hypothesis", "observation"):
        for truth in (0, 1):
            key = f"{kind}_{truth}"
            arrays[f"interpolated_{key}"] = zoom(arrays[f"original_{key}"], 2, order=1)
            arrays[f"original_plus_shifted_{key}"] = np.concatenate(
                [arrays[f"original_{key}"], arrays[f"shifted_{key}"]], axis=0)
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def measure(observation, hypotheses):
    """Inference receives isolated image arrays, without truth or geometry metadata."""
    values = [np.array(value, copy=True) for value in (observation, *hypotheses)]
    before = [array_hash(value) for value in values]
    for value in values:
        value.setflags(write=False)
    try:
        result = classify(values[0], values[1:], ERROR_BOUND)
        residuals = [float(np.max(np.abs(values[0]-value))) for value in values[1:]]
        compatible = [i for i, residual in enumerate(residuals) if residual <= ERROR_BOUND]
        expected = {"residual_linf": residuals, "compatible_indices": compatible,
            "conditional_count": compatible[0]+1 if len(compatible) == 1 else None,
            "status": "unique" if len(compatible) == 1 else ("ambiguous" if compatible else "no_fit")}
        if result != expected or not all(np.isfinite(residuals)):
            raise ValueError("Invalid conditional inference result.")
        return result
    finally:
        if before != [array_hash(value) for value in values]:
            raise ValueError("Inference mutated its isolated inputs.")


def _record(observation, hypotheses, **metadata):
    try:
        return metadata | {"decision": measure(observation, hypotheses), "failure": None}
    except Exception as error:
        return metadata | {"decision": None, "failure": type(error).__name__+": "+str(error)}


def _reindex(value):
    return value.transpose(2, 1, 0)[:, ::-1, :].copy()


def evaluate_pair(info, arrays, construction_failure=None):
    rows, controls, separations = [], [], {}
    for method in METHODS:
        hypotheses = [arrays.get(f"{method}_hypothesis_{i}") for i in (0, 1)]
        for truth in (0, 1):
            metadata = {"pair_id": info["pair_id"], "split": info["split"], "method": method,
                        "truth_index": truth, "physical_count": truth+1}
            if construction_failure:
                row = metadata | {"decision": None, "failure": construction_failure}
            else:
                row = _record(arrays[f"{method}_observation_{truth}"], hypotheses, **metadata)
            rows.append(row)
    if construction_failure:
        return {"results": rows, "controls": controls, "separations_hu": separations,
                "construction_failure": construction_failure}
    for method in (*METHODS, "shifted"):
        hypotheses = [arrays[f"{method}_hypothesis_{i}"] for i in (0, 1)]
        separations[method] = float(np.max(np.abs(hypotheses[0]-hypotheses[1])))
    for row in rows:
        method, truth = row["method"], row["truth_index"]
        for kind, indices in (("hypothesis", (0, 1)), ("observation", (truth,))):
            for i in indices:
                key = f"{method}_{kind}_{i}"
                arrays[f"reindexed_{key}"] = _reindex(arrays[key])
        controls.append(_record(arrays[f"reindexed_{method}_observation_{truth}"],
            [arrays[f"reindexed_{method}_hypothesis_{i}"] for i in (0, 1)],
            kind="reindex", method=method, truth_index=truth, reference_decision=row["decision"]))
    for acquisition in ("original", "shifted"):
        hypotheses = [arrays[f"{acquisition}_hypothesis_{i}"] for i in (0, 1)]
        controls.append(_record(hypotheses[0], [hypotheses[0], hypotheses[0]],
                                kind="identical", acquisition=acquisition))
        key = f"control_{acquisition}_no_fit"
        arrays[key] = np.maximum(*hypotheses)+ERROR_BOUND+1.
        controls.append(_record(arrays[key], hypotheses, kind="no_fit", acquisition=acquisition))
        if separations[acquisition] <= 2*ERROR_BOUND:
            key = f"control_{acquisition}_midpoint"
            arrays[key] = (hypotheses[0]+hypotheses[1])/2
            controls.append(_record(arrays[key], hypotheses, kind="midpoint", acquisition=acquisition))
    for value in arrays.values():
        value.setflags(write=False)
    return {"results": rows, "controls": controls, "separations_hu": separations,
            "construction_failure": None}


def _prior_protocol(known_run):
    return json.loads((known_run / "protocol.json").read_text(encoding="utf-8"))


def _protected_roots(known_run):
    known_run = _external_path(known_run, fresh=False)
    canonical = _external_path(_prior_protocol(known_run)["known_run"], fresh=False)
    roots = (known_run.parent, *prior._protected_roots(canonical))
    if len(set(roots)) != 4 or not all(root.is_dir() for root in roots):
        raise ValueError("Four distinct historical directories are required.")
    return roots


def _snapshot(known_run):
    known_run = _external_path(known_run, fresh=False)
    protocol = _prior_protocol(known_run)
    manifest = json.loads((known_run / "sha256.json").read_text(encoding="utf-8"))
    actual = {path.relative_to(known_run).as_posix(): _sha(path)
              for path in sorted(known_run.rglob("*")) if path.is_file() and path != known_run / "sha256.json"}
    if actual != manifest or protocol != json.loads((known_run.parent / "plan.json").read_text(encoding="utf-8")):
        raise ValueError("Historical aperture manifest or protocol changed.")
    if protocol["experiment"] != "aperture-identifiability-v1":
        raise ValueError("The known run must be the prior aperture experiment.")
    roots = _protected_roots(known_run)
    history = prior._history(Path(protocol["known_run"]).resolve())
    if (history != protocol["historical_baseline"]
            or prior._protected_files(roots[1:]) != protocol["protected_files"]):
        raise ValueError("Historical baseline or protected files changed.")
    sources = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py"))
    sources += [REPO / "scripts" / name for name in ("sampling_information_model.py",
        "sampling_information_experiment.py", "aperture_observation_model.py",
        "aperture_identifiability_experiment.py", "ridge_counting_experiment.py",
        "synthetic_counting_experiment.py", "synthetic_blur_experiment.py")]
    sources += [REPO / path for path in ("docs/SAMPLING_INFORMATION_EXPERIMENT.md",
        "tests/test_sampling_information_experiment.py", "tests/test_sampling_information_model.py",
        "counting-tests.lock", "pyproject.toml")]
    items = scenarios()
    return {"experiment": "sampling-information-v1", "clinical_accuracy": "NOT_MEASURED",
        "known_run": str(known_run), "historical_baseline": history, "scenarios": items,
        "census_sha256": hashlib.sha256(_json(items).encode("utf-8")).hexdigest(),
        "methods": list(METHODS), "physical_counts": {"through": 1, "blind": 2},
        "main_observations_per_method": {"development": 8, "evaluation": 32},
        "error_bound_hu": ERROR_BOUND, "shift_x_mm": SHIFT_X_MM,
        "interpolation": {"zoom": 2, "order": 1}, "rng": "numpy.random.default_rng",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": sys.version, "runtime": {name: version(name) for name in
            ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
        "sources": {path.relative_to(REPO).as_posix(): _sha(path) for path in sources},
        "protected_files": prior._protected_files(roots)}


def freeze(path, known_run):
    known_run = _external_path(known_run, fresh=False)
    path = prior._destination(path, _protected_roots(known_run))
    snapshot = _snapshot(known_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(snapshot)+"\n")
    return path


def summarize(pairs):
    rows = [row for pair in pairs for row in pair["results"]]
    controls = [row for pair in pairs for row in pair["controls"]]
    successful = [row for row in rows if row["failure"] is None]
    table = []
    for split in ("development", "evaluation"):
        for method in METHODS:
            selected = [row for row in rows if row["split"] == split and row["method"] == method]
            counts = dict(correct=0, wrong=0, withheld=0, errors=0)
            for row in selected:
                if row["failure"]:
                    counts["errors"] += 1
                elif row["decision"]["conditional_count"] is None:
                    counts["withheld"] += 1
                else:
                    counts["correct" if row["decision"]["conditional_count"] == row["physical_count"] else "wrong"] += 1
            table.append({"split": split, "method": method, "total": len(selected), **counts,
                "coverage": (counts["correct"]+counts["wrong"])/len(selected) if selected else 0.})
    expected = {(i["pair_id"], i["split"], method, truth)
                for i in scenarios() for method in METHODS for truth in (0, 1)}
    observed = [(r["pair_id"], r["split"], r["method"], r["truth_index"]) for r in rows]
    def control_gate(kind, predicate):
        chosen = [row for row in controls if row["kind"] == kind]
        return bool(chosen) and all(row["failure"] is None and predicate(row) for row in chosen)
    complete_controls = True
    for pair in pairs:
        required = {("reindex", method, truth) for method in METHODS for truth in (0, 1)}
        required |= {(kind, acquisition, None) for kind in ("identical", "no_fit")
                     for acquisition in ("original", "shifted")}
        required |= {("midpoint", acquisition, None) for acquisition in ("original", "shifted")
                     if pair["separations_hu"].get(acquisition, float("inf")) <= 2*ERROR_BOUND}
        actual = [(r["kind"], r.get("method", r.get("acquisition")), r.get("truth_index"))
                  for r in pair["controls"]]
        complete_controls &= len(actual) == len(required) and set(actual) == required
    gates = {"complete_census": len(observed) == len(expected) and set(observed) == expected,
        "complete_controls": complete_controls,
        "no_execution_errors": all(row["failure"] is None for row in (*rows, *controls)),
        "true_hypothesis_compatible": len(successful) == len(rows) and
            all(row["truth_index"] in row["decision"]["compatible_indices"] for row in successful),
        "reindex_consistency": control_gate("reindex", lambda r: r["decision"] == r["reference_decision"]),
        "identical_input_withheld": control_gate("identical", lambda r: r["decision"]["status"] == "ambiguous"),
        "no_fit_withheld": control_gate("no_fit", lambda r: r["decision"]["status"] == "no_fit"),
        "midpoint_ambiguity": control_gate("midpoint", lambda r: r["decision"]["status"] == "ambiguous"),
        "interpolation_separation_not_increased": all(pair["construction_failure"] is None and
            pair["separations_hu"]["interpolated"] <= pair["separations_hu"]["original"]+1e-10 for pair in pairs)}
    paired_changes = []
    for row in rows:
        if row["method"] != "original":
            continue
        matched = [r for r in rows if r["pair_id"] == row["pair_id"] and r["truth_index"] == row["truth_index"]]
        paired_changes.append({"pair_id": row["pair_id"], "split": row["split"], "truth_index": row["truth_index"],
            "decisions": {r["method"]: r["decision"] for r in matched},
            "failures": {r["method"]: r["failure"] for r in matched}})
    combined = next(row for row in table if row["split"] == "evaluation" and row["method"] == "original_plus_shifted")
    numeric_target_met = combined["wrong"] == 0 and combined["correct"] >= 26
    return {"record_count": len(rows), "control_count": len(controls), "table": table, "gates": gates,
        "execution_errors": sum(row["failure"] is not None for row in (*rows, *controls)),
        "utility_target": {"minimum_accepted": 26, "evaluation_denominator": 32,
            "zero_wrong": combined["wrong"] == 0, "coverage_at_least_80_percent":
                combined["correct"]+combined["wrong"] >= 26,
            "criteria_met": numeric_target_met, "software_gates_passed": all(gates.values()),
            "supported": numeric_target_met and all(gates.values()),
            "clinical_meaning": "NONE; investigator-selected exploratory target"},
        "results": rows, "paired_changes": paired_changes, "clinical_accuracy": "NOT_MEASURED",
        "counter_rerun": False, "historical_count_recovery": "NOT_MEASURED"}


def execute(plan_path, destination):
    plan_path = _external_path(plan_path, fresh=False)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    known_run = Path(plan["known_run"]).resolve()
    roots = _protected_roots(known_run)
    destination = prior._destination(destination, roots)
    if plan != _snapshot(known_run):
        raise ValueError("Frozen source, runtime, census or history changed.")
    destination.mkdir(parents=True)
    (destination / "protocol.json").write_text(_json(plan)+"\n", encoding="utf-8")
    pairs = []
    for info in scenarios():
        folder = destination / info["pair_id"]
        folder.mkdir()
        arrays, failure = {}, None
        try:
            arrays = construct(info)
            before = {key: array_hash(value) for key, value in arrays.items()}
            pair = evaluate_pair(info, arrays)
            if before != {key: array_hash(arrays[key]) for key in before}:
                raise ValueError("Pair evaluation mutated the saved input arrays.")
        except Exception as error:
            failure = type(error).__name__+": "+str(error)
            pair = {"results": [{"pair_id": info["pair_id"], "split": info["split"],
                "method": method, "truth_index": truth, "physical_count": truth+1,
                "decision": None, "failure": failure} for method in METHODS for truth in (0, 1)],
                "controls": [], "separations_hu": {}, "construction_failure": failure}
        pair.update(metadata=info, array_hashes={key: array_hash(value) for key, value in arrays.items()})
        np.savez_compressed(folder / "arrays.npz", **arrays)
        (folder / "result.json").write_text(_json(pair)+"\n", encoding="utf-8")
        pairs.append(pair)
    summary = summarize(pairs)
    preserved = plan["protected_files"] == prior._protected_files(roots)
    validation_failure = None
    try:
        unchanged = plan == _snapshot(known_run)
    except Exception as error:
        unchanged = False
        validation_failure = type(error).__name__+": "+str(error)
    summary.update(historical_files_preserved=preserved, historical_baseline=plan.get("historical_baseline"),
                   post_run_validation_failure=validation_failure, pairs=pairs)
    summary["gates"].update(protected_files_preserved=preserved, frozen_state_unchanged=unchanged)
    summary["utility_target"].update(software_gates_passed=all(summary["gates"].values()),
        supported=summary["utility_target"]["criteria_met"] and all(summary["gates"].values()))
    (destination / "results.json").write_text(_json(summary)+"\n", encoding="utf-8")
    lines = ["# Conditional sampling-information experiment", "",
        "Two shape hypotheses and their imaging parameters are supplied. Clinical accuracy is unmeasured.",
        "Interpolation enlarges existing samples; the shifted volume is independently forward-rendered.", "",
        "| Split | Method | Total | Correct | Wrong | Withheld | Errors | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["table"]:
        lines.append("| "+" | ".join(str(row[key]) for key in
            ("split", "method", "total", "correct", "wrong", "withheld", "errors", "coverage"))+" |")
    lines += ["", f"Software gates: {summary['gates']}.", f"Exploratory utility target: {summary['utility_target']}.",
        "Controls are outside the main denominator. Repeat executions are not additional observations.",
        "The historical counter was not run or changed; its withheld counts were not recovered."]
    (destination / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    manifest = {path.relative_to(destination).as_posix(): _sha(path)
                for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "sha256.json").write_text(_json(manifest)+"\n", encoding="utf-8")
    if summary["execution_errors"] or not preserved or not unchanged:
        raise RuntimeError("Experiment retained failures; inspect the saved census.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "run"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--plan", type=Path, required=True)
        subparser.add_argument("--known-run" if command == "freeze" else "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.plan, args.known_run))
    else:
        result = execute(args.plan, args.output)
        print(_json({key: result[key] for key in ("record_count", "table", "gates", "utility_target")}))
        sys.exit(0 if all(result["gates"].values()) else 2)
