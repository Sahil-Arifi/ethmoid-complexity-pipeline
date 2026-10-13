"""Freeze and compare a synthetic repair against all saved native-passage cases."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from canonical_ridge_candidate import count_with_canonical_ridges
from ridge_counting_experiment import ARRAY_NAMES, NULL_COUNTS, _external_path, _json, _sha
from synthetic_counting_experiment import REPO, array_hash

MASK_TYPES = {"partitions": np.int32, "cavities": np.uint16, "air": np.uint16,
              "component_bridge": np.uint8}
GROUPS = ("known_development", "new_perturbations")


def _items(known_run):
    """Validate every saved row and array; never run the baseline or constructors."""
    known_run = Path(known_run).resolve()
    rows = json.loads((known_run / "results.json").read_text(encoding="utf-8"))["results"]
    names = [row["metadata"]["name"] for row in rows]
    if len(rows) != 100 or len(set(names)) != 100:
        raise ValueError("Exactly 100 unique saved native-passage records are required.")
    items = []
    for row, name in zip(rows, names):
        if (not isinstance(name, str) or not name or name in (".", "..")
                or Path(name).name != name or "/" in name or "\\" in name):
            raise ValueError("Invalid saved case name.")
        folder = (known_run / name).resolve()
        if not folder.is_relative_to(known_run):
            raise ValueError("Saved case leaves its input directory.")
        if json.loads((folder / "result.json").read_text(encoding="utf-8")) != row:
            raise ValueError("Saved aggregate and per-case results disagree.")
        with np.load(folder / "input.npz", allow_pickle=False) as saved:
            if len(saved.files) != len(ARRAY_NAMES) or set(saved.files) != set(ARRAY_NAMES):
                raise ValueError("Saved input array manifest changed.")
            arrays = {key: saved[key] for key in ARRAY_NAMES}
        if {key: array_hash(value) for key, value in arrays.items()} != row["input_hashes"]:
            raise ValueError("Saved input arrays changed.")
        with np.load(folder / "evidence.npz", allow_pickle=False) as saved:
            expected = row["output_hashes"]
            if (len(saved.files) != len(expected) or set(saved.files) != set(expected)
                    or "partitions" not in expected):
                raise ValueError("Saved evidence array manifest changed.")
            if {key: array_hash(saved[key]) for key in saved.files} != expected:
                raise ValueError("Saved evidence arrays changed.")
        info = dict(row["metadata"])
        if info["group"] not in GROUPS or type(info["physical_count"]) is not int or info["physical_count"] < 1:
            raise ValueError("Saved scoring metadata is invalid.")
        info.update(original_group=info["group"],
                    baseline_detector_diagnostics=row["detector_diagnostics"],
                    baseline_failure=row.get("failure"),
                    reference_old_partition_hash=row["output_hashes"]["partitions"])
        for value in arrays.values():
            value.setflags(write=False)
        items.append((arrays, info))
    return items


def _protected_roots(known_run):
    known_run = _external_path(known_run, fresh=False)
    prior = json.loads((known_run / "protocol.json").read_text(encoding="utf-8"))["known_run"]
    ridge_run = _external_path(prior, fresh=False)
    if not known_run.is_dir() or not ridge_run.is_dir():
        raise ValueError("Both protected historical directories must exist.")
    return (known_run.parent, ridge_run)


def _protected_files(roots):
    return {str(root): {p.relative_to(root).as_posix(): _sha(p)
                       for p in sorted(root.rglob("*")) if p.is_file()} for root in roots}


def _destination(path, roots, *, fresh=True):
    path = _external_path(path, fresh=fresh)
    if any(path.is_relative_to(root) for root in roots):
        raise ValueError("New artifacts must stay outside both protected historical trees.")
    return path


def _snapshot(known_run):
    known_run = Path(known_run).resolve()
    roots = _protected_roots(known_run)
    items = _items(known_run)
    sources = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py"))
    sources += [REPO / "scripts" / name for name in (
        "canonical_ridge_candidate.py", "canonical_counting_experiment.py", "native_passage_evidence.py",
        "ridge_counting_candidate.py", "ridge_counting_experiment.py", "synthetic_counting_experiment.py",
        "synthetic_blur_experiment.py")]
    sources += [REPO / "docs/CANONICAL_COUNTING_EXPERIMENT.md"]
    return {"experiment": "canonical-component-withholding-v1", "clinical_accuracy": "NOT_MEASURED",
            "known_run": str(known_run), "case_count": len(items), "candidate_budget": 1,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "python": sys.version,
            "runtime": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
            "sources": {path.relative_to(REPO).as_posix(): _sha(path) for path in sources},
            "protected_files": _protected_files(roots),
            "inputs": [{"metadata": info, "hashes": {key: array_hash(value) for key, value in arrays.items()}}
                       for arrays, info in items]}


def freeze(path, known_run):
    known_run = Path(known_run).resolve()
    path = _destination(path, _protected_roots(known_run))
    snapshot = _snapshot(known_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(snapshot) + "\n")
    return path


def measure(arrays):
    """Only copied observed HU, ROI and affine reach the candidate."""
    inputs = {key: arrays[key].copy() for key in ("hu", "roi", "affine")}
    before = {key: array_hash(value) for key, value in inputs.items()}
    for value in inputs.values():
        value.setflags(write=False)
    try:
        output = count_with_canonical_ridges(**inputs)
    finally:
        if before != {key: array_hash(value) for key, value in inputs.items()}:
            raise ValueError("Detector mutated its isolated inputs.")
    masks = {key: np.asarray(output[key]) for key in MASK_TYPES}
    for key, mask in masks.items():
        if mask.shape != inputs["hu"].shape or mask.dtype != MASK_TYPES[key] or np.any(mask < 0):
            raise ValueError("Invalid candidate array shape, dtype or labels.")
        eligible = inputs["roi"] if key == "component_bridge" else inputs["roi"] & (inputs["hu"] < 200)
        if np.any((mask > 0) & ~eligible):
            raise ValueError("Candidate arrays leave their eligible region.")
    if np.any(masks["component_bridge"] > 1):
        raise ValueError("Bridge witness must be binary.")
    if not np.array_equal(masks["air"], np.where(inputs["hu"] < -500, masks["cavities"], 0)):
        raise ValueError("Air mask differs from observed HU.")
    record = {key: output[key] for key in ("cells", "diagnostics", "warnings")}
    cells, details = record["cells"], record["diagnostics"]
    if (not isinstance(cells, list) or not isinstance(record["warnings"], list)
            or any(not isinstance(warning, str) for warning in record["warnings"])):
        raise ValueError("Invalid registry or warnings.")
    if sorted(cell["id"] for cell in cells) != np.unique(masks["cavities"][masks["cavities"] > 0]).tolist():
        raise ValueError("Registry IDs differ from cavity labels.")
    for cell in cells:
        mask = masks["cavities"] == cell["id"]
        center = inputs["affine"][:3, :3] @ np.argwhere(mask).mean(axis=0) + inputs["affine"][:3, 3]
        if (cell["voxel_count"] != int(mask.sum())
                or cell["air_voxel_count"] != int(np.count_nonzero(masks["air"] == cell["id"]))
                or not np.allclose(center, cell["centroid_ras_mm"], rtol=0, atol=1e-8)
                or cell["side"] != ("left" if center[0] <= 0 else "right")):
            raise ValueError("Registry measurements differ from masks.")
    status = details["status"]
    expected = {"left": sum(cell["side"] == "left" for cell in cells),
                "right": sum(cell["side"] == "right" for cell in cells), "total": len(cells)}
    if (status not in ("accepted", "withheld") or details["clinical_accuracy"] != "NOT_MEASURED"
            or details["counts"] != (expected if status == "accepted" else NULL_COUNTS)
            or details["experimental_count_available"] is not (status == "accepted")
            or bool(details["withheld_reasons"]) != (status == "withheld")):
        raise ValueError("Count availability differs from candidate status.")
    if status == "accepted" and any(type(value) is not int for value in details["counts"].values()):
        raise ValueError("Accepted counts must be integers.")
    _json(record)
    return record, masks


def _category(details, physical_count, failure):
    if failure or details is None:
        return "errors"
    if details["status"] == "withheld":
        return "withheld"
    return "correct_accepted" if details["counts"]["total"] == physical_count else "wrong_accepted"


def summarize(rows):
    comparison, old_wrong, new_wrong, single_losses, increases, twin_groups = [], [], [], [], [], {}
    classifications = []
    for row in rows:
        info, current = row["metadata"], row["candidate"]["diagnostics"]
        old = _category(info["baseline_detector_diagnostics"], info["physical_count"], info.get("baseline_failure"))
        new = _category(current, info["physical_count"], row["failure"])
        classifications.append((row, old, new))
        if old == "wrong_accepted": old_wrong.append((info["name"], new))
        if new == "wrong_accepted": new_wrong.append(info["name"])
        if old == "correct_accepted" and info["physical_count"] == 1 and (
                new != "correct_accepted" or current["counts"] != info["baseline_detector_diagnostics"]["counts"]):
            single_losses.append(info["name"])
        if new.endswith("accepted") and (not old.endswith("accepted")
                or current.get("component_bridge_evidence", {}).get("component_count", 0) > 1):
            increases.append(info["name"])
        if info.get("equivalence_group"):
            twin_groups.setdefault(info["equivalence_group"], []).append(row)
    for group in GROUPS:
        chosen = [(row, old, new) for row, old, new in classifications if row["metadata"]["group"] == group]
        keys = ("correct_accepted", "wrong_accepted", "withheld", "errors")
        comparison.append({"group": group, "case_count": len(chosen),
                           "baseline": {key: sum(old == key for _, old, _ in chosen) for key in keys},
                           "candidate": {key: sum(new == key for _, _, new in chosen) for key in keys},
                           "correct_accepted_lost": [row["metadata"]["name"] for row, old, new in chosen
                                                     if old == "correct_accepted" and new != "correct_accepted"]})
    twins = {}
    for name, group in twin_groups.items():
        twins[name] = (len(group) == 2 and {row["metadata"]["physical_count"] for row in group} == {1, 2}
                       and len({tuple(row["input_hashes"][key] for key in ("hu", "roi", "affine")) for row in group}) == 1
                       and len({_json({"candidate": row["candidate"], "hashes": row["output_hashes"]}) for row in group}) == 1
                       and all(row["failure"] is None and row["candidate"]["diagnostics"]["status"] == "withheld"
                               for row in group))
    gates = {"complete_unique_census": len(rows) == len({row["metadata"]["name"] for row in rows}) == 100
             and tuple(group["case_count"] for group in comparison) == (68, 32),
             "all_eight_prior_wrong_acceptances_withheld": len(old_wrong) == 8 and all(new == "withheld" for _, new in old_wrong),
             "zero_new_wrong_acceptances": not new_wrong,
             "all_baseline_correct_single_counts_preserved": not single_losses,
             "material_twins_consistent_and_withheld": len(twins) == 4 and all(twins.values()),
             "no_acceptance_increase_from_wall_path": not increases,
             "no_execution_errors": all(row["failure"] is None for row in rows)}
    return {"case_count": len(rows), "execution_errors": sum(row["failure"] is not None for row in rows),
            "clinical_accuracy": "NOT_MEASURED", "comparison": comparison, "gates": gates,
            "prior_wrong_acceptances": [name for name, _ in old_wrong], "wrong_acceptances": new_wrong,
            "single_count_regressions": single_losses, "acceptance_increases": increases,
            "material_twins": twins, "spatial_reindexing_audit": "NOT_PERFORMED"}


def _failed_prediction(shape):
    record = {"cells": [], "warnings": [], "diagnostics": {
        "status": "withheld", "counts": NULL_COUNTS.copy(), "withheld_reasons": ["execution_failure"],
        "experimental_count_available": False, "clinical_accuracy": "NOT_MEASURED"}}
    return record, {key: np.zeros(shape, dtype) for key, dtype in MASK_TYPES.items()}


def execute(plan_path, destination):
    plan_path = _external_path(plan_path, fresh=False)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    known_run = Path(plan["known_run"]).resolve()
    roots = _protected_roots(known_run)
    destination = _destination(destination, roots)
    if plan != _snapshot(known_run):
        raise ValueError("Frozen source, inputs, rules, runtime or protected files changed.")
    items = _items(known_run)
    destination.mkdir(parents=True)
    (destination / "protocol.json").write_text(_json(plan) + "\n", encoding="utf-8")
    rows = []
    for arrays, info in items:
        folder = destination / info["name"]
        folder.mkdir()
        np.savez_compressed(folder / "input.npz", **arrays)
        failure = None
        try:
            record, detected = measure(arrays)
        except Exception as error:
            failure = type(error).__name__ + ": " + str(error)
            record, detected = _failed_prediction(arrays["hu"].shape)
        row = {"metadata": info, "candidate": record, "failure": failure,
               "input_hashes": {key: array_hash(value) for key, value in arrays.items()},
               "output_hashes": {key: array_hash(value) for key, value in detected.items()}}
        np.savez_compressed(folder / "candidate.npz", **detected)
        (folder / "result.json").write_text(_json(row) + "\n", encoding="utf-8")
        rows.append(row)
        print(info["name"], "ERROR" if failure else record["diagnostics"]["status"], flush=True)
    summary = summarize(rows)
    preserved = plan["protected_files"] == _protected_files(roots)
    summary.update(historical_files_preserved=preserved, results=rows)
    summary["gates"]["protected_files_preserved"] = preserved
    (destination / "results.json").write_text(_json(summary) + "\n", encoding="utf-8")
    lines = ["# Synthetic canonical counting comparison", "",
             "Count gates are preliminary. Full-mask and cell reindexing require an independent saved-artifact audit.",
             "Withholding is not a correct numerical prediction. Clinical accuracy is unmeasured.", "",
             "| Group | Method | Records | Correct accepted | Wrong accepted | Withheld | Errors |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in summary["comparison"]:
        for method in ("baseline", "candidate"):
            counts = group[method]
            lines.append(f"| {group['group']} | {method} | {group['case_count']} | {counts['correct_accepted']} | "
                         f"{counts['wrong_accepted']} | {counts['withheld']} | {counts['errors']} |")
    for group in summary["comparison"]:
        lines += ["", f"Correct accepted counts lost in {group['group']}: {group['correct_accepted_lost']}."]
    lines += ["", f"Preliminary gates: {summary['gates']}.", f"Material twins: {summary['material_twins']}.",
              f"Execution errors: {summary['execution_errors']}. Historical files preserved: {preserved}.",
              "All records are reused development observations; exact reindexings are not independent examples.",
              "A bridge through bone-valued voxels is image evidence and does not establish a physical opening."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {path.relative_to(destination).as_posix(): _sha(path)
                for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "sha256.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    if summary["execution_errors"] or not preserved:
        raise RuntimeError("Experiment retained failures; inspect saved results.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--known-run", type=Path)
    choice.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.known_run is not None:
        print(freeze(args.plan, args.known_run))
    else:
        result = execute(args.plan, args.output)
        print(_json({key: value for key, value in result.items() if key != "results"}))
        sys.exit(0 if all(result["gates"].values()) else 2)
