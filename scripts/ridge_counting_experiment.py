"""Freeze and evaluate one synthetic ridge candidate without rerunning old methods."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_counting_experiment import Fixture, REPO, array_hash
from synthetic_blur_experiment import ImagingCase, score
from ethmoid_pipeline.counting.engine import CountingError

ARRAY_NAMES = ("hu", "labels", "roi", "affine", "highres_labels", "cores")
NULL_COUNTS = {"left": None, "right": None, "total": None}


def count_with_ridges(hu, roi, affine):
    from ridge_counting_candidate import count_with_ridges as candidate
    return candidate(hu, roi, affine)


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays(case):
    return {name: getattr(case.fixture if name in ARRAY_NAMES[:4] else case, name)
            for name in ARRAY_NAMES}


def _hashes(case):
    return {name: array_hash(array) for name, array in _arrays(case).items()}


def _external_path(path, *, fresh=True):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(REPO) or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Synthetic artifacts must stay outside source repositories.")
    if fresh and path.exists():
        raise FileExistsError(path)
    return path


def load_items(known_run):
    """Load frozen historical arrays; only the new stress constructors execute."""
    from ridge_stress_cases import cases, metadata

    known_run = Path(known_run).resolve()
    saved = json.loads((known_run / "results.json").read_text(encoding="utf-8"))["results"]
    if len(saved) != 56 or len({row["name"] for row in saved}) != 56:
        raise ValueError("The known run must contain all 56 unique historical cases.")
    items = []
    for row in saved:
        name = row["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid saved synthetic case name.")
        case_dir = (known_run / name).resolve()
        if not case_dir.is_relative_to(known_run):
            raise ValueError("Saved case escapes the known run.")
        if json.loads((case_dir / "result.json").read_text(encoding="utf-8")) != row:
            raise ValueError("Saved aggregate and per-case results disagree.")
        with np.load(case_dir / "input.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_NAMES):
                raise ValueError("Saved input array manifest differs from the historical format.")
            arrays = {name: archive[name] for name in ARRAY_NAMES}
        if {name: array_hash(array) for name, array in arrays.items()} != row["hashes"]:
            raise ValueError("Saved input arrays differ from their recorded hashes.")
        fixture = Fixture(name, row["description"], *(arrays[key] for key in ARRAY_NAMES[:4]),
                          category=row["category"], components=row["construction_counts"]["total"],
                          required_reasons={int(key): value for key, value in row["required_reasons"].items()})
        case = ImagingCase(fixture, arrays["highres_labels"], arrays["cores"],
                           row["wall_mm"], row["sigma_mm"], row["phase_mm"])
        record = {key: value for key, value in row.items() if key not in ("baseline", "candidate")}
        record.update(group="known_development", original_group=row["group"],
                      expected_abstention=False, equivalence_group=None,
                      saved_baseline=row["baseline"], saved_faint_wall=row["candidate"])
        items.append((case, record))
    for case in cases():
        record = metadata(case)
        record.update(group="fresh_stress", case_kind="material_ambiguity" if record["expected_abstention"] else "control")
        items.append((case, record))
    if len(items) != 68 or len({case.fixture.name for case, _ in items}) != 68:
        raise ValueError("The experiment requires 68 unique cases.")
    for case, _ in items:
        for array in _arrays(case).values():
            array.setflags(write=False)
    return items


def protocol(known_run, items):
    known_run = Path(known_run).resolve()
    sources = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py")) + [
        REPO / "scripts" / name for name in ("ridge_counting_candidate.py", "ridge_counting_experiment.py",
        "ridge_stress_cases.py", "synthetic_blur_experiment.py", "synthetic_counting_experiment.py")]
    saved_files = [known_run / name for name in ("protocol.json", "results.json")]
    for case, record in items:
        if record["group"] == "known_development":
            saved_files.extend(known_run / case.fixture.name / name
                               for name in ("input.npz", "baseline.npz", "candidate.npz", "result.json"))
    return {"experiment": "synthetic-continuous-ridge-v1", "clinical_accuracy": "NOT_MEASURED",
            "scope": "All historical cases are development; fresh constructions are bounded synthetic stress evidence.",
            "candidate_budget": 1, "known_run": str(known_run),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "python": sys.version,
            "runtime": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
            "source_hashes": {path.relative_to(REPO).as_posix(): _sha(path) for path in sources},
            "rules_sha256": _sha(REPO / "docs/SYNTHETIC_RIDGE_EXPERIMENT.md"),
            "saved_file_hashes": {path.relative_to(known_run).as_posix(): _sha(path) for path in saved_files},
            "fixtures": [{"metadata": record, "input_hashes": _hashes(case)} for case, record in items]}


def freeze_plan(path, known_run):
    path = _external_path(path)
    known_run = Path(known_run).expanduser().resolve()
    snapshot = protocol(known_run, load_items(known_run))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(snapshot) + "\n")
    return path


def measure(case):
    """Score against untouched truth, keeping abstention and proposals separate."""
    f = case.fixture
    original = _hashes(case)
    inputs = {name: getattr(f, name).copy() for name in ("hu", "roi", "affine")}
    input_hashes = {name: array_hash(array) for name, array in inputs.items()}
    for array in inputs.values():
        array.setflags(write=False)
    arrays = {name: np.zeros(f.labels.shape, dtype=np.int32 if name == "partitions" else np.uint16)
              for name in ("cavities", "air", "partitions")}
    empty_score = score(case, arrays["cavities"], [])
    cells, warnings, failure = [], [], None
    diagnostics = {"status": "withheld", "counts": NULL_COUNTS.copy(), "withheld_reasons": [],
                   "experimental_count_available": False, "clinical_accuracy": "NOT_MEASURED"}
    try:
        output = count_with_ridges(inputs["hu"], inputs["roi"], inputs["affine"])
        arrays = {name: output[name] for name in arrays}
        cells, warnings, diagnostics = (output[key] for key in ("cells", "warnings", "diagnostics"))
        if not isinstance(cells, list) or not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
            raise ValueError("Invalid registry or warning format.")
        for name, array in arrays.items():
            if array.shape != f.labels.shape or array.dtype != (np.int32 if name == "partitions" else np.uint16) or np.any(array < 0):
                raise ValueError("Invalid prediction array shape, dtype or labels.")
        if not np.array_equal(arrays["air"], np.where(f.hu < -500, arrays["cavities"], 0)):
            raise ValueError("Air labels differ from the original-HU content mask.")
        if any(np.any((array > 0) & (~f.roi | (f.hu >= 200))) for array in arrays.values()):
            raise ValueError("Prediction labels leave eligible space.")
        scored = score(case, arrays["cavities"], cells)
        for cell in cells:
            mask = arrays["cavities"] == cell["id"]
            center = f.affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + f.affine[:3, 3]
            if (cell["voxel_count"] != np.count_nonzero(mask)
                    or cell["air_voxel_count"] != np.count_nonzero(arrays["air"] == cell["id"])
                    or not np.allclose(center, cell["centroid_ras_mm"], rtol=0, atol=1e-8)
                    or cell["uncertain"] != bool(cell["reasons"])):
                raise ValueError("Prediction registry differs from its masks.")
        status = diagnostics["status"]
        if status not in ("accepted", "withheld") or diagnostics["clinical_accuracy"] != "NOT_MEASURED":
            raise ValueError("Invalid experimental status.")
        if (not isinstance(diagnostics["method"], str)
                or not isinstance(diagnostics["withheld_reasons"], list)
                or any(not isinstance(reason, str) for reason in diagnostics["withheld_reasons"])
                or not isinstance(diagnostics["passage_merges"], list)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in (diagnostics["initial_basin_count"], diagnostics["final_basin_count"]))
                or diagnostics["final_basin_count"] != len(np.unique(arrays["partitions"][arrays["partitions"] > 0]))):
            raise ValueError("Invalid basin or uncertainty diagnostics.")
        if (diagnostics["counts"] != (scored["candidate_counts"] if status == "accepted" else NULL_COUNTS)
                or diagnostics["experimental_count_available"] is not (status == "accepted")
                or bool(diagnostics["withheld_reasons"]) != (status == "withheld")):
            raise ValueError("Count availability differs from experimental status.")
        if status == "accepted" and any(not isinstance(value, int) or isinstance(value, bool) for value in diagnostics["counts"].values()):
            raise ValueError("Accepted counts must be integers, excluding booleans.")
        if _hashes(case) != original or any(array_hash(inputs[name]) != value for name, value in input_hashes.items()):
            raise ValueError("Detector mutated its isolated inputs.")
        _json({"cells": cells, "warnings": warnings, "diagnostics": diagnostics})
    except CountingError as error:
        failure = error.code
    except Exception as error:
        failure = "UNEXPECTED_" + type(error).__name__
    preserved = _hashes(case) == original and all(array_hash(inputs[name]) == value for name, value in input_hashes.items())
    if failure:
        arrays = {name: np.zeros(f.labels.shape, dtype=np.int32 if name == "partitions" else np.uint16) for name in arrays}
        cells, warnings, scored = [], [], empty_score
        diagnostics = {"status": "withheld", "counts": NULL_COUNTS.copy(), "withheld_reasons": [failure],
                       "experimental_count_available": False, "clinical_accuracy": "NOT_MEASURED"}
    hashes = {name: array_hash(array) for name, array in arrays.items()}
    fingerprint = hashlib.sha256(_json({"arrays": hashes, "cells": cells, "warnings": warnings,
                                       "diagnostics": diagnostics, "failure_code": failure}).encode()).hexdigest()
    accepted = bool(scored["passed"] and diagnostics["status"] == "accepted" and not failure and preserved)
    result = scored | {"proposal_counts": scored["candidate_counts"], "candidate_counts": diagnostics["counts"],
                       "proposal_passed": bool(scored["passed"] and not failure and preserved),
                       "accepted_recovery": accepted, "passed": accepted, "status": diagnostics["status"],
                       "cells": cells, "warnings": warnings, "diagnostics": diagnostics, "failure_code": failure,
                       "inputs_preserved": preserved, "detected_hashes": hashes, "output_fingerprint": fingerprint}
    return result, arrays


def admission(rows):
    known = [row for row in rows if row["group"] == "known_development"]
    pairs = [row for row in known if row["case_kind"] == "pair"]
    controls = [row for row in known if row["case_kind"] == "control"]
    fresh = [row for row in rows if row["group"] == "fresh_stress"]
    fresh_controls = [row for row in fresh if not row["expected_abstention"]]
    twins = [row for row in fresh if row["expected_abstention"]]
    groups = {}
    for row in twins:
        groups.setdefault(row["equivalence_group"], []).append(row)
    equivalent = {}
    for name, group in groups.items():
        equivalent[name] = (len(group) == 2 and bool(name)
                           and len({tuple(row["input_hashes"][key] for key in ("hu", "roi", "affine")) for row in group}) == 1
                           and len({row["candidate"]["output_fingerprint"] for row in group}) == 1
                           and all(row["candidate"]["status"] == "withheld"
                                   and row["candidate"]["candidate_counts"]["total"] is None
                                   and row["candidate"]["failure_code"] is None for row in group))
    regressions = [row["name"] for row in known if row["saved_faint_wall"]["passed"] and not row["candidate"]["accepted_recovery"]]
    proposal_regressions = [row["name"] for row in known if row["saved_faint_wall"]["passed"] and not row["candidate"]["proposal_passed"]]
    false_splits = [row["name"] for row in controls + fresh_controls if row["candidate"]["status"] == "accepted"
                    and (row["candidate"]["candidate_counts"]["total"] or 0) > 1]
    errors = [row["name"] for row in rows if (row["candidate"]["failure_code"] or "").startswith("UNEXPECTED_")
              or not row["candidate"]["inputs_preserved"]]
    count = lambda selected, key: sum(row["candidate"][key] for row in selected)
    previous = sum(row["saved_faint_wall"]["passed"] for row in pairs)
    gates = {"complete_unique_coverage": len(rows) == len({row["name"] for row in rows}) == 68
             and (len(known), len(pairs), len(controls), len(fresh), len(fresh_controls), len(twins)) == (56, 34, 22, 12, 4, 8),
             "known_pair_improvement": previous == 14 and count(pairs, "accepted_recovery") > previous,
             "faint_wall_successes_preserved": not regressions,
             "all_known_controls_recovered": len(controls) == 22 and count(controls, "accepted_recovery") == 22,
             "all_fresh_controls_recovered": len(fresh_controls) == 4 and count(fresh_controls, "accepted_recovery") == 4,
             "all_material_twins_withheld_identically": len(twins) == 8 and len(groups) == 4 and all(equivalent.values()),
             "no_false_accepted_control_splits": not false_splits, "no_execution_errors": not errors}
    return {"admitted": all(gates.values()), "gates": gates,
            "mechanism_recovery": {"known_pair_denominator": len(pairs),
                                   "known_pair_proposal_recovery": count(pairs, "proposal_passed"),
                                   "previous_faint_pair_recovery": previous,
                                   "known_pair_proposal_improvement": count(pairs, "proposal_passed") > previous,
                                   "faint_wall_proposal_successes_preserved": not proposal_regressions,
                                   "faint_wall_proposal_regressions": proposal_regressions},
            "known_pairs": {"denominator": len(pairs), "saved_faint_wall_recovery": previous,
                            "proposal_recovery": count(pairs, "proposal_passed"), "accepted_recovery": count(pairs, "accepted_recovery")},
            "known_controls": {"denominator": len(controls), "accepted_recovery": count(controls, "accepted_recovery")},
            "fresh_controls": {"denominator": len(fresh_controls), "accepted_recovery": count(fresh_controls, "accepted_recovery")},
            "material_twins": {"denominator": len(twins), "equivalence_groups": equivalent},
            "withheld_cases": [row["name"] for row in rows if row["candidate"]["status"] == "withheld"],
            "faint_wall_success_regressions": regressions, "false_accepted_control_splits": false_splits,
            "execution_errors": errors, "clinical_accuracy": "NOT_MEASURED"}


def run(plan_path, destination):
    destination = _external_path(destination)
    plan_path = _external_path(plan_path, fresh=False)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    known_run = Path(plan["known_run"])
    items = load_items(known_run)
    if protocol(known_run, items) != plan:
        raise ValueError("Inputs, sources, rules or runtime differ from the frozen plan.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "protocol.json").write_text(_json(plan) + "\n", encoding="utf-8")
    rows = []
    for index, (case, metadata) in enumerate(items, start=1):
        case_dir = destination / case.fixture.name
        case_dir.mkdir()
        np.savez_compressed(case_dir / "input.npz", **_arrays(case))
        result, arrays = measure(case)
        np.savez_compressed(case_dir / "proposal.npz", **arrays)
        row = metadata | {"index": index, "input_hashes": _hashes(case), "candidate": result}
        (case_dir / "result.json").write_text(_json(row) + "\n", encoding="utf-8")
        rows.append(row)
    summary = admission(rows)
    summary["gates"]["frozen_inputs_preserved"] = protocol(known_run, items) == plan
    summary["admitted"] = all(summary["gates"].values())
    summary.update(fixture_count=len(rows), results=rows)
    (destination / "results.json").write_text(_json(summary) + "\n", encoding="utf-8")
    lines = ["# Synthetic continuous-ridge experiment", "", f"Admission: {'MET' if summary['admitted'] else 'REJECTED'}.", "",
             f"Known pairs: {summary['known_pairs']}. Known controls: {summary['known_controls']}. Fresh controls: {summary['fresh_controls']}.", "",
             f"Proposal mechanism results: {summary['mechanism_recovery']}. These do not establish an accepted count.", "",
             "All 56 previously exposed cases are development evidence. The 12 new constructions test fixed material and opening rules; they are not a clinical or statistical holdout.", "",
             "Proposal recovery measures spatial masks. Accepted recovery also requires an available experimental count. Withheld counts remain in every denominator.", "",
             "| Case | Group | Status | Declared | Proposed | Accepted | Proposal recovery | Accepted recovery | Worst Dice | Worst core coverage |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        result = row["candidate"]
        lines.append(f"| {row['name']} | {row['group']} | {result['status']} | {result['construction_counts']['total']} | {result['proposal_counts']['total']} | {result['candidate_counts']['total']} | {result['proposal_passed']} | {result['accepted_recovery']} | {min(m['dice'] for m in result['matches']):.3f} | {min(m['core_coverage'] for m in result['matches']):.3f} |")
    lines += ["", f"Admission gates: {summary['gates']}.", "",
              f"Material-twin equivalence: {summary['material_twins']}.", "",
              f"Regressed faint-wall successes: {summary['faint_wall_success_regressions']}.", "",
              f"False accepted control splits: {summary['false_accepted_control_splits']}.", "",
              f"Execution errors: {summary['execution_errors']}.", "",
              "Identical-HU twins must have identical predictions and withheld counts; abstention is not recovered physical counting.", "",
              "The one-candidate budget ends regardless of admission. A second process run checks repeatability and adds no independent observations. Clinical accuracy was not measured."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {path.relative_to(destination).as_posix(): _sha(path)
                for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "sha256.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze-plan", type=Path)
    choice.add_argument("--plan", type=Path)
    parser.add_argument("--known-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze_plan:
        if args.known_run is None or args.output is not None:
            parser.error("--freeze-plan requires --known-run and does not accept --output")
        print(freeze_plan(args.freeze_plan, args.known_run))
    else:
        if args.output is None or args.known_run is not None:
            parser.error("--plan requires --output and does not accept --known-run")
        result = run(args.plan, args.output)
        print(_json({key: value for key, value in result.items() if key != "results"}))
        sys.exit(1 if result["execution_errors"] else 0 if result["admitted"] else 2)
