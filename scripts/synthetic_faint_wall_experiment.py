"""Frozen paired baseline/candidate comparison on synthetic inputs only."""

from dataclasses import asdict
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_counting_experiment import REPO, array_hash, construction_counts
from synthetic_blur_experiment import fixtures as development_cases, record as development_record, score
from synthetic_faint_wall_candidate import METHOD, REASON, count_with_faint_wall, infer_exclusion
from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, METHOD_VERSION, compartment_stage


def case_record(case):
    from synthetic_faint_wall_cases import ChallengeCase, challenge_record

    challenge = isinstance(case, ChallengeCase)
    result = challenge_record(case) if challenge else development_record(case)
    result.update(group="challenge" if challenge else "development",
                  case_kind="pair" if case.fixture.components == 2 else "control")
    return result


def cases():
    from synthetic_faint_wall_cases import challenge_cases

    return development_cases() + challenge_cases()


def protocol(items):
    sources = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py")) + sorted(
        (REPO / "scripts").glob("synthetic_*.py"))
    return {"experiment": "synthetic-faint-wall-v1", "clinical_accuracy": "NOT_MEASURED",
            "baseline_method": METHOD_VERSION, "candidate_method": METHOD,
            "candidate_parameters": {"weak_hu_inclusive": 50, "strong_hu_inclusive": 200,
                                     "connectivity": "face", "attachment": "strong inside ROI"},
            "config": asdict(CountingConfig()), "fixtures": [case_record(case) for case in items],
            "runtime": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
            "python": sys.version,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "source_hashes": {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "rules_sha256": hashlib.sha256((REPO / "docs/SYNTHETIC_FAINT_WALL_EXPERIMENT.md").read_bytes()).hexdigest()}


def freeze_plan():
    plan = REPO / ".local/frozen-faint-wall-plan.json"
    plan.parent.mkdir(exist_ok=True)
    snapshot = protocol(cases())
    with plan.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True, allow_nan=False)
    return plan


def connectivity(case, exclusion):
    free, n = ndi.label(case.fixture.roi & (case.fixture.hu < 200) & ~exclusion)
    return {"components": int(n), "core_component_ids": {
        str(int(i)): [int(v) for v in np.unique(free[case.cores == i])]
        for i in np.unique(case.cores) if i}}


def measure(case, method):
    f, before = case.fixture, case_record(case)
    cavities, air, exclusion = np.zeros_like(f.labels), np.zeros_like(f.labels), np.zeros_like(f.roi)
    empty_score = score(case, cavities, [])
    cells, warnings, failure = [], [], None
    try:
        if method == "baseline":
            cavities, air, cells, warnings = compartment_stage(f.hu, f.roi, f.affine, CountingConfig())
        elif method == "candidate":
            exclusion = infer_exclusion(f.hu, f.roi)
            cavities, air, cells, warnings, actual_exclusion = count_with_faint_wall(f.hu, f.roi, f.affine)
            assert np.array_equal(exclusion, actual_exclusion)
        else:
            raise ValueError("Unknown experiment method")
        scored = score(case, cavities, cells)
        assert cavities.dtype == air.dtype == np.uint16
        assert np.array_equal(air, np.where(f.hu < -500, cavities, 0))
        assert not np.any((cavities > 0) & (~f.roi | exclusion))
        for cell in cells:
            mask = cavities == cell["id"]
            assert cell["voxel_count"] == np.count_nonzero(mask)
            assert cell["air_voxel_count"] == np.count_nonzero(air == cell["id"])
            centroid = f.affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + f.affine[:3, 3]
            assert np.allclose(centroid, cell["centroid_ras_mm"], rtol=0, atol=1e-8)
            assert cell["uncertain"] == bool(cell["reasons"])
            assert (REASON in cell["reasons"]) == bool(method == "candidate" and exclusion.any())
        assert before == case_record(case)
    except CountingError as error:
        failure = error.code
    except Exception as error:
        failure = "UNEXPECTED_" + type(error).__name__
    if failure:
        cavities, air = np.zeros_like(f.labels), np.zeros_like(f.labels)
        cells = []
        scored = empty_score
        scored["candidate_counts"] = {side: None for side in ("left", "right", "total")}
    preserved = before == case_record(case)
    result = scored | {"method": method, "cells": cells, "warnings": warnings, "failure_code": failure,
                       "inputs_preserved": preserved, "weak_evidence_used": bool(exclusion.any()),
                       "inferred_exclusion_voxels": int(exclusion.sum()),
                       "input_connectivity": connectivity(case, exclusion),
                       "detected_hashes": {name: array_hash(a) for name, a in
                                           (("cavities", cavities), ("air", air), ("exclusion", exclusion))}}
    result["passed"] = bool(result["passed"] and preserved and not failure)
    return result, cavities, air, exclusion


def admission(rows):
    recovery = {}
    for group in ("development", "challenge"):
        selected = [r for r in rows if r["group"] == group and r["case_kind"] == "pair"]
        recovery[group] = {method: sum(r[method]["passed"] for r in selected) for method in ("baseline", "candidate")}
    regressions = [r["name"] for r in rows if r["baseline"]["passed"] and not r["candidate"]["passed"]]
    controls = [r for r in rows if r["case_kind"] == "control"]
    false_splits = [r["name"] for r in controls if (r["candidate"]["candidate_counts"]["total"] or 0) > 1]
    execution_errors = [f"{r['name']}:{m}" for r in rows for m in ("baseline", "candidate")
                        if (r[m]["failure_code"] or "").startswith("UNEXPECTED_") or not r[m]["inputs_preserved"]]
    coverage = {(g, k): sum(r["group"] == g and r["case_kind"] == k for r in rows)
                for g in ("development", "challenge") for k in ("pair", "control")}
    gates = {"complete_unique_coverage": len(rows) == len({r["name"] for r in rows}) == 56 and coverage == {
                ("development", "pair"): 18, ("development", "control"): 6,
                ("challenge", "pair"): 16, ("challenge", "control"): 16},
             "development_pair_improvement": recovery["development"]["candidate"] > recovery["development"]["baseline"],
             "challenge_pair_improvement": recovery["challenge"]["candidate"] > recovery["challenge"]["baseline"],
             "baseline_successes_preserved": not regressions,
             "all_controls_recovered": bool(controls) and all(r["candidate"]["passed"] for r in controls),
             "no_false_splits": not false_splits, "no_execution_errors": not execution_errors}
    return {"admitted": all(gates.values()), "gates": gates, "paired_recovery": recovery,
            "controls_recovered": {m: sum(r[m]["passed"] for r in controls) for m in ("baseline", "candidate")},
            "baseline_success_regressions": regressions, "control_false_splits": false_splits,
            "execution_errors": execution_errors, "clinical_accuracy": "NOT_MEASURED"}


def run(destination):
    from PIL import Image, ImageDraw
    from synthetic_counting_experiment import draw_case, font
    import textwrap

    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(REPO) or any((p / ".git").exists() for p in (destination, *destination.parents)):
        raise ValueError("Synthetic artifacts must stay outside source repositories.")
    if destination.exists():
        raise FileExistsError(destination)
    items = cases()
    snapshot = protocol(items)
    plan = json.loads((REPO / ".local/frozen-faint-wall-plan.json").read_text(encoding="utf-8"))
    if plan != snapshot:
        raise ValueError("Inputs, rules, configuration or source differ from the frozen plan.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "protocol.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    rows, pages = [], []
    for index, case in enumerate(items, start=1):
        f = case.fixture
        case_dir = destination / f.name
        case_dir.mkdir()
        np.savez_compressed(case_dir / "input.npz", hu=f.hu, labels=f.labels, roi=f.roi, affine=f.affine,
                            highres_labels=case.highres_labels, cores=case.cores)
        row = case_record(case) | {"index": index}
        for method in ("baseline", "candidate"):
            isolated = copy.deepcopy(case)
            for array in (isolated.fixture.hu, isolated.fixture.labels, isolated.fixture.roi,
                          isolated.fixture.affine, isolated.highres_labels, isolated.cores):
                array.flags.writeable = False
            result, cavities, air, exclusion = measure(isolated, method)
            np.savez_compressed(case_dir / f"{method}.npz", cavities=cavities, air=air, exclusion=exclusion)
            row[method] = result
            page = draw_case(f, result, cavities, case_dir / f"{method}.png")
            draw = ImageDraw.Draw(page)
            draw.rectangle((20, 12, 1540, 208), fill="#f5f5f3")
            draw.text((34, 20), f"SYNTHETIC FAINT-WALL COMPARISON: {row['group'].upper()}", font=font(24, True), fill="#404040")
            draw.text((34, 66), f"Case {index:02d}: {method.capitalize()}", font=font(36, True), fill="#171717")
            for line, text in enumerate(textwrap.wrap(f.description, width=100)):
                draw.text((34, 115 + line * 24), text, font=font(21), fill="#404040")
            draw.text((34, 178), f"Declared spaces: {construction_counts(f)['total']}    Detected: {result['candidate_counts']['total']}    Inferred weak voxels: {result['inferred_exclusion_voxels']}", font=font(23, True), fill="#171717")
            for column, label in enumerate(("Original imaged HU", "Physical reference masks", "Detected masks")):
                draw.rectangle((20 + 510 * column, 208, 525 + 510 * column, 244), fill="#f5f5f3")
                draw.text((34 + 510 * column, 214), label, font=font(22, True), fill="#171717")
            draw.rectangle((20, 858, 1530, 912), fill="#f5f5f3")
            draw.text((34, 863), f"Compound recovery: {'met' if result['passed'] else 'NOT MET'}. Physical connectivity is an engineering convention.", font=font(22, True), fill="#171717")
            draw.text((34, 891), f"Smallest reference Dice: {min(m['dice'] for m in result['matches']):.3f}; core coverage: {min(m['core_coverage'] for m in result['matches']):.1%}. Missing matches score zero.", font=font(20), fill="#404040")
            page.save(case_dir / f"{method}.png")
            pages.append(page)
        (case_dir / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        rows.append(row)
    summary = admission(rows)
    summary.update(fixture_count=len(rows), results=rows)
    summary["paired_counts_correct"] = {g: {m: sum(r[m]["exact_count_pass"] for r in rows
        if r["group"] == g and r["case_kind"] == "pair") for m in ("baseline", "candidate")}
        for g in ("development", "challenge")}
    summary["recovery_failures"] = {m: [r["name"] for r in rows if not r[m]["passed"]
        and f"{r['name']}:{m}" not in summary["execution_errors"]] for m in ("baseline", "candidate")}
    (destination / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    lines = ["# Synthetic faint-wall comparison", "", f"Candidate admission: {'MET' if summary['admitted'] else 'REJECTED'}.", "",
             "One fixed strong-connected 50 HU exclusion mask was compared with the unchanged counter. All 56 cases remain in the denominator.", "",
             f"Compound paired recovery: {summary['paired_recovery']}. All controls recovered: {summary['controls_recovered']} / 22.", "",
             f"Admission gates: {summary['gates']}.", "",
             "The known 24 cases are development evidence; 32 new constructions are a frozen synthetic challenge, not a clinical or statistical holdout.", "",
             "| ID | Case | Group | Kind | Declared | Baseline count | Candidate count | Baseline recovery | Candidate recovery | Worst candidate Dice | Worst core |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b, c = r["baseline"], r["candidate"]
        lines.append(f"| {r['index']:02d} | [{r['name']}]({r['name']}/candidate.png) | {r['group']} | {r['case_kind']} | {r['construction_counts']['total']} | {b['candidate_counts']['total']} | {c['candidate_counts']['total']} | {b['passed']} | {c['passed']} | {min(m['dice'] for m in c['matches']):.3f} | {min(m['core_coverage'] for m in c['matches']):.1%} |")
    lines += ["", f"Regressed baseline successes: {summary['baseline_success_regressions']}.", "",
              f"Candidate controls with extra divisions: {summary['control_false_splits']}.", "",
              f"Unexpected execution errors: {summary['execution_errors']}.", "",
              "Each case retains baseline and candidate NPZ/PNG files, all numerical matches, masks, registry reasons, null failures and original input hashes.", "",
              "A complete inferred exclusion can split a soft-tissue sheet as if it were a wall. Attachment to strong bone does not identify material.", "",
              "The open bone-divider controls use connected-space truth. The prior counter can infer two uncertain cells there; this is a declared engineering convention, not a clinical error verdict.", "",
              "The one-candidate budget ends here regardless of admission. A repeated process verifies integrity and does not add 56 independent observations. No patient, scanner-calibration or anatomical-localization accuracy was measured."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    covers = []
    for start in range(0, len(rows), 28):
        cover = Image.new("RGB", (1560, 1240), "#f5f5f3")
        draw = ImageDraw.Draw(cover)
        draw.text((44, 30), "Faint walls: baseline and one fixed candidate", font=font(34, True), fill="#171717")
        draw.text((44, 88), f"Admission {'MET' if summary['admitted'] else 'REJECTED'}. Development pairs: {summary['paired_recovery']['development']}. Challenge pairs: {summary['paired_recovery']['challenge']}.", font=font(21, True), fill="#171717")
        draw.text((44, 127), f"Controls recovered: {summary['controls_recovered']} / 22. Extra candidate divisions: {len(summary['control_false_splits'])}. Execution errors: {len(summary['execution_errors'])}.", font=font(22), fill="#404040")
        columns = (44, 145, 420, 670, 870, 1080, 1280)
        for x, label in zip(columns, ("Case ID", "Group / kind", "Declared", "Base count", "Cand count", "Base met", "Cand met")):
            draw.text((x, 192), label, font=font(23, True), fill="#171717")
        for i, r in enumerate(rows[start:start + 28]):
            values = (f"{r['index']:02d}", f"{r['group']} / {r['case_kind']}", str(r['construction_counts']['total']),
                      str(r['baseline']['candidate_counts']['total']), str(r['candidate']['candidate_counts']['total']),
                      "yes" if r['baseline']['passed'] else "NO", "yes" if r['candidate']['passed'] else "NO")
            for x, value in zip(columns, values):
                draw.text((x, 242 + i * 29), value, font=font(21), fill="#171717")
        draw.text((44, 1090), "Case IDs map to both numbered comparison pages and the complete Markdown table.", font=font(23), fill="#404040")
        draw.text((44, 1136), "Synthetic connected-space convention. All failures retained. Clinical accuracy unmeasured.", font=font(23), fill="#404040")
        cover.save(destination / f"summary-{start // 28 + 1}.png")
        covers.append(cover)
    covers[0].save(destination / "report.pdf", save_all=True, append_images=covers[1:] + pages, resolution=140)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze-plan", action="store_true")
    choice.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze_plan:
        print(freeze_plan())
    else:
        result = run(args.output)
        sys.exit(1 if result["execution_errors"] else 0 if result["admitted"] else 2)
