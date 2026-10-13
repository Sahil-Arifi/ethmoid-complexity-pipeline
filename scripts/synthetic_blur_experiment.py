"""Fixed physical phantoms, simplified imaging and independent recovery scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_counting_experiment import Fixture, REPO, array_hash, construction_counts, draw_case, font

__all__ = ["ImagingCase", "make_case", "fixtures", "record", "score", "visibility"]

SHAPE = (48, 40, 32)
SPACING_MM = 0.5
FINE_SPACING_MM = 0.125
FACTOR = 4
CONTROL_REASON = "crosses_midline_centroid_laterality_requires_review"


@dataclass(frozen=True)
class ImagingCase:
    fixture: Fixture
    highres_labels: np.ndarray
    cores: np.ndarray
    wall_mm: float
    sigma_mm: float
    phase_mm: float


def _physical_labels(coordinates, wall_mm: float, inset_mm: float = 0) -> np.ndarray:
    x, y, z = coordinates
    labels = np.zeros(tuple(len(axis) for axis in coordinates), dtype=np.uint16)
    yz = ((y[None, :, None] >= -6 + inset_mm) & (y[None, :, None] < 6 - inset_mm)
          & (z[None, None, :] >= -4 + inset_mm) & (z[None, None, :] < 4 - inset_mm))
    boxes = [(-8, 8)] if wall_mm == 0 else [(-8, -wall_mm / 2), (wall_mm / 2, 8)]
    for identifier, (lower, upper) in enumerate(boxes, start=1):
        inside_x = (x[:, None, None] >= lower + inset_mm) & (x[:, None, None] < upper - inset_mm)
        labels[inside_x & yz] = identifier
    return labels


def make_case(wall_mm: float, sigma_mm: float, phase_mm: float) -> ImagingCase:
    """Construct independent physical labels, image HU and 1 mm inset reference cores."""
    if wall_mm not in (0, 0.25, 0.5, 1) or sigma_mm not in (0, 0.25, 0.5) or phase_mm not in (0, 0.25):
        raise ValueError("Parameters must belong to the fixed 24-case design.")
    edges = np.array([-12 + phase_mm, -10, -8], dtype=np.float64)
    centers = tuple(edge + (np.arange(size) + 0.5) * SPACING_MM for edge, size in zip(edges, SHAPE))
    fine_centers = tuple(edge + (np.arange(size * FACTOR) + 0.5) * FINE_SPACING_MM
                         for edge, size in zip(edges, SHAPE))
    highres_labels = _physical_labels(fine_centers, wall_mm).astype(np.uint8)
    labels = _physical_labels(centers, wall_mm)
    cores = _physical_labels(centers, wall_mm, inset_mm=1)
    fine_hu = np.where(highres_labels > 0, -900.0, 800.0)
    if sigma_mm:
        ndi.gaussian_filter(fine_hu, sigma=sigma_mm / FINE_SPACING_MM, output=fine_hu,
                            mode="constant", cval=800.0, truncate=4.0)
    hu = fine_hu.reshape(SHAPE[0], FACTOR, SHAPE[1], FACTOR, SHAPE[2], FACTOR).mean(axis=(1, 3, 5)).astype(np.float32)
    affine = np.diag([SPACING_MM, SPACING_MM, SPACING_MM, 1.0])
    affine[:3, 3] = edges + SPACING_MM / 2
    parameters = "_".join(f"{label}_{value:g}".replace(".", "p") for label, value in
                          (("wall", wall_mm), ("sigma", sigma_mm), ("phase", phase_mm)))
    expected = 2 if wall_mm else 1
    fixture = Fixture(parameters, f"{expected} physical space(s); wall {wall_mm:g} mm, blur sigma {sigma_mm:g} mm, X phase {phase_mm:g} mm.",
                      hu, labels, np.ones(SHAPE, dtype=bool), affine, components=expected,
                      required_reasons={} if wall_mm else {1: CONTROL_REASON})
    assert ndi.label(highres_labels > 0)[1] == expected
    assert np.count_nonzero(highres_labels) * FINE_SPACING_MM ** 3 == (16 - wall_mm) * 12 * 8
    for identifier in range(1, expected + 1):
        assert ndi.label(highres_labels == identifier)[1] == 1
        assert np.any(cores == identifier)
        assert np.all(labels[cores == identifier] == identifier)
    for array in (hu, labels, fixture.roi, affine, highres_labels, cores):
        array.setflags(write=False)
    return ImagingCase(fixture, highres_labels, cores, float(wall_mm), float(sigma_mm), float(phase_mm))


def fixtures() -> list[ImagingCase]:
    return [make_case(wall, sigma, phase) for wall in (0, 0.25, 0.5, 1)
            for sigma in (0, 0.25, 0.5) for phase in (0, 0.25)]


def record(case: ImagingCase) -> dict:
    fixture = case.fixture
    truth_ids = [int(value) for value in np.unique(fixture.labels) if value]
    return {"name": fixture.name, "description": fixture.description, "category": fixture.category,
            "case_kind": "sealed_pair" if case.wall_mm else "absent_wall_control",
            "wall_mm": case.wall_mm, "sigma_mm": case.sigma_mm, "phase_mm": case.phase_mm,
            "shape": list(SHAPE), "spacing_mm": SPACING_MM, "affine": fixture.affine.tolist(),
            "highres_shape": list(case.highres_labels.shape), "highres_spacing_mm": FINE_SPACING_MM,
            "block_factor": FACTOR, "core_inset_mm": 1.0, "gaussian_truncate": 4.0,
            "exterior_hu": 800.0, "air_hu": -900.0,
            "construction_counts": construction_counts(fixture),
            "expected_air_volume_mm3": (16 - case.wall_mm) * 12 * 8,
            "highres_air_volume_mm3": float(np.count_nonzero(case.highres_labels) * FINE_SPACING_MM ** 3),
            "highres_nonbone_components": int(ndi.label(case.highres_labels > 0)[1]),
            "reference_label_voxels": {str(identifier): int(np.count_nonzero(fixture.labels == identifier)) for identifier in truth_ids},
            "reference_core_voxels": {str(identifier): int(np.count_nonzero(case.cores == identifier)) for identifier in truth_ids},
            "required_reasons": {str(identifier): reason for identifier, reason in fixture.required_reasons.items()},
            "hashes": {name: array_hash(array) for name, array in
                       [(name, getattr(fixture, name)) for name in ("hu", "labels", "roi", "affine")]
                       + [("highres_labels", case.highres_labels), ("cores", case.cores)]}}


def _side(affine: np.ndarray, mask: np.ndarray) -> str:
    center = np.argwhere(mask).mean(axis=0)
    x = float(affine[0, :3] @ center + affine[0, 3])
    return "left" if x <= 0 else "right"


def score(case: ImagingCase, detected: np.ndarray, cells: list[dict]) -> dict:
    """Report every truth object; retain numerical failure metrics and unmatched IDs."""
    truth = case.fixture.labels
    if detected.shape != truth.shape:
        raise ValueError("Prediction shape must match reference labels.")
    if not np.issubdtype(detected.dtype, np.integer) or np.any(detected < 0):
        raise ValueError("Predictions must contain nonnegative integer IDs.")
    truth_ids = [int(value) for value in np.unique(truth) if value]
    predicted_ids = [int(value) for value in np.unique(detected) if value]
    if any(isinstance(cell["id"], (bool, np.bool_)) or not isinstance(cell["id"], (int, np.integer)) for cell in cells):
        raise ValueError("Cell registry IDs must be integers, excluding booleans.")
    registry = {cell["id"]: cell for cell in cells}
    if len(registry) != len(cells) or set(registry) != set(predicted_ids) or any(
            cell["side"] not in {"left", "right"} for cell in cells):
        raise ValueError("Prediction IDs and cell registry must agree.")
    truth_sizes = {identifier: int(np.count_nonzero(truth == identifier)) for identifier in truth_ids}
    predicted_sizes = {identifier: int(np.count_nonzero(detected == identifier)) for identifier in predicted_ids}
    components = {identifier: int(ndi.label(detected == identifier)[1]) for identifier in predicted_ids}
    dice = np.zeros((len(truth_ids), len(predicted_ids)), dtype=np.float64)
    for row, truth_id in enumerate(truth_ids):
        for column, predicted_id in enumerate(predicted_ids):
            overlap = int(np.count_nonzero((truth == truth_id) & (detected == predicted_id)))
            dice[row, column] = 2 * overlap / (truth_sizes[truth_id] + predicted_sizes[predicted_id])
    mapping = {}
    if dice.size:
        rows, columns = linear_sum_assignment(dice, maximize=True)
        mapping = {truth_ids[row]: predicted_ids[column] for row, column in zip(rows, columns) if dice[row, column] > 0}
    unmatched = sorted(set(predicted_ids) - set(mapping.values()))
    matches = []
    for row, truth_id in enumerate(truth_ids):
        predicted_id = mapping.get(truth_id)
        core = case.cores == truth_id
        core_size = int(np.count_nonzero(core))
        if not core_size:
            raise ValueError("Every reference label must have a nonempty core.")
        matches.append({"truth_id": truth_id, "matched_id": predicted_id,
                        "truth_voxels": truth_sizes[truth_id], "core_voxels": core_size,
                        "predicted_voxels": predicted_sizes[predicted_id] if predicted_id is not None else 0,
                        "dice": float(dice[row, predicted_ids.index(predicted_id)]) if predicted_id is not None else 0.0,
                        "core_coverage": float(np.count_nonzero(core & (detected == predicted_id)) / core_size) if predicted_id is not None else 0.0,
                        "truth_side": _side(case.fixture.affine, truth == truth_id),
                        "predicted_side": registry[predicted_id]["side"] if predicted_id is not None else None,
                        "predicted_centroid_side": _side(case.fixture.affine, detected == predicted_id) if predicted_id is not None else None,
                        "predicted_components": components[predicted_id] if predicted_id is not None else 0})
    construction = construction_counts(case.fixture)
    candidate = {"left": sum(cell["side"] == "left" for cell in cells),
                 "right": sum(cell["side"] == "right" for cell in cells), "total": len(cells)}
    checks = {"exact_count_pass": candidate["total"] == construction["total"],
              "bijection_pass": len(mapping) == len(truth_ids) == len(predicted_ids) and not unmatched,
              "dice_pass": all(row["dice"] >= 0.90 for row in matches),
              "core_coverage_pass": all(row["core_coverage"] >= 0.95 for row in matches),
              "connected_pass": all(value == 1 for value in components.values()),
              "side_assignment_pass": not case.wall_mm or all(
                  row["truth_side"] == row["predicted_side"] == row["predicted_centroid_side"] for row in matches),
              "flags_pass": bool(case.wall_mm) or all(
                  row["matched_id"] is not None and CONTROL_REASON in registry[row["matched_id"]].get("reasons", []) for row in matches)}
    return {"construction_counts": construction, "candidate_counts": candidate,
            "matches": matches, "unmatched_prediction_ids": unmatched,
            "predicted_instances": [{"id": identifier, "side": registry[identifier]["side"],
                                     "voxel_count": predicted_sizes[identifier], "components": components[identifier]}
                                    for identifier in predicted_ids],
            **checks, "passed": all(checks.values())}


def visibility(case: ImagingCase) -> dict:
    """Independent input diagnostic; it does not change any scoring requirement."""
    free, count = ndi.label(case.fixture.hu < 200)
    core_components = {str(int(identifier)): [int(value) for value in np.unique(free[case.cores == identifier])]
                       for identifier in np.unique(case.cores) if identifier}
    separate = all(len(values) == 1 and values[0] != 0 for values in core_components.values())
    separate = separate and len({values[0] for values in core_components.values()}) == 2
    x = case.fixture.affine[0, 3] + np.arange(SHAPE[0]) * SPACING_MM
    profile = case.fixture.hu[:, SHAPE[1] // 2, SHAPE[2] // 2]
    return {"nonbone_threshold_hu": 200.0, "nonbone_components": int(count),
            "core_nonbone_component_ids": core_components,
            "wall_visible": bool(separate) if case.wall_mm else None,
            "central_profile_max_hu": float(profile[np.abs(x) <= 1.5].max())}


def protocol(cases: list[ImagingCase]) -> dict:
    from dataclasses import asdict
    from importlib.metadata import version
    import hashlib
    import subprocess
    from ethmoid_pipeline.counting.engine import CountingConfig, METHOD_VERSION
    from synthetic_counting_experiment import REPO

    sources = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py")) + [
        REPO / "scripts/synthetic_counting_experiment.py", Path(__file__).resolve()]
    return {
        "experiment": "synthetic-thin-wall-blur-v2-1",
        "scope": "supplied synthetic ROI, unchanged compartment stage, simplified image formation",
        "clinical_accuracy": "NOT_MEASURED", "method_version": METHOD_VERSION,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config": asdict(CountingConfig()), "fixtures": [record(case) for case in cases],
        "runtime": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
        "python": sys.version,
        "source_hashes": {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "rules_sha256": hashlib.sha256((REPO / "docs/SYNTHETIC_BLUR_EXPERIMENT.md").read_bytes()).hexdigest(),
    }


def freeze_plan() -> Path:
    import json
    from synthetic_counting_experiment import REPO

    plan = REPO / ".local/frozen-blur-plan.json"
    plan.parent.mkdir(exist_ok=True)
    snapshot = protocol(fixtures())
    with plan.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, indent=2, sort_keys=True))
    return plan


def measure(case: ImagingCase) -> tuple[dict, np.ndarray, np.ndarray]:
    from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, compartment_stage
    from synthetic_counting_experiment import array_hash, construction_counts

    f = case.fixture
    before = record(case)
    cavities = np.zeros_like(f.labels)
    air = np.zeros_like(f.labels)
    cells, warnings, failure = [], [], None
    scored = score(case, cavities, cells)
    try:
        cavities, air, cells, warnings = compartment_stage(f.hu, f.roi, f.affine, CountingConfig())
        assert set(np.unique(cavities)) - {0} == {cell["id"] for cell in cells}
        assert np.array_equal(air, np.where(f.hu < -500, cavities, 0))
        assert not np.any((cavities > 0) & ~f.roi)
        for cell in cells:
            mask = cavities == cell["id"]
            assert cell["voxel_count"] == np.count_nonzero(mask)
            assert cell["air_voxel_count"] == np.count_nonzero(air == cell["id"])
            center = f.affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + f.affine[:3, 3]
            assert np.allclose(center, cell["centroid_ras_mm"], rtol=0, atol=1e-8)
            assert cell["uncertain"] == bool(cell["reasons"])
        assert before == record(case)
        scored = score(case, cavities, cells)
    except CountingError as error:
        failure = error.code
    except Exception as error:
        failure = "UNEXPECTED_" + type(error).__name__
    if failure:
        cavities, air = np.zeros_like(f.labels), np.zeros_like(f.labels)
        cells = []
    result = before | scored
    result.update({
        "construction_counts": construction_counts(f),
        "candidate_counts": {side: sum(cell["side"] == side for cell in cells) for side in ("left", "right")},
        "cells": cells, "warnings": warnings, "failure_code": failure,
        "input_visibility": visibility(case), "inputs_preserved": before == record(case),
        "detected_hashes": {"cavities": array_hash(cavities), "air": array_hash(air)},
    })
    result["candidate_counts"]["total"] = len(cells)
    if failure:
        result["candidate_counts"] = {side: None for side in ("left", "right", "total")}
    result["passed"] = bool(result["passed"] and result["inputs_preserved"] and not failure)
    return result, cavities, air


def run(destination: Path) -> dict:
    import json
    from PIL import Image, ImageDraw
    from synthetic_counting_experiment import REPO, draw_case, font

    destination = destination.expanduser().resolve()
    if destination.is_relative_to(REPO):
        raise ValueError("Synthetic binary artifacts must stay outside the source repository.")
    if destination.exists():
        raise FileExistsError(destination)
    cases = fixtures()
    snapshot = protocol(cases)
    plan = json.loads((REPO / ".local/frozen-blur-plan.json").read_text(encoding="utf-8"))
    if plan != snapshot:
        raise ValueError("The inputs, rules, configuration or source differ from the premeasurement frozen plan.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "protocol.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    results, pages = [], []
    for case in cases:
        f = case.fixture
        case_dir = destination / f.name
        case_dir.mkdir()
        np.savez_compressed(case_dir / "input.npz", hu=f.hu, labels=f.labels, roi=f.roi, affine=f.affine,
                            highres_labels=case.highres_labels, cores=case.cores)
        result, cavities, air = measure(case)
        np.savez_compressed(case_dir / "detected.npz", cavities=cavities, air=air)
        (case_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        page = draw_case(f, result, cavities, case_dir / "comparison.png")
        draw = ImageDraw.Draw(page)
        draw.rectangle((20, 12, 1300, 50), fill="#f5f5f3")
        draw.text((34, 20), "SYNTHETIC THIN-WALL AND BLUR EXPERIMENT", font=font(24, True), fill="#404040")
        draw.rectangle((20, 208, 525, 244), fill="#f5f5f3")
        draw.text((34, 214), "Input: blurred / averaged HU", font=font(23, True), fill="#171717")
        draw.rectangle((20, 858, 1530, 913), fill="#f5f5f3")
        wall_state = "visible" if result["input_visibility"]["wall_visible"] else "absent at 200 HU"
        if case.wall_mm == 0:
            wall_state = "no divider constructed"
        recovery = "met" if result["passed"] else "NOT MET"
        draw.text((34, 863), f"Physical recovery: {recovery}. Input divider: {wall_state}.", font=font(23, True), fill="#171717")
        draw.text((34, 891), f"Worst matched Dice: {min(row['dice'] for row in result['matches']):.3f}; interior-core coverage: {min(row['core_coverage'] for row in result['matches']):.1%}. Missing matches score zero.", font=font(20), fill="#404040")
        page.save(case_dir / "comparison.png")
        pages.append(page)
        results.append(result)
    summary = {
        "fixture_count": len(results), "sealed_cases": 18, "control_cases": 6,
        "sealed_counts_correct": sum(row["wall_mm"] > 0 and row["exact_count_pass"] for row in results),
        "sealed_recovered": sum(row["wall_mm"] > 0 and row["passed"] for row in results),
        "controls_recovered": sum(row["wall_mm"] == 0 and row["passed"] for row in results),
        "physical_recovery_failures": [row["name"] for row in results if not row["passed"] and not (row["failure_code"] or "").startswith("UNEXPECTED_") and row["inputs_preserved"]],
        "execution_errors": [row["name"] for row in results if (row["failure_code"] or "").startswith("UNEXPECTED_") or not row["inputs_preserved"]],
        "clinical_accuracy": "NOT_MEASURED", "results": results,
    }
    summary["sealed_by_input_visibility"] = [
        {"wall_visible": visible, "cases": sum(row["wall_mm"] > 0 and row["input_visibility"]["wall_visible"] == visible for row in results),
         "recovered": sum(row["wall_mm"] > 0 and row["input_visibility"]["wall_visible"] == visible and row["passed"] for row in results)}
        for visible in (True, False)]
    (destination / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Synthetic thin-wall and blur results", "",
             "Two physical pockets were drawn independently, then blurred and averaged into coarser voxels. "
             "The six absent-wall controls each contain one space. The detector and its default settings were unchanged.", "",
             f"Sealed-pair recovery: {summary['sealed_recovered']}/18. Single-space controls: {summary['controls_recovered']}/6. "
             f"Unexpected execution errors: {len(summary['execution_errors'])}.", "",
             f"Sealed-pair physical totals correct: {summary['sealed_counts_correct']}/18. "
             f"Recovery grouped by threshold-visible input divider: {summary['sealed_by_input_visibility']}.", "",
             "Recovery requires the physical count, one-to-one spatial matching, Dice >= 0.90, core coverage >= 95%, "
             "connected predictions and the predeclared side/flag checks. A correct total alone does not pass.", "",
             "| Case | Wall mm | Blur sigma mm | Phase mm | Physical count | Detected | Divider visible at 200 HU | Worst Dice | Worst core coverage | Recovery |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| [{row['name']}]({row['name']}/comparison.png) | {row['wall_mm']} | {row['sigma_mm']} | "
                     f"{row['phase_mm']} | {row['construction_counts']['total']} | {row['candidate_counts']['total']} | "
                     f"{row['input_visibility']['wall_visible']} | {min(match['dice'] for match in row['matches']):.4f} | "
                     f"{min(match['core_coverage'] for match in row['matches']):.1%} | "
                     f"{'met' if row['passed'] else 'NOT MET'} |")
    lines += ["", "Each result.json retains per-compartment Dice, core coverage, matched IDs, extra predictions, "
              "connected components, physical sides, engine reasons, input visibility and hashes, including failed recovery.", "",
              "The threshold-visible wall diagnostic asks whether the imaged input still separates the two interior cores "
              "at 200 HU. It never changes physical truth or scoring. Loss of that threshold barrier can make recovery "
              "difficult; it does not prove loss of all image information or excuse a failed physical recovery.", "",
              "These 24 cases test one rectangular phantom family and a simplified Gaussian/voxel-averaging model. "
              "A second process checks repeatability; it does not add 24 independent observations. Patient accuracy, "
              "scanner calibration, noise and anatomical localization were not tested. No settings were tuned."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cover = Image.new("RGB", (1560, 1240), "#f5f5f3")
    draw = ImageDraw.Draw(cover)
    draw.text((44, 30), "Thin walls, blur and counting", font=font(40, True), fill="#171717")
    draw.text((44, 95), f"Sealed pairs recovered: {summary['sealed_recovered']}/18   Single-space controls: {summary['controls_recovered']}/6", font=font(28, True), fill="#171717")
    draw.text((44, 142), "Unchanged detector. Known physical objects. Supplied synthetic ROI. Clinical accuracy unmeasured.", font=font(22), fill="#404040")
    by_visibility = summary["sealed_by_input_visibility"]
    draw.text((44, 174), f"Input divider visible at 200 HU: {by_visibility[0]['recovered']}/{by_visibility[0]['cases']} recovered; divider absent: {by_visibility[1]['recovered']}/{by_visibility[1]['cases']}.", font=font(22), fill="#404040")
    draw.text((44, 205), "Wall mm    Blur sigma mm    Phase mm     Detected / physical      Recovery", font=font(25, True), fill="#171717")
    for i, row in enumerate(results):
        text = f"{row['wall_mm']:<10g}  {row['sigma_mm']:<15g}  {row['phase_mm']:<12g}  {row['candidate_counts']['total']} / {row['construction_counts']['total']}           {'met' if row['passed'] else 'NOT MET'}"
        draw.text((44, 257 + i * 34), text, font=font(24), fill="#24613c" if row["passed"] else "#a82828")
    draw.text((44, 1100), "Fine physical construction -> Gaussian blur -> voxel averaging -> unchanged detector.", font=font(24), fill="#404040")
    draw.text((44, 1144), "Numbers on comparison pages identify full 3D objects. All failed cases are retained.", font=font(23), fill="#404040")
    cover.save(destination / "summary.png")
    cover.save(destination / "report.pdf", save_all=True, append_images=pages, resolution=140)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze-plan", action="store_true", help="Save a fresh premeasurement protocol without detector calls.")
    choice.add_argument("--output", type=Path, help="Fresh artifact directory outside the source repository.")
    args = parser.parse_args()
    if args.freeze_plan:
        print(freeze_plan())
    else:
        outcome = run(args.output)
        sys.exit(1 if outcome["execution_errors"] else 0 if not outcome["physical_recovery_failures"] else 2)

