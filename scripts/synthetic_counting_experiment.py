"""Fixed, synthetic-only compartment experiment; never loads a patient scan."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, METHOD_VERSION, compartment_stage


@dataclass
class Fixture:
    name: str
    description: str
    hu: np.ndarray
    labels: np.ndarray
    roi: np.ndarray
    affine: np.ndarray
    category: str = "count"
    components: int = 1
    required_reasons: dict[int, str] = field(default_factory=dict)
    required_any: tuple[str, ...] = ()
    expected_error: str | None = None


def fixtures() -> list[Fixture]:
    shape = (64, 48, 40)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = -(np.asarray(shape) - 1) * 0.25

    def make(name, description, boxes, components=1):
        labels = np.zeros(shape, dtype=np.uint16)
        for index, bounds in enumerate(boxes, start=1):
            labels[tuple(slice(*pair) for pair in bounds)] = index
        hu = np.where(labels, -900, 800).astype(np.float32)
        return Fixture(name, description, hu, labels, np.ones(shape, bool), affine.copy(), components=components)

    def copy(base, name, description, **changes):
        arrays = {key: getattr(base, key).copy() for key in ("hu", "labels", "roi", "affine")}
        return replace(base, name=name, description=description, **(arrays | changes))

    single = make("single_clear", "One sealed air-filled pocket.", [((8, 28), (12, 36), (10, 30))])
    pair = make("two_sealed", "Two sealed pockets separated by a wide bone band.",
                [((8, 24), (12, 36), (10, 30)), ((40, 56), (12, 36), (10, 30))], 2)
    four = make("four_sealed", "Four sealed pockets; two on each physical side.",
                [(x, y, (8, 32)) for x in ((8, 24), (40, 56)) for y in ((6, 20), (28, 42))], 4)
    connected = make("communicating_no_wall", "Two overlapping lobes without an internal wall form one pocket.", [])
    grid = np.indices(shape)
    lobes = sum((grid[i] - c) ** 2 for i, c in enumerate((20, 24, 20))) < 13 ** 2
    lobes |= sum((grid[i] - c) ** 2 for i, c in enumerate((40, 24, 20))) < 13 ** 2
    connected.labels[lobes], connected.hu[lobes] = 1, -900
    connected.required_reasons = {1: "crosses_midline_centroid_laterality_requires_review"}
    thin = make("one_voxel_wall", "Two pockets separated by a continuous 0.5 mm wall (one voxel).",
                [((8, 32), (12, 36), (10, 30)), ((33, 56), (12, 36), (10, 30))], 2)
    thick = make("two_voxel_wall", "Two pockets separated by a continuous 1.0 mm wall (two voxels).",
                 [((8, 31), (12, 36), (10, 30)), ((33, 56), (12, 36), (10, 30))], 2)
    partial = copy(single, "partial_opacity", "The same pocket is half air and half soft tissue.",
                   required_reasons={1: "partly_opacified_candidate"})
    partial.hu[8:18, 12:36, 10:30] = 35
    full = copy(single, "full_opacity", "The same sealed pocket is entirely soft tissue, with no air.",
                required_reasons={1: "fully_opacified_candidate_requires_review"})
    full.hu[full.labels > 0] = 35
    mixed = copy(pair, "mixed_opacity", "One clear and one fully filled pocket; a total alone is insufficient.",
                 required_reasons={2: "fully_opacified_candidate_requires_review"})
    mixed.hu[mixed.labels == 2] = 35
    ambiguity_reasons = ("incomplete_septum_inferred_from_bone_plate",
                         "possible_internal_septa_unresolved_merge_requires_review")
    small = copy(thin, "small_wall_opening", "A 1.5 mm opening through a thin wall: one communicating space or two incompletely separated pockets.",
                 category="ambiguous", components=1, required_any=ambiguity_reasons)
    wide = copy(thin, "wide_wall_opening", "A 4.5 mm opening through the same wall; the interpretation remains unsettled.",
                category="ambiguous", components=1, required_any=ambiguity_reasons)
    for case, aperture in ((small, (slice(22, 25), slice(18, 21))), (wide, (slice(19, 28), slice(16, 25)))):
        location = (32, *aperture)
        case.hu[location], case.labels[location] = -900, 1
    diagonal = make("diagonal_pockets", "Two spherical pockets offset in all three axes.", [], 2)
    for index, center in enumerate(((16, 16, 14), (48, 32, 26)), start=1):
        sphere = sum((grid[i] - c) ** 2 for i, c in enumerate(center)) < 7 ** 2
        diagonal.labels[sphere], diagonal.hu[sphere] = index, -900
    anisotropic = copy(pair, "anisotropic_spacing", "Two pockets sampled at 0.4 x 0.7 x 0.9 mm; no voxel resampling.")
    anisotropic.affine = np.diag([0.4, 0.7, 0.9, 1.0])
    anisotropic.affine[:3, 3] = -(np.asarray(shape) - 1) * np.array([0.4, 0.7, 0.9]) / 2
    transform = np.eye(4)
    transform[0, 0], transform[0, 3] = -1, shape[0] - 1
    flip = copy(pair, "axis_flip", "The same physical construction with the first voxel axis reversed.",
                hu=pair.hu[::-1].copy(), labels=pair.labels[::-1].copy(), roi=pair.roi[::-1].copy(), affine=pair.affine @ transform)
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3)[:, [2, 0, 1]]
    permutation = copy(pair, "axis_permutation", "The same physical construction with voxel axes reordered (Z, X, Y).",
                       hu=pair.hu.transpose(2, 0, 1).copy(), labels=pair.labels.transpose(2, 0, 1).copy(),
                       roi=pair.roi.transpose(2, 0, 1).copy(), affine=pair.affine @ transform)
    oblique = copy(pair, "oblique_grid", "The voxelized construction is rigidly rotated 15 degrees in physical space; no resampling.")
    angle = np.deg2rad(15)
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    oblique.affine = rotation @ pair.affine
    unbounded = make("unbounded_soft_tissue", "Soft tissue without any bone enclosure must not become a cell.", [])
    unbounded.hu[:] = 35
    unbounded.category, unbounded.expected_error = "reject", "NO_SUPPORTED_CELLS"
    pore = make("single_voxel_pore", "A one-voxel-wide soft-tissue channel has no resolved 3D interior.", [])
    pore.hu[8:56, 24, 20] = 35
    pore.category, pore.expected_error = "reject", "NO_SUPPORTED_CELLS"
    cropped = copy(single, "cropped_enclosure", "The supplied ROI cuts through the pocket: withhold or flag the artificial boundary.",
                   category="boundary", required_any=("touches_inferred_anatomical_boundary",))
    cropped.roi[18:] = False
    sheared = copy(pair, "invalid_sheared_affine", "An invalid sheared grid must be rejected before counting.",
                   category="reject", expected_error="INVALID_GEOMETRY")
    sheared.affine[0, 1] = 0.2
    cases = [single, pair, four, connected, thin, thick, partial, full, mixed, small, wide,
             diagonal, anisotropic, flip, permutation, oblique, unbounded, pore, cropped, sheared]
    for case in cases:
        check_fixture(case)
        for array in (case.hu, case.labels, case.roi, case.affine):
            array.setflags(write=False)
    return cases


def check_fixture(case: Fixture) -> None:
    assert case.hu.shape == case.labels.shape == case.roi.shape
    assert case.hu.dtype == np.float32 and case.labels.dtype == np.uint16 and case.roi.dtype == bool
    assert np.isfinite(case.hu).all() and set(np.unique(case.hu)) <= {-900, 35, 800}
    assert not np.any((case.labels > 0) & (case.hu >= 200))
    assert ndi.label(case.hu < 200)[1] == case.components
    for label in set(np.unique(case.labels)) - {0}:
        assert ndi.label(case.labels == label)[1] == 1
    assert case.category in {"count", "ambiguous", "boundary", "reject"}


def array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}".encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def construction_counts(case: Fixture) -> dict[str, int]:
    counts = {"left": 0, "right": 0, "total": 0}
    for label in sorted(set(np.unique(case.labels)) - {0}):
        center = np.argwhere(case.labels == label).mean(axis=0)
        x = float(case.affine[0, :3] @ center + case.affine[0, 3])
        counts["left" if x <= 0 else "right"] += 1
        counts["total"] += 1
    return counts


def fixture_record(case: Fixture) -> dict:
    return {"name": case.name, "description": case.description, "category": case.category,
            "shape": list(case.hu.shape), "affine": case.affine.tolist(),
            "construction_counts": construction_counts(case) if case.category == "count" else None,
            "possible_totals": [1, 2] if case.category == "ambiguous" else None,
            "nonbone_components": case.components, "required_reasons": case.required_reasons,
            "required_any": list(case.required_any), "expected_error": case.expected_error,
            "hashes": {name: array_hash(getattr(case, name)) for name in ("hu", "labels", "roi", "affine")}}


def exact_correspondence(truth: np.ndarray, detected: np.ndarray) -> dict[int, int] | None:
    """One-to-one full voxel equality; IDs may differ, compensating errors may not."""
    truth_ids = sorted(set(np.unique(truth)) - {0})
    detected_ids = set(np.unique(detected)) - {0}
    if len(truth_ids) != len(detected_ids) or not np.array_equal(truth > 0, detected > 0):
        return None
    mapping = {}
    for truth_id in truth_ids:
        assigned = np.unique(detected[truth == truth_id])
        if len(assigned) != 1 or assigned[0] == 0:
            return None
        detected_id = int(assigned[0])
        if not np.array_equal(truth == truth_id, detected == detected_id):
            return None
        mapping[int(truth_id)] = detected_id
    return mapping


def measure(case: Fixture) -> tuple[dict, np.ndarray, np.ndarray]:
    record = fixture_record(case)
    result = record | {"candidate_counts": {"left": None, "right": None, "total": None},
                       "cells": [], "warnings": [], "failure_code": None,
                       "exact_count_pass": None, "spatial_pass": None,
                       "flags_pass": False, "passed": False}
    cavities = np.zeros(case.hu.shape, dtype=np.uint16)
    air = np.zeros_like(cavities)
    try:
        cavities, air, cells, warnings = compartment_stage(case.hu, case.roi, case.affine, CountingConfig())
        result["cells"], result["warnings"] = cells, warnings
        result["candidate_counts"] = {side: sum(cell["side"] == side for cell in cells) for side in ("left", "right")}
        result["candidate_counts"]["total"] = len(cells)
        ids = set(np.unique(cavities)) - {0}
        assert ids == {cell["id"] for cell in cells}
        assert np.array_equal(air, np.where(case.hu < -500, cavities, 0))
        assert not np.any((cavities > 0) & ~case.roi)
        for cell in cells:
            assert cell["voxel_count"] == np.count_nonzero(cavities == cell["id"])
            assert cell["air_voxel_count"] == np.count_nonzero(air == cell["id"])
            assert cell["uncertain"] == bool(cell["reasons"])
        mapping = exact_correspondence(case.labels, cavities)
        registry = {cell["id"]: cell for cell in cells}
        if case.category == "count":
            result["exact_count_pass"] = result["candidate_counts"] == record["construction_counts"]
            result["spatial_pass"] = mapping is not None
            flags = mapping is not None and all(
                reason in registry[mapping[label]]["reasons"] for label, reason in case.required_reasons.items())
            # A correct side total can hide two swapped assignments.
            sides = mapping is not None and all(
                registry[mapping[int(label)]]["side"] == (
                    "left" if float(case.affine[0, :3] @ np.argwhere(case.labels == label).mean(axis=0) + case.affine[0, 3]) <= 0 else "right"
                ) for label in set(np.unique(case.labels)) - {0})
            result["flags_pass"] = bool(flags)
            result["side_assignment_pass"] = bool(sides)
            result["passed"] = bool(result["exact_count_pass"] and result["spatial_pass"] and flags and sides)
        elif case.category in {"ambiguous", "boundary"}:
            result["flags_pass"] = any(cell["uncertain"] and set(cell["reasons"]) & set(case.required_any) for cell in cells)
            result["passed"] = result["flags_pass"]
    except CountingError as error:
        result["failure_code"] = error.code
        result["passed"] = (case.category == "reject" and error.code == case.expected_error) or (
            case.category in {"ambiguous", "boundary"} and error.code == "NO_SUPPORTED_CELLS")
    except Exception as error:
        result["failure_code"] = "UNEXPECTED_" + type(error).__name__
    if result["failure_code"]:
        result["candidate_counts"] = {"left": None, "right": None, "total": None}
        result["cells"] = []
        cavities, air = np.zeros_like(case.labels), np.zeros_like(case.labels)
        if case.category == "count":
            result["exact_count_pass"], result["spatial_pass"] = False, False
    result["inputs_preserved"] = record["hashes"] == fixture_record(case)["hashes"]
    result["passed"] = bool(result["passed"] and result["inputs_preserved"])
    result["detected_hashes"] = {"cavities": array_hash(cavities), "air": array_hash(air)}
    return result, cavities, air


COLORS = np.array([[0, 0, 0], [82, 189, 235], [255, 174, 73], [117, 204, 128], [218, 134, 224]], dtype=np.uint8)


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_case(case: Fixture, result: dict, detected: np.ndarray, destination: Path) -> Image.Image:
    page = Image.new("RGB", (1560, 1060), "#f5f5f3")
    draw = ImageDraw.Draw(page)
    draw.text((34, 20), "SYNTHETIC COUNTING EXPERIMENT", font=font(24, True), fill="#404040")
    draw.text((34, 56), case.name.replace("_", " ").capitalize(), font=font(36, True), fill="#171717")
    # Fixed-width wrapping prevents descriptions/reasons from leaving the page.
    import textwrap
    for line, text in enumerate(textwrap.wrap(case.description, width=115)):
        draw.text((34, 104 + line * 26), text, font=font(22), fill="#404040")
    construction = result["construction_counts"]
    expectation = str(construction["total"]) if construction else (
        "1 or 2 (unsettled)" if case.category == "ambiguous" else "rejection/flag control")
    count = result["candidate_counts"]["total"]
    draw.text((34, 170), f"Construction: {expectation}    Detected candidates: {count if count is not None else 'unavailable'}", font=font(25, True), fill="#171717")
    spacing = np.linalg.norm(case.affine[:3, :3], axis=0)
    for column, title in enumerate(("Input: bone / air / tissue", "Construction outlines", "Detected outlines")):
        draw.text((34 + column * 510, 214), title, font=font(23, True), fill="#171717")
    for row, axis in enumerate((2, 1)):
        target = case.labels if case.labels.any() else (case.hu < 200).astype(np.uint16)
        plane_index = max(range(target.shape[axis]), key=lambda i: (
            len(set(np.unique(np.take(target, i, axis=axis))) - {0}),
            np.count_nonzero(np.take(target, i, axis=axis))))
        ct = np.flipud(np.take(case.hu, plane_index, axis=axis).T)
        remaining = [i for i in range(3) if i != axis]
        extent = np.array([ct.shape[1] * spacing[remaining[0]], ct.shape[0] * spacing[remaining[1]]])
        scale = min(475 / extent[0], 265 / extent[1])
        size = tuple(np.maximum(1, np.rint(extent * scale)).astype(int))
        for column, labels in enumerate((None, case.labels, detected)):
            intensity = np.clip((ct + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
            rgb = np.repeat(intensity[:, :, None], 3, axis=2)
            plane = None if labels is None else np.flipud(np.take(labels, plane_index, axis=axis).T)
            if plane is not None:
                for label in set(np.unique(plane)) - {0}:
                    mask = plane == label
                    rgb[mask] = (0.4 * rgb[mask] + 0.6 * COLORS[1 + (int(label) - 1) % 4]).astype(np.uint8)
            panel = Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)
            x, y = 34 + column * 510 + (475 - size[0]) // 2, 248 + row * 312
            page.paste(panel, (x, y))
            if plane is not None:
                for label in set(np.unique(plane)) - {0}:
                    center = np.argwhere(plane == label).mean(axis=0)
                    position = (x + (center[1] + 0.5) / plane.shape[1] * size[0],
                                y + (center[0] + 0.5) / plane.shape[0] * size[1])
                    draw.text(position, str(label), anchor="mm", font=font(25, True), fill="white", stroke_width=2, stroke_fill="black")
            draw.text((34 + column * 510, y + 269), f"Voxel axis {axis}, plane {plane_index}", font=font(18), fill="#555555")
    passed = "Requirement met" if result["passed"] else "REQUIREMENT NOT MET"
    draw.text((34, 870), passed, font=font(25, True), fill="#24613c" if result["passed"] else "#a82828")
    reasons = sorted({reason for cell in result["cells"] for reason in cell["reasons"]})
    detail = "Failure: " + result["failure_code"] if result["failure_code"] else "Engine reasons: " + (", ".join(reasons) or "none")
    for line, text in enumerate(textwrap.wrap(detail, width=110)[:4]):
        draw.text((34, 914 + line * 23), text, font=font(19), fill="#404040")
    draw.text((34, 1025), "Supplied synthetic ROI; two selected voxel views. Numbers identify 3D labels. No anatomical localization or clinical accuracy test.", font=font(18), fill="#404040")
    page.save(destination)
    return page


def run(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    if destination.is_relative_to(REPO):
        raise ValueError("Synthetic binary artifacts must stay outside the source repository.")
    destination.mkdir(parents=True, exist_ok=False)
    cases = fixtures()
    source_files = sorted((REPO / "src/ethmoid_pipeline/counting").glob("*.py")) + [Path(__file__).resolve()]
    protocol = {"experiment": "synthetic-counting-v2-1", "method_version": METHOD_VERSION,
                "scope": "supplied synthetic ROI; compartment stage only; no clinical accuracy",
                "matching": "one-to-one full voxel equality allowing ID renumbering",
                "config": asdict(CountingConfig()), "fixtures": [fixture_record(case) for case in cases],
                "runtime": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "Pillow")},
                "python": sys.version,
                "source_hashes": {str(path.relative_to(REPO)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
                "rules_sha256": hashlib.sha256((REPO / "docs/SYNTHETIC_COUNTING_EXPERIMENT.md").read_bytes()).hexdigest()}
    frozen = REPO / ".local/frozen-synthetic-plan.json"
    if frozen.exists():
        plan = json.loads(frozen.read_text(encoding="utf-8"))
        if plan["fixtures"] != json.loads(json.dumps(protocol["fixtures"])) or plan["config"] != protocol["config"] or plan["rules_sha256"] != protocol["rules_sha256"]:
            raise ValueError("The design differs from the premeasurement frozen plan.")
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    results, pages = [], []
    for case in cases:
        case_dir = destination / case.name
        case_dir.mkdir()
        np.savez_compressed(case_dir / "input.npz", hu=case.hu, labels=case.labels, roi=case.roi, affine=case.affine)
        result, cavities, air = measure(case)
        np.savez_compressed(case_dir / "detected.npz", cavities=cavities, air=air)
        (case_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        pages.append(draw_case(case, result, cavities, case_dir / "comparison.png"))
        results.append(result)
    summary = {"fixture_count": len(results),
               "exact_cases": sum(row["category"] == "count" for row in results),
               "exact_requirements_met": sum(row["category"] == "count" and row["passed"] for row in results),
               "control_cases": sum(row["category"] != "count" for row in results),
               "control_requirements_met": sum(row["category"] != "count" and row["passed"] for row in results),
               "failed_requirements": [row["name"] for row in results if not row["passed"]],
               "advancement_gate_passed": all(row["passed"] for row in results),
               "clinical_accuracy": "NOT_MEASURED", "results": results}
    (destination / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Synthetic counting experiment", "",
             "Generated volumes with supplied ROIs; unchanged compartment engine. No patient scans or anatomical localization.", "",
             f"Exact construction requirements: {summary['exact_requirements_met']}/{summary['exact_cases']}. "
             f"Control requirements: {summary['control_requirements_met']}/{summary['control_cases']}. "
             "Controls are not additional exact-count accuracy cases.", "",
             "| Fixture | Category | Construction total | Candidate total | Count | Exact masks | Flags | Requirement |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    display = lambda value: "—" if value is None else str(value)
    for row in results:
        expected = row["construction_counts"]["total"] if row["construction_counts"] else (
            "1 or 2; unsettled" if row["category"] == "ambiguous" else "withhold/flag")
        lines.append(f"| [{row['name']}]({row['name']}/comparison.png) | {row['category']} | {expected} | "
                     f"{display(row['candidate_counts']['total'])} | {display(row['exact_count_pass'])} | "
                     f"{display(row['spatial_pass'])} | {row['flags_pass']} | {'met' if row['passed'] else 'NOT MET'} |")
    lines += ["", "See each result.json for side counts, per-cell reasons, exceptions and hashes. "
              "The saved protocol precedes every detector call; all 20 rows are retained.", "",
              "The exact mask criterion is deliberately strict for constructed voxel objects. "
              "It is not a clinical overlap measurement. Filled candidates remain review flags. "
              "Open-wall interpretations remain unsettled even if an engine count equals an alternative.", "",
              "One process execution is recorded here; repeatability must be checked against a separate saved execution. "
              "No settings were tuned. Real image formation, localization and clinical accuracy remain untested."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cover = Image.new("RGB", (1560, 1060), "#f5f5f3")
    draw = ImageDraw.Draw(cover)
    draw.text((44, 30), "Synthetic counting experiment", font=font(40, True), fill="#171717")
    draw.text((44, 90), f"Exact recovery: {summary['exact_requirements_met']}/14    Controls: {summary['control_requirements_met']}/6", font=font(28, True), fill="#171717")
    draw.text((44, 132), "Frozen settings. Supplied synthetic ROI. Clinical accuracy is not measured.", font=font(23), fill="#404040")
    for index, row in enumerate(results):
        expected = str(row["construction_counts"]["total"]) if row["construction_counts"] else ("1 or 2" if row["category"] == "ambiguous" else "control")
        actual = display(row["candidate_counts"]["total"])
        draw.text((44, 200 + index * 35), f"{row['name']:<28}  construction {expected:<8}  detected {actual:<4}  {'met' if row['passed'] else 'NOT MET'}", font=font(22), fill="#171717")
    draw.text((44, 970), "Construction truth is independent of detector output. Correct totals with wrong masks fail.", font=font(23), fill="#404040")
    cover.save(destination / "summary.png")
    cover.save(destination / "report.pdf", save_all=True, append_images=pages, resolution=140)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh local directory outside the source repository.")
    args = parser.parse_args()
    outcome = run(args.output)
    # A completed experiment can fail its advancement gate; report that distinctly.
    sys.exit(0 if outcome["advancement_gate_passed"] else 2)
