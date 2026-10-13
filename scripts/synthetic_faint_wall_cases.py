"""Predeclared physical challenge phantoms; never calls a counting detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_blur_experiment import (
    CONTROL_REASON, FACTOR, FINE_SPACING_MM, ImagingCase, SHAPE, SPACING_MM,
)
from synthetic_counting_experiment import Fixture, array_hash, construction_counts

PAIR_ROWS = ((0.375, 0.125, 0.125), (0.375, 0.375, 0.125),
             (0.75, 0.375, 0.375), (0.75, 0.625, 0.375))
CONTROL_ROWS = ((0.125, 0.125), (0.375, 0.375))
GEOMETRIES = ("off_center_flat", "oblique", "curved", "axis_permutation")
CONTROLS = ("no_divider", "perforated_divider", "attached_ridge", "isolated_fragment",
            "content_ramp", "gaussian_noise", "soft_sheet_35", "soft_sheet_100")


@dataclass(frozen=True)
class ChallengeCase(ImagingCase):
    construction: str
    noise_seed: int | None = None


def _coordinates(edges, spacing, shape):
    return tuple(edge + (np.arange(size) + 0.5) * spacing
                 for edge, size in zip(edges, shape))


def _reference(coordinates, construction, wall_mm, inset_mm=0):
    """Evaluate physical predicates independently at the requested grid centers."""
    x, y, z = (coordinates[0][:, None, None], coordinates[1][None, :, None],
               coordinates[2][None, None, :])
    inside = ((x >= -8 + inset_mm) & (x < 8 - inset_mm)
              & (y >= -6 + inset_mm) & (y < 6 - inset_mm)
              & (z >= -4 + inset_mm) & (z < 4 - inset_mm))
    labels = np.zeros(inside.shape, dtype=np.uint16)
    if wall_mm:
        if construction == "oblique":
            d, margin = x + 0.75 - 0.25 * y, inset_mm * np.sqrt(1 + 0.25 ** 2)
        elif construction == "curved":
            d = x - 0.5 - 0.04 * (y * y - 12)
            margin = inset_mm * np.sqrt(1 + 0.48 ** 2)
        else:
            d, margin = x - 1, inset_mm
        labels[inside & (d < -wall_mm / 2 - margin)] = 1
        labels[inside & (d >= wall_mm / 2 + margin)] = 2
    else:
        if construction == "perforated_divider":
            bone = ((x >= -0.375) & (x < 0.375)
                    & ~((np.abs(y) < 1.5) & (np.abs(z) < 1.5)))
        elif construction == "attached_ridge":
            bone = (x >= -0.375) & (x < 0.375) & (y < -2)
        elif construction == "isolated_fragment":
            bone = ((x >= -0.375) & (x < 0.375)
                    & (np.abs(y) < 2) & (np.abs(z) < 2))
        else:
            bone = np.zeros(inside.shape, dtype=bool)
        labels[inside & ~bone] = 1
        if inset_mm:
            # Expanded bounding boxes conservatively exclude the 1 mm bone neighborhood.
            for bounds in _bone_boxes(construction):
                near = np.ones(inside.shape, dtype=bool)
                for axis, (lower, upper) in zip((x, y, z), bounds):
                    near &= (axis >= lower - inset_mm) & (axis <= upper + inset_mm)
                labels[near] = 0
    return labels


def _bone_boxes(construction):
    x = (-0.375, 0.375)
    if construction == "perforated_divider":
        return ((x, (-6, -1.5), (-4, 4)), (x, (1.5, 6), (-4, 4)),
                (x, (-1.5, 1.5), (-4, -1.5)), (x, (-1.5, 1.5), (1.5, 4)))
    if construction == "attached_ridge":
        return ((x, (-6, -2), (-4, 4)),)
    if construction == "isolated_fragment":
        return ((x, (-2, 2), (-2, 2)),)
    return ()


def make_challenge_case(construction: str, row: int) -> ChallengeCase:
    if construction not in GEOMETRIES + CONTROLS or isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("Construction and row must belong to the fixed challenge design.")
    rows = PAIR_ROWS if construction in GEOMETRIES else CONTROL_ROWS
    if row not in range(len(rows)):
        raise ValueError("Construction and row must belong to the fixed challenge design.")
    wall, sigma, phase = rows[row] if construction in GEOMETRIES else (0, *rows[row])
    edges = np.array([-12 + phase, -10, -8], dtype=np.float64)
    centers = _coordinates(edges, SPACING_MM, SHAPE)
    fine_centers = _coordinates(edges, FINE_SPACING_MM, tuple(size * FACTOR for size in SHAPE))
    highres_labels = _reference(fine_centers, construction, wall).astype(np.uint8)
    labels = _reference(centers, construction, wall)
    cores = _reference(centers, construction, wall, inset_mm=1)
    fine_hu = np.where(highres_labels, -900.0, 800.0)
    fine_x = fine_centers[0][:, None, None]
    if construction == "content_ramp":
        content = -900 + 300 * (fine_x + 8) / 16
        fine_hu = np.where(highres_labels, content, fine_hu)
    elif construction in ("soft_sheet_35", "soft_sheet_100"):
        sheet = (highres_labels > 0) & (fine_x >= -0.375) & (fine_x < 0.375)
        fine_hu[sheet] = 35 if construction == "soft_sheet_35" else 100
    ndi.gaussian_filter(fine_hu, sigma=sigma / FINE_SPACING_MM, output=fine_hu,
                        mode="constant", cval=800.0, truncate=4.0)
    hu = fine_hu.reshape(SHAPE[0], FACTOR, SHAPE[1], FACTOR, SHAPE[2], FACTOR).mean(axis=(1, 3, 5))
    seed = 2026101301 + row if construction == "gaussian_noise" else None
    if seed is not None:
        hu += np.random.Generator(np.random.PCG64(seed)).normal(0, 75, size=SHAPE)
    hu = hu.astype(np.float32)
    affine = np.diag([SPACING_MM, SPACING_MM, SPACING_MM, 1.0])
    affine[:3, 3] = edges + SPACING_MM / 2
    if construction == "axis_permutation":
        hu, labels, cores, highres_labels = (np.swapaxes(array, 0, 1)
                                            for array in (hu, labels, cores, highres_labels))
        affine[:, [0, 1]] = affine[:, [1, 0]]
    expected = 2 if wall else 1
    fixture = Fixture(f"challenge_{construction}_row_{row + 1}",
                      f"{expected} physical space(s); {construction.replace('_', ' ')}; "
                      f"blur sigma {sigma:g} mm, X phase {phase:g} mm.",
                      hu, labels, np.ones(labels.shape, dtype=bool), affine, components=expected,
                      required_reasons={} if wall else {1: CONTROL_REASON})
    assert ndi.label(highres_labels > 0)[1] == expected
    assert np.isfinite(hu).all()
    for identifier in range(1, expected + 1):
        assert ndi.label(highres_labels == identifier)[1] == 1
        assert np.any(cores == identifier) and ndi.label(cores == identifier)[1] == 1
        assert np.all(labels[cores == identifier] == identifier)
    for array in (hu, labels, fixture.roi, affine, highres_labels, cores):
        array.setflags(write=False)
    return ChallengeCase(fixture, highres_labels, cores, float(wall), float(sigma), float(phase),
                         construction, seed)


def challenge_cases() -> list[ChallengeCase]:
    return ([make_challenge_case(name, row) for name in GEOMETRIES for row in range(4)]
            + [make_challenge_case(name, row) for name in CONTROLS for row in range(2)])


def challenge_record(case: ChallengeCase) -> dict:
    f = case.fixture
    volume = 1536 - case.wall_mm * 96 if case.wall_mm else {
        "perforated_divider": 1470.75, "attached_ridge": 1512, "isolated_fragment": 1524,
    }.get(case.construction, 1536)
    truth_ids = [int(value) for value in np.unique(f.labels) if value]
    return {"name": f.name, "description": f.description, "category": f.category,
            "case_kind": "sealed_pair" if case.wall_mm else "one_space_control",
            "group": "challenge", "construction": case.construction,
            "wall_mm": case.wall_mm, "material_sheet_width_mm": 0.75 if case.construction in
            ("perforated_divider", "attached_ridge", "isolated_fragment", "soft_sheet_35", "soft_sheet_100") else case.wall_mm,
            "wall_width_convention": "signed coordinate width; not normal thickness",
            "sigma_mm": case.sigma_mm, "phase_mm": case.phase_mm,
            "noise_seed": case.noise_seed, "noise_generator": "PCG64" if case.noise_seed else None,
            "noise_std_hu": 75 if case.noise_seed else 0,
            "shape": list(f.hu.shape), "affine": f.affine.tolist(), "spacing_mm": SPACING_MM,
            "highres_shape": list(case.highres_labels.shape), "highres_spacing_mm": FINE_SPACING_MM,
            "block_factor": FACTOR, "core_inset_mm": 1.0, "gaussian_truncate": 4.0,
            "exterior_hu": 800.0, "air_hu": -900.0,
            "content_hu": [35] if case.construction == "soft_sheet_35" else
            [100] if case.construction == "soft_sheet_100" else
            [-900, -600] if case.construction == "content_ramp" else [-900],
            "physical_cavity_volume_mm3": float(volume),
            "highres_cavity_volume_mm3": float(np.count_nonzero(case.highres_labels) * FINE_SPACING_MM ** 3),
            "highres_physical_components": int(ndi.label(case.highres_labels > 0)[1]),
            "construction_counts": construction_counts(f),
            "reference_label_voxels": {str(identifier): int(np.count_nonzero(f.labels == identifier)) for identifier in truth_ids},
            "reference_core_voxels": {str(identifier): int(np.count_nonzero(case.cores == identifier)) for identifier in truth_ids},
            "required_reasons": {str(identifier): reason for identifier, reason in f.required_reasons.items()},
            "hashes": {name: array_hash(array) for name, array in
                       [(name, getattr(f, name)) for name in ("hu", "labels", "roi", "affine")]
                       + [("highres_labels", case.highres_labels), ("cores", case.cores)]}}
