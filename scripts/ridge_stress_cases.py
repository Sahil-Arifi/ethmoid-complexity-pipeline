"""Twelve analytic stress cases, independent of the existing 56-case generator.

The material twins deliberately have identical measured HU but different latent
counts. In the central 0.5 mm X voxel, a centered bone sheet occupies fraction
(HU + 900) / 1700, with 800 HU bone and -900 HU air. Its exact volume average
equals a full voxel of 35 or 100 HU content. This is an information-loss example,
not a scanner simulation. The other observations use analytic center sampling.
Reference labels are evaluated from physical predicates; the separately sampled
2x labels check topology and never generate the HU. No detector is called here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_blur_experiment import CONTROL_REASON, ImagingCase
from synthetic_counting_experiment import Fixture

SHAPE = (41, 31, 25)
SPACING_MM = np.array((0.5, 0.6, 0.8))
HALF_EXTENT_MM = np.array((7.25, 6.3, 6.8))
APERTURE_CENTER_Y_MM = 0.15
APERTURE_WIDTH_MM = 2.4
FRAGMENT_BOUNDS_MM = ((-0.25, 0.25), (-2.55, -0.45), (0.1, 2.3))


@dataclass(frozen=True)
class StressCase(ImagingCase):
    construction: str
    observed_sheet_hu: int | None
    expected_abstention: bool
    equivalence_group: str | None
    transform: str = "base"


def _coordinates(factor: int = 1) -> tuple[np.ndarray, ...]:
    edges = -np.asarray(SHAPE) * SPACING_MM / 2
    return tuple(edge + (np.arange(size * factor) + 0.5) * step / factor
                 for edge, size, step in zip(edges, SHAPE, SPACING_MM))


def _reference(coordinates, construction: str, wall_mm: float,
               inset_mm: float = 0) -> np.ndarray:
    x, y, z = (coordinates[0][:, None, None], coordinates[1][None, :, None],
               coordinates[2][None, None, :])
    inside = ((np.abs(x) < HALF_EXTENT_MM[0] - inset_mm)
              & (np.abs(y) < HALF_EXTENT_MM[1] - inset_mm)
              & (np.abs(z) < HALF_EXTENT_MM[2] - inset_mm))
    labels = np.zeros(inside.shape, dtype=np.uint16)
    if construction == "bone":
        labels[inside & (x < -wall_mm / 2 - inset_mm)] = 1
        labels[inside & (x > wall_mm / 2 + inset_mm)] = 2
        return labels
    bone = np.zeros(inside.shape, dtype=bool)
    if construction == "opening":
        aperture = ((np.abs(y - APERTURE_CENTER_Y_MM) < APERTURE_WIDTH_MM / 2 - inset_mm)
                    & (np.abs(z) < APERTURE_WIDTH_MM / 2 - inset_mm))
        bone = (np.abs(x) <= 0.25 + inset_mm) & ~aperture
    elif construction == "fragment":
        bone = np.ones(inside.shape, dtype=bool)
        for axis, (lower, upper) in zip((x, y, z), FRAGMENT_BOUNDS_MM):
            bone &= (axis >= lower - inset_mm) & (axis <= upper + inset_mm)
    labels[inside & ~bone] = 1
    return labels


def _base(construction: str, sheet_hu: int | None = None) -> StressCase:
    fraction = (sheet_hu + 900) / 1700 if sheet_hu is not None else 0.0
    wall_mm = fraction * SPACING_MM[0] if construction == "bone" else 0.0
    coordinates = _coordinates()
    labels = _reference(coordinates, construction, wall_mm)
    cores = _reference(coordinates, construction, wall_mm, inset_mm=1.0)
    highres = _reference(_coordinates(2), construction, wall_mm).astype(np.uint8)
    if sheet_hu is None:
        hu = np.where(labels > 0, -900, 800).astype(np.float32)
    else:
        # The common observation is constructed without either twin's truth labels.
        cavity = _reference(coordinates, "content", 0) > 0
        sheet = cavity & (np.abs(coordinates[0][:, None, None]) < SPACING_MM[0] / 2)
        hu = np.where(cavity, -900, 800).astype(np.float32)
        hu[sheet] = sheet_hu
    affine = np.diag([*SPACING_MM, 1.0])
    affine[:3, 3] = -(np.asarray(SHAPE) - 1) * SPACING_MM / 2
    name = f"stress_{construction}_{sheet_hu}" if sheet_hu is not None else f"stress_{construction}"
    count = 2 if construction == "bone" else 1
    fixture = Fixture(name, f"{count} analytic physical space(s); {construction}; no blur or noise.",
                      hu, labels, np.ones(SHAPE, dtype=bool), affine, components=count,
                      required_reasons={} if count == 2 else {1: CONTROL_REASON})
    group = f"material_{sheet_hu}_base" if sheet_hu is not None else None
    return StressCase(fixture, highres, cores, float(wall_mm), 0.0, 0.0,
                      construction, sheet_hu, sheet_hu is not None, group)


def _transformed(case: StressCase) -> StressCase:
    def transform(array):
        return array.transpose(2, 1, 0)[:, ::-1, :].copy()

    mapping = np.eye(4)
    mapping[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    mapping[1, 3] = case.fixture.hu.shape[1] - 1
    f = case.fixture
    fixture = replace(f, name=f.name + "_swap_xz_flip_y", hu=transform(f.hu),
                      labels=transform(f.labels), roi=transform(f.roi), affine=f.affine @ mapping)
    group = case.equivalence_group.replace("_base", "_swap_xz_flip_y") if case.equivalence_group else None
    return replace(case, fixture=fixture, highres_labels=transform(case.highres_labels),
                   cores=transform(case.cores), equivalence_group=group, transform="swap_xz_flip_y")


def cases() -> list[StressCase]:
    """Return six fresh constructions and their exact physical reindexings."""
    bases = [_base(kind, hu) for hu in (35, 100) for kind in ("bone", "content")]
    bases.extend((_base("opening"), _base("fragment")))
    result = bases + [_transformed(case) for case in bases]
    for case in result:
        for array in (case.fixture.hu, case.fixture.labels, case.fixture.roi,
                      case.fixture.affine, case.highres_labels, case.cores):
            array.setflags(write=False)
    return result


def metadata(case: StressCase) -> dict:
    """Return scoring-only construction facts; none is a detector input."""
    notes = {
        "bone": "Two spaces separated by a centered subvoxel bone sheet; central voxel center is bone.",
        "content": "One space containing a full central voxel sheet of soft content; content is not a wall.",
        "opening": "One communicating space through a physical 2.4 x 2.4 mm aperture centered at Y=0.15, Z=0 mm in a 0.5 mm wall.",
        "fragment": "One space around an isolated 0.5 x 2.1 x 2.2 mm bone fragment.",
    }
    hu = case.observed_sheet_hu
    return {"name": case.fixture.name, "expected_abstention": case.expected_abstention,
            "equivalence_group": case.equivalence_group, "transform": case.transform,
            "physical_truth": notes[case.construction], "physical_count": case.fixture.components,
            "observed_sheet_hu": hu, "bone_fraction_in_twin": (hu + 900) / 1700 if hu is not None else None,
            "wall_mm": case.wall_mm, "reference_core_inset_mm": 1.0,
            "observation_model": "exact central-voxel volume average or full-voxel content" if hu is not None else "analytic center sampling"}
