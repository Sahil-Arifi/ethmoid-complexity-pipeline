"""Fixed physical falsifiers for a synthetic native-passage evidence experiment.

Air (-900 HU) and bone (800 HU) are sampled on an ideal 0.125 mm grid,
optionally blurred, and volume-averaged into 0.5 mm voxels. Optional Gaussian
noise is added after averaging. This is an idealized model, not a calibrated
scanner simulation or clinical evidence. Reference labels and 1 mm inset cores
come directly from physical predicates, independently of observed HU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_counting_experiment import Fixture

SHAPE = (32, 28, 24)
SPACING_MM = 0.5
FINE_SPACING_MM = 0.125
FACTOR = 4
HALF_EXTENT_MM = np.array((5.0, 4.0, 3.0))
NOISE_SEED = 1729
CORE_INSET_MM = 1.0


@dataclass(frozen=True)
class NativePassageCase:
    fixture: Fixture
    highres_labels: np.ndarray
    cores: np.ndarray
    wall_mm: float
    aperture_mm: float
    sigma_mm: float
    phase_mm: float
    noise_hu: float
    transform: str = "base"


def _coordinates(phase_mm: float, factor: int = 1) -> tuple[np.ndarray, ...]:
    edges = -np.asarray(SHAPE) * SPACING_MM / 2
    edges[0] += phase_mm
    return tuple(edge + (np.arange(size * factor) + 0.5) * SPACING_MM / factor
                 for edge, size in zip(edges, SHAPE))


def _reference(coordinates, wall_mm: float, aperture_mm: float,
               inset_mm: float = 0.0) -> np.ndarray:
    x, y, z = (coordinates[0][:, None, None], coordinates[1][None, :, None],
               coordinates[2][None, None, :])
    inside = np.ones(tuple(len(axis) for axis in coordinates), dtype=bool)
    for axis, extent in zip((x, y, z), HALF_EXTENT_MM):
        inside &= (axis >= -extent + inset_mm) & (axis < extent - inset_mm)
    # Half-open bounds retain the two fine-grid samples in a 0.25 mm aperture.
    plate = (x >= -wall_mm / 2 - inset_mm) & (x < wall_mm / 2 + inset_mm)
    aperture = ((y >= -aperture_mm / 2 + inset_mm) & (y < aperture_mm / 2 - inset_mm)
                & (z >= -aperture_mm / 2 + inset_mm) & (z < aperture_mm / 2 - inset_mm))
    free = inside & (~plate | aperture)
    labels = np.zeros(inside.shape, dtype=np.uint16)
    if aperture_mm:
        labels[free] = 1
    else:
        labels[free & (x < 0)] = 1
        labels[free & (x >= 0)] = 2
    return labels


def _base(wall_mm: float, aperture_mm: float, sigma_mm: float,
          phase_mm: float = 0.0, noise_hu: float = 0.0) -> NativePassageCase:
    coordinates = _coordinates(phase_mm)
    labels = _reference(coordinates, wall_mm, aperture_mm)
    cores = _reference(coordinates, wall_mm, aperture_mm, CORE_INSET_MM)
    highres = _reference(_coordinates(phase_mm, FACTOR), wall_mm, aperture_mm).astype(np.uint8)
    fine_hu = np.where(highres > 0, -900.0, 800.0)
    if sigma_mm:
        ndi.gaussian_filter(fine_hu, sigma=sigma_mm / FINE_SPACING_MM, output=fine_hu,
                            mode="constant", cval=800.0, truncate=4.0)
    hu = fine_hu.reshape(SHAPE[0], FACTOR, SHAPE[1], FACTOR, SHAPE[2], FACTOR).mean(axis=(1, 3, 5))
    if noise_hu:
        hu += np.random.default_rng(NOISE_SEED).normal(0.0, noise_hu, SHAPE)
    hu = hu.astype(np.float32)
    affine = np.diag([SPACING_MM, SPACING_MM, SPACING_MM, 1.0])
    affine[:3, 3] = [axis[0] for axis in coordinates]
    count = 1 if aperture_mm else 2
    parameters = "_".join(f"{key}_{value:g}".replace(".", "p") for key, value in
                          (("wall", wall_mm), ("aperture", aperture_mm), ("sigma", sigma_mm),
                           ("phase", phase_mm), ("noise", noise_hu)))
    fixture = Fixture("native_" + parameters,
                      f"{count} physical space(s); wall {wall_mm:g} mm; square aperture {aperture_mm:g} mm; "
                      f"blur sigma {sigma_mm:g} mm; X phase {phase_mm:g} mm; noise SD {noise_hu:g} HU.",
                      hu, labels, np.ones(SHAPE, dtype=bool), affine, components=count)
    assert ndi.label(highres > 0)[1] == count
    assert np.all(labels[cores > 0] == cores[cores > 0])
    return NativePassageCase(fixture, highres, cores, wall_mm, aperture_mm,
                             sigma_mm, phase_mm, noise_hu)


def _transformed(case: NativePassageCase) -> NativePassageCase:
    def transform(array):
        return array.transpose(2, 1, 0)[:, ::-1, :].copy()

    mapping = np.eye(4)
    mapping[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    mapping[1, 3] = case.fixture.hu.shape[1] - 1
    f = case.fixture
    fixture = replace(f, name=f.name + "_swap_xz_flip_y", hu=transform(f.hu),
                      labels=transform(f.labels), roi=transform(f.roi), affine=f.affine @ mapping)
    return replace(case, fixture=fixture, highres_labels=transform(case.highres_labels),
                   cores=transform(case.cores), transform="swap_xz_flip_y")


def cases() -> list[NativePassageCase]:
    """Return the fixed 16 physical constructions and 16 exact reindexings."""
    bases = [_base(wall, 0.0, sigma) for wall in (0.125, 0.5, 1.0) for sigma in (0.0, 0.3)]
    bases += [_base(1.0, aperture, sigma) for aperture in (0.25, 0.5, 1.5) for sigma in (0.0, 0.3)]
    bases += [_base(wall, aperture, 0.3, phase_mm=0.125) for wall, aperture in ((0.5, 0.0), (1.0, 0.5))]
    bases += [_base(wall, aperture, 0.3, noise_hu=25.0) for wall, aperture in ((0.5, 0.0), (1.0, 0.5))]
    result = bases + [_transformed(case) for case in bases]
    for case in result:
        for array in (case.fixture.hu, case.fixture.labels, case.fixture.roi,
                      case.fixture.affine, case.highres_labels, case.cores):
            array.setflags(write=False)
    return result


def metadata(case: NativePassageCase) -> dict:
    """Scoring-only physical facts; never supply these to a detector."""
    return {"name": case.fixture.name, "expected_count": case.fixture.components,
            "physical_count": case.fixture.components,
            "case_kind": "communicating_opening" if case.aperture_mm else "sealed_pair",
            "wall_mm": case.wall_mm, "aperture_mm": case.aperture_mm,
            "sigma_mm": case.sigma_mm, "phase_mm": case.phase_mm,
            "noise_hu": case.noise_hu, "noise_seed": NOISE_SEED if case.noise_hu else None,
            "transform": case.transform, "native_spacing_mm": SPACING_MM,
            "fine_spacing_mm": FINE_SPACING_MM, "reference_core_inset_mm": CORE_INSET_MM,
            "observation_model": "Ideal fine-grid air/bone sampling, Gaussian blur, voxel averaging, then Gaussian noise; not clinical or scanner-calibrated.",
            "reference_model": "Half-open physical box, wall and aperture predicates, independent of observed HU.",
            "clinical_accuracy": "NOT_MEASURED"}
