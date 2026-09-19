"""Equation-defined engineering phantoms for observed-wall proposal research.

These are not patient images, sinus anatomy, expert annotations, or a clinical
validation set. The geometry is chosen before image sampling/augmentation. Its
outer-wall material supplies independent supervision; a candidate extractor,
graph decoder, or existing counting prediction never supplies the target.

Array axes are X, Y, Z in millimetres, with +Y away from the posterior seed.
Partial volume uses a local-plane voxel-footprint approximation. This is an
explicit imaging approximation, not a scanner reconstruction simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np


VERSION = "procedural-wall-phantoms-2"
GEOMETRY_FAMILIES = (
    "plane", "smooth_bowl", "tilted", "steep_ridge", "asymmetric",
    "saddle", "wavy", "offcenter_dome",
)
MAX_VOXELS = 2_000_000
BONE_HU = 200.0


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class WallGeometry:
    family: str
    index: int
    generation_seed: int
    radius_x_mm: float
    radius_z_mm: float
    wall_offset_mm: float
    amplitude_mm: float
    tilt_x: float
    tilt_z: float
    ridge_scale_mm: float
    wall_thickness_mm: float
    internal_offset_mm: float
    external_offset_mm: float
    distractor_thickness_mm: float
    distractor_phase: float
    extent_mm: tuple[float, float, float] = (32.0, 36.0, 28.0)

    @property
    def geometry_id(self) -> str:
        return "geometry-" + _digest({"version": VERSION, **asdict(self)})[:24]


@dataclass(frozen=True)
class SceneAugmentation:
    spacing_mm: tuple[float, float, float] = (0.8, 0.8, 0.8)
    opacity: str = "air"
    noise_std_hu: float = 0.0
    blur_sigma_mm: float = 0.0
    wall_state: str = "present"
    seed: int = 0


@dataclass(frozen=True)
class SyntheticWallScene:
    hu: np.ndarray
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    seed: np.ndarray
    analytic_seed: np.ndarray
    outer_wall: np.ndarray  # Equation-defined material footprint, before blur.
    wall_support: np.ndarray  # Observed leading faces causally supported after blur.
    target_surface_y_mm: np.ndarray
    target_leading_surface_y_mm: np.ndarray
    support_columns: np.ndarray
    observable: bool
    ambiguous: bool
    cropped: bool
    geometry_id: str
    family: str
    augmentation_id: str
    support_fraction: float
    wall_state: str
    soft_tissue_hu: float
    provenance: str = "procedural_equations_not_expert_or_patient_anatomy"


def make_geometry(index: int, family: str = "smooth_bowl", *, seed: int = 0) -> WallGeometry:
    """Choose a complete geometry, independently of image augmentations.

    True and distractor thicknesses are independent Uniform(1.2, 2.0) mm
    draws from the same distribution, never class-specific thickness ranges.
    """
    if family not in GEOMETRY_FAMILIES or type(index) is not int or index < 0 or type(seed) is not int:
        raise ValueError("A known geometry family and nonnegative integer index are required")
    rng = np.random.default_rng(int(_digest([VERSION, family, index, seed])[:16], 16))
    return WallGeometry(
        family=family, index=index, generation_seed=seed,
        radius_x_mm=float(rng.uniform(11.0, 13.0)), radius_z_mm=float(rng.uniform(9.0, 11.0)),
        wall_offset_mm=float(rng.uniform(3.5, 5.0)), amplitude_mm=float(rng.uniform(2.5, 4.5)),
        tilt_x=float(rng.uniform(-0.12, 0.12)), tilt_z=float(rng.uniform(-0.10, 0.10)),
        ridge_scale_mm=float(rng.uniform(1.8, 3.2)), wall_thickness_mm=float(rng.uniform(1.2, 2.0)),
        internal_offset_mm=float(rng.uniform(3.0, 4.5)), external_offset_mm=float(rng.uniform(3.5, 5.0)),
        distractor_thickness_mm=float(rng.uniform(1.2, 2.0)), distractor_phase=float(rng.uniform(-0.6, 0.6)),
    )


def split_geometry_families(geometries: Iterable[WallGeometry], *, seed: int = 0,
                            fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, tuple[WallGeometry, ...]]:
    """Split entire families before rendering; no geometry can cross splits.

Validation and test therefore exercise withheld equation families as well as
unseen geometries. Augmentations must be rendered from the assigned members.
"""
    values = tuple(geometries)
    if (len(fractions) != 3 or any(not math.isfinite(x) or x <= 0 for x in fractions)
            or not math.isclose(sum(fractions), 1.0) or type(seed) is not int):
        raise ValueError("Three positive split fractions summing to one are required")
    if any(not isinstance(g, WallGeometry) or g.family not in GEOMETRY_FAMILIES for g in values):
        raise ValueError("Splits require known complete geometry specifications")
    ids = [g.geometry_id for g in values]
    if len(ids) != len(set(ids)):
        raise ValueError("A complete geometry cannot be duplicated in a split release")
    families = sorted({g.family for g in values}, key=lambda f: _digest([VERSION, "split", seed, f]))
    if len(families) < 3:
        raise ValueError("At least three independent geometry families are required")
    ntrain = min(len(families) - 2, max(1, round(len(families) * fractions[0])))
    nvalidation = min(len(families) - ntrain - 1, max(1, round(len(families) * fractions[1])))
    groups = (families[:ntrain], families[ntrain:ntrain + nvalidation], families[ntrain + nvalidation:])
    return {name: tuple(sorted((g for g in values if g.family in members), key=lambda g: g.geometry_id))
            for name, members in zip(("train", "validation", "test"), groups)}


def _surface(g: WallGeometry, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Y surface and its two physical derivatives."""
    u, v = x / g.radius_x_mm, z / g.radius_z_mm
    a = g.amplitude_mm
    y = g.wall_offset_mm + g.tilt_x * x + g.tilt_z * z
    dx, dz = np.full_like(y, g.tilt_x), np.full_like(y, g.tilt_z)
    if g.family == "smooth_bowl":
        y = y + a * (0.6 * u**2 + 0.4 * v**2 - 0.35)
        dx, dz = dx + 1.2 * a * u / g.radius_x_mm, dz + 0.8 * a * v / g.radius_z_mm
    elif g.family == "tilted":
        y, dx = y + 0.45 * x, dx + 0.45
    elif g.family == "steep_ridge":
        t = np.tanh(x / g.ridge_scale_mm)
        y, dx = y + a * t, dx + a * (1 - t**2) / g.ridge_scale_mm
    elif g.family == "asymmetric":
        y = y + a * (0.45 * u**3 + 0.3 * u + 0.35 * v**2)
        dx, dz = dx + a * (1.35 * u**2 + 0.3) / g.radius_x_mm, dz + 0.7 * a * v / g.radius_z_mm
    elif g.family == "saddle":
        y = y + 0.5 * a * (u**2 - v**2)
        dx, dz = dx + a * u / g.radius_x_mm, dz - a * v / g.radius_z_mm
    elif g.family == "wavy":
        y = y + 0.7 * a * np.sin(1.5 * u) * np.cos(0.7 * v)
        dx = dx + 1.05 * a * np.cos(1.5 * u) * np.cos(0.7 * v) / g.radius_x_mm
        dz = dz - 0.49 * a * np.sin(1.5 * u) * np.sin(0.7 * v) / g.radius_z_mm
    elif g.family == "offcenter_dome":
        y = y + a * ((u - 0.3)**2 + 0.45 * (v + 0.2)**2 - 0.5)
        dx, dz = dx + 2 * a * (u - 0.3) / g.radius_x_mm, dz + 0.9 * a * (v + 0.2) / g.radius_z_mm
    elif g.family != "plane":
        raise ValueError("Unknown geometry family")
    return y, dx, dz


def _slab_fraction(distance: np.ndarray, thickness: float, footprint: np.ndarray | float) -> np.ndarray:
    """Fraction of a local-plane slab intersecting the voxel's projected width."""
    low = np.maximum(distance - footprint / 2, -thickness / 2)
    high = np.minimum(distance + footprint / 2, thickness / 2)
    return np.clip((high - low) / footprint, 0, 1).astype(np.float32)


def _wall_signal_support(base: np.ndarray, outer_signal: np.ndarray,
                         other_signal: np.ndarray) -> np.ndarray:
    """Attribute noiseless threshold evidence to the independently rendered wall.

    The point-spread function can move its signal beyond the material footprint.
    A positive blur tail alone is insufficient: the wall must dominate the other
    material contribution and its removal must remove threshold-level evidence.
    No sampled noise, candidate rank, fitted surface, or learned score enters.
    """
    without_wall = base + other_signal
    before_noise = base + outer_signal + other_signal
    return ((outer_signal > 0) & (before_noise >= BONE_HU)
            & (outer_signal >= other_signal) & (without_wall < BONE_HU))


def render_scene(geometry: WallGeometry, augmentation: SceneAugmentation | None = None,
                 *, max_voxels: int = MAX_VOXELS) -> SyntheticWallScene:
    """Sample an owned, bounded CT-like phantom without any learned model.

    Missing walls have no material. Averaged-away walls retain a subvoxel material
    definition but cannot receive support from noise alone. Cropped scenes expose
    only the acquired part. Ambiguous scenes contain two coherent rival sheets;
    their candidates are ignored for forced-choice supervision.

    Each augmentation draws one Uniform(-20, 80) HU soft-tissue value for both
    opaque contents and surrounding tissue. Their class cannot be inferred from
    an artificial fixed inside/outside intensity difference.

    Material occupancy remains separate from observed support. The latter uses
    the blurred, noiseless signal attributable to that material and cannot label
    a threshold crossing caused only by noise or an independently visible septum.
    """
    a = augmentation or SceneAugmentation()
    if not isinstance(geometry, WallGeometry) or not isinstance(a, SceneAugmentation):
        raise ValueError("A complete geometry and sampling specification are required")
    spacing = tuple(float(s) for s in a.spacing_mm)
    if (len(spacing) != 3 or any(not math.isfinite(s) or not 0 < s <= 3.5 for s in spacing)
            or type(max_voxels) is not int or not 0 < max_voxels <= MAX_VOXELS
            or a.opacity not in {"air", "partial", "full"}
            or a.wall_state not in {"present", "missing", "averaged_away", "cropped", "ambiguous"}
            or type(a.seed) is not int or not math.isfinite(a.noise_std_hu) or not 0 <= a.noise_std_hu <= 300
            or not math.isfinite(a.blur_sigma_mm) or not 0 <= a.blur_sigma_mm <= 3):
        raise ValueError("Invalid phantom sampling, contrast, condition, or resource limit")
    numeric = [value for key, value in asdict(geometry).items()
               if key not in {"family", "index", "generation_seed", "extent_mm"}]
    if (geometry.family not in GEOMETRY_FAMILIES or any(not math.isfinite(x) for x in numeric)
            or len(geometry.extent_mm) != 3 or any(not math.isfinite(x) or not 0 < x <= 64 for x in geometry.extent_mm)
            or min(geometry.radius_x_mm, geometry.radius_z_mm, geometry.ridge_scale_mm,
                   geometry.wall_thickness_mm, geometry.distractor_thickness_mm) <= 0):
        raise ValueError("Invalid analytic geometry")
    lower = (-geometry.extent_mm[0] / 2, -16.0, -geometry.extent_mm[2] / 2)
    upper = (geometry.extent_mm[0] / 2,
             geometry.wall_offset_mm if a.wall_state == "cropped" else lower[1] + geometry.extent_mm[1],
             geometry.extent_mm[2] / 2)
    dimensions = tuple((hi - lo) / step for lo, hi, step in zip(lower, upper, spacing))
    if any(not math.isfinite(n) for n in dimensions):
        raise ValueError("Synthetic scene exceeds its bounded voxel budget")
    shape = tuple(math.ceil(n) for n in dimensions)
    # No voxel arrays or coordinate grids may be allocated before this gate.
    if min(shape) < 2 or math.prod(shape) > max_voxels:
        raise ValueError("Synthetic scene exceeds its bounded voxel budget")
    origin = tuple(lo + step / 2 for lo, step in zip(lower, spacing))
    axes = [lo + (np.arange(n, dtype=np.float32) + 0.5) * step for lo, n, step in zip(lower, shape, spacing)]
    x, z = np.meshgrid(axes[0], axes[2], indexing="ij")
    surface, dx, dz = _surface(geometry, x, z)
    distance = (axes[1][None, :, None] - surface[:, None, :]) / np.sqrt(1 + dx[:, None, :]**2 + dz[:, None, :]**2)
    projected = (spacing[1] + np.abs(dx) * spacing[0] + np.abs(dz) * spacing[2]) / np.sqrt(1 + dx**2 + dz**2)
    ellipse = (x / geometry.radius_x_mm)**2 + (z / geometry.radius_z_mm)**2
    interior_xz = ellipse <= 1
    chamber = interior_xz[:, None, :] & (axes[1][None, :, None] > -12.0) & (axes[1][None, :, None] < surface[:, None, :])
    rng = np.random.default_rng(int(_digest([geometry.geometry_id, asdict(a)])[:16], 16))
    soft_tissue_hu = float(np.float32(rng.uniform(-20, 80)))
    fill = np.full((shape[0], 1, shape[2]), -900.0 if a.opacity != "full" else soft_tissue_hu, dtype=np.float32)
    if a.opacity == "partial":
        fill = np.where(z[:, None, :] > 0.1 * geometry.radius_z_mm, soft_tissue_hu, -900.0).astype(np.float32)
    base = np.where(chamber, fill, soft_tissue_hu).astype(np.float32)
    thickness = geometry.wall_thickness_mm * (0.01 if a.wall_state == "averaged_away" else 1)
    outer_fraction = _slab_fraction(distance, thickness, projected[:, None, :]) * interior_xz[:, None, :]
    if a.wall_state == "missing":
        outer_fraction.fill(0)
    # Independently generated partial septa on both sides of the specified wall.
    internal_region = interior_xz & (np.abs(z / geometry.radius_z_mm + geometry.distractor_phase) < 0.45)
    external_region = interior_xz & (np.abs(x / geometry.radius_x_mm - geometry.distractor_phase) < 0.5)
    normal = np.sqrt(1 + dx**2 + dz**2)[:, None, :]
    internal = _slab_fraction(distance + geometry.internal_offset_mm / normal,
                              geometry.distractor_thickness_mm, projected[:, None, :]) * internal_region[:, None, :]
    external = _slab_fraction(distance - geometry.external_offset_mm / normal,
                              geometry.distractor_thickness_mm, projected[:, None, :]) * external_region[:, None, :]
    back = _slab_fraction(axes[1][None, :, None] + 12.0, 1.6, spacing[1]) * interior_xz[:, None, :]
    sides = ((ellipse > 1) & (ellipse < 1.3))[:, None, :] & (axes[1][None, :, None] >= -12) & (axes[1][None, :, None] <= surface[:, None, :])
    other_fraction = np.clip(internal + external + back + sides, 0, 1)
    if a.wall_state == "ambiguous":
        rival = _slab_fraction(distance - 3.0 / normal, thickness, projected[:, None, :]) * interior_xz[:, None, :]
        other_fraction = np.clip(other_fraction + rival, 0, 1)
    total = np.clip(outer_fraction + other_fraction, 0, 1)
    share = np.divide(outer_fraction, outer_fraction + other_fraction,
                      out=np.zeros_like(outer_fraction), where=(outer_fraction + other_fraction) > 0)
    signal = (1400.0 - base) * total
    outer_signal, other_signal = signal * share, signal * (1 - share)
    if a.blur_sigma_mm:
        from scipy.ndimage import gaussian_filter
        sigma = tuple(a.blur_sigma_mm / s for s in spacing)
        base = gaussian_filter(base, sigma, mode="nearest")
        outer_signal = gaussian_filter(outer_signal, sigma, mode="constant", cval=0)
        other_signal = gaussian_filter(other_signal, sigma, mode="constant", cval=0)
    before_noise = base + outer_signal + other_signal
    hu = np.clip(before_noise + rng.normal(0, a.noise_std_hu, size=shape).astype(np.float32), -1200, 3000).astype(np.float32)
    analytic_seed = ((ellipse < 0.52)[:, None, :] & (axes[1][None, :, None] >= -9.0)
                     & (axes[1][None, :, None] <= -6.0) & chamber)
    # Like a bone-excluded localization interior, the supplied seed cannot
    # contain a sampled bone voxel, including an opacity/noise threshold hit.
    # Retain the independent procedural region to make this distinction clear.
    seed = analytic_seed & (hu < BONE_HU)
    if not np.any(seed):
        raise ValueError("The sampled scene contains no posterior seed")
    bone = hu >= BONE_HU
    leading = bone.copy()
    leading[:, 1:, :] &= ~bone[:, :-1, :]
    leading[:, 0, :] = False
    outer_wall = outer_fraction > 1e-6
    seed_columns = np.any(seed, axis=1)
    wall_support = (leading & _wall_signal_support(base, outer_signal, other_signal)
                    & seed_columns[:, None, :] & (axes[1][None, :, None] > -6.0))
    # Rival sheets have no uniquely adjudicated target even though bone exists.
    if a.wall_state == "ambiguous":
        wall_support.fill(False)
    support_columns = np.any(wall_support, axis=1)
    fraction = float(np.count_nonzero(support_columns) / np.count_nonzero(seed_columns))
    # The scoring target is the seed-facing slab boundary, not its center.
    # Normal thickness projects to Y thickness * sqrt(1 + dx^2 + dz^2).
    leading_surface = surface - thickness * np.sqrt(1 + dx**2 + dz**2) / 2
    arrays = (hu, seed, analytic_seed, outer_wall, wall_support, surface, leading_surface, support_columns)
    for array in arrays:
        array.setflags(write=False)
    return SyntheticWallScene(
        hu=hu, spacing_mm=spacing, origin_mm=origin, seed=seed, analytic_seed=analytic_seed,
        outer_wall=outer_wall, wall_support=wall_support,
        target_surface_y_mm=surface, target_leading_surface_y_mm=leading_surface,
        support_columns=support_columns,
        observable=bool(fraction >= 0.8 and a.wall_state not in {"ambiguous", "cropped"}),
        ambiguous=a.wall_state == "ambiguous", cropped=a.wall_state == "cropped",
        geometry_id=geometry.geometry_id, family=geometry.family,
        augmentation_id="augmentation-" + _digest(asdict(a))[:24],
        support_fraction=fraction, wall_state=a.wall_state, soft_tissue_hu=soft_tissue_hu,
    )


def candidate_targets(scene: SyntheticWallScene, positions_ijk: np.ndarray) -> np.ndarray:
    """Label supplied positions by generative provenance, never candidate rank.

    1 = observed leading face supported by the noiseless outer-wall signal;
    0 = not independently supported as the outer wall, including noise-only hits;
    -1 = ambiguous scene, excluded from forced-choice training/evaluation targets.
    Coordinates alone are accepted so no extractor or decoder is imported here.
    """
    points = np.asarray(positions_ijk)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "iu"
            or np.any(points < 0) or np.any(points >= np.asarray(scene.hu.shape))):
        raise ValueError("Candidate positions must be integer native voxel coordinates")
    if scene.ambiguous:
        return np.full(len(points), -1, dtype=np.int8)
    return scene.wall_support[tuple(points.T)].astype(np.int8)
