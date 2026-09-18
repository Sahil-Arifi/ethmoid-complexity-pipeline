"""Automatic anatomical bounds from TotalSegmentator tasks 115, 775, and 297.

Neither network labels an ethmoid cell, a sphenoid sinus, or a named ethmoid
variant. Those boundaries below are explicit, unvalidated geometric proposals.
Missing evidence is a failure, never a request for a hand-drawn initial ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .engine import CountingConfig, CountingError, _sha256, geometry

TASKS = ("craniofacial_structures", "head_glands_cavities", "total")
MASK_TASK = {
    "skull": TASKS[0], "sinus_maxillary": TASKS[0], "sinus_frontal": TASKS[0],
    "eye_left": TASKS[1], "eye_right": TASKS[1],
    "optic_nerve_left": TASKS[1], "optic_nerve_right": TASKS[1],
    "nasal_cavity_left": TASKS[1], "nasal_cavity_right": TASKS[1],
    "brain": TASKS[2],
}


@dataclass
class Localization:
    region: np.ndarray
    excluded: np.ndarray
    variants: dict[str, np.ndarray]
    midline_ras_x: float
    warnings: list[str]


def localize_native(hu: np.ndarray, affine: np.ndarray, masks: dict[str, np.ndarray], config: CountingConfig, plan: dict, *, localization_policy: dict | None = None, diagnostics: dict | None = None) -> Localization:
    """Map anatomical membership back before any cavity or air classification."""
    options = {}
    if localization_policy is not None:
        options["localization_policy"] = localization_policy
    if diagnostics is not None:
        options["diagnostics"] = diagnostics
    if plan["mode"] == "native_axis_aligned":
        return automatic_region(hu, affine, masks, config, **options)
    from .grids import anatomy_proxy, resample

    proxy_hu, proxy_affine, proxy_masks, valid = anatomy_proxy(hu, affine, masks, plan)
    localized = automatic_region(proxy_hu, proxy_affine, proxy_masks, config, **options)
    if np.any(localized.region & (~valid | ndi.binary_dilation(~valid))):
        raise CountingError("TARGET_CLIPPED", "The anatomical proxy reaches the acquired source field-of-view boundary.")

    def native(mask: np.ndarray, outside: int = 0) -> np.ndarray:
        return resample(mask.astype(np.uint8), proxy_affine, hu.shape, affine, order=0, outside=outside).astype(bool)

    excluded = native(localized.excluded, outside=1)
    # Reapply native learned contents: interpolation must not reintroduce brain,
    # eyes, or nerves that the original source-bound masks explicitly exclude.
    for name in ("brain", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right"):
        excluded |= masks[name]
    region = native(localized.region) & ~excluded
    if not region.any():
        raise CountingError("EMPTY_REGION", "No anatomical membership remains on the original source grid.")
    lo, hi = _bounds(region)
    if any(lo[i] == 0 or hi[i] == hu.shape[i] - 1 for i in range(3)):
        raise CountingError("TARGET_CLIPPED", "The mapped counting region reaches the source field-of-view edge.")
    variants = {name: native(mask) & ~excluded for name, mask in localized.variants.items()}
    return Localization(region, excluded, variants, localized.midline_ras_x,
                        localized.warnings + ["Oblique anatomical bounds were mapped from a linear-HU localization proxy; all cavity, septum and air decisions use original native CT samples."])


def _predict_task(source: Path, destination: Path, task: str, *, full_extent: bool = False) -> dict[str, Any]:
    """Bounded GPU-then-CPU attempt; preserve source/native output geometry."""
    import torch
    from totalsegmentator.python_api import get_task_config, totalsegmentator

    def predict(device: str) -> None:
        brain_only = task == "total"
        if not full_extent:
            totalsegmentator(input=source, output=destination, task=task,
                robust_crop=True, fast=brain_only, roi_subset=["brain"] if brain_only else None, device=device,
                nr_thr_resamp=2, nr_thr_saving=1, statistics=False, preview=False, quiet=True)
            return
        # The pinned public wrapper does not expose crop=None. Its own lower-level
        # inference API does, using the same pinned task/trainer/plans/settings.
        from totalsegmentator.nnunet import nnUNet_predict_image
        settings = get_task_config(task, fast=brain_only, fastest=False, quiet=True, robust_crop=True, plans="nnUNetPlans", model_size="big")
        nnUNet_predict_image(source, destination, settings["task_id"],
            model=settings["model"], folds=settings["folds"], trainer=settings["trainer"],
            tta=False, multilabel_image=False, resample=settings["resample"],
            crop=None, crop_path=None, task_name=task, preview=False,
            nr_threads_resampling=2, nr_threads_saving=1, force_split=False,
            output_type="nifti", statistics=False, quiet=True, verbose=False,
            device="cuda" if device == "gpu" else "cpu", plans=settings["plans"],
            resampling_order=1, output_task_name=task, roi_subset=["brain"] if brain_only else None)

    try:
        predict("gpu")
        return {"device": "gpu", "cpu_fallback": False, "full_extent": full_extent}
    except RuntimeError as error:
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    predict("cpu")
    return {"device": "cpu", "cpu_fallback": True, "full_extent": full_extent}


def run_localizers(source: Path, destination: Path, *, full_extent: bool = False) -> None:
    """Use only the preloaded, network-blocked inference runtime.

    Bootstrap owns downloading/hashing the public weights *before* patient work.
    The runtime's sitecustomize blocks outbound sockets by default. Requiring it
    here prevents accidental use of a normal network-enabled Python environment.
    """
    runtime = Path(sys.prefix)
    manifest = runtime / "weights-manifest.json"
    if not manifest.is_file() or os.environ.get("ETHMOID_INFERENCE_OFFLINE", "1") != "1":
        raise CountingError("LOCALIZER_NOT_READY", "The preloaded offline anatomical-localizer runtime is required.")
    try:
        import sitecustomize  # type: ignore[import-not-found]

        # Require the bootstrap's own guard, not an unrelated sitecustomize.
        if "counting-runtime" not in str(Path(sitecustomize.__file__).resolve()).lower():
            raise CountingError("LOCALIZER_NOT_READY", "The offline inference guard is unavailable.")
        config_path = runtime / "totalsegmentator" / "config.json"
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        if settings.get("send_usage_stats") is not False:
            raise CountingError("LOCALIZER_NOT_READY", "Localizer telemetry must be disabled before processing.")
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        package_version = metadata.version("TotalSegmentator")
        if provenance.get("totalsegmentator_version") != package_version:
            raise CountingError("LOCALIZER_INTEGRITY", "The localizer package differs from its recorded model provenance.")
        manifest_sha256 = _sha256(manifest)
        models = {item["task_id"]: item for item in provenance.get("models", [])}
        weights_root = (runtime / "totalsegmentator" / "nnunet" / "results").resolve()
        if set((115, 775, 297, 298)) - set(models):
            raise CountingError("LOCALIZER_NOT_READY", "Required model provenance is incomplete.")
        for task_id in (115, 775, 297, 298):
            entries = models[task_id].get("files", [])
            if not entries:
                raise CountingError("LOCALIZER_NOT_READY", "Required model provenance is incomplete.")
            for entry in entries:
                path = (weights_root / entry["path"]).resolve()
                if not path.is_relative_to(weights_root) or not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
                    raise CountingError("LOCALIZER_INTEGRITY", "A pretrained localizer file differs from its recorded provenance.")
        os.environ["TOTALSEG_HOME_DIR"] = str(runtime / "totalsegmentator")
        os.environ["TOTALSEG_WEIGHTS_PATH"] = str(runtime / "totalsegmentator" / "nnunet" / "results")
        attempts = {}
        for task in TASKS:
            attempts[task] = _predict_task(source, destination / task, task, full_extent=full_extent)
        cache = {
            "schema_version": 1, "source_sha256": _sha256(source),
            "totalsegmentator_version": package_version,
            "weights_manifest_sha256": manifest_sha256,
            "tasks": [{"task": task, **attempts[task], "weights_manifest_sha256": manifest_sha256,
                "model_task_id": {"craniofacial_structures": 115, "head_glands_cavities": 775, "total": 297}[task],
                "fast": task == "total", "roi_subset": ["brain"] if task == "total" else None,
                "robust_crop": not full_extent, "masks": [
                {"class": name, "filename": f"{name}.nii.gz", "sha256": _sha256(destination / task / f"{name}.nii.gz")}
                for name, model_task in MASK_TASK.items() if model_task == task
            ]} for task in TASKS],
        }
        (destination / "localizers.local.json").write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    except CountingError:
        raise
    except Exception:
        raise CountingError("LOCALIZER_FAILED", "The offline anatomical localizers could not complete.") from None


def cache_producer_warnings(directory: Path) -> list[str]:
    """Reject contradictory producer evidence; explicitly identify legacy caches."""
    provenance = json.loads((directory / "localizers.local.json").read_text(encoding="utf-8"))
    manifest = Path(sys.prefix) / "weights-manifest.json"
    expected_digest = _sha256(manifest) if manifest.is_file() else None
    try:
        package_version = metadata.version("TotalSegmentator")
    except metadata.PackageNotFoundError:
        package_version = None
    recorded_version = provenance.get("totalsegmentator_version")
    if recorded_version and package_version and recorded_version != package_version:
        raise CountingError("LANDMARK_PROVENANCE", "Cached anatomical localizers were produced by a different package version.")
    tasks = {task["task"]: task for task in provenance.get("tasks", [])}
    verified = bool(expected_digest and package_version and recorded_version == package_version)
    for task in TASKS:
        digest = tasks.get(task, {}).get("weights_manifest_sha256", provenance.get("weights_manifest_sha256"))
        if digest and expected_digest and digest != expected_digest:
            raise CountingError("LANDMARK_PROVENANCE", "Cached anatomical localizers were produced with a different recorded model manifest.")
        verified &= bool(digest and expected_digest and digest == expected_digest)
    if verified:
        return []
    return ["Legacy anatomical cache: producer package/weight binding is incomplete or unavailable; current-runtime model provenance must not be attributed to these cached masks."]


def cache_producer_summary(directory: Path) -> dict[str, Any]:
    path = directory / "localizers.local.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    verified = not cache_producer_warnings(directory)
    tasks = {item["task"]: item for item in provenance.get("tasks", [])}
    return {
        "producer_binding": "verified" if verified else "unverified_legacy",
        "cache_manifest_sha256": _sha256(path),
        "totalsegmentator_version": provenance.get("totalsegmentator_version"),
        "tasks": [{"task": task,
            "model_task_id": tasks.get(task, {}).get("model_task_id"),
            "weights_manifest_sha256": tasks.get(task, {}).get("weights_manifest_sha256", provenance.get("weights_manifest_sha256")),
        } for task in TASKS],
    }


def load_landmarks(directory: Path, source: nib.spatialimages.SpatialImage, *, source_sha256: str | None = None) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    checksums: dict[tuple[str, str], str] = {}
    if source_sha256 is not None and directory.is_dir():
        manifest = directory / "localizers.local.json"
        if not manifest.is_file():
            raise CountingError("LANDMARK_PROVENANCE", "Cached anatomical localizers lack source provenance.")
        try:
            provenance = json.loads(manifest.read_text(encoding="utf-8"))
            if provenance.get("source_sha256") != source_sha256:
                raise CountingError("LANDMARK_SOURCE", "Cached anatomical localizers belong to a different source volume.")
            checksums = {(task["task"], mask["class"]): mask["sha256"] for task in provenance["tasks"] for mask in task["masks"]}
            cache_producer_warnings(directory)
        except (OSError, ValueError, KeyError, TypeError):
            raise CountingError("LANDMARK_PROVENANCE", "Cached anatomical-localizer provenance is invalid.") from None
    for name, task in MASK_TASK.items():
        path = directory / task / f"{name}.nii.gz"
        if not path.is_file():
            path = directory / f"{name}.nii.gz"
        if not path.is_file():
            raise CountingError("LANDMARKS_MISSING", "Required automatic anatomical-localizer masks are missing.")
        if source_sha256 is not None and checksums.get((task, name)) != _sha256(path):
            raise CountingError("LANDMARK_INTEGRITY", "A cached anatomical-localizer mask differs from its recorded provenance.")
        image = nib.as_closest_canonical(nib.load(path))
        if image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-4):
            raise CountingError("LANDMARK_GEOMETRY", "An anatomical-localizer mask does not match the source grid.")
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
            raise CountingError("LANDMARK_VALUES", "An anatomical-localizer output is not a binary structure mask.")
        masks[name] = data.astype(bool)
    return masks


def _bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.where(mask)
    if not len(points[0]):
        raise CountingError("LANDMARK_EMPTY", "A required anatomical structure was not detected.")
    return np.array([p.min() for p in points]), np.array([p.max() for p in points])


def _near(mask: np.ndarray, distance_mm: float, spacing: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= distance_mm


def _fill_surfaces(surface: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise CountingError("ORBIT_WALL_UNRESOLVED", "The medial bony orbital boundary could not be recovered.")
    indices = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return surface[tuple(indices)]


def _bone_enclosed_planes(bone: np.ndarray, axis: int) -> np.ndarray:
    """Enclosed non-bone profiles; no bone gap is fabricated by closing.

    Profiles are evidence, not final instances: cavities that communicate in a
    different plane are subsequently resolved with the 3D compartment method.
    """
    result = np.zeros_like(bone)
    for index in range(bone.shape[axis]):
        selection = [slice(None)] * 3
        selection[axis] = index
        plane = bone[tuple(selection)]
        result[tuple(selection)] = ndi.binary_fill_holes(plane) & ~plane
    return result


def _orbit_from_bone(eye: np.ndarray, profiles: np.ndarray, bone: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, int]:
    labels, _ = ndi.label(profiles)
    seeds, overlap = np.unique(labels[eye], return_counts=True)
    usable = [(int(n), int(count)) for n, count in zip(seeds, overlap) if n > 0]
    if usable:
        label = max(usable, key=lambda item: item[1])[0]
        orbit = labels == label
        # An unclosed orbital/skull boundary must not turn the brain into orbit.
        if orbit.sum() > eye.sum() * 12:
            orbit = np.zeros_like(eye)
    else:
        orbit = np.zeros_like(eye)
    lo, hi = _bounds(eye)
    floors: list[int] = []
    # Native bone immediately below eye columns provides the actual floor,
    # rather than substituting the lower border of the eyeball mask.
    max_depth = max(2, int((hi[2] - lo[2] + 1) * 0.8))
    for x, y in zip(*np.where(eye.any(axis=2))):
        eye_z = np.flatnonzero(eye[x, y])
        top = int(eye_z.min())
        bottom = max(0, top - max_depth)
        candidates = np.flatnonzero(bone[x, y, bottom:top])
        if candidates.size:
            floors.append(bottom + int(candidates[-1]))
    if len(floors) < max(10, int(eye.any(axis=2).sum() * 0.15)):
        raise CountingError("ORBIT_FLOOR_UNRESOLVED", "Native bony evidence for the inferior orbital plane is insufficient.")
    floor = int(np.quantile(floors, 0.25))
    # Eye and optic content cannot be a cell even where an orbital ring is open.
    orbit |= eye
    return orbit, floor


def _medial_orbital_surface(seed: np.ndarray, bone: np.ndarray, midline: float, side: str, spacing: np.ndarray) -> np.ndarray:
    """Cast rays from known orbital contents, never from arbitrary image rows."""
    surface = np.zeros(bone.shape[1:], dtype=np.int32)
    valid = np.zeros(surface.shape, dtype=bool)
    for yy, zz in zip(*np.where(seed.any(axis=0))):
        xs = np.flatnonzero(seed[:, yy, zz])
        if side == "left":
            start = int(xs.max())
            if start >= midline:
                continue
            positions = np.flatnonzero(bone[start:int(np.ceil(midline)), yy, zz])
            if positions.size:
                surface[yy, zz] = start + int(positions[0])
                valid[yy, zz] = True
        else:
            end = int(xs.min())
            start = int(np.floor(midline))
            if end <= midline:
                continue
            positions = np.flatnonzero(bone[start:end + 1, yy, zz])
            if positions.size:
                surface[yy, zz] = start + int(positions[-1])
                valid[yy, zz] = True
    if valid.sum() < max(10, seed.any(axis=0).sum() * 0.15):
        raise CountingError("ORBIT_WALL_UNRESOLVED", "Bone-supported rays from orbital contents do not resolve a medial orbital wall.")
    # Suppress stair-step/local internal-bone outliers. These interpolated surfaces
    # are anatomical proposals, not voxel-exact labels of the lamina papyracea.
    size = tuple(max(1, int(np.ceil(3.0 / step))) for step in spacing[1:])
    return ndi.median_filter(_fill_surfaces(surface, valid), size=size)


def _intracranial_roof(hu: np.ndarray, bone: np.ndarray, brain_mask: np.ndarray, nasal: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refine the inferior learned brain surface against native skull-base bone.

    The 3 mm brain prior is an exclusion/localization anchor, never a cell model.
    Inferred enclosed tissue alone cannot reliably distinguish brain from sinus.
    """
    z = np.arange(hu.shape[2])[None, None, :]
    brain = np.asarray(brain_mask, dtype=bool)
    if not brain.any():
        raise CountingError("ROOF_UNRESOLVED", "The pretrained intracranial localization anchor is empty.")
    lowest = np.min(np.where(brain, z, hu.shape[2]), axis=2).astype(np.int32)
    roof = np.zeros(hu.shape[:2], dtype=np.int32)
    valid = np.zeros(roof.shape, dtype=bool)
    depth = max(1, int(np.ceil(3.0 / spacing[2])))
    for xx, yy in zip(*np.where(lowest < hu.shape[2])):
        top = int(lowest[xx, yy])
        bottom = max(0, top - depth)
        upper = min(hu.shape[2], top + depth + 1)
        positions = np.flatnonzero(bone[xx, yy, bottom:upper])
        if positions.size:
            absolute = positions + bottom
            roof[xx, yy] = int(absolute[np.argmin(np.abs(absolute - top))])
            valid[xx, yy] = True
    if np.count_nonzero(valid & nasal.any(axis=2)) < 10:
        raise CountingError("ROOF_UNRESOLVED", "The intracranial component lacks sufficient native bony support above the nasal anatomy.")
    size = tuple(max(1, int(np.ceil(3.0 / step))) for step in spacing[:2])
    return ndi.median_filter(_fill_surfaces(roof, valid), size=size), brain


def _continuous_wall(wall: np.ndarray, valid: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Fit a curved anterior surface; global Y variation is not inconsistency."""
    xx, zz = np.where(valid)
    if len(xx) < 12 or np.ptp(xx) < 2 or np.ptp(zz) < 2:
        raise CountingError("SPHENOID_UNRESOLVED", "The anterior wall lacks spatially distributed bone evidence.")
    x = (xx - np.mean(xx)) * spacing[0] / 10
    z = (zz - np.mean(zz)) * spacing[2] / 10
    design = np.column_stack((np.ones(len(x)), x, z, x * x, x * z, z * z))
    observed = wall[valid] * spacing[1]
    weights = np.ones(len(x))
    prediction = observed.copy()
    for _ in range(6):
        root_weights = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(design * root_weights[:, None], observed * root_weights, rcond=None)
        if rank < 6:
            raise CountingError("SPHENOID_UNRESOLVED", "Anterior-wall support is geometrically degenerate.")
        prediction = design @ coefficients
        residual = np.abs(prediction - observed)
        weights = np.minimum(1.0, 1.5 / np.maximum(residual, 1e-6))
    # This is an engineering residual tolerance in millimetres, not a normative
    # claim about sphenoid shape. A broad but coherent curved wall is allowed.
    inlier = residual <= max(2.0, 3 * float(spacing[1]))
    if inlier.mean() < 0.70:
        raise CountingError("SPHENOID_AMBIGUOUS", "Anterior bony-wall rays do not support a coherent curved surface.")
    fitted = wall.copy()
    fitted[xx[inlier], zz[inlier]] = np.rint(prediction[inlier] / spacing[1]).astype(np.int32)
    supported = np.zeros_like(valid)
    supported[xx[inlier], zz[inlier]] = True
    return _fill_surfaces(fitted, supported)


def _closed_nasal_pockets(nasal: np.ndarray, bone: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Keep only non-bone nasal contents separated from the open parent lumen."""
    labels, _ = ndi.label(nasal & ~bone & ~brain)
    boundary = nasal & ~ndi.binary_erosion(nasal)
    open_ids = np.unique(labels[boundary])
    return (labels > 0) & ~np.isin(labels, open_ids)


def _sphenoid_boundary(hu: np.ndarray, bone: np.ndarray, enclosed: np.ndarray, nasal: np.ndarray, nerves: np.ndarray, eye_bounds: tuple[np.ndarray, np.ndarray], floor: int, spacing: np.ndarray, midline: float, *, localization_policy: dict | None = None, diagnostics: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Select posterior central bone-enclosed sinus parent, including opacity.

    Candidates must lie posterior to the optic-nerve centroid, beneath their
    superior extent, contact the midline neighborhood, and have an anterior
    bony wall. This is a provisional anatomical heuristic, not a learned label.
    """
    v4_policy = isinstance(localization_policy, dict) and (
        set(localization_policy) == {"sphenoid_seed_clearance_mm", "physical_ray_extent"}
        and type(localization_policy["sphenoid_seed_clearance_mm"]) in (int, float)
        and localization_policy["sphenoid_seed_clearance_mm"] == 1.0
        and localization_policy["physical_ray_extent"] is True
    )
    v5_policy = isinstance(localization_policy, dict) and (
        set(localization_policy) == {"sphenoid_seed_mode", "sphenoid_fallback_clearance_mm",
                                    "fallback_requires_max_spacing_gt_mm", "physical_ray_extent"}
        and localization_policy["sphenoid_seed_mode"] == "preserve_legacy_then_thick_empty_fallback"
        and type(localization_policy["sphenoid_fallback_clearance_mm"]) in (int, float)
        and localization_policy["sphenoid_fallback_clearance_mm"] == 1.0
        and type(localization_policy["fallback_requires_max_spacing_gt_mm"]) in (int, float)
        and localization_policy["fallback_requires_max_spacing_gt_mm"] == 1.0
        and localization_policy["physical_ray_extent"] is True
    )
    if localization_policy is not None and not (v4_policy or v5_policy):
        raise CountingError("INVALID_CONFIG", "The research sphenoid localization policy is unsupported.")
    # The research setting affects a localization seed only. It does not alter
    # native cell enclosure, separator support, or accepted-count coverage.
    clearance = 1.0 if v4_policy else max(1.0, float(spacing.max()) * 2)
    diagnostic = {}
    if diagnostics is not None:
        diagnostics["sphenoid"] = diagnostic
        diagnostic.update(seed_clearance_mm=clearance, spacing_mm=spacing.tolist())
        if v5_policy:
            diagnostic.update(initial_seed_clearance_mm=clearance, chosen_seed_clearance_mm=clearance,
                seed_spacing_reference="current_anatomical_localization_grid",
                seed_fallback_used=False, seed_fallback_reason="legacy_seed_not_yet_evaluated")
    _, eye_hi = eye_bounds
    nerve_points = np.where(nerves)
    cutoff_y = int(np.median(nerve_points[1]))
    max_z = int(np.quantile(nerve_points[2], 0.95))
    eye_width = max(1, int(eye_hi[0] - eye_bounds[0][0]))
    x = np.arange(hu.shape[0])[:, None, None]
    y = np.arange(hu.shape[1])[None, :, None]
    z = np.arange(hu.shape[2])[None, None, :]
    near_nasal = _near(nasal, float(eye_width * spacing[0] * 0.30), spacing)
    search = enclosed & near_nasal & (y < cutoff_y) & (z > floor) & (z <= max_z)
    search &= np.abs(x - midline) <= eye_width * 0.18
    # A closed profile can contain opaque tissue. Do not seed from air alone.
    connected, n = ndi.label(search)
    choices: list[tuple[float, int, int]] = []
    for label, slices in enumerate(ndi.find_objects(connected), start=1):
        if slices is None:
            continue
        mask = connected[slices] == label
        size = int(mask.sum())
        if size * float(np.prod(spacing)) < 20:
            continue
        points = np.where(mask)
        anterior = int(np.quantile(points[1], 0.90)) + slices[1].start
        posterior = int(np.quantile(points[1], 0.10)) + slices[1].start
        if (anterior - posterior) * spacing[1] < 3:
            continue
        full = connected == label
        surface = ndi.binary_dilation(full) & ~full
        support = float(np.count_nonzero(surface & bone)) / max(1, int(surface.sum()))
        if support < 0.20:
            continue
        # Bigger midline posterior compartments are the major sinus parent;
        # superior/lateral smaller compartments are retained for Onodi review.
        score = size * support
        choices.append((score, label, anterior))
    if diagnostics is not None:
        diagnostic.update(search_voxels=int(search.sum()), connected_candidates=int(n), eligible_candidates=len(choices))
    if not choices:
        raise CountingError("SPHENOID_UNRESOLVED", "A bone-supported posterior sphenoid boundary could not be established automatically.")
    choices.sort(reverse=True)
    best_score, label, anterior = choices[0]
    if len(choices) > 1 and choices[1][0] > best_score * 0.85 and abs(choices[1][2] - anterior) * spacing[1] > 4:
        raise CountingError("SPHENOID_AMBIGUOUS", "Competing posterior compartments give incompatible sphenoid boundaries.")
    parent_seed = connected == label
    interior_distance = ndi.distance_transform_edt(parent_seed, sampling=spacing)
    interior_seed = parent_seed & (interior_distance >= clearance)
    if v5_policy:
        initial_size = int(interior_seed.sum())
        if initial_size:
            reason = "legacy_seed_supported"
        elif float(spacing.max()) > 1.0:
            clearance = 1.0
            interior_seed = parent_seed & (interior_distance >= clearance)
            reason = "legacy_seed_empty_thick_grid"
        else:
            reason = "legacy_seed_empty_thin_grid"
        if diagnostics is not None:
            diagnostic.update(initial_interior_seed_voxels=initial_size,
                chosen_seed_clearance_mm=clearance, seed_clearance_mm=clearance,
                seed_fallback_used=reason == "legacy_seed_empty_thick_grid", seed_fallback_reason=reason)
    if diagnostics is not None:
        diagnostic.update(parent_voxels=int(parent_seed.sum()), available_interior_radius_mm=float(interior_distance.max()),
                          interior_seed_voxels=int(interior_seed.sum()), seed_columns=int(interior_seed.any(axis=1).sum()))
    if not interior_seed.any():
        raise CountingError("SPHENOID_UNRESOLVED", "The posterior sinus candidate has no spatially resolved interior.")
    wall = np.zeros((hu.shape[0], hu.shape[2]), dtype=np.int32)
    valid = np.zeros(wall.shape, dtype=bool)
    # A seed quantile is not a bony wall. Follow the lumen anteriorly to actual
    # native bone on each supported column, with an anatomy-scaled search limit.
    ray_extent_voxels = eye_width * 0.25
    if localization_policy is not None:
        ray_extent_voxels *= spacing[0] / spacing[1]
    limit_y = int(np.median(np.where(nerves)[1]) + ray_extent_voxels)
    limit_y = min(limit_y, hu.shape[1] - 1)
    if diagnostics is not None:
        diagnostic.update(ray_search_extent_mm=float(ray_extent_voxels * spacing[1]),
                          clipped_ray_search_extent_mm=float((limit_y - np.median(nerve_points[1])) * spacing[1]))
    # Smoothing is used only to estimate a surface normal, never to change the
    # HU volume or manufacture a counted septum.
    normal_image = ndi.gaussian_filter(hu, sigma=0.5 / spacing)
    bone_hit_rays = 0
    for xx, zz in zip(*np.where(interior_seed.any(axis=1))):
        start = int(np.flatnonzero(interior_seed[xx, :, zz]).max())
        hits = np.flatnonzero(bone[xx, start:limit_y + 1, zz])
        if hits.size:
            bone_hit_rays += 1
            hit = start + int(hits[0])
            if xx < 1 or zz < 1 or xx >= hu.shape[0] - 1 or zz >= hu.shape[2] - 1 or hit < 2:
                continue
            gy = (normal_image[xx, hit, zz] - normal_image[xx, hit - 2, zz]) / (2 * spacing[1])
            gx = (normal_image[xx + 1, hit - 1, zz] - normal_image[xx - 1, hit - 1, zz]) / (2 * spacing[0])
            gz = (normal_image[xx, hit - 1, zz + 1] - normal_image[xx, hit - 1, zz - 1]) / (2 * spacing[2])
            if gy <= 0 or gy < max(abs(gx), abs(gz)):
                continue
            wall[xx, zz] = hit
            valid[xx, zz] = True
    if diagnostics is not None:
        diagnostic.update(bone_hit_rays=bone_hit_rays, supported_wall_rays=int(valid.sum()),
                          required_wall_rays=float(max(12, interior_seed.any(axis=1).sum() * 0.5)))
    if valid.sum() < max(12, interior_seed.any(axis=1).sum() * 0.5):
        raise CountingError("SPHENOID_UNRESOLVED", "The posterior sinus candidate lacks a supported anterior bony wall.")
    size = (max(1, int(np.ceil(2 / spacing[0]))), max(1, int(np.ceil(2 / spacing[2]))))
    wall = ndi.median_filter(_continuous_wall(wall, valid, spacing), size=size)
    parent = enclosed & near_nasal & (y < wall[:, None, :]) & (z > floor) & (z <= max_z)
    parent &= np.abs(x - midline) <= eye_width * 0.18
    return parent, wall[:, None, :]


def automatic_region(hu: np.ndarray, affine: np.ndarray, masks: dict[str, np.ndarray], config: CountingConfig, *, localization_policy: dict | None = None, diagnostics: dict | None = None) -> Localization:
    spacing = geometry(affine)
    if nib.aff2axcodes(affine) != ("R", "A", "S"):
        raise CountingError("LOCALIZATION_ORIENTATION", "Automatic localization requires a canonical RAS view of the source grid.")
    if not np.allclose(affine[:3, :3] / spacing, np.eye(3), atol=1e-4):
        raise CountingError("OBLIQUE_GRID_UNSUPPORTED", "An oblique source requires a separately validated anatomical reorientation before this prototype can localize boundaries.")
    for name in ("skull", "brain", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right", "nasal_cavity_left", "nasal_cavity_right", "sinus_maxillary"):
        if name not in masks or masks[name].shape != hu.shape or not masks[name].any():
            raise CountingError("LANDMARK_EMPTY", "A required anatomical structure was not detected.")
    left, right = masks["eye_left"], masks["eye_right"]
    left_center, right_center = np.array(ndi.center_of_mass(left)), np.array(ndi.center_of_mass(right))
    if left_center[0] >= right_center[0]:
        raise CountingError("LANDMARK_LATERALITY", "Automatic eye laterality is inconsistent with the physical coordinate system.")
    eyes = left | right
    lo, hi = _bounds(eyes)
    if any(lo[i] <= 1 or hi[i] >= hu.shape[i] - 2 for i in range(3)):
        raise CountingError("TARGET_CLIPPED", "Required orbital anatomy reaches the source field-of-view edge.")
    nasal = masks["nasal_cavity_left"] | masks["nasal_cavity_right"]
    nerves = masks["optic_nerve_left"] | masks["optic_nerve_right"]
    bone = hu >= config.bone_hu
    # Work on localizer-derived anatomical extent, not a user-prepared ROI.
    coronal_holes = _bone_enclosed_planes(bone, 1)
    left_orbit, left_floor = _orbit_from_bone(left, coronal_holes, bone, spacing)
    right_orbit, right_floor = _orbit_from_bone(right, coronal_holes, bone, spacing)
    midline = float((left_center[0] + right_center[0]) / 2)
    floor = min(left_floor, right_floor)
    midline_ras = float(nib.affines.apply_affine(affine, [midline, 0, 0])[0])
    x = np.arange(hu.shape[0])[:, None, None]
    y = np.arange(hu.shape[1])[None, :, None]
    z = np.arange(hu.shape[2])[None, None, :]
    width_mm = float((right_center[0] - left_center[0]) * spacing[0])
    eye_radius_mm = float(np.median([(b[2] - a[2]) * spacing[2] / 2 for a, b in (_bounds(left), _bounds(right))]))
    if not 5 <= eye_radius_mm <= 20 or not 20 <= width_mm <= 100:
        raise CountingError("LANDMARK_SCALE", "The orbital localizers have an implausible adult anatomical scale.")
    left_wall = _medial_orbital_surface(left_orbit | masks["optic_nerve_left"], bone, midline, "left", spacing)
    right_wall = _medial_orbital_surface(right_orbit | masks["optic_nerve_right"], bone, midline, "right", spacing)
    interorbital = (x >= left_wall[None, :, :]) & (x <= right_wall[None, :, :])
    axial_holes = _bone_enclosed_planes(bone, 2)
    roof, brain = _intracranial_roof(hu, bone, masks["brain"], nasal, spacing)
    below_roof = z <= roof[:, :, None]
    # The nasal surface supplies the shape; orbital scale supplies an explicit
    # global dilation. This is not a fixed central rectangular crop.
    envelope = _near(nasal, eye_radius_mm * 1.25, spacing) & (z >= floor) & below_roof
    enclosed = coronal_holes | axial_holes
    skull_core = ndi.binary_erosion(masks["skull"], iterations=1) & (hu > -100)
    options = {}
    if localization_policy is not None:
        options["localization_policy"] = localization_policy
    if diagnostics is not None:
        options["diagnostics"] = diagnostics
    sphenoid, posterior_y = _sphenoid_boundary(hu, bone, enclosed & ~skull_core & ~brain, nasal, nerves, (lo, hi), floor, spacing, midline, **options)
    nasal_anterior = int(np.quantile(np.where(nasal)[1], 0.97))
    frontal = masks["sinus_frontal"]
    if frontal.any():
        # The inferior posterior part of the frontal-sinus model anchors the
        # recess. No full frontal sinus is automatically counted as an ethmoid.
        fp = np.where(frontal)
        recess_y = int(np.quantile(fp[1][fp[2] <= np.quantile(fp[2], 0.25)], 0.10))
        anterior_y = min(nasal_anterior, max(recess_y, int(np.quantile(np.where(eyes)[1], 0.6))))
    else:
        anterior_y = int(np.quantile(np.where(eyes)[1], 0.85))
    core = envelope & interorbital & (y >= posterior_y) & (y <= anterior_y)
    # Exclude the learned central nasal parent. A concha-bullosa candidate is
    # retained only when native bone separates its interior from the nasal mask
    # boundary; a window through open nasal contents is not a named variant.
    concha_closed = _closed_nasal_pockets(nasal, bone, brain)
    concha_closed &= below_roof & (y >= posterior_y)
    nasal_parent = nasal & ~concha_closed
    # Variants are *candidate neighborhoods*, never asserted anatomical labels.
    # All need a separately bone-supported 3D cavity before they can be counted.
    maxillary = masks["sinus_maxillary"]
    near_maxillary = _near(maxillary, eye_radius_mm * 0.35, spacing)
    near_frontal = _near(frontal, eye_radius_mm * 0.5, spacing) if frontal.any() else np.zeros_like(nasal)
    near_nerves = _near(nerves, eye_radius_mm * 0.6, spacing)
    variants = {
        "agger_nasi_or_frontal_recess": envelope & interorbital & (y > anterior_y) & (y <= nasal_anterior),
        "haller": near_maxillary & enclosed & (z >= floor - eye_radius_mm * 0.3 / spacing[2]) & (z <= floor + eye_radius_mm * 0.45 / spacing[2]) & (y >= posterior_y),
        "onodi": envelope & enclosed & near_nerves & (y < posterior_y) & ~sphenoid,
        "supraorbital_or_frontal": near_frontal & enclosed & (z >= hi[2] - eye_radius_mm * 0.5 / spacing[2]) & (y >= posterior_y),
        "concha_bullosa": concha_closed,
    }
    # Semantic sinus masks can absorb adjacent variants. Only their eroded
    # central contents seed major-parent exclusion; septations remain observable.
    # Connected free-space expansion then stays on the same side of native bone.
    free_labels, _ = ndi.label(enclosed & ~bone)
    excluded = left_orbit | right_orbit | eyes | nerves | sphenoid | brain | nasal_parent
    # A high-confidence interior of the learned skull is not a sinus cell.
    # Erosion avoids using the coarse semantic boundary to erase thin septa or
    # adjacent cavity contents; native HU still supplies all counted boundaries.
    excluded |= skull_core
    for major in (maxillary, frontal):
        if not major.any():
            continue
        core_seed = ndi.distance_transform_edt(major, sampling=spacing) > 2.0
        for half in (x <= midline, x > midline):
            candidates, sizes = np.unique(free_labels[core_seed & half], return_counts=True)
            candidates, sizes = candidates[candidates > 0], sizes[candidates > 0]
            if len(candidates):
                # Only the dominant parent on each side is excluded. Removing
                # every model-overlapping compartment would erase Haller and
                # frontal-region cells absorbed by a semantic sinus mask.
                best = int(np.argmax(sizes))
                if sizes[best] >= 8:
                    excluded |= free_labels == candidates[best]
    # Select complete separated profiles for variant review; intersecting a
    # broad neighborhood is not permission to include a clipped major-sinus rim.
    region = core.copy()
    variant_candidates = np.unique(free_labels[np.logical_or.reduce(list(variants.values())) & ~excluded])
    whole_variants = {name: np.zeros_like(core) for name in variants}
    objects = ndi.find_objects(free_labels)
    for value in variant_candidates:
        if value == 0:
            continue
        bounds = objects[int(value) - 1]
        if bounds is None:
            continue
        candidate = free_labels[bounds] == value
        if np.any(candidate & excluded[bounds]):
            continue
        for name, neighborhood in variants.items():
            overlap = int(np.count_nonzero(candidate & neighborhood[bounds]))
            if overlap >= max(4, int(candidate.sum() * 0.5)):
                whole_variants[name][bounds] |= candidate
                region[bounds] |= candidate
    variants = whole_variants
    variant_union = np.logical_or.reduce(list(variants.values()))
    # Include native boundary neighbors; do not create or close missing bone.
    region |= ndi.binary_dilation(variant_union, structure=ndi.generate_binary_structure(3, 1))
    region &= ~excluded
    if not region.any():
        raise CountingError("EMPTY_REGION", "The anatomy-derived region contains no resolvable ethmoid candidates.")
    rb_lo, rb_hi = _bounds(region)
    if any(rb_lo[i] == 0 or rb_hi[i] == hu.shape[i] - 1 for i in range(3)):
        raise CountingError("TARGET_CLIPPED", "The inferred counting region reaches the source field-of-view edge.")
    warnings = [
        "Needs clinical review: ethmoid roof, orbital walls, frontal recess, and sphenoid boundary are automatically inferred geometric bounds, not validated ethmoid labels.",
        "Named-variant neighborhoods are included provisionally; their anatomical membership and complete coverage are unvalidated.",
        "The inferior core plane uses native bony orbital-floor evidence; included variant neighborhoods may extend beyond the core.",
        "Frontal cells are interpreted as frontal-region ethmoid cells, excluding the main frontal sinus.",
        "Bone-supported inferred septations take precedence over apparent communication; uncertain splits require review.",
        "A separate 3 mm task-297 brain inference anchors intracranial exclusion; skull-base boundaries are refined on native CT bone.",
        "Central nasal contents are excluded; concha-bullosa candidates require separate native bone enclosure and may be missed where a wall is unresolved.",
    ]
    if not frontal.any():
        warnings.append("The frontal-sinus localizer is empty; frontal-recess localization uses orbital/nasal evidence and may be incomplete.")
    return Localization(region, excluded, variants, midline_ras, warnings)
