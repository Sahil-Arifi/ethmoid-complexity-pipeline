"""Bone-supported compartment proposals, not a clinically validated cell model.

The learned part of this pipeline is anatomical localization. Instance proposals
use native-HU geometry. Air is a *content* mask; it is never the identity of a
cell. In particular, a bone-enclosed opaque compartment can have no air voxels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

METHOD_VERSION = "native-bone-watershed-0.2.1"
PROTOCOL_VERSION = "ethmoid-count-prototype-v2"


class CountingError(RuntimeError):
    """A PHI-safe, expected failure; never includes source file contents."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CountingConfig:
    """Global engineering settings. These are not per-case clinical cutoffs."""

    air_hu: float = -500.0
    bone_hu: float = 200.0
    seed_prominence_mm: float = 0.8
    plate_half_width_mm: float = 1.5
    separator_support_mm: float = 1.5
    separator_support_fraction: float = 0.60
    minimum_enclosure_fraction: float = 0.55
    maximum_roi_boundary_fraction: float = 0.25
    minimum_air_coverage_fraction: float = 0.50
    boundary_growth_mm: float = 3.0
    minimum_voxels: int = 4
    max_instances: int = 65535

    @classmethod
    def parse(cls, value: dict[str, Any] | "CountingConfig" | None) -> "CountingConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise CountingError("INVALID_CONFIG", "Counting configuration must be an object.")
        # A protocol snapshot may wrap the fixed algorithm parameters.
        if "algorithm" in value:
            value = value["algorithm"]
        allowed = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or set(value) - allowed:
            raise CountingError("INVALID_CONFIG", "Counting configuration has unsupported parameters.")
        try:
            result = cls(**value)
            numbers = asdict(result)
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in numbers.values()):
                raise ValueError
            if not -1000 < result.air_hu < 0 or not 50 <= result.bone_hu <= 1000:
                raise ValueError
            if not 0 < result.seed_prominence_mm <= 5 or not 0 < result.plate_half_width_mm <= 5:
                raise ValueError
            if not 0 < result.separator_support_mm <= 5:
                raise ValueError
            if not 0 <= result.boundary_growth_mm <= 5:
                raise ValueError
            if any(not 0 < v <= 1 for v in (result.separator_support_fraction, result.minimum_enclosure_fraction, result.maximum_roi_boundary_fraction, result.minimum_air_coverage_fraction)):
                raise ValueError
            if int(result.minimum_voxels) != result.minimum_voxels or result.minimum_voxels < 1:
                raise ValueError
            if int(result.max_instances) != result.max_instances or not 1 <= result.max_instances <= 65535:
                raise ValueError
            return result
        except (TypeError, ValueError):
            raise CountingError("INVALID_CONFIG", "Counting configuration values are invalid.") from None


def geometry(affine: np.ndarray) -> np.ndarray:
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CountingError("INVALID_GEOMETRY", "The volume affine is invalid.")
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(spacing <= 0) or not np.allclose((matrix[:3, :3] / spacing).T @ (matrix[:3, :3] / spacing), np.eye(3), atol=1e-4):
        raise CountingError("INVALID_GEOMETRY", "Sheared or degenerate grids require a validated conversion.")
    return spacing


def _bone_support_radius_mm(spacing: np.ndarray) -> float:
    return min(max(float(spacing.max()) * 1.01, 0.8), 1.01)


def _shift(mask: np.ndarray, axis: int, offset: int) -> np.ndarray:
    out = np.zeros_like(mask)
    source = [slice(None)] * 3
    target = [slice(None)] * 3
    source[axis] = slice(None, -offset) if offset > 0 else slice(-offset, None)
    target[axis] = slice(offset, None) if offset > 0 else slice(None, offset)
    out[tuple(target)] = mask[tuple(source)]
    return out


def _merge_unsupported_separators(labels: np.ndarray, free: np.ndarray, bone: np.ndarray, spacing: np.ndarray, config: CountingConfig, *, trace: dict | None = None) -> tuple[np.ndarray, set[int]]:
    """Retain watershed splits only near a thin bone plate with lumen on both sides.

    A distance saddle alone is insufficient evidence for two cells. A plate may
    be incomplete, but bone that only borders exterior tissue is not a septum.
    The resulting split remains an uncertain engineering inference.
    """
    n = int(labels.max())
    parent = np.arange(n + 1)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    evidence: dict[tuple[int, int], list[int]] = {}
    for axis in range(3):
        step = max(1, int(np.ceil(config.plate_half_width_mm / spacing[axis])))
        plate = bone & _shift(free, axis, step) & _shift(free, axis, -step)
        near_plate = ndi.distance_transform_edt(~plate, sampling=spacing) <= config.separator_support_mm if plate.any() else np.zeros_like(bone)
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(low)], labels[tuple(high)]
        boundary = (a != b) & (a > 0) & (b > 0)
        if not boundary.any():
            continue
        pairs = np.sort(np.column_stack((a[boundary], b[boundary])), axis=1)
        support = (near_plate[tuple(low)] | near_plate[tuple(high)])[boundary]
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        totals = np.bincount(inverse)
        backed = np.bincount(inverse, weights=support.astype(np.uint8))
        for pair, total, count in zip(unique, totals, backed):
            key = (int(pair[0]), int(pair[1]))
            # Use the dominant interface normal. Stair-step side faces around
            # an aperture are not independent evidence against its septum.
            if key not in evidence or total > evidence[key][0]:
                evidence[key] = [int(total), int(count), axis]
    preserved = {pair for pair, (total, support, _) in evidence.items() if support / total >= config.separator_support_fraction}
    # A communication around a third basin must not undo an already supported
    # septum. Preserve these cannot-merge constraints during graph contraction.
    unsupported = [(support / total, a, b) for (a, b), (total, support, _) in evidence.items() if (a, b) not in preserved]
    merge_decisions = [] if trace is not None else None
    for _, a, b in sorted(unsupported):
        left, right = root(a), root(b)
        if left == right:
            continue
        if any({root(u), root(v)} == {left, right} for u, v in preserved):
            if merge_decisions is not None:
                merge_decisions.append({"basins": [a, b], "merged": False, "reason": "supported_septum_constraint"})
            continue
        parent[max(left, right)] = min(left, right)
        if merge_decisions is not None:
            merge_decisions.append({"basins": [a, b], "merged": True, "reason": "insufficient_separator_support"})
    mapping = np.array([root(i) for i in range(n + 1)], dtype=np.int32)
    uncertain = {int(mapping[i]) for pair in preserved for i in pair if mapping[pair[0]] != mapping[pair[1]]}
    if trace is not None:
        trace["interfaces"] = [{"basins": list(pair), "dominant_axis": axis,
            "interface_faces": total, "supported_faces": support,
            "support_fraction": support / total, "preserved": pair in preserved}
            for pair, (total, support, axis) in sorted(evidence.items())]
        trace["merge_decisions"] = merge_decisions
        trace["basin_to_merged_label"] = mapping.tolist()
    return mapping[labels], uncertain


def compartment_stage(hu: np.ndarray, roi: np.ndarray, affine: np.ndarray, config: dict[str, Any] | CountingConfig | None = None, *, midline_ras_x: float = 0.0, variant_regions: dict[str, np.ndarray] | None = None, excluded: np.ndarray | None = None, reconstruction_region: np.ndarray | None = None, trace: dict | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    """Propose compartments in an already localized region.

    This low-level method supports synthetic validation. Real-volume entry points
    always obtain the region automatically through the learned localizers.
    """
    cfg = CountingConfig.parse(config)
    values = np.asarray(hu)
    region = np.asarray(roi, dtype=bool)
    if values.ndim != 3 or values.shape != region.shape or not np.isfinite(values).all():
        raise CountingError("INVALID_VOLUME", "A finite three-dimensional HU volume and matching region are required.")
    spacing = geometry(affine)
    if not region.any():
        raise CountingError("EMPTY_REGION", "Automatic anatomical localization produced no region.")
    search = region if reconstruction_region is None else np.asarray(reconstruction_region, dtype=bool)
    if search.shape != region.shape or np.any(region & ~search):
        raise CountingError("INVALID_GEOMETRY", "The reconstruction region must contain the anatomical membership region.")
    bone = values >= cfg.bone_hu
    free = search & ~bone
    if excluded is not None:
        if excluded.shape != region.shape:
            raise CountingError("INVALID_GEOMETRY", "The exclusion geometry does not match the source.")
        free &= ~excluded
    if not free.any():
        raise CountingError("NO_COMPARTMENTS", "No non-bone compartment can be resolved in the localized region.")
    # Padding is essential: the edge of a processing crop is not open-ended infinity.
    distance = ndi.distance_transform_edt(np.pad(free, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    maxima = h_maxima(distance.astype(np.float32), cfg.seed_prominence_mm) & free
    if trace is not None:
        trace["seed_counts"] = {"free_maxima_voxels": int(maxima.sum()), "air_maxima_voxels": 0}
    # Thick soft-tissue contents can shift a free-space distance maximum away
    # from a small aerated chamber. Air-lumen maxima add detection seeds only;
    # the final cavity and split/merge decisions still use bone evidence.
    air_lumen = free & (values < cfg.air_hu)
    air_distance = ndi.distance_transform_edt(np.pad(air_lumen, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    if air_lumen.any():
        air_maxima = h_maxima(air_distance.astype(np.float32), cfg.seed_prominence_mm) & air_lumen
        maxima |= air_maxima
        if trace is not None:
            trace["seed_counts"]["air_maxima_voxels"] = int(air_maxima.sum())
    seeds, _ = ndi.label(maxima, structure=ndi.generate_binary_structure(3, 1))
    initial_markers = int(seeds.max())
    components, ncomponents = ndi.label(free)
    # Small but bone-resolved enclosed spaces must not vanish solely because their
    # radius is below the seed prominence setting.
    seeded = set(np.unique(components[seeds > 0]).tolist())
    next_seed = int(seeds.max()) + 1
    for component, bounds in enumerate(ndi.find_objects(components), start=1):
        if component not in seeded:
            if bounds is None:
                continue
            positions = np.flatnonzero((components[bounds] == component).ravel())
            if positions.size:
                best = positions[np.argmax(distance[bounds].ravel()[positions])]
                local = np.unravel_index(best, components[bounds].shape)
                position = tuple(local[i] + bounds[i].start for i in range(3))
                seeds[position] = next_seed
                next_seed += 1
    del components
    labels = watershed(-distance, markers=seeds, mask=free, connectivity=1).astype(np.int32)
    if trace is not None:
        trace["arrays"] = {"membership": region, "excluded": np.zeros_like(region) if excluded is None else excluded,
            "search": search, "markers": seeds, "watershed": labels, "affine": np.asarray(affine)}
        trace["seed_counts"].update(initial_markers=initial_markers, component_fallbacks=int(seeds.max()) - initial_markers)
    labels, inferred_split = _merge_unsupported_separators(labels, free, bone, spacing, cfg, trace=trace)
    if trace is not None:
        trace["arrays"]["merged"] = labels
        trace["proposals"] = []
    # A coarse sampling interval must not make distant bone look like enclosure.
    bone_near = ndi.distance_transform_edt(~bone, sampling=spacing) <= _bone_support_radius_mm(spacing)
    roi_edge = search & ~ndi.binary_erosion(search)
    proposals: list[tuple[tuple[slice, ...], np.ndarray, dict[str, Any]]] = []
    proposal_records = {} if trace is not None else None
    rejected = 0
    for value, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        mask = labels[bounds] == value
        size = int(mask.sum())
        # A one-voxel-wide pore is not a resolvable 3D cavity on this grid.
        resolved_interior = float(distance[bounds][mask].max()) > float(spacing.max()) * 1.01
        if trace is not None:
            record = {"merged_label": value, "voxel_count": size,
                "maximum_interior_radius_mm": float(distance[bounds][mask].max()), "rejection": None}
            trace["proposals"].append(record)
            proposal_records[value] = record
        if size < cfg.minimum_voxels or not resolved_interior:
            if trace is not None:
                record["rejection"] = "minimum_size" if size < cfg.minimum_voxels else "unresolved_interior"
            rejected += 1
            continue
        membership_fraction = float(np.count_nonzero(mask & region[bounds])) / size
        if trace is not None:
            record["membership_fraction"] = membership_fraction
        if membership_fraction < 0.5:
            if trace is not None:
                record["rejection"] = "membership"
            rejected += 1
            continue
        surface = mask & ~ndi.binary_erosion(mask)
        surface_count = int(surface.sum())
        enclosure = float(np.count_nonzero(surface & bone_near[bounds])) / max(1, surface_count)
        roi_contact = float(np.count_nonzero(surface & roi_edge[bounds])) / max(1, surface_count)
        unsupported_roi_contact = float(np.count_nonzero(surface & roi_edge[bounds] & ~bone_near[bounds])) / max(1, surface_count)
        air_count = int(np.count_nonzero(mask & (values[bounds] < cfg.air_hu)))
        # An arbitrary ROI face must not manufacture an opaque anatomical cavity.
        if enclosure < cfg.minimum_enclosure_fraction or unsupported_roi_contact > cfg.maximum_roi_boundary_fraction:
            if trace is not None:
                record.update(rejection="bony_enclosure" if enclosure < cfg.minimum_enclosure_fraction else "unsupported_roi_boundary",
                    enclosure_fraction=enclosure, unsupported_roi_contact_fraction=unsupported_roi_contact)
            rejected += 1
            continue
        voxel_centroid = np.array(ndi.center_of_mass(mask)) + np.array([sl.start for sl in bounds])
        centroid = nib.affines.apply_affine(affine, voxel_centroid)
        reasons: list[str] = []
        if int(value) in inferred_split:
            reasons.append("incomplete_septum_inferred_from_bone_plate")
        internal_plate = np.zeros_like(mask)
        for axis in range(3):
            step = max(1, int(np.ceil(cfg.plate_half_width_mm / spacing[axis])))
            internal_plate |= bone[bounds] & _shift(mask, axis, step) & _shift(mask, axis, -step)
        if np.count_nonzero(internal_plate) >= max(4, int(surface_count * 0.02)):
            reasons.append("possible_internal_septa_unresolved_merge_requires_review")
        if air_count == 0:
            reasons.append("fully_opacified_candidate_requires_review")
        elif air_count < size * 0.8:
            reasons.append("partly_opacified_candidate")
        if enclosure < 0.85:
            reasons.append("incomplete_bony_enclosure")
        if roi_contact > 0:
            reasons.append("touches_inferred_anatomical_boundary")
        if membership_fraction < 1:
            reasons.append("cavity_reconstructed_beyond_membership_boundary")
        if size < 20:
            reasons.append("small_resolution_limited_candidate")
        if abs(float(centroid[0]) - midline_ras_x) <= float(np.abs(affine[0, :3]).sum()):
            reasons.append("midline_laterality_requires_review")
        positions = np.where(mask)
        physical_x = np.full(size, float(affine[0, 3]))
        for axis in range(3):
            physical_x += affine[0, axis] * (positions[axis] + bounds[axis].start)
        if physical_x.min() < midline_ras_x < physical_x.max():
            reasons.append("crosses_midline_centroid_laterality_requires_review")
        for name, extension in sorted((variant_regions or {}).items()):
            if np.any(mask & extension[bounds]):
                reasons.append(f"possible_{name}_variant_requires_review")
        cell = {
            "id": 0, "side": "left" if centroid[0] <= midline_ras_x else "right",
            "centroid_ras_mm": [float(v) for v in centroid], "origin": "automatic",
            "included": True, "uncertain": bool(reasons), "reasons": reasons,
            "voxel_count": size, "air_voxel_count": air_count,
        }
        proposals.append((bounds, mask, cell))
        if trace is not None:
            record.update(enclosure_fraction=enclosure, unsupported_roi_contact_fraction=unsupported_roi_contact)
    if len(proposals) > cfg.max_instances:
        raise CountingError("INSTANCE_LIMIT", "The number of proposals exceeds the instance format limit.")
    # Physical coordinates, not array traversal or NIfTI axis signs, determine IDs.
    proposals.sort(key=lambda item: tuple(round(v, 6) for v in item[2]["centroid_ras_mm"]))
    cavities = np.zeros(values.shape, dtype=np.uint16)
    cells: list[dict[str, Any]] = []
    for index, (bounds, mask, cell) in enumerate(proposals, start=1):
        cavities[bounds][mask] = index
        cell["id"] = index
        cells.append(cell)
        if trace is not None:
            merged_label = int(labels[bounds][mask][0])
            proposal_records[merged_label]["cell_id"] = index
    air = np.where(values < cfg.air_hu, cavities, 0).astype(np.uint16)
    warnings = ["Unvalidated automatic compartment proposals; clinician reference counts are not yet available."]
    observed_air = int(np.count_nonzero(region & (values < cfg.air_hu)))
    assigned_air = int(np.count_nonzero((air > 0) & region))
    if observed_air:
        warnings.append(f"Assigned {100.0 * assigned_air / observed_air:.1f}% of aerated voxels in the inferred search region to accepted candidates; this is a geometric coverage diagnostic, not clinical sensitivity.")
    if observed_air and assigned_air / observed_air < 0.5:
        warnings.append("Low geometric coverage: substantial aerated structure remains unresolved. The candidate total must not be represented as a complete anatomical count.")
    if rejected:
        warnings.append(f"{rejected} geometric candidates were rejected for insufficient size, bony support, or excessive region-edge contact; unresolved cells may remain.")
    if not cells:
        raise CountingError("NO_SUPPORTED_CELLS", "No sufficiently bone-supported cell candidates could be resolved; this is not a zero-cell count.")
    return cavities, air, cells, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_volume(volume_path: str | Path, output_dir: str | Path, landmarks_dir: str | Path | None = None, config: dict[str, Any] | CountingConfig | None = None, *, acquisition: dict[str, Any] | None = None, _strategy: Any = None, diagnostic_stages: bool = False) -> dict[str, Any]:
    """Run a fully automatic first count and persist a source-grid result.

    Existing output directories must be empty; immutable prior results are never
    overwritten. Localizer inputs are optional *machine-generated* cache masks,
    not a human-prepared ethmoid ROI or seeds.
    """
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise CountingError("OUTPUT_EXISTS", "Counting output must be a fresh or empty directory.")
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1, "status": "failed", "method_version": METHOD_VERSION,
        "counting_engine_version": METHOD_VERSION,
        "source_sha256": None, "protocol_version": PROTOCOL_VERSION, "config": {},
        "counts": {"left": None, "right": None, "total": None}, "cells": [],
        "warnings": [], "artifacts": {}, "diagnostics": {}, "localizer_provenance": {},
        "acquisition": {}, "quality": {}, "localization_geometry": {},
    }
    try:
        from .localization import localize_native, cache_producer_summary, cache_producer_warnings, load_landmarks, run_localizers
        from .grids import check_proxy_limit, localization_grid
        from .quality import acquisition_snapshot, resolution_quality

        if _strategy is not None:
            result.update(_strategy.initial_metadata())
        if isinstance(config, dict) and "protocol_version" in config and config["protocol_version"] != result["protocol_version"]:
            raise CountingError("INVALID_CONFIG", "This worker cannot execute a different frozen protocol version.")
        cfg = CountingConfig.parse(config)
        result["config"] = asdict(cfg)
        source = Path(volume_path)
        if not source.is_file():
            raise CountingError("SOURCE_UNAVAILABLE", "The source volume is unavailable.")
        result["source_sha256"] = _sha256(source)
        try:
            if acquisition is not None and (not isinstance(acquisition, dict) or acquisition.get("source_sha256") != result["source_sha256"]):
                raise ValueError("Acquisition snapshot must bind the exact source")
            result["acquisition"] = acquisition_snapshot(acquisition, result["source_sha256"])
        except ValueError:
            raise CountingError("INVALID_ACQUISITION", "The source-bound acquisition metadata is invalid.") from None
        result["quality"] = resolution_quality(result["acquisition"])
        image = nib.load(source)
        if len(image.shape) != 3:
            raise CountingError("INVALID_VOLUME", "Only one three-dimensional CT volume is supported.")
        spacing = geometry(image.affine)
        result["quality"] = resolution_quality(result["acquisition"], image.affine)
        canonical = nib.as_closest_canonical(image)
        result["localization_geometry"] = localization_grid(canonical.shape, canonical.affine)
        if _strategy is None:
            check_proxy_limit(result["localization_geometry"])
        # No interpolation or sharpening is used to invent resolution.
        hu = canonical.get_fdata(dtype=np.float32)
        if not np.isfinite(hu).all() or float(np.percentile(hu, 0.1)) > -300 or float(np.percentile(hu, 99.9)) < 250:
            raise CountingError("HU_RANGE_INVALID", "Expected calibrated CT air and bone values were not found.")
        if _strategy is not None:
            localization = _strategy.localize(source, image, canonical, hu, destination, landmarks_dir, cfg, result)
        elif landmarks_dir is None:
            landmarks_dir = destination / "localizers"
            run_localizers(source, Path(landmarks_dir))
        if _strategy is None:
            retried_extent = False
            try:
                landmarks = load_landmarks(Path(landmarks_dir), canonical, source_sha256=result["source_sha256"])
                if (Path(landmarks_dir) / "localizers.local.json").is_file():
                    result["localizer_provenance"] = cache_producer_summary(Path(landmarks_dir))
                localization = localize_native(hu, canonical.affine, landmarks, cfg, result["localization_geometry"])
            except CountingError as error:
                if error.code not in {"LANDMARK_EMPTY", "TARGET_CLIPPED"}:
                    raise
                # One full-source-field retry can recover anatomy omitted by the
                # rough crop. A genuinely clipped acquisition still fails afterward.
                retry_directory = destination / "localizers-full-extent"
                run_localizers(source, retry_directory, full_extent=True)
                landmarks = load_landmarks(retry_directory, canonical, source_sha256=result["source_sha256"])
                if (retry_directory / "localizers.local.json").is_file():
                    result["localizer_provenance"] = cache_producer_summary(retry_directory)
                localization = localize_native(hu, canonical.affine, landmarks, cfg, result["localization_geometry"])
                landmarks_dir = retry_directory
                retried_extent = True
            if retried_extent:
                localization.warnings.append("Automatic anatomical localization required one full-field retry after missing or clipped landmark evidence.")
            cache_manifest = Path(landmarks_dir) / "localizers.local.json" if landmarks_dir else None
            if cache_manifest is not None and cache_manifest.is_file():
                localization.warnings.extend(cache_producer_warnings(Path(landmarks_dir)))
        region = localization.region
        if _strategy is not None:
            _strategy.stage("counting")
        canonical_spacing = geometry(canonical.affine)
        search = ndi.distance_transform_edt(~region, sampling=canonical_spacing) <= cfg.boundary_growth_mm
        search &= ~localization.excluded
        search |= region
        indices = np.where(search)
        # Keep all source bone within the physical support radius on fine grids.
        padding = [max(2, int(np.ceil(_bone_support_radius_mm(canonical_spacing) / step))) for step in canonical_spacing]
        crop = tuple(slice(max(0, int(a.min()) - padding[i]), min(region.shape[i], int(a.max()) + padding[i] + 1)) for i, a in enumerate(indices))
        start = np.array([sl.start for sl in crop])
        crop_affine = canonical.affine.copy()
        crop_affine[:3, 3] = nib.affines.apply_affine(canonical.affine, start)
        trace = {} if diagnostic_stages else None
        try:
            cavities_crop, air_crop, cells, warnings = compartment_stage(
                hu[crop], region[crop], crop_affine, cfg,
                midline_ras_x=localization.midline_ras_x,
                variant_regions={name: mask[crop] for name, mask in localization.variants.items()},
                excluded=localization.excluded[crop], reconstruction_region=search[crop], trace=trace,
            )
        finally:
            if trace is not None and "arrays" in trace:
                # Explicit opt-in only. These protected sidecars never enter public artifacts.
                np.savez_compressed(destination / "diagnostic-stages.local.npz", **trace.pop("arrays"))
                trace.update(schema_version=1, source_sha256=result["source_sha256"], config=result["config"], engine_sha256=_sha256(Path(__file__)),
                    method_version=result["method_version"], protocol_version=result["protocol_version"],
                    crop_bounds=[[s.start, s.stop] for s in crop], midline_ras_x=localization.midline_ras_x,
                    coordinate_system="cropped canonical voxels with physical RAS affine; not native array indices")
                (destination / "diagnostic-stages.local.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        cavities = np.zeros(hu.shape, dtype=np.uint16)
        air = np.zeros(hu.shape, dtype=np.uint16)
        cavities[crop], air[crop] = cavities_crop, air_crop
        if _strategy is not None:
            _strategy.assign_sides(cavities, cells)
        physical_x = (canonical.affine[0, 0] * np.arange(hu.shape[0])[:, None, None]
                      + canonical.affine[0, 1] * np.arange(hu.shape[1])[None, :, None]
                      + canonical.affine[0, 2] * np.arange(hu.shape[2])[None, None, :]
                      + canonical.affine[0, 3])
        region_labels = np.where(region | (cavities > 0), np.where(physical_x <= localization.midline_ras_x, 1, 2), 0).astype(np.uint8)
        for cell in cells:
            region_labels[cavities == cell["id"]] = 1 if cell["side"] == "left" else 2
            if result["quality"]["flags"]:
                cell["uncertain"] = True
                cell["reasons"].extend(result["quality"]["flags"])
        orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine), nib.orientations.io_orientation(image.affine))
        for name, array in (("cavities", cavities), ("air", air), ("region", region_labels)):
            native = nib.orientations.apply_orientation(array, orientation)
            header = image.header.copy()
            header.set_data_dtype(np.uint8 if name == "region" else np.uint16)
            header.set_slope_inter(1, 0)
            out = nib.Nifti1Image(native, image.affine, header=header)
            out.set_qform(image.affine, code=1)
            out.set_sform(image.affine, code=1)
            nib.save(out, destination / f"{name}.nii.gz")
        result["warnings"] = localization.warnings + warnings
        observed_air = int(np.count_nonzero(region & (hu < cfg.air_hu)))
        assigned_air = int(np.count_nonzero((air > 0) & region))
        coverage = assigned_air / observed_air if observed_air else None
        result["diagnostics"] = {"observed_air_voxels": observed_air, "assigned_air_voxels": assigned_air, "air_coverage_fraction": coverage}
        if coverage is not None and coverage < cfg.minimum_air_coverage_fraction:
            # Keep proposals only in the protected work directory for engineering
            # review. The public result has no count, registry, or artifact links.
            (destination / "diagnostic-candidates.local.json").write_text(json.dumps({"cells": cells, "clinical_count_available": False}, indent=2), encoding="utf-8")
            raise CountingError("INSUFFICIENT_COVERAGE", "Substantial aerated anatomy remains unresolved. An anatomical cell count is unavailable; protected diagnostic proposals require review.")
        result.update(
            status="completed", cells=cells,
            counts={"left": sum(c["side"] == "left" for c in cells), "right": sum(c["side"] == "right" for c in cells), "total": len(cells)},
            warnings=localization.warnings + warnings,
            artifacts={"cavities": "cavities.nii.gz", "air": "air.nii.gz", "region": "region.nii.gz"},
        )
        if _strategy is not None:
            _strategy.verify_artifacts(source, destination, result)
    except CountingError as error:
        result.update(failure_code=error.code, message=str(error))
    except (OSError, ValueError, nib.filebasedimages.ImageFileError):
        result.update(failure_code="VOLUME_OR_OUTPUT_INVALID", message="The local source, landmarks, or output could not be read or written.")
    except Exception:
        # Raw third-party exception text may contain paths or image metadata.
        (destination / "engine-error.local.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result.update(failure_code="ENGINE_ERROR", message="The local counting engine failed; inspect protected local runtime diagnostics.")
    if _strategy is not None:
        _strategy.finalize(result)
    result["warnings"] = result["quality"].get("warnings", []) + result["warnings"]
    temporary = destination / ".result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination / "result.json")
    return result
