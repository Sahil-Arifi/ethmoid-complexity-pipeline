"""Source-bound native boundary labels and local Slicer review interchange.

Unknown voxels stay -1. No separator labels are inferred from core edges.
All source volumes, labels and releases must remain outside Git workspaces.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import re
import shutil
import zipfile

import nibabel as nib
import numpy as np

from . import instance_experiment as legacy
from .boundary_io import new_output

VERSION = "boundary-instance-v1"
SAMPLE_FIELDS = {"sample_id", "patient_group", "scope", "source", "annotation",
                 "protocol_sha256", "provenance"}
ARRAY_FIELDS = {"foreground_labels", "separator_labels", "instance_ids",
                "evaluation_region", "affine"}
HEADS = ("foreground_labels", "separator_labels")
SCOPES = {"synthetic_truth", "conditional_native", "adjudicated_expert"}
MAX_CROP_VOXELS = 64_000_000
MAX_RELEASE_VOXELS = 128_000_000
MAX_SOURCE_VOXELS = 128_000_000
ExperimentError = legacy.ExperimentError
require = legacy.require


def _token(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _sample_record(sample):
    require(isinstance(sample, dict) and set(sample) == SAMPLE_FIELDS, "BOUNDARY_SAMPLE_SCHEMA_INVALID")
    require(_token(sample["sample_id"]) and _token(sample["patient_group"]), "BOUNDARY_SAMPLE_ID_INVALID")
    require(isinstance(sample["scope"], str) and sample["scope"] in SCOPES
            and _digest(sample["protocol_sha256"]), "BOUNDARY_SCOPE_PROTOCOL_INVALID")
    provenance = sample["provenance"]
    require(isinstance(provenance, dict) and provenance.get("kind") == sample["scope"],
            "BOUNDARY_PROVENANCE_INVALID")
    if sample["scope"] == "synthetic_truth":
        require(set(provenance) == {"kind", "generator"} and isinstance(provenance["generator"], str)
                and 0 < len(provenance["generator"].strip()) <= 256,
                "SYNTHETIC_PROVENANCE_REQUIRED")
    elif sample["scope"] == "conditional_native":
        require(set(provenance) == {"kind", "review_status"} and provenance["review_status"] == "conditional",
                "CONDITIONAL_PROVENANCE_REQUIRED")
    else:
        require(set(provenance) == {"kind", "annotator", "reviewer", "review_status", "independent_review"}
                and _token(provenance["annotator"]) and _token(provenance["reviewer"])
                and provenance["annotator"] != provenance["reviewer"]
                and provenance["review_status"] == "adjudicated" and provenance["independent_review"] is True,
                "INDEPENDENT_ADJUDICATED_REVIEW_REQUIRED")
    annotation = sample["annotation"]
    require(isinstance(annotation, dict) and set(annotation) == {"path", "sha256"}
            and _digest(annotation["sha256"]), "BOUNDARY_ANNOTATION_REQUIRED")
    path = legacy.external(annotation["path"])
    require(path.is_file() and legacy.sha256(path) == annotation["sha256"], "ANNOTATION_HASH_MISMATCH")
    try:
        source = legacy.source_record(sample["source"])
    except ExperimentError:
        raise
    except (OSError, ValueError, TypeError):
        raise ExperimentError("BOUNDARY_SOURCE_LOAD_FAILED") from None
    return {**deepcopy(sample), "source": source,
            "annotation": {"path": str(path), "sha256": annotation["sha256"]}}


def _arrays(values, shape, affine):
    require(set(values) == ARRAY_FIELDS, "BOUNDARY_ANNOTATION_ARRAYS_INVALID")
    for key in (*HEADS, "instance_ids", "evaluation_region"):
        require(values[key].shape == shape, "BOUNDARY_ANNOTATION_SHAPE_INVALID")
    for key in HEADS:
        require(values[key].dtype.kind == "i" and np.isin(values[key], [-1, 0, 1]).all(),
                "BOUNDARY_HEAD_VALUES_INVALID")
    instances = values["instance_ids"]
    require(instances.dtype.kind == "i" and np.all(instances >= -1)
            and np.all(instances <= np.iinfo(np.int32).max), "BOUNDARY_INSTANCE_VALUES_INVALID")
    require(values["evaluation_region"].dtype == np.dtype(bool), "BOUNDARY_EVALUATION_REGION_INVALID")
    supplied_affine = values["affine"]
    require(supplied_affine.shape == (4, 4) and supplied_affine.dtype.kind in "fiu"
            and np.isfinite(supplied_affine).all()
            and np.allclose(supplied_affine, affine, rtol=0, atol=1e-6), "BOUNDARY_AFFINE_MISMATCH")
    foreground = values["foreground_labels"]
    require(not np.any((instances > 0) & (foreground == 0))
            and not np.any((instances == 0) & (foreground == 1)), "BOUNDARY_ANNOTATION_CONTRADICTION")
    require(not np.any(values["evaluation_region"] & ((instances < 0) | (foreground < 0))),
            "BOUNDARY_EVALUATION_REGION_INCOMPLETE")
    return {key: np.asarray(value).copy() for key, value in values.items()}


def _bounded_shape(source):
    # Canonical reorientation in the legacy loader may materialize the source.
    try:
        image = nib.load(source["input"])
    except (OSError, ValueError, nib.filebasedimages.ImageFileError):
        raise ExperimentError("BOUNDARY_SOURCE_LOAD_FAILED") from None
    require(len(image.shape) == 3 and math.prod(image.shape) <= MAX_SOURCE_VOXELS,
            "BOUNDARY_SOURCE_VOXEL_BUDGET_EXCEEDED")
    shape = tuple(b - a for a, b in source["crop_bounds"])
    require(math.prod(shape) <= MAX_CROP_VOXELS, "BOUNDARY_CROP_VOXEL_BUDGET_EXCEEDED")
    return shape


def _load_archive(path, shape):
    # Inspect bounded NPY headers before np.load can allocate decompressed arrays.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(ARRAY_FIELDS) and set(names) == {f"{key}.npy" for key in ARRAY_FIELDS},
                "BOUNDARY_ANNOTATION_ARRAYS_INVALID")
        for info in archive.infolist():
            expected = (4, 4) if info.filename == "affine.npy" else shape
            require(info.file_size <= math.prod(expected) * 8 + 10000, "BOUNDARY_ARCHIVE_BUDGET_EXCEEDED")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in {(1, 0), (2, 0)}, "BOUNDARY_ARCHIVE_FORMAT_INVALID")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                array_shape, _, dtype = reader(stream)
                require(tuple(array_shape) == expected, "BOUNDARY_ANNOTATION_SHAPE_INVALID")
                require(not dtype.hasobject and dtype.kind in "bifu" and dtype.itemsize <= 8,
                        "BOUNDARY_ARCHIVE_DTYPE_INVALID")
                require(info.file_size == stream.tell() + math.prod(expected) * dtype.itemsize,
                        "BOUNDARY_ARCHIVE_SIZE_INVALID")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_sample(sample):
    """Validate hashes/geometry and return float32 HU, crop affine and native arrays."""
    record = _sample_record(sample)
    try:
        shape = _bounded_shape(record["source"])
        values = _load_archive(record["annotation"]["path"], shape)
        _, _, hu, affine = legacy.load_crop(record["source"])
        return hu, affine, _arrays(values, hu.shape, affine)
    except ExperimentError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, nib.filebasedimages.ImageFileError):
        raise ExperimentError("BOUNDARY_SAMPLE_LOAD_FAILED") from None


def _head_counts(array):
    return {"unknown": int(np.sum(array == -1)), "negative": int(np.sum(array == 0)),
            "positive": int(np.sum(array == 1))}


def validate_release(manifest_dict):
    """Validate an exact dataset manifest; return records and a path-free readiness report."""
    require(isinstance(manifest_dict, dict) and set(manifest_dict) == {"schema_version", "version", "samples"}
            and type(manifest_dict["schema_version"]) is int and manifest_dict["schema_version"] == 1
            and manifest_dict["version"] == VERSION and isinstance(manifest_dict["samples"], list),
            "BOUNDARY_RELEASE_SCHEMA_INVALID")
    samples, rows, seen_ids, source_groups, source_crops = [], [], set(), {}, set()
    totals = {head: Counter(unknown=0, negative=0, positive=0) for head in HEADS}
    scopes = Counter()
    total_voxels = 0
    for sample in manifest_dict["samples"]:
        record = _sample_record(sample)
        identity, group = record["sample_id"], record["patient_group"]
        require(identity not in seen_ids, "DUPLICATE_SAMPLE_ID")
        seen_ids.add(identity)
        source_hash = record["source"]["source_sha256"]
        require(source_groups.get(source_hash, group) == group, "SOURCE_PATIENT_GROUP_CONFLICT")
        source_groups[source_hash] = group
        crop_key = (source_hash, tuple(tuple(pair) for pair in record["source"]["crop_bounds"]))
        require(crop_key not in source_crops, "DUPLICATE_SOURCE_CROP")
        source_crops.add(crop_key)
        total_voxels += math.prod(_bounded_shape(record["source"]))
        require(total_voxels <= MAX_RELEASE_VOXELS, "BOUNDARY_RELEASE_VOXEL_BUDGET_EXCEEDED")
        samples.append(record)
    for record in samples:
        identity, group = record["sample_id"], record["patient_group"]
        _, _, arrays = load_sample(record)
        counts = {head: _head_counts(arrays[head]) for head in HEADS}
        for head in HEADS:
            totals[head].update(counts[head])
        rows.append({"sample_id": identity, "patient_group": group, "scope": record["scope"],
                     "head_counts": counts, "instance_count": int(len(np.unique(arrays["instance_ids"][arrays["instance_ids"] > 0]))),
                     "known_instance_voxels": int(np.sum(arrays["instance_ids"] >= 0)),
                     "evaluation_voxels": int(np.sum(arrays["evaluation_region"]))})
        scopes[record["scope"]] += 1
    groups = sorted({sample["patient_group"] for sample in samples})
    missing = [f"{head.upper()}_{label.upper()}_MISSING" for head in HEADS
               for label in ("negative", "positive") if not totals[head][label]]
    if not samples:
        missing.insert(0, "SAMPLES_MISSING")
    training_groups = groups[:-1] if len(groups) > 1 else groups
    training_counts = {head: Counter(unknown=0, negative=0, positive=0) for head in HEADS}
    validation_counts = {head: Counter(unknown=0, negative=0, positive=0) for head in HEADS}
    for row in rows:
        if row["patient_group"] in training_groups:
            for head in HEADS:
                training_counts[head].update(row["head_counts"][head])
        elif len(groups) > 1:
            for head in HEADS:
                validation_counts[head].update(row["head_counts"][head])
    fold_missing = [f"TRAINING_{head.upper()}_{label.upper()}_MISSING" for head in HEADS
                    for label in ("negative", "positive") if not training_counts[head][label]]
    validation_missing = [f"HELDOUT_{head.upper()}_KNOWN_SUPERVISION_MISSING" for head in HEADS
                          if len(groups) > 1 and not (validation_counts[head]["negative"] + validation_counts[head]["positive"])]
    reasons = list(dict.fromkeys(missing + fold_missing + validation_missing))
    if len(groups) == 1:
        reasons.append("SINGLE_PATIENT_GROUP_MECHANISM_ONLY")
    training_allowed = bool(samples) and not missing and not fold_missing and not validation_missing
    status = "NOT_READY" if not training_allowed else (
        "MECHANISM_ONLY" if len(groups) == 1 else "READY_GROUP_HOLDOUT")
    return {"samples": samples, "readiness": {"version": VERSION, "status": status,
        "training_allowed": training_allowed, "validation_available": len(groups) > 1 and not validation_missing,
        "holdout_group_available": len(groups) > 1,
        "reasons": reasons, "sample_count": len(samples), "patient_groups": groups,
        "patient_group_count": len(groups), "scope_counts": dict(scopes), "samples": rows,
        "head_counts": {head: dict(totals[head]) for head in HEADS},
        "group_holdout": {"training_groups": training_groups,
            "validation_groups": groups[-1:] if len(groups) > 1 else [],
            "training_head_counts": {head: dict(training_counts[head]) for head in HEADS},
            "training_missing_labels": fold_missing,
            "validation_head_counts": {head: dict(validation_counts[head]) for head in HEADS},
            "validation_missing_labels": validation_missing},
        "clinical_validation_established": False}}


def readiness(manifest_path, output_path):
    """Write a fresh external readiness report without copying patient artifacts."""
    manifest_path = legacy.external(manifest_path)
    result = validate_release(legacy.read_json(manifest_path))["readiness"]
    result["manifest_sha256"] = legacy.sha256(manifest_path)
    output = new_output(output_path)
    legacy.write_json(output / "readiness.local.json", result)
    return result


def _write_layer(directory, key, array, affine):
    path = directory / f"{key}.nii.gz"
    require(not path.exists(), "FRESH_OUTPUT_REQUIRED")
    nib.save(nib.Nifti1Image(array, affine), path)
    return {"file": path.name, "sha256": legacy.sha256(path)}


def export(manifest_path, output_path):
    """Export a fresh protected release, exact NPZ copies and native Slicer layers."""
    manifest_path = legacy.external(manifest_path)
    validated = validate_release(legacy.read_json(manifest_path))
    output = new_output(output_path)
    exported = []
    for record in validated["samples"]:
        hu, affine, arrays = load_sample(record)
        folder = output / record["sample_id"]
        folder.mkdir()
        copied = folder / "annotation.npz"
        with Path(record["annotation"]["path"]).open("rb") as incoming, copied.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        require(legacy.sha256(copied) == record["annotation"]["sha256"], "ANNOTATION_COPY_HASH_MISMATCH")
        layers = {"source_volume": _write_layer(folder, "source_volume", hu, affine)}
        for key in (*HEADS, "instance_ids"):
            dtype = np.int8 if key in HEADS else np.int32
            layers[key] = _write_layer(folder, key, arrays[key].astype(dtype), affine)
            layers[f"{key}_known"] = _write_layer(folder, f"{key}_known", (arrays[key] >= 0).astype(np.uint8), affine)
        layers["evaluation_region"] = _write_layer(folder, "evaluation_region",
                                                   arrays["evaluation_region"].astype(np.uint8), affine)
        metadata = {"schema_version": 1, "version": VERSION, "sample_id": record["sample_id"],
                    "patient_group": record["patient_group"],
                    "source": record["source"], "annotation": record["annotation"],
                    "protocol_sha256": record["protocol_sha256"], "scope": record["scope"],
                    "provenance": record["provenance"], "layers": layers}
        legacy.write_json(folder / "review.local.json", metadata)
        require(legacy.sha256(record["source"]["input"]) == record["source"]["source_sha256"],
                "SOURCE_HASH_MISMATCH")
        exported.append({**record, "annotation": {"path": str(copied), "sha256": legacy.sha256(copied)}})
    released = {"schema_version": 1, "version": VERSION, "samples": exported}
    legacy.write_json(output / "release.local.json", released)
    legacy.write_json(output / "readiness.local.json", validated["readiness"])
    return {"status": "EXPORTED", "version": VERSION, "sample_count": len(exported),
            "release_sha256": legacy.sha256(output / "release.local.json"), "readiness": validated["readiness"]}


def _review_layer(directory, key, shape, affine, *, mask=False):
    try:
        image = nib.load(directory / f"{key}.nii.gz")
        require(image.shape == shape and np.allclose(image.affine, affine, rtol=0, atol=1e-6),
                "BOUNDARY_REVIEW_GEOMETRY_MISMATCH")
        dtype = image.get_data_dtype()
        require((dtype == np.dtype(np.uint8) if mask else dtype.kind in "if") and dtype.itemsize <= 8,
                "BOUNDARY_REVIEW_DTYPE_INVALID")
        require(image.dataobj.slope == 1 and image.dataobj.inter == 0, "BOUNDARY_REVIEW_SCALING_INVALID")
        array = np.asarray(image.dataobj)
        require(np.isfinite(array).all(), "BOUNDARY_REVIEW_VALUES_INVALID")
        if mask:
            require(np.isin(array, [0, 1]).all(), "BOUNDARY_REVIEW_MASK_INVALID")
        elif key != "source_volume":
            require(array.dtype.kind == "i", "BOUNDARY_REVIEW_DTYPE_INVALID")
        return array
    except ExperimentError:
        raise
    except (OSError, ValueError, TypeError, nib.filebasedimages.ImageFileError):
        raise ExperimentError("BOUNDARY_REVIEW_LOAD_FAILED") from None


def import_review(sample, review_directory, output_path):
    """Convert edited Slicer NIfTIs into a fresh NPZ annotation path/hash record.

    Edit both a label layer and its uint8 ``*_known`` layer when adding or
    retracting supervision. Unknown masks restore -1 from signed -1 or Slicer's
    background 0; positive labels outside known masks are rejected. Geometry
    and the source volume cannot change. This helper never
    promotes conditional provenance to expert adjudication.
    """
    record = _sample_record(sample)
    hu, affine, _ = load_sample(record)
    directory = legacy.external(review_directory)
    require(directory.is_dir(), "BOUNDARY_REVIEW_DIRECTORY_REQUIRED")
    metadata = legacy.read_json(directory / "review.local.json")
    expected = {"schema_version": 1, "version": VERSION, "sample_id": record["sample_id"],
                "patient_group": record["patient_group"],
                "source": record["source"],
                "protocol_sha256": record["protocol_sha256"], "scope": record["scope"],
                "provenance": record["provenance"]}
    require(set(metadata) == {*expected, "annotation", "layers"}
            and all(metadata.get(key) == value for key, value in expected.items())
            and isinstance(metadata["annotation"], dict) and set(metadata["annotation"]) == {"path", "sha256"}
            and metadata["annotation"]["sha256"] == record["annotation"]["sha256"],
            "BOUNDARY_REVIEW_BINDING_MISMATCH")
    layer_keys = {"source_volume", *HEADS, "instance_ids", "evaluation_region",
                  *(f"{key}_known" for key in (*HEADS, "instance_ids"))}
    require(isinstance(metadata["layers"], dict) and set(metadata["layers"]) == layer_keys,
            "BOUNDARY_REVIEW_BINDING_MISMATCH")
    for key, value in metadata["layers"].items():
        require(isinstance(value, dict) and set(value) == {"file", "sha256"}
                and value["file"] == f"{key}.nii.gz" and _digest(value["sha256"]),
                "BOUNDARY_REVIEW_BINDING_MISMATCH")
    require(legacy.sha256(directory / "source_volume.nii.gz") == metadata["layers"]["source_volume"]["sha256"],
            "BOUNDARY_REVIEW_SOURCE_CHANGED")
    reviewed_source = _review_layer(directory, "source_volume", hu.shape, affine)
    require(np.array_equal(reviewed_source, hu), "BOUNDARY_REVIEW_SOURCE_CHANGED")
    arrays = {"affine": affine, "evaluation_region": _review_layer(
        directory, "evaluation_region", hu.shape, affine, mask=True).astype(bool)}
    for key in (*HEADS, "instance_ids"):
        labels = _review_layer(directory, key, hu.shape, affine)
        known = _review_layer(directory, f"{key}_known", hu.shape, affine, mask=True).astype(bool)
        require(not np.any(known & (labels < 0)) and not np.any(~known & (labels > 0)),
                "BOUNDARY_REVIEW_KNOWN_MASK_MISMATCH")
        labels = labels.copy()
        labels[~known] = -1
        arrays[key] = labels
    arrays = _arrays(arrays, hu.shape, affine)
    output = legacy.external(output_path, exists=False)
    require(output.suffix.lower() == ".npz" and output.parent.is_dir() and not output.exists(), "FRESH_OUTPUT_REQUIRED")
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": str(output), "sha256": legacy.sha256(output)}
