"""Isolated, source-bound instance-learning experiment; never a production method.

Annotations, checkpoints and reports are external research artifacts. Prediction
accepts no annotations or reference totals. Unknown labels never become negatives.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from . import instance_learning as core

VERSION = "video-guided-instance-prototype-1"
CONFIG = {"spacing_mm": 0.5, "patch_size": 64, "batch_size": 2,
          "steps": 2000, "maximum_gpu_seconds": 3600, "seed": 20261102,
          "learning_rate": 0.001, "weight_decay": 0.0001,
          "noise_std_hu": 20.0, "threshold": 0.5, "minimum_native_voxels": 4,
          "channels": [8, 16, 32, 64], "inference_stride": 32}
SOURCE_FIELDS = {"input", "source_sha256", "crop_bounds", "midline_ras_x"}


class ExperimentError(ValueError):
    """Fixed error codes avoid exposing protected paths or headers."""


def require(condition, code):
    if not condition:
        raise ExperimentError(code)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def external(path, *, exists=True):
    try:
        require(Path(path).is_absolute(), "EXTERNAL_ARTIFACT_REQUIRED")
        result = Path(path).resolve(strict=exists)
        require(result.is_absolute() and not any((p/".git").exists() for p in (result, *result.parents)),
                "EXTERNAL_ARTIFACT_REQUIRED")
        return result
    except (OSError, TypeError, ValueError):
        raise ExperimentError("EXTERNAL_ARTIFACT_REQUIRED") from None


def read_json(path):
    try:
        value = json.loads(external(path).read_text(encoding="utf-8-sig"),
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        require(isinstance(value, dict), "MANIFEST_INVALID")
        return value
    except (OSError, ValueError):
        raise ExperimentError("MANIFEST_INVALID") from None


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def new_output(path):
    output = external(path, exists=False)
    require(not output.exists() and output.parent.is_dir(), "FRESH_OUTPUT_REQUIRED")
    output.mkdir()
    return output


def failure_code(error):
    """Classify expected runtime failures without returning protected messages."""
    if isinstance(error, ExperimentError) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}",str(error)):
        return str(error)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(error,torch.cuda.OutOfMemoryError):
        return "CUDA_OUT_OF_MEMORY"
    if isinstance(error,RuntimeError) and "does not have a deterministic implementation" in str(error):
        return "DETERMINISTIC_KERNEL_UNAVAILABLE"
    return "EXPERIMENT_EXECUTION_FAILED"


def record_failure(command, output_path, error):
    """Preserve a fixed-code failure record only in an existing external run."""
    code = failure_code(error)
    try:
        output = external(output_path)
        require(output.is_dir(), "EXISTING_OUTPUT_REQUIRED")
        last_step = 0
        progress = output/"progress.local.jsonl"
        if command == "train" and progress.is_file():
            for line in progress.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                step = row.get("step")
                if type(step) is int and 0 <= step <= CONFIG["steps"]:
                    last_step = max(last_step,step)
        name = type(error).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",name):
            name = "Exception"
        write_json(output/"failure.local.json",{"status":"FAILED","version":VERSION,
            "command":command if command in {"readiness","train","infer","mechanism","report"} else "unknown",
            "error_code":code,"exception_class":name,
            "last_recorded_completed_step":last_step if command == "train" else None,
            "exact_completed_steps_known":False,
            "checkpoint_file_present":(output/"final-checkpoint.local.pt").is_file(),
            "input_preservation":"NOT_CHECKED_AFTER_FAILURE"})
    except Exception:
        # Never overwrite an earlier failure or create a new unvalidated run.
        pass
    return code


def source_record(value):
    require(isinstance(value, dict) and set(value) == SOURCE_FIELDS, "SOURCE_SCHEMA_INVALID")
    path = external(value["input"])
    require(isinstance(value["source_sha256"], str) and len(value["source_sha256"]) == 64
            and sha256(path) == value["source_sha256"], "SOURCE_HASH_MISMATCH")
    bounds = value["crop_bounds"]
    require(isinstance(bounds, list) and len(bounds) == 3 and all(
        isinstance(pair, list) and len(pair) == 2 and all(type(n) is int for n in pair)
        and 0 <= pair[0] < pair[1] for pair in bounds), "CROP_INVALID")
    require(type(value["midline_ras_x"]) in (float, int) and np.isfinite(value["midline_ras_x"]),
            "MIDLINE_INVALID")
    return {**value, "input": str(path)}


def load_crop(source):
    original = nib.load(source["input"])
    require(len(original.shape) == 3, "SOURCE_DIMENSIONS_INVALID")
    canonical = nib.as_closest_canonical(original)
    affine = np.asarray(canonical.affine, float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    require(np.isfinite(affine).all() and np.all(spacing > 0)
            and np.allclose((affine[:3, :3]/spacing).T @ (affine[:3, :3]/spacing), np.eye(3), atol=1e-5),
            "SOURCE_GEOMETRY_INVALID")
    bounds = np.array(source["crop_bounds"], dtype=int)
    require(np.all(bounds[:, 1] <= canonical.shape), "CROP_OUTSIDE_SOURCE")
    hu = np.asarray(canonical.dataobj[tuple(slice(a, b) for a, b in bounds)], dtype=np.float32)
    require(np.isfinite(hu).all(), "HU_NONFINITE")
    crop_affine = affine.copy()
    crop_affine[:3, 3] = nib.affines.apply_affine(affine, bounds[:, 0])
    return original, canonical, hu, crop_affine


def model_coordinates(shape, spacing):
    shape = np.asarray(shape, int)
    spacing = np.asarray(spacing, float)
    require(shape.shape == spacing.shape == (3,) and np.all(shape > 0)
            and np.isfinite(spacing).all() and np.all(spacing > 0), "GRID_INVALID")
    model_shape = np.ceil((shape-1)*spacing/CONFIG["spacing_mm"]).astype(int)+1
    require(int(np.prod(model_shape)) <= 256**3, "MODEL_GRID_BUDGET_EXCEEDED")
    coords = np.meshgrid(*(np.arange(n, dtype=float)*CONFIG["spacing_mm"]/s
                           for n, s in zip(model_shape, spacing)), indexing="ij")
    # Acquired voxels occupy half-voxel footprints, not only their center box.
    # The upper half-index tie belongs to the next (unacquired) native voxel.
    valid = np.logical_and.reduce([(c >= -.5) & (c < n-.5) for c, n in zip(coords, shape)])
    return np.asarray(coords), valid


def model_grid(hu, affine, labels=None, foreground=None):
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    coordinates, valid = model_coordinates(hu.shape, spacing)
    values = ndi.map_coordinates(hu, coordinates, order=1, mode="nearest", prefilter=False).astype(np.float32)
    values[~valid] = -1000
    if labels is None:
        return values, valid
    require(labels.shape == foreground.shape == hu.shape, "ANNOTATION_SHAPE_INVALID")
    def resample_known(array):
        # An annotation is native-voxel occupancy, not a point observation.
        # Nearest categorical sampling preserves its half-voxel footprint;
        # SciPy order=0 sends exact half-index ties to the higher native index.
        # Unknown native voxels stay unknown, with no linear coverage erosion.
        sampled = ndi.map_coordinates(array, coordinates, order=0, mode="nearest", prefilter=False)
        sampled[~valid] = -1
        return sampled
    return values, valid, resample_known(labels), resample_known(foreground)


def _annotation(manifest, source, hu, affine):
    annotation = manifest.get("annotation")
    require(isinstance(annotation, dict) and set(annotation) == {"path", "sha256"}, "ANNOTATION_REQUIRED")
    path = external(annotation["path"])
    require(sha256(path) == annotation["sha256"], "ANNOTATION_HASH_MISMATCH")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"instance_ids", "foreground_labels", "affine"}, "ANNOTATION_ARRAYS_INVALID")
        labels, foreground = archive["instance_ids"], archive["foreground_labels"]
        require(labels.dtype.kind in "iu" and foreground.dtype.kind in "iu"
                and labels.shape == foreground.shape == hu.shape, "ANNOTATION_SHAPE_INVALID")
        require(np.all(labels >= -1) and np.isin(foreground, [-1, 0, 1]).all()
                and np.allclose(archive["affine"], affine, rtol=0, atol=1e-6), "ANNOTATION_VALUES_INVALID")
        require(not np.any((labels > 0) & (foreground == 0))
                and not np.any((labels == 0) & (foreground == 1)), "ANNOTATION_CONTRADICTION")
        return labels.copy(), foreground.copy(), path


def inspect_readiness(manifest):
    require(set(manifest) <= {"schema_version", "source", "annotation", "distinct_pairs", "evidence", "notes"}
            and manifest.get("schema_version") == 1, "ANNOTATION_MANIFEST_INVALID")
    source = source_record(manifest.get("source"))
    result = {"status": "NOT_READY", "training_allowed": False, "reasons": [],
              "version": VERSION, "config": CONFIG, "source_sha256": source["source_sha256"],
              "source_match_uncertainty": [], "boundary_uncertainty": [], "label_counts": {}, "evidence_hashes": {}}
    evidence = manifest.get("evidence", [])
    require(isinstance(evidence, list), "EVIDENCE_INVALID")
    accepted = []
    for record in evidence:
        require(isinstance(record, dict) and record.get("kind") in
                {"same_cell", "different_cell", "foreground", "background"}, "EVIDENCE_INVALID")
        ids = record.get("point_ids", [])
        require(isinstance(ids, list) and all(type(i) is int and 1 <= i <= 10 for i in ids), "EVIDENCE_INVALID")
        source_match = record.get("source_match")
        boundary = record.get("boundary_support")
        require(source_match in {"consistent", "provisional", "ambiguous", "mismatch"}
                and boundary in {"supported", "ambiguous"}, "EVIDENCE_UNCERTAINTY_INVALID")
        result["source_match_uncertainty"].append(source_match)
        result["boundary_uncertainty"].append(boundary)
        if record.get("review_only") or 10 in ids or source_match in {"ambiguous", "mismatch"} or boundary != "supported":
            continue
        times, frames = record.get("timestamps_seconds"), record.get("frames")
        require(isinstance(times, list) and len(times) >= 1 and all(type(t) in (int, float)
                and np.isfinite(t) and t >= 0 for t in times), "TIMESTAMP_EVIDENCE_REQUIRED")
        if len(set(times)) < 2:
            views = record.get("native_views", [])
            require(isinstance(views, list) and len(views) >= 2, "MULTISLICE_EVIDENCE_REQUIRED")
            original = nib.load(source["input"])
            canonical = nib.as_closest_canonical(original)
            bounds = np.asarray(source["crop_bounds"])
            corners = np.array(np.meshgrid(*[(a,b-1) for a,b in bounds],indexing="ij")).reshape(3,-1).T
            transform = np.linalg.inv(original.affine) @ canonical.affine
            native_corners = nib.affines.apply_affine(transform,corners)
            checked = set()
            for view in views:
                require(isinstance(view,dict) and set(view) == {"axis","index"}
                        and type(view["axis"]) is int and view["axis"] in (0,1,2)
                        and type(view["index"]) is int, "NATIVE_VIEW_INVALID")
                axis,index = view["axis"],view["index"]
                require(0 <= index < original.shape[axis]
                        and native_corners[:,axis].min()-1e-6 <= index <= native_corners[:,axis].max()+1e-6,
                        "NATIVE_VIEW_OUTSIDE_SOURCE_CROP")
                checked.add((axis,index))
            require(len(checked) >= 2, "MULTISLICE_EVIDENCE_REQUIRED")
        require(isinstance(frames, list) and len(frames) >= 2, "FRAME_EVIDENCE_REQUIRED")
        for frame in frames:
            require(isinstance(frame, dict) and set(frame) == {"path", "sha256"}
                    and sha256(external(frame["path"])) == frame["sha256"], "FRAME_HASH_MISMATCH")
            result["evidence_hashes"][str(external(frame["path"]))] = frame["sha256"]
        accepted.append(record)
    kinds = Counter(record["kind"] for record in accepted)
    result["accepted_evidence_kinds"] = dict(kinds)
    if not kinds["same_cell"]:
        result["reasons"].append("SUPPORTED_MULTISLICE_SAME_CELL_REGIONS_MISSING")
    if not kinds["different_cell"]:
        result["reasons"].append("SUPPORTED_MULTISLICE_DIFFERENT_CELL_REGIONS_MISSING")
    if manifest.get("annotation") is None:
        result["reasons"].append("SUPPORTED_VOXEL_ANNOTATIONS_MISSING")
        return result
    _, _, hu, affine = load_crop(source)
    labels, foreground, _ = _annotation(manifest, source, hu, affine)
    pairs = manifest.get("distinct_pairs", [])
    supported_pairs = set()
    same_ids = set()
    for record in accepted:
        if record["kind"] in {"same_cell", "different_cell"}:
            label_ids = record.get("label_ids")
            expected = 1 if record["kind"] == "same_cell" else 2
            require(isinstance(label_ids, list) and len(label_ids) == expected
                    and len(set(label_ids)) == expected and all(type(i) is int and i > 0 for i in label_ids),
                    "EVIDENCE_LABEL_IDS_REQUIRED")
            require(all(np.any(labels == i) for i in label_ids), "EVIDENCE_LABEL_ID_ABSENT")
            if expected == 2:
                supported_pairs.add(tuple(sorted(label_ids)))
            else:
                same_ids.add(label_ids[0])
    require(isinstance(pairs, list) and all(isinstance(pair, list) and len(pair) == 2
            and all(type(i) is int and i > 0 for i in pair) and pair[0] != pair[1] for pair in pairs),
            "DISTINCT_PAIR_INVALID")
    require(set(tuple(sorted(pair)) for pair in pairs) <= supported_pairs, "UNSUPPORTED_DISTINCT_PAIR")
    require(set(int(i) for i in np.unique(labels[labels > 0])) <= same_ids, "UNSUPPORTED_INSTANCE_VOXELS")
    require(not np.any((foreground == 1) & (labels <= 0)) or kinds["foreground"] > 0,
            "FOREGROUND_ONLY_EVIDENCE_REQUIRED")
    require(not np.any((foreground == 0) | (labels == 0)) or kinds["background"] > 0, "BACKGROUND_EVIDENCE_REQUIRED")
    for identity in same_ids:
        coordinates = np.argwhere(labels == identity)
        require(len(coordinates) >= 2 and np.any(np.ptp(coordinates, axis=0) >= 1), "MULTISLICE_REGION_REQUIRED")
    targets, masks = core.build_targets(labels, distinct_pairs=pairs, foreground_labels=foreground)
    result["label_counts"] = {"known_foreground": int(np.sum(masks[0] & (targets[0] == 1))),
                              "explicit_background": int(np.sum(masks[0] & (targets[0] == 0))),
                              "same_cell_edges": int(np.sum(masks[1:] & (targets[1:] == 1))),
                              "different_cell_edges": int(np.sum(masks[1:] & (targets[1:] == 0))),
                              "unknown_foreground_voxels": int(np.sum(~masks[0]))}
    if not all(result["label_counts"][key] > 0 for key in
               ("known_foreground", "explicit_background", "same_cell_edges", "different_cell_edges")):
        result["reasons"].append("BOTH_FOREGROUND_AND_AFFINITY_CLASSES_REQUIRED")
    if not result["reasons"]:
        result.update(status="READY_CONDITIONAL_DEVELOPMENT", training_allowed=True)
    return result


def readiness(manifest_path, output_path):
    manifest = read_json(manifest_path)
    result = inspect_readiness(manifest)
    result["manifest_sha256"] = sha256(manifest_path)
    output = new_output(output_path)
    write_json(output/"readiness.local.json", result)
    return result


def deterministic_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    require(torch.cuda.is_available(), "CUDA_RUNTIME_REQUIRED")
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def patch(array, center, size, fill):
    result = np.full((size,)*3, fill, dtype=array.dtype)
    low = np.asarray(center, int)-size//2
    high = low+size
    a, b = np.maximum(low, 0), np.minimum(high, array.shape)
    if np.all(b > a):
        result[tuple(slice(x, y) for x, y in zip(a-low, b-low))] = array[tuple(slice(x, y) for x, y in zip(a, b))]
    return result


def train(manifest_path, output_path):
    manifest = read_json(manifest_path)
    result = inspect_readiness(manifest)
    output = new_output(output_path)
    write_json(output/"readiness.local.json", result)
    require(result["training_allowed"], "READINESS_FAILED_NO_TRAINING")
    source = source_record(manifest["source"])
    _, _, hu, affine = load_crop(source)
    labels, foreground, annotation_path = _annotation(manifest, source, hu, affine)
    hu, _, labels, foreground = model_grid(hu, affine, labels, foreground)
    targets, masks = core.build_targets(labels, distinct_pairs=manifest.get("distinct_pairs", []), foreground_labels=foreground)
    require(np.any(masks[1:] & (targets[1:] == 0)) and np.any(masks[1:] & (targets[1:] == 1)),
            "RESAMPLED_AFFINITY_CLASSES_MISSING")
    centers = [np.argwhere(masks[0] & (targets[0] == v)) for v in (0,1)]
    require(all(len(c) for c in centers), "RESAMPLED_ANNOTATION_CLASSES_MISSING")
    started = time.monotonic()
    torch = deterministic_torch()
    model = core.build_model().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    rng = np.random.default_rng(CONFIG["seed"])
    input_hashes = {str(external(manifest_path)): sha256(manifest_path), source["input"]: source["source_sha256"],
                    str(annotation_path): manifest["annotation"]["sha256"]}
    input_hashes.update(result["evidence_hashes"])
    losses = []
    torch.cuda.synchronize()
    complete = True
    for step in range(CONFIG["steps"]):
        if time.monotonic()-started >= CONFIG["maximum_gpu_seconds"]:
            complete = False
            break
        xs, ys, ms = [], [], []
        for i in range(CONFIG["batch_size"]):
            pool = centers[(step+i) % 2]
            center = pool[rng.integers(len(pool))]
            h = patch(hu, center, CONFIG["patch_size"], -1000)
            l = patch(labels, center, CONFIG["patch_size"], -1)
            f = patch(foreground, center, CONFIG["patch_size"], -1)
            if rng.random() < .5:
                h, l, f = (np.flip(a, axis=0).copy() for a in (h, l, f))
            h = h+rng.normal(0, CONFIG["noise_std_hu"], h.shape).astype(np.float32)
            target, mask = core.build_targets(l, distinct_pairs=manifest.get("distinct_pairs", []), foreground_labels=f)
            xs.append(core.normalize_hu(h)); ys.append(target); ms.append(mask)
        x = torch.from_numpy(np.stack(xs)).to("cuda")
        y = torch.from_numpy(np.stack(ys)).to("cuda")
        m = torch.from_numpy(np.stack(ms)).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = core.masked_balanced_bce(model(x), y, m)
        require(bool(torch.isfinite(loss).item()), "NONFINITE_TRAINING_LOSS")
        loss.backward(); optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))
        if (step+1) % 100 == 0:
            with (output/"progress.local.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"step":step+1,"loss":losses[-1],"elapsed_seconds":time.monotonic()-started},allow_nan=False)+"\n")
        if time.monotonic()-started >= CONFIG["maximum_gpu_seconds"]:
            complete = False
            break
    preserved = all(sha256(path) == digest for path, digest in input_hashes.items())
    require(preserved, "TRAINING_INPUT_CHANGED")
    summary = {"status": "COMPLETE" if complete and len(losses) == CONFIG["steps"] else "INCOMPLETE_BUDGET",
               "version": VERSION, "config": CONFIG, "steps_completed": len(losses),
               "elapsed_gpu_session_seconds": time.monotonic()-started,
               "input_hashes": input_hashes, "inputs_unchanged": preserved, "losses": losses}
    if summary["status"] == "COMPLETE":
        checkpoint = output/"final-checkpoint.local.pt"
        torch.save({"version": VERSION, "config": CONFIG, "model_version": core.MODEL_VERSION,
                    "state_dict": model.cpu().state_dict(), "steps_completed": len(losses),
                    "annotation_manifest_sha256": sha256(manifest_path)}, checkpoint)
        summary["checkpoint_sha256"] = sha256(checkpoint)
    write_json(output/"training.local.json", summary)
    return summary


def starts(length, size=64, stride=32):
    if length <= size:
        return [0]
    return sorted(set(range(0, length-size+1, stride)) | {length-size})


def tiled_probabilities(hu, model, torch):
    shape = np.array(hu.shape)
    sums = np.zeros((7, *shape), np.float32)
    weights = np.zeros((7,*shape), np.float32)
    model.eval()
    with torch.inference_mode():
        for x in starts(shape[0]):
            for y in starts(shape[1]):
                for z in starts(shape[2]):
                    low = np.array([x, y, z]); high = np.minimum(low+64, shape)
                    box = tuple(slice(a, b) for a, b in zip(low, high))
                    h = np.full((64,)*3, -1000, np.float32)
                    sub = tuple(slice(0, b-a) for a, b in zip(low, high))
                    h[sub] = hu[box]
                    probabilities = torch.sigmoid(model(torch.from_numpy(core.normalize_hu(h)[None]).to("cuda")))[0].cpu().numpy()
                    sums[(0,*box)] += probabilities[(0,*sub)]
                    weights[(0,*box)] += 1
                    for channel,offset in enumerate(core.OFFSETS,start=1):
                        edge_high = high-np.array(offset)
                        if np.any(edge_high <= low):
                            continue
                        edge_box = tuple(slice(a,b) for a,b in zip(low,edge_high))
                        edge_sub = tuple(slice(0,b-a) for a,b in zip(low,edge_high))
                        sums[(channel,*edge_box)] += probabilities[(channel,*edge_sub)]
                        weights[(channel,*edge_box)] += 1
    require(np.all(weights[0] > 0), "TILING_COVERAGE_FAILED")
    for channel,offset in enumerate(core.OFFSETS,start=1):
        source,_ = core.edge_slices(tuple(shape),offset)
        require(np.all(weights[(channel,*source)] > 0), "TILING_EDGE_COVERAGE_FAILED")
    return np.divide(sums,weights,out=np.zeros_like(sums),where=weights > 0)


def native_labels(model_labels, crop_shape, crop_affine):
    spacing = np.linalg.norm(np.asarray(crop_affine)[:3, :3], axis=0)
    coords = np.meshgrid(*(np.arange(n)*s/CONFIG["spacing_mm"] for n, s in zip(crop_shape, spacing)), indexing="ij")
    labels = ndi.map_coordinates(model_labels, np.asarray(coords), order=0, mode="nearest", prefilter=False).astype(np.int32)
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids[(ids > 0) & (counts >= CONFIG["minimum_native_voxels"])]
    require(len(keep) <= 65535, "TOO_MANY_INSTANCES")
    lookup = np.zeros(int(labels.max())+1, np.uint16)
    lookup[keep] = np.arange(1, len(keep)+1, dtype=np.uint16)
    return lookup[labels]


def label_registry(labels, hu, affine, midline):
    registry = []
    for identity in np.unique(labels):
        if not identity:
            continue
        indices = np.argwhere(labels == identity)
        world = nib.affines.apply_affine(affine, indices)
        centroid = world.mean(axis=0)
        # RAS X grows toward the patient's right; exact midline centroids stay flagged.
        side = "right" if centroid[0] > midline else "left"
        values = hu[tuple(indices.T)]
        registry.append({"id": int(identity), "side": side, "voxel_count": len(indices),
                         "volume_mm3": float(len(indices)*abs(np.linalg.det(affine[:3, :3]))),
                         "centroid_ras_mm": centroid.tolist(), "mean_hu": float(values.mean()),
                         "flags": ["crosses_midline"] if world[:, 0].min() <= midline <= world[:, 0].max() else []})
    counts = Counter(item["side"] for item in registry)
    return registry, {"left": counts["left"], "right": counts["right"], "total": len(registry)}


def infer(manifest_path, output_path):
    manifest = read_json(manifest_path)
    require(set(manifest) == {"schema_version", "case_id", "source", "checkpoint", "checkpoint_sha256"}
            and manifest["schema_version"] == 1 and isinstance(manifest["case_id"], str)
            and manifest["case_id"].isascii() and manifest["case_id"].replace("_", "").isalnum(),
            "INFERENCE_MANIFEST_INVALID")
    source = source_record(manifest["source"])
    checkpoint = external(manifest["checkpoint"])
    require(sha256(checkpoint) == manifest["checkpoint_sha256"], "CHECKPOINT_HASH_MISMATCH")
    torch = deterministic_torch()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved.get("version") == VERSION and saved.get("model_version") == core.MODEL_VERSION
            and saved.get("config") == CONFIG and saved.get("steps_completed") == CONFIG["steps"], "CHECKPOINT_PROTOCOL_MISMATCH")
    model = core.build_model(); model.load_state_dict(saved["state_dict"], strict=True); model.to("cuda")
    original, canonical, hu, affine = load_crop(source)
    grid, valid = model_grid(hu, affine)
    probabilities = tiled_probabilities(grid, model, torch)
    probabilities[0, ~valid] = 0
    decoded = core.decode_probabilities(probabilities)
    labels = native_labels(decoded, hu.shape, affine)
    registry, counts = label_registry(labels, hu, affine, source["midline_ras_x"])
    full = np.zeros(canonical.shape, np.uint16)
    bounds = source["crop_bounds"]
    full[tuple(slice(a, b) for a, b in bounds)] = labels
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
                                                nib.orientations.io_orientation(original.affine))
    stored = nib.orientations.apply_orientation(full, orientation)
    require(stored.shape == original.shape, "NATIVE_PROJECTION_FAILED")
    require(sha256(source["input"]) == source["source_sha256"]
            and sha256(checkpoint) == manifest["checkpoint_sha256"], "INFERENCE_INPUT_CHANGED")
    output = new_output(output_path)
    artifact = output/"instances.local.nii.gz"
    nib.save(nib.Nifti1Image(stored, original.affine), artifact)
    result = {"status": "COMPLETE", "version": VERSION, "case_id": manifest["case_id"],
              "worker_identity": str(uuid.uuid4()), "worker_pid": os.getpid(),
              "counts": counts, "registry": registry, "source_sha256": source["source_sha256"],
              "checkpoint_sha256": manifest["checkpoint_sha256"], "config_sha256": fingerprint(CONFIG),
              "annotation_manifest_sha256": saved.get("annotation_manifest_sha256"),
              "native_array_sha256": hashlib.sha256(stored.tobytes(order="C")).hexdigest(),
              "native_affine": original.affine.tolist(), "native_shape": list(original.shape),
              "artifact_sha256": sha256(artifact), "inputs_unchanged": True}
    write_json(output/"prediction.local.json", result)
    return result


def mechanism(manifest_path, output_path):
    """Score only supported annotation voxels after prediction has closed."""
    manifest = read_json(manifest_path)
    require(set(manifest) == {"schema_version", "annotation_manifest", "prediction"}
            and manifest["schema_version"] == 1, "MECHANISM_MANIFEST_INVALID")
    annotation = read_json(manifest["annotation_manifest"])
    ready = inspect_readiness(annotation)
    require(ready["training_allowed"], "MECHANISM_ANNOTATIONS_NOT_READY")
    source = source_record(annotation["source"])
    prediction = read_json(manifest["prediction"])
    require(prediction.get("status") == "COMPLETE" and prediction.get("source_sha256") == source["source_sha256"],
            "MECHANISM_SOURCE_MISMATCH")
    require(prediction.get("annotation_manifest_sha256") == sha256(manifest["annotation_manifest"]),
            "MECHANISM_ANNOTATION_CHANGED")
    artifact = external(Path(manifest["prediction"]).parent/"instances.local.nii.gz")
    require(sha256(artifact) == prediction.get("artifact_sha256"), "MECHANISM_ARTIFACT_MISMATCH")
    predicted = nib.as_closest_canonical(nib.load(artifact))
    _, canonical, hu, affine = load_crop(source)
    require(predicted.shape == canonical.shape and np.allclose(predicted.affine, canonical.affine, rtol=0, atol=1e-6),
            "MECHANISM_GEOMETRY_MISMATCH")
    bounds = source["crop_bounds"]
    values = np.asarray(predicted.dataobj[tuple(slice(a,b) for a,b in bounds)])
    labels, foreground, _ = _annotation(annotation, source, hu, affine)
    regions = []
    assignment = {}
    for identity in np.unique(labels[labels > 0]):
        sampled = values[labels == identity]
        ids = np.unique(sampled[sampled > 0]).tolist()
        assignment[int(identity)] = ids[0] if len(ids) == 1 and np.all(sampled > 0) else None
        regions.append({"annotation_id":int(identity), "known_voxels":len(sampled),
                        "covered_voxels":int(np.sum(sampled > 0)), "prediction_ids":ids,
                        "split":len(ids) > 1, "fully_covered":bool(np.all(sampled > 0))})
    pairs = []
    for a,b in annotation.get("distinct_pairs", []):
        left, right = assignment.get(a), assignment.get(b)
        pairs.append({"annotation_ids":[a,b], "prediction_ids":[left,right],
                      "separate":left is not None and right is not None and left != right})
    known_positive = (foreground == 1) | (labels > 0)
    known_negative = (foreground == 0) | (labels == 0)
    missing = int(np.sum(known_positive & (values == 0)))
    false_positive = int(np.sum(known_negative & (values > 0)))
    gates = {"three_supported_distinctions":len(pairs) >= 3 and all(p["separate"] for p in pairs),
             "annotated_within_cell_connections":bool(regions) and all(r["fully_covered"] and not r["split"] for r in regions),
             "known_foreground_coverage":missing == 0,
             "explicit_exclusions_preserved":false_positive == 0}
    result = {"status":"PASS_CONDITIONAL_MECHANISM" if all(gates.values()) else "FAIL",
              "version":VERSION, "gates":gates, "regions":regions, "distinctions":pairs,
              "known_foreground_voxels":int(np.sum(known_positive)), "missing_known_foreground_voxels":missing,
              "explicit_background_voxels":int(np.sum(known_negative)), "false_positive_known_background_voxels":false_positive,
              "unknown_voxels_unscored":int(np.sum(~(known_positive|known_negative))),
              "annotation_manifest_sha256":sha256(manifest["annotation_manifest"]),
              "prediction_sha256":sha256(manifest["prediction"]),
              "checkpoint_sha256":prediction.get("checkpoint_sha256"),
              "interpretation":"Conditional development annotation agreement, not complete-cell or clinical accuracy."}
    output = new_output(output_path)
    write_json(output/"mechanism.local.json",result)
    return result


def report(manifest_path, output_path):
    manifest = read_json(manifest_path)
    require(set(manifest) <= {"schema_version", "entries", "readiness", "notes", "mechanism"}
            and manifest.get("schema_version") == 1 and isinstance(manifest.get("entries"), list)
            and len(manifest["entries"]) == 8, "REPORT_MANIFEST_INVALID")
    rows, unique = [], set()
    checkpoint_hashes = set()
    total_error = 0
    exact_cases = exact_sides = complete = 0
    repeatable = True
    for entry in manifest["entries"]:
        require(set(entry) == {"case_id", "reference", "baseline", "source_sha256", "predictions"}, "REPORT_ENTRY_INVALID")
        case = entry["case_id"]
        require(isinstance(case, str) and case not in unique, "DUPLICATE_REPORT_CASE")
        unique.add(case)
        for key in ("reference", "baseline"):
            require(isinstance(entry[key], dict) and set(entry[key]) == {"left", "right"}
                    and all(type(v) is int and v >= 0 for v in entry[key].values()), "REPORT_COUNTS_INVALID")
        row = {"case_id": case, "reference": entry["reference"], "baseline": entry["baseline"],
               "status": "NOT_RUN", "prototype": None, "absolute_side_error": None,
               "bilateral_exact": False, "side_regressions": [], "repeatable": False}
        predictions = entry["predictions"]
        require(isinstance(predictions, list) and len(predictions) in (0, 2), "TWO_REPEAT_PATHS_REQUIRED")
        if predictions:
            records = []
            for path in predictions:
                try:
                    record = read_json(path)
                    artifact = external(Path(path).parent/"instances.local.nii.gz")
                    image = nib.load(artifact)
                    values = np.asanyarray(image.dataobj)
                    identity = np.unique(values)
                    expected_ids = identity[identity > 0].tolist()
                    registry = record.get("registry", [])
                    counts = record.get("counts", {})
                    require(record.get("status") == "COMPLETE" and record.get("version") == VERSION
                            and record.get("case_id") == case and record.get("source_sha256") == entry["source_sha256"]
                            and record.get("inputs_unchanged") is True
                            and sha256(artifact) == record.get("artifact_sha256")
                            and hashlib.sha256(values.tobytes(order="C")).hexdigest() == record.get("native_array_sha256")
                            and list(values.shape) == record.get("native_shape")
                            and np.allclose(image.affine, record.get("native_affine"), rtol=0, atol=1e-6)
                            and values.dtype.kind in "iu" and np.all(values >= 0)
                            and sorted(r["id"] for r in registry) == expected_ids
                            and set(counts) == {"left", "right", "total"}
                            and all(type(v) is int and v >= 0 for v in counts.values())
                            and counts["total"] == len(registry) == counts["left"]+counts["right"]
                            and all(r["side"] in {"left", "right"} for r in registry)
                            and all(counts[side] == sum(r["side"] == side for r in registry) for side in ("left", "right")),
                            "PREDICTION_ARTIFACT_INVALID")
                    records.append(record)
                except (ExperimentError, OSError, ValueError, TypeError, KeyError, nib.filebasedimages.ImageFileError):
                    records.append(None)
            valid = all(r is not None for r in records)
            if valid:
                a, b = records
                checkpoint_hashes.update(r.get("checkpoint_sha256") for r in records)
                row["repeatable"] = (str(external(predictions[0])) != str(external(predictions[1]))
                    and a.get("worker_identity") != b.get("worker_identity")
                    and a.get("config_sha256") == b.get("config_sha256") == fingerprint(CONFIG)
                    and a.get("checkpoint_sha256") == b.get("checkpoint_sha256")
                    and a["native_array_sha256"] == b["native_array_sha256"] and a["counts"] == b["counts"])
                row["status"] = "COMPLETE" if row["repeatable"] else "NONREPEATABLE"
                row["prototype"] = {side: a["counts"][side] for side in ("left", "right")}
                require(all(type(v) is int and v >= 0 for v in row["prototype"].values()), "PREDICTION_COUNTS_INVALID")
                errors = {side: abs(row["prototype"][side]-entry["reference"][side]) for side in ("left", "right")}
                row["absolute_side_error"] = sum(errors.values())
                row["bilateral_exact"] = not any(errors.values())
                row["side_regressions"] = [side for side in errors if errors[side] > abs(entry["baseline"][side]-entry["reference"][side])]
                total_error += row["absolute_side_error"]
                exact_cases += int(row["bilateral_exact"])
                exact_sides += sum(error == 0 for error in errors.values())
                complete += 1
            else:
                row["status"] = "FAILED"
        repeatable &= row["repeatable"]
        rows.append(row)
    baseline_error = sum(abs(r["baseline"][s]-r["reference"][s]) for r in rows for s in ("left", "right"))
    require(baseline_error == 37, "FROZEN_BASELINE_ERROR_MISMATCH")
    original = next((r for r in rows if r["case_id"] == "Case008_Series004"), None)
    training = next((r for r in rows if r["case_id"] == "Case024_Series002"), None)
    require(original is not None and training is not None, "FROZEN_CASES_REQUIRED")
    other_error = total_error-training["absolute_side_error"] if complete == 8 else None
    gates = {"all_eight_outputs": complete == 8, "at_least_two_exact_bilateral": exact_cases >= 2,
             "absolute_error_below_37": complete == 8 and total_error < 37,
             "preserve_original_exact_case": original["bilateral_exact"],
             "other_seven_error_below_35": other_error is not None and other_error < 35,
             "two_repeat_identical_native_masks_and_counts": repeatable,
             "single_frozen_checkpoint": complete == 8 and len(checkpoint_hashes) == 1 and None not in checkpoint_hashes}
    counts_pass = all(gates.values())
    mechanism_status = "NOT_EVALUATED"
    if manifest.get("mechanism") is not None:
        mechanism_record = read_json(manifest["mechanism"])
        mechanism_status = mechanism_record.get("status", "INVALID")
        require(mechanism_record.get("version") == VERSION, "MECHANISM_VERSION_MISMATCH")
        training_entry = next(entry for entry in manifest["entries"] if entry["case_id"] == "Case024_Series002")
        bound = (complete == 8 and mechanism_record.get("checkpoint_sha256") in checkpoint_hashes
                 and mechanism_record.get("prediction_sha256") in {sha256(path) for path in training_entry["predictions"]})
        if not bound:
            mechanism_status = "UNBOUND_TO_COMPARISON"
    gates["supported_cell_identity_mechanism"] = mechanism_status == "PASS_CONDITIONAL_MECHANISM"
    summary = {"status": "PASS_CONDITIONAL_DEVELOPMENT" if all(gates.values()) else ("NOT_RUN" if not complete else "FAIL"),
               "version": VERSION, "scheduled_cases": 8, "scheduled_sides": 16,
               "count_status":"PASS_DEVELOPMENT_COUNTS" if counts_pass else "FAIL_OR_NOT_RUN",
               "mechanism_status":mechanism_status,
               "complete_cases": complete, "bilateral_exact": exact_cases, "side_exact": exact_sides,
               "absolute_side_error": total_error if complete == 8 else None,
               "other_seven_absolute_side_error": other_error, "gates": gates, "cases": rows,
               "interpretation": "Development count agreement only; cell-identity mechanism review remains separate."}
    output = new_output(output_path)
    write_json(output/"comparison.local.json", summary)
    text = ["# Eight-case development comparison", "", "| Scan | Reference L/R | Baseline L/R | Prototype L/R | Status |", "|---|---:|---:|---:|---|"]
    for row in rows:
        pair = lambda value: f"{value['left']}/{value['right']}" if value else "Not run"
        text.append(f"| {row['case_id']} | {pair(row['reference'])} | {pair(row['baseline'])} | {pair(row['prototype'])} | {row['status']} |")
    text.extend(["", "No absent prediction is replaced by a baseline count. All eight cases remain in the denominator.",
                 "Count agreement is development evidence, not individual-cell or clinical accuracy."])
    (output/"REPORT.local.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary
