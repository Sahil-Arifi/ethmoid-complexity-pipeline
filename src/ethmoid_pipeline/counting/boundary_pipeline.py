"""Protected local runner for the versioned boundary-instance experiment."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import time
import uuid

import nibabel as nib
import numpy as np

from . import boundary_learning as learning
from . import instance_experiment as old
from . import instance_learning as backbone
from .boundary_io import new_output
from .training import instance_agreement

VERSION = learning.VERSION


def _reference(value, code="ARTIFACT_REFERENCE_INVALID"):
    old.require(isinstance(value, dict) and set(value) == {"path", "sha256"}, code)
    path = old.external(value["path"])
    old.require(re.fullmatch(r"[a-f0-9]{64}", str(value["sha256"])) is not None
                and old.sha256(path) == value["sha256"], "ARTIFACT_HASH_MISMATCH")
    return path


def _manifest(path, fields, optional=()):
    value = old.read_json(path)
    old.require(set(fields) <= set(value) <= set(fields) | set(optional)
                and value.get("schema_version") == 1 and value.get("version") == VERSION,
                "BOUNDARY_MANIFEST_INVALID")
    return value


def _sources():
    from . import boundary_data, boundary_partition
    return {Path(module.__file__).name: old.sha256(module.__file__)
            for module in (learning, backbone, old, boundary_data, boundary_partition)}


def _runtime(torch):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "cudnn": torch.backends.cudnn.version()}


def _gpu_budget(torch):
    free, _ = torch.cuda.mem_get_info()
    old.require(free >= 3*1024**3, "GPU_MEMORY_PREFLIGHT_FAILED")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(min(1., 12*1024**3/torch.cuda.get_device_properties(0).total_memory))


def _torch():
    torch = old.deterministic_torch()
    torch.manual_seed(learning.CONFIG["seed"])
    torch.cuda.manual_seed_all(learning.CONFIG["seed"])
    return torch


def _pools(samples):
    """Group observed head/class labels into annotation sampling regions."""
    from scipy.ndimage import generate_binary_structure, label

    pools = [[] for _ in range(4)]
    connectivity = generate_binary_structure(3, 1)
    for sample_index, (_, _, arrays) in enumerate(samples):
        y, m = learning.targets(arrays)
        for head in range(2):
            for cls in (0, 1):
                observed = m[head] & (y[head] == cls)
                components, count = label(observed, connectivity)
                positions = np.flatnonzero(observed)
                if len(positions):
                    ids = components.ravel()[positions]
                    ordered = positions[np.argsort(ids, kind="stable")]
                    sizes = np.bincount(ids, minlength=count+1)[1:]
                    # Retaining every index costs CPU memory proportional to known
                    # supervision, bounded by the release, and preserves rare regions.
                    regions = np.split(ordered, np.cumsum(sizes)[:-1])
                    pools[head*2+cls].append((sample_index, regions))
    old.require(all(pools), "TRAIN_PARTITION_HEAD_CLASSES_MISSING")
    return pools


def _sample_anchor(pool, rng, component_rng):
    """Keep eligible released samples/crops uniform, then choose component and voxel."""
    sample_index, regions = pool[rng.integers(len(pool))]
    positions = regions[component_rng.integers(len(regions))]
    return sample_index, positions[rng.integers(len(positions))]


def _validation(samples, model, torch, deadline=None):
    losses = [[[], []], [[], []]]
    for hu, _, arrays in samples:
        p = learning.probabilities(hu, model, torch, deadline=deadline)
        target, known = learning.targets(arrays)
        for head in range(2):
            for cls in (0, 1):
                selected = known[head] & (target[head] == cls)
                if np.any(selected):
                    values = np.clip(p[head, selected], 1e-7, 1-1e-7)
                    values = -np.log(values if cls else 1-values)
                    losses[head][cls].append((float(values.sum(dtype=np.float64)), len(values)))
    heads = []
    for classes in losses:
        means = [sum(v for v, _ in items)/sum(n for _, n in items) for items in classes if items]
        old.require(bool(means), "VALIDATION_HEAD_SUPERVISION_MISSING")
        heads.append(float(np.mean(means)))
    return float(np.mean(heads))


def train(manifest_path, output_path):
    from . import boundary_data as data
    manifest = _manifest(manifest_path, {"schema_version", "version", "dataset", "purpose"},
                         {"synthetic_steps"})
    dataset_path = _reference(manifest["dataset"])
    release = data.validate_release(old.read_json(dataset_path))
    ready = release["readiness"]
    output = new_output(output_path)
    old.write_json(output/"readiness.local.json", ready)
    old.require(ready["training_allowed"], "READINESS_FAILED_NO_TRAINING")
    records = release["samples"]
    synthetic = all(record["scope"] == "synthetic_truth" for record in records)
    old.require(manifest["purpose"] in {"synthetic_engineering", "native_development"}
                and (manifest["purpose"] != "synthetic_engineering" or synthetic), "TRAINING_PURPOSE_INVALID")
    steps = manifest.get("synthetic_steps", learning.CONFIG["steps"])
    old.require(type(steps) is int and 0 < steps <= learning.CONFIG["steps"]
                and ("synthetic_steps" not in manifest or synthetic), "TRAINING_BUDGET_INVALID")
    groups = sorted({r["patient_group"] for r in records})
    validation_group = groups[-1] if len(groups) > 1 else None
    training_records = [r for r in records if r["patient_group"] != validation_group]
    validation_records = [r for r in records if r["patient_group"] == validation_group]
    training_samples = [data.load_sample(r) for r in training_records]
    validation_samples = [data.load_sample(r) for r in validation_records]
    pools = _pools(training_samples)
    inputs = {str(dataset_path): old.sha256(dataset_path), str(old.external(manifest_path)): old.sha256(manifest_path)}
    for record in records:
        inputs[record["source"]["input"]] = record["source"]["source_sha256"]
        inputs[record["annotation"]["path"]] = record["annotation"]["sha256"]
    torch = _torch()
    _gpu_budget(torch)
    model = learning.build_model().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning.CONFIG["learning_rate"],
                                 weight_decay=learning.CONFIG["weight_decay"])
    rng = np.random.default_rng(learning.CONFIG["seed"])
    component_rng = np.random.default_rng(learning.CONFIG["seed"])
    patch_rng = np.random.default_rng(learning.CONFIG["seed"])
    best_loss, selected_step = float("inf"), 0
    checkpoint = output/"selected-checkpoint.local.pt"
    started = time.monotonic()
    deadline = started + learning.CONFIG["maximum_gpu_seconds"]
    completed = 0
    selection = "patient_disjoint_annotated_validation_loss" if validation_samples else "last_saved_step_single_group_mechanism_only"
    for step in range(1, steps+1):
        if time.monotonic()-started >= learning.CONFIG["maximum_gpu_seconds"]:
            break
        xs, ys, masks = [], [], []
        for batch in range(learning.CONFIG["batch_size"]):
            pool = pools[((step-1)*learning.CONFIG["batch_size"]+batch) % 4]
            sample_index, position = _sample_anchor(pool, rng, component_rng)
            hu, _, arrays = training_samples[sample_index]
            anchor = np.unravel_index(position, hu.shape)
            size = learning.CONFIG["patch_size"]
            center = learning.sample_patch_center(hu.shape, anchor, patch_rng, size=size)
            h = old.patch(hu, center, size, -1000)
            a = {key: old.patch(arrays[key], center, size, -1)
                 for key in ("foreground_labels", "separator_labels")}
            for axis in range(3):
                if rng.random() < .5:
                    h = np.flip(h, axis).copy()
                    a = {key: np.flip(value, axis).copy() for key, value in a.items()}
            h += rng.normal(0, 20, h.shape).astype(np.float32)
            y, known = learning.targets(a)
            xs.append(backbone.normalize_hu(h)); ys.append(y); masks.append(known)
        x, y, known = [torch.from_numpy(np.stack(v)).to("cuda") for v in (xs, ys, masks)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = learning.masked_loss(model(x), y, known)
        old.require(bool(torch.isfinite(loss)), "NONFINITE_TRAINING_LOSS")
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        completed = step
        validate_now = step % learning.CONFIG["validation_interval"] == 0 or step == steps
        if validate_now:
            val_loss = _validation(validation_samples, model, torch, deadline) if validation_samples else None
            row = {"step": step, "loss": float(loss.detach().cpu()), "validation_loss": val_loss,
                   "elapsed_seconds": time.monotonic()-started}
            with (output/"progress.local.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            if val_loss is None or val_loss < best_loss:
                if val_loss is not None:
                    best_loss = val_loss
                selected_step = step
                torch.save({"version": VERSION, "model_version": learning.MODEL_VERSION,
                            "config": learning.CONFIG, "model_sources": _sources(), "model_runtime": _runtime(torch),
                            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "selected_step": step, "steps_requested": steps,
                            "selection": selection, "dataset_sha256": old.sha256(dataset_path),
                            "evidence_scopes": sorted({r["scope"] for r in records})}, checkpoint)
        old.require(torch.cuda.max_memory_allocated() <= 12*1024**3, "GPU_MEMORY_BUDGET_EXCEEDED")
    old.require(selected_step > 0, "NO_VALIDATED_CHECKPOINT_WITHIN_BUDGET")
    old.require(all(old.sha256(path) == digest for path, digest in inputs.items()), "TRAINING_INPUT_CHANGED")
    result = {"status": "COMPLETE" if completed == steps else "COMPLETE_BOUNDED",
              "version": VERSION, "model_version": learning.MODEL_VERSION, "config": learning.CONFIG,
              "steps_completed": completed, "steps_requested": steps, "selected_step": selected_step,
              "selection": selection, "validation_loss": best_loss if validation_samples else None,
              "training_groups": groups[:-1] if validation_group else groups,
              "validation_groups": [validation_group] if validation_group else [],
              "evidence_scopes": sorted({r["scope"] for r in records}), "checkpoint_sha256": old.sha256(checkpoint),
              "elapsed_seconds": time.monotonic()-started, "inputs_unchanged": True,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "input_hashes": inputs,
              "producer_sha256": old.sha256(__file__), "model_sources": _sources(), "model_runtime": _runtime(torch),
              "anatomical_accuracy": "NOT_ESTABLISHED"}
    old.write_json(output/"training.local.json", result)
    return result


def _load_model(path, digest, torch):
    checkpoint = old.external(path)
    old.require(old.sha256(checkpoint) == digest, "CHECKPOINT_HASH_MISMATCH")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    old.require(saved.get("version") == VERSION and saved.get("model_version") == learning.MODEL_VERSION
                and saved.get("config") == learning.CONFIG and saved.get("model_sources") == _sources()
                and saved.get("model_runtime") == _runtime(torch)
                and type(saved.get("selected_step")) is int and 0 < saved["selected_step"] <= learning.CONFIG["steps"],
                "CHECKPOINT_PROTOCOL_MISMATCH")
    model = learning.build_model()
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to("cuda"), saved


def native_output(labels, original, canonical, source):
    old.require(labels.dtype.kind in "iu" and np.all(labels >= 0)
                and labels.max(initial=0) <= 65535, "NATIVE_LABEL_RANGE_INVALID")
    full = np.zeros(canonical.shape, np.uint16)
    full[tuple(slice(a, b) for a, b in source["crop_bounds"])] = labels
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
                                               nib.orientations.io_orientation(original.affine))
    stored = nib.orientations.apply_orientation(full, transform)
    old.require(stored.shape == original.shape, "NATIVE_PROJECTION_FAILED")
    return stored


def _quality_flags(labels, registry):
    flags = []
    if any(np.any(np.take(labels, index, axis=axis)) for axis in range(3) for index in (0, -1)):
        flags.append("candidate_touches_crop_boundary")
    if any(r["flags"] for r in registry):
        flags.append("crosses_midline_requires_review")
    if not registry:
        flags.append("no_supported_candidates")
    return flags


def infer(manifest_path, output_path):
    from .boundary_data import _bounded_shape
    from .boundary_partition import partition, _validate_arrays
    manifest = _manifest(manifest_path, {"schema_version", "version", "case_id", "source", "checkpoint",
                                         "checkpoint_sha256", "partition_runtime"})
    old.require(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(manifest["case_id"])) is not None, "CASE_ID_INVALID")
    source = old.source_record(manifest["source"])
    shape = _bounded_shape(source)
    # The CPU partition budget is stricter than the native source limit. Check
    # it before device access, using a zero-stride view without allocating a crop.
    _validate_arrays(np.broadcast_to(np.zeros(1, np.float32), (2, *shape)), np.eye(4))
    runtime = _reference(manifest["partition_runtime"])
    bounds = np.asarray(source["crop_bounds"])
    old.require(int(np.prod(bounds[:, 1]-bounds[:, 0])) <= learning.CONFIG["maximum_native_voxels"], "NATIVE_RESOURCE_LIMIT")
    output = new_output(output_path)
    torch = _torch()
    _gpu_budget(torch)
    model, saved = _load_model(manifest["checkpoint"], manifest["checkpoint_sha256"], torch)
    original, canonical, hu, affine = old.load_crop(source)
    p = learning.probabilities(hu, model, torch, deadline=time.monotonic()+900)
    probability_path = output/"probabilities.local.npz"
    np.savez_compressed(probability_path, probabilities=p, affine=affine)
    labels, diagnostics = partition(p, affine, str(runtime))
    old.require(labels.shape == hu.shape, "PARTITION_SHAPE_MISMATCH")
    registry, candidate_counts = old.label_registry(labels, hu, affine, source["midline_ras_x"])
    flags = _quality_flags(labels, registry)
    stored = native_output(labels, original, canonical, source)
    artifact = output/"instances.local.nii.gz"
    nib.save(nib.Nifti1Image(stored, original.affine), artifact)
    old.require(old.sha256(source["input"]) == source["source_sha256"]
                and old.sha256(manifest["checkpoint"]) == manifest["checkpoint_sha256"]
                and old.sha256(runtime) == manifest["partition_runtime"]["sha256"], "INFERENCE_INPUT_CHANGED")
    result = {"status": "COMPLETE", "version": VERSION, "case_id": manifest["case_id"],
              "worker_identity": str(uuid.uuid4()), "worker_pid": os.getpid(),
              "source_sha256": source["source_sha256"], "source": source,
              "checkpoint_sha256": manifest["checkpoint_sha256"], "config_sha256": old.fingerprint(learning.CONFIG),
              "partition_runtime_sha256": manifest["partition_runtime"]["sha256"],
              "producer_sha256": old.sha256(__file__), "model_sources": _sources(), "model_runtime": _runtime(torch),
              "counts": {"left": None, "right": None, "total": None} if flags else candidate_counts,
              "candidate_counts": candidate_counts, "count_status": "WITHHELD" if flags else "DEVELOPMENT_ONLY",
              "registry": registry, "quality_flags": flags, "anatomical_completeness": "NOT_ESTABLISHED",
              "evidence_scopes": saved["evidence_scopes"], "native_shape": list(original.shape),
              "native_affine": original.affine.tolist(), "artifact_sha256": old.sha256(artifact),
              "probabilities_sha256": old.sha256(probability_path),
              "native_array_sha256": hashlib.sha256(stored.tobytes(order="C")).hexdigest(),
              "partition": diagnostics, "inputs_unchanged": True}
    old.write_json(output/"prediction.local.json", result)
    return result


def _evaluation_references(value):
    if value is None:
        return {key: None for key in ("left", "right", "total")}
    old.require(isinstance(value, dict) and bool(value) and set(value) <= {"left", "right", "total"}
                and all(v is None or (type(v) is int and v >= 0) for v in value.values()),
                "REFERENCE_COUNTS_INVALID")
    result = {key: value.get(key) for key in ("left", "right", "total")}
    if all(result[key] is not None for key in result):
        old.require(result["left"] + result["right"] == result["total"], "REFERENCE_COUNTS_INVALID")
    if result["total"] is not None:
        old.require(all(result[key] is None or result[key] <= result["total"] for key in ("left", "right")),
                    "REFERENCE_COUNTS_INVALID")
    return result


def _evaluation_failure(reference, entry):
    path = _reference(reference)
    failure = old.read_json(path)
    old.require(failure.get("version") == VERSION and failure.get("status") == "FAILED"
                and re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", str(failure.get("error_code"))) is not None
                and failure.get("command", "infer") == "infer"
                and failure.get("case_id", entry["case_id"]) == entry["case_id"]
                and failure.get("source_sha256", entry["source_sha256"]) == entry["source_sha256"],
                "FAILURE_RECORD_INVALID")
    return {"error_code": failure["error_code"], "record_sha256": reference["sha256"],
            "source_binding": "RECORDED" if "source_sha256" in failure else "DECLARED_BY_EVALUATION_ENTRY"}


def _evaluation_identity(prediction):
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None

    identities = tuple(prediction.get(field) for field in
                       ("checkpoint_sha256", "config_sha256", "partition_runtime_sha256", "producer_sha256"))
    sources, runtime = prediction.get("model_sources"), prediction.get("model_runtime")
    old.require(all(digest(value) for value in identities)
                and isinstance(sources, dict) and bool(sources)
                and all(isinstance(key, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}\.py", key)
                        and digest(value) for key, value in sources.items())
                and isinstance(runtime, dict) and set(runtime) == {"python", "torch", "cuda", "cudnn"}
                and all(isinstance(runtime[key], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+()-]{0,99}", runtime[key])
                        for key in ("python", "torch", "cuda"))
                and (runtime["cudnn"] is None or (type(runtime["cudnn"]) is int and runtime["cudnn"] >= 0)),
                "PREDICTION_CONFIGURATION_INVALID")
    # Compare saved provenance, so historical reports remain verifiable after source changes.
    return (*identities, old.fingerprint(sources), old.fingerprint(runtime))


def _evaluation_prediction(reference, entry, contexts):
    from .boundary_data import _bounded_shape

    path = _reference(reference)
    prediction = old.read_json(path)
    old.require(prediction.get("version") == VERSION and prediction.get("status") == "COMPLETE"
                and prediction.get("case_id") == entry["case_id"]
                and prediction.get("source_sha256") == entry["source_sha256"], "PREDICTION_PROVENANCE_MISMATCH")
    source = old.source_record(prediction.get("source"))
    old.require(source["source_sha256"] == entry["source_sha256"], "PREDICTION_SOURCE_MISMATCH")
    key = old.fingerprint(source)
    if key not in contexts:
        _bounded_shape(source)
        contexts[key] = old.load_crop(source)
    original, canonical, hu, affine = contexts[key]
    mask = old.external(path.parent / "instances.local.nii.gz")
    old.require(old.sha256(mask) == prediction.get("artifact_sha256"), "PREDICTION_ARTIFACT_CHANGED")
    image = nib.load(mask)
    old.require(image.shape == original.shape and image.get_data_dtype() == np.dtype(np.uint16)
                and np.array_equal(image.affine, original.affine)
                and prediction.get("native_shape") == list(original.shape)
                and np.array_equal(np.asarray(prediction.get("native_affine")), original.affine),
                "PREDICTION_SOURCE_GEOMETRY_MISMATCH")
    array = np.asanyarray(image.dataobj)
    old.require(array.dtype == np.dtype(np.uint16)
                and hashlib.sha256(array.tobytes(order="C")).hexdigest() == prediction.get("native_array_sha256"),
                "PREDICTION_ARRAY_CHANGED")
    full = np.asanyarray(nib.as_closest_canonical(image).dataobj)
    old.require(full.shape == canonical.shape, "PREDICTION_SOURCE_GEOMETRY_MISMATCH")
    crop = full[tuple(slice(a, b) for a, b in source["crop_bounds"])]
    old.require(crop.shape == hu.shape and np.count_nonzero(full) == np.count_nonzero(crop),
                "PREDICTION_LABELS_OUTSIDE_CROP")
    registry, candidates = old.label_registry(crop, hu, affine, source["midline_ras_x"])
    flags = _quality_flags(crop, registry)
    counts = {key: None for key in ("left", "right", "total")} if flags else candidates
    count_status = "WITHHELD" if flags else "DEVELOPMENT_ONLY"
    old.require(old.fingerprint(prediction.get("registry")) == old.fingerprint(registry)
                and old.fingerprint(prediction.get("candidate_counts")) == old.fingerprint(candidates)
                and old.fingerprint(prediction.get("counts")) == old.fingerprint(counts)
                and prediction.get("quality_flags") == flags
                and prediction.get("count_status") == count_status, "PREDICTION_REGISTRY_COUNTS_MISMATCH")
    old.require(type(prediction.get("worker_pid")) is int and prediction["worker_pid"] > 0
                and isinstance(prediction.get("worker_identity"), str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", prediction["worker_identity"]) is not None,
                "PREDICTION_WORKER_INVALID")
    identities = _evaluation_identity(prediction)
    old.require(old.sha256(source["input"]) == source["source_sha256"]
                and old.sha256(path) == reference["sha256"]
                and old.sha256(mask) == prediction["artifact_sha256"], "EVALUATION_INPUT_CHANGED")
    return prediction, array, source, identities


def _evaluation_instances(sample, prediction, array, source_hash):
    from .boundary_data import _bounded_shape, load_sample

    old.require(isinstance(sample, dict) and isinstance(sample.get("source"), dict)
                and sample["source"].get("source_sha256") == source_hash, "REFERENCE_SOURCE_MISMATCH")
    sample_source = old.source_record(sample["source"])
    old.require(sample_source == prediction["source"], "REFERENCE_SOURCE_MISMATCH")
    _bounded_shape(sample_source)
    _, _, arrays = load_sample(sample)
    canonical = nib.as_closest_canonical(nib.Nifti1Image(array, np.asarray(prediction["native_affine"])))
    crop = np.asanyarray(canonical.dataobj[tuple(slice(x, y) for x, y in sample_source["crop_bounds"])])
    old.require(crop.shape == arrays["instance_ids"].shape, "REFERENCE_CROP_MISMATCH")
    region = arrays["evaluation_region"]
    coverage = {"evaluated_voxels": int(np.count_nonzero(region)),
                "region_fraction": float(np.count_nonzero(region) / region.size)}
    if not np.any(region):
        return {"status": "NOT_SCORED_NO_COMPLETE_REGION", "reference_scope": sample["scope"], **coverage}
    old.require(np.all(arrays["instance_ids"][region] >= 0), "REFERENCE_REGION_HAS_UNKNOWN_INSTANCES")
    metric = instance_agreement(np.where(region, crop, 0).astype(np.int32),
                                np.where(region, arrays["instance_ids"], 0).astype(np.int32))
    metric["scope"] = "supplied_complete_evaluation_region_only"
    metric["reference_scope"] = sample["scope"]
    metric["clinical_accuracy_scope"] = {
        "synthetic_truth": "not_established_supplied_synthetic_truth_only",
        "conditional_native": "not_established_supplied_conditional_native_reference_only",
        "adjudicated_expert": "only_supplied_adjudicated_expert_reference_instances",
    }[sample["scope"]]
    predicted_foreground = (crop > 0) & region
    reference_foreground = (arrays["instance_ids"] > 0) & region
    predicted_voxels = int(np.count_nonzero(predicted_foreground))
    reference_voxels = int(np.count_nonzero(reference_foreground))
    overlap_voxels = int(np.count_nonzero(predicted_foreground & reference_foreground))
    metric.update(coverage, predicted_foreground_voxels=predicted_voxels,
                  reference_foreground_voxels=reference_voxels,
                  foreground_overlap_voxels=overlap_voxels,
                  reference_foreground_voxel_coverage=(overlap_voxels / reference_voxels if reference_voxels else None),
                  foreground_dice=(2 * overlap_voxels / (predicted_voxels + reference_voxels)
                                   if predicted_voxels + reference_voxels else 1.))
    return metric


def evaluate(manifest_path, output_path):
    manifest = _manifest(manifest_path, {"schema_version", "version", "entries"})
    entries = manifest["entries"]
    old.require(isinstance(entries, list) and bool(entries), "EVALUATION_ENTRIES_INVALID")
    cases, seen, checkpoints = [], set(), set()
    paired_error = paired_sides = exact_sides = exact_studies = referenced_sides = 0
    states = Counter()
    for entry in entries:
        old.require(isinstance(entry, dict)
                    and {"case_id", "source_sha256", "predictions"} <= set(entry)
                    <= {"case_id", "source_sha256", "predictions", "reference_counts", "sample", "failure"},
                    "EVALUATION_ENTRY_INVALID")
        case_id = entry["case_id"]
        old.require(isinstance(case_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", case_id)
                    and case_id not in seen and isinstance(entry["predictions"], list)
                    and isinstance(entry["source_sha256"], str)
                    and re.fullmatch(r"[a-f0-9]{64}", entry["source_sha256"]), "EVALUATION_CASE_INVALID")
        seen.add(case_id)
        reference = _evaluation_references(entry.get("reference_counts"))
        referenced_sides += sum(reference[side] is not None for side in ("left", "right"))
        old.require(len(entry["predictions"]) in (0, 2)
                    and not (entry["predictions"] and "failure" in entry), "TWO_FRESH_PREDICTIONS_REQUIRED")
        row = {"case_id": case_id, "state": "NOT_RUN", "repeat_equal": None, "validation_passes": False,
               "reference_counts": reference, "side_errors": {"left": None, "right": None},
               "paired_sides": 0, "absolute_side_error": None}
        if not entry["predictions"]:
            if "failure" in entry:
                row.update(state="FAILED", failure=_evaluation_failure(entry["failure"], entry))
            if "sample" in entry:
                row["instances"] = {"status": "NOT_SCORED_NO_PREDICTION"}
            states[row["state"]] += 1
            cases.append(row)
            continue
        contexts = {}
        predictions = [_evaluation_prediction(item, entry, contexts) for item in entry["predictions"]]
        a, b = predictions[0][0], predictions[1][0]
        old.require(a["worker_identity"] != b["worker_identity"] and a["worker_pid"] != b["worker_pid"],
                    "FRESH_WORKERS_REQUIRED")
        checkpoints.update(item[3] for item in predictions)
        repeat = (np.array_equal(predictions[0][1], predictions[1][1])
                  and predictions[0][2] == predictions[1][2]
                  and all(a[key] == b[key] for key in ("registry", "counts", "candidate_counts", "count_status",
                                                       "quality_flags", "native_shape", "native_affine")))
        row.update(state=("WITHHELD" if a["count_status"] == "WITHHELD" else "AVAILABLE") if repeat else "REPEAT_MISMATCH",
                   repeat_equal=repeat, validation_passes=repeat,
                   counts=a["counts"] if repeat else {key: None for key in ("left", "right", "total")},
                   candidate_counts=a["candidate_counts"] if repeat else None)
        if not repeat:
            row["repeat_candidate_counts"] = [a["candidate_counts"], b["candidate_counts"]]
        states[row["state"]] += 1
        exact = []
        for side in ("left", "right"):
            count = row["counts"][side]
            if count is not None and reference[side] is not None:
                error = abs(count - reference[side])
                row["side_errors"][side] = error
                paired_error += error; paired_sides += 1
                exact_sides += error == 0; exact.append(error == 0)
        row["paired_sides"] = len(exact)
        row["absolute_side_error"] = sum(error for error in row["side_errors"].values()
                                         if error is not None) if exact else None
        exact_studies += len(exact) == 2 and all(exact)
        if "sample" in entry:
            row["instances"] = (_evaluation_instances(entry["sample"], a, predictions[0][1], entry["source_sha256"])
                                if repeat else {"status": "NOT_SCORED_REPEAT_MISMATCH"})
        cases.append(row)
        del predictions, contexts
    old.require(len(checkpoints) <= 1, "COHORT_CHECKPOINT_OR_CONFIGURATION_MISMATCH")
    result = {"status": "COMPLETE", "version": VERSION, "scheduled_studies": len(entries),
              "validation_passes": all(case["validation_passes"] for case in cases),
              "validation_scope": "fresh_worker_repeatability_only",
              "scheduled_sides": 2 * len(entries), "states": dict(states), "paired_sides": paired_sides,
              "referenced_sides": referenced_sides, "unpaired_reference_sides": referenced_sides - paired_sides,
              "absolute_side_error": paired_error if paired_sides else None,
              "exact_sides": exact_sides, "exact_bilateral_studies": exact_studies,
              "repeatable_studies": sum(c["repeat_equal"] is True for c in cases), "cases": cases,
              "evaluation_scope": "exposed_development_not_external_validation", "anatomical_accuracy": "NOT_ESTABLISHED"}
    output = new_output(output_path)
    old.write_json(output / "evaluation.local.json", result)
    return result
