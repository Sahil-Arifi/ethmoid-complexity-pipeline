"""Count-blind diagnosis of the frozen, unpromoted legacy affinity checkpoint."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

from . import instance_experiment as legacy
from . import instance_learning as core

VERSION = "boundary-instance-v1"
require = legacy.require


def _reference(value, code):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, code)
    require(isinstance(value["path"], str) and isinstance(value["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is not None, code)
    return value


def _manifest(value):
    required = {"schema_version", "version", "source", "checkpoint", "checkpoint_sha256", "annotation_manifest"}
    require(isinstance(value, dict) and required <= set(value)
            and set(value) <= required | {"annotation_manifest_sha256", "baseline_prediction"}
            and type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["version"] == VERSION, "DIAGNOSTIC_MANIFEST_INVALID")
    checkpoint = _reference({"path": value["checkpoint"], "sha256": value["checkpoint_sha256"]},
                            "DIAGNOSTIC_CHECKPOINT_INVALID")
    annotation = value["annotation_manifest"]
    if isinstance(annotation, str):
        annotation = {"path": annotation, "sha256": value.get("annotation_manifest_sha256")}
    else:
        require("annotation_manifest_sha256" not in value, "DIAGNOSTIC_ANNOTATION_INVALID")
    annotation = _reference(annotation, "DIAGNOSTIC_ANNOTATION_INVALID")
    baseline = value.get("baseline_prediction")
    if baseline is not None:
        _reference(baseline, "DIAGNOSTIC_BASELINE_INVALID")
    return checkpoint, annotation, baseline


def _checked(reference, code):
    path = legacy.external(reference["path"])
    require(legacy.sha256(path) == reference["sha256"], code)
    return path


def _producer():
    return {"source_files": {Path(module.__file__).name: legacy.sha256(Path(module.__file__))
                             for module in (legacy, core, sys.modules[__name__])},
            "python_version": sys.version.split()[0], "legacy_version": legacy.VERSION,
            "legacy_model_version": core.MODEL_VERSION, "legacy_config": legacy.CONFIG,
            "legacy_config_sha256": legacy.fingerprint(legacy.CONFIG)}


def _reproduce(source, checkpoint):
    """Use the original closed architecture, processing, device and decoder."""
    from .models import _require_offline_runtime

    _require_offline_runtime()
    torch = legacy.deterministic_torch()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved.get("version") == legacy.VERSION and saved.get("model_version") == core.MODEL_VERSION
            and saved.get("config") == legacy.CONFIG
            and saved.get("steps_completed") == legacy.CONFIG["steps"], "CHECKPOINT_PROTOCOL_MISMATCH")
    model = core.build_model()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to("cuda")
    original, canonical, hu, affine = legacy.load_crop(source)
    grid, valid = legacy.model_grid(hu, affine)
    probabilities = legacy.tiled_probabilities(grid, model, torch)
    probabilities[0, ~valid] = 0
    decoded = core.decode_probabilities(probabilities)
    native_crop = legacy.native_labels(decoded, hu.shape, affine)
    full = np.zeros(canonical.shape, np.uint16)
    full[tuple(slice(a, b) for a, b in source["crop_bounds"])] = native_crop
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
                                                nib.orientations.io_orientation(original.affine))
    native = nib.orientations.apply_orientation(full, orientation)
    require(native.shape == original.shape, "NATIVE_PROJECTION_FAILED")
    return {"probabilities": probabilities, "valid": valid, "decoded": decoded,
            "native_crop": native_crop, "native": native, "native_affine": original.affine,
            "hu": hu, "crop_affine": affine,
            "training_annotation_manifest_sha256": saved.get("annotation_manifest_sha256"),
            "torch_version": str(torch.__version__)}


def _save_prediction(output, result, source, checkpoint_reference):
    affine = result["crop_affine"]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    model_affine = affine.copy()
    model_affine[:3, :3] = affine[:3, :3] / spacing * legacy.CONFIG["spacing_mm"]
    coordinates = {f"crop_index_axis_{axis}": np.arange(size, dtype=np.float64)
                   * legacy.CONFIG["spacing_mm"] / spacing[axis]
                   for axis, size in enumerate(result["valid"].shape)}
    arrays = output / "probabilities.local.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, probabilities=result["probabilities"], valid=result["valid"],
                            model_affine=model_affine, crop_affine=affine, **coordinates)
    native = output / "instances.local.nii.gz"
    nib.save(nib.Nifti1Image(result["native"], result["native_affine"]), native)
    producer = _producer()
    producer["torch_version"] = result["torch_version"]
    record = {"schema_version": 1, "version": VERSION, "status": "SOURCE_ONLY_PREDICTION_CLOSED",
              "source": source, "checkpoint_sha256": checkpoint_reference["sha256"],
              "probabilities_sha256": legacy.sha256(arrays), "native_sha256": legacy.sha256(native),
              "grid_shape": list(result["valid"].shape), "crop_shape": list(result["hu"].shape),
              "coordinate_representation": "separable_native_crop_index_axes",
              "producer": producer, "annotation_loaded": False}
    legacy.write_json(output / "prediction-closed.local.json", record)
    return record


def _baseline(reference, result, source, checkpoint_hash):
    if reference is None:
        return {"status": "NOT_SUPPLIED"}
    path = _checked(reference, "BASELINE_HASH_MISMATCH")
    record = legacy.read_json(path)
    require(record.get("status") == "COMPLETE"
            and record.get("source_sha256") == source["source_sha256"]
            and record.get("checkpoint_sha256") == checkpoint_hash
            and record.get("config_sha256") == legacy.fingerprint(legacy.CONFIG), "BASELINE_BINDING_MISMATCH")
    artifact = _checked({"path": str(path.parent / "instances.local.nii.gz"),
                         "sha256": record.get("artifact_sha256")}, "BASELINE_ARTIFACT_MISMATCH")
    image = nib.load(artifact)
    equal = (image.shape == result["native"].shape
             and np.array_equal(image.affine, result["native_affine"])
             and np.array_equal(np.asanyarray(image.dataobj), result["native"]))
    require(equal, "BASELINE_REPRODUCTION_MISMATCH")
    return {"status": "EXACT_NATIVE_MATCH", "prediction_sha256": reference["sha256"],
            "native_array_equal": True, "native_affine_equal": True}


def _positive_graph(probabilities):
    foreground = probabilities[0] >= core.THRESHOLD
    flat = np.arange(foreground.size, dtype=np.int32).reshape(foreground.shape)
    rows, columns = [], []
    for channel, offset in enumerate(core.OFFSETS, 1):
        a, b = core.edge_slices(foreground.shape, offset)
        selected = foreground[a] & foreground[b] & (probabilities[(channel, *a)] >= core.THRESHOLD)
        rows.append(flat[a][selected])
        columns.append(flat[b][selected])
    a, b = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(2 * len(a), np.uint8),
                        (np.concatenate((a, b)), np.concatenate((b, a)))),
                       shape=(foreground.size, foreground.size)).tocsr()
    graph.sort_indices()
    return graph


def _traversal(coordinates, labels, foreground):
    counts = Counter()
    for point in coordinates:
        index = tuple(point)
        kind = ("known_background" if labels[index] == 0 or foreground[index] == 0 else
                "known_instance" if labels[index] > 0 else
                "known_foreground_without_identity" if foreground[index] == 1 else "unknown")
        counts[kind] += 1
    return {key: counts[key] for key in ("known_instance", "known_foreground_without_identity",
                                        "known_background", "unknown")}


def _path(probabilities, labels, foreground, decoded, pair, graph):
    left, right = pair
    a = np.unique(decoded[labels == left]); b = np.unique(decoded[labels == right])
    shared = np.intersect1d(a[a > 0], b[b > 0])
    record = {"annotation_ids": list(pair), "connected": bool(len(shared)),
              "selection": "smallest_shared_component_then_smallest_source_flat_index_then_sorted_neighbor_BFS"}
    if not len(shared):
        record["status"] = "DISCONNECTED_OR_MISSING_FOREGROUND"
        return record
    selected = int(shared[0])
    start = int(np.flatnonzero((labels == left) & (decoded == selected))[0])
    order, predecessors = breadth_first_order(graph, start, directed=True, return_predecessors=True)
    targets = (labels.ravel()[order] == right) & (decoded.ravel()[order] == selected)
    require(np.any(targets), "PATH_COMPONENT_INCONSISTENT")
    end = int(order[np.flatnonzero(targets)[0]])
    nodes = [end]
    while nodes[-1] != start:
        parent = int(predecessors[nodes[-1]])
        require(parent >= 0 and len(nodes) <= labels.size, "PATH_PREDECESSOR_INVALID")
        nodes.append(parent)
    coordinates = np.array(np.unravel_index(nodes[::-1], labels.shape)).T
    steps, offsets, affinities = [], Counter(), []
    for a, b in zip(coordinates[:-1], coordinates[1:]):
        delta = b - a
        absolute = tuple(np.abs(delta).tolist())
        channel = core.OFFSETS.index(absolute) + 1
        anchor = a if np.all(delta >= 0) else b
        affinity = float(probabilities[(channel, *anchor)])
        affinities.append(affinity)
        offsets[",".join(map(str, delta.tolist()))] += 1
        length = int(np.abs(delta).sum())
        if length > 1:
            interior = [a + np.sign(delta) * step for step in range(1, length)]
            traversal = _traversal(interior, labels, foreground)
            steps.append({"from": a.tolist(), "to": b.tolist(), "affinity": affinity,
                          "intermediate_traversal": traversal,
                          "predicted_background_voxels": sum(int(probabilities[(0, *p)] < core.THRESHOLD)
                                                             for p in interior)})
    record.update(status="POSITIVE_PATH", coordinates=coordinates.tolist(), edge_count=len(affinities),
                  bottleneck_affinity=min(affinities), offset_counts=dict(sorted(offsets.items())),
                  node_traversal=_traversal(coordinates, labels, foreground), long_edges=steps,
                  long_edge_background_bypasses=sum(s["predicted_background_voxels"] > 0 for s in steps),
                  long_edge_known_background_crossings=sum(s["intermediate_traversal"]["known_background"] > 0 for s in steps))
    return record


def audit_arrays(probabilities, valid, model_labels, model_foreground, distinct_pairs, *, decoded=None):
    """Pure array diagnostics. Unknown supervision never becomes a negative."""
    decoded = core.decode_probabilities(probabilities) if decoded is None else decoded
    targets, known = core.build_targets(model_labels, distinct_pairs=distinct_pairs,
                                        foreground_labels=model_foreground)
    require(valid.shape == model_labels.shape == model_foreground.shape == probabilities.shape[1:]
            and valid.dtype == np.bool_, "DIAGNOSTIC_ARRAY_SHAPE_INVALID")
    heads = []
    for channel in range(7):
        domain = valid.copy() if channel == 0 else np.zeros(valid.shape, bool)
        if channel:
            a, b = core.edge_slices(valid.shape, core.OFFSETS[channel - 1])
            domain[a] = valid[a] & valid[b]
        mask = known[channel] & domain
        truth, predicted = targets[channel][mask] > 0, probabilities[channel][mask] >= core.THRESHOLD
        heads.append({"channel": channel, "offset": None if not channel else list(core.OFFSETS[channel - 1]),
                      "positive": int(truth.sum()), "negative": int((~truth).sum()),
                      "unknown": int(np.sum(domain & ~known[channel])), "outside_domain": int(np.sum(~domain)),
                      "true_positive": int(np.sum(truth & predicted)), "true_negative": int(np.sum(~truth & ~predicted)),
                      "false_positive": int(np.sum(~truth & predicted)), "false_negative": int(np.sum(truth & ~predicted))})
    graph = _positive_graph(probabilities) if distinct_pairs else None
    pairs = sorted(set(tuple(sorted(pair)) for pair in distinct_pairs))
    paths = [_path(probabilities, model_labels, model_foreground, decoded, pair, graph) for pair in pairs]
    return {"heads": heads, "distinct_pair_paths": paths}


def _native_mechanism(predicted, labels, foreground):
    regions = []
    for identity in np.unique(labels[labels > 0]):
        values = predicted[labels == identity]
        components = np.unique(values[values > 0])
        regions.append({"annotation_id": int(identity), "known_voxels": len(values),
                        "covered_voxels": int(np.sum(values > 0)), "prediction_ids": components.tolist(),
                        "fragmented": len(components) > 1, "fully_covered": bool(np.all(values > 0))})
    positive = (labels > 0) | (foreground == 1)
    negative = (labels == 0) | (foreground == 0)
    return {"cores": regions, "known_foreground_voxels": int(positive.sum()),
            "missing_known_foreground_voxels": int(np.sum(positive & (predicted == 0))),
            "explicit_exclusion_voxels": int(negative.sum()),
            "violated_exclusion_voxels": int(np.sum(negative & (predicted > 0))),
            "unknown_voxels_unscored": int(np.sum(~(positive | negative)))}


def diagnose(manifest_path, output_path):
    """Seal source-only predictions before reading any annotation data."""
    output = None
    try:
        manifest = legacy.read_json(manifest_path)
        checkpoint_reference, annotation_reference, baseline = _manifest(manifest)
        source = legacy.source_record(manifest["source"])
        checkpoint = _checked(checkpoint_reference, "CHECKPOINT_HASH_MISMATCH")
        from .boundary_io import new_output
        output = new_output(output_path)
        result = _reproduce(source, checkpoint)
        require(legacy.sha256(source["input"]) == source["source_sha256"]
                and legacy.sha256(checkpoint) == checkpoint_reference["sha256"], "DIAGNOSTIC_INPUT_CHANGED")
        closed = _save_prediction(output, result, source, checkpoint_reference)
        baseline_match = _baseline(baseline, result, source, checkpoint_reference["sha256"])
        annotation_path = _checked(annotation_reference, "ANNOTATION_MANIFEST_HASH_MISMATCH")
        require(result["training_annotation_manifest_sha256"] == annotation_reference["sha256"],
                "CHECKPOINT_ANNOTATION_MISMATCH")
        annotation = legacy.read_json(annotation_path)
        require(legacy.source_record(annotation.get("source")) == source, "DIAGNOSTIC_ANNOTATION_SOURCE_MISMATCH")
        readiness = legacy.inspect_readiness(annotation)
        require(readiness["training_allowed"], "DIAGNOSTIC_ANNOTATION_NOT_READY")
        labels, foreground, annotation_array_path = legacy._annotation(annotation, source, result["hu"], result["crop_affine"])
        _, valid, model_labels, model_foreground = legacy.model_grid(result["hu"], result["crop_affine"], labels, foreground)
        audit = audit_arrays(result["probabilities"], valid, model_labels, model_foreground,
                             annotation.get("distinct_pairs", []), decoded=result["decoded"])
        preserved = (legacy.sha256(source["input"]) == source["source_sha256"]
                     and legacy.sha256(checkpoint) == checkpoint_reference["sha256"]
                     and legacy.sha256(annotation_path) == annotation_reference["sha256"]
                     and legacy.sha256(annotation_array_path) == annotation["annotation"]["sha256"]
                     and legacy.sha256(output / "probabilities.local.npz") == closed["probabilities_sha256"])
        require(preserved, "DIAGNOSTIC_INPUT_CHANGED")
        report = {"schema_version": 1, "version": VERSION, "status": "COMPLETE", **audit,
                  "native_mechanism": _native_mechanism(result["native_crop"], labels, foreground),
                  "baseline_match": baseline_match, "source_sha256": source["source_sha256"],
                  "checkpoint_sha256": checkpoint_reference["sha256"],
                  "annotation_manifest_sha256": annotation_reference["sha256"],
                  "prediction_closed_sha256": legacy.sha256(output / "prediction-closed.local.json"),
                  "inputs_unchanged": preserved, "threshold": core.THRESHOLD,
                  "interpretation": "Conditional known-region and path diagnosis only; unknown anatomy is unscored. No training, threshold selection or count-accuracy claim."}
        legacy.write_json(output / "diagnosis.local.json", report)
        return report
    except Exception as error:
        code = legacy.failure_code(error)
        if output is not None:
            try:
                legacy.write_json(output / "failure.local.json", {"status": "FAILED", "version": VERSION, "error_code": code})
            except Exception:
                pass
        raise legacy.ExperimentError(code) from None
