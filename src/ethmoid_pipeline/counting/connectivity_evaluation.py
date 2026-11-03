"""CT-only prediction and post-prediction evaluation for the isolated fork."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import uuid

import nibabel as nib
import numpy as np

from . import instance_experiment as ex
from . import instance_learning as core

_WORKER_IDENTITY = str(uuid.uuid4())
_INFERENCE_FIELDS = {"schema_version", "case_id", "source", "checkpoint", "checkpoint_sha256"}
_SHARED_GEOMETRY = ("spacing_mm", "patch_size", "inference_stride", "threshold",
                    "minimum_native_voxels", "channels")


def _runner():
    # CLI dispatch imports this module after the training runner has initialized.
    from . import connectivity_experiment
    return connectivity_experiment


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inference_manifest(manifest):
    ex.require(set(manifest) == _INFERENCE_FIELDS and manifest.get("schema_version") == 1
               and isinstance(manifest.get("case_id"), str) and manifest["case_id"].isascii()
               and manifest["case_id"].replace("_", "").isalnum(), "INFERENCE_MANIFEST_INVALID")
    source = ex.source_record(manifest["source"])
    checkpoint = ex.external(manifest["checkpoint"])
    ex.require(_hash(manifest["checkpoint_sha256"])
               and ex.sha256(checkpoint) == manifest["checkpoint_sha256"], "CHECKPOINT_HASH_MISMATCH")
    return source, checkpoint


def project_native(labels, original, canonical, bounds):
    """Put crop labels back on the unchanged input grid and orientation."""
    full = np.zeros(canonical.shape, np.uint16)
    full[tuple(slice(a, b) for a, b in bounds)] = labels
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
                                                nib.orientations.io_orientation(original.affine))
    stored = nib.orientations.apply_orientation(full, orientation)
    ex.require(stored.shape == original.shape, "NATIVE_PROJECTION_FAILED")
    return stored


def infer(manifest_path, output_path):
    """Prediction opens only the strict CT/crop/checkpoint manifest and its inputs."""
    manifest = ex.read_json(manifest_path)
    source, checkpoint = _inference_manifest(manifest)
    run = _runner()
    ex.require(all(run.CONFIG[key] == ex.CONFIG[key] for key in _SHARED_GEOMETRY),
               "SHARED_GEOMETRY_PROTOCOL_MISMATCH")
    manifest_hash = ex.sha256(manifest_path)
    torch = ex.deterministic_torch()
    model, saved = run.load_checkpoint(checkpoint, torch)
    model.to("cuda")
    original, canonical, hu, affine = ex.load_crop(source)
    hu_hash = hashlib.sha256(hu.tobytes(order="C")).hexdigest()
    affine_before = affine.copy()
    grid, valid = ex.model_grid(hu, affine)
    probabilities = ex.tiled_probabilities(grid, model, torch)
    probabilities[0, ~valid] = 0
    decoded = core.decode_probabilities(probabilities)
    labels = ex.native_labels(decoded, hu.shape, affine)
    registry, counts = ex.label_registry(labels, hu, affine, source["midline_ras_x"])
    stored = project_native(labels, original, canonical, source["crop_bounds"])
    ex.require(ex.sha256(source["input"]) == source["source_sha256"]
               and ex.sha256(checkpoint) == manifest["checkpoint_sha256"]
               and ex.sha256(manifest_path) == manifest_hash
               and hashlib.sha256(hu.tobytes(order="C")).hexdigest() == hu_hash
               and np.array_equal(affine, affine_before), "INFERENCE_INPUT_CHANGED")
    output = ex.new_output(output_path)
    artifact = output/"instances.local.nii.gz"
    nib.save(nib.Nifti1Image(stored, original.affine), artifact)
    result = {"status": "COMPLETE", "version": run.VERSION, "case_id": manifest["case_id"],
              "worker_identity": _WORKER_IDENTITY, "worker_pid": os.getpid(),
              "counts": counts, "registry": registry, "source_sha256": source["source_sha256"],
              "source_geometry_sha256": ex.fingerprint(source),
              "inference_manifest_path": str(ex.external(manifest_path)),
              "inference_manifest_sha256": manifest_hash,
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "config_sha256": ex.fingerprint(run.CONFIG),
              "protocol_sha256": saved.get("protocol_sha256"),
              "annotation_manifest_sha256": saved.get("annotation_manifest_sha256"),
              "native_array_sha256": hashlib.sha256(stored.tobytes(order="C")).hexdigest(),
              "native_affine": original.affine.tolist(), "native_shape": list(original.shape),
              "artifact_sha256": ex.sha256(artifact), "inputs_unchanged": True}
    ex.write_json(output/"prediction.local.json", result)
    return result


def _prediction(path, *, case_id=None, source_sha256=None, source_geometry_sha256=None):
    """Validate a closed prediction without loading a model or annotation."""
    run = _runner()
    record = ex.read_json(path)
    ex.require(record.get("status") == "COMPLETE" and record.get("version") == run.VERSION
               and record.get("config_sha256") == ex.fingerprint(run.CONFIG)
               and record.get("inputs_unchanged") is True, "PREDICTION_PROTOCOL_INVALID")
    manifest_path = ex.external(record.get("inference_manifest_path"))
    ex.require(ex.sha256(manifest_path) == record.get("inference_manifest_sha256"),
               "PREDICTION_MANIFEST_CHANGED")
    manifest = ex.read_json(manifest_path)
    source, _ = _inference_manifest(manifest)
    ex.require(record.get("case_id") == manifest["case_id"]
               and (case_id is None or record["case_id"] == case_id)
               and record.get("source_sha256") == source["source_sha256"]
               and (source_sha256 is None or record["source_sha256"] == source_sha256)
               and record.get("source_geometry_sha256") == ex.fingerprint(source)
               and (source_geometry_sha256 is None or record["source_geometry_sha256"] == source_geometry_sha256)
               and record.get("checkpoint_sha256") == manifest["checkpoint_sha256"],
               "PREDICTION_SOURCE_MISMATCH")
    artifact = ex.external(Path(path).parent/"instances.local.nii.gz")
    ex.require(ex.sha256(artifact) == record.get("artifact_sha256"), "PREDICTION_ARTIFACT_CHANGED")
    image = nib.load(artifact)
    values = np.asanyarray(image.dataobj)
    ex.require(values.dtype == np.dtype(np.uint16) and values.ndim == 3
               and hashlib.sha256(values.tobytes(order="C")).hexdigest() == record.get("native_array_sha256")
               and list(values.shape) == record.get("native_shape")
               and np.shape(record.get("native_affine")) == (4, 4)
               and np.allclose(image.affine, record["native_affine"], rtol=0, atol=1e-6),
               "PREDICTION_NATIVE_ARRAY_INVALID")
    original, canonical, hu, affine = ex.load_crop(source)
    ex.require(image.shape == original.shape
               and np.allclose(image.affine, original.affine, rtol=0, atol=1e-6),
               "PREDICTION_SOURCE_GEOMETRY_MISMATCH")
    canonical_prediction = nib.as_closest_canonical(image)
    ex.require(canonical_prediction.shape == canonical.shape
               and np.allclose(canonical_prediction.affine, canonical.affine, rtol=0, atol=1e-6),
               "PREDICTION_SOURCE_GEOMETRY_MISMATCH")
    crop = np.asarray(canonical_prediction.dataobj[tuple(slice(a, b) for a, b in source["crop_bounds"])])
    ex.require(np.count_nonzero(values) == np.count_nonzero(crop), "PREDICTION_OUTSIDE_CROP")
    ids, sizes = np.unique(crop[crop > 0], return_counts=True)
    ex.require(ids.tolist() == list(range(1, len(ids)+1))
               and np.all(sizes >= run.CONFIG["minimum_native_voxels"]), "PREDICTION_INSTANCE_IDS_INVALID")
    registry, counts = ex.label_registry(crop, hu, affine, source["midline_ras_x"])
    ex.require(ex.fingerprint(record.get("registry")) == ex.fingerprint(registry)
               and record.get("counts") == counts, "PREDICTION_REGISTRY_INVALID")
    ex.require(_hash(record.get("annotation_manifest_sha256")), "PREDICTION_ANNOTATION_BINDING_INVALID")
    ex.require(_hash(record.get("protocol_sha256")), "PREDICTION_PROTOCOL_BINDING_INVALID")
    return record, crop, source


def score_supported_regions(values, labels, foreground, distinct_pairs):
    """Unknown voxels and unannotated whole-cell extents remain unscored."""
    ex.require(values.shape == labels.shape == foreground.shape, "MECHANISM_SHAPE_MISMATCH")
    regions, assignment, observed_ids = [], {}, {}
    for identity in np.unique(labels[labels > 0]):
        sampled = values[labels == identity]
        ids = np.unique(sampled[sampled > 0]).tolist()
        observed_ids[int(identity)] = ids
        assignment[int(identity)] = ids[0] if len(ids) == 1 and np.all(sampled > 0) else None
        regions.append({"annotation_id": int(identity), "known_voxels": len(sampled),
                        "covered_voxels": int(np.sum(sampled > 0)), "prediction_ids": ids,
                        "split": len(ids) > 1, "fully_covered": bool(np.all(sampled > 0))})
    pairs = [{"annotation_ids": [a, b], "prediction_ids": [assignment.get(a), assignment.get(b)],
              "observed_prediction_ids": [observed_ids.get(a, []), observed_ids.get(b, [])],
              "shared_prediction_ids": sorted(set(observed_ids.get(a, [])) & set(observed_ids.get(b, []))),
              "separate": assignment.get(a) is not None and assignment.get(b) is not None
                          and assignment[a] != assignment[b]} for a, b in distinct_pairs]
    positive = (foreground == 1) | (labels > 0)
    negative = (foreground == 0) | (labels == 0)
    missing = int(np.sum(positive & (values == 0)))
    false_positive = int(np.sum(negative & (values > 0)))
    gates = {"three_supported_distinctions": len(pairs) == 3 and all(p["separate"] for p in pairs),
             "annotated_within_cell_connections": bool(regions) and all(r["fully_covered"] and not r["split"] for r in regions),
             "known_foreground_coverage": bool(np.any(positive)) and missing == 0,
             "explicit_exclusions_preserved": false_positive == 0}
    return {"gates": gates, "regions": regions, "distinctions": pairs,
            "supported_merges": [pair for pair in pairs if pair["shared_prediction_ids"]],
            "supported_splits": [region for region in regions if region["split"]],
            "known_foreground_voxels": int(np.sum(positive)), "missing_known_foreground_voxels": missing,
            "known_foreground_coverage_fraction": float(np.sum(positive & (values > 0))/np.sum(positive)) if np.any(positive) else None,
            "explicit_background_voxels": int(np.sum(negative)), "false_positive_known_background_voxels": false_positive,
            "unknown_voxels_unscored": int(np.sum(~(positive | negative))),
            "whole_cell_false_splits": "UNMEASURED_OUTSIDE_PROVISIONAL_CORES"}


def mechanism(manifest_path, output_path):
    manifest = ex.read_json(manifest_path)
    ex.require(set(manifest) == {"schema_version", "annotation_manifest", "prediction"}
               and manifest.get("schema_version") == 1, "MECHANISM_MANIFEST_INVALID")
    # Finish checking the saved prediction before opening its evaluation labels.
    prediction, values, source = _prediction(manifest["prediction"], case_id="Case024_Series002")
    annotation = ex.read_json(manifest["annotation_manifest"])
    ready = ex.inspect_readiness(annotation)
    ex.require(ready["training_allowed"], "MECHANISM_ANNOTATIONS_NOT_READY")
    annotation_hash = ex.sha256(manifest["annotation_manifest"])
    ex.require(prediction["annotation_manifest_sha256"] == annotation_hash
               and ex.fingerprint(ex.source_record(annotation["source"])) == ex.fingerprint(source),
               "MECHANISM_ANNOTATION_CHANGED")
    _, _, hu, affine = ex.load_crop(source)
    labels, foreground, _ = ex._annotation(annotation, source, hu, affine)
    scored = score_supported_regions(values, labels, foreground, annotation.get("distinct_pairs", []))
    run = _runner()
    result = {**scored, "status": "PASS_CONDITIONAL_MECHANISM" if all(scored["gates"].values()) else "FAIL",
              "version": run.VERSION, "config_sha256": ex.fingerprint(run.CONFIG),
              "annotation_manifest_sha256": annotation_hash, "prediction_sha256": ex.sha256(manifest["prediction"]),
              "checkpoint_sha256": prediction["checkpoint_sha256"], "protocol_sha256": prediction.get("protocol_sha256"),
              "source_match_uncertainty": ready["source_match_uncertainty"],
              "boundary_uncertainty": ready["boundary_uncertainty"],
              "interpretation": "Conditional partial-core agreement; whole-cell false splits and clinical accuracy remain unmeasured."}
    output = ex.new_output(output_path)
    ex.write_json(output/"mechanism.local.json", result)
    return result


def _identical_fresh_repeats(paths, records):
    a, b = records
    workers = [record.get("worker_identity") for record in records]
    pids = [record.get("worker_pid") for record in records]
    return (str(ex.external(paths[0])) != str(ex.external(paths[1]))
            and all(isinstance(worker, str) and worker for worker in workers) and workers[0] != workers[1]
            and all(type(pid) is int and pid > 0 for pid in pids) and pids[0] != pids[1]
            and all(a.get(key) == b.get(key) for key in ("checkpoint_sha256", "config_sha256",
                "annotation_manifest_sha256", "protocol_sha256", "source_geometry_sha256",
                "native_array_sha256", "native_shape", "native_affine", "counts", "registry")))


def _evaluation_metadata(path):
    if path is None:
        return None
    try:
        return ex.read_json(path)
    except (ex.ExperimentError, OSError, ValueError, TypeError):
        return None


def report(manifest_path, output_path):
    """Score the frozen development cohort after two closed prediction runs."""
    run = _runner()
    manifest = ex.read_json(manifest_path)
    ex.require(set(manifest) <= {"schema_version", "entries", "readiness", "notes", "mechanism"}
               and manifest.get("schema_version") == 1 and isinstance(manifest.get("entries"), list)
               and len(manifest["entries"]) == 8, "REPORT_MANIFEST_INVALID")
    # Reference totals exist only in post-prediction scoring, never in infer().
    frozen = {"Case024_Series002": ((13, 10), (13, 8)), "Case012_Series007": ((9, 16), (12, 17)),
              "Case010_Series002": ((14, 11), (9, 4)), "Case008_Series004": ((10, 5), (10, 5)),
              "Case003_Series007": ((9, 12), (6, 7)), "Case006_Series007": ((15, 12), (13, 10)),
              "Case025_Series001": ((12, 14), (9, 12)), "Case016_Series003": ((9, 8), (8, 9))}
    rows, seen, checkpoint_hashes, annotation_hashes, protocol_hashes = [], set(), set(), set(), set()
    for entry in manifest["entries"]:
        ex.require(isinstance(entry, dict) and set(entry) == {"case_id", "reference", "baseline", "source_sha256",
                                                            "source_geometry_sha256", "predictions"},
                   "REPORT_ENTRY_INVALID")
        case = entry["case_id"]
        ex.require(isinstance(case, str) and case in frozen and case not in seen, "FROZEN_CASE_SET_MISMATCH")
        seen.add(case)
        for key, expected in zip(("reference", "baseline"), frozen[case]):
            value = entry[key]
            ex.require(isinstance(value, dict) and set(value) == {"left", "right"}
                       and all(type(value[side]) is int for side in ("left", "right"))
                       and (value["left"], value["right"]) == expected, "FROZEN_COUNTS_MISMATCH")
        ex.require(_hash(entry["source_sha256"]) and _hash(entry["source_geometry_sha256"]), "REPORT_SOURCE_HASH_INVALID")
        paths = entry["predictions"]
        ex.require(isinstance(paths, list) and len(paths) in (0, 2), "TWO_REPEAT_PATHS_REQUIRED")
        row = {"case_id": case, "reference": entry["reference"], "baseline": entry["baseline"],
               "status": "NOT_RUN", "prototype": None, "absolute_side_errors": None,
               "absolute_side_error": None, "bilateral_exact": False, "side_regressions": [],
               "repeatable": False, "repeat_counts": [], "failure_codes": []}
        records = []
        for path in paths:
            try:
                record, _, _ = _prediction(path, case_id=case, source_sha256=entry["source_sha256"],
                                           source_geometry_sha256=entry["source_geometry_sha256"])
                records.append(record)
                row["repeat_counts"].append(record["counts"])
            except (ex.ExperimentError, OSError, ValueError, TypeError, KeyError, IndexError,
                    nib.filebasedimages.ImageFileError) as error:
                records.append(None)
                row["repeat_counts"].append(None)
                row["failure_codes"].append(ex.failure_code(error))
        if paths:
            row["status"] = "FAILED"
            if all(record is not None for record in records):
                row["repeatable"] = _identical_fresh_repeats(paths, records)
                row["status"] = "COMPLETE" if row["repeatable"] else "NONREPEATABLE"
                if row["repeatable"]:
                    a = records[0]
                    checkpoint_hashes.add(a["checkpoint_sha256"])
                    annotation_hashes.add(a["annotation_manifest_sha256"])
                    protocol_hashes.add(a.get("protocol_sha256"))
                    row["prototype"] = {side: a["counts"][side] for side in ("left", "right")}
                    errors = {side: abs(row["prototype"][side]-entry["reference"][side]) for side in ("left", "right")}
                    row["absolute_side_errors"] = errors
                    row["absolute_side_error"] = sum(errors.values())
                    row["bilateral_exact"] = not any(errors.values())
                    row["side_regressions"] = [side for side in errors
                        if errors[side] > abs(entry["baseline"][side]-entry["reference"][side])]
        rows.append(row)
    ex.require(seen == set(frozen), "FROZEN_CASE_SET_MISMATCH")
    complete = sum(row["status"] == "COMPLETE" for row in rows)
    exact_cases = sum(row["bilateral_exact"] for row in rows)
    exact_sides = sum(error == 0 for row in rows for error in (row["absolute_side_errors"] or {}).values())
    total_error = sum(row["absolute_side_error"] for row in rows) if complete == 8 else None
    training = next(row for row in rows if row["case_id"] == "Case024_Series002")
    original = next(row for row in rows if row["case_id"] == "Case008_Series004")
    other_error = total_error-training["absolute_side_error"] if complete == 8 else None
    baseline_error = sum(abs(row["baseline"][side]-row["reference"][side]) for row in rows for side in ("left", "right"))
    ex.require(baseline_error == 37 and baseline_error-2 == 35, "FROZEN_BASELINE_ERROR_MISMATCH")
    gates = {"all_eight_outputs": complete == 8, "at_least_two_exact_bilateral": exact_cases >= 2,
             "absolute_error_below_37": total_error is not None and total_error < 37,
             "preserve_original_exact_case": original["bilateral_exact"],
             "other_seven_error_below_35": other_error is not None and other_error < 35,
             "two_repeat_identical_native_masks_and_counts": all(row["repeatable"] for row in rows),
             "single_frozen_checkpoint": complete == 8 and len(checkpoint_hashes) == 1,
             "single_annotation_release": complete == 8 and len(annotation_hashes) == 1,
             "single_frozen_protocol": complete == 8 and len(protocol_hashes) == 1
                                       and all(_hash(value) for value in protocol_hashes)}
    counts_pass = all(gates.values())
    mechanism_record = _evaluation_metadata(manifest.get("mechanism"))
    mechanism_status = "NOT_EVALUATED" if manifest.get("mechanism") is None else "INVALID_OR_UNBOUND"
    if mechanism_record is not None:
        expected_gates = {"three_supported_distinctions", "annotated_within_cell_connections",
                          "known_foreground_coverage", "explicit_exclusions_preserved"}
        training_entry = next(entry for entry in manifest["entries"] if entry["case_id"] == "Case024_Series002")
        prediction_hashes = {ex.sha256(ex.external(path)) for path in training_entry["predictions"]
                             if isinstance(path, str) and Path(path).is_absolute() and Path(path).is_file()}
        bound = (complete == 8 and mechanism_record.get("version") == run.VERSION
                 and mechanism_record.get("config_sha256") == ex.fingerprint(run.CONFIG)
                 and mechanism_record.get("checkpoint_sha256") in checkpoint_hashes
                 and mechanism_record.get("annotation_manifest_sha256") in annotation_hashes
                 and mechanism_record.get("protocol_sha256") in protocol_hashes
                 and mechanism_record.get("prediction_sha256") in prediction_hashes)
        if bound:
            mechanism_status = mechanism_record.get("status", "INVALID")
            mechanism_gates = mechanism_record.get("gates")
            if not isinstance(mechanism_gates, dict) or set(mechanism_gates) != expected_gates or not all(
                    value is True for value in mechanism_gates.values()):
                mechanism_status = "FAIL"
    gates["supported_cell_identity_mechanism"] = mechanism_status == "PASS_CONDITIONAL_MECHANISM"
    ready = _evaluation_metadata(manifest.get("readiness"))
    gates["bound_conditional_readiness"] = bool(ready is not None and mechanism_record is not None
        and ready.get("version") == run.VERSION and ready.get("training_allowed") is True
        and ready.get("evolving_full_graph") is True and ready.get("partial_supervision_only") is True
        and ready.get("long_continuation_unmeasured") is True
        and ready.get("annotation_manifest_sha256") == mechanism_record.get("annotation_manifest_sha256")
        and ready.get("annotation_manifest_sha256") in annotation_hashes
        and ready.get("protocol_sha256") in protocol_hashes)
    beyond_training = ("NOT_EVALUABLE" if complete != 8 else
                       "IMPROVEMENT_BEYOND_TRAINING_CASE" if other_error < 35 else
                       "TRAINING_CASE_FIT_ONLY" if training["absolute_side_error"] < 2 else
                       "NO_DEMONSTRATED_COUNT_IMPROVEMENT")
    status = "PASS_CONDITIONAL_DEVELOPMENT" if all(gates.values()) else (
        "NOT_RUN" if all(row["status"] == "NOT_RUN" for row in rows) else "FAIL")
    summary = {"status": status, "version": run.VERSION, "config_sha256": ex.fingerprint(run.CONFIG),
               "scheduled_cases": 8, "scheduled_sides": 16, "complete_cases": complete,
               "bilateral_exact": exact_cases, "side_exact": exact_sides, "absolute_side_error": total_error,
               "other_seven_absolute_side_error": other_error, "baseline_absolute_side_error": 37,
               "baseline_other_seven_absolute_side_error": 35,
               "count_status": "PASS_DEVELOPMENT_COUNTS" if counts_pass else "FAIL_OR_NOT_RUN",
               "beyond_training_case": beyond_training,
               "mechanism_status": mechanism_status, "gates": gates, "cases": rows,
               "whole_cell_false_splits": "UNMEASURED_OUTSIDE_PROVISIONAL_CORES",
               "interpretation": "Previously inspected development data; sparse cell-identity checks do not establish whole-cell or clinical accuracy."}
    output = ex.new_output(output_path)
    ex.write_json(output/"comparison.local.json", summary)
    text = ["# Eight-case connectivity development comparison", "", f"Status: {status}.", "",
            "| Scan | Reference L/R | Baseline L/R | Prototype L/R | Side error | Regressions | Status |",
            "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        pair = lambda value: f"{value['left']}/{value['right']}" if value else "Not available"
        text.append(f"| {row['case_id']} | {pair(row['reference'])} | {pair(row['baseline'])} | "
                    f"{pair(row['prototype'])} | {row['absolute_side_error']} | "
                    f"{', '.join(row['side_regressions']) or 'None'} | {row['status']} |")
    text.extend(["", f"Exact bilateral: {exact_cases}/8. Exact sides: {exact_sides}/16.",
                 f"Total absolute side error: {total_error}. Other seven: {other_error}.",
                 f"Beyond the training case: {beyond_training}.", ""])
    text.extend(f"- {gate}: {'PASS' if passed else 'FAIL'}" for gate, passed in gates.items())
    text.extend(["", "Missing, failed and nonrepeatable predictions remain in the denominator; no baseline fallback is used.",
                 "Supported core splits and merges are scored separately. Whole-cell false splits outside those cores remain unmeasured.",
                 "Count agreement alone does not establish correct cell identities or clinical accuracy."])
    with (output/"REPORT.local.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(text)+"\n")
    return summary
