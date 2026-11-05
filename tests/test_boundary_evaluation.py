"""Synthetic mutation checks for source-bound, repeat-aware evaluation."""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_pipeline as pipeline
from ethmoid_pipeline.counting import instance_experiment as legacy


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": legacy.sha256(path)}


def read(reference):
    return json.loads(Path(reference["path"]).read_text())


def rewrite(reference, value):
    reference.update(write(Path(reference["path"]), value))


def source(tmp_path, *, flipped=False):
    affine = np.diag([-.5 if flipped else .5, .75, 1.25, 1.])
    affine[0, 3] = 3 if flipped else -3
    path = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.full((14, 10, 8), -800, np.float32), affine), path)
    return {"input": str(path), "source_sha256": legacy.sha256(path),
            "crop_bounds": [[1, 13], [1, 9], [1, 7]], "midline_ras_x": 0.}


def labels():
    result = np.zeros((12, 8, 6), np.uint16)
    result[2:4, 2:4, 2:4] = 1
    result[8:10, 2:4, 2:4] = 2
    return result


def prediction(tmp_path, record, crop, worker):
    folder = tmp_path / f"worker-{worker}"
    folder.mkdir()
    original, canonical, hu, affine = legacy.load_crop(record)
    stored = pipeline.native_output(crop, original, canonical, record)
    artifact = folder / "instances.local.nii.gz"
    nib.save(nib.Nifti1Image(stored, original.affine), artifact)
    registry, candidates = legacy.label_registry(crop, hu, affine, record["midline_ras_x"])
    flags = pipeline._quality_flags(crop, registry)
    value = {"version": pipeline.VERSION, "status": "COMPLETE", "case_id": "synthetic",
        "source": record, "source_sha256": record["source_sha256"],
        "worker_identity": f"worker-{worker}", "worker_pid": 1000 + worker,
        "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64, "partition_runtime_sha256": "c" * 64,
        "producer_sha256": "e" * 64, "model_sources": {"fixture.py": "f" * 64},
        "model_runtime": {"python": "3.11.9", "torch": "2.5.1", "cuda": "12.1", "cudnn": 8907},
        "native_shape": list(original.shape), "native_affine": original.affine.tolist(),
        "artifact_sha256": legacy.sha256(artifact),
        "native_array_sha256": hashlib.sha256(stored.tobytes(order="C")).hexdigest(),
        "registry": registry, "candidate_counts": candidates, "quality_flags": flags,
        "count_status": "WITHHELD" if flags else "DEVELOPMENT_ONLY",
        "counts": {key: None for key in ("left", "right", "total")} if flags else candidates}
    return write(folder / "prediction.local.json", value)


def entry(tmp_path, *, crops=None, flipped=False):
    record = source(tmp_path, flipped=flipped)
    crops = [labels(), labels()] if crops is None else crops
    return {"case_id": "synthetic", "source_sha256": record["source_sha256"],
            "predictions": [prediction(tmp_path, record, crop, index + 1) for index, crop in enumerate(crops)],
            "reference_counts": {"left": 1, "right": 1, "total": 2}}, record


def evaluate(tmp_path, entries):
    manifest = write(tmp_path / "evaluation-manifest.json", {"schema_version": 1,
                     "version": pipeline.VERSION, "entries": entries})
    return pipeline.evaluate(manifest["path"], tmp_path / "evaluation")


def sample(tmp_path, record, instances=None):
    _, _, hu, affine = legacy.load_crop(record)
    instances = labels().astype(np.int32) if instances is None else instances
    path = tmp_path / "reference.npz"
    np.savez_compressed(path, foreground_labels=(instances > 0).astype(np.int8),
                        separator_labels=np.full(hu.shape, -1, np.int8), instance_ids=instances,
                        evaluation_region=np.ones(hu.shape, bool), affine=affine)
    return {"sample_id": "reference", "patient_group": "synthetic-group", "scope": "synthetic_truth",
            "source": record, "annotation": {"path": str(path), "sha256": legacy.sha256(path)},
            "protocol_sha256": "d" * 64, "provenance": {"kind": "synthetic_truth", "generator": "test-only"}}


@pytest.mark.parametrize("flipped", [False, True])
def test_recomputed_native_geometry_registry_and_complete_region_match(tmp_path, flipped):
    case, record = entry(tmp_path, flipped=flipped)
    case["sample"] = sample(tmp_path, record)
    result = evaluate(tmp_path, [case])
    assert result["states"] == {"AVAILABLE": 1} and result["paired_sides"] == 2
    assert result["exact_bilateral_studies"] == 1 and result["absolute_side_error"] == 0
    assert result["status"] == "COMPLETE" and result["validation_passes"] is True
    row = result["cases"][0]
    assert row["side_errors"] == {"left": 0, "right": 0} and row["paired_sides"] == 2
    assert row["instances"]["matched_instances"] == 2
    assert row["instances"]["evaluated_voxels"] == labels().size
    assert row["instances"]["region_fraction"] == 1.
    assert row["instances"]["reference_foreground_voxel_coverage"] == 1.
    assert row["instances"]["foreground_dice"] == 1.
    assert row["instances"]["clinical_accuracy_scope"] == "not_established_supplied_synthetic_truth_only"


@pytest.mark.parametrize("field,value", [
    ("counts", {"left": 99, "right": 99, "total": 198}),
    ("counts", {"left": True, "right": 1, "total": 2}),
    ("candidate_counts", {"left": 99, "right": 99, "total": 198}),
    ("registry", []), ("quality_flags", ["fabricated"]), ("count_status", "WITHHELD")])
def test_hash_bound_metadata_cannot_override_mask_derived_counts(tmp_path, field, value):
    case, _ = entry(tmp_path)
    reference = case["predictions"][0]
    changed = read(reference); changed[field] = value; rewrite(reference, changed)
    with pytest.raises(legacy.ExperimentError, match="PREDICTION_REGISTRY_COUNTS_MISMATCH"):
        evaluate(tmp_path, [case])


def test_shifted_mask_and_matching_forged_metadata_fail_actual_source_binding(tmp_path):
    case, _ = entry(tmp_path)
    reference = case["predictions"][0]
    changed = read(reference)
    artifact = Path(reference["path"]).parent / "instances.local.nii.gz"
    image = nib.load(artifact)
    array, affine = np.asanyarray(image.dataobj).copy(), image.affine.copy()
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(array, affine), artifact)
    changed.update(native_affine=affine.tolist(), artifact_sha256=legacy.sha256(artifact))
    rewrite(reference, changed)
    with pytest.raises(legacy.ExperimentError, match="PREDICTION_SOURCE_GEOMETRY_MISMATCH"):
        evaluate(tmp_path, [case])


def test_artifact_bytes_and_uint16_values_are_checked_before_scoring(tmp_path):
    case, _ = entry(tmp_path)
    artifact = Path(case["predictions"][0]["path"]).parent / "instances.local.nii.gz"
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(legacy.ExperimentError, match="PREDICTION_ARTIFACT_CHANGED"):
        evaluate(tmp_path, [case])


def test_positive_labels_outside_declared_crop_are_rejected(tmp_path):
    case, _ = entry(tmp_path)
    reference = case["predictions"][0]
    changed = read(reference)
    artifact = Path(reference["path"]).parent / "instances.local.nii.gz"
    image = nib.load(artifact)
    array = np.asanyarray(image.dataobj).copy(); array[0, 0, 0] = 9
    nib.save(nib.Nifti1Image(array, image.affine), artifact)
    changed.update(artifact_sha256=legacy.sha256(artifact),
                   native_array_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    rewrite(reference, changed)
    with pytest.raises(legacy.ExperimentError, match="PREDICTION_LABELS_OUTSIDE_CROP"):
        evaluate(tmp_path, [case])


def test_distinct_uuids_in_one_process_do_not_satisfy_fresh_workers(tmp_path):
    case, _ = entry(tmp_path)
    first, second = case["predictions"]
    changed = read(second); changed["worker_pid"] = read(first)["worker_pid"]; rewrite(second, changed)
    with pytest.raises(legacy.ExperimentError, match="FRESH_WORKERS_REQUIRED"):
        evaluate(tmp_path, [case])


def test_repeat_mismatch_suppresses_all_count_and_instance_scoring(tmp_path, monkeypatch):
    other = labels(); other[2, 2, 2] = 0
    case, _ = entry(tmp_path, crops=[labels(), other])
    case["sample"] = {"must_not_be_loaded": True}
    monkeypatch.setattr(pipeline, "_evaluation_instances", lambda *_: pytest.fail("must not score one repeat"))
    result = evaluate(tmp_path, [case])
    assert result["states"] == {"REPEAT_MISMATCH": 1} and result["paired_sides"] == 0
    assert result["absolute_side_error"] is None and result["exact_bilateral_studies"] == 0
    assert result["status"] == "COMPLETE" and result["validation_passes"] is False
    assert result["cases"][0]["instances"]["status"] == "NOT_SCORED_REPEAT_MISMATCH"
    assert all(value is None for value in result["cases"][0]["counts"].values())


def test_repeat_source_crop_and_midline_bindings_must_match(tmp_path):
    case, _ = entry(tmp_path)
    reference = case["predictions"][1]
    changed = read(reference); changed["source"]["midline_ras_x"] = .1; rewrite(reference, changed)
    result = evaluate(tmp_path, [case])
    assert result["states"] == {"REPEAT_MISMATCH": 1} and result["paired_sides"] == 0


def test_withheld_candidates_are_never_scored_as_numeric_counts(tmp_path):
    case, _ = entry(tmp_path, crops=[np.zeros_like(labels()), np.zeros_like(labels())])
    case["reference_counts"] = {"left": 0, "right": 0, "total": 0}
    result = evaluate(tmp_path, [case])
    assert result["states"] == {"WITHHELD": 1} and result["paired_sides"] == 0
    assert result["unpaired_reference_sides"] == 2 and result["exact_sides"] == 0


@pytest.mark.parametrize("references,paired", [({"right": 1}, 1), ({"left": None, "right": 1, "total": None}, 1),
                                               ({"total": 2}, 0), (None, 0)])
def test_partial_references_remain_unknown_without_inferred_targets(tmp_path, references, paired):
    case, _ = entry(tmp_path); case["reference_counts"] = references
    result = evaluate(tmp_path, [case])
    assert result["paired_sides"] == paired and result["exact_bilateral_studies"] == 0


@pytest.mark.parametrize("references", [{"left": -1}, {"right": True}, {"left": 2, "right": 2, "total": 1},
                                        {"left": 5, "total": 2}, {"bad": 1}, {}])
def test_invalid_references_fail_even_when_not_run(tmp_path, references):
    case = {"case_id": "not-run", "source_sha256": "a" * 64, "predictions": [], "reference_counts": references}
    with pytest.raises(legacy.ExperimentError, match="REFERENCE_COUNTS_INVALID"):
        evaluate(tmp_path, [case])


def test_failed_and_not_run_entries_remain_in_all_scheduled_denominators(tmp_path):
    failure = write(tmp_path / "failure.json", {"version": pipeline.VERSION, "status": "FAILED",
                    "command": "infer", "error_code": "CUDA_OUT_OF_MEMORY"})
    entries = [{"case_id": "failed", "source_sha256": "a" * 64, "predictions": [], "failure": failure,
                "reference_counts": {"left": 3, "right": 4, "total": 7}},
               {"case_id": "not-run", "source_sha256": "b" * 64, "predictions": [], "reference_counts": {"right": 2}}]
    result = evaluate(tmp_path, entries)
    assert result["scheduled_studies"] == 2 and result["scheduled_sides"] == 4
    assert result["states"] == {"FAILED": 1, "NOT_RUN": 1} and result["unpaired_reference_sides"] == 3
    assert result["status"] == "COMPLETE" and result["validation_passes"] is False
    assert result["cases"][0]["failure"]["error_code"] == "CUDA_OUT_OF_MEMORY"


@pytest.mark.parametrize("mutation", ["hash", "source", "status"])
def test_failed_entry_needs_verified_failure_record(tmp_path, mutation):
    record = {"version": pipeline.VERSION, "status": "FAILED", "error_code": "CUDA_OUT_OF_MEMORY"}
    if mutation == "source":
        record["source_sha256"] = "b" * 64
    if mutation == "status":
        record["status"] = "COMPLETE"
    reference = write(tmp_path / "failure.json", record)
    if mutation == "hash":
        reference["sha256"] = "0" * 64
    case = {"case_id": "failed", "source_sha256": "a" * 64, "predictions": [], "failure": reference}
    with pytest.raises(legacy.ExperimentError, match="ARTIFACT_HASH_MISMATCH|FAILURE_RECORD_INVALID"):
        evaluate(tmp_path, [case])


def test_unknown_instance_voxels_inside_complete_reference_region_are_not_background(tmp_path):
    case, record = entry(tmp_path)
    instances = labels().astype(np.int32); instances[0, 0, 0] = -1
    case["sample"] = sample(tmp_path, record, instances)
    with pytest.raises(legacy.ExperimentError, match="BOUNDARY_EVALUATION_REGION_INCOMPLETE|REFERENCE_REGION_HAS_UNKNOWN_INSTANCES"):
        evaluate(tmp_path, [case])


def test_partial_complete_region_reports_coverage_and_conditional_scope(tmp_path):
    case, record = entry(tmp_path)
    annotation = sample(tmp_path, record)
    annotation["scope"] = "conditional_native"
    annotation["provenance"] = {"kind": "conditional_native", "review_status": "conditional"}
    path = Path(annotation["annotation"]["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["evaluation_region"][:6] = False
    np.savez_compressed(path, **arrays)
    annotation["annotation"]["sha256"] = legacy.sha256(path)
    case["sample"] = annotation
    result = evaluate(tmp_path, [case])
    metric = result["cases"][0]["instances"]
    assert metric["region_fraction"] == .5 and metric["evaluated_voxels"] == labels().size // 2
    assert metric["reference_foreground_voxels"] == 8 and metric["foreground_dice"] == 1.
    assert metric["clinical_accuracy_scope"] == "not_established_supplied_conditional_native_reference_only"


def test_per_row_errors_include_only_supplied_known_sides(tmp_path):
    case, _ = entry(tmp_path)
    case["reference_counts"] = {"right": 4}
    row = evaluate(tmp_path, [case])["cases"][0]
    assert row["side_errors"] == {"left": None, "right": 3}
    assert row["paired_sides"] == 1 and row["absolute_side_error"] == 3


def test_reference_crop_and_midline_must_match_prediction_source_binding(tmp_path):
    case, record = entry(tmp_path)
    case["sample"] = sample(tmp_path, {**record, "midline_ras_x": .1})
    with pytest.raises(legacy.ExperimentError, match="REFERENCE_SOURCE_MISMATCH"):
        evaluate(tmp_path, [case])


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "d" * 64), ("producer_sha256", "d" * 64),
    ("model_sources", {"fixture.py": "d" * 64}),
    ("model_runtime", {"python": "3.12.1", "torch": "2.5.1", "cuda": "12.1", "cudnn": 8907})])
def test_mixed_checkpoint_or_producer_identities_cannot_be_aggregated(tmp_path, field, value):
    case, _ = entry(tmp_path)
    reference = case["predictions"][1]
    changed = read(reference); changed[field] = value; rewrite(reference, changed)
    with pytest.raises(legacy.ExperimentError, match="COHORT_CHECKPOINT_OR_CONFIGURATION_MISMATCH"):
        evaluate(tmp_path, [case])


@pytest.mark.parametrize("field,value", [
    ("producer_sha256", "missing"), ("model_sources", {}),
    ("model_sources", {"../file.py": "a" * 64}), ("model_sources", {"file.py": True}),
    ("model_runtime", {"python": "3.11.9"}),
    ("model_runtime", {"python": "3.11.9", "torch": "2.5.1", "cuda": "12.1", "cudnn": True})])
def test_configuration_provenance_has_closed_shapes_and_digest_values(tmp_path, field, value):
    case, _ = entry(tmp_path)
    reference = case["predictions"][0]
    changed = read(reference); changed[field] = value; rewrite(reference, changed)
    with pytest.raises(legacy.ExperimentError, match="PREDICTION_CONFIGURATION_INVALID"):
        evaluate(tmp_path, [case])
