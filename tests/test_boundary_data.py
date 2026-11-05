"""Synthetic-only release validation and native Slicer label interchange."""
from copy import deepcopy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_data as data
from ethmoid_pipeline.counting import instance_experiment as legacy


def sample(tmp_path, name="sample-a", group="group-a", affine=None):
    affine = np.diag([.5, .75, 1.25, 1.]) if affine is None else affine
    volume = tmp_path / f"{name}.nii.gz"
    hu = np.arange(9 * 8 * 7, dtype=np.float32).reshape(9, 8, 7) - 1000
    nib.save(nib.Nifti1Image(hu, affine), volume)
    source = {"input": str(volume), "source_sha256": legacy.sha256(volume),
              "crop_bounds": [[1, 8], [1, 7], [1, 6]], "midline_ras_x": 0.}
    _, _, crop, crop_affine = legacy.load_crop(source)
    foreground = np.full(crop.shape, -1, np.int8)
    separator = np.full(crop.shape, -1, np.int8)
    instances = np.full(crop.shape, -1, np.int32)
    foreground[1:3, 1:4, 1:4], instances[1:3, 1:4, 1:4] = 1, 1
    foreground[4:6, 1:4, 1:4], instances[4:6, 1:4, 1:4] = 1, 2
    foreground[0], instances[0] = 0, 0
    separator[3, 1:4, 1:4] = 1
    separator[1, 1:4, 1:4] = 0
    arrays = {"foreground_labels": foreground, "separator_labels": separator,
              "instance_ids": instances, "evaluation_region": np.zeros(crop.shape, bool),
              "affine": crop_affine}
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return {"sample_id": name, "patient_group": group, "scope": "synthetic_truth", "source": source,
            "annotation": {"path": str(path), "sha256": legacy.sha256(path)},
            "protocol_sha256": "a" * 64,
            "provenance": {"kind": "synthetic_truth", "generator": "synthetic-test"}}, crop, arrays


def release(*samples):
    return {"schema_version": 1, "version": data.VERSION, "samples": list(samples)}


def rewrite_annotation(record, **updates):
    path = Path(record["annotation"]["path"])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(updates)
    np.savez_compressed(path, **values)
    record["annotation"]["sha256"] = legacy.sha256(path)


def test_native_crop_preserves_signed_unknowns_and_counts_only_supplied_labels(tmp_path):
    record, expected_hu, expected_arrays = sample(tmp_path)
    hu, affine, arrays = data.load_sample(record)
    assert hu.dtype == np.float32
    np.testing.assert_array_equal(hu, expected_hu)
    for key in expected_arrays:
        np.testing.assert_array_equal(arrays[key], expected_arrays[key])
    result = data.validate_release(release(record))
    readiness = result["readiness"]
    assert readiness["status"] == "MECHANISM_ONLY"
    assert readiness["training_allowed"] and not readiness["validation_available"]
    assert readiness["head_counts"]["separator_labels"]["positive"] == 9
    assert readiness["head_counts"]["foreground_labels"]["unknown"] > 0
    assert not readiness["clinical_validation_established"]
    assert record == result["samples"][0]
    np.testing.assert_array_equal(affine, expected_arrays["affine"])


@pytest.mark.parametrize("key", ["foreground_labels", "separator_labels", "instance_ids", "evaluation_region"])
def test_shape_rejected_for_every_label_layer(tmp_path, key):
    record, _, arrays = sample(tmp_path)
    rewrite_annotation(record, **{key: arrays[key][:-1]})
    with pytest.raises(data.ExperimentError, match="BOUNDARY_ANNOTATION_SHAPE_INVALID"):
        data.load_sample(record)


def test_exact_schemas_hashes_and_affine_prevent_unbound_or_reference_metadata(tmp_path):
    record, _, arrays = sample(tmp_path)
    with pytest.raises(data.ExperimentError, match="BOUNDARY_RELEASE_SCHEMA_INVALID"):
        data.validate_release({**release(record), "reference_total": 2})
    with pytest.raises(data.ExperimentError, match="BOUNDARY_SAMPLE_SCHEMA_INVALID"):
        data.load_sample({**record, "reference_total": 2})
    changed = deepcopy(record)
    changed["source"]["reference_total"] = 2
    with pytest.raises(data.ExperimentError, match="SOURCE_SCHEMA_INVALID"):
        data.load_sample(changed)
    changed = deepcopy(record)
    changed["annotation"]["sha256"] = "0" * 64
    with pytest.raises(data.ExperimentError, match="ANNOTATION_HASH_MISMATCH"):
        data.load_sample(changed)
    changed = deepcopy(record)
    changed["source"]["source_sha256"] = "0" * 64
    with pytest.raises(data.ExperimentError, match="SOURCE_HASH_MISMATCH"):
        data.load_sample(changed)
    shifted = arrays["affine"].copy()
    shifted[0, 3] += .1
    rewrite_annotation(record, affine=shifted)
    with pytest.raises(data.ExperimentError, match="BOUNDARY_AFFINE_MISMATCH"):
        data.load_sample(record)


@pytest.mark.parametrize("updates,code", [
    ({"foreground_labels": np.zeros((7, 6, 5), np.uint8)}, "BOUNDARY_HEAD_VALUES_INVALID"),
    ({"separator_labels": np.full((7, 6, 5), 2, np.int8)}, "BOUNDARY_HEAD_VALUES_INVALID"),
    ({"instance_ids": np.full((7, 6, 5), -2, np.int32)}, "BOUNDARY_INSTANCE_VALUES_INVALID"),
    ({"evaluation_region": np.ones((7, 6, 5), np.uint8)}, "BOUNDARY_EVALUATION_REGION_INVALID"),
])
def test_label_values_and_signed_types_are_explicit(tmp_path, updates, code):
    record, _, _ = sample(tmp_path)
    rewrite_annotation(record, **updates)
    with pytest.raises(data.ExperimentError, match=code):
        data.load_sample(record)


def test_expert_requires_independent_adjudication_and_conditional_is_explicit(tmp_path):
    record, _, _ = sample(tmp_path)
    record["scope"] = "adjudicated_expert"
    record["provenance"] = {"kind": "adjudicated_expert", "annotator": "rater-a", "reviewer": "rater-b",
                            "review_status": "adjudicated", "independent_review": True}
    assert data.validate_release(release(record))["readiness"]["scope_counts"] == {"adjudicated_expert": 1}
    record["provenance"]["reviewer"] = "rater-a"
    with pytest.raises(data.ExperimentError, match="INDEPENDENT_ADJUDICATED_REVIEW_REQUIRED"):
        data.load_sample(record)
    record["scope"], record["provenance"] = "conditional_native", {
        "kind": "conditional_native", "review_status": "conditional"}
    assert data.validate_release(release(record))["readiness"]["scope_counts"] == {"conditional_native": 1}


def test_source_cannot_leak_across_patient_groups(tmp_path):
    record, _, _ = sample(tmp_path)
    changed = deepcopy(record)
    changed.update(sample_id="sample-b", patient_group="group-b")
    with pytest.raises(data.ExperimentError, match="SOURCE_PATIENT_GROUP_CONFLICT"):
        data.validate_release(release(record, changed))
    changed["patient_group"] = record["patient_group"]
    with pytest.raises(data.ExperimentError, match="DUPLICATE_SOURCE_CROP"):
        data.validate_release(release(record, changed))


def test_group_holdout_requires_both_classes_in_training_partition(tmp_path):
    train, _, arrays = sample(tmp_path, "train", "a")
    valid, _, _ = sample(tmp_path, "validation", "z")
    # Make a genuinely different source while keeping its geometry.
    source = Path(valid["source"]["input"])
    image = nib.load(source)
    nib.save(nib.Nifti1Image(image.get_fdata().astype(np.float32) + 1, image.affine), source)
    valid["source"]["source_sha256"] = legacy.sha256(source)
    separator = arrays["separator_labels"].copy()
    separator[separator == 1] = -1
    rewrite_annotation(train, separator_labels=separator)
    result = data.validate_release(release(train, valid))["readiness"]
    assert result["validation_available"] and not result["training_allowed"]
    assert result["group_holdout"]["training_groups"] == ["a"]
    assert "TRAINING_SEPARATOR_LABELS_POSITIVE_MISSING" in result["reasons"]
    assert result["head_counts"]["separator_labels"]["positive"] > 0


def test_readiness_requires_fresh_external_output(tmp_path):
    record, _, _ = sample(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(release(record)), encoding="utf-8")
    output = tmp_path / "readiness"
    assert data.readiness(path, output)["training_allowed"]
    assert (output / "readiness.local.json").is_file()
    with pytest.raises(data.ExperimentError, match="FRESH_OUTPUT_REQUIRED"):
        data.readiness(path, output)
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / ".git").mkdir()
    with pytest.raises(data.ExperimentError, match="EXTERNAL_ARTIFACT_REQUIRED"):
        data.readiness(path, protected / "release")


def exported(tmp_path, *, affine=None):
    record, hu, arrays = sample(tmp_path, affine=affine)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(release(record)), encoding="utf-8")
    output = tmp_path / "exported"
    result = data.export(manifest, output)
    exported_record = json.loads((output / "release.local.json").read_text())["samples"][0]
    return record, exported_record, output / record["sample_id"], result, hu, arrays


def test_export_roundtrip_preserves_canonical_geometry_hashes_unknowns_and_signed_layers(tmp_path):
    # A noncanonical source proves review layers use canonical native crop geometry.
    affine = np.array([[-.5, 0, 0, 10], [0, 0, 1.25, -4], [0, .75, 0, 2], [0, 0, 0, 1]])
    original, record, folder, result, hu, arrays = exported(tmp_path, affine=affine)
    assert result["status"] == "EXPORTED"
    assert legacy.sha256(original["annotation"]["path"]) == legacy.sha256(record["annotation"]["path"])
    manifest = json.loads((folder.parent / "release.local.json").read_text())
    assert set(manifest) == {"schema_version", "version", "samples"}
    assert data.validate_release(manifest)["readiness"]["training_allowed"]
    for key in data.HEADS:
        assert nib.load(folder / f"{key}.nii.gz").get_data_dtype() == np.dtype(np.int8)
        assert nib.load(folder / f"{key}_known.nii.gz").get_data_dtype() == np.dtype(np.uint8)
    destination = tmp_path / "roundtrip.npz"
    annotation = data.import_review(record, folder, destination)
    reread = {**record, "annotation": annotation}
    actual_hu, actual_affine, actual = data.load_sample(reread)
    np.testing.assert_array_equal(actual_hu, hu)
    np.testing.assert_array_equal(actual_affine, arrays["affine"])
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])
    # The source-bound original manifest is also accepted after the byte copy.
    data.import_review(original, folder, tmp_path / "roundtrip-original.npz")
    with pytest.raises(data.ExperimentError, match="FRESH_OUTPUT_REQUIRED"):
        data.import_review(record, folder, destination)


def test_review_addition_requires_explicit_known_mask_and_never_infers_separator_labels(tmp_path):
    _, record, folder, _, _, arrays = exported(tmp_path)
    key = "separator_labels"
    image = nib.load(folder / f"{key}.nii.gz")
    changed = np.asarray(image.dataobj).copy()
    coordinate = tuple(np.argwhere(changed == -1)[0])
    changed[coordinate] = 1
    nib.save(nib.Nifti1Image(changed, image.affine), folder / f"{key}.nii.gz")
    with pytest.raises(data.ExperimentError, match="BOUNDARY_REVIEW_KNOWN_MASK_MISMATCH"):
        data.import_review(record, folder, tmp_path / "not-written.npz")
    assert not (tmp_path / "not-written.npz").exists()
    mask_image = nib.load(folder / f"{key}_known.nii.gz")
    known = np.asarray(mask_image.dataobj).copy()
    known[coordinate] = 1
    nib.save(nib.Nifti1Image(known, image.affine), folder / f"{key}_known.nii.gz")
    annotation = data.import_review(record, folder, tmp_path / "edited.npz")
    _, _, actual = data.load_sample({**record, "annotation": annotation})
    np.testing.assert_array_equal(actual[key], changed)
    for unchanged in ("foreground_labels", "instance_ids", "evaluation_region"):
        np.testing.assert_array_equal(actual[unchanged], arrays[unchanged])


@pytest.mark.parametrize("tamper,code", [
    ("geometry", "BOUNDARY_REVIEW_GEOMETRY_MISMATCH"),
    ("source", "BOUNDARY_REVIEW_SOURCE_CHANGED"),
    ("metadata", "BOUNDARY_REVIEW_BINDING_MISMATCH"),
    ("mask_dtype", "BOUNDARY_REVIEW_DTYPE_INVALID"),
])
def test_review_rejects_changed_geometry_source_binding_and_mask_type(tmp_path, tamper, code):
    _, record, folder, _, _, _ = exported(tmp_path)
    if tamper == "metadata":
        path = folder / "review.local.json"
        metadata = json.loads(path.read_text())
        metadata["protocol_sha256"] = "b" * 64
        path.write_text(json.dumps(metadata))
    else:
        key = "source_volume" if tamper == "source" else "foreground_labels"
        if tamper == "mask_dtype":
            key += "_known"
        path = folder / f"{key}.nii.gz"
        image = nib.load(path)
        values, affine = np.asarray(image.dataobj).copy(), image.affine.copy()
        if tamper == "geometry":
            affine[1, 3] += 1
        elif tamper == "mask_dtype":
            values = values.astype(np.int8)
        else:
            values.flat[0] += 10
        nib.save(nib.Nifti1Image(values, affine), path)
    with pytest.raises(data.ExperimentError, match=code):
        data.import_review(record, folder, tmp_path / "rejected.npz")
    assert not (tmp_path / "rejected.npz").exists()


def test_crop_and_release_budgets_apply_before_loading_voxels(tmp_path, monkeypatch):
    record, _, _ = sample(tmp_path)
    monkeypatch.setattr(data, "MAX_CROP_VOXELS", 100)
    monkeypatch.setattr(legacy, "load_crop", lambda _: pytest.fail("must not load over-budget data"))
    with pytest.raises(data.ExperimentError, match="BOUNDARY_CROP_VOXEL_BUDGET_EXCEEDED"):
        data.load_sample(record)
    monkeypatch.setattr(data, "MAX_CROP_VOXELS", 1000)
    monkeypatch.setattr(data, "MAX_RELEASE_VOXELS", 100)
    with pytest.raises(data.ExperimentError, match="BOUNDARY_RELEASE_VOXEL_BUDGET_EXCEEDED"):
        data.validate_release(release(record))


def test_empty_release_and_absent_head_classes_have_explicit_reasons(tmp_path):
    empty = data.validate_release(release())["readiness"]
    assert not empty["training_allowed"] and "SAMPLES_MISSING" in empty["reasons"]
    assert empty["head_counts"]["separator_labels"] == {"unknown": 0, "negative": 0, "positive": 0}
    record, _, arrays = sample(tmp_path)
    separator = np.full(arrays["separator_labels"].shape, -1, np.int8)
    rewrite_annotation(record, separator_labels=separator)
    result = data.validate_release(release(record))["readiness"]
    assert not result["training_allowed"]
    assert "SEPARATOR_LABELS_NEGATIVE_MISSING" in result["reasons"]
    assert "SEPARATOR_LABELS_POSITIVE_MISSING" in result["reasons"]


def test_unsupported_arrays_corrupt_archive_and_private_source_errors_are_fixed_codes(tmp_path):
    record, _, _ = sample(tmp_path)
    rewrite_annotation(record, reference_count=np.array([2]))
    with pytest.raises(data.ExperimentError, match="BOUNDARY_ANNOTATION_ARRAYS_INVALID"):
        data.load_sample(record)
    annotation = Path(record["annotation"]["path"])
    annotation.write_bytes(b"not a zip")
    record["annotation"]["sha256"] = legacy.sha256(annotation)
    with pytest.raises(data.ExperimentError, match="BOUNDARY_SAMPLE_LOAD_FAILED"):
        data.load_sample(record)
    source = Path(record["source"]["input"])
    source.write_bytes(b"synthetic-private-header")
    record["source"]["source_sha256"] = legacy.sha256(source)
    with pytest.raises(data.ExperimentError, match="BOUNDARY_SOURCE_LOAD_FAILED") as error:
        data.validate_release(release(record))
    assert "private" not in str(error.value) and str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("unknown", ["instance_ids", "foreground_labels"])
def test_evaluation_region_requires_known_instances_and_foreground(tmp_path, unknown):
    record, _, arrays = sample(tmp_path)
    evaluation = np.zeros(arrays["instance_ids"].shape, bool)
    evaluation[0, 0, 0] = True
    values = arrays[unknown].copy()
    values[0, 0, 0] = -1
    rewrite_annotation(record, evaluation_region=evaluation, **{unknown: values})
    with pytest.raises(data.ExperimentError, match="BOUNDARY_EVALUATION_REGION_INCOMPLETE"):
        data.load_sample(record)


def test_complete_instance_evaluation_does_not_require_separator_labels(tmp_path):
    record, _, arrays = sample(tmp_path)
    evaluation = np.zeros(arrays["instance_ids"].shape, bool)
    evaluation[0] = True
    rewrite_annotation(record, evaluation_region=evaluation,
                       separator_labels=np.full(evaluation.shape, -1, np.int8))
    data.load_sample(record)


def test_slicer_zero_background_restores_unknowns_from_explicit_known_masks(tmp_path):
    _, record, folder, _, _, arrays = exported(tmp_path)
    for key in (*data.HEADS, "instance_ids"):
        image = nib.load(folder / f"{key}.nii.gz")
        labels = np.asarray(image.dataobj).copy()
        labels[labels < 0] = 0
        nib.save(nib.Nifti1Image(labels, image.affine), folder / f"{key}.nii.gz")
    annotation = data.import_review(record, folder, tmp_path / "slicer-roundtrip.npz")
    _, _, actual = data.load_sample({**record, "annotation": annotation})
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])


@pytest.mark.parametrize("head", data.HEADS)
def test_holdout_needs_some_known_supervision_in_each_head_without_fold_merging(tmp_path, head):
    train, _, _ = sample(tmp_path, "train", "a")
    valid, _, arrays = sample(tmp_path, "heldout", "z", affine=np.diag([.6, .8, 1.25, 1.]))
    rewrite_annotation(valid, **{head: np.full(arrays[head].shape, -1, np.int8)})
    result = data.validate_release(release(train, valid))["readiness"]
    reason = f"HELDOUT_{head.upper()}_KNOWN_SUPERVISION_MISSING"
    assert result["status"] == "NOT_READY" and not result["training_allowed"]
    assert not result["validation_available"] and result["holdout_group_available"]
    assert reason in result["reasons"]
    assert result["group_holdout"]["validation_missing_labels"] == [reason]
    assert result["group_holdout"]["training_groups"] == ["a"]
    assert result["group_holdout"]["validation_groups"] == ["z"]
