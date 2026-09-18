"""Contracts for real expert data/models; fixtures are explicitly synthetic."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.counting.profiles import file_sha256
from ethmoid_pipeline.counting.training import export_nnunet_dataset, patient_grouped_splits, validate_annotation


def annotation(tmp_path, index=0):
    image = nib.Nifti1Image(np.zeros((12, 12, 12), np.int16), np.eye(4))
    source = tmp_path / f"source-{index}.nii.gz"
    nib.save(image, source)
    labels = np.zeros(image.shape, np.uint8)
    labels[2:4, 2:8, 2:8] = 1
    labels[7:9, 2:8, 2:8] = 2
    labels[4:7, 8:10, 2:8] = 3
    mask = tmp_path / f"anatomy-{index}.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), mask)
    return {"sample_id": f"sample_{index}", "patient_group": f"group_{index // 2}",
        "source_path": str(source), "source_sha256": file_sha256(source),
        "anatomy_path": str(mask), "anatomy_sha256": file_sha256(mask),
        "protocol_sha256": "a" * 64,
        "expert_review": {"status": "adjudicated", "source": "independent_expert", "reviewer_token": "synthetic_test_only"}}


def test_patient_groups_never_leak_between_folds():
    records = [{"sample_id": f"s{i}", "patient_group": f"p{i // 3}"} for i in range(18)]
    splits = patient_grouped_splits(records, folds=3)
    groups = {r["sample_id"]: r["patient_group"] for r in records}
    assert splits == patient_grouped_splits(list(reversed(records)), folds=3)
    assert sorted(s for split in splits for s in split["val"]) == sorted(groups)
    for split in splits:
        assert not {groups[s] for s in split["train"]} & {groups[s] for s in split["val"]}


def test_split_needs_independent_patients_not_many_reconstructions():
    records = [{"sample_id": f"s{i}", "patient_group": "one_patient"} for i in range(12)]
    with pytest.raises(ValueError, match="fewer independent"):
        patient_grouped_splits(records, folds=2)


def test_split_rejects_duplicate_samples():
    with pytest.raises(ValueError, match="Duplicate"):
        patient_grouped_splits([{"sample_id": "s", "patient_group": "a"}, {"sample_id": "s", "patient_group": "b"}], folds=2)


@pytest.mark.parametrize("change", ["unreviewed", "automatic", "hash", "shape", "labels"])
def test_annotation_must_be_expert_bound_and_aligned(tmp_path, change):
    record = annotation(tmp_path)
    if change == "unreviewed":
        record["expert_review"]["status"] = "pending"
    elif change == "automatic":
        record["expert_review"]["source"] = "automatic_proposal"
    elif change == "hash":
        record["source_sha256"] = "0" * 64
    else:
        image = nib.load(record["anatomy_path"])
        data = np.asarray(image.dataobj).copy()
        if change == "shape":
            data = data[:-1]
        else:
            data[1, 1, 1] = 5
        nib.save(nib.Nifti1Image(data, image.affine), record["anatomy_path"])
        record["anatomy_sha256"] = file_sha256(record["anatomy_path"])
    with pytest.raises(ValueError):
        validate_annotation(record)


def test_dataset_export_is_reproducible_reviewed_and_does_not_train(tmp_path):
    records = [annotation(tmp_path, i) for i in range(4)]
    result = export_nnunet_dataset(records, tmp_path / "dataset", folds=2)
    assert result["state"] == "expert_dataset_exported_not_trained" and result["patient_groups"] == 2
    dataset = json.loads((tmp_path / "dataset" / "dataset.json").read_text())
    assert dataset["labels"] == {"background": 0, "ethmoid_left": 1, "ethmoid_right": 2, "sphenoid": 3}
    assert len(list((tmp_path / "dataset" / "imagesTr").glob("*.nii.gz"))) == 4
    with pytest.raises(ValueError, match="new directory"):
        export_nnunet_dataset(records, tmp_path / "dataset", folds=2)


def test_evaluation_distinguishes_equal_counts_from_instance_correctness():
    pytest.importorskip("scipy")
    from ethmoid_pipeline.counting.training import instance_agreement
    reference = np.zeros((16, 16, 16), np.uint16)
    reference[2:6, 2:6, 2:6] = 1
    reference[9:13, 9:13, 9:13] = 2
    predicted = np.zeros_like(reference)
    predicted[2:4, 2:6, 2:6] = 1
    predicted[4:6, 2:6, 2:6] = 2
    report = instance_agreement(predicted, reference)
    assert report["absolute_count_error"] == 0
    assert report["missed_instances"] == report["false_instances"] == 1
    assert report["split_reference_instances"] == 1
    merged = instance_agreement((reference > 0).astype(np.uint16), reference)
    assert merged["merged_predicted_instances"] == 1 and merged["count_error"] == -1


def test_empty_instance_evaluation_keeps_misses_and_false_instances():
    pytest.importorskip("scipy")
    from ethmoid_pipeline.counting.training import instance_agreement
    empty = np.zeros((4, 4, 4), np.uint16)
    full = np.ones_like(empty)
    assert instance_agreement(empty, full)["missed_instances"] == 1
    assert instance_agreement(full, empty)["false_instances"] == 1
    assert instance_agreement(empty, empty)["matched_instances"] == 0


def model_bundle(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    from ethmoid_pipeline.counting.models import LABELS
    root = tmp_path / "model"
    (root / "fold_0").mkdir(parents=True)
    (root / "fold_0" / "checkpoint_final.pth").write_bytes(b"synthetic-not-a-network")
    (root / "plans.json").write_text("{}")
    (root / "dataset.json").write_text(json.dumps({"labels": LABELS, "channel_names": {"0": "CT"}, "file_ending": ".nii.gz"}))
    report = tmp_path / "expert-evaluation.json"
    report.write_text(json.dumps({"reference_source": "independent_expert", "promotion_approved": True,
        "acceptance_thresholds": {"synthetic_test_only": True}, "patient_group_disjoint": True, "clinical_readiness_claim": False}))
    manifest = {"schema_version": 1, "backend": "nnunet_v2", "output_labels": dict(LABELS),
        "status": "expert_reference_review_passed", "model_id": "synthetic_test_only", "model_directory": "model",
        "folds": [0], "nnunet_version": "synthetic_test_only", "evaluation_report": report.name, "evaluation_sha256": file_sha256(report),
        "files": {str(p.relative_to(root)).replace("\\", "/"): file_sha256(p) for p in root.rglob("*") if p.is_file()}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, root


def test_no_direct_model_is_an_explicit_unavailable_outcome(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    from ethmoid_pipeline.counting.models import load_model_manifest
    from ethmoid_pipeline.counting.engine import CountingError
    with pytest.raises(CountingError) as error:
        load_model_manifest(tmp_path / "not-installed.json")
    assert error.value.code == "DIRECT_MODEL_UNAVAILABLE"


@pytest.mark.parametrize("changed", ["weights", "labels", "evidence", "escape", "unreviewed", "folds"])
def test_model_promotion_contract_rejects_unsafe_or_unreviewed_bundle(tmp_path, changed):
    path, manifest, root = model_bundle(tmp_path)
    from ethmoid_pipeline.counting.models import load_model_manifest
    from ethmoid_pipeline.counting.engine import CountingError
    assert load_model_manifest(path)[0]["model_id"] == "synthetic_test_only"
    if changed == "weights":
        (root / "fold_0" / "checkpoint_final.pth").write_bytes(b"changed")
    elif changed == "labels":
        manifest["output_labels"]["ethmoid_left"] = 9
    elif changed == "evidence":
        manifest["evaluation_sha256"] = "0" * 64
    elif changed == "escape":
        manifest["model_directory"] = ".."
    elif changed == "unreviewed":
        manifest["status"] = "untrained"
    else:
        manifest["folds"] = [0, 0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(CountingError) as error:
        load_model_manifest(path)
    assert error.value.code == "DIRECT_MODEL_INTEGRITY"


@pytest.mark.parametrize("fault", [None, "geometry", "missing_sphenoid"])
def test_direct_adapter_pins_runtime_and_verifies_native_labels(tmp_path, monkeypatch, fault):
    path, manifest, _ = model_bundle(tmp_path)
    import sys
    from types import SimpleNamespace
    import ethmoid_pipeline.counting.models as models
    from ethmoid_pipeline.counting.engine import CountingError
    reference = annotation(tmp_path, 6)
    source = Path(reference["source_path"])
    canonical = nib.as_closest_canonical(nib.load(source))
    calls = []
    class Predictor:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def initialize_from_trained_model_folder(self, model, **kwargs):
            assert kwargs == {"use_folds": (0,), "checkpoint_name": "checkpoint_final.pth"}
        def predict_from_files(self, inputs, outputs, **kwargs):
            assert inputs == [[str(source)]] and kwargs["num_processes_preprocessing"] == 1
            raw = nib.load(reference["anatomy_path"])
            data, affine = np.asarray(raw.dataobj).copy(), raw.affine.copy()
            if fault == "geometry":
                affine[0, 3] += 1
            if fault == "missing_sphenoid":
                data[data == 3] = 0
            nib.save(nib.Nifti1Image(data, affine), outputs[0] + ".nii.gz")
    monkeypatch.setattr(models, "_require_offline_runtime", lambda: None)
    monkeypatch.setattr(models.metadata, "version", lambda _: manifest["nnunet_version"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda x: x, cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "nnunetv2.inference.predict_from_raw_data", SimpleNamespace(nnUNetPredictor=Predictor))
    if fault:
        with pytest.raises(CountingError) as error:
            models.run_direct_model(path, source, canonical, tmp_path / "predicted")
        assert error.value.code == ("DIRECT_MODEL_GEOMETRY" if fault == "geometry" else "EMPTY_REGION")
    else:
        local, identity, sides = models.run_direct_model(path, source, canonical, tmp_path / "predicted")
        assert identity["localization"] == "nnunet_v2_direct_anatomy"
        assert local.region.shape == canonical.shape and local.excluded.any()
        assert set(np.unique(sides)) == {0, 1, 2}
        assert calls[0]["perform_everything_on_device"] is False and calls[0]["use_mirroring"] is False
