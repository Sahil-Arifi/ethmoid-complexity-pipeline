"""Synthetic contracts for CT-only connectivity-fork evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import connectivity_evaluation as evaluation
from ethmoid_pipeline.counting import instance_experiment as ex


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def protocol(monkeypatch):
    runner = SimpleNamespace(VERSION="video-guided-connectivity-prototype-2",
                             CONFIG={**ex.CONFIG, "steps": 128, "connectivity_loss": True})
    monkeypatch.setattr(evaluation, "_runner", lambda: runner)
    return runner


def source(tmp_path, shape=(12, 8, 8), affine=None):
    affine = np.diag([.5, .5, .5, 1.]) if affine is None else affine
    path = tmp_path/"synthetic.nii"
    hu = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)-1000
    nib.save(nib.Nifti1Image(hu, affine), path)
    return {"input": str(path), "source_sha256": ex.sha256(path),
            "crop_bounds": [[0, n] for n in shape], "midline_ras_x": 2.5}


def inference_manifest(tmp_path, record):
    checkpoint = tmp_path/"checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint, never loaded by torch")
    return {"schema_version": 1, "case_id": "Synthetic", "source": record,
            "checkpoint": str(checkpoint), "checkpoint_sha256": ex.sha256(checkpoint)}


@pytest.mark.parametrize("extra", ["annotation", "reference", "readiness", "points"])
def test_infer_rejects_extra_fields_before_model_or_source_read(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(evaluation, "_runner", lambda: pytest.fail("model module must stay unopened"))
    monkeypatch.setattr(ex, "source_record", lambda _: pytest.fail("reject top-level leakage first"))
    manifest = write(tmp_path/"manifest.json", {"schema_version": 1, "case_id": "Synthetic", "source": {},
                    "checkpoint": "unused", "checkpoint_sha256": "unused", extra: "do not open"})
    with pytest.raises(ex.ExperimentError, match="INFERENCE_MANIFEST_INVALID"):
        evaluation.infer(manifest, tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_nested_reference_is_rejected_before_checkpoint_read(tmp_path, monkeypatch):
    manifest = inference_manifest(tmp_path, {**source(tmp_path), "reference": [1, 2]})
    path = write(tmp_path/"manifest.json", manifest)
    monkeypatch.setattr(evaluation, "_runner", lambda: pytest.fail("no checkpoint/model read"))
    with pytest.raises(ex.ExperimentError, match="SOURCE_SCHEMA_INVALID"):
        evaluation.infer(path, tmp_path/"output")


def test_projection_restores_native_orientation_and_preserves_inputs():
    affine = np.diag([-.5, .75, 1., 1.]); affine[0, 3] = 5
    hu = np.arange(8*9*10, dtype=np.float32).reshape(8, 9, 10)
    original = nib.Nifti1Image(hu.copy(), affine)
    canonical = nib.as_closest_canonical(original)
    bounds = [[1, 5], [2, 6], [3, 7]]
    labels = np.ones((4, 4, 4), np.uint16)
    stored = evaluation.project_native(labels, original, canonical, bounds)
    restored = nib.as_closest_canonical(nib.Nifti1Image(stored, original.affine))
    expected = np.zeros(canonical.shape, np.uint16); expected[1:5, 2:6, 3:7] = 1
    assert np.array_equal(np.asanyarray(restored.dataobj), expected)
    assert np.array_equal(np.asanyarray(original.dataobj), hu)
    assert np.array_equal(original.affine, affine)
    assert stored.shape == original.shape and stored.dtype == np.uint16


def test_ct_only_infer_reuses_geometry_and_never_opens_labels(tmp_path, monkeypatch, protocol):
    record = source(tmp_path)
    record["crop_bounds"] = [[1, 11], [1, 7], [1, 7]]
    path = write(tmp_path/"manifest.json", inference_manifest(tmp_path, record))
    source_hash = ex.sha256(record["input"])
    calls = []
    class Model:
        def to(self, device): calls.append(device)
    protocol.load_checkpoint = lambda checkpoint, torch: (Model(), {"annotation_manifest_sha256": "a"*64,
        "protocol_sha256": "b"*64, "annotation_path": "must not open"})
    protocol.select_constraints = lambda *_: pytest.fail("inference cannot select constraints")
    monkeypatch.setattr(ex, "deterministic_torch", lambda: object())
    monkeypatch.setattr(ex, "inspect_readiness", lambda *_: pytest.fail("inference cannot inspect readiness"))
    monkeypatch.setattr(ex, "_annotation", lambda *_: pytest.fail("inference cannot open labels"))
    def probabilities(hu, model, torch):
        result = np.ones((7, *hu.shape), np.float32)
        result[0] = 0
        result[0, 1:3, 1:3, 1:3] = 1
        result[0, 7:9, 1:3, 1:3] = 1
        return result
    monkeypatch.setattr(ex, "tiled_probabilities", probabilities)
    result = evaluation.infer(path, tmp_path/"output")
    assert calls == ["cuda"] and result["counts"]["total"] == 2
    assert result["version"] == protocol.VERSION
    assert result["config_sha256"] == ex.fingerprint(protocol.CONFIG)
    assert result["annotation_manifest_sha256"] == "a"*64
    assert ex.sha256(record["input"]) == source_hash
    validated, _, _ = evaluation._prediction(tmp_path/"output/prediction.local.json")
    assert validated == result
    result["registry"][0]["side"] = "right"
    write(tmp_path/"output/prediction.local.json", result)
    with pytest.raises(ex.ExperimentError, match="PREDICTION_REGISTRY_INVALID"):
        evaluation._prediction(tmp_path/"output/prediction.local.json")


def partial_cores():
    labels = np.full((9, 5, 4), -1, np.int32)
    for identity in range(1, 8): labels[identity, 1, 1:3] = identity
    foreground = np.full_like(labels, -1)
    foreground[labels > 0] = 1; foreground[0, 0, 0] = 0
    return labels, foreground, [[1, 5], [2, 3], [6, 7]]


def test_supported_regions_ignore_unknowns_and_expose_split_merge_and_missing():
    labels, foreground, pairs = partial_cores()
    values = np.maximum(labels, 0).astype(np.uint16); values[8, 4, 3] = 99
    scored = evaluation.score_supported_regions(values, labels, foreground, pairs)
    assert all(scored["gates"].values()) and scored["unknown_voxels_unscored"] > 0
    assert scored["false_positive_known_background_voxels"] == 0
    assert scored["whole_cell_false_splits"] == "UNMEASURED_OUTSIDE_PROVISIONAL_CORES"
    merged = values.copy(); merged[labels == 3] = 2
    scored = evaluation.score_supported_regions(merged, labels, foreground, pairs)
    assert scored["supported_merges"][0]["annotation_ids"] == [2, 3]
    separated = values.copy(); separated[1, 1, 2] = 100
    scored = evaluation.score_supported_regions(separated, labels, foreground, pairs)
    assert scored["supported_splits"][0]["annotation_id"] == 1
    assert not scored["gates"]["annotated_within_cell_connections"]
    values[labels == 7] = 0
    scored = evaluation.score_supported_regions(values, labels, foreground, pairs)
    assert not scored["gates"]["known_foreground_coverage"]
    assert scored["missing_known_foreground_voxels"] == 2


def test_supported_regions_report_simultaneous_split_and_merge():
    labels, foreground, pairs = partial_cores()
    values = np.maximum(labels, 0).astype(np.uint16)
    values[2, 1, 2] = 3
    scored = evaluation.score_supported_regions(values, labels, foreground, pairs)
    merged = scored["supported_merges"]
    assert len(merged) == 1 and merged[0]["annotation_ids"] == [2, 3]
    assert merged[0]["observed_prediction_ids"] == [[2, 3], [3]]
    assert merged[0]["shared_prediction_ids"] == [3]
    assert merged[0]["prediction_ids"] == [None, 3]
    assert not merged[0]["separate"]
    assert [r["annotation_id"] for r in scored["supported_splits"]] == [2]
    assert not scored["gates"]["three_supported_distinctions"]
    assert not scored["gates"]["annotated_within_cell_connections"]


def test_supported_regions_report_partial_missing_core_and_merge():
    labels, foreground, pairs = partial_cores()
    values = np.maximum(labels, 0).astype(np.uint16)
    values[2, 1, 1] = 0; values[2, 1, 2] = 3
    scored = evaluation.score_supported_regions(values, labels, foreground, pairs)
    merged = scored["supported_merges"]
    assert len(merged) == 1 and merged[0]["annotation_ids"] == [2, 3]
    assert merged[0]["observed_prediction_ids"] == [[3], [3]]
    assert merged[0]["shared_prediction_ids"] == [3]
    assert merged[0]["prediction_ids"] == [None, 3]
    assert not merged[0]["separate"]
    assert not scored["supported_splits"] and scored["missing_known_foreground_voxels"] == 1
    assert not scored["gates"]["three_supported_distinctions"]
    assert not scored["gates"]["known_foreground_coverage"]


def frozen_entries():
    pairs = [("Case024_Series002", (13, 10), (13, 8)), ("Case012_Series007", (9, 16), (12, 17)),
             ("Case010_Series002", (14, 11), (9, 4)), ("Case008_Series004", (10, 5), (10, 5)),
             ("Case003_Series007", (9, 12), (6, 7)), ("Case006_Series007", (15, 12), (13, 10)),
             ("Case025_Series001", (12, 14), (9, 12)), ("Case016_Series003", (9, 8), (8, 9))]
    return [{"case_id": case, "reference": dict(zip(("left", "right"), reference)),
             "baseline": dict(zip(("left", "right"), baseline)), "source_sha256": "a"*64,
             "source_geometry_sha256": "b"*64, "predictions": []} for case, reference, baseline in pairs]


def closed_comparison(tmp_path, protocol):
    entries = frozen_entries()
    checkpoint = tmp_path/"one-checkpoint.pt"; checkpoint.write_bytes(b"one frozen synthetic checkpoint")
    for index, entry in enumerate(entries):
        directory = tmp_path/entry["case_id"]; directory.mkdir()
        left, right = entry["reference"].values(); maximum = max(left, right)
        affine = np.diag([.5, .5, .5, 1.]); affine[0, 3] = -.5*maximum
        record = source(directory, shape=(2*maximum+1, 4, 2), affine=affine)
        record["midline_ras_x"] = 0
        entry["source_sha256"] = record["source_sha256"]
        entry["source_geometry_sha256"] = ex.fingerprint(record)
        manifest = {"schema_version": 1, "case_id": entry["case_id"], "source": record,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": ex.sha256(checkpoint)}
        manifest_path = write(directory/"ct-only.json", manifest)
        labels = np.zeros((2*maximum+1, 4, 2), np.uint16)
        for identity in range(1, left+1): labels[identity-1, :2] = identity
        for identity in range(1, right+1): labels[maximum+identity, :2] = left+identity
        _, _, hu, crop_affine = ex.load_crop(record)
        registry, counts = ex.label_registry(labels, hu, crop_affine, 0)
        assert (counts["left"], counts["right"]) == (left, right)
        for repeat in range(2):
            output = directory/f"repeat-{repeat}"; output.mkdir()
            artifact = output/"instances.local.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), artifact)
            prediction = {"status": "COMPLETE", "version": protocol.VERSION, "case_id": entry["case_id"],
                "worker_identity": f"synthetic-process-{index}-{repeat}", "worker_pid": 100+index*2+repeat,
                "counts": counts, "registry": registry, "source_sha256": record["source_sha256"],
                "source_geometry_sha256": ex.fingerprint(record), "inference_manifest_path": str(manifest_path),
                "inference_manifest_sha256": ex.sha256(manifest_path), "checkpoint_sha256": ex.sha256(checkpoint),
                "config_sha256": ex.fingerprint(protocol.CONFIG), "protocol_sha256": "b"*64,
                "annotation_manifest_sha256": "a"*64, "native_array_sha256": ex.hashlib.sha256(labels.tobytes()).hexdigest(),
                "native_affine": affine.tolist(), "native_shape": list(labels.shape),
                "artifact_sha256": ex.sha256(artifact), "inputs_unchanged": True}
            entry["predictions"].append(str(write(output/"prediction.local.json", prediction)))
    mechanism = {"status": "PASS_CONDITIONAL_MECHANISM", "version": protocol.VERSION,
        "config_sha256": ex.fingerprint(protocol.CONFIG), "checkpoint_sha256": ex.sha256(checkpoint),
        "annotation_manifest_sha256": "a"*64, "protocol_sha256": "b"*64,
        "prediction_sha256": ex.sha256(entries[0]["predictions"][0]),
        "gates": {"three_supported_distinctions": True, "annotated_within_cell_connections": True,
                  "known_foreground_coverage": True, "explicit_exclusions_preserved": True}}
    readiness = {"version": protocol.VERSION, "training_allowed": True, "evolving_full_graph": True,
        "partial_supervision_only": True, "long_continuation_unmeasured": True,
        "annotation_manifest_sha256": "a"*64, "protocol_sha256": "b"*64}
    return {"schema_version": 1, "entries": entries,
            "mechanism": str(write(tmp_path/"mechanism.json", mechanism)),
            "readiness": str(write(tmp_path/"readiness.json", readiness))}


def test_report_preserves_eight_case_failure_denominator_and_no_fallback(tmp_path, protocol):
    manifest = write(tmp_path/"report.json", {"schema_version": 1, "entries": frozen_entries()})
    result = evaluation.report(manifest, tmp_path/"report-output")
    assert result["status"] == "NOT_RUN" and result["scheduled_cases"] == 8
    assert result["scheduled_sides"] == 16 and result["complete_cases"] == 0
    assert result["bilateral_exact"] == result["side_exact"] == 0
    assert result["absolute_side_error"] is None
    assert all(row["prototype"] is None for row in result["cases"])
    bad = frozen_entries(); bad[-1]["case_id"] = "DifferentStudy"
    write(tmp_path/"wrong-cohort.json", {"schema_version": 1, "entries": bad})
    with pytest.raises(ex.ExperimentError, match="FROZEN_CASE_SET_MISMATCH"):
        evaluation.report(tmp_path/"wrong-cohort.json", tmp_path/"bad-output")


def test_report_binds_fresh_repeats_mechanism_and_readiness(tmp_path, protocol):
    manifest = closed_comparison(tmp_path, protocol)
    path = write(tmp_path/"report.json", manifest)
    result = evaluation.report(path, tmp_path/"report-output")
    assert result["status"] == "PASS_CONDITIONAL_DEVELOPMENT"
    assert all(result["gates"].values()) and result["bilateral_exact"] == 8 and result["side_exact"] == 16
    assert result["absolute_side_error"] == result["other_seven_absolute_side_error"] == 0
    assert result["baseline_absolute_side_error"] == 37 and result["baseline_other_seven_absolute_side_error"] == 35
    prediction_path = Path(manifest["entries"][0]["predictions"][1])
    prediction = json.loads(prediction_path.read_text()); prediction["worker_pid"] -= 1
    write(prediction_path, prediction)
    result = evaluation.report(path, tmp_path/"same-process-output")
    assert result["status"] == "FAIL" and result["complete_cases"] == 7
    assert result["bilateral_exact"] == 7 and result["side_exact"] == 14
    assert result["cases"][0]["status"] == "NONREPEATABLE" and result["cases"][0]["prototype"] is None
    assert result["absolute_side_error"] is None
    assert not result["gates"]["two_repeat_identical_native_masks_and_counts"]
    Path(manifest["entries"][1]["predictions"][0]).unlink()
    result = evaluation.report(path, tmp_path/"missing-output")
    assert result["complete_cases"] == 6 and result["cases"][1]["status"] == "FAILED"
    assert result["cases"][1]["failure_codes"] == ["MANIFEST_INVALID"]


def test_report_rejects_old_versions_registry_tampering_and_unbound_readiness(tmp_path, protocol):
    manifest = closed_comparison(tmp_path, protocol)
    path = write(tmp_path/"report.json", manifest)
    ready_path = Path(manifest["readiness"])
    ready = json.loads(ready_path.read_text()); ready["annotation_manifest_sha256"] = "c"*64
    write(ready_path, ready)
    result = evaluation.report(path, tmp_path/"unbound-output")
    assert result["count_status"] == "PASS_DEVELOPMENT_COUNTS"
    assert result["status"] == "FAIL" and not result["gates"]["bound_conditional_readiness"]
    prediction_path = Path(manifest["entries"][0]["predictions"][0])
    prediction = json.loads(prediction_path.read_text()); prediction["version"] = ex.VERSION
    write(prediction_path, prediction)
    result = evaluation.report(path, tmp_path/"old-output")
    assert result["complete_cases"] == 7 and result["cases"][0]["failure_codes"] == ["PREDICTION_PROTOCOL_INVALID"]
    prediction["version"] = protocol.VERSION; prediction["registry"][0]["side"] = "right"
    write(prediction_path, prediction)
    result = evaluation.report(path, tmp_path/"tampered-output")
    assert result["complete_cases"] == 7 and result["cases"][0]["failure_codes"] == ["PREDICTION_REGISTRY_INVALID"]


def test_report_requires_preserved_crop_binding_and_fails_malformed_mechanism(tmp_path, protocol):
    manifest = closed_comparison(tmp_path, protocol)
    path = write(tmp_path/"report.json", manifest)
    mechanism_path = Path(manifest["mechanism"])
    mechanism = json.loads(mechanism_path.read_text()); mechanism["gates"] = 7
    write(mechanism_path, mechanism)
    result = evaluation.report(path, tmp_path/"malformed-mechanism")
    assert result["status"] == "FAIL" and result["count_status"] == "PASS_DEVELOPMENT_COUNTS"
    assert result["mechanism_status"] == "FAIL"
    manifest["entries"][0]["source_geometry_sha256"] = "f"*64
    write(path, manifest)
    result = evaluation.report(path, tmp_path/"wrong-crop")
    assert result["complete_cases"] == 7
    assert result["cases"][0]["failure_codes"] == ["PREDICTION_SOURCE_MISMATCH"]*2
    assert result["cases"][0]["prototype"] is None
    del manifest["entries"][0]["source_geometry_sha256"]
    write(path, manifest)
    with pytest.raises(ex.ExperimentError, match="REPORT_ENTRY_INVALID"):
        evaluation.report(path, tmp_path/"missing-crop-binding")
