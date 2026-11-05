"""Synthetic contracts for count-blind saved-checkpoint diagnosis."""
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_diagnostics as diagnostics
from ethmoid_pipeline.counting import instance_experiment as legacy
from ethmoid_pipeline.counting import instance_learning as core


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def release(tmp_path, monkeypatch):
    affine = np.diag([.5, .5, .5, 1.])
    hu = np.full((8, 8, 8), -800, np.float32)
    source_path = tmp_path / "synthetic.nii"
    nib.save(nib.Nifti1Image(hu, affine), source_path)
    source = {"input": str(source_path), "source_sha256": legacy.sha256(source_path),
              "crop_bounds": [[0, 8]] * 3, "midline_ras_x": 2.}
    ids = np.full(hu.shape, -1, np.int32)
    ids[1:4, 1:6, 1:6] = 1
    ids[4:7, 1:6, 1:6] = 2
    fg = np.full(hu.shape, -1, np.int8)
    fg[ids > 0] = 1
    fg[0] = 0
    arrays = tmp_path / "labels.npz"
    np.savez_compressed(arrays, instance_ids=ids, foreground_labels=fg, affine=affine)
    frame = tmp_path / "synthetic-view.txt"
    frame.write_text("synthetic evidence", encoding="utf-8")
    evidence = [{"kind": kind, "source_match": "provisional", "boundary_support": "supported",
                 "point_ids": [1], "timestamps_seconds": [1, 2], "label_ids": labels,
                 "frames": [{"path": str(frame), "sha256": legacy.sha256(frame)}] * 2}
                for kind, labels in (("same_cell", [1]), ("same_cell", [2]),
                                     ("different_cell", [1, 2]), ("background", []))]
    annotation = write(tmp_path / "annotation.json", {"schema_version": 1, "source": source,
        "annotation": {"path": str(arrays), "sha256": legacy.sha256(arrays)},
        "distinct_pairs": [[1, 2]], "evidence": evidence})
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint; synthetic worker is injected")
    probabilities = np.ones((7, *hu.shape), np.float32)
    probabilities[0, 0] = 0
    decoded = core.decode_probabilities(probabilities)
    native = legacy.native_labels(decoded, hu.shape, affine)
    result = {"probabilities": probabilities, "valid": np.ones(hu.shape, bool), "decoded": decoded,
              "native_crop": native, "native": native, "native_affine": affine,
              "hu": hu, "crop_affine": affine, "torch_version": "synthetic",
              "training_annotation_manifest_sha256": legacy.sha256(annotation)}
    monkeypatch.setattr(diagnostics, "_reproduce", lambda *_: result)
    manifest = {"schema_version": 1, "version": diagnostics.VERSION, "source": source,
                "checkpoint": str(checkpoint), "checkpoint_sha256": legacy.sha256(checkpoint),
                "annotation_manifest": {"path": str(annotation), "sha256": legacy.sha256(annotation)}}
    return manifest, annotation, result


def test_alternate_positive_path_reconnects_correctly_negative_direct_edge():
    probabilities = np.zeros((7, 2, 2, 1), np.float32)
    probabilities[0] = 1
    probabilities[1, 0, 0, 0] = .1
    probabilities[2, 0, 0, 0] = .8
    probabilities[1, 0, 1, 0] = .8
    probabilities[2, 1, 0, 0] = .8
    ids = np.full((2, 2, 1), -1, np.int32)
    ids[0, 0, 0], ids[1, 0, 0] = 1, 2
    result = diagnostics.audit_arrays(probabilities, np.ones(ids.shape, bool), ids,
                                      np.full_like(ids, -1), [(1, 2)])
    assert result["heads"][1]["true_negative"] == 1
    path = result["distinct_pair_paths"][0]
    assert path["connected"] and path["edge_count"] == 3
    assert path["coordinates"] == [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    assert path["bottleneck_affinity"] == pytest.approx(.8)
    assert path["node_traversal"]["unknown"] == 2


def test_long_edge_reports_background_jump_and_unknown_interior():
    probabilities = np.zeros((7, 5, 1, 1), np.float32)
    probabilities[0, [0, 4], 0, 0] = 1
    probabilities[4, 0, 0, 0] = .7
    ids = np.full((5, 1, 1), -1, np.int32)
    ids[0, 0, 0], ids[4, 0, 0], ids[2, 0, 0] = 1, 2, 0
    result = diagnostics.audit_arrays(probabilities, np.ones(ids.shape, bool), ids,
                                      np.full_like(ids, -1), [(1, 2)])
    path = result["distinct_pair_paths"][0]
    assert path["edge_count"] == 1 and path["long_edge_background_bypasses"] == 1
    assert path["long_edge_known_background_crossings"] == 1
    assert path["long_edges"][0]["predicted_background_voxels"] == 3
    assert path["long_edges"][0]["intermediate_traversal"]["unknown"] == 2
    assert result["heads"][4]["false_positive"] == 1


def test_bfs_ties_use_sorted_flat_neighbor_indices_and_pairs_are_canonical():
    probabilities = np.ones((7, 2, 2, 1), np.float32)
    ids = np.full((2, 2, 1), -1, np.int32)
    ids[0, 0, 0], ids[1, 1, 0] = 1, 2
    outputs = [diagnostics.audit_arrays(probabilities, np.ones(ids.shape, bool), ids,
               np.full_like(ids, -1), pairs) for pairs in ([(1, 2)], [(2, 1), (1, 2)])]
    assert outputs[0] == outputs[1]
    assert outputs[0]["distinct_pair_paths"][0]["coordinates"] == [[0, 0, 0], [0, 1, 0], [1, 1, 0]]


def test_missing_foreground_is_reported_without_inventing_path():
    probabilities = np.zeros((7, 2, 1, 1), np.float32)
    ids = np.array([1, 2], np.int32).reshape(2, 1, 1)
    result = diagnostics.audit_arrays(probabilities, np.ones(ids.shape, bool), ids,
                                      np.full_like(ids, -1), [(1, 2)])
    assert result["heads"][0]["false_negative"] == 2
    assert not result["distinct_pair_paths"][0]["connected"]


def test_predictions_are_closed_before_annotation_is_read_and_inputs_stay_identical(tmp_path, release, monkeypatch):
    manifest, annotation, _ = release
    output = tmp_path / "diagnosis"
    read = legacy.read_json
    observed = []
    def checked_read(path):
        if str(path) == str(annotation):
            closed = json.loads((output / "prediction-closed.local.json").read_text())
            assert closed["annotation_loaded"] is False
            assert closed["probabilities_sha256"] == legacy.sha256(output / "probabilities.local.npz")
            observed.append(True)
        return read(path)
    monkeypatch.setattr(legacy, "read_json", checked_read)
    before = {str(p): legacy.sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    result = diagnostics.diagnose(write(tmp_path / "manifest.json", manifest), output)
    assert observed and result["status"] == "COMPLETE" and result["inputs_unchanged"]
    assert result["native_mechanism"]["violated_exclusion_voxels"] == 0
    assert all(legacy.sha256(path) == digest for path, digest in before.items())
    with np.load(output / "probabilities.local.npz", allow_pickle=False) as arrays:
        assert arrays["probabilities"].shape == (7, 8, 8, 8)
        np.testing.assert_array_equal(arrays["crop_index_axis_0"], np.arange(8))
        np.testing.assert_array_equal(arrays["model_affine"], np.diag([.5, .5, .5, 1.]))


@pytest.mark.parametrize("extra", [{"reference_counts": {"left": 1}}, {"threshold": .1}, {"target": 8}])
def test_reference_or_threshold_injection_fails_before_prediction(tmp_path, release, monkeypatch, extra):
    manifest, _, _ = release
    monkeypatch.setattr(diagnostics, "_reproduce", lambda *_: pytest.fail("must not predict"))
    with pytest.raises(legacy.ExperimentError, match="DIAGNOSTIC_MANIFEST_INVALID"):
        diagnostics.diagnose(write(tmp_path / "bad.json", {**manifest, **extra}), tmp_path / "never")
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("field,code", [("checkpoint_sha256", "CHECKPOINT_HASH_MISMATCH"),
                                       ("annotation_manifest", "ANNOTATION_MANIFEST_HASH_MISMATCH")])
def test_wrong_hashes_fail_closed_at_the_correct_stage(tmp_path, release, field, code):
    manifest, _, _ = release
    manifest[field] = "0" * 64 if field == "checkpoint_sha256" else {**manifest[field], "sha256": "0" * 64}
    output = tmp_path / "failed"
    with pytest.raises(legacy.ExperimentError, match=code):
        diagnostics.diagnose(write(tmp_path / "bad.json", manifest), output)
    assert (output / "prediction-closed.local.json").exists() == (field == "annotation_manifest")


def test_unexpected_error_is_sanitized_and_original_output_cannot_be_overwritten(tmp_path, release, monkeypatch):
    manifest, _, _ = release
    secret = "private-path-or-header"
    def fail(*_):
        raise OSError(secret)
    monkeypatch.setattr(diagnostics, "_reproduce", fail)
    output = tmp_path / "failed"
    path = write(tmp_path / "manifest.json", manifest)
    with pytest.raises(legacy.ExperimentError) as error:
        diagnostics.diagnose(path, output)
    assert str(error.value) == "EXPERIMENT_EXECUTION_FAILED" and secret not in str(error.value)
    previous = (output / "failure.local.json").read_bytes()
    assert secret.encode() not in previous
    with pytest.raises(legacy.ExperimentError, match="FRESH_OUTPUT_REQUIRED"):
        diagnostics.diagnose(path, output)
    assert (output / "failure.local.json").read_bytes() == previous


def test_optional_baseline_is_bound_and_native_arrays_match(tmp_path, release):
    manifest, _, reproduced = release
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    artifact = baseline / "instances.local.nii.gz"
    nib.save(nib.Nifti1Image(reproduced["native"], reproduced["native_affine"]), artifact)
    record = write(baseline / "prediction.local.json", {"status": "COMPLETE",
        "source_sha256": manifest["source"]["source_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": legacy.fingerprint(legacy.CONFIG), "artifact_sha256": legacy.sha256(artifact)})
    manifest["baseline_prediction"] = {"path": str(record), "sha256": legacy.sha256(record)}
    manifest["annotation_manifest_sha256"] = manifest["annotation_manifest"]["sha256"]
    manifest["annotation_manifest"] = manifest["annotation_manifest"]["path"]
    result = diagnostics.diagnose(write(tmp_path / "manifest.json", manifest), tmp_path / "diagnosis")
    assert result["baseline_match"]["status"] == "EXACT_NATIVE_MATCH"
