"""Boundary orchestration checks use generated arrays only."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_pipeline as pipeline
from ethmoid_pipeline.counting import boundary_learning as learning
from ethmoid_pipeline.counting import instance_experiment as safe


@pytest.mark.parametrize("affine", [np.diag([.4, .8, 1.2, 1]),
    np.array([[-.4, 0, 0, 5], [0, .8, 0, -3], [0, 0, 1.2, 1], [0, 0, 0, 1]]),
    np.array([[0, 0, -.4, 5], [0, .8, 0, -3], [1.2, 0, 0, 1], [0, 0, 0, 1]])])
def test_native_projection_preserves_voxel_positions_and_original_geometry(affine):
    original = nib.Nifti1Image(np.zeros((7, 8, 9), np.float32), affine)
    canonical = nib.as_closest_canonical(original)
    bounds = [[1, n-1] for n in canonical.shape]
    labels = np.zeros(tuple(b-a for a, b in bounds), np.int32)
    labels[1, 2, 3] = 9
    stored = pipeline.native_output(labels, original, canonical, {"crop_bounds": bounds})
    restored = nib.as_closest_canonical(nib.Nifti1Image(stored, affine))
    expected = np.zeros(canonical.shape, np.uint16)
    expected[tuple(slice(a, b) for a, b in bounds)] = labels
    assert np.array_equal(restored.dataobj, expected)
    assert np.array_equal(restored.affine, canonical.affine)


def test_checkpoint_tampering_and_configuration_mismatch_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path/"checkpoint.pt"
    torch.save({"version": "old-affinity-version"}, path)
    with pytest.raises(ValueError, match="CHECKPOINT_HASH_MISMATCH"):
        pipeline._load_model(path, "0"*64, torch)
    with pytest.raises(ValueError, match="CHECKPOINT_PROTOCOL_MISMATCH"):
        pipeline._load_model(path, safe.sha256(path), torch)


def test_inference_rejects_reference_inputs_before_device_access(tmp_path, monkeypatch):
    path = tmp_path/"request.json"
    safe.write_json(path, {"schema_version": 1, "version": learning.VERSION, "case_id": "generated",
                          "source": {}, "checkpoint": "unused", "checkpoint_sha256": "0"*64,
                          "partition_runtime": {}, "reference_counts": {"total": 2}})
    monkeypatch.setattr(pipeline, "_torch", lambda: pytest.fail("device accessed"))
    with pytest.raises(ValueError, match="BOUNDARY_MANIFEST_INVALID"):
        pipeline.infer(path, tmp_path/"output")


def test_deadline_expires_before_any_patch_prediction():
    torch = pytest.importorskip("torch")
    class Forbidden(torch.nn.Module):
        def forward(self, x):
            pytest.fail("expired inference ran model")
    with pytest.raises(ValueError, match="INFERENCE_TIME_BUDGET_EXCEEDED"):
        learning.probabilities(np.zeros((4, 4, 4), np.float32), Forbidden(), torch,
                               device="cpu", patch_size=8, stride=8, deadline=time.monotonic()-1)


def test_training_pools_reject_missing_separator_supervision():
    heads = {"foreground_labels": np.array([[[0, 1]]], np.int8),
             "separator_labels": np.array([[[-1, -1]]], np.int8)}
    with pytest.raises(ValueError, match="TRAIN_PARTITION_HEAD_CLASSES_MISSING"):
        pipeline._pools([(None, None, heads)])


def test_validation_cannot_select_checkpoint_from_only_one_head(monkeypatch):
    arrays = {"foreground_labels": np.zeros((4, 4, 4), np.int8),
              "separator_labels": np.full((4, 4, 4), -1, np.int8)}
    monkeypatch.setattr(learning, "probabilities", lambda *_, **__: np.full((2, 4, 4, 4), .5))
    with pytest.raises(ValueError, match="VALIDATION_HEAD_SUPERVISION_MISSING"):
        pipeline._validation([(np.zeros((4, 4, 4)), None, arrays)], None, None)


def test_partition_resource_preflight_precedes_device_access(tmp_path, monkeypatch):
    from ethmoid_pipeline.counting import boundary_data
    request = tmp_path/"request.json"
    safe.write_json(request, {"schema_version": 1, "version": learning.VERSION, "case_id": "generated",
                             "source": {}, "checkpoint": "unused", "checkpoint_sha256": "0"*64,
                             "partition_runtime": {}})
    monkeypatch.setattr(safe, "source_record", lambda _: {})
    monkeypatch.setattr(boundary_data, "_bounded_shape", lambda _: (200, 200, 200))
    monkeypatch.setattr(pipeline, "_torch", lambda: pytest.fail("device accessed"))
    with pytest.raises(ValueError, match="PARTITION_MEMORY_BUDGET_EXCEEDED"):
        pipeline.infer(request, tmp_path/"output")


def _cli():
    path = Path(__file__).parents[1]/"scripts"/"boundary_cell_pipeline.py"
    spec = importlib.util.spec_from_file_location("boundary_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_failure_is_sanitized_and_never_mutates_existing_output(tmp_path, monkeypatch, capsys):
    cli = _cli()
    manifest = tmp_path/"manifest.json"
    safe.write_json(manifest, {})
    output = tmp_path/"run"
    monkeypatch.setattr(sys, "argv", ["boundary_cell_pipeline.py", "evaluate", "--manifest", str(manifest),
                                     "--output", str(output)])
    monkeypatch.setattr(pipeline, "evaluate", lambda *_: (_ for _ in ()).throw(RuntimeError("PRIVATE_PATH_VALUE")))
    assert cli.main() == 2
    failure = json.loads((output/"failure.local.json").read_text())
    assert failure["error_code"] == "EXPERIMENT_EXECUTION_FAILED"
    assert "PRIVATE_PATH_VALUE" not in capsys.readouterr().err
    previous = (output/"failure.local.json").read_bytes()
    assert cli.main() == 2
    assert (output/"failure.local.json").read_bytes() == previous


def test_successful_export_returns_zero(tmp_path, monkeypatch, capsys):
    from ethmoid_pipeline.counting import boundary_data
    cli = _cli()
    monkeypatch.setattr(sys, "argv", ["boundary_cell_pipeline.py", "export", "--manifest", "unused",
                                     "--output", str(tmp_path/"unused")])
    monkeypatch.setattr(boundary_data, "export", lambda *_: {"status": "EXPORTED"})
    assert cli.main() == 0
    assert capsys.readouterr().out.strip() == "EXPORTED"


def test_supervisor_timeout_records_fixed_failure_and_terminates_worker(tmp_path, monkeypatch, capsys):
    import subprocess
    cli = _cli()
    worker = type("Worker", (), {"pid": 123456, "killed": False})()
    def communicate(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("PRIVATE_PATH", timeout)
        return "", ""
    worker.communicate = communicate
    worker.kill = lambda: setattr(worker, "killed", True)
    monkeypatch.delenv("ETHMOID_BOUNDARY_CHILD", raising=False)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_, **__: worker)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_, **__: None)
    monkeypatch.setattr(cli, "new_output", safe.new_output)
    output = tmp_path/"timed-out"
    monkeypatch.setattr(sys, "argv", ["boundary_cell_pipeline.py", "train", "--manifest", "unused",
                                     "--output", str(output)])
    assert cli.main() == 2 and worker.killed
    assert json.loads((output/"failure.local.json").read_text())["error_code"] == "PROCESS_TIME_BUDGET_EXCEEDED"
    assert capsys.readouterr().err.strip() == "PROCESS_TIME_BUDGET_EXCEEDED"


def test_quality_flags_withhold_empty_truncated_and_cross_midline():
    empty = np.zeros((5, 5, 5), np.int32)
    assert pipeline._quality_flags(empty, []) == ["no_supported_candidates"]
    empty[0, 2, 2] = 1
    assert pipeline._quality_flags(empty, [{"flags": []}]) == ["candidate_touches_crop_boundary"]
    empty[:] = 0; empty[2, 2, 2] = 1
    assert pipeline._quality_flags(empty, [{"flags": ["crosses_midline"]}]) == ["crosses_midline_requires_review"]
