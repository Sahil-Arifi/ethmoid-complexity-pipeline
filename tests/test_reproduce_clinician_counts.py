"""Synthetic inputs exercise reproduction bookkeeping, never patient accuracy."""
import importlib.util
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "scripts/reproduce_clinician_counts.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v2_reproduction_binds_current_counting_component():
    from ethmoid_pipeline.counting.engine import METHOD_VERSION, PROTOCOL_VERSION
    assert runner.METHODS["v2"] == (METHOD_VERSION, PROTOCOL_VERSION)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "synthetic.nii"
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), -900, np.int16), np.eye(4)), source)
    cache = tmp_path / "landmarks"
    cache.mkdir()
    write_json(cache / "localizers.local.json", {"synthetic": True})
    entry = {"case_id": "Synthetic_A", "input": str(source), "source_sha256": runner.sha256(source),
             "landmarks": str(cache), "acquisition": {"source_sha256": runner.sha256(source)},
             "reference": {"left": 1, "right": 1}}
    manifest = tmp_path / "inputs.json"
    write_json(manifest, {"entries": [entry]})
    return manifest, tmp_path / "runs", entry


def fake_worker(left=1, right=1, *, bad_registry=False, missing=False, changed_artifact=False, failed=False,
                wrong_method=False, tamper_source=False, ignore_config=False, floating_artifact=False,
                wrong_air=False, wrong_measurement=False):
    calls = []

    def execute(command, environment):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        if missing:
            return subprocess.CompletedProcess(command, 0, "", "")
        source = Path(command[command.index("--input") + 1])
        method = command[command.index("--method") + 1]
        versions = runner.METHODS[method]
        result = {"source_sha256": runner.sha256(source), "status": "failed" if failed else "completed",
                  "method_version": "different" if wrong_method else versions[0], "protocol_version": versions[1],
                  "config": runner.read_json(command[command.index("--config") + 1]),
                  "counts": {"left": None, "right": None, "total": None}, "cells": [], "artifacts": {}}
        if ignore_config:
            result["config"]["bone_hu"] += 1
        if not failed:
            count = left + right
            result["counts"] = {"left": left, "right": right, "total": count}
            labels = np.zeros((4, 4, 4), np.uint16)
            for i in range(count):
                labels[i + 1, 1, 1] = i + 1
            if changed_artifact and len(calls) == 2:
                labels = labels[:, ::-1].copy()
            result["cells"] = [{"id": i + 1, "side": "left" if i < left else "right", "included": True,
                                "origin": "automatic", "uncertain": False,
                                "voxel_count": 2 if wrong_measurement else 1, "air_voxel_count": 1,
                                "centroid_ras_mm": np.argwhere(labels == i + 1).mean(axis=0).tolist()}
                               for i in range(count)]
            for name in ("cavities", "air", "region"):
                path = output / (name + ".nii.gz")
                data = labels.copy()
                if name == "region":
                    data = np.where(labels == 0, 0, np.where(labels <= left, 1, 2)).astype(np.uint8)
                elif name == "air" and wrong_air:
                    data[:] = 0
                    for cell in result["cells"]:
                        cell["air_voxel_count"] = 0
                if name == "cavities" and floating_artifact:
                    data = data.astype(np.float32)
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                result["artifacts"][name] = path.name
            if bad_registry:
                result["cells"][0]["side"] = "right"
        write_json(output / "result.json", result)
        if tamper_source:
            source.write_bytes(b"modified during inference")
        return subprocess.CompletedProcess(command, 2 if failed else 0, "synthetic worker", "")

    execute.calls = calls
    return execute


def test_separate_workers_match_both_sides_and_bind_inputs_without_sending_targets(inputs):
    manifest, output, entry = inputs
    execute = fake_worker()
    summary = runner.run(manifest, output, repeat=3, worker=execute)
    assert summary["success"] and summary["exact_runs"] == summary["scheduled_runs"] == 3
    assert summary["cases"][0]["repeatable"] is True
    assert len(execute.calls) == 3
    for command in execute.calls:
        assert "--manifest" not in command and "--reference" not in command
        assert str(manifest) not in command and "--method" in command
    provenance = json.loads((output / "provenance.local.json").read_text())
    assert provenance["manifest_sha256"] == runner.sha256(manifest)
    assert provenance["sources"] == {entry["case_id"]: entry["source_sha256"]}
    assert provenance["code_sha256"] and provenance["inputs_unchanged"]


def test_matching_total_with_wrong_sides_is_mismatch(inputs):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=2, worker=fake_worker(0, 2))
    assert not summary["success"] and summary["exact_runs"] == 0
    assert summary["records"][0]["deltas"] == {"left": -1, "right": 1, "total": 0}


def test_matching_counts_with_changed_native_labels_are_not_repeatable(inputs):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=2, worker=fake_worker(changed_artifact=True))
    assert summary["exact_runs"] == 2
    assert not summary["success"] and summary["cases"][0]["repeatable"] is False


@pytest.mark.parametrize("options,code", [({"missing": True}, "WORKER_RESULT_MISSING"),
                                        ({"bad_registry": True}, "COUNT_REGISTRY_MISMATCH"),
                                        ({"wrong_method": True}, "WORKER_METHOD_MISMATCH"),
                                        ({"ignore_config": True}, "WORKER_CONFIG_MISMATCH"),
                                        ({"floating_artifact": True}, "ARTIFACT_DTYPE_INVALID"),
                                        ({"wrong_air": True}, "NATIVE_HU_AIR_MISMATCH"),
                                        ({"wrong_measurement": True}, "NATIVE_ARTIFACT_IDENTITY_MISMATCH")])
def test_invalid_result_remains_a_failed_row_in_every_repeat(inputs, options, code):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=2, worker=fake_worker(**options))
    assert len(summary["records"]) == summary["scheduled_runs"] == 2
    assert summary["technical_failures"] == 2 and not summary["success"]
    assert all(row["failure_code"] == code for row in summary["records"])


def test_source_mutation_invalidates_batch_and_does_not_drop_later_attempts(inputs):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=2, worker=fake_worker(tamper_source=True))
    assert summary["scheduled_runs"] == len(summary["records"]) == 2
    assert summary["technical_failures"] == 2 and not summary["inputs_unchanged"]
    assert not summary["success"]
    assert summary["records"][0]["failure_code"] == "SOURCE_CHANGED_DURING_RUN"


def test_worker_exception_is_retained_as_failed_attempts(inputs):
    manifest, output, _ = inputs

    def broken(command, environment):
        raise OSError("synthetic inaccessible worker")

    summary = runner.run(manifest, output, repeat=2, worker=broken)
    assert summary["technical_failures"] == len(summary["records"]) == 2
    assert all(row["failure_code"] == "RUNNER_TECHNICAL_FAILURE" for row in summary["records"])


def test_final_unreadable_protected_file_invalidates_summary_without_losing_rows(inputs, monkeypatch):
    manifest, output, entry = inputs
    execute = fake_worker()
    original_hash = runner.sha256
    cache_manifest = Path(entry["landmarks"]) / "localizers.local.json"

    def guarded_hash(path):
        if len(execute.calls) == 2 and Path(path) == cache_manifest:
            raise PermissionError("synthetic protected input unreadable")
        return original_hash(path)

    monkeypatch.setattr(runner, "sha256", guarded_hash)
    summary = runner.run(manifest, output, repeat=2, worker=execute)
    assert len(summary["records"]) == summary["exact_runs"] == 2
    assert not summary["inputs_unchanged"] and not summary["success"]
    assert (output / "summary.local.json").is_file()


def test_invalid_global_config_is_rejected_before_output(inputs):
    manifest, output, _ = inputs
    config = manifest.with_name("config.json")
    write_json(config, {"unsupported_field": 1})
    with pytest.raises(runner.ReproductionError, match="CONFIG_INVALID"):
        runner.run(manifest, output, config=config, worker=fake_worker())
    assert not output.exists()


def test_worker_abstention_retains_nulls_and_cannot_be_repeatability_success(inputs):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=2, worker=fake_worker(failed=True))
    assert summary["withheld_runs"] == 2 and summary["exact_runs"] == 0
    assert not summary["success"] and summary["cases"][0]["repeatable"] is False
    assert all(row["counts"] == {"left": None, "right": None, "total": None} for row in summary["records"])


def test_one_run_cannot_prove_repeatability(inputs):
    manifest, output, _ = inputs
    summary = runner.run(manifest, output, repeat=1, worker=fake_worker())
    assert summary["exact_runs"] == 1 and summary["cases"][0]["repeatable"] is None
    assert not summary["success"]


def test_output_is_immutable_even_if_empty(inputs):
    manifest, output, _ = inputs
    output.mkdir()
    with pytest.raises(runner.ReproductionError, match="OUTPUT_EXISTS"):
        runner.run(manifest, output, worker=fake_worker())
    assert list(output.iterdir()) == []


def test_output_cannot_be_written_inside_checkout(inputs, tmp_path, monkeypatch):
    manifest, _, _ = inputs
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(runner.ReproductionError, match="EXTERNAL_DATA_PATH_REQUIRED"):
        runner.run(manifest, tmp_path / "repo-output", worker=fake_worker())


@pytest.mark.parametrize("fault,code", [("duplicate", "DUPLICATE_CASE"), ("tamper", "SOURCE_HASH_MISMATCH"),
                                      ("boolean", "REFERENCE_INVALID"), ("acquisition", "ACQUISITION_SOURCE_MISMATCH")])
def test_bad_manifest_fails_before_any_inference_or_output(inputs, fault, code):
    manifest, output, entry = inputs
    data = {"entries": [entry]}
    if fault == "duplicate":
        data["entries"].append({**entry, "case_id": entry["case_id"].lower()})
    elif fault == "tamper":
        Path(entry["input"]).write_bytes(b"changed source")
    elif fault == "boolean":
        entry["reference"]["left"] = True
    else:
        entry["acquisition"]["source_sha256"] = "0" * 64
    write_json(manifest, data)
    execute = fake_worker()
    with pytest.raises(runner.ReproductionError, match=code):
        runner.run(manifest, output, worker=execute)
    assert execute.calls == [] and not output.exists()
