"""Synthetic cohorts and subprocesses test evaluation without medical data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("evaluate_counting_cohort", Path(__file__).resolve().parents[1] / "scripts/evaluate_counting_cohort.py")
assert spec and spec.loader
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


FAKE_WORKER = '''import argparse, hashlib, json, os, sys
from pathlib import Path
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser()
for name in ('input','output','config','landmarks'):
    p.add_argument('--'+name)
a=p.parse_args()
source=Path(a.input)
destination=Path(a.output)
destination.mkdir()
image=nib.load(source)
config=json.loads(Path(a.config).read_text())['algorithm']
result={'status':'completed','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol_version':'ethmoid-count-prototype-v1','method_version':'synthetic-1','config':config,'counts':{'left':1,'right':0,'total':1},'cells':[],'artifacts':{}}
cavities=np.zeros(image.shape,dtype=np.uint16)
cavities[1:3,2:4,3:5]=1
air=np.zeros_like(cavities)
result['cells']=[{'id':1,'side':'left','included':True,'origin':'automatic','uncertain':True,'voxel_count':8,'air_voxel_count':0,'centroid_ras_mm':nib.affines.apply_affine(image.affine,[1.5,2.5,3.5]).tolist()}]
for name,data in [('cavities',cavities),('air',air),('region',(cavities>0).astype(np.uint8))]:
    filename=name+'.nii.gz'
    nib.save(nib.Nifti1Image(data,image.affine),destination/filename)
    result['artifacts'][name]=filename
fault=os.environ.get('EVALUATION_TEST_FAULT','')
if fault=='one_failure' and source.name.startswith('Case002'):
    result.update(status='failed',counts={'left':None,'right':None,'total':None},cells=[],artifacts={},failure_code='TARGET_CLIPPED')
elif fault=='false_count_failure':
    result.update(status='failed',failure_code='TARGET_CLIPPED')
elif fault=='source_mutation':
    source.write_bytes(source.read_bytes()+b'changed')
elif fault=='code_mutation':
    target=Path(__file__).with_name('engine.py')
    target.write_text(target.read_text()+'\\n# changed during worker\\n')
elif fault=='config_mutation':
    result['config']['air_hu']=-400
elif fault=='missing_result':
    sys.exit(1)
elif fault=='final_cohort_mutation' and source.name.startswith('Case005'):
    target=source.parent/'cohort.v1.local.json'
    target.write_bytes(target.read_bytes()+b'\\n')
(destination/'result.json').write_text(json.dumps(result))
print('SYNTHETIC-PROTECTED-WORKER-TEXT')
sys.exit(0 if result['status']=='completed' else 2)
'''


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    work = tmp_path / "local/EthmoidComplexity/work"
    (work / "counting").mkdir(parents=True)
    pipeline = tmp_path / "code"
    package = pipeline / "src/ethmoid_pipeline/counting"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "engine.py").write_text("class CountingConfig:\n    air_hu: float = -500.0\n    bone_hu: float = 200.0\n")
    (package / "localization.py").write_text("# Synthetic test stub; no inference.\n")
    (package / "worker.py").write_text(FAKE_WORKER)
    algorithm = {"air_hu": -500.0, "bone_hu": 200.0}
    (package / "protocol.json").write_text(json.dumps({"protocol_version": evaluation.PROTOCOL_VERSION, "algorithm": algorithm}))
    (pipeline / ".gitignore").write_text("__pycache__/\n")
    git(pipeline, "init", "-q")
    git(pipeline, "config", "user.name", "Synthetic tests")
    git(pipeline, "config", "user.email", "synthetic@example.invalid")
    git(pipeline, "config", "core.autocrlf", "false")
    git(pipeline, "add", ".")
    git(pipeline, "commit", "-q", "-m", "Synthetic fixture")
    entries = []
    for index in range(1, 11):
        name = f"Case{index:03}"
        path = work / "counting" / (name + ".nii.gz")
        affine = np.diag([-0.6, 0.7, 0.8, 1.0])
        affine[:3, 3] = (3, -4, -2)
        data = np.full((6, 7, 8), -900, dtype=np.int16)
        data[0, 0, 0] -= index
        nib.save(nib.Nifti1Image(data, affine), path)
        entries.append({"case": name, "series": "Series001", "split": "development" if index <= 5 else "held_out", "nifti": path.relative_to(work).as_posix(), "nifti_sha256": evaluation.sha256(path), "automated_qc": "PASS", "hu_max_absolute_error": 0})
    manifest = {"schema_version": 1, "protocol_version": evaluation.PROTOCOL_VERSION, "entries": entries}
    manifest_path = work / "counting/cohort.v1.local.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(evaluation, "_runtime_provenance", lambda _: {"synthetic_test_only": True, "weights_manifest_sha256": "0" * 64})
    return {"work_root": work, "pipeline_root": pipeline, "runtime_python": Path(sys.executable), "output": work / "counting/evaluation-synthetic", "split": "development"}, manifest, algorithm


def test_serial_subprocesses_have_consistent_artifacts_and_null_failures(cohort, monkeypatch, capsys):
    options, manifest, algorithm = cohort
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    summary = evaluation.evaluate(**options)
    assert summary["attempted"] == 5 and summary["completed"] == 4 and summary["failed"] == 1
    assert summary["failure_codes"] == {"TARGET_CLIPPED": 1}
    assert summary["counts_from_completed_only"] == {"left": 4, "right": 0, "total": 4}
    assert summary["accuracy_measured"] is False
    assert summary["diagnostics_from_completed_only"] == {"fully_opaque_candidates": 4, "partly_opaque_candidates": 0, "uncertain_candidates": 4}
    assert "geometric_air_coverage" not in summary
    attempt = evaluation.read_json(options["output"] / "Case002/attempt.local.json")
    assert attempt["counts"] == evaluation.NULL_COUNTS and attempt["failure_code"] == "TARGET_CLIPPED"
    assert attempt["elapsed_seconds"] > 0
    success = evaluation.read_json(options["output"] / "Case001/attempt.local.json")
    assert success["native_geometry"] == "PASS" and success["count_identity"] == "PASS"
    provenance = evaluation.read_json(options["output"] / "provenance.local.json")
    assert provenance["algorithm_sha256"] == evaluation.canonical_sha256(algorithm)
    assert provenance["code"]["git_commit"] == git(options["pipeline_root"], "rev-parse", "HEAD")
    assert provenance["code"]["git_dirty"] is False
    assert len(provenance["code"]["module_sha256"]) == 4
    assert capsys.readouterr().out == ""
    assert "SYNTHETIC-PROTECTED-WORKER-TEXT" in (options["output"] / "Case001/worker.local.log").read_text()
    before = evaluation.sha256(options["output"] / "summary.local.json")
    with pytest.raises(evaluation.EvaluationError, match="IMMUTABLE_OUTPUT"):
        evaluation.evaluate(**options)
    assert evaluation.sha256(options["output"] / "summary.local.json") == before


def test_development_never_opens_heldout_images(cohort):
    options, manifest, _ = cohort
    for entry in manifest["entries"]:
        if entry["split"] == "held_out":
            (options["work_root"] / entry["nifti"]).unlink()
    assert evaluation.evaluate(**options)["completed"] == 5


def test_heldout_requires_explicit_complete_clean_freeze(cohort):
    options, _, algorithm = cohort
    options["split"] = "held_out"
    with pytest.raises(evaluation.EvaluationError, match="EXPLICIT_HELD_OUT_FREEZE"):
        evaluation.evaluate(**options)
    options.update(expected_code_sha=git(options["pipeline_root"], "rev-parse", "HEAD"), expected_algorithm_sha256=evaluation.canonical_sha256(algorithm), expected_cohort_sha256=evaluation.sha256(options["work_root"] / "counting/cohort.v1.local.json"))
    with pytest.raises(evaluation.EvaluationError, match="FROZEN_CODE_SHA_MISMATCH"):
        evaluation.evaluate(**{**options, "expected_code_sha": "0" * 40})
    with pytest.raises(evaluation.EvaluationError, match="FROZEN_ALGORITHM_SHA_MISMATCH"):
        evaluation.evaluate(**{**options, "expected_algorithm_sha256": "0" * 64})
    with pytest.raises(evaluation.EvaluationError, match="FROZEN_COHORT_SHA_MISMATCH"):
        evaluation.evaluate(**{**options, "expected_cohort_sha256": "0" * 64})
    (options["pipeline_root"] / "uncommitted.txt").write_text("Synthetic change")
    with pytest.raises(evaluation.EvaluationError, match="CLEAN_COMMIT"):
        evaluation.evaluate(**options)
    (options["pipeline_root"] / "uncommitted.txt").unlink()
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == 5
    assert (options["output"] / "Case006/attempt.local.json").is_file()
    assert not (options["output"] / "Case001").exists()


def test_algorithm_defaults_cannot_silently_fill_unfrozen_fields(cohort):
    options, _, _ = cohort
    path = options["pipeline_root"] / "src/ethmoid_pipeline/counting/protocol.json"
    value = evaluation.read_json(path)
    del value["algorithm"]["bone_hu"]
    path.write_text(json.dumps(value))
    with pytest.raises(evaluation.EvaluationError, match="COMPLETE_FROZEN_ALGORITHM"):
        evaluation.evaluate(**options)
    assert not options["output"].exists()


@pytest.mark.parametrize("fault,code", [("false_count_failure", "WORKER_RESULT_INVALID"), ("source_mutation", "PROVENANCE_CHANGED"), ("code_mutation", "PROVENANCE_CHANGED"), ("config_mutation", "EFFECTIVE_ALGORITHM_MISMATCH"), ("missing_result", "LOCAL_JSON_INVALID")])
def test_worker_failures_and_drift_never_report_counts(cohort, monkeypatch, fault, code):
    options, _, _ = cohort
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == 0 and summary["failed"] == 5
    assert summary["counts_from_completed_only"] == evaluation.NULL_COUNTS
    attempt = evaluation.read_json(options["output"] / "Case001/attempt.local.json")
    assert attempt["failure_code"] == code and attempt["counts"] == evaluation.NULL_COUNTS


def test_protected_paths_and_source_hash_checks_precede_output_creation(cohort, tmp_path):
    options, manifest, _ = cohort
    with pytest.raises(evaluation.EvaluationError, match="PROTECTED_PATH"):
        evaluation.evaluate(**{**options, "output": tmp_path / "outside"})
    source = options["work_root"] / manifest["entries"][0]["nifti"]
    source.write_bytes(source.read_bytes() + b"modified")
    with pytest.raises(evaluation.EvaluationError, match="COHORT_SOURCE_CHANGED"):
        evaluation.evaluate(**options)
    assert not options["output"].exists()


def test_invalid_and_cross_source_cache_is_rejected(cohort):
    options, manifest, _ = cohort
    entry = manifest["entries"][0]
    source = options["work_root"] / entry["nifti"]
    cache = options["work_root"] / "counting/localizers/Case001"
    (cache / "total").mkdir(parents=True)
    image = nib.load(source)
    mask_path = cache / "total/brain.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(image.shape, dtype=np.uint8), image.affine), mask_path)
    value = {"source_sha256": entry["nifti_sha256"], "tasks": [{"task": "total", "masks": [{"class": "brain", "filename": "brain.nii.gz", "sha256": evaluation.sha256(mask_path)}]}]}
    path = cache / "localizers.local.json"
    path.write_text(json.dumps(value))
    info = evaluation.validate_cache(cache, source, entry["nifti_sha256"], options["work_root"])
    assert info["source_sha256"] == entry["nifti_sha256"] and info["producer_provenance"] == "unverified"
    runtime_info = {"weights_manifest_sha256": "1" * 64, "packages": {"totalsegmentator": "synthetic-test"}}
    value.update(weights_manifest_sha256="1" * 64, totalsegmentator_version="synthetic-test")
    path.write_text(json.dumps(value))
    assert evaluation.validate_cache(cache, source, entry["nifti_sha256"], options["work_root"], runtime_info)["producer_provenance"] == "verified"
    value["source_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(evaluation.EvaluationError, match="LANDMARK_SOURCE"):
        evaluation.validate_cache(cache, source, entry["nifti_sha256"], options["work_root"])
    summary = evaluation.evaluate(**options)
    assert summary["failed"] == 1 and summary["completed"] == 4
    assert not (options["output"] / "Case001/worker.local.log").exists()


@pytest.mark.parametrize("change,code", [("counts", "COUNT_IDENTITY"), ("affine", "GEOMETRY"), ("label", "IDENTITY"), ("laterality", "LATERALITY"), ("centroid", "CENTROID"), ("voxels", "IDENTITY"), ("fractional", "VALUES")])
def test_completed_artifact_consistency_rejects_false_pass(cohort, tmp_path, change, code):
    options, manifest, algorithm = cohort
    source = options["work_root"] / manifest["entries"][0]["nifti"]
    directory = tmp_path / "synthetic-result"
    config = options["pipeline_root"] / "src/ethmoid_pipeline/counting/protocol.json"
    assert evaluation._worker(Path(sys.executable), options["pipeline_root"], source, directory, config, None, tmp_path / "worker.log", 30) == 0
    result = evaluation.read_json(directory / "result.json")
    if change == "counts":
        result["counts"]["total"] = 99
    elif change == "centroid":
        result["cells"][0]["centroid_ras_mm"][0] += 0.2
    elif change == "voxels":
        result["cells"][0]["voxel_count"] = 99
    else:
        path = directory / ("region.nii.gz" if change == "laterality" else "cavities.nii.gz")
        image = nib.load(path)
        data, affine = image.get_fdata(), image.affine.copy()
        if change == "affine":
            affine[0, 3] += 0.1
        elif change == "label":
            data[data > 0] = 2
        elif change == "laterality":
            data[data > 0] = 2
        elif change == "fractional":
            data[0, 0, 0] = 0.5
        nib.save(nib.Nifti1Image(data, affine), path)
    with pytest.raises(evaluation.EvaluationError, match=code):
        evaluation.validate_completed(result, directory, source, algorithm)


def test_timeout_kills_worker_and_returns_fixed_error(cohort, tmp_path):
    options, manifest, _ = cohort
    worker = options["pipeline_root"] / "src/ethmoid_pipeline/counting/worker.py"
    worker.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(evaluation.EvaluationError, match="WORKER_TIMEOUT"):
        evaluation._worker(Path(sys.executable), options["pipeline_root"], options["work_root"] / manifest["entries"][0]["nifti"], tmp_path / "output", worker, None, tmp_path / "timeout.log", 1)


def test_main_prints_aggregate_only(cohort, monkeypatch, capsys):
    options, _, _ = cohort
    summary = {"attempted": 5, "completed": 4, "failed": 1, "accuracy_measured": False}
    monkeypatch.setattr(evaluation, "evaluate", lambda **_: summary)
    assert evaluation.main(["--pipeline-root", str(options["pipeline_root"]), "--output", str(options["output"]), "--split", "development"]) == 2
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("field", ["nifti", "nifti_sha256"])
def test_duplicate_sources_cannot_leak_between_splits(cohort, field):
    options, manifest, _ = cohort
    manifest["entries"][5][field] = manifest["entries"][0][field]
    (options["work_root"] / "counting/cohort.v1.local.json").write_text(json.dumps(manifest))
    with pytest.raises(evaluation.EvaluationError, match="COHORT_DUPLICATE_SOURCE"):
        evaluation.evaluate(**options)
    assert not options["output"].exists()


def test_final_case_cohort_drift_is_not_missed(cohort, monkeypatch):
    options, _, _ = cohort
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "final_cohort_mutation")
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == 4 and summary["failed"] == 1
    assert summary["failure_codes"] == {"PROVENANCE_CHANGED": 1}


def test_runtime_drift_invalidates_worker_results(cohort, monkeypatch):
    options, _, _ = cohort
    calls = []
    def changed_runtime(_):
        calls.append(True)
        return {"weights_manifest_sha256": "0" * 64 if len(calls) == 1 else "1" * 64}
    monkeypatch.setattr(evaluation, "_runtime_provenance", changed_runtime)
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == 0 and summary["counts_from_completed_only"] == evaluation.NULL_COUNTS
    assert summary["failure_codes"] == {"PROVENANCE_CHANGED": 5}


def test_zero_cell_completed_result_is_not_a_valid_count(cohort, tmp_path):
    options, manifest, algorithm = cohort
    source = options["work_root"] / manifest["entries"][0]["nifti"]
    result = {"config": algorithm, "cells": [], "counts": {"left": 0, "right": 0, "total": 0}}
    with pytest.raises(evaluation.EvaluationError, match="EMPTY_COMPLETED"):
        evaluation.validate_completed(result, tmp_path, source, algorithm)


def test_bilateral_region_centers_detect_physical_side_reversal(cohort, tmp_path):
    options, manifest, algorithm = cohort
    source = options["work_root"] / manifest["entries"][0]["nifti"]
    directory = tmp_path / "synthetic-result"
    config = options["pipeline_root"] / "src/ethmoid_pipeline/counting/protocol.json"
    assert evaluation._worker(Path(sys.executable), options["pipeline_root"], source, directory, config, None, tmp_path / "worker.log", 30) == 0
    result = evaluation.read_json(directory / "result.json")
    path = directory / "region.nii.gz"
    image = nib.load(path)
    data = np.asanyarray(image.dataobj).copy()
    # Native X decreases in RAS. Label 2 here is physically LEFT of label 1,
    # despite the cavity agreeing with its own registry's label-1 assertion.
    data[4:, :, :] = 2
    nib.save(nib.Nifti1Image(data, image.affine), path)
    with pytest.raises(evaluation.EvaluationError, match="LATERALITY"):
        evaluation.validate_completed(result, directory, source, algorithm)


def test_structured_coverage_remains_a_geometric_diagnostic():
    assert evaluation.structured_diagnostics({"diagnostics": {"air_coverage_fraction": 0.4, "observed_air_voxels": 10, "assigned_air_voxels": 4}, "warnings": ["SYNTHETIC PRIVATE TEXT"]}) == {"air_coverage_fraction": 0.4, "observed_air_voxels": 10, "assigned_air_voxels": 4}
    assert evaluation.structured_diagnostics({"diagnostics": {"air_coverage_fraction": None, "observed_air_voxels": 0}})["air_coverage_fraction"] is None
    with pytest.raises(evaluation.EvaluationError, match="DIAGNOSTIC_INVALID"):
        evaluation.structured_diagnostics({"diagnostics": {"air_coverage_fraction": 1.1}})


def test_heldout_refuses_cache_without_frozen_producer_provenance(cohort):
    options, manifest, algorithm = cohort
    entry = manifest["entries"][5]
    source = options["work_root"] / entry["nifti"]
    cache = options["work_root"] / "counting/localizers/Case006"
    (cache / "total").mkdir(parents=True)
    image = nib.load(source)
    mask_path = cache / "total/brain.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(image.shape, dtype=np.uint8), image.affine), mask_path)
    (cache / "localizers.local.json").write_text(json.dumps({"source_sha256": entry["nifti_sha256"], "tasks": [{"task": "total", "masks": [{"class": "brain", "filename": "brain.nii.gz", "sha256": evaluation.sha256(mask_path)}]}]}))
    options.update(split="held_out", expected_code_sha=git(options["pipeline_root"], "rev-parse", "HEAD"), expected_algorithm_sha256=evaluation.canonical_sha256(algorithm), expected_cohort_sha256=evaluation.sha256(options["work_root"] / "counting/cohort.v1.local.json"))
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == 4 and summary["failure_codes"] == {"HELD_OUT_CACHE_PRODUCER_UNVERIFIED": 1}
    assert not (options["output"] / "Case006/worker.local.log").exists()
