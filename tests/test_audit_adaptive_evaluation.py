"""Independent audit checks use generated volumes and local synthetic archives."""
import importlib.util
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest

from tests.test_evaluate_adaptive_catalog import runnable as evaluation_fixture, evaluation
from tests.test_evaluate_counting_cohort import git

spec = importlib.util.spec_from_file_location("audit_adaptive_evaluation", Path(__file__).resolve().parents[1] / "scripts/audit_adaptive_evaluation.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    options = evaluation_fixture.__wrapped__(tmp_path, monkeypatch)
    root = options["pipeline_root"]
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "evaluate_adaptive_catalog.py").write_bytes((Path(__file__).resolve().parents[1] / "scripts/evaluate_adaptive_catalog.py").read_bytes())
    worker = root / "src/ethmoid_pipeline/counting/worker.py"
    content = worker.read_text()
    content = content.replace("fault=os.environ", """result['schema_version']=1
result['model_identity']={'localization':'v2_geometric_baseline','direct_ethmoid_model':'unavailable_without_promoted_weights'}
result['artifact_verification']['source_sha256']=result['source_sha256']
fault=os.environ""")
    worker.write_text(content)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Synthetic independent audit fixture")
    manifest = auditor.checks.read_json(options["manifest"])
    manifest.update(studies=2, series=4, accuracy_measured=False, unavailable=[
        {"study_label": "Case001", "series_label": "Series002", "reason_code": "SCOUT"},
        {"study_label": "Case002", "series_label": "Series002", "reason_code": "PROJECTION"}])
    options["manifest"].write_text(json.dumps(manifest))
    options.update(split="all", expected_code_sha=git(root, "rev-parse", "HEAD"), expected_manifest_sha256=auditor.checks.sha256(options["manifest"]))
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    result = evaluation.evaluate(**options)
    assert result["completed"] == result["withheld"] == 1
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: {"synthetic_test_only": True})
    return {"work_root": options["work_root"], "pipeline_root": root, "runtime_python": options["runtime_python"],
        "manifest": options["manifest"], "evaluation": options["output"], "expected_code_sha": options["expected_code_sha"],
        "expected_manifest_sha256": options["expected_manifest_sha256"], "expected_protocol_sha256": options["expected_protocol_sha256"],
        "expected_studies": 2, "expected_series": 4, "expected_ready": 2}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def change_result(options, identity, transform):
    folder = options["evaluation"] / identity
    result_path = folder / "worker/result.json"
    result = auditor.checks.read_json(result_path)
    transform(result)
    write_json(result_path, result)
    attempt_path = folder / "attempt.local.json"
    attempt = auditor.checks.read_json(attempt_path)
    attempt["result_sha256"] = auditor.checks.sha256(result_path)
    write_json(attempt_path, attempt)


def assert_failed(options, code):
    with pytest.raises(auditor.checks.EvaluationError, match=code):
        auditor.audit(**options)


def test_full_audit_verifies_native_success_null_failure_and_independent_summary(completed):
    before = {path: auditor.checks.sha256(path) for path in completed["evaluation"].rglob("*") if path.is_file()}
    report = auditor.audit(**completed)
    assert report["status"] == "verified" and report["accuracy_measured"] is False
    assert report["coverage"] == {"studies": 2, "series": 4, "unique_attempts": 2, "unavailable": 2}
    assert report["successful_native_results_verified"] == 1
    assert report["summary"]["outcomes"] == {"automatic_candidate_count": 1, "needs_review": 1}
    assert report["runtime_by_nominal_thickness_mm"]["2.5"]["attempted"] == 2
    assert report["v2_comparison"] is None
    assert before == {path: auditor.checks.sha256(path) for path in before}


def test_incomplete_evaluation_fails_before_source_or_runtime_reads(completed, monkeypatch):
    (completed["evaluation"] / "summary.local.json").unlink()
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: pytest.fail("Do not probe runtime before completion"))
    assert_failed(completed, "AUDIT_EVALUATION_NOT_FINISHED")


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_attempt_coverage_cannot_omit_or_duplicate_series(completed, kind):
    path = completed["evaluation"] / "Case001_Series001/attempt.local.json"
    if kind == "missing":
        path.unlink()
    else:
        destination = completed["evaluation"] / "extra"
        destination.mkdir()
        shutil.copyfile(path, destination / "attempt.local.json")
    assert_failed(completed, "AUDIT_ATTEMPT_COVERAGE_INCOMPLETE_OR_DUPLICATE")


def test_duplicate_catalog_identity_is_rejected_even_with_resealed_manifest(completed):
    catalog = auditor.checks.read_json(completed["manifest"])
    catalog["unavailable"][0].update(study_label="Case001", series_label="Series001")
    write_json(completed["manifest"], catalog)
    digest = auditor.checks.sha256(completed["manifest"])
    completed["expected_manifest_sha256"] = digest
    path = completed["evaluation"] / "provenance.local.json"
    provenance = auditor.checks.read_json(path)
    provenance["manifest_sha256"] = digest
    write_json(path, provenance)
    assert_failed(completed, "AUDIT_CATALOG_DUPLICATE_IDENTITY")


@pytest.mark.parametrize("kind", ["source", "result", "log", "cavity"])
def test_source_result_log_and_artifacts_are_hash_bound(completed, kind):
    folder = completed["evaluation"] / "Case001_Series001"
    if kind == "source":
        entry = auditor.checks.read_json(completed["manifest"])["entries"][0]
        path = completed["work_root"] / entry["nifti"]
    else:
        path = folder / {"result": "worker/result.json", "log": "worker.local.log", "cavity": "worker/cavities.nii.gz"}[kind]
    path.write_bytes(path.read_bytes() + b"synthetic mutation")
    assert_failed(completed, "AUDIT_ARTIFACT_HASH_CHANGED|AUDIT_RECORDED_VALIDATION_MISMATCH")


@pytest.mark.parametrize("field,value,code", [
    ("counts", {"left": 0, "right": 0, "total": 0}, "AUDIT_FAILED_COUNT_EXPOSED"),
    ("artifacts", {"cavities": "cavities.nii.gz"}, "AUDIT_FAILED_COUNT_EXPOSED"),
    ("outcome", "automatic_candidate_count", "AUDIT_FAILURE_OUTCOME_INVALID"),
    ("analysis_mode", "assisted", "AUDIT_UNEXPECTED_ASSISTANCE"),
])
def test_failed_results_never_expose_numeric_counts_or_wrong_mode(completed, field, value, code):
    change_result(completed, "Case002_Series001", lambda result: result.update({field: value}))
    assert_failed(completed, code)


def test_native_hu_is_recomputed_even_when_corrupt_air_hashes_are_resealed(completed):
    folder = completed["evaluation"] / "Case001_Series001"
    path = folder / "worker/air.nii.gz"
    image = nib.load(path)
    data = np.asarray(image.dataobj).copy()
    data[1, 2, 3] = 0
    nib.save(nib.Nifti1Image(data, image.affine), path)
    digest = auditor.checks.sha256(path)
    def mutate(result):
        result["artifact_verification"]["sha256"]["air"] = digest
        result["cells"][0]["air_voxel_count"] = 7
    change_result(completed, "Case001_Series001", mutate)
    path = folder / "attempt.local.json"
    attempt = auditor.checks.read_json(path)
    attempt["artifacts"]["air"]["sha256"] = digest
    attempt["diagnostics"]["partly_opaque_candidates"] = 1
    write_json(path, attempt)
    assert_failed(completed, "AUDIT_NATIVE_HU_AIR_MISMATCH")


@pytest.mark.parametrize("kind,code", [("centroid", "NATIVE_ARTIFACT_CENTROID_MISMATCH"), ("voxel_count", "NATIVE_ARTIFACT_IDENTITY_MISMATCH")])
def test_native_measurements_are_independently_recomputed(completed, kind, code):
    def mutate(result):
        if kind == "centroid":
            result["cells"][0]["centroid_ras_mm"][0] += 1
        else:
            result["cells"][0]["voxel_count"] += 1
    change_result(completed, "Case001_Series001", mutate)
    assert_failed(completed, code)


def test_recomputed_summary_detects_changed_completion_or_time(completed):
    path = completed["evaluation"] / "summary.local.json"
    summary = auditor.checks.read_json(path)
    summary["elapsed_seconds"] += 1
    write_json(path, summary)
    assert_failed(completed, "AUDIT_AGGREGATE_SUMMARY_MISMATCH")


def test_frozen_runtime_and_code_are_verified(completed, monkeypatch):
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: {"synthetic_test_only": False})
    assert_failed(completed, "AUDIT_RUNTIME_CHANGED")
    completed["expected_code_sha"] = "0" * 40
    assert_failed(completed, "AUDIT_FROZEN_CODE_CHANGED")


def test_rejected_diagnostic_proposals_remain_protected_not_counted(completed):
    worker = completed["evaluation"] / "Case002_Series001/worker"
    write_json(worker / "diagnostic-candidates.local.json", {"clinical_count_available": False, "cells": [{"id": 1}]})
    report = auditor.audit(**completed)
    assert report["failed_attempts_with_protected_proposals"] == 1
    assert report["summary"]["completed"] == 1
    assert report["attempt_bindings"][1]["protected_rejected_proposal_hashes"]


def test_v2_matching_is_repeatability_only_and_requires_same_source(completed):
    archive = completed["work_root"] / "old-v2"
    archive.mkdir()
    write_json(archive / "protocol.local.json", {"protocol_version": "ethmoid-count-prototype-v2"})
    write_json(archive / "provenance.local.json", {"code": {"git_commit": "synthetic"}, "protocol_sha256": "a" * 64})
    for name in ("Case001_Series001", "Case002_Series001"):
        destination = archive / name
        destination.mkdir()
        attempt = auditor.checks.read_json(completed["evaluation"] / name / "attempt.local.json")
        (destination / "worker").mkdir()
        raw = auditor.checks.read_json(completed["evaluation"] / name / "worker/result.json")
        raw['protocol_version'] = 'ethmoid-count-prototype-v2'
        if name.startswith("Case002"):
            attempt["source_sha256"] = "b" * 64
            raw['source_sha256'] = 'b' * 64
        write_json(destination / 'worker/result.json', raw)
        shutil.copyfile(completed['evaluation'] / name / 'worker.local.log', destination / 'worker.local.log')
        attempt['result_sha256'] = auditor.checks.sha256(destination / 'worker/result.json')
        write_json(destination / "attempt.local.json", attempt)
    report = auditor.audit(**completed, v2_evaluations=[archive])
    comparison = report["v2_comparison"]
    assert comparison["totals"]["same_source_pairs"] == 1
    assert comparison["totals"]["source_changed_not_comparable"] == 1
    assert comparison["interpretation"] == "method_repeatability_only_not_ground_truth_or_accuracy"


@pytest.mark.parametrize('mutation', [None, 'hash', 'geometry'])
def test_cached_anatomy_hash_and_geometry_bindings(tmp_path, mutation):
    source = tmp_path / 'source.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 7), np.int16), np.eye(4)), source)
    cache = tmp_path / 'cache'
    (cache / 'craniofacial_structures').mkdir(parents=True)
    mask = cache / 'craniofacial_structures/skull.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.uint8), np.eye(4)), mask)
    runtime = {'weights_manifest_sha256': 'a' * 64, 'packages': {'totalsegmentator': 'synthetic'}}
    manifest = {'source_sha256': auditor.checks.sha256(source), 'weights_manifest_sha256': 'a' * 64,
        'totalsegmentator_version': 'synthetic', 'tasks': [{'task': 'craniofacial_structures',
            'masks': [{'class': 'skull', 'filename': 'skull.nii.gz', 'sha256': auditor.checks.sha256(mask)}]}]}
    write_json(cache / 'localizers.local.json', manifest)
    expected = auditor.checks.validate_cache(cache, source, manifest['source_sha256'], tmp_path, runtime)
    if mutation == 'hash':
        mask.write_bytes(mask.read_bytes() + b'changed')
    elif mutation == 'geometry':
        affine = np.eye(4)
        affine[0, 3] = 1
        nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.uint8), affine), mask)
        digest = auditor.checks.sha256(mask)
        manifest['tasks'][0]['masks'][0]['sha256'] = digest
        write_json(cache / 'localizers.local.json', manifest)
        expected['manifest_sha256'] = auditor.checks.sha256(cache / 'localizers.local.json')
        expected['mask_sha256']['craniofacial_structures/skull.nii.gz'] = digest
    if mutation:
        with pytest.raises(auditor.checks.EvaluationError, match='AUDIT_ARTIFACT_HASH_CHANGED|AUDIT_CACHE_GEOMETRY_INVALID'):
            auditor.audit_cache(cache, expected, source, manifest['source_sha256'], tmp_path, runtime)
    else:
        auditor.audit_cache(cache, expected, source, manifest['source_sha256'], tmp_path, runtime)
