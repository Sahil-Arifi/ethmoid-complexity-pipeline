"""Independent research audit with tiny synthetic CT and real worker subprocesses."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest

from tests.test_research_evaluator import runnable as research_fixture, evaluation, RUNTIME
from tests.test_evaluate_counting_cohort import git

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_audit", ROOT / "scripts/audit_research_evaluation.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    options = research_fixture.__wrapped__(tmp_path, monkeypatch)
    worker = options["pipeline_root"] / "src/ethmoid_pipeline/counting/worker.py"
    worker.write_text(worker.read_text().replace("fault=os.environ", "result['schema_version']=1\nfault=os.environ"))
    git(options["pipeline_root"], "add", ".")
    git(options["pipeline_root"], "commit", "-q", "-m", "Synthetic audit schema")
    options["expected_code_sha"] = git(options["pipeline_root"], "rev-parse", "HEAD")
    catalog = evaluation.checks.read_json(options["manifest"])
    quality = evaluation.adaptive.quality_helpers(options["pipeline_root"])
    for entry in catalog["entries"]:
        entry["acquisition"] = evaluation.adaptive.normalized_acquisition(entry, quality)
    write(options["manifest"], catalog)
    options["expected_manifest_sha256"] = evaluation.checks.sha256(options["manifest"])
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    summary = evaluation.evaluate(**options)
    assert summary["completed"] == summary["withheld"] == 1
    baseline = options["work_root"] / "baseline-v3"
    shutil.copytree(options["output"], baseline)
    profiles, _ = evaluation.research_helpers(options["pipeline_root"])
    old_attempts = []
    for attempt_path in baseline.glob("*/attempt.local.json"):
        result_path = attempt_path.parent / "worker/result.json"
        result = evaluation.checks.read_json(result_path)
        result.update(protocol_version="ethmoid-count-prototype-v3", method_version="scan-adaptive-native-0.3.0",
            processing_plan=profiles.build_processing_plan(result["scan_profile"]),
            model_identity={"localization": "v2_geometric_baseline", "direct_ethmoid_model": "unavailable_without_promoted_weights"})
        result.pop("research_status")
        attempt = evaluation.checks.read_json(attempt_path)
        attempt.update(method_version=result["method_version"], processing_plan_sha256=result["processing_plan"]["sha256"])
        if result["status"] == "failed":
            result["failure_code"] = attempt["failure_code"] = "SPHENOID_UNRESOLVED"
            attempt["diagnostics"] = evaluation.checks.structured_diagnostics(result)
        write(result_path, result)
        attempt["result_sha256"] = evaluation.checks.sha256(result_path)
        write(attempt_path, attempt)
        old_attempts.append(attempt)
    write(baseline / "protocol.local.json", evaluation.checks.read_json(options["pipeline_root"] / "src/ethmoid_pipeline/counting/protocol.v3.json"))
    write(baseline / "provenance.local.json", {"manifest_sha256": options["expected_manifest_sha256"],
        "accuracy_measured": False, "code": {"git_commit": "b" * 40, "git_dirty": False}, "runtime": deepcopy(RUNTIME)})
    write(baseline / "summary.local.json", evaluation.adaptive.summarize(old_attempts))
    _, binding = evaluation.load_baseline(baseline, catalog["entries"], options["work_root"], options["expected_manifest_sha256"], options["pipeline_root"])
    provenance_path = options["output"] / "provenance.local.json"
    provenance = evaluation.checks.read_json(provenance_path)
    provenance["baseline"] = binding
    write(provenance_path, provenance)
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: deepcopy(RUNTIME))
    return {**{key: value for key, value in options.items() if key != "output"}, "evaluation": options["output"],
            "expected_studies": 2, "expected_series": 2, "expected_ready": 2}


def reseal(options, name):
    directory = options["evaluation"]
    batch_path = directory / "batch-verification.local.json"
    batch = auditor.checks.read_json(batch_path)
    batch["attempt_sha256"][name] = auditor.checks.sha256(directory / name / "attempt.local.json")
    write(batch_path, batch)
    summary_path = directory / "summary.local.json"
    summary = auditor.checks.read_json(summary_path)
    summary["batch_verification_sha256"] = auditor.checks.sha256(batch_path)
    write(summary_path, summary)


def change_result(options, name, transform):
    folder = options["evaluation"] / name
    path = folder / "worker/result.json"
    result = auditor.checks.read_json(path)
    transform(result)
    write(path, result)
    path = folder / "attempt.local.json"
    attempt = auditor.checks.read_json(path)
    attempt["result_sha256"] = auditor.checks.sha256(folder / "worker/result.json")
    write(path, attempt)
    reseal(options, name)


def rejected(options, code):
    with pytest.raises((auditor.checks.EvaluationError, auditor.runner.checks.EvaluationError), match=code):
        auditor.audit(**options)


def test_full_audit_preserves_every_byte_and_emits_aggregate_only(completed):
    before = {path: auditor.checks.sha256(path) for path in completed["work_root"].rglob("*") if path.is_file()}
    report = auditor.audit(**completed)
    assert report["status"] == "verified" and report["accuracy_measured"] is False
    assert report["coverage"] == {"studies": 2, "series": 2, "unique_sources": 2, "unique_attempts": 2, "unavailable": 0}
    assert report["successful_native_results_verified"] == report["baseline_native_results_verified"] == 1
    assert report["v3_comparison"]["totals"]["same_source_pairs"] == 2
    assert report["v3_comparison"]["failure_transitions"]["completed -> completed"] == 1
    aggregates = report["candidate_aggregates"]
    assert aggregates["overall"] == {"attempted": 2, "completed_volumes": 1, "studies_with_completed_volume": 1, "candidate_instances": 1, "uncertain_instances": 1}
    assert aggregates["by_nominal_thickness_mm"]["2.5"]["candidate_instances"] == 1
    encoded = json.dumps(report)
    assert all(value not in encoded for value in ("Case001", "Series001", str(completed["work_root"]), "SYNTHETIC-PROTECTED"))
    assert before == {path: auditor.checks.sha256(path) for path in completed["work_root"].rglob("*") if path.is_file()}


def test_incomplete_returns_before_runtime_or_manifest_read(completed, monkeypatch):
    (completed["evaluation"] / "summary.local.json").unlink()
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: pytest.fail("runtime not allowed"))
    completed["manifest"] = completed["work_root"] / "missing.json"
    report = auditor.audit(**completed)
    assert report["status"] == "incomplete" and report["finished_attempt_records"] == 2
    assert report["accuracy_measured"] is False


@pytest.mark.parametrize("kind", ["source", "result", "log", "mask", "profile_input", "plan_input", "baseline_result", "baseline_log", "attempt_seal"])
def test_unchanged_hash_bindings_are_required(completed, kind):
    folder = completed["evaluation"] / "Case001_Series001"
    names = {"result": "worker/result.json", "log": "worker.local.log", "mask": "worker/cavities.nii.gz",
             "profile_input": "scan-profile.local.json", "plan_input": "processing-plan.local.json", "attempt_seal": "attempt.local.json"}
    if kind == "source":
        path = completed["work_root"] / auditor.checks.read_json(completed["manifest"])["entries"][0]["nifti"]
    elif kind.startswith("baseline_"):
        path = completed["work_root"] / "baseline-v3/Case001_Series001" / names[kind.removeprefix("baseline_")]
    else:
        path = folder / names[kind]
    path.write_bytes(path.read_bytes() + b" ")
    rejected(completed, "AUDIT_ARTIFACT_HASH_CHANGED|BASELINE_RESULT_BINDING_INVALID|AUDIT_ATTEMPT_SEAL_CHANGED")


@pytest.mark.parametrize("field,value,code", [
    ("counts", {"left": 0, "right": 0, "total": 0}, "AUDIT_FAILED_COUNT_EXPOSED"),
    ("artifacts", {"cavities": "cavities.nii.gz"}, "AUDIT_FAILED_COUNT_EXPOSED"),
    ("outcome", "automatic_candidate_count", "AUDIT_FAILURE_OUTCOME_INVALID"),
    ("analysis_mode", "assisted", "AUDIT_MODEL_OR_ASSISTANCE_MISMATCH"),
])
def test_resealed_failures_cannot_expose_counts(completed, field, value, code):
    change_result(completed, "Case002_Series001", lambda result: result.update({field: value}))
    rejected(completed, code)


@pytest.mark.parametrize("kind,code", [("centroid", "NATIVE_ARTIFACT_CENTROID_MISMATCH"),
    ("voxel_count", "NATIVE_ARTIFACT_IDENTITY_MISMATCH"), ("plan", "AUDIT_PLAN_MISMATCH"),
    ("profile", "AUDIT_PROFILE_MISMATCH"), ("escape", "AUDIT_SUCCESS_MANIFEST_INVALID"),
    ("model", "AUDIT_MODEL_OR_ASSISTANCE_MISMATCH")])
def test_resealed_results_still_undergo_independent_native_and_policy_checks(completed, kind, code):
    def mutate(result):
        if kind == "centroid": result["cells"][0]["centroid_ras_mm"][0] += 1
        elif kind == "voxel_count": result["cells"][0]["voxel_count"] += 1
        elif kind == "plan": result["processing_plan"]["counting_grid"] = "resampled"
        elif kind == "profile": result["scan_profile"]["native_spacing_mm"] = [9, 9, 9]
        elif kind == "escape": result["artifacts"]["cavities"] = "../cavities.nii.gz"
        else: result["model_identity"]["direct_ethmoid_model"] = "trained"
    change_result(completed, "Case001_Series001", mutate)
    rejected(completed, code)


def test_native_hu_recomputed_after_air_and_registry_and_all_hashes_are_resealed(completed):
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
    write(path, attempt)
    reseal(completed, "Case001_Series001")
    rejected(completed, "AUDIT_NATIVE_HU_AIR_MISMATCH")


def test_shifted_native_affine_rejected_even_after_resealing(completed):
    path = completed["evaluation"] / "Case001_Series001/worker/cavities.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 2
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), path)
    def mutate(result):
        result["artifact_verification"]["sha256"]["cavities"] = auditor.checks.sha256(path)
    change_result(completed, "Case001_Series001", mutate)
    rejected(completed, "NATIVE_ARTIFACT_GEOMETRY_INVALID")


@pytest.mark.parametrize("fault", [None, "origin", "binding", "mask", "geometry"])
def test_baseline_generated_cache_requires_original_source_and_geometry(tmp_path, fault):
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 7), np.int16), np.eye(4)), source)
    entry = {"study_label": "Case001", "series_label": "Series001", "source_sha256": auditor.checks.sha256(source)}
    previous = tmp_path / "baseline/Case001_Series001"
    cache = previous / "worker/localizers"
    (cache / "craniofacial_structures").mkdir(parents=True)
    path = cache / "craniofacial_structures/skull.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.uint8), np.eye(4)), path)
    manifest = {"source_sha256": entry["source_sha256"], "weights_manifest_sha256": RUNTIME["weights_manifest_sha256"],
        "totalsegmentator_version": RUNTIME["packages"]["totalsegmentator"], "tasks": [{"task": "craniofacial_structures",
        "masks": [{"class": "skull", "filename": "skull.nii.gz", "sha256": auditor.checks.sha256(path)}]}]}
    write(cache / "localizers.local.json", manifest)
    expected = auditor.checks.validate_cache(cache, source, entry["source_sha256"], tmp_path, RUNTIME)
    baseline = {("Case001", "Series001"): {"directory": previous, "attempt": {"generated_localizers": {"localizers": deepcopy(expected)}}}}
    attempt = {"cache_origin": "verified_v3_generated", "localizer_cache": cache.relative_to(tmp_path).as_posix(),
        "cached_localizers": deepcopy(expected), "generated_localizers": {}}
    if fault == "origin": attempt["cache_origin"] = "fresh_inference"
    elif fault == "binding": attempt["cached_localizers"]["source_sha256"] = "b" * 64
    elif fault == "mask": path.write_bytes(path.read_bytes() + b"changed")
    elif fault == "geometry":
        affine = np.eye(4)
        affine[2, 3] = 1
        nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.uint8), affine), path)
        manifest["tasks"][0]["masks"][0]["sha256"] = auditor.checks.sha256(path)
        write(cache / "localizers.local.json", manifest)
        expected.update(manifest_sha256=auditor.checks.sha256(cache / "localizers.local.json"))
        expected["mask_sha256"]["craniofacial_structures/skull.nii.gz"] = auditor.checks.sha256(path)
        attempt["cached_localizers"] = deepcopy(expected)
        baseline[("Case001", "Series001")]["attempt"]["generated_localizers"]["localizers"] = deepcopy(expected)
    args = entry, attempt, tmp_path / "new/Case001_Series001", baseline, source, tmp_path, RUNTIME
    if fault:
        with pytest.raises(auditor.checks.EvaluationError, match="AUDIT_CACHE_ORIGIN_MISMATCH|AUDIT_BASELINE_CACHE_CHANGED|AUDIT_ARTIFACT_HASH_CHANGED|AUDIT_CACHE_GEOMETRY_INVALID"):
            auditor._cache_bindings(*args)
    else:
        auditor._cache_bindings(*args)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_attempt_coverage_is_exact(completed, kind):
    path = completed["evaluation"] / "Case001_Series001/attempt.local.json"
    if kind == "missing": path.unlink()
    else:
        duplicate = completed["evaluation"] / "extra"
        duplicate.mkdir()
        shutil.copyfile(path, duplicate / "attempt.local.json")
    rejected(completed, "AUDIT_ATTEMPT_COVERAGE_INCOMPLETE_OR_DUPLICATE")


def test_duplicate_source_cannot_pass_resealed_catalog(completed):
    path = completed["manifest"]
    catalog = auditor.checks.read_json(path)
    catalog["entries"][1]["source_sha256"] = catalog["entries"][0]["source_sha256"]
    write(path, catalog)
    completed["expected_manifest_sha256"] = auditor.checks.sha256(path)
    path = completed["evaluation"] / "provenance.local.json"
    provenance = auditor.checks.read_json(path)
    provenance["manifest_sha256"] = completed["expected_manifest_sha256"]
    write(path, provenance)
    rejected(completed, "AUDIT_SOURCE_DUPLICATE")


def test_summary_is_recomputed(completed):
    path = completed["evaluation"] / "summary.local.json"
    summary = auditor.checks.read_json(path)
    summary["elapsed_seconds"] += 1
    write(path, summary)
    rejected(completed, "AUDIT_AGGREGATE_SUMMARY_MISMATCH")


def test_runtime_drift_and_incomplete_batch_verification_rejected(completed, monkeypatch):
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: {**RUNTIME, "changed": True})
    rejected(completed, "AUDIT_RUNTIME_CHANGED")
    monkeypatch.setattr(auditor.checks, "_runtime_provenance", lambda _: deepcopy(RUNTIME))
    path = completed["evaluation"] / "batch-verification.local.json"
    batch = auditor.checks.read_json(path)
    batch["status"] = "failed"
    write(path, batch)
    reseal(completed, "Case001_Series001")
    rejected(completed, "AUDIT_BATCH_VERIFICATION_FAILED")


def test_candidate_groups_count_studies_and_reconstruction_range_without_identifiers():
    def record(study, count, thickness, spacing):
        return ({"study_label": study, "status": "completed", "counts": {"total": count},
            "slice_thickness_mm": thickness, "diagnostics": {"uncertain_candidates": 2}}, {"native_spacing_mm": spacing})
    report = auditor._completed_groups([record("protected-a", 8, .5, [.5, .5, .5]),
        record("protected-a", 11, 2.5, [.5, .5, 2.5]), record("protected-b", 5, .5, [.5, .5, .5])])
    assert report["overall"]["candidate_instances"] == 24
    assert report["overall"]["studies_with_completed_volume"] == 2
    assert report["overall"]["uncertain_instances"] == 6
    assert report["by_native_spacing_mm"]["0.5,0.5,0.5"]["candidate_instances"] == 13
    assert report["within_study_reconstruction_counts"]["total_count_range_width_distribution"] == {"3": 1}
    assert "protected" not in json.dumps(report)


def test_cli_writes_only_new_protected_report_and_incomplete_is_distinct(completed, capsys):
    args = [part for key, value in completed.items() for part in ("--" + key.replace("_", "-"), str(value))]
    output = completed["work_root"] / "audit.local.json"
    assert auditor.main(args + ["--output", str(output)]) == 0
    assert auditor.checks.read_json(output)["status"] == "verified"
    assert auditor.main(args + ["--output", str(output)]) == 2
    assert "AUDIT_OUTPUT_ALREADY_EXISTS" in capsys.readouterr().out
    assert auditor.main(args + ["--output", str(completed["evaluation"] / "audit.local.json")]) == 2
    assert "AUDIT_OUTPUT_INSIDE_EVALUATION" in capsys.readouterr().out
    assert auditor.main(args + ["--output", str(completed["work_root"] / "baseline-v3/audit.local.json")]) == 2
    assert "AUDIT_OUTPUT_INSIDE_EVALUATION" in capsys.readouterr().out
    (completed["evaluation"] / "summary.local.json").unlink()
    assert auditor.main(args) == 3
    assert '"status": "incomplete"' in capsys.readouterr().out


def test_cli_invalid_image_reason_does_not_print_protected_path(monkeypatch, capsys):
    def invalid_image(**_):
        raise nib.filebasedimages.ImageFileError("synthetic-protected-patient-path")
    monkeypatch.setattr(auditor, "audit", invalid_image)
    args = [part for name in ("pipeline-root", "manifest", "evaluation", "expected-code-sha", "expected-manifest-sha256", "expected-protocol-sha256")
            for part in ("--" + name, "unused")]
    assert auditor.main(args) == 2
    output = capsys.readouterr().out
    assert "AUDIT_INPUT_UNREADABLE_OR_INVALID" in output
    assert "synthetic-protected-patient-path" not in output
