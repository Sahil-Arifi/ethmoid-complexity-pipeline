"""V5 frozen runner and explicit audit-generation checks on synthetic CT only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.test_research_evaluator import runnable as v4_fixture, make_baseline, RUNTIME
from tests.test_evaluate_counting_cohort import git
from tests.test_research_audit import completed as v4_completed, auditor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_v5_evaluation", ROOT / "scripts/evaluate_research_v5_catalog.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def upgrade(options):
    options = dict(options)
    pipeline = options["pipeline_root"]
    package = pipeline / "src/ethmoid_pipeline/counting"
    (package / "research_v5_policy.py").write_bytes((ROOT / "src/ethmoid_pipeline/counting/research_v5_policy.py").read_bytes())
    protocol = json.loads((package / "protocol.research.v4.json").read_text())
    protocol["protocol_version"] = evaluation.PROTOCOL
    (package / "protocol.research.v5.json").write_text(json.dumps(protocol))
    worker = (package / "worker.py").read_text()
    worker = worker.replace("ethmoid-count-research-v4", evaluation.PROTOCOL).replace("research-native-0.4.0", evaluation.METHOD)
    worker = worker.replace("from ethmoid_pipeline.counting.research_policy import", "from ethmoid_pipeline.counting.research_v5_policy import")
    worker = worker.replace("assert a.method=='research-v4'", "assert a.method=='research-v5'")
    (package / "worker.py").write_text(worker)
    (pipeline / "scripts/evaluate_research_v5_catalog.py").write_bytes((ROOT / "scripts/evaluate_research_v5_catalog.py").read_bytes())
    git(pipeline, "add", ".")
    git(pipeline, "commit", "-q", "-m", "Synthetic v5 frozen runner")
    return {**options, "expected_code_sha": git(pipeline, "rev-parse", "HEAD"),
            "expected_protocol_sha256": evaluation.checks.sha256(package / "protocol.research.v5.json")}


@pytest.fixture
def runnable(tmp_path, monkeypatch):
    options = upgrade(v4_fixture.__wrapped__(tmp_path, monkeypatch))
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (2, 2, 2))
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: deepcopy(RUNTIME))
    return options


def test_serial_v5_counts_and_abstentions_bind_versions_and_both_runtime_fingerprints(runnable, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: calls.append(True) or deepcopy(RUNTIME))
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    summary = evaluation.evaluate(**runnable)
    assert summary["attempted"] == 2 and summary["verified_completed"] == summary["withheld"] == 1
    assert len(calls) == 2
    folder = runnable["output"] / "Case001_Series001"
    result = evaluation.checks.read_json(folder / "worker/result.json")
    assert result["method_version"] == evaluation.METHOD == "research-native-0.5.0"
    assert result["protocol_version"] == evaluation.PROTOCOL == "ethmoid-count-research-v5"
    assert result["processing_plan"]["version"] == evaluation.PLAN == "research-localization-plan-2"
    assert result["processing_plan"]["supported_legacy_seed_policy"] == "preserve_exact_existing_voxels"
    failed = evaluation.checks.read_json(runnable["output"] / "Case002_Series001/attempt.local.json")
    assert failed["counts"] == evaluation.checks.NULL_COUNTS and failed["outcome"] == "needs_review"
    provenance = evaluation.checks.read_json(runnable["output"] / "provenance.local.json")
    assert set(provenance["evaluator_dependencies"]) == {"evaluate_research_v5_catalog.py", "evaluate_adaptive_catalog.py", "evaluate_counting_cohort.py"}
    verification = evaluation.checks.read_json(runnable["output"] / "batch-verification.local.json")
    assert verification["runtime_before_sha256"] == verification["runtime_after_sha256"]
    with pytest.raises(evaluation.checks.EvaluationError, match="IMMUTABLE"):
        evaluation.evaluate(**runnable)


@pytest.mark.parametrize("fault,code", [
    ("plan_mutation", "WORKER_ADAPTATION_MISMATCH"), ("false_model", "WORKER_ADAPTATION_MISMATCH"),
    ("zero_failure", "WORKER_FAILURE_INVALID"), ("incorrect_centroid", "NATIVE_ARTIFACT_CENTROID_MISMATCH"),
    ("hu_mismatch", "NATIVE_HU_AIR_MISMATCH"), ("mask_affine", "NATIVE_ARTIFACT_GEOMETRY_INVALID"),
    ("input_plan_mutation", "WORKER_INPUT_SNAPSHOT_CHANGED"), ("source_mutation", "CATALOG_SOURCE_CHANGED"),
])
def test_v5_never_publishes_invalid_native_or_frozen_policy_results(runnable, monkeypatch, fault, code):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    result = evaluation.evaluate(**runnable)
    assert result["verified_completed"] == 0 and result["failure_codes"] == {code: 2}
    for path in runnable["output"].glob("*/attempt.local.json"):
        attempt = evaluation.checks.read_json(path)
        assert attempt["counts"] == evaluation.checks.NULL_COUNTS and attempt["outcome"] == "technical_error"


def test_v5_runtime_change_invalidates_batch(runnable, monkeypatch):
    values = iter([RUNTIME, {**RUNTIME, "changed": True}])
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: next(values))
    result = evaluation.evaluate(**runnable)
    assert result["completed"] == 2 and result["verified_completed"] == 0 and result["verification_status"] == "failed"


def test_v5_selection_is_source_bound_to_v3_baseline(runnable):
    baseline = make_baseline(runnable)
    result = evaluation.evaluate(**runnable, baseline_evaluation=baseline, selection="seed_failures")
    assert result["selected"] == result["verified_completed"] == 1
    assert result["unselected"] == 1
    saved = evaluation.checks.read_json(runnable["output"] / "provenance.local.json")
    assert saved["baseline"]["sha256"] and saved["baseline"]["runtime"] == RUNTIME


def test_v5_reuses_only_verified_v3_generated_cache_without_changing_manifest(runnable):
    baseline = make_baseline(runnable)
    catalog = evaluation.checks.read_json(runnable["manifest"])
    entry = catalog["entries"][0]
    source = runnable["work_root"] / entry["nifti"]
    folder = baseline / "Case001_Series001"
    cache = folder / "worker/localizers"
    (cache / "head").mkdir(parents=True)
    image = nib.load(source)
    mask = cache / "head/eyes_left.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(image.shape, np.uint8), image.affine), mask)
    evaluation.checks.write_new_json(cache / "localizers.local.json", {
        "source_sha256": entry["source_sha256"], "weights_manifest_sha256": RUNTIME["weights_manifest_sha256"],
        "totalsegmentator_version": RUNTIME["packages"]["totalsegmentator"], "tasks": [{"task": "head", "masks": [
            {"filename": mask.name, "sha256": evaluation.checks.sha256(mask)}]}]})
    info = evaluation.checks.validate_cache(cache, source, entry["source_sha256"], runnable["work_root"], RUNTIME)
    attempt = evaluation.checks.read_json(folder / "attempt.local.json")
    attempt["generated_localizers"] = {"localizers": info}
    (folder / "attempt.local.json").write_text(json.dumps(attempt))
    original_hash = evaluation.checks.sha256(runnable["manifest"])
    summary = evaluation.evaluate(**runnable, baseline_evaluation=baseline, selection="seed_failures")
    assert summary["verified_completed"] == 1
    saved = evaluation.checks.read_json(runnable["output"] / "Case001_Series001/attempt.local.json")
    assert saved["cache_origin"] == "verified_v3_generated" and saved["cached_localizers"] == info
    assert evaluation.checks.sha256(runnable["manifest"]) == original_hash
    records, _ = evaluation.load_baseline(baseline, catalog["entries"], runnable["work_root"], original_hash, runnable["pipeline_root"])
    mask.write_bytes(mask.read_bytes() + b"tampered")
    with pytest.raises(evaluation.checks.EvaluationError, match="LANDMARK_HASH_MISMATCH"):
        evaluation.cache_for(entry, records, source, runnable["work_root"], RUNTIME)


def test_v5_missing_catalog_entries_or_wrong_plan_version_cannot_launch(runnable, monkeypatch):
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (26, 230, 98))
    with pytest.raises(evaluation.checks.EvaluationError, match="CATALOG_COVERAGE_MISMATCH"):
        evaluation.evaluate(**runnable)
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (2, 2, 2))
    path = runnable["pipeline_root"] / "src/ethmoid_pipeline/counting/research_v5_policy.py"
    path.write_text(path.read_text().replace("research-localization-plan-2", "research-localization-plan-1"))
    git(runnable["pipeline_root"], "add", ".")
    git(runnable["pipeline_root"], "commit", "-q", "-m", "Wrong synthetic plan version")
    runnable["expected_code_sha"] = git(runnable["pipeline_root"], "rev-parse", "HEAD")
    with pytest.raises(evaluation.checks.EvaluationError, match="RESEARCH_POLICY_VERSION_MISMATCH"):
        evaluation.evaluate(**runnable)
    assert not runnable["output"].exists()


@pytest.mark.parametrize("field", ["expected_code_sha", "expected_manifest_sha256", "expected_protocol_sha256"])
def test_v5_requires_exact_freeze(runnable, field):
    runnable[field] = "0" * 64
    with pytest.raises(evaluation.checks.EvaluationError, match="FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)
    assert not runnable["output"].exists()


def test_v5_rejects_mutated_evaluator_even_when_git_commit_is_clean(runnable):
    path = runnable["pipeline_root"] / "scripts/evaluate_research_v5_catalog.py"
    path.write_text(path.read_text() + "\n# a different frozen evaluator\n")
    git(runnable["pipeline_root"], "add", ".")
    git(runnable["pipeline_root"], "commit", "-q", "-m", "Different synthetic evaluator")
    runnable["expected_code_sha"] = git(runnable["pipeline_root"], "rev-parse", "HEAD")
    with pytest.raises(evaluation.checks.EvaluationError, match="EVALUATOR_FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)


def test_v5_audit_is_explicit_and_default_v4_does_not_certify_it(tmp_path, monkeypatch):
    original = v4_completed.__wrapped__(tmp_path, monkeypatch)
    options = upgrade({key: value for key, value in original.items() if key not in {"evaluation", "expected_studies", "expected_series", "expected_ready"}})
    options["output"] = options["work_root"] / "research-v5"
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (2, 2, 2))
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: deepcopy(RUNTIME))
    summary = evaluation.evaluate(**options, baseline_evaluation=options["work_root"] / "baseline-v3")
    assert summary["verified_completed"] == 1
    audit_options = {**{key: value for key, value in options.items() if key != "output"},
        "evaluation": options["output"], "expected_studies": 2, "expected_series": 2, "expected_ready": 2}
    with pytest.raises(auditor.checks.EvaluationError, match="AUDIT_ARTIFACT_HASH_CHANGED"):
        auditor.audit(**audit_options)
    report = auditor.audit(**audit_options, generation=5)
    assert report["status"] == "verified" and report["research_generation"] == 5
    assert report["successful_native_results_verified"] == report["baseline_native_results_verified"] == 1
    assert report["v3_comparison"]["totals"]["same_source_pairs"] == 2


def test_unsupported_audit_generation_cannot_read_sources():
    with pytest.raises(auditor.checks.EvaluationError, match="AUDIT_RESEARCH_GENERATION_UNSUPPORTED"):
        auditor.audit(work_root=Path("absent"), pipeline_root=Path("absent"), runtime_python=Path("absent"),
            manifest=Path("absent"), evaluation=Path("absent"), expected_code_sha="0" * 40,
            expected_manifest_sha256="0" * 64, expected_protocol_sha256="0" * 64, generation=6)
