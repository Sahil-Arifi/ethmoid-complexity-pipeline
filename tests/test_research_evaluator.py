"""Frozen research runner acceptance with synthetic CT and real subprocesses."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.test_evaluate_adaptive_catalog import runnable as v3_runnable
from tests.test_evaluate_counting_cohort import git

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_evaluation", ROOT / "scripts/evaluate_research_catalog.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
RUNTIME = {"synthetic_test_only": True, "weights_manifest_sha256": "a" * 64,
           "packages": {"totalsegmentator": "synthetic-1"}}


@pytest.fixture
def runnable(tmp_path, monkeypatch):
    options = v3_runnable.__wrapped__(tmp_path, monkeypatch)
    options.pop("split")
    pipeline = options["pipeline_root"]
    package = pipeline / "src/ethmoid_pipeline/counting"
    (package / "research_policy.py").write_bytes((ROOT / "src/ethmoid_pipeline/counting/research_policy.py").read_bytes())
    protocol = json.loads((package / "protocol.v3.json").read_text())
    protocol["protocol_version"] = evaluation.PROTOCOL
    (package / "protocol.research.v4.json").write_text(json.dumps(protocol))
    worker = (package / "worker.py").read_text()
    worker = worker.replace("'method','progress'", "'method','progress','processing-plan'")
    worker = worker.replace("ethmoid-count-prototype-v3", evaluation.PROTOCOL).replace("scan-adaptive-native-0.3.0", evaluation.METHOD)
    worker = worker.replace("result['processing_plan']=build_processing_plan(result['scan_profile'])", "from ethmoid_pipeline.counting.research_policy import build_research_plan\nresult['processing_plan']=build_research_plan(result['scan_profile'])\nassert json.loads(Path(a.processing_plan).read_text())==result['processing_plan']\nassert a.method=='research-v4'")
    worker = worker.replace("'status':'verified','sha256'", "'status':'verified','source_sha256':result['source_sha256'],'sha256'")
    worker = worker.replace("fault=os.environ", "result['model_identity']={'localization':'research_geometric_baseline','direct_ethmoid_model':'not_used'}\nresult['research_status']='NOT_ACCURACY_VALIDATED'\nfault=os.environ")
    worker = worker.replace("(destination/'result.json')", """if fault=='zero_failure': result.update(status='failed',counts={'left':0,'right':0,'total':0},cells=[],artifacts={},failure_code='TARGET_CLIPPED',outcome='needs_review')
if fault=='incorrect_centroid': result['cells'][0]['centroid_ras_mm']=[0,0,0]
if fault=='false_model': result['model_identity']['direct_ethmoid_model']='trained'
if fault=='hu_mismatch':
    nib.save(nib.Nifti1Image(np.zeros(image.shape,np.uint16),image.affine),destination/'air.nii.gz')
    result['cells'][0]['air_voxel_count']=0
    result['artifact_verification']['sha256']['air']=file_sha256(destination/'air.nii.gz')
if fault=='mask_affine':
    affine=image.affine.copy(); affine[0,3]+=3
    nib.save(nib.Nifti1Image(cavities,affine),destination/'cavities.nii.gz')
    result['artifact_verification']['sha256']['cavities']=file_sha256(destination/'cavities.nii.gz')
if fault=='artifact_escape': result['artifacts']['cavities']='../cavities.nii.gz'
if fault=='zero_success': result.update(counts={'left':0,'right':0,'total':0},cells=[])
if fault=='input_plan_mutation': Path(a.processing_plan).write_text('{}')
(destination/'result.json')""")
    (package / "worker.py").write_text(worker)
    (pipeline / "scripts").mkdir()
    for name in ("evaluate_research_catalog.py", "evaluate_adaptive_catalog.py", "evaluate_counting_cohort.py"):
        (pipeline / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    catalog = json.loads(options["manifest"].read_text())
    catalog.update(studies=2, series=2, unavailable=[], accuracy_measured=False)
    options["manifest"].write_text(json.dumps(catalog))
    git(pipeline, "add", ".")
    git(pipeline, "commit", "-q", "-m", "Synthetic research runner fixture")
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (2, 2, 2))
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: deepcopy(RUNTIME))
    return {**options, "expected_code_sha": git(pipeline, "rev-parse", "HEAD"),
        "expected_manifest_sha256": evaluation.checks.sha256(options["manifest"]),
        "expected_protocol_sha256": evaluation.checks.sha256(package / "protocol.research.v4.json")}


def make_baseline(options):
    baseline = options["work_root"] / "baseline-v3"
    baseline.mkdir()
    catalog = evaluation.checks.read_json(options["manifest"])
    protocol = evaluation.checks.read_json(options["pipeline_root"] / "src/ethmoid_pipeline/counting/protocol.v3.json")
    profiles, _ = evaluation.research_helpers(options["pipeline_root"])
    quality = evaluation.adaptive.quality_helpers(options["pipeline_root"])
    attempts = []
    for number, entry in enumerate(catalog["entries"]):
        directory = baseline / "_".join(evaluation.identity(entry))
        (directory / "worker").mkdir(parents=True)
        acquisition = evaluation.adaptive.normalized_acquisition(entry, quality)
        # Normalize the same frozen catalog snapshots as the real manifest creator.
        entry["acquisition"] = acquisition
        source = options["work_root"] / entry["nifti"]
        profile = profiles.build_scan_profile(source, acquisition)
        plan = profiles.build_processing_plan(profile)
        result = dict(source_sha256=entry["source_sha256"], protocol_version=protocol["protocol_version"], config=protocol["algorithm"],
            method_version="scan-adaptive-native-0.3.0", analysis_mode="automatic", assistance=None,
            acquisition=acquisition, scan_profile=profile, processing_plan=plan)
        if number == 0:
            result.update(status="failed", counts=dict(evaluation.checks.NULL_COUNTS), cells=[], artifacts={},
                failure_code="SPHENOID_UNRESOLVED", message=evaluation.SEED_MESSAGE, outcome="needs_review")
        else:
            result.update(status="completed", counts={"left": 1, "right": 0, "total": 1},
                outcome="automatic_candidate_count", artifact_verification={"status": "verified"})
        evaluation.checks.write_new_json(directory / "worker/result.json", result)
        (directory / "worker.local.log").write_text("Synthetic baseline log")
        attempt = dict(study_label=entry["study_label"], series_label=entry["series_label"], source_sha256=entry["source_sha256"],
            acquisition=acquisition, status=result["status"], counts=result["counts"], outcome=result["outcome"],
            scan_profile_sha256=profile["sha256"], processing_plan_sha256=plan["sha256"], slice_thickness_mm=2.5,
            elapsed_seconds=1, result_sha256=evaluation.checks.sha256(directory / "worker/result.json"),
            worker_log_sha256=evaluation.checks.sha256(directory / "worker.local.log"), generated_localizers={})
        if "failure_code" in result:
            attempt["failure_code"] = result["failure_code"]
        evaluation.checks.write_new_json(directory / "attempt.local.json", attempt)
        attempts.append(attempt)
    options["manifest"].write_text(json.dumps(catalog))
    options["expected_manifest_sha256"] = evaluation.checks.sha256(options["manifest"])
    evaluation.checks.write_new_json(baseline / "protocol.local.json", protocol)
    evaluation.checks.write_new_json(baseline / "provenance.local.json", {
        "manifest_sha256": options["expected_manifest_sha256"], "accuracy_measured": False,
        "code": {"git_commit": "b" * 40, "git_dirty": False}, "runtime": deepcopy(RUNTIME)})
    evaluation.checks.write_new_json(baseline / "summary.local.json", evaluation.adaptive.summarize(attempts))
    return baseline


def test_real_subprocesses_preserve_failures_bind_plan_and_immutable_history(runnable, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: calls.append(True) or deepcopy(RUNTIME))
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    summary = evaluation.evaluate(**runnable)
    assert summary["attempted"] == 2 and summary["verified_completed"] == 1 and summary["withheld"] == 1
    assert summary["verification_status"] == "verified"
    assert summary["outcomes"] == {"automatic_candidate_count": 1, "needs_review": 1}
    assert summary["accuracy_measured"] is False and summary["catalog_ready"] == 2
    assert len(calls) == 2
    success = evaluation.checks.read_json(runnable["output"] / "Case001_Series001/attempt.local.json")
    assert success["native_geometry"] == success["count_identity"] == success["source_hu_air_consistency"] == "PASS"
    assert success["cache_origin"] == "fresh_inference"
    failed = evaluation.checks.read_json(runnable["output"] / "Case002_Series001/attempt.local.json")
    assert failed["counts"] == evaluation.checks.NULL_COUNTS
    verification = evaluation.checks.read_json(runnable["output"] / "batch-verification.local.json")
    assert len(verification["attempt_sha256"]) == 2
    assert verification["runtime_before_sha256"] == verification["runtime_after_sha256"]
    assert "SYNTHETIC-PROTECTED-WORKER-TEXT" not in capsys.readouterr().out
    with pytest.raises(evaluation.checks.EvaluationError, match="IMMUTABLE"):
        evaluation.evaluate(**runnable)


@pytest.mark.parametrize("fault,code", [
    ("profile_mutation", "WORKER_ADAPTATION_MISMATCH"), ("plan_mutation", "WORKER_ADAPTATION_MISMATCH"),
    ("wrong_mode", "WORKER_ADAPTATION_MISMATCH"), ("false_model", "WORKER_ADAPTATION_MISMATCH"),
    ("zero_failure", "WORKER_FAILURE_INVALID"), ("false_count_failure", "WORKER_FAILURE_INVALID"),
    ("verification_missing", "WORKER_VERIFICATION_MISSING"), ("artifact_hash", "WORKER_ARTIFACT_HASH_CHANGED"),
    ("incorrect_centroid", "NATIVE_ARTIFACT_CENTROID_MISMATCH"), ("hu_mismatch", "NATIVE_HU_AIR_MISMATCH"),
    ("mask_affine", "NATIVE_ARTIFACT_GEOMETRY_INVALID"), ("artifact_escape", "WORKER_SUCCESS_INVALID"),
    ("zero_success", "EMPTY_COMPLETED_COUNT_UNSUPPORTED"), ("config_mutation", "WORKER_PROVENANCE_MISMATCH"),
    ("input_plan_mutation", "WORKER_INPUT_SNAPSHOT_CHANGED"),
])
def test_invalid_results_never_publish_candidate_counts(runnable, monkeypatch, fault, code):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    summary = evaluation.evaluate(**runnable)
    assert summary["completed"] == summary["verified_completed"] == 0
    assert summary["failure_codes"] == {code: 2}
    for attempt in runnable["output"].glob("*/attempt.local.json"):
        value = evaluation.checks.read_json(attempt)
        assert value["counts"] == evaluation.checks.NULL_COUNTS and value["outcome"] == "technical_error"


def test_runtime_drift_invalidates_batch_without_rewriting_attempts(runnable, monkeypatch):
    values = iter([RUNTIME, {**RUNTIME, "changed": True}])
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: next(values))
    summary = evaluation.evaluate(**runnable)
    assert summary["completed"] == 2 and summary["verified_completed"] == 0
    assert summary["verification_status"] == "failed"
    assert evaluation.checks.read_json(runnable["output"] / "batch-verification.local.json")["failure_code"] == "RUNTIME_CHANGED"


@pytest.mark.parametrize("field", ["expected_code_sha", "expected_manifest_sha256", "expected_protocol_sha256"])
def test_freeze_mismatch_cannot_launch(runnable, field):
    runnable[field] = "0" * 64
    with pytest.raises(evaluation.checks.EvaluationError, match="FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)
    assert not runnable["output"].exists()


def test_incomplete_catalog_and_uncommitted_script_are_rejected(runnable, monkeypatch):
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (26, 230, 98))
    with pytest.raises(evaluation.checks.EvaluationError, match="CATALOG_COVERAGE_MISMATCH"):
        evaluation.evaluate(**runnable)
    monkeypatch.setattr(evaluation, "EXPECTED_CATALOG", (2, 2, 2))
    path = runnable["pipeline_root"] / "scripts/evaluate_research_catalog.py"
    path.write_text("# not the frozen runner")
    git(runnable["pipeline_root"], "add", ".")
    git(runnable["pipeline_root"], "commit", "-q", "-m", "Different frozen runner")
    runnable["expected_code_sha"] = git(runnable["pipeline_root"], "rev-parse", "HEAD")
    with pytest.raises(evaluation.checks.EvaluationError, match="EVALUATOR_FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)


@pytest.mark.parametrize("selection,index", [("seed_failures", 0), ("baseline_completed", 1)])
def test_diagnostic_selection_is_bound_to_verified_v3_results(runnable, selection, index):
    baseline = make_baseline(runnable)
    summary = evaluation.evaluate(**runnable, baseline_evaluation=baseline, selection=selection)
    assert summary["selected"] == summary["verified_completed"] == 1 and summary["unselected"] == 1
    assert (runnable["output"] / f"Case{index + 1:03}_Series001/attempt.local.json").is_file()
    provenance = evaluation.checks.read_json(runnable["output"] / "provenance.local.json")
    assert provenance["baseline"]["sha256"]


def test_subsets_need_baseline_and_seed_reason_is_exact(runnable):
    with pytest.raises(evaluation.checks.EvaluationError, match="BASELINE_REQUIRED"):
        evaluation.evaluate(**runnable, selection="seed_failures")
    entries = [{"study_label": "Case001", "series_label": "Series001"}]
    baseline = {("Case001", "Series001"): {"result": {"failure_code": "SPHENOID_UNRESOLVED", "message": "Different unresolved anatomy"}}}
    with pytest.raises(evaluation.checks.EvaluationError, match="EMPTY_EVALUATION_SELECTION"):
        evaluation.select_entries(entries, baseline, "seed_failures")
    baseline[("Case001", "Series001")]["result"]["failure_code"] = "RESOURCE_LIMIT"
    assert evaluation.select_entries(entries, baseline, "resource_limits") == entries


@pytest.mark.parametrize("fault", ["result", "log", "source", "missing_attempt", "summary", "profile"])
def test_tampered_baseline_cannot_select_or_supply_cache(runnable, fault):
    baseline = make_baseline(runnable)
    folder = baseline / "Case001_Series001"
    if fault == "result":
        (folder / "worker/result.json").write_text("{}")
    elif fault == "log":
        (folder / "worker.local.log").write_text("changed")
    elif fault == "source":
        path = folder / "worker/result.json"
        result = evaluation.checks.read_json(path)
        result["source_sha256"] = "0" * 64
        path.write_text(json.dumps(result))
        attempt = evaluation.checks.read_json(folder / "attempt.local.json")
        attempt["result_sha256"] = evaluation.checks.sha256(path)
        (folder / "attempt.local.json").write_text(json.dumps(attempt))
    elif fault == "profile":
        path = folder / "worker/result.json"
        result = evaluation.checks.read_json(path)
        result["scan_profile"]["native_spacing_mm"] = [9, 9, 9]
        path.write_text(json.dumps(result))
        attempt = evaluation.checks.read_json(folder / "attempt.local.json")
        attempt["result_sha256"] = evaluation.checks.sha256(path)
        (folder / "attempt.local.json").write_text(json.dumps(attempt))
    elif fault == "summary":
        (baseline / "summary.local.json").write_text("{}")
    else:
        (folder / "attempt.local.json").unlink()
    with pytest.raises(evaluation.checks.EvaluationError, match="BASELINE"):
        evaluation.evaluate(**runnable, baseline_evaluation=baseline, selection="seed_failures")
    assert not runnable["output"].exists()


def test_verified_v3_generated_cache_is_reused_without_manifest_rewrite(runnable):
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
        "totalsegmentator_version": RUNTIME["packages"]["totalsegmentator"],
        "tasks": [{"task": "head", "masks": [{"filename": mask.name, "sha256": evaluation.checks.sha256(mask)}]}]})
    info = evaluation.checks.validate_cache(cache, source, entry["source_sha256"], runnable["work_root"], RUNTIME)
    attempt = evaluation.checks.read_json(folder / "attempt.local.json")
    attempt["generated_localizers"] = {"localizers": info}
    (folder / "attempt.local.json").write_text(json.dumps(attempt))
    manifest_before = evaluation.checks.sha256(runnable["manifest"])
    summary = evaluation.evaluate(**runnable, baseline_evaluation=baseline, selection="seed_failures")
    assert summary["verified_completed"] == 1
    saved = evaluation.checks.read_json(runnable["output"] / "Case001_Series001/attempt.local.json")
    assert saved["cache_origin"] == "verified_v3_generated" and saved["cached_localizers"] == info
    assert evaluation.checks.sha256(runnable["manifest"]) == manifest_before
    records, _ = evaluation.load_baseline(baseline, catalog["entries"], runnable["work_root"], manifest_before, runnable["pipeline_root"])
    mask.write_bytes(mask.read_bytes() + b"tampered")
    with pytest.raises(evaluation.checks.EvaluationError, match="LANDMARK_HASH_MISMATCH"):
        evaluation.cache_for(entry, records, source, runnable["work_root"], RUNTIME)


def test_timeout_is_retained_and_does_not_stop_later_attempts(runnable, monkeypatch):
    calls = []
    def unavailable_worker(*args):
        calls.append(True)
        raise evaluation.checks.EvaluationError("WORKER_TIMEOUT")
    monkeypatch.setattr(evaluation, "worker", unavailable_worker)
    summary = evaluation.evaluate(**runnable)
    assert len(calls) == 2 and summary["failure_codes"] == {"WORKER_TIMEOUT": 2}
    assert summary["verified_completed"] == 0 and summary["outcomes"] == {"technical_error": 2}


def test_source_change_cannot_produce_count(runnable, monkeypatch):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "source_mutation")
    summary = evaluation.evaluate(**runnable)
    assert summary["completed"] == 0 and summary["failure_codes"] == {"CATALOG_SOURCE_CHANGED": 2}


def test_worker_timeout_terminates_its_process_group(tmp_path, monkeypatch):
    import subprocess
    class Process:
        pid = 987654
        waits = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -1
    stopped = []
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(evaluation.subprocess, "run", lambda command, **k: stopped.append(command))
    monkeypatch.setattr(evaluation.os, "killpg", lambda *a: stopped.append(a), raising=False)
    with pytest.raises(evaluation.checks.EvaluationError, match="WORKER_TIMEOUT"):
        evaluation.worker(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, None, tmp_path / "log", 1)
    assert stopped
