"""Real subprocess plumbing with synthetic outputs, never clinical evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_evaluate_catalog import runnable as v2_runnable
from tests.test_evaluate_counting_cohort import git

spec = importlib.util.spec_from_file_location("evaluate_adaptive_catalog", Path(__file__).resolve().parents[1] / "scripts/evaluate_adaptive_catalog.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture
def runnable(tmp_path, monkeypatch):
    options = v2_runnable.__wrapped__(tmp_path, monkeypatch)
    root = options["pipeline_root"]
    package = root / "src/ethmoid_pipeline/counting"
    (package / "profiles.py").write_bytes((Path(__file__).resolve().parents[1] / "src/ethmoid_pipeline/counting/profiles.py").read_bytes())
    protocol = json.loads((package / "protocol.json").read_text())
    protocol["protocol_version"] = "ethmoid-count-prototype-v3"
    (package / "protocol.v3.json").write_text(json.dumps(protocol))
    worker = (package / "worker.py").read_text()
    worker = worker.replace("'landmarks','acquisition'", "'landmarks','acquisition','method','progress'")
    worker = worker.replace("ethmoid-count-prototype-v2", "ethmoid-count-prototype-v3").replace("'synthetic-1'", "'scan-adaptive-native-0.3.0'")
    worker = worker.replace("air=np.zeros_like(cavities)", "air=cavities.copy()").replace("'air_voxel_count':0", "'air_voxel_count':8")
    worker = worker.replace("fault=os.environ", """from ethmoid_pipeline.counting.profiles import build_scan_profile, build_processing_plan, file_sha256
result['scan_profile']=build_scan_profile(source,result['acquisition'])
result['processing_plan']=build_processing_plan(result['scan_profile'])
result['analysis_mode']='automatic'
result['assistance']=None
result['artifact_verification']={'status':'verified','sha256':{name:file_sha256(destination/filename) for name,filename in result['artifacts'].items()}}
fault=os.environ""")
    worker = worker.replace("(destination/'result.json')", """result['outcome']='automatic_candidate_count' if result['status']=='completed' else 'needs_review'
if fault=='profile_mutation': result['scan_profile']['slice_thickness_mm']=0.5
if fault=='plan_mutation': result['processing_plan']['counting_grid']='resampled'
if fault=='verification_missing': result.pop('artifact_verification')
if fault=='artifact_hash': result['artifact_verification']['sha256']['cavities']='0'*64
if fault=='wrong_mode': result['analysis_mode']='assisted'
(destination/'result.json')""")
    (package / "worker.py").write_text(worker)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Synthetic adaptive worker")
    monkeypatch.setattr(evaluation.checks, "_runtime_provenance", lambda _: {"synthetic_test_only": True})
    return {**options, "expected_code_sha": git(root, "rev-parse", "HEAD"),
        "expected_protocol_sha256": evaluation.checks.sha256(package / "protocol.v3.json")}


def test_v3_subprocess_counts_and_abstentions_are_complete_and_immutable(runnable, monkeypatch):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", "one_failure")
    summary = evaluation.evaluate(**runnable)
    assert summary["completed"] == 1 and summary["withheld"] == 1
    assert summary["outcomes"] == {"automatic_candidate_count": 1, "needs_review": 1}
    assert summary["accuracy_measured"] is False
    assert summary["evaluation_scope"] == "existing_development_collection_not_independent_clinical_validation"
    attempt = evaluation.checks.read_json(runnable["output"] / "Case001_Series001/attempt.local.json")
    assert len(attempt["scan_profile_sha256"]) == len(attempt["processing_plan_sha256"]) == 64
    with pytest.raises(evaluation.checks.EvaluationError, match="IMMUTABLE"):
        evaluation.evaluate(**runnable)


@pytest.mark.parametrize("fault,expected", [
    ("profile_mutation", "WORKER_ADAPTATION_MISMATCH"), ("plan_mutation", "WORKER_ADAPTATION_MISMATCH"),
    ("verification_missing", "WORKER_VERIFICATION_MISSING"), ("artifact_hash", "WORKER_ARTIFACT_HASH_CHANGED"),
    ("wrong_mode", "WORKER_ADAPTATION_MISMATCH"),
])
def test_v3_invalid_provenance_cannot_become_a_success(runnable, monkeypatch, fault, expected):
    monkeypatch.setenv("EVALUATION_TEST_FAULT", fault)
    summary = evaluation.evaluate(**runnable)
    assert summary["completed"] == 0 and summary["withheld"] == 2
    assert summary["failure_codes"] == {expected: 2}


def test_v3_requires_exact_protocol_freeze(runnable):
    runnable["expected_protocol_sha256"] = "0" * 64
    with pytest.raises(evaluation.checks.EvaluationError, match="FREEZE_MISMATCH"):
        evaluation.evaluate(**runnable)
    assert not runnable["output"].exists()
