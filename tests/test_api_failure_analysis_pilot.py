"""Offline safety and seeded-regression checks; never contacts the API."""
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest

SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "scripts/api_failure_analysis_pilot.py"
spec = importlib.util.spec_from_file_location("pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_exact_mutations_preserve_original_and_reject_anchor_drift():
    source = (SOURCE / "scripts/evaluation_scoring.py").read_text(encoding="utf-8")
    for name, (before, after) in pilot.MUTATIONS.items():
        mutated = pilot.mutate(source, name)
        assert before not in mutated
        assert mutated.count(after) == 1
        assert mutated.replace(after, before, 1) == source
        with pytest.raises(ValueError):
            pilot.mutate(mutated, name)


def test_byte_reservation_and_hard_caps_include_reasoning_maximum():
    body = pilot.request_body("Ã©" * 50)
    assert pilot.reservation(body) > pilot.MAX_OUTPUT * pilot.OUTPUT_RATE
    assert pilot.can_spend(9, .5, 9)
    assert not pilot.can_spend(9.99, .02, 9)
    assert not pilot.can_spend(0, .01, 10)
    assert body["store"] is False and "tools" not in body
    assert body["model"] == "gpt-5.4-mini-2026-03-17"


def test_frozen_offline_preparation_runs_real_baseline_and_mutant_tests(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline preparation attempted credential access or network.")
    monkeypatch.setattr(pilot, "load_key", forbidden)
    monkeypatch.setattr(pilot, "post", forbidden)
    output = tmp_path / "offline"
    assert pilot.main(["--source-root", str(SOURCE), "--output", str(output)]) == 0
    checks = json.loads((output / "local_checks.json").read_text())
    assert checks["baseline"]["returncode"] == 0
    assert all(checks[name]["returncode"] == 1 for name in pilot.MUTATIONS)
    tests_bytes = (SOURCE / "tests/test_evaluation_scoring.py").read_bytes()
    for name in ("baseline", *pilot.MUTATIONS):
        assert (output / name / "tests/test_evaluation_scoring.py").read_bytes() == tests_bytes
        assert (output / name / "pytest.txt").is_file()
    tasks = json.loads((output / "tasks.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert len(tasks) == 5 and len(protocol["order"]) == 10
    assert protocol["runner_sha256"] == pilot.digest(SCRIPT.read_bytes())
    assert protocol["tasks_sha256"] == pilot.digest((output / "tasks.json").read_bytes())
    assert len({tuple(entry) for entry in protocol["order"]}) == 10
    assert not (output / "accounting.json").exists()
    for task in tasks:
        prompt = task["request"]["input"][0]["content"]
        assert "def synthetic_demo" not in prompt
        assert "# Python integer pairs" not in prompt
        assert "def test_" not in prompt
    before = (output / "protocol.json").read_bytes()
    assert pilot.main(["--source-root", str(SOURCE), "--output", str(output)]) == 1
    assert (output / "protocol.json").read_bytes() == before


def test_error_metadata_redacts_key_and_body_and_has_no_retry(tmp_path):
    fake_key = "sk-fake-test-key"
    task = {"id": "task-01", "request": pilot.request_body("synthetic only")}
    protocol = {"order": [["task-01", 1], ["task-01", 2]]}
    calls = []
    def failing(body, key):
        calls.append(key)
        raise urllib.error.HTTPError(pilot.ENDPOINT, 403, fake_key, {}, None)
    with pytest.raises(RuntimeError, match="no retry") as caught:
        pilot.run_api(tmp_path, protocol, [task], fake_key, transport=failing)
    assert calls == [fake_key]
    saved = (tmp_path / "accounting.json").read_text()
    assert fake_key not in saved and fake_key not in str(caught.value)
    ledger = json.loads(saved)
    assert ledger[0]["http_status"] == 403
    assert ledger[0]["charged_or_reserved_usd"] == ledger[0]["reservation_usd"]


def test_no_request_after_cap_and_invalid_answers_remain_outcomes(tmp_path):
    tasks = [{"id": "task-01", "request": pilot.request_body("synthetic")}]
    calls = []
    def incomplete(body, key):
        calls.append(1)
        return {"id": "invented", "model": pilot.MODEL, "service_tier": "default", "status": "incomplete", "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 10}}
    order = [["task-01", i] for i in range(11)]
    with pytest.raises(ValueError, match="cap"):
        pilot.run_api(tmp_path, {"order": order}, tasks, "unused", transport=incomplete)
    ledger = json.loads((tmp_path / "accounting.json").read_text())
    assert len(calls) == len(ledger) == 10
    assert all(row["state"] == "invalid_or_incomplete_response" for row in ledger)


def test_model_or_service_tier_mismatch_is_retained_without_fallback(tmp_path):
    tasks = [{"id": "task-01", "request": pilot.request_body("synthetic")}]
    calls = []
    def mismatched(body, key):
        calls.append(1)
        return {"model": "different-model", "service_tier": "priority", "status": "completed",
                "output": [], "usage": {"input_tokens": 10, "output_tokens": 10}}
    ledger = pilot.run_api(tmp_path, {"order": [["task-01", 1]]}, tasks, "unused", transport=mismatched)
    assert len(calls) == 1
    assert ledger[0]["state"] == "response_configuration_mismatch"
    saved = json.loads((tmp_path / "task-01-repeat-1.json").read_text())
    assert saved["parsed"] is None
    assert saved["response"]["model"] == "different-model"


def test_parser_and_source_comment_stripping():
    answer = {"status": "no_change", "diagnosis": "No demonstrated defect.",
              "edits": [], "falsification_test": "Run invariants."}
    response = {"status": "completed", "output": [{"type": "message", "content":
                [{"type": "output_text", "text": json.dumps(answer)}]}]}
    assert pilot.parsed_answer(response) == answer
    answer["edits"] = [{"before": 1, "after": "x"}]
    response["output"][0]["content"][0]["text"] = json.dumps(answer)
    assert pilot.parsed_answer(response) is None
    assert "private comment" not in pilot.stripped_source("a = '#literal' # private comment\n")
    assert "#literal" in pilot.stripped_source("a = '#literal' # private comment\n")
