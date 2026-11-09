"""Frozen synthetic software-review pilot; no patient inputs or generated-code execution."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import tokenize
import urllib.error
import urllib.request

MODEL = "gpt-5.4-mini-2026-03-17"
ENDPOINT = "https://api.openai.com/v1/responses"
MAX_OUTPUT = 5000
CAP_USD = 10.0
MAX_REQUESTS = 10
INPUT_RATE = .75 / 1_000_000
OUTPUT_RATE = 4.5 / 1_000_000
MUTATIONS = {
    "coverage": ('"answer_coverage": _ratio(c, n)', '"answer_coverage": _ratio(c, p)'),
    "equal_totals": ('"exact_instance_match": geometry and sides',
                     '"exact_instance_match": len(reference_sides) == len(prediction_sides)'),
    "float_ids": ('set(zip(reference[both].tolist(), prediction[both].tolist()))',
                  'set(map(tuple, np.stack((reference[both], prediction[both]), axis=1).tolist()))'),
}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["defect", "no_change", "insufficient_information"]},
        "diagnosis": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                  "properties": {"before": {"type": "string"}, "after": {"type": "string"}},
                  "required": ["before", "after"]}},
        "falsification_test": {"type": "string"},
    },
    "required": ["status", "diagnosis", "edits", "falsification_test"],
}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def stripped_source(source):
    source = source.split("\ndef synthetic_demo():", 1)[0]
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return tokenize.untokenize(token for token in tokens if token.type != tokenize.COMMENT)


def mutate(source, name):
    before, after = MUTATIONS[name]
    if source.count(before) != 1:
        raise ValueError("Mutation anchor must occur exactly once.")
    return source.replace(before, after, 1)


def request_body(prompt):
    return {"model": MODEL, "input": [{"role": "user", "content": prompt}],
            "store": False, "service_tier": "default", "reasoning": {"effort": "medium"},
            "max_output_tokens": MAX_OUTPUT,
            "text": {"format": {"type": "json_schema", "name": "synthetic_review",
                                "strict": True, "schema": SCHEMA}}}


def reservation(body):
    # A UTF-8 byte is a conservative upper bound for one input text token.
    size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return size * INPUT_RATE + MAX_OUTPUT * OUTPUT_RATE


def can_spend(spent, reserve, count):
    return count < MAX_REQUESTS and spent + reserve <= CAP_USD


def load_key(path):
    if path is None or not path.is_file() or path.is_symlink():
        raise ValueError("Live mode requires an explicit regular env file.")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"\s*OPENAI_API_KEY\s*=\s*(.*?)\s*", line)
        if match:
            key = match.group(1).strip('"\'')
            if key.startswith("sk-") and not any(c.isspace() for c in key):
                return key
    raise ValueError("No usable API key in the approved env file.")


def post(body, key):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, method="POST",
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    # Do not follow redirects with a credential-bearing request.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=180) as response:
        return json.loads(response.read())


def parsed_answer(response):
    if response.get("status") != "completed":
        return None
    parts = [part["text"] for item in response.get("output", [])
             if item.get("type") == "message" for part in item.get("content", [])
             if part.get("type") == "output_text"]
    try:
        answer = json.loads("".join(parts))
    except (ValueError, TypeError):
        return None
    if (not isinstance(answer, dict) or set(answer) != set(SCHEMA["required"])
            or answer["status"] not in SCHEMA["properties"]["status"]["enum"]
            or not isinstance(answer["diagnosis"], str)
            or not isinstance(answer["falsification_test"], str)
            or not isinstance(answer["edits"], list)):
        return None
    if any(not isinstance(e, dict) or set(e) != {"before", "after"}
           or any(not isinstance(v, str) for v in e.values()) for e in answer["edits"]):
        return None
    return answer


def test_fixture(root, source, tests, label):
    fixture = root / label
    (fixture / "scripts").mkdir(parents=True)
    (fixture / "tests").mkdir()
    (fixture / "scripts" / "evaluation_scoring.py").write_text(source, encoding="utf-8")
    (fixture / "tests" / "test_evaluation_scoring.py").write_bytes(tests)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_evaluation_scoring.py", "-q"],
                            cwd=fixture, capture_output=True, text=True, timeout=180)
    (fixture / "pytest.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return {"returncode": result.returncode, "log": label + "/pytest.txt"}


def prepare(source_root, output):
    source_root = source_root.resolve(strict=True)
    if output.resolve().is_relative_to(source_root):
        raise ValueError("Output must be outside the source repository.")
    source_path = source_root / "scripts" / "evaluation_scoring.py"
    test_path = source_root / "tests" / "test_evaluation_scoring.py"
    if source_path.is_symlink() or test_path.is_symlink():
        raise ValueError("Source files must be regular files.")
    original, tests = source_path.read_bytes(), test_path.read_bytes()
    source = original.decode("utf-8-sig")
    output.mkdir(parents=True, exist_ok=False)
    source_variants = {name: mutate(source, name) for name in MUTATIONS}
    checks = {"baseline": test_fixture(output, source, tests, "baseline")}
    for name, variant in source_variants.items():
        checks[name] = test_fixture(output, variant, tests, name)
    write_json(output / "local_checks.json", checks)
    if checks["baseline"]["returncode"] != 0 or any(checks[n]["returncode"] != 1 for n in MUTATIONS):
        raise ValueError("Baseline/mutant checks failed their predefined gate; no API requests permitted.")
    instructions = ("Review this isolated synthetic evaluation helper. No patient data is involved. "
                    "Identify a demonstrated software defect, propose a minimal exact-text edit only if justified, "
                    "or return no_change. Preserve strict validation, unavailable predictions, exact spatial "
                    "recovery, large integer identities, and uncertainty. Tests are withheld. Do not claim "
                    "clinical accuracy. Do not generate an unrelated refactor. Explain a falsification test.\n")
    symptoms = {
        "coverage": "An invented batch has three scheduled left units, three eligible references, two available answers and one exact answer. Reported answer coverage is 1.0.",
        "equal_totals": "Reference labels [1,1,0,2,2,0,3,3] and prediction labels [5,5,0,5,5,0,6,7], both reshaped to (8,1,1), have all-left registries. exact_instance_match is True.",
        "float_ids": "A shared-grid comparison with reference np.uint64([2**63+1,0,2**63+2]) and prediction np.int64([7,0,9]), reshaped to (3,1,1), raises KeyError. Registry keys are exact Python integers.",
    }
    tasks = []
    for number, name in enumerate(MUTATIONS, 1):
        prompt = instructions + "Observed result: " + symptoms[name] + "\nSource:\n" + stripped_source(source_variants[name])
        tasks.append({"id": "task-0" + str(number), "expected": "defect", "request": request_body(prompt)})
    clean = instructions + "No failing assertion is supplied. Inspect this unchanged helper; do not invent a defect solely because review was requested.\nSource:\n" + stripped_source(source)
    tasks.append({"id": "task-04", "expected": "no_change", "request": request_body(clean)})
    ambiguity = ("Synthetic engineering thought experiment only. Two specified physical constructions have "
                 "different true compartment counts: one connected space and two sealed spaces. Under the "
                 "declared acquisition model, they produce exactly the same full observed voxel array. "
                 "No prior probabilities, shape restrictions, additional views, or independent measurements "
                 "are available. Can changing an intensity threshold, interpolation or a deterministic counter "
                 "guarantee the correct count for both? Return the supplied schema. Propose code edits only "
                 "if evidence establishes a software defect. State what would falsify your conclusion.")
    tasks.append({"id": "task-05", "expected": "insufficient_information", "request": request_body(ambiguity)})
    order = [(t["id"], repeat) for repeat in (1, 2) for t in tasks]
    random.Random(20261108).shuffle(order)
    protocol = {"schema_version": 1, "evidence_type": "synthetic_software_review_only",
                "clinical_accuracy": "NOT_MEASURED", "source_sha256": digest(original),
                "tests_sha256": digest(tests), "model": MODEL, "endpoint": ENDPOINT,
                "runner_sha256": digest(Path(__file__).read_bytes()),
                "requests_max": MAX_REQUESTS, "cap_usd": CAP_USD, "order": order,
                "input_usd_per_million": .75, "output_usd_per_million": 4.5,
                "policy": "No generated edits executed; no retries or model fallback; all outcomes retained."}
    write_json(output / "tasks.json", tasks)
    protocol["tasks_sha256"] = digest((output / "tasks.json").read_bytes())
    write_json(output / "protocol.json", protocol)
    return protocol, tasks


def run_api(output, protocol, tasks, key, transport=post):
    lookup = {task["id"]: task for task in tasks}
    ledger = []
    for task_id, repetition in protocol["order"]:
        task = lookup[task_id]
        reserve = reservation(task["request"])
        spent = sum(row["charged_or_reserved_usd"] for row in ledger)
        if not can_spend(spent, reserve, len(ledger)):
            raise ValueError("Local API spending/request cap reached.")
        entry = {"task_id": task_id, "repetition": repetition, "reservation_usd": reserve,
                 "charged_or_reserved_usd": reserve, "state": "pending"}
        ledger.append(entry)
        # Persist the full reservation before networking: a crash cannot erase an in-flight charge.
        accounting = output / "accounting.json"
        accounting.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        started = time.monotonic()
        try:
            response = transport(task["request"], key)
            entry["latency_seconds"] = time.monotonic() - started
            config_matches = response.get("model") == MODEL and response.get("service_tier") == "default"
            answer = parsed_answer(response) if config_matches else None
            usage = response.get("usage", {})
            if (isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int)
                    and min(usage["input_tokens"], usage["output_tokens"]) >= 0):
                entry["charged_or_reserved_usd"] = usage["input_tokens"] * INPUT_RATE + usage["output_tokens"] * OUTPUT_RATE
                entry["usage"] = usage
            entry["state"] = "valid_response" if answer is not None else "invalid_or_incomplete_response"
            if not config_matches:
                entry["state"] = "response_configuration_mismatch"
            safe_response = {field: response.get(field) for field in ("id", "model", "service_tier", "status", "output", "usage", "incomplete_details")}
            write_json(output / f"{task_id}-repeat-{repetition}.json", {"response": safe_response, "parsed": answer})
        except Exception as error:
            entry["state"] = "request_error"
            entry["latency_seconds"] = time.monotonic() - started
            entry["error_type"] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                entry["http_status"] = error.code
            # Error messages/bodies can contain request or credential material; retain metadata only.
            accounting.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError("API request failed; safe metadata retained, no retry performed.") from None
        accounting.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-api", action="store_true")
    args = parser.parse_args(argv)
    try:
        protocol, tasks = prepare(args.source_root, args.output)
        if args.run_api:
            run_api(args.output, protocol, tasks, load_key(args.env_file))
        print("Synthetic pilot complete: " + ("live API responses retained." if args.run_api else "offline preparation only."))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        print("Pilot stopped; inspect safe local artifacts. No automatic retry.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
