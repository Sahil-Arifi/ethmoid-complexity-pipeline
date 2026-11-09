# API failure-analysis pilot

This pilot measures whether API assistance can diagnose and repair deliberately seeded defects in the existing synthetic evaluation helper. It does not change the anatomical counter or measure patient accuracy. The defects and correct source are known to the experiment designers; results are a symptom-guided seeded debugging benchmark.

## Frozen design

- Source baseline: `ee7c251`; only `scripts/evaluation_scoring.py` is sent as code, with comments and demo fixtures removed. Tests stay local.
- Three separate defects: answer-coverage denominator, equal totals incorrectly accepted as exact spatial recovery, and floating-point promotion of mixed signed/unsigned label identities.
- Two controls: unchanged scorer source and a thought experiment that explicitly supplies identical observations for incompatible physical counts.
- Five tasks, each requested twice in an order shuffled with seed 20261108. No quality-selected retries or model fallback. Failed, malformed and incomplete answers remain outcomes.
- Model snapshot: `gpt-5.4-mini-2026-03-17`, medium reasoning, standard service tier, maximum 5,000 output tokens, structured JSON, `store: false`, no model tools.
- Maximum ten requests and $10 estimated standard token cost. Before each request, reserve a conservative input-byte/token bound plus maximum output cost. Keep failed-request reservations. This is a local runner limit, not an account-wide billing limit.
- Before calls, require all unchanged scorer tests to pass and each seeded version to fail the same unchanged test module. Preserve source/test hashes and exact request packets.
- Retain proposed edits as inert text. Review each patch before applying it to an isolated copy. A repair succeeds only when the unchanged complete scorer test module passes. Do not modify test assertions or production source.
- Clean control succeeds with `no_change` and no edits. Ambiguity control succeeds with `insufficient_information`, no edits, and no promised recovery from the same observations. Inspect the reasoning manually.
- Report every task and repetition separately, including repair failures, false bug reports, incomplete outcomes, API tokens, latency and estimated cost. No claim of time saved is made without a timed comparator.

The primary decision is whether a further bounded debugging pilot is justified. At least one independently verified repair without regression, correct ambiguity handling, and an honest clean-code result are required; this does not establish general review accuracy.

## Data and credentials

Only allowlisted software source and invented synthetic descriptions leave the machine. Patient scans, images, metadata, clinical notes, secret files and existing private research outputs are excluded. The key is loaded only from an explicit approved local env file and sent in the Authorization header to the fixed OpenAI endpoint. Redirects are disabled. Key contents are never logged. Generated artifacts and the local source graph remain outside Git.

## Sources

The [model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini) specifies the frozen snapshot and standard text rates of $0.75 per million input tokens and $4.50 per million output tokens. The runner conservatively ignores any cached-input discount when estimating cost. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) defines the Responses API JSON-schema format. Usage-derived estimates are not a billing statement.

## Review

Three independent advisors and anonymous peer review were used instead of the council skill's default five, conserving the remaining Codex allowance. Review favored executable scorer checks and a valid-code control over another attempt to infer unresolved anatomy. Graphify was restricted to the scorer and its tests; its graph stays local.

## Results

Completed on November 8, 2026. All 80 local scorer/runner tests passed before the live run. The copied baseline passed all 73 scorer tests; the three seeded copies failed 9, 6 and 1 tests respectively. Every copied test file retained the original bytes.

Ten requests completed without HTTP errors, retries or model substitutions. Eight responses contained valid structured answers. Both unchanged-code reviews exhausted the fixed 5,000-output-token allowance and produced no completed answer; they remain two unsuccessful control outcomes.

| Task | Repetition 1 | Repetition 2 |
| --- | --- | --- |
| Coverage denominator | Proposed reference-eligible denominator instead of all scheduled units; 8 tests failed, 65 passed | Wrong denominator and unmatched edit anchor; could not apply as written |
| Equal totals versus exact recovery | Correctly identified count-parity error, but omitted side assignment; 1 test failed, 72 passed | Same omitted safeguard; 1 test failed, 72 passed |
| Mixed integer label precision | Replaced float-promoting overlap construction with exact integer pairs; all 73 tests passed | Added redundant integer conversions without preventing float promotion; 1 test failed, 72 passed |
| Unchanged code | Incomplete at output limit | Incomplete at output limit |
| Explicitly indistinguishable observations | Correct uncertainty reasoning and required `insufficient_information` status | Correct uncertainty reasoning, but returned `no_change`; fails the frozen status rule |

The six defective packets were all labeled `defect`; this alone is not repair success. After manual inspection, exact edit hashes were whitelisted for a separate local verification phase. Five proposals applied to isolated copies and one had an unmatched anchor. **One of six proposed repairs passed the entire unchanged test module.** Four retained failures; one was not executable as written. The runner itself never executes generated edits. The one successful proposal restores behavior already present in the unmodified scorer; no new counter improvement resulted.

Both ambiguity responses rejected guaranteed recovery from identical observations and proposed no edits. Only one met the exact status rule. This control supplies the indistinguishability premise explicitly; it does not test independent discovery of an imaging limitation. The clean-code controls provide no evidence about false-positive review behavior because neither response completed.

API usage was **29,270 input tokens and 25,307 output tokens**, including 23,262 reasoning tokens. The full-input-rate estimate is **$0.135834**; cached-input discounts may reduce the bill. This is a usage-based upper estimate at the documented standard rates, not a billing-statement reconciliation. Summed request latency was 132.37 seconds, median 10.38 seconds. Local test/review time is additional and no time-saving comparison was measured.

The gate for scaling this configuration was not met: repairs were inconsistent and the unchanged-code control did not complete. API analysis is usable as a source of suggestions under manual review and executable checks. Automatic application of these outputs is not supported by this result. A future comparison would need a separately frozen design and must retain this run; no further requests were made to rescue its scores.

Artifacts are local under `%LOCALAPPDATA%/EthmoidComplexity/api-pilots/2026-11-08-01`: exact request packets, protocol/source/test/runner hashes, all responses, cost accounting, original and seeded fixtures, manually reviewed candidate copies, all test logs, and independent audit. The safe source graph is separately under the local API-pilot staging folder. No patient input was processed or uploaded; production counting behavior and prior experiments remain unchanged.

Reproduce the offline checks from this checkout:

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_api_failure_analysis_pilot.py tests/test_evaluation_scoring.py -q
```

The pilot runner defaults to offline preparation. A live run requires an explicit `--run-api`, source root, new output directory, and approved env-file path. Do not use patient material as inputs or automatically execute returned edits.
