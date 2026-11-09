# Counter failure investigation, 2026-11-08

The attempted count recovery was rejected. It increased accepted correct totals on the existing synthetic observations, but a separate positive-opening example produced a wrong accepted count. The canonical counting safeguards remain in place. Clinical accuracy is unmeasured.

## Measured comparison

Both methods received the same 100 saved native-passage observations: 68 development records and 32 perturbation records. The latter include 16 exact physical reindexings, rather than 32 independent constructions. Only copied, read-only HU, ROI and affine arrays reached either detector. Physical labels, cores, names and expected totals were used after prediction for scoring.

| Method | Correct accepted | Wrong accepted | Withheld | Execution errors |
|---|---:|---:|---:|---:|
| Current canonical counter | 22 | 0 | 78 | 0 |
| Isolated conditional candidate | 25 | 0 | 75 | 0 |

The candidate recovered these three observations, including one exact reindexing:

- `wall_1_sigma_0_phase_0`
- `native_wall_1_aperture_0_sigma_0_phase_0_noise_0`
- `native_wall_1_aperture_0_sigma_0_phase_0_noise_0_swap_xz_flip_y`

All 22 current correct acceptances were preserved. Every partition, cavity, air and bridge mask remained identical to the canonical output. The three recovered masks also matched their reference labels exactly, allowing label-ID renaming. All four material-twin groups remained withheld. These are reused development observations, not an untouched test set or clinical accuracy measurement.

## What was tried and why it failed

`scripts/saturated_barrier_candidate.py` tries one narrowly declared hypothesis: under exact, noiseless, unquantized averaging of known air and bone endpoints, a saturated bone voxel would contain no air. It removes only the subvoxel-communication veto when all samples are exactly -900 or 800 HU, the ROI covers the array, two enclosed free components agree under 6- and 26-connectivity, and both match the existing cavity proposals exactly. Other vetoes remain active. The module carries the model assumptions and rejection warning and is not imported by production code.

Stored endpoint equality does not establish that observation model. The independent falsifier places a strictly positive square channel through the entire wall. Its exact average is computed with Decimal arithmetic and is below 800 HU. Casting to float32 rounds it to 800, so the complete observed input is byte-identical to the sealed construction. The physical connected-space count is one; the candidate accepts two. The canonical counter withholds both.

The runner uses a 0.000001 mm square aperture to demonstrate this numerical limit. It is a mathematical falsifier under the existing positive-opening definition, not a claim about clinical anatomy or scanner resolution. The candidate unit tests contain another independently constructed rounding witness. Unknown material density, clipping and acquisition uncertainty would require additional assumptions as well. A 25/100 result on the saved cases therefore does not justify replacing the current 22/100 acceptance policy.

## Software defect repaired

Malformed affine values previously escaped `canonical_ridge_candidate._inputs` as raw `ValueError`, `TypeError` or `OverflowError`. They now produce the existing `CountingError` with code `INVALID_GEOMETRY`, before proposal generation. Four regression cases failed before the repair and passed afterward. Valid-input count behavior is unchanged. This improves error handling, not count accuracy.

## Reproduction and retained evidence

New artifacts are outside Git under:

`%LOCALAPPDATA%/EthmoidComplexity/api-pilots/2026-11-08-counter-01`

The input census remains at:

`%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/native-passage-v1/2026-10-13-01/run-01`

The original historical directories were hashed before and after execution. New files include `baseline-plan.json`, `baseline-run-01`, `baseline-run-02`, `candidate-plan.json`, `candidate-run-01`, `candidate-run-02`, API requests/responses/accounting and an independent saved-artifact audit. No original result was replaced. Each new candidate run retains every case, complete masks, diagnostics, input/output hashes, spatial correspondence and failures. The plan binds source files, runtime, inputs, candidate budget and rejection rule before evaluation.

The independent saved-artifact audit passed **1,428 checks** without importing or rerunning either detector. Both baseline runs match byte for byte, as do both candidate runs. All **820** protected historical files retain their bytes and inventory. All **22** physical reindexing pairs preserve complete masks, registries, statuses and counts. The locked CPU counting CI command, extended with the two new test modules, passed **776 tests without skips**. Existing upstream scikit-image/NumPy deprecation warnings remain.

The broader application run passed 356 tests, skipped 23 scientific-stack modules covered by the CPU job, and deselected nine integration cases. Its one failure was the HTTP index route returning 503 because the fresh worktree had no frontend build. After installing the existing locked frontend dependencies and building successfully, that test passed on rerun. No application source fix was needed. The tracked-file medical-data guard and `git diff --check` passed. The initial counter-only phase did not require local browser reruns; the later dependency repair did.

The first remote run passed counting, application, frontend and browser checks, then failed its dependency audit on the existing development dependency versions. The follow-up repair pins [Vitest 4.1.12](https://github.com/vitest-dev/vitest/security/advisories/GHSA-qxg6-89mg-rqv8) and resolves [Undici 7.30.2](https://github.com/nodejs/undici/security/advisories/GHSA-4w7w-5xrc-9h26) within jsdom's unchanged dependency range. No other package versions changed. A clean locked install, 42 frontend tests, production build, three local synthetic browser workflows and the dependency audit passed; the audit reports zero vulnerabilities.

Use the locked CPU counting runtime with the active checkout's `src` on `PYTHONPATH`:

```powershell
python scripts/counter_failure_experiment.py --plan <fresh-external-plan.json> --known-run <saved-native-run>
python scripts/counter_failure_experiment.py --plan <fresh-external-plan.json> --output <fresh-external-output>
```

The historical canonical runner's `comparison.baseline` refers to the older ridge result (32 correct, eight wrong, 60 withheld). This new runner explicitly reruns today's canonical method beside the candidate, so those older numbers are not misreported as the current baseline.

## API review and decision review

The existing authorized key was reused. Only explicitly allowlisted counter source, synthetic documentation, synthetic tests and synthetic result summaries were sent to the regular OpenAI Responses API. No patient material or credentials were included in prompts. All requests used `gpt-5.4-2026-03-05`, `store: false`, and a local $5 cap; no provider or authentication switch occurred.

Three requests were recorded. The first timed out without a response; its $0.384265 reservation remains charged against the local cap because billing is unknown. A smaller request exhausted its 6,000-token output allowance entirely on reasoning and returned no usable answer. The final source/result review completed with reasoning disabled and recommended rejecting promotion, citing the rounding counterexample. Returned usage totals $0.135 in uncached token-price estimates; adding the timeout reservation gives $0.519265. This is local accounting, not verified invoice spend.

The API review also called reuse of the baseline's own proposals "leakage." That criticism was not adopted: reusing a detector's derived proposals is legitimate, and no reference labels reach the candidate. The demonstrable wrong acceptance is sufficient for rejection regardless of that critique. API opinion is advisory; saved outputs and independent checks determine the result.

The llm-council workflow was adapted to three independent advisors on the existing authorized connection, followed by anonymous proposal review and root synthesis, to conserve the remaining Codex window. Their shared conclusion was to test the conditional hypothesis, retain its falsifier and preserve the canonical guards. Graphify used local AST extraction on 16 allowlisted source/test files; its output remains outside Git.

The next count-improvement experiment needs independently established acquisition information or a declared physical model that can be falsified on different geometry. An LLM's confidence or a better score on these same records cannot resolve image-identical opposite counts.
