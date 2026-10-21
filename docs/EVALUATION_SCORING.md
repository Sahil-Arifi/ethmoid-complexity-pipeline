# Synthetic evaluation scoring harness

Prepared October 20, 2026. `scripts/evaluation_scoring.py` checks count-evaluation arithmetic and spatial error reporting using invented records and complete synthetic label arrays. It prepares the software needed for a future evaluation. Its output declares `evidence_type: synthetic_engineering_only` and `clinical_accuracy: NOT_MEASURED`.

The user authorized this harness and an extension of the [draft evaluation protocol](CLINICIAN_COUNT_EVALUATION_DRAFT.md) while awaiting more data. This phase uses no patient files, runs no detector, and creates no clinical references or case assignments. Existing experimental scorers, frozen artifacts, production counting and the application remain unchanged.

## Run the demonstration

From the repository root, using the existing Python environment with the project's numerical dependencies:

```powershell
python scripts/evaluation_scoring.py --demo --output "$env:TEMP\ethmoid-scoring-demo.json"
```

Choose a new output filename. The command writes one deterministic JSON report with exclusive creation, so an existing file causes an error. The command accepts the built-in demonstration only; it has no patient-data importer. The report contains the invented census and records, synthetic label sequences, computed results and the scorer's source hash. The sequences represent arrays shaped `(length, 1, 1)` with all demonstration objects assigned left. The repeated-character source/method hashes and reference locks are invented binding examples, not hashes or approvals of clinical files. Repeating the same source to a different new filename should produce identical report bytes.

`tests/test_evaluation_scoring.py` contains the executable checks for the input boundary, arithmetic and spatial cases. Tests and demonstration results are software evidence; they cannot establish the counter's clinical performance.

## Count input boundary

The pure function `evaluate_counts(census, records)` takes an explicit scheduled census and normalized current records. It does not read files or resolve worksheet revision histories.

The census has `schema_version: 1`, `evidence_type: synthetic`, an `evaluation_freeze_id`, a `method_sha256`, and cases with `case_id`, `participant_group_id` and `source_sha256`. Case IDs, source hashes and participant groups must be unique. Hashes use lowercase SHA-256 hex strings. One case per participant is supported. Each case schedules left, right and total endpoints, even if there is no corresponding record. Multiple reconstructions per participant need a future grouped design and adapter.

Each record belongs to one `(evaluation_freeze_id, case_id, endpoint)` and carries reference and prediction fields. Duplicate identities and records outside the census are invalid. A missing record remains in the census with a missing reference and unknown prediction.

| Record | Fields and rules |
| --- | --- |
| Reference | Nested under `reference`: `state`, `count`, `independence`, `source_sha256`, `scope`, `locked_at` and `evidence_ref`. State is `final`, `provisional`, `missing` or `conflict`. Final counts require nonnegative integers; provisional counts allow integers or `null`; missing/conflicting counts must be `null`. Independence is `independent`, `model_assisted` or `unknown`. Scope is `full_3d`, `landmark_2d`, `partial_volume` or `unknown`. A supplied lock must be an ISO timestamp with a timezone. |
| Prediction | Nested under `prediction`: `state`, `count`, `kind`, `source_sha256`, `method_sha256`, `run_id`, `uncertainty` and `reason`. State is `available`, `withheld`, `technical_failure`, `not_run` or `unknown`. An available count requires `kind: original_automatic`, matching source and method hashes, and a run ID. Uncertainty flags remain a list. Unavailable states carry `null` counts and a nonempty reason. |

An eligible reference must be final, independent, matched to the exact census source, full 3D, locked and supported by an evidence reference. The evaluator derives eligibility from those fields. It does not use a manual `paired_error_eligible` override. Provisional, conflicting, source-unmatched or nonindependent references stay visible without entering final-reference error calculations; they do not reduce the scheduled or available-answer denominators.

Malformed records reject the batch before scoring. Examples include booleans supplied as counts, negative or fractional counts, duplicate endpoint records, an available prediction without its required bindings, or a numeric prediction marked unavailable. An explicit zero is a completed count; `null` means no count.

Field validation and matching hashes check internal consistency only. They cannot authenticate a file, prove that a reviewer was independent, confirm a lock actually happened, or establish that a source mapping is correct. A future validated adapter must obtain and verify that evidence from protected records before patient evaluation is authorized.

## Metrics and acceptance rule

For each endpoint separately, the report retains the complete census and calculates the sets defined in the [draft protocol](CLINICIAN_COUNT_EVALUATION_DRAFT.md): N is scheduled, R has eligible references, C has available automatic answers, P has both, and E is exactly equal within P. Prediction states partition N. Reference states and eligibility reasons remain available for review.

The report includes per-record signed and absolute errors on P, mean bias, mean/median/maximum absolute error, exact agreement E/P, answer coverage C/N, reference coverage R/N, end-to-end exact success E/R, and census exact success E/N. Rates retain their numerator and denominator. An empty denominator has a `null` value and is not estimable; it is never reported as 0% or 100%.

The fixed harness rule retains available numeric answers with their uncertainty flags visible. It does not exclude flagged answers after seeing the results. Withheld answers and technical failures remain in coverage and end-to-end denominators but have no numeric error. There is no tolerance setting or clinical pass threshold.

Left, right and total are independent reported endpoints. Available predictions for a case must come from the same original run, including when its total is missing. The evaluator checks an explicitly supplied total against compatible available side counts and rejects a conflicting sum. Even when one side is missing, the total cannot be smaller than the known side. A feasible partial set remains `not_checkable`; the evaluator does not derive or replace a total. A correct total cannot erase opposite left/right errors. Incompatible scopes or unavailable sides cannot justify deriving a full-volume sum.

The draft's invented example is an arithmetic check: references 4, 5 and 6 with predictions 4, withheld and 8 give N = R = 3, C = P = 2 and E = 1. Signed errors are 0 and +2; mean absolute error is 1, E/P is 1/2, C/N is 2/3 and E/R is 1/3. If the third reference becomes ineligible, N and C remain unchanged while R and P shrink. Both the changed denominator and its reason must remain visible.

## Synthetic spatial diagnostics

`score_synthetic_instances` takes complete known reference and prediction labels on the same grid, with explicit known sides for each object. Label IDs identify objects; matching numeric IDs are not required. Every nonzero voxel overlap connects a reference object to a prediction object. Connected components of this bipartite graph define the diagnostic relationships:

| Reference objects | Predicted objects | Relationship |
| --- | --- | --- |
| 1 | 1 | `one_to_one` |
| More than 1 | 1 | `merge` |
| 1 | More than 1 | `split` |
| More than 1 | More than 1 | `complex` |
| 1 | 0 | `missed` |
| 0 | 1 | `extra` |

A one-to-one relationship alone does not establish exact recovery. Exact spatial success requires equal object masks and correct side assignment. The demonstration and focused tests exercise missed and extra objects, merge/split relationships, compensation that preserves a total, and side errors. Known complete synthetic coverage makes these classifications possible.

Any overlap, including a single boundary voxel, creates an edge under this diagnostic convention. It is not a validated clinical correspondence rule or an overlap threshold for noisy annotations. Video counts alone provide neither shared-grid masks nor complete spatial references. This helper therefore does not classify clinician videos or infer a missed/extra cell from a count difference. Existing blur-experiment scoring and its thresholds are unchanged.

## Decisions and remaining work

The scope decision used a resource-bounded adaptation of the council workflow: two independent advisors on the existing authorized model connection, followed by anonymous peer review. The implementation keeps census arithmetic separate from complete synthetic spatial checks. It avoids changing historical experiment rules or treating the manual worksheets as an already validated data pipeline.

CSV revision resolution, source-record adapters, statistical inference and clinical acceptance thresholds are not implemented. Before patient use, validate an adapter against the protected worksheet and saved-output histories; resolve participant exposure, exact reconstruction and reference independence; and freeze the remaining clinical decisions in the draft protocol. The synthetic evidence marker is not a deidentification check. Patient processing still requires authorization for the actual data and workflow.

## Executed software verification

On October 20, 2026, the focused scorer suite passed **73 tests**. The dedicated counting suite passed **716 tests without skips**, including these scorer checks. The application environment passed **357 tests**, with 21 SciPy-dependent modules skipped and nine integration tests deselected under its existing configuration; the dedicated counting suite executed the skipped modules. These suite totals overlap and must not be summed into a unique-test count. Existing dependency deprecation warnings remain.

The medical-data guard passed against the complete staged source index. Two separate demonstration processes wrote byte-identical reports under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/evaluation-scoring/2026-10-20-01`. The focused tests also verified exclusive output creation, preserved existing report bytes, input immutability and record-order invariance. No historical experiment was rerun or rescored by this demonstration.

An independent saved-output audit passed **270 assertions**, recomputing every count row, denominator, rate and error statistic, and all eight spatial examples without importing the scorer. It verified both report hashes and their scorer-source binding. The audit is saved beside the reports as `independent-audit.json`. Spatial side checks use the source-bound all-left demonstration convention; the saved label sequences alone do not establish laterality.
