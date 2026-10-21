# Clinician video review preparation

Status: templates ready; no new recordings or data reviewed in this phase. Prepared October 20, 2026, after the user reported that the clinician will send narrated manual-count videos and more data. This is a preparation workflow, not a new counting experiment or clinical protocol approval.

The existing clinical answers remain the starting point. Capture what the recordings resolve before asking any targeted follow-up. The historical clinical-review questionnaire is not a new task list: its unresolved labels may predate later answers. Do not request annotation exports that the clinician has already said are unavailable.

## Start here when a recording arrives

1. Preserve the original in the existing protected local work area. Record its hash and relative location. Do not upload videos, transcripts, scan identifiers, filled worksheets or case mappings to Git, Graphify, a new transcription service, or the online blank guide. Neutral labels alone do not deidentify pixels or audio.
2. Add the participant and case to the case register before opening content for development. Related series, repeat scans and videos stay in the same participant group. If identity or prior exposure is unknown, leave the role unassigned. No actual cases are assigned by this preparation.
3. Add one review session for each video segment that concerns one exact reconstruction and one measurement scope. If the recording switches series, make another session; the same video hash may occur in several sessions. Preserve the distinction between a full-volume 3D count and a selected-slice or landmark count.
4. Record observations in time order. Give a reference cell a stable local label; reuse it when the clinician revisits that cell on another slice. Keep her explicit words, what is visibly shown, and our interpretation in separate columns.
5. Record final counts separately from intermediate counting actions. A spoken final total may support a count reference even when the video cannot support cell-by-cell correspondence. An incomplete recording is not automatically a complete cell reference.
6. Compare only to a preserved output bound to the same source volume. Record the run, algorithm, configuration, result hash and revision before examining disagreement. Keep unresolved matches pending; do not select the series or run whose count is closest.

Use the draft [evaluation plan](CLINICIAN_COUNT_EVALUATION_DRAFT.md) before assigning any future evaluation cases. No new inference, correction of application records, model fitting or threshold changes are authorized by these preparation templates.

## Blank local worksheets

Header-only CSV copies and this guide are prepared outside Git under `%LOCALAPPDATA%/EthmoidComplexity/work/clinician-video-review/preparation-20261020`. Open the CSV files in a spreadsheet and import identifiers, timestamps and hashes as text. The files are manual worksheets; they do not enforce rules automatically. Keep these originals blank and work in a new protected local batch directory when data arrives.

Every `*_id` is a neutral local identifier, not a patient name. File and source references resolve only within protected local storage. Every row has a unique `row_id`; append corrections with `supersedes_row_id`, recorder, date and reason instead of deleting the earlier row. The current version is the unsuperseded row in a linear revision chain. Conflicting branches require resolution and remain unscored. Dates use ISO format; video times use `HH:MM:SS.mmm`. A cell label alone is not proof of spatial correspondence.

### 1. Case register: participant grouping and exposure history

One row per case state; append a new state when the case's role or exposure changes. `case_id` groups sessions; `participant_group_id` groups all related scans, reconstructions and recordings. Keep the real identity mapping separate and protected.

```csv
row_id,supersedes_row_id,case_id,participant_group_id,identity_status,prior_exposure,first_development_view_at,role,role_reason,selection_freeze_id,recorded_by,recorded_at,change_reason
```

- `identity_status`: `confirmed`, `unknown` or `conflict`; record the identity evidence in `role_reason` by local reference, without copying identifiers into this template.
- `prior_exposure`: `none_documented`, `development`, `prior_evaluation` or `unknown`. A claim of none needs an exposure-history check.
- `role`: `unassigned`, `development`, `evaluation_reserved`, `evaluation_used` or `excluded`. All related records must agree. Reasons accompany every role and change.
- Existing cases whose images, counts or disagreements influenced development remain development. New data is not automatically a holdout. Viewing a narrated explanation for development marks the whole participant group as development. A former evaluation participant used for tuning cannot later be restored to untouched status; preserve its historical evaluation record.
- A reserved evaluation group needs verified identity/exposure, a predeclared selection rule and freeze ID. Record any later exposure; do not quietly remove difficult cases from the original evaluation denominator.

### 2. Review sessions: source and reference provenance

```csv
row_id,supersedes_row_id,session_id,case_id,video_id,video_sha256,video_relpath,segment_start,segment_end,received_at,clinician_id,series_ref,source_volume_sha256,series_match,match_evidence_ref,measurement_scope,rules_source_ref,model_visibility,cell_reference_coverage,limitations,recorded_by,recorded_at,change_reason
```

- `series_ref` points to the protected series mapping. `series_match` is `unconfirmed`, `verified` or `conflict`; only verified matching permits numerical comparison. Record the evidence and verifier in `match_evidence_ref`. Similar appearance, display slice number, filename or count alone is insufficient.
- `measurement_scope`: `full_3d`, `landmark_2d`, `partial_volume` or `unknown`. Never sum landmark counts into a full-volume count. `rules_source_ref` points to the existing applicable clinician answer/protocol revision, not a newly invented definition.
- `model_visibility`: `not_seen_confirmed`, `seen_before`, `seen_during` or `unknown`. Model-assisted commentary can support development but is not an independent reference.
- `cell_reference_coverage`: `complete`, `partial` or `unknown`; complete requires explicit support for the reviewed region and distinct cell identities. `limitations` records skipped areas, inaudible remarks, unverifiable locations or unresolved orientation. Coverage is about cell correspondence, not whether a final number was spoken.
- Keep an unconfirmed session for qualitative learning. Do not infer physical coordinates, scanner calibration or left/right from screen placement. Record visible orientation evidence; uncertain laterality stays unknown.

### 3. Cell observations: what was counted and why

```csv
row_id,supersedes_row_id,session_id,reference_cell_id,time_start,time_end,side,action,location_evidence,clinician_statement,visible_action,our_interpretation,certainty,disposition,recorded_by,recorded_at,change_reason
```

- `side`: `left`, `right` or `unknown`. `action`: `new_cell`, `revisit`, `revise`, `exclude`, `uncertain` or `final_count_statement`.
- Reuse `reference_cell_id` for revisits. Leave it blank when identity is unresolved; do not manufacture an instance from a timestamp. If a correction replaces a group of cells, document all affected IDs and the relationship in `change_reason`, preserving prior observations.
- `location_evidence` includes the visible plane/location and orientation evidence, plus a local source reference where available. A video timestamp locates evidence in the video, not a unique point in the CT volume.
- `clinician_statement` preserves the supported quote or faithful paraphrase and its timestamp. `visible_action` describes what can be seen. `our_interpretation` is explicitly provisional and never silently becomes a clinician label.
- `certainty`: `explicit`, `uncertain` or `not_stated`; it describes the clinician's expressed certainty, not an invented probability. `disposition`: `included`, `excluded`, `unresolved` or `not_applicable`.

### 4. Saved-output comparisons: spatial relationships

```csv
row_id,supersedes_row_id,comparison_id,session_id,source_volume_sha256,run_id,algorithm_version,configuration_sha256,result_sha256,prediction_kind,revision,reference_cell_ids,candidate_ids,relationship,correspondence_status,evidence_ref,reviewed_region_coverage,reviewer_id,recorded_by,recorded_at,change_reason
```

- Bind each comparison to the source and exact saved result. `prediction_kind` is `original_automatic` or `corrected`; store corrected comparisons separately and never substitute them for automatic performance. Reuse existing run/revision identity from `counting_service.get_run`, `result.json` and provenance records; do not assume that the latest revision is the original output.
- Separate set members in `reference_cell_ids` and `candidate_ids` with semicolons. IDs are scoped to the session and saved run. `relationship`: `one_to_one`, `merge`, `split`, `complex`, `missed`, `extra` or `unresolved`.
- A merge links several reference cells to one candidate; a split links one reference cell to several candidates. Complex many-to-many correspondence stays explicit. An empty set means no matching object only after correspondence review; it does not mean the reviewer has not looked.
- `correspondence_status`: `confirmed`, `pending` or `conflict`. Confirmed requires reproducible spatial evidence and a named reviewer. A total-count difference is not enough to label a merge, split, missed cell or extra candidate. `reviewed_region_coverage` is `complete`, `partial` or `unknown`.
- A missed/extra classification requires complete reference coverage of the relevant region and sufficient saved-output evidence. A flagged relation can remain a provisional hypothesis in `change_reason` while `relationship` stays unresolved. Do not count relationship rows as independent cells or cases.

### 5. Count comparisons: one row per endpoint

```csv
row_id,supersedes_row_id,case_id,evaluation_freeze_id,session_id,comparison_id,endpoint,reference_state,reference_count,reference_evidence_ref,reference_independence,reference_locked_at,prediction_state,predicted_count,prediction_uncertainty,paired_error_eligible,reason,recorded_by,recorded_at,change_reason
```

- For a future evaluation, retain one current analysis row per `(evaluation_freeze_id, case_id, endpoint)` for the preselected reconstruction and method. The frozen census supplies the planned scope even if no recording or output exists. Leave `evaluation_freeze_id` blank for development records; do not invent a freeze or assign actual cases during preparation.
- `endpoint`: `left`, `right` or `total`. The linked session supplies observed measurement scope; the linked comparison supplies the saved run. Leave either link blank when its evidence does not exist and explain why in `reason`. Missing videos, missing references and not-run predictions still need census rows. A comparison may have empty cell sets and unresolved correspondence while its count is still comparable.
- Multiple sessions may support one endpoint row; list supporting local references in `reference_evidence_ref`. They do not increase N, R, C or P. Conflicting observations require resolution or a conflict state. A superseding row replaces its predecessor's analytic role without deleting history.
- `reference_state`: `final`, `provisional`, `missing` or `conflict`; `reference_independence`: `independent`, `model_assisted` or `unknown`. Independent requires supported reviewer provenance and confirmed absence of model influence, not merely a blank visibility field. Preserve disagreements and adjudication provenance before locking a reference.
- `prediction_state`: `available`, `withheld`, `technical_failure`, `not_run` or `unknown`. `prediction_uncertainty` preserves the saved flag/reason even when a number is available. Never turn that flag into a claim of clinical reliability.
- A blank numeric field is missing, never zero. Enter zero only for an explicit completed zero count. Withheld/failed/not-run/unknown predictions have blank numeric values. Provisional numbers remain visible but do not enter final-reference accuracy metrics.
- `paired_error_eligible`: `yes`, `no` or `pending`, with a reason. A yes requires the frozen endpoint scope, verified exact series, final eligible reference and an available original automatic answer from the correct saved run. This field controls numeric-error pairing only; it never removes a selected unit from the evaluation census. Withheld or failed predictions stay in coverage/end-to-end denominators even though numeric error is unavailable. Qualitative observations can remain useful when the numerical gate fails.
- Record clinician-stated total separately from any derived sum. Use `reference_evidence_ref` to mark a total as stated or derived. Derive only from finalized left/right values with the same scope and no unresolved side assignment; retain any stated-total disagreement as conflict rather than overwriting it.

## Readiness check before a comparison is used

For a paired numerical comparison, the case, session and run links resolve; hashes refer to the exact preserved files; the scope and required orientation are known; corrections have one current row; repeat views retain cell identity; the applicable clinical answers are referenced; reference completeness and model visibility are recorded; and the participant role is consistent. Missing-evidence census rows may lack session/run links but must retain their planned case/endpoint identity and a reason. They remain in the scheduled denominator. If any item is unknown, use only the evidence that remains supported; missing values remain unavailable.

For a worked check, imagine two explicitly identified reference cells matching one saved candidate. One spatial comparison is a merge only with confirmed correspondence. Revisiting one of those cells adds an observation, not a third cell. If the series is unconfirmed, keep the commentary but withhold the numerical comparison. These are invented worksheet examples, not new experiment results.

## Existing project sources

- [Current scope](V2_SCOPE.md): exact-series matching, protected evidence and the automatic-accuracy requirement before website work.
- [Saved cohort evaluation](COUNTING_EVALUATION.md): source/configuration provenance, participant split and unavailable counts. Its historical held-out split is not automatically a fresh clinical holdout.
- [Sampling-information results](SAMPLING_INFORMATION_RESULTS.md): controlled synthetic findings and their limits.
- `src/ethmoid_pipeline/webapp/counting_service.py`: original counts, selected revision, source hash and immutable correction lineage. This preparation uses those concepts without editing the application.
