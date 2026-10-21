# Clinical count evaluation plan - draft

Prepared October 20, 2026. This draft defines records and calculations, not a clinical pass threshold or permission to run patient inference. No new case assignments, reference counts or patient results were created. Complete and freeze the outstanding decisions before evaluation; do not invent a sample-size or error target from the synthetic results.

## Question and unit of evaluation

Can a frozen automatic method give sufficiently accurate left, right and total unique-cell counts on eligible, previously unused participants, while providing answers often enough to be useful? Report left, right and total separately. A correct total may hide compensating left/right errors.

Use one preselected reconstruction per participant for the primary analysis. If multiple reconstructions or repeat scans are studied, predeclare a secondary grouped analysis; never count them, slices, video frames or cells as independent participants. Keep the eligibility census and all reasons for exclusion. Reconcile earlier reviewed and historically evaluated participants before claiming any new group is untouched.

Keep one current analysis row per `(evaluation_freeze_id, case_id, endpoint)` for the frozen method and selected reconstruction, including scheduled units with no video, reference or output. Multiple supporting sessions or comparisons do not add evaluation units. A superseding row replaces the earlier row's analytic role while preserving history; unresolved reference conflicts stay in the census without entering paired error. This preparation creates no actual evaluation assignments.

The primary reference scope is a complete 3D unique-cell count. Landmark/selected-slice and partial-volume counts are separate descriptive endpoints unless a distinct comparison is explicitly defined before evaluation. Video commentary is not a voxel mask and cannot establish segmentation overlap or Dice accuracy.

## Development and reference independence

- Videos watched to understand or tune the algorithm, prior disagreements, and cases already used for decisions belong to development. All related scans and reconstructions follow the participant group.
- Reserve a future evaluation group only after identity, prior exposure, eligibility and selection rules are verified. Leave roles unassigned when that evidence is missing. Record role changes rather than overwriting their history.
- Manual counts should be completed and locked without seeing the algorithm's counts or masks. Record reviewer, scope, rules source, timestamp, uncertainty and any adjudication. Preserve original opinions. Model-assisted or unknown-independence references are development/descriptive evidence, not independent performance evidence.
- If evaluation findings later influence tuning, preserve that evaluation result and mark the group as exposed. Subsequent performance needs a new untouched group or an explicitly declared reuse analysis.

## What must be fixed before evaluation

| Decision | Current state |
| --- | --- |
| Clinical decision owner and reference-review/adjudication process | Pending agreement |
| Participant selection, eligibility rules, primary reconstruction rule and sample-size/precision rationale | Pending agreement |
| Exact applicable counting definitions and their source revision | Carry forward existing answers; reconcile only concrete conflicts |
| Primary endpoint and left/right/total reporting | Proposed here; clinical agreement pending |
| Maximum acceptable count error and exact-agreement target | Not set |
| Minimum answer coverage and handling of uncertain available answers | Not set |
| Handling of missing/conflicting references and reference review completion | Predeclare before outcomes are known |
| Method code, configuration, output-acceptance rule, source manifest and evaluation script | Freeze before execution |
| Authorized execution, independent reviewer access and protected transfer/storage | Confirm for the actual data and workflow |

This is an internal preparation list, not another broad questionnaire for the clinician. Use incoming explanations and existing answers first. Do not change targets after seeing results. Later amendments require a new dated version and explicit separation of affected results.

## Denominators and metrics

For each endpoint separately, define the following sets after predeclared eligibility review. Report both numerator and denominator, not percentages alone. Report the screened total and pre-evaluation exclusions with reasons alongside these sets.

| Symbol | Definition |
| --- | --- |
| N | All selected eligible endpoint units scheduled for the frozen evaluation. Later technical failures, withholding and missing references stay in this census. |
| R | Units in N with a final, independent, source-matched reference for this exact endpoint/scope. |
| C | Units in N with an available original automatic numeric answer under the frozen acceptance rule, including any retained uncertainty flags. |
| P | Paired units in both R and C. |
| E | Units in P with exactly equal automatic and reference counts. |

| Measure | Calculation and interpretation |
| --- | --- |
| Signed count error | Automatic minus reference on P; positive means overcount, negative means undercount. Show per-case values and mean bias. |
| Absolute count error | Absolute signed error on P; report mean, median and maximum with the number of pairs. |
| Exact agreement among available answers | E / P; undefined when P is zero. This is selective agreement, not a statement about every selected case. |
| Answer coverage | C / N, alongside separate withheld, technical-failure, not-run and unknown counts. Those prediction states must partition N. |
| Reference coverage | R / N, with reasons for missing, conflicting, provisional, unmatched or nonindependent references. |
| End-to-end exact success where a reference exists | E / R, retaining withheld and failed outputs as cases without exact success. Also give E / N as a census description; do not call unknown-reference cases proven errors. |
| Agreement within an agreed tolerance | Number with absolute error no greater than the predeclared tolerance divided by P, and separately by R. Tolerance remains unset. |
| Cell-level error categories | Report confirmed missed/extra/split/merge relationships only where spatial correspondence and relevant reference coverage support them. Give the reviewed denominator; do not extrapolate to unreviewed cells. |

Never score a missing number as zero or count an abstention as correct. A numeric count with an uncertainty flag remains visibly flagged. If an acceptance rule excludes such answers, declare it before evaluation and classify them as withheld for that analysis. When a denominator is zero, report `not estimable` rather than 0% or 100%.

Report confidence/precision estimates appropriate to the number of independent participants. The method and precision target remain to be selected before evaluation; multiple reconstructions require participant-level grouping. A small development sample cannot establish population-level performance.

## Illustrative calculation check

This is invented arithmetic for checking the worksheet, not study evidence. Suppose three eligible left-side units have independent final references 4, 5 and 6. Their automatic outputs are 4, withheld and 8. Then N = R = 3, C = P = 2 and E = 1. Signed errors are 0 and +2; mean absolute error is 1. Exact agreement among available answers is 1/2, answer coverage is 2/3, and end-to-end exact success is 1/3. The withheld case has no numeric error; it remains in N and R.

If the third reference is unconfirmed, N stays 3 and C stays 2, while R becomes 2 and P becomes 1. Report reference coverage 2/3 and the reason. Do not report improved accuracy without disclosing that changed denominator. If left is overcounted by one and right undercounted by one, total agreement can still be exact; the side errors remain.

## Deliverable after evidence arrives

First produce a source-matched development review with unresolved items retained. Then propose one bounded algorithm experiment based on those observations for user authorization. Clinical evaluation can proceed only after the missing design decisions and independent evidence are in place. The website remains behind the user's automatic-accuracy checkpoint.

See the [review guide and blank worksheet schemas](CLINICIAN_VIDEO_REVIEW.md). No machine-readable clinical approval, training label set or runnable evaluator is created by this draft.
