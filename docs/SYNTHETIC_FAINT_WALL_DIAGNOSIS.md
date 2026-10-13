# Faint-wall failure diagnosis

The 20 missed synthetic pairs have two different causes: **12 start with one seed and one basin; eight start with two useful basins that the merger collapses.** None is lost through final proposal rejection. This diagnosis identifies where an algorithm revision must act; it does not establish a successful replacement.

The completed comparison remains rejected. Its measured source, parameters, fixtures, scoring rules and outputs were not changed. This follow-up used saved synthetic arrays and one instrumented replay of the unchanged candidate on all 56 cases. No patient data, application or database was processed.

## The counting definition was already answered

The September 16 answered questionnaire was rechecked against its email attachment and the complete text of the matching Word document. The project definitions distinguish a real opening connecting two spaces, counted as one cell, from a thin, incomplete or unclear bony wall, which can separate two cells. Reasking that conceptual question is unnecessary. Private source links and document contents remain outside Git.

Earlier summaries described this definition as clinically unsettled. That was too broad. The unresolved engineering question is how to distinguish a genuine opening from an apparent gap caused by imaging a thin wall. The failed perforated-divider control has a declared physical opening, so its one-cell target agrees with the answered requirement. This does not establish the identity of any patient structure.

The existing baseline test requires two uncertain proposals across a 1.8 mm synthetic opening. The failed challenge has a 3 mm opening and also produces two proposals. The baseline test documents implemented behavior, not authority to override the answered requirement. Both historical results remain preserved; a future implementation must explicitly reconcile this mismatch.

## Why the 50 HU candidate misses pairs

All 20 misses retain a face-connected path between their reference interiors after applying the saved exclusion mask. Every path includes only values below 50 HU. Across all 34 paired fixtures, the mask omitted zero eligible ROI voxels in the interval from 50 to 200 HU. Removing the strong-bone attachment rule therefore cannot fix these particular failures.

In 18 of the 20 missed pairs, none of 128 sampled interior profiles reaches 50 HU. The other two have patchy evidence: 64/128 profiles in the angled case and 32/128 in the curved case. These are measurements at known synthetic divider locations for diagnosis only; those locations must never become detector inputs.

All 14 recovered pairs have two face-connected free regions. Two reconnect if diagonal neighbors are allowed, so this association is connectivity-convention dependent. It is not a general definition of a cell, and the frozen face-connectivity scores remain unchanged.

## Stage attribution from the unchanged replay

| Stage responsible for the final merge | Missed pairs | Observed behavior |
| --- | ---: | --- |
| Seed/proposal generation | 12 | One seed and one pre-merge basin contain both reference interiors. |
| Separator merging | 8 | Two basins own 100% of different reference interiors, then merge into one. |
| Final proposal rejection | 0 | The post-merge basin survives filtering in every missed pair. |

Distinct reference-interior ownership does not prove the pre-merge shapes meet the complete recovery score. A merger-only revision has a relevant mechanism in eight cases; it cannot fix the twelve single-basin cases.

Keeping every initial split would introduce errors. The current merger correctly combines two basins in the isolated-fragment, blurred-opening and 100 HU soft-sheet controls. The other perforated-divider row retains both proposals and fails. A replacement must distinguish those cases rather than disabling merging or changing expected counts.

## Algorithm direction and decision

Do not change production defaults or continue the completed threshold experiment. The evidence rules out an attachment-only fix and a keep-all-splits fix. It supports a separate experimental method with two coupled responsibilities:

1. Use continuous image evidence to propose divisions that the current binary free-space and air-distance seeds miss.
2. Evaluate each proposed divider against local wall evidence, genuine openings and soft contents before retaining or merging it. Keep the ability to merge a proposal; do not turn weak evidence into an irreversible excluded wall.

An intensity ridge alone is not a bone classifier. As a simple information-loss example, averaging 55% material at 800 HU with 45% air at -900 HU yields 35 HU, identical to a voxel filled with 35 HU content. This is an ideal volume-averaging counterexample, not proof that all 20 saved cases are indistinguishable or a calibrated scanner model. It explains why the method must retain uncertainty and sometimes withhold a count when material identity is unresolved.

The next implementation should remain synthetic-only and separate from the measured method. Before measuring it, specify one candidate's seed rule, separator evidence, opening rule and abstention behavior; freeze its code and tests. Retain all 56 now-known cases as development evidence and add independently specified material-confusion and opening controls. Abstention is not a recovered count. Report full spatial recovery, false splits and withheld counts separately. Do not use reference labels, desired counts or known wall positions as inference inputs.

This turn completed diagnosis, not a new candidate comparison. No numerical rule for the new method has been validated. A local merger change alone would leave most missed pairs unresolved; introducing a second candidate under the old frozen budget would misrepresent the experiment.

## Verification and retained evidence

The instrumented replay calls the original watershed and merger through observing wrappers and restores both functions after each case. **56/56 replays exactly reproduce all saved cavity, air and exclusion arrays, every registry field and every warning.** All six input-array hashes per case, saved input/candidate/result file hashes and all 12 measured source-file hashes remain unchanged. No output, threshold or return value was altered.

The independent saved-array census also retained the original artifact hashes. The trace records seed counts, pre/post-merge basin counts, reference-interior ownership and final rejection counts. It does not distinguish air-derived from free-space-derived seeds or record numeric interface-support ratios. Only the candidate was replayed; the original paired baseline results were read without rerunning them.

Evidence is outside Git under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/faint-wall-v1/2026-10-13-diagnosis-01`: `failure_census.py/.json/.csv`, `mechanism_saved_array_checks.py/.json`, `mechanism_review.md`, and `unchanged_candidate_trace.py/.json` with its complete Markdown findings table. The trace refuses to overwrite its result.

To respect the user's usage preference, the LLM Council workflow was reduced to two independent technical analyses, one anonymous peer review and a chair synthesis on the existing authorized connection. No additional provider was used. Source relationships were checked through the existing source-only graph and direct code inspection. There is no new accuracy denominator, clinical validation or production promotion.
