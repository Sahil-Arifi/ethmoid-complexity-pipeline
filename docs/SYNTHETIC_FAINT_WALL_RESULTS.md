# Synthetic faint-wall findings

The single fixed candidate improved recovery of constructed pairs but **failed admission**. It recovered 8/18 known development pairs versus the baseline's 6/18, and 6/16 new challenge pairs versus 1/16. Both methods recovered 21/22 single-space controls. One perforated bone-divider control still produced two candidates in both methods. Every baseline success was preserved, and neither execution reported an unexpected error. These are synthetic engineering results; patient accuracy remains unmeasured.

## What was compared

The production counter and its defaults were unchanged. The helper-only candidate identifies image values from 50 HU up to, but excluding, 200 HU that are face-connected to strong bone inside the supplied region. It passes that weak-evidence mask to the counter's existing exclusion hook. HU is the CT image's intensity scale; this fixed convention does not prove that a faint feature is bone. Original image values still determine air measurements and strong-bone support. Candidate cells receive an experimental review flag whenever a weak mask is used.

[The rules](SYNTHETIC_FAINT_WALL_EXPERIMENT.md) were fixed before measurement. The 24 prior cases remain development evidence. The 32 new deterministic constructions exercise shifted, angled and curved dividers, a corresponding array-axis permutation, openings, partial ridges, fragments, content variation, noise and soft-tissue sheets. Related sampling rows and the axis-permuted geometry are not independent patient observations or a statistical holdout.

Recovery requires the correct total **and** each complete space's shape, interior coverage, connectivity and physical side assignment. A merged pair cannot pass merely by overlapping part of each reference. The generated region bypasses anatomical localization.

| Evaluation group | Cases | Baseline recovery | Candidate recovery |
| --- | --- | --- | --- |
| Known paired spaces | 18 | 6 | 8 |
| New paired spaces | 16 | 1 | 6 |
| Known single-space controls | 6 | 6 | 6 |
| New single-space controls | 16 | 15 | 15 |

Among the new pairs, off-center flat dividers improved from 0/4 to 2/4, angled dividers from 0/4 to 1/4, and axis-permuted flat dividers from 0/4 to 2/4. Curved dividers remained 1/4. The two additional development successes were the 0.5 mm divider with 0.25 mm blur/phase and the 1 mm divider with 0.5 mm blur/zero phase. These improvements do not resolve the 20 remaining missed pairs.

## Why the candidate was rejected

The acceptance rules required improvement in both paired groups, preservation of every baseline success, complete coverage, no execution errors, recovery of all 22 controls and no extra control divisions. Both pair-improvement gates, the baseline-preservation gate, the coverage gate and the execution-error gate passed. Both control gates failed.

`challenge_perforated_divider_row_1` contains a bone divider with an opening connecting its two sides. Its fine physical reference and imaged non-bone region each form one connected space. Both methods nevertheless propose two cells with incomplete-divider review flags. The candidate retained this pre-existing split; it did not introduce a new control split. Its best matched reference has Dice 0.6803 and about 50.02% core coverage, below the required 0.90 and 95%. The second candidate is unmatched. All failures remain in the denominator.

A clinician may define two communicating cells despite an opening. This experiment explicitly uses connected-space truth, so that disagreement is an engineering convention, not proof of a clinical counting error. It was declared before measurement and was not changed to admit the candidate.

All four soft-sheet cases passed under the specified blur and sampling conditions. That does not establish that the mask identifies bone: attachment to a strong boundary cannot distinguish materials, and a complete inferred exclusion can make a false barrier that the present merger cannot reconnect. The one-candidate budget ends with this result; no threshold sweep, post-result tuning or production promotion was performed.

## Verification and retained evidence

Before measurement, **186 dedicated counting tests passed without skips**, including 58 new checks. Independent source review found and corrected admission accepting an extra unknown row, failed scoring on mutated reference cores and cross-method input contamination. Their regression tests failed before the fixes and passed afterward. Each method receives an isolated read-only copy; original saved inputs and every failure score remain intact. The complete tracked-file medical-data guard passed.

Both fresh executions used source commit `4424094c9222203d52841454236a0ec17939c5c3` and the same source/rules/configuration/runtime and six-array input manifests. They each retain all 56 cases, both methods' numerical scores and registries, inferred masks, original inputs, 112 comparison images, two summary covers, a complete Markdown table and a 114-page PDF. Two process executions verify repeatability; they do not create 112 independent cases.

Canonical reports are `run-01/report.pdf` and `run-02/report.pdf` outside Git under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/faint-wall-v1/2026-10-13-01`. The frozen plan and measured source snapshot are retained there. Independent saved-array and report audit records accompany the reports.

The independent saved-array audit passed **11,670 assertions with zero discrepancies**, without importing either detector or the existing scorer. It rebuilt the declared geometry, physical references, affines, discrete Gaussian convolution and block averages; reconstructed images matched with zero HU error. It verified all 1,344 saved input/method-array hashes across both runs and all 672 repeated array pairs, then independently recomputed assignment, overlap, core coverage, connectivity, physical centroids, air measurements, uncertainty and admission. All 24 development baseline inputs, cavity/air arrays and overlapping result fields matched the prior thin-wall/blur `run-04`, along with prior baseline source/configuration/runtime.

The independent report audit passed 1,318 checks: all 56 Markdown rows and case-name mappings, the captions, all 114 ordered pages per PDF and cross-process files. All 341 non-PDF files per run are byte-identical; PDF bytes match after normalizing only creation/modification timestamps. Both covers and representative recovered, merged, perforated-divider, soft-sheet and permuted examples were visually inspected. The PDF is raster; the Markdown table supplies searchable full case names.

Both [GitHub Actions jobs at the measured source commit](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/actions/runs/38370734096) passed: 186 dedicated counting tests with no skips; 284 application Python tests, with seven SciPy-dependent modules covered by the dedicated job and nine integration tests excluded; 42 frontend tests; three synthetic browser workflows; the production build; zero dependency-audit vulnerabilities; and the tracked-file medical-data guard. Passing software checks does not satisfy the failed candidate-admission gates.

The next decision needs to address the clinical meaning of communicating spaces and inspect why the remaining synthetic pairs stay merged. A new behavior comparison would require a separately approved frozen design. Patient scans, scanner calibration, anatomical localization, clinical reference masks and clinical accuracy were not tested.
