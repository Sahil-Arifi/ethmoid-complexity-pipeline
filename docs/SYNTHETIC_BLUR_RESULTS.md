# Thin-wall and blur findings

The unchanged detector recovered both physical pockets in **6 of 18** constructed paired-space cases. It merged the other **12 pairs into one candidate**. All **six single-space controls** met their count, overlap, core-coverage, connectivity and review-flag requirements. There were no unexpected execution errors. These results describe one artificial rectangular enclosure and a declared imaging model; patient counting accuracy remains unmeasured.

## What the test changed

The earlier ideal experiment used sharp, directly drawn detector voxels and passed all 14 exact construction cases plus six controls. This follow-up first constructs known physical spaces at 0.125 mm, blurs their intensities, then averages into 0.5 mm detector voxels. It varies divider thickness, blur and the position of the sampling grid. The detector, its bone/air settings, anatomical-localizer bypass and supplied-ROI scope remain unchanged. [The frozen design](SYNTHETIC_BLUR_EXPERIMENT.md) defines all geometry, imaging, matching and scoring conventions.

| Physical divider | Paired cases | Correct physical count and recovery | What happened |
| --- | --- | --- | --- |
| 0.25 mm | 6 | 0 | All six became one detected space. |
| 0.5 mm | 6 | 1 | Only zero blur with a 0.25 mm X sampling phase recovered both spaces. |
| 1.0 mm | 6 | 5 | The 0.5 mm blur / zero-phase case became one detected space. |
| No divider; one-space controls | 6 | 6 | All six remained one detected space with the required crossing-midline reason. |

Blur is expressed as the standard deviation of the constructed Gaussian kernel, not a measured scanner resolution. A sampling phase moves the image grid relative to the same physical objects. Changing that alignment alone changed recovery in the 0.5 mm zero-blur pair and the 1 mm / 0.5 mm blur pair. None of these cases estimates performance on anatomically shaped cells.

## What the failures mean

All six paired inputs with a separating barrier at the fixed **200 HU bone threshold** recovered both physical spaces. None of the 12 inputs without that threshold barrier did. The physical divider still exists in those 12 constructions. The visibility diagnostic describes how the imaged input is classified at this threshold; it does not redefine truth, remove a failed row or establish that all intensity evidence disappeared.

Each missed pair produced one connected candidate spanning both reference pockets. A one-to-one matcher assigns that candidate to one reference; the second reference remains unmatched with overlap and core coverage zero. The matched reference has Dice approximately 0.644 to 0.667 and full core coverage. This prevents a merged object from receiving credit as two correctly recovered pockets. Continuous scores and all engine reasons remain in each case's saved result.

There was no threshold sweep, alternative counting model, detector improvement, training or patient processing. A separate frozen experiment would be needed to test whether weaker wall evidence can improve recovery while avoiding invented divisions in the one-space controls. These findings identify an undercounting limit; they do not authorize promoting the current counter as clinically reliable.

## Verification and retained evidence

The source-bound premeasurement checks passed **128 dedicated counting tests without skips**, including 38 new generator, scorer, persistence and error-path checks. Independent code review found and corrected three harness issues before detector execution: malformed registry IDs, JSON key normalization in saved-plan comparison, and scorer exceptions escaping the case error record. Regression tests first reproduced each defect. The counting source and default settings were not changed.

Two initial fresh processes produced identical protocol/results bytes, 192 saved arrays and 25 comparison/summary PNGs. An independent saved-array audit performed 2,634 checks of hashes, geometry, fine physical topology/volume, affine/side assignment, masks, registries, matching, overlap, core coverage and input visibility. It confirmed every outcome and retained all 12 misses in the denominator. PDF bytes differed only in their creation/modification timestamps. Two process executions do not create 48 independent cases.

Inspection required display-only corrections to fixed column positions and threshold/mask captions. Earlier runs, their exact source snapshots and frozen plans remain preserved. The generated inputs, reference identities, matching rules, scores, detector and settings stayed unchanged. Final reports are `run-04/report.pdf` and `run-05/report.pdf`, bound to source commit `7ea483c`. Their 192 arrays and all case-result bytes match the original execution; their protocol/results/Markdown and 25 PNGs match each other. PDF bytes match after normalizing creation/modification timestamps. A final independent report audit checked the summary and selected comparison pages, all 24 recorded outcomes and the failure-score range. These reports and verification records remain outside Git under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/thin-wall-blur-v2/2026-10-12-01`; the directory date records when the authorized task began.

Both [GitHub Actions jobs at the final measured source commit](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/actions/runs/38361210220) passed. This checks the counting experiment and the broader repository workflow; it does not turn the 12 physical recovery failures into successes.

For this experiment, the source repository retains the runner, tests, protocol and this aggregate summary. NPZ files do not encode runtime writeability flags: immutable-in-memory inputs are supported by the pinned generator and preservation checks, while saved hashes bind their actual values. No image is a patient image. Scanner response, noise, irregular anatomy, automatic localization, full-volume acceptance gates and clinical accuracy remain untested.
