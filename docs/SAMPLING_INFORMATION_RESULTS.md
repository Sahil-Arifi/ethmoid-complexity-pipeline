# New samples versus image enlargement: results

Additional simulated measurements improved discrimination between two supplied shapes in this fixed experiment. The combined observations selected the correct supplied explanation in 31 of 32 evaluation observations and withheld one. The original observations selected 13; interpolating those observations selected six. This does not measure clinical counting accuracy or recover any of the historical counter's withheld counts.

## Scope and fixed comparison

The [design](SAMPLING_INFORMATION_EXPERIMENT.md), cases, seeds, error bound and decision rule were frozen at `64fc1b39fc92449c61933a0cd6f636bc2499e20b` before evaluation. The four development geometry pairs and sixteen evaluation pairs each supply both possible truths: a through-opening with one connected air space and opposed blind recesses with two spaces. Shape alternatives, material values, blur and sampling geometry are given exactly to inference. These parameter variations are not independent patients or unseen anatomical families.

The original grid has 0.5 mm voxels. Linear interpolation enlarges those existing samples. The added view is independently forward-rendered from the continuous geometry on a grid shifted 0.125 mm in X, then combined with the original view. This is not interpolation and does not establish that a different reconstruction from existing CT data supplies independent information. One seeded bounded error field per acquisition is shared between the two truths in a pair. The maximum error is an illustrative 5 HU per voxel, not measured scanner noise.

| Split | Input | Observations | Correct selections | Wrong selections | Withheld | Execution errors | Conditional coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Development | Original | 8 | 4 | 0 | 4 | 0 | 50.0% |
| Development | Interpolated original | 8 | 3 | 0 | 5 | 0 | 37.5% |
| Development | Original plus newly sampled view | 8 | 8 | 0 | 0 | 0 | 100.0% |
| Evaluation | Original | 32 | 13 | 0 | 19 | 0 | 40.625% |
| Evaluation | Interpolated original | 32 | 6 | 0 | 26 | 0 | 18.75% |
| Evaluation | Original plus newly sampled view | 32 | 31 | 0 | 1 | 0 | 96.875% |

The combined measurements retain every original correct selection and add eighteen conditional selections on these same evaluation observations. The interpolation operator smooths differences used by the fixed maximum-residual decision rule; seven original selections become withheld. This is a result for this operator and rule, not a general claim about visual enlargement or clinical interpretation.

The predeclared exploratory target of at least 26/32 correct selections and zero wrong selections was met, with all software gates passing. The target was investigator-selected and has no clinical approval or clinical-performance meaning. Because the true generating hypothesis is supplied and its error is bounded by construction, a unique compatible selection cannot be wrong within this closed set. Zero wrong therefore is not evidence of real-world diagnostic accuracy.

## Noise sensitivity and remaining ambiguity

The 31/32 result concerns the selected seeded observations. It is not a guarantee over every permitted error pattern. A separately labeled posthoc description of the saved hypothesis separations found that the combined predictions differ by more than 10 HU in eight of sixteen evaluation pairs, versus one pair with the original samples. Only those pairs are guaranteed distinguishable for every per-voxel error within +/-5 HU under this two-hypothesis model. This supplementary calculation did not change the frozen target, cases or decision rule.

The withheld combined observation is the blind-recess member of `pair_07`: opening width 0.0625 mm, recess depth 0.375 mm and blur sigma 0.15 mm. The maximum combined hypothesis difference is about 1.229 HU, and both supplied explanations remain compatible with that observation. It stays in the denominator.

## Verification and evidence

All **643 dedicated counting tests passed without skips**, including 38 new model tests and 30 new runner tests. Both new modules first failed against explicit unimplemented stubs; adversarial tests also exposed missing control-census and failure-retention checks before freezing. The repository medical-data guard and whitespace checks passed. The plan/task files follow the guard's existing documentation directory boundary; the guard was not weakened.

Both frozen executions completed **120 method-observations and 226 controls** without errors. Those 120 records are the same forty synthetic observations evaluated by three methods, not 120 independent examples. Reindexing and ambiguity controls remain outside the main denominator. The independent audit passed **1,326 checks**, reconstructing geometry, quadrature, interpolation, seeded errors, residuals, decisions and controls without importing the inference function. Maximum independent integration discrepancy was **6.83 x 10^-13 HU**. All **44 files per execution** were byte-identical; **19 frozen source hashes** and **1,604 protected historical files** matched.

Frozen evidence is outside Git at `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/sampling-information-v1/2026-10-19-01`: `plan.json`, `run-01`, `run-02`, `audit_saved.py`, `audit.json`, `independent-audit.json` and `separation-description.json`. Each run includes all per-pair arrays, controls, metadata, decisions, hashes and failures. The illustrative figure uses the predeclared development shape with opening 0.25 mm, recess depth 0.25 mm and zero blur: original predictions have identical values, while the newly sampled predictions differ by at most 53.125 HU. It is not patient CT.

## Project consequence

The result supports studying which acquisition information is needed to distinguish particular shapes. It does not supply missing information to existing patient scans or establish that another acquisition is indicated. It also does not infer unknown anatomy, handle the full clinical inclusion rules, estimate scanner calibration, or validate patient counts. No clinician outreach, annotation export request, patient processing, website work or production promotion occurred.

The historical counter remains **22 correct accepted counts, zero wrong and 78 withheld out of 100 synthetic examples**, verified from saved evidence without running it. Existing clinician counts should be used where the exact source series is confirmed, with development and evaluation evidence kept separate. Automatic counting and website readiness remain unmet. The fixed experiment is complete and was not retuned after results.
