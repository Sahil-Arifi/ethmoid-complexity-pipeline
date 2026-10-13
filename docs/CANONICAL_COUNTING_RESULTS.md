# Canonical counting and communication withholding: results

The two targeted engineering repairs passed their frozen checks. Exact axis changes now produce matching physical proposals, and all eight previously incorrect accepted counts become withheld. The tradeoff is ten lost correct two-space counts. The successor returns a correct count in 22 of 100 synthetic records and withholds the other 78; automatic counting and website readiness remain unmet.

## Frozen comparison

The [fixed design](CANONICAL_COUNTING_EXPERIMENT.md) binds source commit `7aed71855c91efa94c438cec309a51c982527806`, 15 source hashes, runtime and all 100 saved observations. The candidate sees only copied HU, ROI and affine arrays. Baseline decisions come from the saved preceding experiment. Two separate processes completed the comparison without execution errors or parameter changes.

| Records | Method | Correct accepted | Wrong accepted | Withheld | Errors |
|---|---|---:|---:|---:|---:|
| 68 known development | Saved baseline | 28 | 0 | 40 | 0 |
| 68 known development | Successor | 22 | 0 | 46 | 0 |
| 32 prior perturbations | Saved baseline | 4 | 8 | 20 | 0 |
| 32 prior perturbations | Successor | 0 | 0 | 32 | 0 |
| All 100 | Saved baseline | 32 | 8 | 60 | 0 |
| All 100 | Successor | 22 | 0 | 78 | 0 |

Withholding is not a correct numerical prediction. All 22 previously correct accepted one-space counts remain available. The successor loses six correct pair counts from the known group and four from the perturbation group. Those four are two sealed-wall constructions and their exact reindexings. All ten names are retained in each run's `report.md` and `results.json` and independently checked in `audit.json`.

All records are reused development observations. The 32 perturbation records comprise 16 physical constructions and their exact reindexings; repeated orientations are not independent validation examples. Zero wrong accepted counts in this batch does not establish clinical accuracy.

## What changed

The complete proposal pipeline now runs after a lossless physical-axis permutation and reflection. It uses no resampling, and every spatial output returns to the input grid. All 22 saved base/reindex pairs match across partitions, cavities, air and bridge masks, with matching physical registries. This includes the thin-wall plateau that previously changed 109 proposal voxels. Seven saved partition arrays differ from the old proposals; their names and changed label-voxel totals are recorded by the audit. Old inputs and outputs remain intact.

The successor withholds counts whenever the native free-space mask contains multiple disconnected components. This applies even when watershed proposals do not touch or bridge evidence is unavailable. It removes the eight unsupported counts of two for physically connected small-opening constructions. Existing uncertainty rules still apply to one-component images, so a single component does not guarantee an accepted count.

A diagnostic now searches the entire supplied ROI, including voxels classified as bone. All 18 two-component records have a measured minimum-HU bridge; the remaining 82 records have one native component and an explicit `not_two_components` status. At zero blur, the measured bottlenecks are 693.75 HU for the 0.25 mm opening, 375 HU for the 0.5 mm opening and 800 HU for the sealed 1 mm wall. Five saved bridge witnesses touch the ROI boundary and are flagged. These paths expose signal excluded by the old 200 HU cutoff; they do not establish an anatomical opening or authorize a merge. No threshold was selected from these outcomes.

All four exact-input material-twin groups remain identical and withheld despite their opposite physical references. These observations still cannot uniquely identify wall material.

## Validation and provenance

All **479 dedicated counting tests passed without skips**. Regressions failed against the old passthrough before implementation. The tests cover every one of the 48 signed axis permutations on each of three grids, including anisotropic oblique geometry; all spatial outputs and physical registries are checked. Focused bridge tests cover strong bone, attenuated bone, excluded paths, missing terminals, disconnected ROIs and larger component sets. Runner tests cover immutable plans, truth isolation, input mutation, complete failure records, protected destinations and scoring denominators. The tracked-file medical-data guard passed.

An independent reviewer audited saved arrays without importing or rerunning the candidate. The helper self-check and full audit passed on their first execution. The audit verified all 100 records, all 18 bridge minimality and complete-witness proofs, all 22 reindex pairs, the four material-twin groups, count and registry measurements, all 15 frozen source hashes and all **820 protected historical files**. All **304 files per run** matched byte for byte across the two processes. All predefined repair gates passed.

Plans, complete results, inputs, masks and the independent audit remain outside Git under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/canonical-component-v1/2026-10-13-01`, using `plan.json`, `run-01`, `run-02`, `audit_saved.py` and `audit.json`. Prior failed audits are preserved in their original experiment directory. The fixed source and design were committed before freezing; this results document was added after measurement.

## Remaining algorithm requirement

The repair establishes orientation consistency and removes these unsupported acceptances. It does not recover a reliable count for uncertain multi-space anatomy. A subsequent, separately specified wall/opening model would need to recover correct pair counts while retaining small-opening alternatives and acknowledging image-identical cases. Independent reference evaluation, full-volume localization and clinical accuracy requirements remain unresolved. Production counting, patient scans and the website were not changed or promoted by this experiment.
