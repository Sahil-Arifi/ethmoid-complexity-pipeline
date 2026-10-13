# Continuous-ridge results

The replacement candidate recovers the complete proposed shapes of **34/34 known paired spaces**, compared with the prior faint-wall method's **14/34**. It also corrects the previously failed real-opening control. The method remains **rejected for count admission**: it withholds 28 pair counts and four known soft-content counts, and proposed boundaries falsely divide those four soft sheets. Improved proposals are not established anatomical counts.

## Measured behavior

| Group | Cases | Prior faint-wall spatial recovery | Ridge proposal recovery | Ridge accepted recovery |
| --- | ---: | ---: | ---: | ---: |
| Known pairs | 34 | 14 | 34 | 6 |
| Known one-space controls | 22 | 21 | 18 | 18 |
| Fresh opening/fragment controls | 4 | Not rerun | 4 | 4 |

All eight fresh material-twin cases withhold their counts and produce identical inference within each identical-input pair, as required. Their opposite physical identities cannot be determined from the deliberately identical observed HU. This is an ideal averaging counterexample, not a measurement of clinical ambiguity frequency.

Across all 68 cases, 28 experimental counts are available and 40 are withheld. Every available count meets the frozen full spatial recovery test. There are zero accepted false control splits and zero unexpected execution errors. These are outcomes on generated inputs, not estimates of clinical accuracy.

The 20 pairs identified in the earlier diagnosis now have two spatially correct proposals. Recovery still requires the exact total, complete one-to-one matching, Dice of at least 0.90, at least 95% reference-interior coverage, connected shapes and correct physical side assignment. This is stronger than merely generating two labels.

The four known soft sheets also produce two proposals, although their physical reference is one space. All four counts are withheld. The corresponding four prior spatial successes therefore regress at the proposal level. Eight previously successful pair counts also become withheld, making twelve regressions at the accepted-count level. Both changes remain in the reported denominators.

The previously failed perforated-divider control now has one complete proposal and one available experimental count. All known and fresh declared openings and isolated-fragment controls meet their count and shape requirements. Unit validation also covers a genuine opening separately from the historical baseline test that required two proposals across one.

## Admission remains rejected

The [frozen rules](SYNTHETIC_RIDGE_EXPERIMENT.md) require improved accepted pair recovery, preservation of every prior success and all known controls recovered. Those three gates fail. Complete coverage, all four fresh controls, all eight identical-input abstentions, absence of accepted false control splits, execution integrity and frozen-input preservation pass. No threshold, acceptance rule or fixture was changed after measurement.

The candidate is useful as a proposal-generation prototype with explicit uncertainty. It is not a promoted automatic counter. The remaining task is to evaluate material and opening evidence that can support a count without splitting soft contents. The matched-HU counterexamples require continued abstention wherever the detector has no distinguishing evidence; a future method must preserve that limitation rather than learn their case identities.

## Implementation and verification

`ridge_counting_candidate.py` generates continuous-HU minima and watershed proposals, contracts original basin edges with resolved physical passages, and checks strong-bone support and geometric acceptance. It preserves original-HU air measurements and physical centroid IDs. Material ambiguity, unresolved communication and rejected geometric proposals withhold the total. Production code, its defaults and the old frozen experiment are unchanged.

The compact peer review caught a real merge-order defect before freezing: recomputing intensity percentiles on merged masks let an array reflection change whether a count was available. The three-basin regression failed before fixing edge evidence to the original partition and passed afterward. No evaluation case had run before that correction.

**237 dedicated counting tests passed without skips**, including 51 new candidate, fixture and runner tests. They cover malformed inputs/results, immutable provenance, complete failure denominators, physical reindexing, original-HU measurements, material-twin equality and accepted-versus-proposal scoring. The complete tracked-file medical-data guard passed. The installed scikit-image code emits an existing NumPy shape-assignment deprecation warning; it did not fail validation.

Both fresh process executions used source commit `9e6644095e40abc4cf90c6f8f0f3197b4ae21f37`, one fixed plan and identical inputs/runtime. Their experiment outputs are under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/continuous-ridge-v1/2026-10-13-01`, in `run-01` and `run-02`. Each retains the complete report, all 68 input sets, partition/cavity/air arrays, registries, scores, withheld reasons, protocol and file manifest. The parent directory retains the frozen plan, source bundles and an illustrative comparison image. The renderer reads saved arrays only; its slices are illustrations, not a substitute for full-volume scoring.

An independent saved-artifact audit recomputed all 180 spatial scores without importing the project scorer and confirmed byte-for-byte repeatability across all 208 files in each run. It verified the reported denominators, withheld counts, admission decision, input integrity and unchanged historical source. Both [GitHub Actions jobs at the measured source](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/actions/runs/38380419583) passed, including the broader application checks.

The first audit retained one packaging failure: `git archive` normalized the scorer's final blank line, so that archived file differed from the exact measured checkout bytes. Its Python statements were unchanged. The original archive and audit remain preserved; the additional `measured-worktree-source.zip` contains the exact frozen source and rules bytes, plus separately identified validation files. Packaging correction did not change the detector, inputs, rules or measured outputs.

All 56 old cases are development evidence. The twelve new cases are six constructions and their exact physical reindexings; they are not twelve independent scans. Supplied regions bypass localization. Patient data, clinical reference masks, scanner calibration and clinical accuracy were not tested. The one-candidate evaluation budget ends with these results.
