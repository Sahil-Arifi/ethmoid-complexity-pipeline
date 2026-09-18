# Research completion evaluations

**Latest audited result: research v5 completed 15/98 automatic candidate counts (15.3%), with 83 anatomical review outcomes and no technical errors.** It retained all 12 v3 completions with identical left/right/total counts and added three previously resource-limited completions. All 306 candidate entries remain uncertain. **The requested 98/98 automatic-count target is not met.** Anatomical accuracy remains unmeasured, and no expert target labels or new public target-label dataset were obtained. The application default remains v3; these protected research results have not been imported into production application history.

The historical v4 evidence below is retained unchanged in substance. Its lost completion and changed side counts describe v4, not the subsequent v5 result documented separately.

## Frozen v4 evaluation: historical evidence

The frozen research v4 method produced **15 automatic candidate counts from 98 prepared volumes (15.3%)**, compared with **12/98 (12.2%)** for frozen v3. All 98 attempts finished with a recorded outcome. The remaining **83 required anatomical review**; none was reported as a successful zero-cell count. The independent integrity audit verified the completed batch on 2026-09-18.

This is a development-method comparison, **not an accuracy evaluation**. Four previously unsuccessful volumes completed, but one previously completed volume became an abstention. More available counts do not establish better anatomical counting. The method remains an unvalidated research experiment.

### Frozen methods and scope

| Item | Evaluated evidence |
| --- | --- |
| v4 code | [Frozen commit `ce6a0b2`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/ce6a0b2ddc2b7943fce8a2b849652965eb564403) |
| v4 method / protocol | `research-native-0.4.0` / `ethmoid-count-research-v4` |
| Baseline | [Frozen v3 commit `a1e3ae4`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/a1e3ae4ca6359c73bb4592448f414cc6da554c67), `scan-adaptive-native-0.3.0` |
| Localization | `research_geometric_baseline`; no direct ethmoid model used or promoted |
| Analysis mode | Automatic for all 98 attempts; no assisted results included |
| Catalog | 26 studies, 230 series |
| Prepared and attempted | 98 unique converted sources across 24 studies: 92 native CT and six volumetric MPR series |
| Unavailable | 132 series; two studies had no prepared volume |

V4 tested three engineering changes: a fixed 1 mm sphenoid-localization interior seed, a physical-unit correction to anterior wall-ray extent, and a bounded anatomy-only proxy when the original-resolution anatomy grid exceeded the resource budget. The fallback proxy permits 0.5–1 mm spacing and at most 64 million voxels. Final cavity separation, air classification, cell IDs, measurements and the conservative native cavity parameters remain on original voxels. The v4 algorithm configuration is identical to v3; the changes concern anatomical localization.

All six previously resource-limited MPR volumes reached anatomical analysis through the explicit bounded proxy. Three completed, two withheld counts because of an unresolved sphenoid boundary, and one found no supported cells. This removes those six resource-limit outcomes in this batch; it does not recover detail missing from the acquired images.

All **26 studies are development evidence**. Earlier regression/acceptance partition names remain historical provenance, not independent test-set status. Reconstructions from one patient must stay in the same evaluation group. Independent expert references and new locked studies are still required. Nominal slice thickness remains separate from physical spacing; neither thin nominal thickness nor conversion verification establishes clinical eligibility.

### Final outcomes

| Outcome | Frozen v3 | Research v4 |
| --- | ---: | ---: |
| Automatic candidate count | 12 | 15 |
| Needs review | 80 | 83 |
| Technical error: resource limit | 6 | 0 |
| Total attempted | **98** | **98** |

The 15 completed v4 results contain **316 candidate registry entries, all marked uncertain**, across 11 studies. This sums entries across series, including repeated reconstructions; it is not a unique patient-level cell total. There were no assisted outcomes or expert-referenced accuracy measurements in this batch.

Every withheld result has an explicit reason:

| Reason | v3 | v4 |
| --- | ---: | ---: |
| `SPHENOID_UNRESOLVED` | 57 | 40 |
| `SPHENOID_AMBIGUOUS` | 18 | 25 |
| `INSUFFICIENT_COVERAGE` | 4 | 8 |
| `NO_SUPPORTED_CELLS` | 1 | 10 |
| `RESOURCE_LIMIT` | 6 | 0 |
| Total withheld | **86** | **83** |

The final v4 failure messages distinguish the following mechanisms:

| Recorded mechanism | Attempts |
| --- | ---: |
| Posterior candidate lacks a supported anterior bony wall | 27 |
| Anterior wall rays do not support one coherent curved surface | 21 |
| No sufficiently bone-supported cell candidates | 10 |
| No eligible bone-supported posterior sphenoid compartment | 8 |
| Substantial aerated anatomy remains unassigned | 8 |
| Anterior wall lacks spatially distributed evidence | 5 |
| Competing posterior compartments give incompatible boundaries | 4 |
| Total | **83** |

All 132 unavailable series remain in catalog accounting and were not submitted to counting:

| Preparation reason | Series |
| --- | ---: |
| `PROJECTION` | 78 |
| `SCOUT` | 26 |
| `HU_UNCALIBRATED` | 16 |
| `MIXED_ACQUISITION` | 8 |
| `GEOMETRY_RECONCILIATION_UNRESOLVED` | 4 |
| Total | **132** |

### Paired regression findings

All 98 v4 attempts were paired with v3 attempts using the same converted sources. Of the 12 previous completions, **11 still completed and one now required review**. Four previous failures completed: three resource-limited volumes with nominal 0.6 mm thickness and one sphenoid-unresolved volume with nominal 1.25 mm thickness. The newly withheld result also had nominal 1.25 mm thickness and failed the sphenoid ambiguity check.

Among the 11 volumes completed by both methods:

- Seven retained identical left/right/total counts; four changed their side counts.
- Nine retained their total count; one total decreased by one and another increased by one.
- Two of the unchanged-total pairs shifted one candidate from right to left.
- The paired aggregate stayed at 226 candidates, with left/right totals changing from 118/108 to 119/107.

All 98 v4 attempts reused verified anatomical localizer caches: 92 from the frozen manifest and six generated during the earlier v3 evaluation. No new localizer inference was needed. These comparisons therefore document the consequences of the research processing changes on the available evidence; they do not identify which count is anatomically correct.

Three studies had multiple completed reconstructions. Their within-study total-count ranges were 1, 2 and 12 cells. This is reconstruction repeatability evidence, not agreement with ground truth. The net increase of three completions must be considered alongside the new abstention and changed successful counts.

### Runtime by nominal slice thickness

| Thickness (mm) | Attempts | Candidate counts | Withheld | Mean (s) | Median (s) | Range (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.600 | 7 | 5 | 2 | 105.3 | 130.1 | 28.6–182.0 |
| 0.625 | 44 | 9 | 35 | 48.9 | 41.1 | 24.5–99.2 |
| 1.250 | 13 | 1 | 12 | 18.9 | 17.5 | 12.1–30.1 |
| 2.000 | 5 | 0 | 5 | 28.1 | 31.7 | 8.3–53.7 |
| 2.500 | 29 | 0 | 29 | 11.1 | 9.6 | 6.8–19.7 |

Summed attempt elapsed time was **3,598.325 seconds (60.0 minutes)**, compared with 3,688.710 seconds for v3. These are serial local elapsed times including cache verification, anatomical processing and abstentions. They exclude initial upload/preparation and do not measure uncached localizer latency. Early abstention can shorten a run. These measurements apply to this workstation, frozen runtime, cache state and collection; see the [runtime contract](INFERENCE_RUNTIME.md) and [v3 evaluation](SCAN_ADAPTIVE_EVALUATION.md).

### Integrity and preservation

The independent read-only audit reproduced the final aggregates and verified all 98 unique attempt/source bindings and all 230 catalog entries. It checked the frozen code, protocol, runtime before/after the batch, model/cache provenance, input profiles and plans, logs, results and artifact bindings. It also rechecked the 12 completed baseline results for the paired comparison.

For all **15 completed v4 results**, the auditor checked native mask shape, affine and dtype; source-aligned cell IDs and registry consistency; per-side/total counts; physical centroids, laterality and voxel measurements; and exact air-mask membership against the calibrated source-volume HU threshold. Finished cell IDs were not resampled. This verifies geometry and source-HU consistency, not the anatomical correctness of the selected regions or cell separations.

All **83 withheld results** retained null left/right/total counts, empty public cell registries and no accepted segmentation-artifact links. The eight coverage-abstention attempts retained protected diagnostic proposals for investigation; these were not promoted to completed counts. Historical v3 evidence was preserved and verified as the baseline.

A separate read-only preservation audit rehashed every original file linked to the 98 prepared entries: **13,543 files totaling 7,989,710,984 bytes**. All original-series fingerprints matched the frozen records and previous preservation audit, and catalog source/identity mappings remained unchanged. It used one hashing worker and made no database or original-image writes. The 132 unavailable entries were catalog-checked, not byte-rehashed. This audit **did not decode source pixels again or repeat DICOM-to-volume conversion verification**; original-byte preservation, prior conversion proof and current converted-volume HU checks are separate evidence.

### Diagnostic limits and next experiments

The final diagnostics support investigating both wall-candidate selection and the global surface model, but do not prove either proposed replacement will improve anatomy:

- **Wall selection:** 24 of the 27 supported-wall failures had enough rays encountering bone to meet the raw required count, but too few survived the subsequent orientation/edge checks. Current counters do not distinguish a tilted genuine wall, an incorrect internal bone hit and edge rejection.
- **Surface model:** 21 attempts passed the ray-support gate but failed the global quadratic fit. Residual maps and detailed accepted/rejected hit geometry are absent, so a restrictive fit cannot yet be separated from inconsistent anatomical evidence. Five further attempts lacked spatially distributed wall support; four had competing compartments.
- **Cell acceptance:** eight coverage failures assigned only **0–18.5%** of regional air, below the unchanged 50% requirement. The ten `NO_SUPPORTED_CELLS` attempts have no per-rule rejection breakdown. Size/interior resolution, anatomical membership, enclosure and boundary-contact rejection must be recorded separately before attributing these failures or changing thresholds.

The [research source review](RESEARCH_DATA_SOURCES.md) describes orientation-aware wall proposals and observed-bone-constrained graph/local surfaces. Future engineering tests must reject absent walls, misleading internal septa, competing surfaces, disconnected support and cropped anatomy. Synthetic controls and preserved native geometry can be tested without clinician labels; anatomical accuracy, model promotion and clinical acceptance still require independent expert regions, cell references and per-side/total counts. The reported completion increase does not justify relaxing failed checks.

## Frozen v5 evaluation: latest verified result

Research v5 is frozen at [commit `1c3a3f5`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/1c3a3f5afb0035a2ce26a2da285baad01cf773af), with method `research-native-0.5.0`, protocol `ethmoid-count-research-v5` and processing plan `research-localization-plan-2`. Its full 98-volume evaluation and independent read-only audit are **verified**. The same 26-study, 230-series catalog is accounted for: 98 unique prepared sources across 24 studies and 132 unavailable series. All 26 studies remain development evidence.

V5 preserves any supported legacy interior seed exactly and permits the 1 mm fallback only when that interior is empty on the current anatomical localization grid with maximum spacing above 1 mm. The physical wall-ray correction, bounded anatomy-only resource policy and native cavity checks remain. This is a shared geometry policy, not tuning by study identity or expected count. It is a regression-control experiment, not clinical promotion.

| Outcome | v3 | v4 | v5 |
| --- | ---: | ---: | ---: |
| Automatic candidate count | 12 | 15 | 15 |
| Needs review | 80 | 83 | 83 |
| Technical error: resource limit | 6 | 0 | 0 |
| Total attempted | **98** | **98** | **98** |

V5 contains **306 candidate entries, all uncertain, across 15 completed volumes and 11 studies**. These include repeated reconstructions and are not a unique patient-level cell total. The three newly completed volumes were prior resource-limit cases with nominal 0.6 mm thickness. The other three prior resource-limit cases reached anatomy processing but withheld counts: two unresolved sphenoid boundaries and one unsupported-cell outcome. All 80 v3 anatomical review cases still require review under v5.

### Paired retention and limitations

All **12/12 v3 completed results retain identical left/right/total counts**; none became withheld and none changed its side counts. The paired total remains 236 candidates. A separately verified comparison also found **all 12 cell registries identical and all saved native air, cavity and region artifacts byte-identical** by their nonempty SHA-256 mappings. This establishes retained outputs on these frozen development pairs, not anatomical accuracy or a universal regression guarantee. Three studies have multiple completed v5 reconstructions; their total-count ranges are 2, 5 and 12 cells. Reconstruction agreement or disagreement is not a reference count.

V4 and v5 share 14 completed volumes: v5 restored the v3 success lost by v4, while one v4-only completion now withholds its count with `SPHENOID_UNRESOLVED`. Among those 14 shared completions, ten have identical side counts, three have identical full cell registries and two have identical native artifact mappings. Both methods completed 15 volumes, but their completed subsets and outputs differ; pooled candidate entries changed from 316 to 306. The sealed comparison checked all 98 same-source pairs across the three methods. This distinguishes v5's exact v3 retention from the different historical v4 results.

Every v5 failure retained null counts, an empty public cell registry and no accepted segmentation-artifact links:

| Failure code | Attempts | Final recorded mechanisms |
| --- | ---: | --- |
| `SPHENOID_UNRESOLVED` | 44 | 28 unsupported anterior walls; eight without an eligible posterior compartment; six with spatially concentrated wall evidence; two geometrically degenerate wall fits. |
| `SPHENOID_AMBIGUOUS` | 22 | 18 incompatible global curved-wall fits; four competing posterior compartments. |
| `INSUFFICIENT_COVERAGE` | 9 | Only 0–44.6% of regional air assigned, below the unchanged 50% requirement. |
| `NO_SUPPORTED_CELLS` | 8 | No proposed compartment passed all native acceptance checks; no per-rule rejection breakdown is available. |
| Total withheld | **83** | No failed attempt was converted into a successful zero-cell result. |

The wall bottlenecks remain: 24 of the 28 unsupported-wall attempts had enough raw bone-hit rays, but too few survived the following orientation/edge checks. Diagnostics still cannot distinguish a tilted true wall from an incorrect hit or edge rejection, and detailed surface residuals are absent. These are reasons to investigate the bounded experiments above, not evidence for lowering the acceptance thresholds. No automatic count was obtained for the nominal 2 or 2.5 mm groups.

### V5 runtime and verification

| Thickness (mm) | Attempts | Candidate counts | Withheld | Mean (s) | Median (s) | Range (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.600 | 7 | 5 | 2 | 112.3 | 136.2 | 34.6–182.8 |
| 0.625 | 44 | 9 | 35 | 48.2 | 40.1 | 25.1–96.2 |
| 1.250 | 13 | 1 | 12 | 19.6 | 18.0 | 12.8–41.0 |
| 2.000 | 5 | 0 | 5 | 29.7 | 33.2 | 8.5–56.3 |
| 2.500 | 29 | 0 | 29 | 11.2 | 9.8 | 6.7–18.4 |

Summed v5 attempt time was **3,635.101 seconds (60.6 minutes)**. These serial local timings include verified cache reuse and abstentions, not initial upload/preparation or uncached inference latency. They are not a general runtime guarantee.

The independent v5 audit verified all 98 source/result identities, code/protocol/runtime and cache bindings, frozen profiles/plans, logs and results. It independently checked all 15 successful native outputs and all 12 successful v3 baseline outputs: shape, affine, dtype, registry/IDs, side totals, physical centroids/measurements and exact source-HU air membership. It also checked null-failure behavior and the retained protected proposals for the nine coverage abstentions. These checks verify software consistency and provenance, not expert anatomical agreement or a fresh original-DICOM pixel-decode pass.

A separate final production-preservation comparison verified **all 20 database tables unchanged** from the task-start snapshot. Research outputs remain protected and separate from saved production app histories; no research-result import into production, production restart or opt-in setting change was performed. The earlier original-file preservation evidence remains described above. The unset application default remains v3, and assisted runs remain on v3; v5 is available only through explicit local operator opt-in.

Software and synthetic checks cover the tested geometry, job-recovery, upload and review paths; they do not establish support for every DICOM export or correct counts on every scan. The [public data/model review](RESEARCH_DATA_SOURCES.md) did not produce the required conventional-CT ethmoid/sphenoid expert supervision or individual-cell references. No new target-label dataset or weights were downloaded, and no direct ethmoid model was trained or promoted. Accuracy and clinical acceptance remain pending independent expert labels, patient-grouped evaluation and new locked studies. **Producing correct automatic counts for all 98 volumes has not been demonstrated.**

## Evidence and reproduction

The protected completed v4 and v5 audits, separate final attempt/result archives, frozen summaries and preservation reports are the evidence for this document. Patient identifiers, source paths, source fingerprints and individual examples remain outside Git.

A protected local comparison CSV records all 98 v3/v4/v5 attempt outcomes, reasons and side counts, with blank count fields for the 83 v5 abstentions. Its read-back verification binds it to the audited summaries and results. It is available for local review and is not published in this repository.

- [Research v4 evaluator](../scripts/evaluate_research_catalog.py) freezes inputs and retains every outcome in a new protected output directory.
- [Independent research auditor](../scripts/audit_research_evaluation.py) verifies the finished batch and recomputes paired aggregates without inference or database writes.
- [Research v5 evaluator](../scripts/evaluate_research_v5_catalog.py) evaluates the separately frozen opt-in method; it does not overwrite v4.
- [Counting definitions](COUNTING_PROTOTYPE.md), [scan-adaptive contract](SCAN_ADAPTIVE.md) and [data protection rules](../SECURITY.md) describe the retained scope and safeguards.
