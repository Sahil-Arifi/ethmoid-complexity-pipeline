# Scan-adaptive development evaluation

The frozen v3 method produced automatic candidate counts for **12 of 98 prepared volumes (12.2%)**. Another 80 attempts required anatomical review, and six reached an explicit resource limit. Every prepared volume has a terminal outcome, and every catalog series is accounted for. Automatic completion did not improve over v2. Anatomical accuracy remains unmeasured.

This is an engineering evaluation of the existing development collection. It does not establish clinical accuracy, compatibility with every DICOM export, or a guarantee that every uploaded scan can yield a count.

## Frozen method and collection

| Item | Evaluated snapshot |
| --- | --- |
| Code | `a1e3ae4ca6359c73bb4592448f414cc6da554c67`, clean checkout |
| Method | `scan-adaptive-native-0.3.0` |
| Protocol | `ethmoid-count-prototype-v3` |
| Anatomical localization | `v2_geometric_baseline`; no promoted expert-trained ethmoid model |
| Analysis mode | Automatic for all 98 attempts; no assisted results included |
| Catalog | 26 studies, 230 series |
| Prepared volumes | 98 across 24 studies: 92 native CT and six volumetric MPR series |
| Unavailable series | 132; two studies had no prepared volume |

The evaluator froze the protocol, source manifest, code, runtime, model provenance, scan profiles and processing plans. Later application, upload and review fixes do not retroactively change these results. Original scans and historical v1/v2 evidence remain preserved.

The frozen legacy acquisition records did not carry normalized reconstruction or kernel categories: all 98 saved profiles correctly retain `unknown` for those fields. They do record nominal thickness and physical slice interval separately. The native/MPR inventory above comes from the catalog and geometry evidence, not a retrospective rewrite of those profiles. Later descriptor enrichment improves the metadata available to future application runs; it is not evidence that this frozen batch used the enriched categories.

All 26 studies are development evidence. Previous regression/acceptance partition names are retained for provenance only; none of this collection is now an independent held-out accuracy set. Reconstructions from the same patient must remain in one evaluation group. New locked studies and independent expert references are still required.

## Outcomes and reasons

| Outcome | Volumes | Interpretation |
| --- | ---: | --- |
| Automatic candidate count | 12 | A result passed the defined software consistency checks; clinical correctness is unknown. |
| Needs review | 80 | The method withheld a count because anatomy or supported compartments could not be resolved. |
| Technical error: resource limit | 6 | The anatomy-processing grid could not fit the fixed resource policy. These remain unsuccessful attempts. |
| Total attempted | **98** | No attempt omitted from the denominator. |

The 12 completed results contain **236 candidate registry entries, all flagged uncertain**. This is a sum across series, including repeat reconstructions, not a count of unique cells across patients. There are no expert-referenced automatic or assisted accuracy measurements in this batch.

All 86 withheld counts have an explicit reason:

| Reason | Volumes | Meaning |
| --- | ---: | --- |
| `SPHENOID_UNRESOLVED` | 57 | The posterior boundary with the sphenoid sinus could not be located. |
| `SPHENOID_AMBIGUOUS` | 18 | The posterior boundary was ambiguous. |
| `INSUFFICIENT_COVERAGE` | 4 | Too much aerated anatomy in the proposed region remained unassigned for a supported count. |
| `NO_SUPPORTED_CELLS` | 1 | Proposed compartments did not satisfy the conservative cavity checks; this is not a zero-cell result. |
| `RESOURCE_LIMIT` | 6 | The anatomy-processing grid exceeded the supported resource budget. |

The six resource-limited attempts are the six MPR series: three with nominal 0.6 mm thickness and three with 2 mm thickness. Their oblique anatomy proxies use the smallest native spacing, approximately 0.203–0.220 mm isotropic. Full grids require 703.7–949.3 million voxels. Landmark crops with 12 mm margins still require 161.8–316.1 million voxels; 24 mm margins require 256.3–429.2 million. All 12 bounded crop attempts were rejected before allocation because they exceeded the fixed **128-million-voxel** limit. The resource constraint therefore remains verified and explicit; the evaluation did not coarsen source detail to force a result.

The 132 unavailable catalog series were retained in coverage accounting and did not enter counting:

| Preparation reason | Series |
| --- | ---: |
| `PROJECTION` | 78 |
| `SCOUT` | 26 |
| `HU_UNCALIBRATED` | 16 |
| `MIXED_ACQUISITION` | 8 |
| `GEOMETRY_RECONCILIATION_UNRESOLVED` | 4 |
| Total unavailable | **132** |

## Runtime by nominal slice thickness

| Thickness (mm) | Attempts | Candidate counts | Withheld | Mean (s) | Median (s) | Range (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.600 | 7 | 2 | 5 | 113.1 | 149.4 | 28.2–191.5 |
| 0.625 | 44 | 9 | 35 | 46.3 | 37.4 | 25.7–99.3 |
| 1.250 | 13 | 1 | 12 | 18.6 | 17.2 | 12.0–40.0 |
| 2.000 | 5 | 0 | 5 | 71.0 | 101.2 | 8.9–124.2 |
| 2.500 | 29 | 0 | 29 | 9.0 | 8.8 | 7.1–13.0 |

Summed per-attempt elapsed time was **3,688.710 seconds (61.5 minutes)**. These are serial local attempt times, including cache verification and abstentions, not end-to-end upload latency or uncached inference benchmarks. **92 attempts reused source- and model-verified localizer caches**. Shorter thick-slice times often reflect earlier abstention, not faster successful anatomical analysis. Thickness is the nominal DICOM property; it must not be substituted for physical voxel spacing.

The tested workstation has an RTX 5070 Ti with 16 GiB VRAM and approximately 32 GB system RAM. The frozen isolated runtime used Python 3.12.14, PyTorch 2.14.0+cu130, TotalSegmentator 2.18.0 and nnU-Net v2 2.8.1. These timings apply to this host, cache state and collection. The installed nnU-Net framework and surrounding-anatomy localizers do not supply a validated ethmoid-cell counter. See [the runtime contract](INFERENCE_RUNTIME.md).

## Independent integrity audit

The independent read-only auditor verified the completed evaluation and reproduced its aggregates. It checked all 98 unique source/result bindings, all 230 catalog entries, the frozen code/runtime/protocol, profiles and plans, and source, cache, log and result artifacts. For all 12 completed results it rechecked native mask shape, affine and dtype; registry IDs; per-side and total counts; physical centroids, laterality and measurements; and exact air membership against the calibrated CT voxels under the frozen HU threshold. Finished instance IDs were not resampled.

All 86 unsuccessful results retained unavailable/null counts, empty public cell registries and no accepted segmentation-artifact links. Four coverage-abstention attempts retained protected rejected proposals for investigation; those proposals were not promoted to successful counts.

A separate final raw-source audit rehashed the original DICOM files linked to all 98 ready entries: **13,543 files totaling 7,989,710,984 bytes**. Their fingerprints matched the frozen source records with no reported discrepancies. This audit opened the database read-only, used one hashing worker and started no inference. It **did not perform a new source-pixel decode or repeat full DICOM-to-volume conversion verification**. The calibrated converted-volume checks above and the original preparation proof are distinct from this final byte-integrity check.

These checks establish consistency and provenance of the stored evidence. They do not establish that the proposed regions, cell splits or total counts match expert anatomy.

## Comparison with v2 and remaining work

All 98 attempts had comparable v2 results from the same converted source. Every status, failure reason and successful total count was unchanged; automatic completion remained **12/98**. Side counts also matched for 97 pairs. One completed result shifted one candidate from right to left, leaving the pooled successful counts at 236: the aggregate left/right distribution changed from 121/115 to 122/114.

The side-count difference was not merely an ID relabeling: some candidate geometry also changed. The paired attempts used different verified localizer outputs despite matching source, algorithm settings, runtime and model weights. This comparison therefore includes localizer repeatability variation; it cannot isolate a method improvement or prove the particular cause of that difference. No anatomical accuracy gain is claimed. The committed counting source and configuration remained unchanged after the evaluated freeze; later application fixes did not rerun or overwrite these outputs.

Expert region masks, individual-cell references and independent left/right/total counts remain the dependency for supervised model training, promotion and accuracy evaluation. More produced counts alone would not justify promotion. Clinical acceptance thresholds must be agreed before locked testing, with count error, missed/false cells, split/merge errors and boundary agreement reported by scan quality and reconstruction.

The assisted review workflow is available for difficult scans, but no assisted recovery rate is inferred from this automatic batch. The six resource constraints and 80 anatomical abstentions remain visible engineering limitations of the evaluated method.

## Evidence and reproduction

The protected local evaluation contains immutable attempt records, frozen summary/provenance, the independent aggregate audit and the final raw-source audit. Patient data, local source paths, identities, source fingerprints and individual examples are intentionally excluded from this report.

- [Freeze the catalog](../scripts/freeze_catalog_evaluation.py) to account for prepared and unavailable series and bind their source evidence.
- [Run the frozen adaptive evaluation](../scripts/evaluate_adaptive_catalog.py) from a clean checkout with the expected code, protocol and manifest bindings; retain every terminal attempt in a fresh protected output directory.
- [Audit the completed evaluation](../scripts/audit_adaptive_evaluation.py) independently, including optional same-source v2 comparison. The audit does not launch inference or modify the database.
- Read the [scan-adaptive implementation contract](SCAN_ADAPTIVE.md), [prior v2 evaluation](CATALOG_EVALUATION.md), [counting definitions](COUNTING_PROTOTYPE.md) and [data protection rules](../SECURITY.md) for scope and limitations.
