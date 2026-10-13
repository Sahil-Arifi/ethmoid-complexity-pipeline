# Native passage evidence: results

The added measurement does not justify changing count acceptance. It detects some real openings, but blurred sealed walls also admit low-intensity paths. The fixed batch exposes another failure in the unchanged ridge counter: four physical opening constructions, each repeated under exact reindexing, receive an incorrect accepted count of two. Automatic readiness remains unmet.

## What was measured

The [frozen design](NATIVE_PASSAGE_EXPERIMENT.md) binds source commit `e974e13`, dependencies, rules and input hashes. Two separate processes each completed all 100 records without execution errors. The 68 historical records reuse their saved partitions; the unchanged detector supplies partitions for 32 new records. The new extractor measures native-HU bottlenecks, voxel-center clearance and contact evidence between touching proposals. It receives no reference count, construction label or core.

| Record group | All records | Physical sealed pairs | Physical one-space records | Records with one measured contacting pair | Records with no contacting pair |
|---|---:|---:|---:|---:|---:|
| Known development | 68 | 38 | 30 | 40 | 28 |
| New perturbations | 32 | 16 | 16 | 20 | 12 |

The 32 new records represent 16 physical constructions and their exact X/Z swap plus Y reflection. They are not 32 independent examples. None of the 100 records is a clinical validation case. No record had multiple contacting pairs or an unavailable terminal measurement in this batch.

## Fixed threshold diagnostics

These are counterfactual connection flags, not accepted counts or selected settings. Among the 40 eligible historical records, eight physically contain one space and 32 contain a sealed pair. Among the 20 eligible new records, eight contain one space and twelve contain a sealed pair. The other 40 records remain in the census above.

| Native bottleneck at or below (HU) | Historical one-space flags / 8 | Historical false sealed-pair flags / 32 | New one-space flags / 8 | New false sealed-pair flags / 12 |
|---:|---:|---:|---:|---:|
| -900 | 0 | 0 | 2 | 0 |
| -800 | 0 | 0 | 2 | 0 |
| -700 | 0 | 0 | 4 | 0 |
| -500 | 0 | 2 | 4 | 2 |
| 0 | 3 | 20 | 4 | 12 |
| 199 | 8 | 32 | 8 | 12 |

For example, a blurred sealed 0.125 mm wall has a native bottleneck of about -682.31 HU, despite physically separating two spaces. A blurred 1.5 mm square opening has a bottleneck of about -735.34 HU. A blurred 0.5 mm opening needs about +152.70 HU. These examples illustrate overlap and sensitivity to image formation; they do not define a clinically safe intensity cutoff. Voxel-center clearance is not the physical opening width.

The eight historical material-twin records still produce identical evidence within each exact-input pair despite opposite physical references. Image-identical bone and soft-content constructions remain ambiguous to this measurement.

## False acceptances in the unchanged counter

The new records produce twelve accepted counts: four correct sealed-pair records and eight incorrect counts of two for physically connected spaces. Twenty records withhold their counts. All eight incorrect records have no contacting proposal pair, so the new pair-level extractor produces no passage measurement for them.

| Physical construction, all with a 1 mm wall | Blur sigma (mm) | X sampling phase (mm) | Incorrect records including reindexing |
|---|---:|---:|---:|
| 0.25 mm square opening | 0 | 0 | 2 |
| 0.25 mm square opening | 0.3 | 0 | 2 |
| 0.5 mm square opening | 0 | 0 | 2 |
| 0.5 mm square opening | 0.3 | 0.125 | 2 |

The reference openings are connected on the independent 0.125 mm physical grid. After averaging to 0.5 mm voxels, they fail to form a connected non-bone path under the counter's fixed 200 HU cutoff. This does not mean all image information has disappeared. At zero blur, the 0.25 mm and 0.5 mm openings leave wall voxels at 693.75 and 375 HU respectively, compared with 800 HU for sealed bone. The counter still classifies those voxels as bone and excludes them from the path check. An empty pair report therefore cannot establish that a wall is sealed.

## Validation and remaining work

All 278 dedicated counting tests passed without skips. The new tests cover analytic paths, competing corridors, physical constructions, reindexing, input isolation, mutation detection, retained failures and protected output paths. An independent graph-search calculation agreed with the extractor on twelve randomized grids. The tracked-file medical-data guard passed in the application environment.

The independent saved-artifact audit verified all 100 records, all 60 measured pairs' minimum-HU and maximum-clearance proofs, each witness component, complete threshold-table denominators, four exact-input twin groups and all 32 new fine-grid reference geometries. All 15 frozen source hashes and 208 protected historical files matched. All 303 files in each run matched byte for byte across processes.

**The overall audit fails one reindexing requirement.** In the unblurred 0.125 mm sealed-wall construction, exact reindexing changes 109 proposal voxels on the same -475 HU plateau. The maximum physical centroid-coordinate difference is 0.118069 mm; this is a real boundary reassignment, not numerical noise to ignore. Both versions withhold their count, and their bottleneck, clearance, remaining contact/terminal measurements and witness masks agree exactly. The other fifteen transformed pairs have identical corresponding proposals. Saved-output inspection and the installed watershed's queue-order tie handling suggest an ordering-sensitive plateau; the precise contribution of smoothing versus watershed ordering remains unproven because intermediate arrays were not saved. Both failed audit records remain preserved.

The next engineering target is the wall region currently excluded by the bone cutoff, together with orientation-stable proposals. A wall/opening model must retain small-opening alternatives and distinguish unavailable evidence from a closed wall. These results do not support changing a threshold or promoting the current candidate. Clinical targets, independent reference evaluation and full-volume localization remain separate requirements for website readiness.

All plans, saved arrays, witness masks, per-case records, repeatability outputs and audit records remain outside Git under `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/native-passage-v1/2026-10-13-01`. The canonical measurement folders are `run-01` and `run-02`; the frozen design and failed evidence remain preserved. Original patient scans, historical runs, production counting and the application were not changed.
