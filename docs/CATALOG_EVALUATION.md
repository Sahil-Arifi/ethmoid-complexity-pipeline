# Frozen all-series evaluation

The v2 evaluation measures completion, abstention, native-coordinate consistency
and runtime across the configured collection. It does not measure anatomical
accuracy: independent expert 3D reference counts are still unavailable.

Preparation and evaluation are separate. Catalog startup prepares images only;
the application runs inference after Run is clicked. The engineering evaluation
below deliberately invokes workers outside the application and retains every
attempt in protected local storage. It does not populate or replace clinical
review histories.

## Freeze and study split

The local collection preparation completed with 26 studies, 230 series and
98 verified volumes (92 native CT and six MPR), across 24 studies. The 132
unavailable series comprise 78 projections, 26 scouts, 16 without supported HU
calibration, eight mixed acquisitions and four unresolved MPR grids. All remain
in the frozen collection's coverage accounting.

The current v2 evaluation is in progress using these immutable inputs:

- Code: `23b547893b4514a841df4552e6ef947e559d5335`.
- Method: `native-bone-watershed-0.2.0`.
- Protocol: `ethmoid-count-prototype-v2`, SHA256
  `fcf7b2244737f4b660f40318d6d60b551c31b1ae7c3ed0c88ace2934b29c2652`.
- Protected collection manifest SHA256:
  `165ed694eea5b391da389e8230b00548c852ca5e89c61f68e527d04c6fecc7fc`.
- Regression: 43 prepared series from the ten previously examined studies.
- Acceptance: 55 prepared series from 14 of the remaining sixteen studies;
  two studies have no supported 3D volume.

Regression has finished; acceptance is in progress. A final collection result
requires an immutable attempt for each of the 98 prepared series, including every
abstention or resource-limit outcome. Execution is serial on the local RTX 5070 Ti
with 16 GB VRAM and approximately 32 GB system RAM. No runtime or accuracy claim
is generalized beyond this host and collection.

Finish every preparation job first. `scripts/freeze_catalog_evaluation.py`
snapshots every ready series and every unavailable reason. It verifies the
current raw DICOM fingerprints, linked immutable volume identity, converted
volume hash and engineering validation state. The snapshot is a new protected
file; it must never be committed. Source paths, UIDs, images, masks, result
registries and logs remain local.

The ten previously examined studies are the regression set: Case001, Case002,
Case003, Case004, Case005, Case006, Case009, Case015, Case021 and Case025. All
reconstructions of each study remain in its split. The remaining sixteen
studies are the acceptance set. Unavailable series remain in coverage accounting
even though they cannot enter inference.

Run regression with a committed clean checkout. Resolve software defects before
acceptance, then freeze the method, complete algorithm settings, protocol, source
manifest, runtime and model provenance. The acceptance run uses that same clean
snapshot. Keep the original v1 evidence unchanged. Agreement between different
reconstructions is repeat-acquisition evidence, never independent ground truth.

## Runner contract

`scripts/evaluate_catalog.py` accepts a separate clean pipeline worktree, the
protected manifest, a fresh protected output directory, the study split and the
expected Git/manifest/protocol hashes. It launches one offline worker at a time
using the isolated inference runtime. Complete logs and per-attempt records are
retained even when inference abstains. A timeout terminates the worker process
tree and records an unavailable count.

The runner checks source, code, settings, model and cache provenance before and
after each attempt. Historical caches are reused only when their source, native
geometry, masks and producer/model hashes verify. Rejected caches stay unchanged;
the new attempt performs fresh localization instead.

A completed result must have matching acquisition facts and recomputed quality,
native uint16 masks, consistent registry IDs, counts, laterality, centroids and
physical measurements. Failed attempts must have unavailable counts, no cells
and no successful segmentation artifacts. Missing sources and invalid results
are retained as failures rather than silently omitted or reported as zero.

## Reporting

### Completed regression gate

All 43 regression series have a unique immutable attempt: **five completed and
38 withheld** (11.6% completion by series). Summed attempt time was **4,100.990
seconds**. The withheld reasons were 22 unresolved sphenoid boundaries, seven
ambiguous sphenoid boundaries, six anatomy-proxy resource limits, two insufficient
coverage outcomes and one absence of supported cells. No unexpected runtime or
evaluation-integrity failure occurred. Failed attempts contain no numeric count.

| Nominal thickness (mm) | Attempted | Completed | Withheld | Summed attempt seconds |
| --- | ---: | ---: | ---: | ---: |
| 0.6 | 7 | 2 | 5 | 614.635 |
| 0.625 | 16 | 3 | 13 | 1,817.110 |
| 1.25 | 6 | 0 | 6 | 662.165 |
| 2 | 5 | 0 | 5 | 206.301 |
| 2.5 | 9 | 0 | 9 | 800.779 |

All ten matching original-cohort series retained their v1 outcomes: five completed
and five identical anatomical abstentions. Sources and algorithm values matched.
All five successful current artifact sets independently passed native geometry,
instance/registry consistency, centroids, laterality and exact native-HU air-mask
classification. All ten required localizer-mask sets passed hash, binary-value
and native-grid checks. Five identical verified caches reproduced identical
behavior; five unverified legacy caches were replaced with verified outputs.
Three successful boundary/registry comparisons changed alongside localizer changes,
but no count outcome changed. Incomplete legacy producer evidence prevents precise
attribution of those differences. No unexpected fixed-input regression was found.

The protected comparison report SHA256 is
`0dc1cba8b08c2ff23004b40d6dda29ee769883a96eb140f740d466d45540acef`.
Neither these comparisons nor the 107 flagged uncertain regression candidates are
expert reference labels. Broader acquisition support did not produce a completed
thick-slice count in this regression batch; accepting an input is not evidence that
lost septa or cells were recovered. Acceptance outcomes are still pending.

Report the exact number of cataloged, prepared, attempted, completed and withheld
series, unique studies, reasons for unavailability/abstention, uncertainty and
runtime by nominal thickness. Keep nominal slice thickness, source slice interval
and native voxel spacing distinct. Thin reconstructed voxels do not prove that
thin septa survived acquisition. All candidate totals remain experimental until
independent manual 3D counts and expert boundary review establish accuracy.

Synthetic runner tests cover freeze mismatch, invalid output, source/code/config
drift, unknown acquisition thickness, wrong quality metadata, partial collection
failure, immutable output, timeout process cleanup and raw-source changes. The
existing v1 runner and its evidence are retained unchanged.
