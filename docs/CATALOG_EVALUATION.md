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

The completed v2 evaluation used these immutable inputs:

- Code: `23b547893b4514a841df4552e6ef947e559d5335`.
- Method: `native-bone-watershed-0.2.0`.
- Protocol: `ethmoid-count-prototype-v2`, SHA256
  `fcf7b2244737f4b660f40318d6d60b551c31b1ae7c3ed0c88ace2934b29c2652`.
- Protected collection manifest SHA256:
  `165ed694eea5b391da389e8230b00548c852ca5e89c61f68e527d04c6fecc7fc`.
- Regression: 43 prepared series from the ten previously examined studies.
- Acceptance: 55 prepared series from 14 of the remaining sixteen studies;
  two studies have no supported 3D volume.

Both splits finished with an immutable attempt for each of the 98 prepared series,
including every abstention or resource-limit outcome. Evaluator execution was
serial on the local RTX 5070 Ti
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
and no public segmentation-artifact links. Rejected coverage proposals may remain
in protected diagnostic files with `clinical_count_available: false`; they cannot
be downloaded as completed count results. Missing sources and invalid results
are retained as failures rather than silently omitted or reported as zero.

## Reporting

### Final collection outcome

All **230 catalog series** have a recorded outcome: **98 prepared and evaluated**,
and **132 unavailable** with the preparation reasons above. The final integrity
audit verified unique split coverage, current volume hashes, immutable result/log
and successful artifact hashes, clean frozen code/protocol, acquisition/settings
bindings, and null counts with no public artifact links on every failed attempt.
No unexpected runtime or evaluation-integrity failure occurred.

| Split | Attempted series | Completed | Withheld | Series completion | Studies with a completed count |
| --- | ---: | ---: | ---: | ---: | --- |
| Regression | 43 | 5 | 38 | 11.6% | 5 of 10 |
| Acceptance | 55 | 7 | 48 | 12.7% | 6 of 14 prepared; 6 of 16 cataloged |
| Combined | 98 | 12 | 86 | 12.2% | 11 of 24 prepared; 11 of 26 cataloged |

The 86 abstentions comprise **57 unresolved sphenoid boundaries, 18 ambiguous
sphenoid boundaries, six anatomy-proxy resource limits, four insufficient-coverage
outcomes and one absence of supported cells**. Every failed result has unavailable
counts rather than zero. Four coverage failures retain rejected proposals only in
protected diagnostic storage. Independent live API checks returned 409 for all
16 artifact/export requests across the application's four failed runs.

All **236 candidate instances across reconstructions** carried uncertainty flags.
This number pools reconstructions, not distinct anatomical cells across people.
There are no independent expert 3D counts, measured clinical accuracy, calibrated
confidence estimates or evidence that thin details lost at acquisition were
recovered. One 1.25 mm attempt completed; none of the 2 or 2.5 mm attempts completed.
Within-protocol acquisitions completed 11 of 48 attempts; broader experimental
acquisitions completed one of 50. Acquisition classification is not clinical
eligibility or proof of count completeness.

| Nominal thickness (mm) | Attempted | Completed | Withheld | Mean seconds | Median seconds | Range seconds |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.6 | 7 | 2 | 5 | 87.8 | 121.3 | 2.6–174.2 |
| 0.625 | 44 | 9 | 35 | 134.1 | 134.3 | 29.9–194.7 |
| 1.25 | 13 | 1 | 12 | 111.9 | 107.7 | 104.7–136.0 |
| 2 | 5 | 0 | 5 | 41.3 | 2.4 | 2.0–104.0 |
| 2.5 | 29 | 0 | 29 | 97.6 | 99.7 | 11.0–120.3 |

Summed per-attempt time was **11,006.571 seconds** (about 3.06 hours): 4,100.990
regression seconds and 6,905.581 acceptance seconds. These timings include fast
abstentions, verified cache reuse and normal application/engineering activity on
this host. They are not an isolated hardware benchmark or a latency guarantee.
Slice thickness, physical slice interval and native voxel spacing remain separate
fields in every protected attempt.

All seven successful acceptance outputs passed independent native geometry,
uint16 ID/registry, voxel count, centroid/laterality, exact HU-air classification,
source/acquisition/quality/protocol/config and physical-measurement checks.
Physical volumes were recomputed from native voxel counts and the affine
determinant. The first-five and final-two audit reports form an exact disjoint
cover of the seven successful attempt/result hashes. The original ten-series
regression comparison is detailed below.

Protected audit report SHA256 values:

- Final coverage and integrity:
  `939d799e48a0a27338a252e557cd5bbb1ca981ce72e028f18e31d1ef67bbcfb8`.
- First five successful acceptance outputs:
  `035b4c3f9ba00327b7a026b09bb34c8a76363822d7d1f1624e702d0fbba872fc`.
- Final two successful acceptance outputs:
  `04d122bba92da37617aa152a06b3cd5002088ac089e13f3dea6cb21e2b9889de`.

All scan data, result registries, masks, source identities and detailed reports
remain local. The 98 evaluation attempts did not populate application histories.
The original imports, annotation revisions and count runs remain preserved. A
separate live application request during evaluation is retained with its normal
history; the source of that UI/API action was not inferred from logs.

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
lost septa or cells were recovered. The completed acceptance findings above use
the same frozen method without scan-specific tuning.

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
