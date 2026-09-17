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
