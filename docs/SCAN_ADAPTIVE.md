# Scan-adaptive local labeling and counting

## Scope and interpretation

This increment extends the local React/TypeScript/NiiVue and FastAPI application. The app runtime remains Python 3.13; image inference remains in the isolated, pinned Python 3.12 PyTorch/TotalSegmentator runtime with its offline network guard. SQLite and protected local files hold resumable work and immutable history. There is no hosting, patient-data upload to a third party, account system or billing.

Every submission must receive a visible outcome. Execution state (queued/running/completed/failed) is distinct from `automatic_candidate_count`, `assisted_candidate_count`, `needs_review`, `unsupported_input`, `technical_error` and `cancelled`. An unresolved anatomical count remains unavailable, never zero. An automatic completion is not an accuracy measurement.

## Durable counting phase

Schema 4 adds count-job ownership, heartbeats, attempt/retry counters, checkpoints and anatomical-boundary revisions. The additive migration backs up earlier databases. An OS lock permits one local counting owner; a second server cannot steal a live job based on a stale heartbeat. Windows inference starts suspended, enters a kill-on-close Job Object, then resumes; its descendants terminate if the server dies abruptly. Restart resolves pending cancellations and recovers previously authorized unfinished runs only. It does not launch new counts on the existing catalog.

Each attempt writes into a new directory. After source, frozen protocol, output hashes, native geometry, cell measurements and exact HU air classification pass verification, publication renames the directory atomically. A restart can reuse only a hash-verified checkpoint. Historical runs are not rewritten. Transient processing failures receive at most two retries; repeating the same anatomical abstention is not a retry policy. Cancellation terminates the tracked inference process tree and keeps a nonnumeric outcome.

New APIs retain the existing case/count-run routes and add `cancel`, `retry`, `boundaries`, `boundaries/{id}/mask`, and `assisted-runs` below `/api/cases/{case_id}/count-runs/{run_id}`. Boundary masks use the original source grid and uint8 labels 0 (background), 1 (left ethmoid membership), 2 (right ethmoid membership). The saved revision is bound to source hash, exact affine, mask hash and parent run. Assisted counting preserves the automatic parent and exports explicit assistance lineage. Reviewers remain responsible for anatomical membership, including protocol variants and neighboring-sinus exclusion.

## Adaptive method and model contract

The method is `scan-adaptive-native-0.3.0`, protocol `ethmoid-count-prototype-v3`. V1 and V2 remain readable and selectable by their existing worker contracts. A sealed scan profile records native shape/affine, actual spacing, nominal thickness, normalized reconstruction categories, physical coverage and descriptive source-HU samples. Samples are not advertised as validated artifact/noise diagnosis.

Shared policies derive a sealed processing plan from those facts. No policy keys on case names, patient identity or free-text scanner descriptions. Thin, thick, unknown and reconstructed acquisitions stay distinguishable. Thickness and slice interval remain separate, as specified in the [DICOM image-plane standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html). The clinician's at-most-1-mm study criterion remains visible on thicker experimental attempts.

For oversized native anatomy or oblique anatomy proxies, verified coarse anatomical landmarks define a bounded region with progressively larger physical margins. Artificial crop boundaries are checked before acceptance. The same 128-million-voxel anatomy limit remains; unresolved crop/resource constraints are explicit. CT interpolation is limited to anatomy localization, region mapping uses nearest neighbors, and final cell IDs are never resampled. The v2 cavity rules and 1.01-mm enclosure-support cap remain unchanged.

Without promoted weights, the backend clearly identifies `v2_geometric_baseline`. A direct nnU-Net adapter requires a versioned manifest, exact weight/config hashes, native output alignment, defined left/right ethmoid and sphenoid labels, and recorded expert-reference promotion evidence. A manifest flag does not replace clinical review. The implementation includes expert-source validation, dataset export, patient-grouped folds and reference-based count/instance error utilities; no training has occurred. All reconstructions from a patient share an evaluation group. See the [official nnU-Net dataset contract](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

## Verification and remaining work

Core tests cover native flipped/oblique coordinates, known cavities, side conflicts, resource bounds, missing/tampered models, patient-group leakage, crash/restart checkpoint reuse, two competing services, retry exhaustion, disk-full errors, source/boundary drift and preservation of the automatic parent. A real CPU subprocess integration drives assisted counting from the API through export on a synthetic source. It does not use an inference stub.

The separate `scripts/evaluate_adaptive_catalog.py` requires a clean frozen checkout and hash-bound manifest/protocol. It writes immutable protected local evidence, retains every success/abstention/error, verifies outputs against original HU, and accounts for unavailable catalog series. All existing 26 studies are development evidence, including the former v1/v2 held-out partitions. New locked studies and independent expert references are required for an external accuracy claim.

Upload/codec acceptance, browser acceptance and the 98-volume reevaluation will be recorded here as those phases complete. No inference failures are hidden to improve a completion percentage.
