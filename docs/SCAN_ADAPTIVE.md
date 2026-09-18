# Scan-adaptive local labeling and counting

## Scope and interpretation

This increment extends the local React/TypeScript/NiiVue and FastAPI application. The app runtime remains Python 3.13; image inference remains in the isolated, pinned Python 3.12 PyTorch/TotalSegmentator runtime with its offline network guard. SQLite and protected local files hold resumable work and immutable history. There is no hosting, patient-data upload to a third party, account system or billing.

Every submission must receive a visible outcome. Execution state (queued/running/completed/failed) is distinct from `automatic_candidate_count`, `assisted_candidate_count`, `needs_review`, `unsupported_input`, `technical_error` and `cancelled`. An unresolved anatomical count remains unavailable, never zero. An automatic completion is not an accuracy measurement.

## Durable counting phase

Schema 4 adds count-job ownership, heartbeats, attempt/retry counters, checkpoints and anatomical-boundary revisions. The additive migration backs up earlier databases. An OS lock permits one local counting owner; a second server cannot steal a live job based on a stale heartbeat. Windows inference starts suspended, enters a kill-on-close Job Object, then resumes; its descendants terminate if the server dies abruptly. Restart resolves pending cancellations and recovers previously authorized unfinished runs only. It does not launch new counts on the existing catalog.

Each attempt writes into a new directory. After source, frozen protocol, output hashes, native geometry, cell measurements and exact HU air classification pass verification, publication renames the directory atomically. A restart can reuse only a hash-verified checkpoint. Historical runs are not rewritten. Transient processing failures receive at most two retries; repeating the same anatomical abstention is not a retry policy. Cancellation terminates the tracked inference process tree and keeps a nonnumeric outcome.

New APIs retain the existing case/count-run routes and add `cancel`, `retry`, `boundaries`, `boundaries/{id}/mask`, and `assisted-runs` below `/api/cases/{case_id}/count-runs/{run_id}`. Boundary masks use the original source grid and uint8 labels 0 (background), 1 (left ethmoid membership), 2 (right ethmoid membership). The saved revision is bound to source hash, exact affine, mask hash and parent run. Assisted counting preserves the automatic parent and exports explicit assistance lineage. Reviewers remain responsible for anatomical membership, including protocol variants and neighboring-sinus exclusion.

`GET .../boundaries` returns saved revisions newest first, including when the automatic parent could not localize anatomy. Listing and opening a revision check its current source and parent identity; serving its mask also verifies the stored hash. This makes a saved review discoverable after reload before an assisted run has been submitted.

The viewer restores the latest saved boundary and offers older revisions without overwriting them. Unsaved changes retain navigation protection. The "Cells needing review" overlay filters existing uncertainty flags in a display copy, leaving native IDs, measurements and totals unchanged. Retry requests keep their idempotency key after an unacknowledged response.

## Adaptive method and model contract

The method is `scan-adaptive-native-0.3.0`, protocol `ethmoid-count-prototype-v3`. V1 and V2 remain readable and selectable by their existing worker contracts. A sealed scan profile records native shape/affine, actual spacing, nominal thickness, normalized reconstruction categories, physical coverage and descriptive source-HU samples. Samples are not advertised as validated artifact/noise diagnosis.

Shared policies derive a sealed processing plan from those facts. No policy keys on case names, patient identity or free-text scanner descriptions. Thin, thick, unknown and reconstructed acquisitions stay distinguishable. Thickness and slice interval remain separate, as specified in the [DICOM image-plane standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html). The clinician's at-most-1-mm study criterion remains visible on thicker experimental attempts.

For oversized native anatomy or oblique anatomy proxies, verified coarse anatomical landmarks define a bounded region with progressively larger physical margins. Artificial crop boundaries are checked before acceptance. The same 128-million-voxel anatomy limit remains; unresolved crop/resource constraints are explicit. CT interpolation is limited to anatomy localization, region mapping uses nearest neighbors, and final cell IDs are never resampled. The v2 cavity rules and 1.01-mm enclosure-support cap remain unchanged.

Without promoted weights, the backend clearly identifies `v2_geometric_baseline`. A direct nnU-Net adapter requires a versioned manifest, exact weight/config hashes, native output alignment, defined left/right ethmoid and sphenoid labels, and recorded expert-reference promotion evidence. A manifest flag does not replace clinical review. The implementation includes expert-source validation, dataset export, patient-grouped folds and reference-based count/instance error utilities; no training has occurred. All reconstructions from a patient share an evaluation group. See the [official nnU-Net dataset contract](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

## Verification and remaining work

Core tests cover native flipped/oblique coordinates, known cavities, side conflicts, resource bounds, missing/tampered models, patient-group leakage, crash/restart checkpoint reuse, two competing services, retry exhaustion, disk-full errors, source/boundary drift and preservation of the automatic parent. A real CPU subprocess integration drives assisted counting from the API through export on a synthetic source. It does not use an inference stub.

The separate `scripts/evaluate_adaptive_catalog.py` requires a clean frozen checkout and hash-bound manifest/protocol. It writes immutable protected local evidence, retains every success/abstention/error, verifies outputs against original HU, and accounts for unavailable catalog series. All existing 26 studies are development evidence, including the former v1/v2 held-out partitions. New locked studies and independent expert references are required for an external accuracy claim.

`scripts/audit_adaptive_evaluation.py` independently checks the completed evaluation. It refuses unfinished batches before reading large artifacts, requires all 98 unique attempts and all 230 catalog entries, rechecks frozen code/runtime/profile/plan bindings and source/result/cache hashes, recomputes native measurements and HU air membership, and verifies failed outcomes retain null counts. Its optional v2 comparison measures repeatability only. It verifies current converted sources; raw DICOM verification is the prerequisite recorded by the frozen catalog. The auditor's 23 synthetic tests passed, including deliberately changed artifacts, sources, summaries, geometry and failed-count exposure.

Upload/codec acceptance is recorded below. Frontend acceptance passed 50 unit tests, a production build and five browser workflows, including interrupted upload/reselection and API/viewer assisted review through reload/export. The upload browser transport is simulated; the assisted browser scenario uses synthetic inference. The separate CPU subprocess test uses the actual algorithm. The 98-volume reevaluation is running from frozen code `a1e3ae4`; results remain pending. No inference failures are hidden to improve a completion percentage.

## Complete uploads and DICOM compatibility

The new `/api/uploads` workflow keeps every submitted original in protected local storage outside the processing root and Git. Folder selection and ZIP archives enter the same persistent study/series catalog. Exact study/series identities and source-file hash multisets determine reuse; filenames and case labels do not. Additional series of an existing study retain their own source roots. Conflicting identities preserve the existing volume/history and expose a review reason. Ancillary and unreadable members are individually accounted for.

The initial limits are 20 GiB expanded data, 50,000 files, and 8 MiB per transfer chunk. ZIP extraction accepts stored/deflated entries, rejects traversal, absolute paths, links, encrypted/unsupported entries and expansion over quota, and verifies CRC while streaming. Originals survive interrupted indexing and failed extraction. The local worker owns an OS lock, resumes finalized uploads after restart, and survives temporary database errors. An unavailable configured collection does not disable new uploads.

| API | Contract |
| --- | --- |
| `POST /api/uploads` | JSON `{files:[{name,size}]}` creates a durable session and returns `upload_id`, status, per-file offsets, study outcomes, issues and configured limits. |
| `PUT /api/uploads/{id}/files/{index}?offset=N` | Raw bytes, at most 8 MiB. Returns the committed `received_bytes`. A repeated completed prefix is accepted only if every byte matches; mismatched reselection is rejected. A zero-length source needs one empty PUT. |
| `GET /api/uploads/{id}` | Saved transfer offsets, indexing/preparation state, neutral study/series navigation, count-run identity and fixed actionable reasons. |
| `POST /api/uploads/{id}/finalize` | Idempotent finalization, accepted only after all file hashes are complete. An explicit repeat retries transient indexing/storage failures; unsafe archives remain explained terminal outcomes. |

Finalization authorizes preparation and one automatic analysis intent per uploaded study. Inference waits for verified geometry and HU; a stable hash-based idempotency key bridges a crash between count creation and upload-intent publication. Repeated identical uploads do not silently produce new runs. Session `completed` means ingestion/preparation and scheduling have finished; `analysis_status` separately reports the count job. Other series remain available for explicit Run. Count scheduling errors are visible and do not abandon unrelated uploads.

An interrupted or mismatched upload does not block another submission. The dialog can start a separate upload and return to earlier sessions; it stores only opaque session IDs in browser history and retains originals on the server. Switching between two incomplete folder sessions preserves directory-selection behavior.

Classic CT and Enhanced CT share physical-position validation. Enhanced functional groups supply frame-specific position, orientation, pixel measures and HU scaling; missing/conflicting groups, nonspatial acquisitions and duplicate planes fail explicitly. A private classic-frame adapter feeds the pinned converter while originals, original SOP identities, frame indices and hashes remain in provenance. Frame order is established physically before converter staging. Output geometry and **every calibrated source voxel** are checked independently before a volume becomes ready.

The application pins `pylibjpeg==2.1.0`, `pylibjpeg-libjpeg==2.4.0`, `pylibjpeg-openjpeg==2.5.0` and `pyjpegls==1.5.1` in addition to `pydicom==3.0.1`. Accepted decoding includes uncompressed transfer syntaxes, native RLE, lossless JPEG, JPEG-LS and JPEG 2000. Lossy transfer syntaxes and a recorded lossy history are rejected. Compressed inference inputs use the selected lossless decoder, never an implicit Pillow fallback. See [pydicom's decoder guidance](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html) and [the DICOM Enhanced CT functional-group specification](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.15.3.html).

### Executed upload/codec verification

The targeted upload, frame, catalog, conversion and geometry suite passed **139 tests** before the additional legacy-signature regression test. Synthetic lossless decode fixtures cover implicit/explicit uncompressed, RLE, JPEG process 14/SV1, JPEG-LS and JPEG 2000. Real pinned dcm2niix integration covers five lossless encoding paths and shuffled Enhanced CT with varying frame calibration: all reported **zero maximum HU error**, with unchanged originals. Tests also exercised partial transfer recovery, byte-equal prefix replay, concurrent retries/finalization, malformed/unsafe archives, expansion quota, storage exhaustion, worker ownership/restart, mixed studies, exact reuploads, identity conflicts, source-root coexistence and the actual HTTP contract. These are software and conversion checks, not anatomical accuracy measurements.

A protected **copy** of the production schema-3 database was migrated to schema 4, then synchronized twice against the read-only source inventory. One migration backup was created. All rows and content hashes remained identical in 98 cases, 28 annotation revisions, 2 imports, 109 count runs, 98 case links, 230 catalog series and 26 catalog studies; there were no cell revisions. The catalog retained 98 ready and 132 unavailable series. No count callback was invoked. This metadata audit did not run inference or alter the production database. The frozen inventory-signature projection prevents new display/profile fields from invalidating unchanged existing scans.

Signed-pixel coverage was subsequently extended to known negative/positive extrema and transitions at 16 bits stored and 12 bits stored with high bit 11. All seven tested encodings passed exact decoded-pixel comparisons. Across classic and shuffled Enhanced CT, all 28 new converter combinations passed full-volume HU verification with zero error and unchanged originals, including varying Enhanced frame calibration. The complete codec module passed 62 tests in the clean application runtime. These fixtures do not establish compatibility with every vendor export.

The actual local application was then upgraded from a fresh deployment backup. The same production tables passed an exact row comparison immediately after migration, retaining 109 runs and 28 annotation revisions. The live API reported 26 studies, 230 series, 98 ready and 132 unavailable, with zero count jobs. Browser inspection verified the upload options and the assisted-boundary review path on an existing failed automatic attempt. GitHub's application/browser and counting-engine jobs both passed at `960e14e`.

### Reproducing the application dependency lock

The hashed lock includes the pinned build backend because both bootstrap and CI use `--no-build-isolation` for the editable project. Keep `requirements-build.in` synchronized with `build-system.requires` in `pyproject.toml`. Regenerate from both inputs:

```powershell
uv pip compile pyproject.toml requirements-build.in --extra test --generate-hashes --python-version 3.13 --no-header -o requirements.lock
```

The corrected lock was installed with `pip install --require-hashes` into a fresh Python 3.13 environment containing only pip, without inherited site-packages or a manually supplied `.pth` file. Editable installation with `--no-build-isolation --no-deps` succeeded and `pip check` reported no broken requirements. Two consecutive lock generations produced identical bytes. Bootstrap now stops immediately if environment creation, dependency installation or editable installation fails.

The live environment was separately rebuilt using `uv pip sync`, which need not install `pip` as a runtime package. Conversion provenance now records an absent installer as `null`; missing required image-processing libraries still fail. Actual decoder versions are included. Real classic and shuffled Enhanced CT uploads exercised the full catalog/converter/HU pipeline in that environment without supplied or mocked provenance: both had zero HU error, unchanged originals and exactly one scheduling callback after verification. The unchanged geometry-failure regression and related provenance/catalog tests passed (22 tests).
