# Ethmoid Complexity Pipeline

This repository contains the engineering foundation for the UCSD ethmoid-complexity project:

- a fail-closed DICOM-to-NIfTI conversion pipeline with geometry and HU verification; and
- a function-first local web application for scan review and provisional manual annotation.

The current development milestone is a fully automatic, experimental 3D cell-counting prototype. Clinical accuracy is not established, and no composite complexity score is defined.

## 3D counting implementation status

The latest clinician email and answered questionnaire request a count of unique cells throughout the full CT volume. They supersede the older selected-landmark and anterior/posterior counting proposal. Preliminary inference is authorized before manual 3D reference counts are available. Review and correction follow the first automatic result; no patient-specific ROI, seeds, or parameter tuning precede it.

The implementation contract and phased plan are in [Full-volume counting prototype](docs/COUNTING_PROTOTYPE.md). Patient images and derived outputs remain local; the private repository receives only engineering artifacts and synthetic tests.

| Phase | Status | Delivered / next action |
| --- | --- | --- |
| 1. Contract and verification | Completed | Separate experimental protocol, requirement mapping, deterministic autosave regression, synthetic CI, complete-index medical-data guard |
| 2. Runtime and cohort | Completed | Hash-locked offline GPU environment, four model manifests, ten verified volumes and immutable study split |
| 3. Automatic counting | In progress in an isolated worktree | Paired anatomical localization and native-resolution cavity instances |
| 4. Persistent run API | In progress in an isolated worktree | Jobs, original results, correction revisions and exports |
| 5. Review interface | In progress in an isolated worktree | Numbered overlay, counts, corrections and presentation exports |
| 6. Held-out evaluation | Pending | Frozen five-development/five-held-out study evaluation and handoff |

Executed Phase 1 checks: Python **54 passed**; frontend **27 passed**; production TypeScript/Vite build **passed**; real Edge/WebGL synthetic workflow **1 passed**; npm audit **0 vulnerabilities**; full Git-index medical-data guard **passed**. The browser run emitted a Windows connection-reset message during shutdown but completed successfully. One existing third-party Starlette deprecation warning and the existing large imaging-library chunk warnings remain. The previous intermittent brightness/autosave test now waits for the viewer-ready microtask and explicitly advances the debounce timer without relaxing its persistence assertion. GitHub CI is new and is checked after publishing.

Current data readiness was refreshed read-only before implementation: 24 sharp reconstruction candidates passed engineering preflight; the pilot's full-volume source-to-NIfTI comparison again produced zero HU error. These are engineering findings, not clinical suitability or counting-accuracy results. The questionnaire gives positive feedback on the submitted conversion comparison; historic QC files remain immutable and are not silently upgraded to approve other scans.

The first remote CI run exposed an ordering dependency: the HTTP test serves the built frontend, so it needs the production build before Python tests. The workflow now builds first, matching the successfully executed local sequence. No test assertion was removed.

Phase 2 conversion finding: two decimal-spacing series exposed a small dcm2niix sform precision drift. The converter now reconciles only bounded form disagreement against validated DICOM geometry when one original form already passes the existing voxel-grid tolerance. It preserves encoded voxel bytes and HU scaling and then runs the unchanged full-volume verification. Nine focused conversion tests passed, including axis flips and oblique geometry. Separate local copies of both affected real outputs verified with zero HU error and unchanged voxel payloads; official cohort conversions retain committed provenance.

Phase 2 runtime and cohort are complete; see [Runtime setup and findings](docs/INFERENCE_RUNTIME.md). The isolated Python 3.12.14 environment has 122 consistent packages and four prefetched model sets (28 hashed files). CUDA matrix and 3D-convolution smoke checks matched CPU results; external Python DNS is blocked and TotalSegmentator telemetry is disabled. Both localizers ran on the pilot and produced 26 binary masks matching its native geometry. These models supply landmarks, not ethmoid-cell labels.

The protected frozen cohort contains five development and five held-out studies. All ten full-volume HU comparisons passed with maximum error **0.0**, and a repeat preparation run reused every verified image without changing the manifest. Acquisition eligibility does not establish clinical suitability. The phase worktree passed **60 Python tests** and its production build. Counts are still under development: initial real-scan runs exposed localization and candidate-rejection defects, so they are not accuracy evidence or presentable anatomical totals.

## Documentation rule

This README is the durable project handoff and must be updated after every meaningful project work session. Record completed work, important findings or decisions, current validation status, newly created outputs, unresolved questions, and the next action. Keep completed and pending work clearly separated so a future chat can resume without reconstructing prior context.

## Existing application and conversion baseline (2026-08-28)

### Validated conversion pilot

One audited CT series, `Case003 / Series007`, has been converted from DICOM to NIfTI and independently checked by the pipeline.

Automated status: **PASS, pending manual Slicer and clinical review**.

- The 224 source DICOM slices were assembled into one `512 x 512 x 224` NIfTI volume.
- Pixel spacing is `0.488281 x 0.488281 x 0.625 mm`.
- All 58,720,256 voxels match the source DICOM after HU rescaling; maximum error is `0 HU`.
- The affine, physical coordinates, slice order, qform, sform, source hashes, and conversion provenance passed.
- A visual QC package was generated with orthogonal views, coverage montages, DICOM/NIfTI comparisons, and a 224-frame axial cine.
- A concise review PDF and email draft were prepared for the study team.

### Local web application MVP

A functional single-user MVP now runs at `http://127.0.0.1:8765` with:

- a case dashboard;
- NIfTI import with optional matching QC-report validation;
- DICOM-folder import, neutral series selection, strict preflight, conversion, and full geometry/HU verification;
- linked axial, coronal, and sagittal NiiVue views plus a 3D volume toggle;
- crosshairs, orientation markers, coordinates, zoom/pan, window/level, and the `300/3000 HU` bone preset;
- provisional four-region manual annotation with draw, erase, fill, brush size, overlay visibility/opacity, undo, and redo;
- immutable `uint8` NIfTI mask revisions, annotation/QC/notes autosave, multiplanar/3D camera-state autosave, and exact reload;
- separate automated-validation and manual-QC states;
- reviewer notes and `Draft`, `Ready for review`, `Needs changes`, and `Approved` workflow states;
- per-label voxel counts and physical volumes; and
- neutral ZIP export containing the mask, manifest, protocol, QC, notes, status, and measurements.

The interface now uses a purpose-built **Ethmoid Atlas** visual system rather than a generic admin layout. It includes:

- a four-step first-run guide that can be replayed from the persistent `Help` button;
- keyboard-, hover-, and click-accessible explanations beside the main viewer, annotation, image, validation, and QC controls;
- grouped annotation and image-display controls that keep the CT canvas visually neutral;
- dedicated `Brightness` (`-1000` to `2000 HU`) and `Contrast` (`100` to `5000 HU`) sliders with exact numeric entry; and
- the exact `Bone · 300/3000` display preset.

Saved brightness and contrast are applied when a scan first opens. Changes made with the canvas window/level tool also synchronize back to the sliders and saved viewer state. These controls alter only the display; they do not alter the CT voxel values.

The CT volume and saved annotation are served to the embedded viewer as inline resources. Opening or navigating a case therefore does not trigger browser download requests. Only the explicit `Export` action is served as a downloadable attachment.

The existing verified `Case003` NIfTI and QC report are loaded into the local app as a `VERIFIED` case. Its current local `Draft` contains saved annotation revisions; those marks are provisional and must not be treated as clinical ground truth or training data. The import was repeated after tightening the QC-report contract and still passed. Real-browser checks confirmed CT rendering, multiplanar/3D switching, drawing, autosave, mask reload, viewer-state persistence, and measurement display.

Application labels are explicitly `PROVISIONAL_NOT_FOR_TRAINING`:

1. Left anterior ethmoid
2. Right anterior ethmoid
3. Left posterior ethmoid
4. Right posterior ethmoid

These names are a temporary protocol configuration, not a clinical decision. The application deliberately displays no model prediction or complexity score.

### Stakeholder feedback and Protocol v2 draft

The study-team response now gives a clearer intended end point: local, single-scan DICOM processing; bilateral ethmoid-compartment review; individual air-cell identification and correction; side-specific counts and measurements; and comparison with manual counts at three coronal landmark locations.

The supplied annotation reference contains nine completed examples with right/left point counts at middle-turbinate, anterior-ethmoid, and posterior-ethmoid planes, plus one incomplete/template page. These screenshots are useful workflow references, but they are not full-volume contours or individual-cell training masks.

Three aligned Protocol v2 artifacts have been drafted:

- [Ethmoid Air-Cell Annotation Protocol v2](docs/ANNOTATION_PROTOCOL_V2_DRAFT.md), which separates confirmed requirements, provisional implementation choices, and blocking clinical decisions;
- [Protocol v2 Clinical Review Form](docs/PROTOCOL_V2_CLINICAL_REVIEW.md), which requests only the remaining definitions, roles, thresholds, and approvals; and
- [`protocol.v2.draft.json`](src/ethmoid_pipeline/webapp/protocol.v2.draft.json), an inactive machine-readable representation of the proposed two-mask, instance-registry, and landmark-point contract.

The v2 draft is `DRAFT_FOR_CLINICAL_REVIEW` and `NOT_FOR_TRAINING`. It is not packaged, loaded, or returned by the application. Protocol v1 and all current app behavior remain unchanged.

An editable Google Docs copy of the focused clinical review form has been created in the project owner's Drive under the `ChatGPT` folder. The private Drive link is not stored in Git; the tracked Markdown review form remains the source used when incorporating approved answers into the next frozen protocol revision.

### Validation evidence

- Python suite: **50 passed**; one third-party Starlette TestClient deprecation warning. The suite now verifies Protocol v2 artifact alignment, inactive/not-for-training status, the narrow repository-guard exception, package-data isolation, and continued Protocol v1 runtime selection.
- Frontend suite: **27 passed** across seven test files, including first-use guidance, hoverable contextual help, modal focus containment, brightness/contrast synchronization and safe restore bounds, pending-change navigation, and approved-case read-only behavior.
- Automated Microsoft Edge/WebGL workflow: **1 passed**. It completes the first-run guide, imports a synthetic directional NIfTI, verifies orientation-aware 2D and 3D rendering, proves that slider changes alter the rendered image, reloads and restores exact non-default brightness and contrast, draws and autosaves a mask, reloads the annotation and viewer mode, proves browser-Back review persistence, completes QC, finalizes for review, and downloads the ZIP export.
- Production TypeScript/Vite build: **PASS**.
- Runtime/development dependency audit: **0 vulnerabilities**.
- The repository guard passed against an isolated index containing the complete would-be commit.
- Installed `ethmoid-app` command, health endpoint, static frontend, protocol endpoint, case API, and 117,440,864-byte pilot-volume delivery: **PASS**.
- Real browser: dashboard, NIfTI decoding, 2D/3D CT display, drawing, autosave, reload, coordinates, and metrics: **PASS**.

The current web-app implementation is committed and published to the private GitHub repository on branch `feature/function-first-webapp`. The protected baseline on `main` remains commit `e95b1c9`; this feature branch has not been merged.

## What the conversion means

A DICOM CT study is a set of separate scanner images plus scanner and patient metadata. NIfTI stores the same ordered image pixels as a single geometry-aware 3D volume that image-analysis software can use reliably. The conversion does not invent anatomy, enhance the scan, segment structures, or create a 3D surface model.

The axial cine is not a 3D rendering. It is a slideshow through the 224 two-dimensional slices in the 3D NIfTI volume. The web app can volume-render the CT, but a meaningful rotatable ethmoid structure requires an approved definition and a segmentation mask.

## Work completed

- Audited the local DICOM export and identified candidate axial bone CT series.
- Built metadata preflight checks that reject mixed acquisitions, duplicate positions, missing planes, changing geometry, unsupported transfer syntax, and derived/localizer images.
- Built reproducible DICOM-to-NIfTI conversion with pinned tools and dependencies.
- Generalized the pilot-only converter into reusable single-series conversion and verification services while preserving the strict `Case003 / Series007` regression path.
- Built full-volume geometry, orientation, source-integrity, and HU verification.
- Added synthetic conversion, rejection, repository-guard, API, persistence, export, and frontend tests.
- Added an automated production-build browser test using a directional NIfTI phantom and real NiiVue/WebGL rendering.
- Added the local FastAPI, SQLite, React, Vite, and NiiVue application described above.
- Required external QC reports to match the complete audited check schema and volume hash before a NIfTI can be marked `VERIFIED`.
- Added cleanup for converted, failed, and cancelled DICOM import staging areas.
- Added a navigation guard that flushes pending annotation, review, and viewer state before both in-app and browser-Back navigation.
- Made approved cases read-only for masks, QC, and review content while preserving scan navigation and export; the backend also rejects any attempt to re-finalize and downgrade an approved case.
- Added the Ethmoid Atlas interface, versioned first-run guide, contextual tooltips, grouped controls, brightness/contrast sliders, exact HU inputs, and bidirectional NiiVue window synchronization.
- Stopped normal CT and annotation loading from being advertised as file downloads while preserving attachment behavior for explicit ZIP exports.
- Generated patient-derived visual review material outside Git.
- Prepared a reviewer package asking for conversion approval and application requirements.
- Reviewed the returned application requirements and landmark-count examples without treating screenshot points as 3D training ground truth.
- Drafted Protocol v2, its focused clinical review form, and an inactive machine-readable configuration for bilateral compartments, individual-cell instances, and physical landmark observations.
- Created and verified an editable Google Docs version of the clinical review form for stakeholder completion.
- Reviewed the published SinusSegment approach. Its released model predicts one combined sinonasal foreground rather than separate anterior/posterior or left/right ethmoid structures, and the publication does not report ethmoid-only performance. It has therefore not been adopted as the project solution.

## Not completed

- Manual 3D Slicer comparison.
- Clinical confirmation of ethmoid-region coverage and suitability.
- Clinical approval of Protocol v2, its anatomical rules, reviewer roles, and predeclared pilot thresholds.
- A clinically defined and validated complexity score; Protocol v2 deliberately omits one.
- Expert ground-truth annotations or an annotation-consistency pilot.
- Runtime support for Protocol v2 multi-layer `uint16` individual-cell annotation; the current app remains a single `uint8` four-region editor.
- Automated ethmoid segmentation, model training, inference, or complexity scoring.
- Bulk conversion of the remaining studies.
- Remote hosting, authentication, multi-user assignment, or collaboration features.

## Next milestone

Complete the six-phase automatic counting prototype described above. The older inactive Protocol v2 documents describe a stricter annotation/training workflow and do not block the newly authorized exploratory inference path. Their historical approvals are not fabricated or changed. Reference counts will be used to evaluate untouched automatic predictions when available; model-assisted corrections remain separate from independent ground truth. Training a bespoke supervised clinical model is not part of this first prototype.

## Start the application

From PowerShell at the repository root:

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
ethmoid-app start
```

The command opens `http://127.0.0.1:8765`. The default app data directory is outside Git under `%LOCALAPPDATA%\EthmoidComplexity\work\webapp`.

Basic workflow:

1. Select `Import scan` or open an existing case.
2. Import a DICOM folder or `.nii`/`.nii.gz` file.
3. Review the linked CT views and automated validation result.
4. Complete the separate manual QC checklist.
5. Draw provisional labels and save or allow autosave.
6. Add notes, finalize for review, and approve only after the required review.
7. Export the neutral annotation package.

An external NIfTI without a matching QC report remains `UNVERIFIED` and viewable but cannot be approved. A failed DICOM conversion cannot be finalized.

## Conversion commands

Set the source location only in the current shell; do not save it in tracked configuration:

```powershell
$env:ETHMOID_DICOM_ROOT = '<read-only DICOM root>'
ethmoid-pipeline inventory
ethmoid-pipeline convert --case Case003 --series Series007
ethmoid-pipeline verify --case Case003 --series Series007
```

`inventory` assigns neutral labels using case-insensitive sorted relative paths and saves the UID/path mapping only in the local work root. The standalone `convert` command remains intentionally restricted to the audited pilot fingerprint. The web app uses the reusable single-series service only after its own strict preflight.

## Privacy boundary

Raw DICOM, the local UID map, staged files, NIfTI volumes, masks, converter logs, Slicer state, screenshots, QC outputs, SQLite app data, and exports remain outside this repository. The repository contains only source, configuration, documentation, and synthetic tests.

See [SECURITY.md](SECURITY.md) before handling real data.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
npm --prefix web test
npm --prefix web run build
npm --prefix web run test:e2e
npm --prefix web audit --audit-level=moderate
python scripts/check_staged.py
```

All test DICOMs and NIfTIs are generated synthetically. Real medical data is never copied into the test tree or CI.
