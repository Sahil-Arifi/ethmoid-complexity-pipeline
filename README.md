# Ethmoid Complexity Pipeline

This repository contains the engineering foundation for the UCSD ethmoid-complexity project:

- a fail-closed DICOM-to-NIfTI conversion pipeline with geometry and HU verification; and
- a local web application for automatic experimental 3D counting, scan review and revision-based correction.

The experimental 3D cell-counting prototype is implemented. Its workflow is functional, but the frozen method produced candidate counts on only two of five held-out studies and withheld the other three. Clinical accuracy is unmeasured, and no composite complexity score is defined.

## Study catalog and broader counting work

Implementation is in progress for the approved 26-study, 230-series local catalog. The Cases page will group every reconstruction under its study, prepare usable 3D volumes automatically and run counting only on request. Existing volume IDs, annotations and original predictions must be preserved. The clinician questionnaire explicitly specifies at most 1 mm slice thickness; broader experimental attempts will retain that distinction, with thickness recorded separately from actual slice spacing. The original v1 evaluation remains immutable.

Completed conversion foundation: a separate exploratory preflight profile accepts calibrated volumetric reformats and orthogonal oblique acquisitions without disabling geometry, HU or source-integrity checks. Scouts, projection averages, mixed image types, in-plane drift and missing/duplicate planes are rejected explicitly. The original strict profile remains the default. **48 preflight/conversion tests passed**, including actual dcm2niix conversions at 0, 30 and 90 degrees with zero full-volume HU error and unchanged synthetic source hashes.

Completed broader engine: method `native-bone-watershed-0.2.0` and protocol `ethmoid-count-prototype-v2` attempt verified thicker volumes while retaining the questionnaire's thickness criterion as an acquisition warning. Native voxels determine air, instances and measurements. Orthogonal oblique inputs use an anatomy-only RAS proxy with complete physical extent, linear CT interpolation, nearest-neighbor region mapping, padding exclusion and a 128-million-voxel limit. Enclosure support is capped at 1.01 mm; unresolved boundaries still withhold counts. The exact v1 protocol and historical results remain readable and exportable. **82 CPU geometry/engine/quality tests passed without skips**; **101 application, report and guard checks passed**. No new real-scan counting-accuracy result is implied by these tests.

The read-only collection preflight accounted for all **26 studies and 230 series**: **102** series passed exploratory 3D preflight (92 native and 10 volumetric reformats), across 24 studies. The remaining 128 remain visible with limitations; two studies have no supported 3D stack. Full conversion may identify further failures.

Completed persistent catalog: production migration created a protected database backup and registered all **26 studies / 230 series**. Both existing volume identities were verified against source pixels and correctly grouped under study Case003: legacy Case001 is Series009 and legacy Case003 is Series007. Comparing every legacy table row against the backup confirmed both cases, all 28 annotation revisions and all four count runs unchanged. Repeated synchronization retained the same coverage without duplicates. The one-worker queue supports interrupted recovery, explicit retry/priority promotion, deterministic recommendations and exact source checks before new counts. New files and raw identities remain protected locally. See [Catalog operation and API](docs/CATALOG.md).

Completed grouped interface: study rows expose every series, preparation/count status and distinct acquisition thickness/spacing. Switching series protects unsaved work and clears previous masks/results. Integrated checks passed **39 frontend tests**, **three synthetic browser workflows**, and the production build. The integrated Python suite passed **243 tests**, with two SciPy-dependent modules separately covered by the dedicated CPU suite; the final catalog hardening suite passed **28 tests**. Those catalog/preparation tests include real synthetic conversion, unchanged source hashes, zero inference jobs from preparation, reuse rejection for mismatched voxels or displaced affines, process-lock ownership and ready-state persistence. Automatic real-collection preparation and the frozen full-collection evaluation are next; clinical accuracy remains unmeasured.

The separate [all-series evaluation procedure](docs/CATALOG_EVALUATION.md) now has a read-only manifest freezer and a serial offline runner. They verify current raw-source fingerprints, immutable image hashes, complete acquisition/quality metadata, native outputs and frozen code/model/settings provenance. Every attempted series retains a success or failure record. **35 evaluator/freezer tests passed**; the unchanged v1 evaluator also remains covered. Real preparation is running from clean commit `b0baa49`; no all-series inference results are available yet.

## 3D counting implementation status

The latest clinician email and answered questionnaire request a count of unique cells throughout the full CT volume. They supersede the older selected-landmark and anterior/posterior counting proposal. Preliminary inference is authorized before manual 3D reference counts are available. Review and correction follow the first automatic result; no patient-specific ROI, seeds, or parameter tuning precede it.

The implementation contract and phased plan are in [Full-volume counting prototype](docs/COUNTING_PROTOTYPE.md). Patient images and derived outputs remain local; the private repository receives only engineering artifacts and synthetic tests.

| Phase | Status | Delivered / next action |
| --- | --- | --- |
| 1. Contract and verification | Completed | Separate experimental protocol, requirement mapping, deterministic autosave regression, synthetic CI, complete-index medical-data guard |
| 2. Runtime and cohort | Completed | Hash-locked offline GPU environment, four model manifests, ten verified volumes and immutable study split |
| 3. Automatic counting | Completed as experimental inference | Three anatomical priors, native-resolution cavity instances, explicit abstention and uncertainty; settings frozen for evaluation |
| 4. Persistent run API | Completed; real offline worker acceptance passed | Durable jobs, immutable predictions, correction lineage, source checks and per-response exports |
| 5. Review interface | Completed; synthetic browser and real report checks passed | Numbered native-coordinate overlays, revision-bound corrections and PDF/PNG/CSV/NIfTI exports |
| 6. Held-out evaluation | Completed as engineering feasibility evaluation | All ten attempts recorded; 3/5 development and 2/5 held-out outputs; clinical accuracy unmeasured |

Executed Phase 1 checks: Python **54 passed**; frontend **27 passed**; production TypeScript/Vite build **passed**; real Edge/WebGL synthetic workflow **1 passed**; npm audit **0 vulnerabilities**; full Git-index medical-data guard **passed**. The browser run emitted a Windows connection-reset message during shutdown but completed successfully. One existing third-party Starlette deprecation warning and the existing large imaging-library chunk warnings remain. The previous intermittent brightness/autosave test now waits for the viewer-ready microtask and explicitly advances the debounce timer without relaxing its persistence assertion. GitHub CI is new and is checked after publishing.

Current data readiness was refreshed read-only before implementation: 24 sharp reconstruction candidates passed engineering preflight; the pilot's full-volume source-to-NIfTI comparison again produced zero HU error. These are engineering findings, not clinical suitability or counting-accuracy results. The questionnaire gives positive feedback on the submitted conversion comparison; historic QC files remain immutable and are not silently upgraded to approve other scans.

The first remote CI run exposed an ordering dependency: the HTTP test serves the built frontend, so it needs the production build before Python tests. The workflow now builds first, matching the successfully executed local sequence. No test assertion was removed.

Phase 2 conversion finding: two decimal-spacing series exposed a small dcm2niix sform precision drift. The converter now reconciles only bounded form disagreement against validated DICOM geometry when one original form already passes the existing voxel-grid tolerance. It preserves encoded voxel bytes and HU scaling and then runs the unchanged full-volume verification. Nine focused conversion tests passed, including axis flips and oblique geometry. Separate local copies of both affected real outputs verified with zero HU error and unchanged voxel payloads; official cohort conversions retain committed provenance.

Phase 2 runtime and cohort are complete; see [Runtime setup and findings](docs/INFERENCE_RUNTIME.md). The isolated Python 3.12.14 environment has 122 consistent packages and four prefetched model sets (28 hashed files). CUDA matrix and 3D-convolution smoke checks matched CPU results; external Python DNS is blocked and TotalSegmentator telemetry is disabled. Both localizers ran on the pilot and produced 26 binary masks matching its native geometry. These models supply landmarks, not ethmoid-cell labels.

The protected frozen cohort contains five development and five held-out studies. All ten full-volume HU comparisons passed with maximum error **0.0**, and a repeat preparation run reused every verified image without changing the manifest. Acquisition eligibility does not establish clinical suitability. The phase worktree passed **60 Python tests** and its production build. Counts are still under development: initial real-scan runs exposed localization and candidate-rejection defects, so they are not accuracy evidence or presentable anatomical totals.

Phase 4 backend work proceeded alongside algorithm development. It now persists single-worker jobs, reuses idempotent requests, handles interrupted runs, and migrates existing databases with a local backup. Every result is tied to its source and protocol. Corrections preserve revision 0, allocate new IDs for split/merge, enforce stale-revision checks, support IDs above 255, and keep anatomical cavities separate from air-only masks. Approved cases cannot be edited or recounted. Active jobs block deletion; completed case deletion removes its count artifacts.

The integrated application suite passed **96 tests** before the final export fix; the final **34 counting API/revision tests** passed afterward. The export regression holds a ZIP open to reproduce Windows file locking: each response now owns a separate temporary export and cleans it up after streaming. The backend worktree's complete suite passed **84 tests**. These checks use synthetic inference. The automatic engine is still being evaluated on development scans; a working job API is not evidence of correct clinical counts.

Phase 5 makes whole-volume counting the default workspace. It shows left/right/total candidate counts, numbered air or cavity overlays, physical-coordinate cell focus, run history and the untouched original prediction. After inference, reviewers can add, remove, split, merge or paint a boundary; each correction is bound to its parent revision. A conflicting edit cannot silently overwrite a newer revision. Previous manual annotations remain accessible separately. Viewer controls and camera position persist locally.

Exports include original and reviewed counts, cell registries, native NIfTI labels, a paginated PDF summary and a representative PNG. Reports use physical coordinates for oblique or anisotropic scans, carry orientation labels and uncertainty markers, and avoid copying source header metadata. Report hashes bind the presentation files to the export manifest. These are protected local study outputs and must not be committed.

Executed Phase 5 checks: **124 Python tests**, **34 frontend tests**, **2 browser workflows**, production build, and npm audit (**0 vulnerabilities**). The tests cover an ID above 255, opaque count-only corrections, revision conflicts, original/current export agreement, actual ZIP download bytes, and preserved annotation behavior. Browser checks use isolated Playwright Chromium; a host Edge extension was intercepting ZIP requests, so CI no longer relies on the host browser profile. The test-only inference server accepts only its exact generated phantom and is not part of the production application entry point. Clinical accuracy remains unmeasured; next action is to freeze the development-tested counting engine and evaluate every held-out study.

Phase 3 now implements the full automatic worker. It uses skull/sinus/orbital/nasal landmarks plus a brain prior, then native-bone compartment proposals with stable physical-coordinate IDs. A clean source scan requires no manual outlines, seeds or case-specific settings. Earlier development attempts with brain or nasal leakage were rejected. The final shared method produced three exploratory outputs and withheld two of the five development scans because the sphenoid boundary was ambiguous or unresolved. Visible thin septa still create possible merged cells; uncertainty flags preserve this limitation. Named-variant and fully opaque-cell accuracy are not established. Oblique automatic localization is explicitly unsupported in this version, even though import, conversion verification and reports support oblique geometry.

The engine passed **34 CPU tests without skips**, including whole-volume identity, septa/communication, partial/complete opacity, physical laterality, native labels, cache provenance, fallback control paths and unsupported geometry. A separate hash-locked CPU environment lets CI execute these tests without CUDA or model downloads. A full uncached real pilot run completed with all three localizers and verified producer hashes; its preliminary total agreed with the cached development run. This is repeat execution evidence, not clinical accuracy. The frozen evaluation runner passed **28 tests**, rejects split/source/model/code drift and empty false-success outputs, and records failures with unavailable counts.

Additional visual QA corrected an upstream indexed-color rendering defect that tinted label-zero background for high cell IDs. The viewer now uses an RGBA display copy while preserving the canonical uint16 image for IDs and edits. A browser pixel regression verifies unchanged background and changed cell pixels; full browser workflows (**2 passed**), frontend tests (**34 passed**) and build passed again. Desktop, tablet and narrow-phone screenshots were checked; toolbar controls wrap without clipping. Exported bone views now match the interface's 300/3000 HU preset. The held-out images have not yet been used for inference; the next phase runs the committed frozen method once across their fixed split.

Final freeze checks: the application Python suite passed **153 tests** with the inference module intentionally skipped because SciPy is isolated from the application environment; that module passed all **34 tests** in the dedicated CPU environment. Small GPU localizer-mask differences occurred between fresh and earlier runs, so bitwise GPU reproducibility is not claimed. Counts agreed on the pilot. [Frozen evaluation procedure](docs/COUNTING_EVALUATION.md) records how to reproduce the next phase without changing code, settings or study membership.

The first frozen development evaluation completed all five attempts in 343.6 seconds: three outputs passed artifact/count consistency checks and two were withheld for sphenoid-boundary failures. All 56 accepted candidates were uncertain; this aggregate is not a cohort ground-truth count. Historical development caches retain unverified producer-binding flags. The engine and twelve parameters remain frozen at `fc8206d770c113e17f678c3e54d81500deae6481` while application handoff checks continue.

CI setup correction: GitHub's Python action did not provide Windows Python 3.12.14, so the engine job stopped before tests. It now uses the same uv-managed Python distribution as the local runtime, with uv 0.12.5 and the existing hash-locked CPU dependencies; [the pinned setup action](https://github.com/astral-sh/setup-uv) provides that installation path. The application job passed. Production workers now explicitly disable user-site packages and nnUNet compilation to match the evaluation environment. A focused launcher test passed; the counting/revision suite passed **34 tests**, frontend **34 tests**, and the production build passed after the handoff wording update. No frozen algorithm or threshold changed.

Both remote CI jobs passed after that environment correction. Production API acceptance then ran the real offline worker on the existing verified pilot in **181.6 seconds**, with no injected executor or manual prompts. It preserved the source, all **28** prior annotation revisions and the complete case/QC metadata; schema migration created its backup. The count is saved in the local application. PDF pages, registry/count text, image bounds, PNG views, CSV, source hashes and native labels passed local automated checks. These checks are not human anatomical review. User-reviewable reports and acceptance evidence are protected under `work\counting\app-acceptance`; none is tracked in Git.

The API now rejects mismatched effective settings and cell centroids, in addition to its existing source/mask/count checks. The persisted real result passed those stricter checks without modification. The updated API/revision suite passed **39 tests**. The counting browser workflow also passed with an explicit nonblank 3D-render check, return to multiplanar review, stable ID sampling, correction, reload and export. No JavaScript errors occurred. Actual background server launch was rejected by automatic approval review with the reason `blocked by policy`; it was not retried. The production API was verified in-process without a listening server. The prepared application can be launched manually with the command below.

Phase 6 is complete: all five held-out studies ran once against the frozen engine, unchanged settings, fixed split and verified model provenance. **Two produced candidate outputs; three withheld counts** (one ambiguous and two unresolved sphenoid boundaries). The run took **746.9 seconds** including local checks. Every accepted held-out candidate is uncertain. No clinical reference counts were available, so neither execution success nor geometric air coverage is counting accuracy. [Full evaluation findings](docs/COUNTING_EVALUATION.md) record all aggregates, failures and provenance limits. Per-study results, numeric review and clinician-review montages remain protected locally.

Final held-out numeric review passed the two outputs' native geometry, source binding, IDs, air values, centroids, volumes and laterality checks. All **71** evaluator-recorded file hashes were unchanged. The **51** uncertain candidates include **31** possible internal-septum merge flags. Six protected montages are ready for human review; automated overlap checks do not establish anatomical correctness.

The final implementation CI at `b50633f` passed: **152 application tests** (four local integration tests deselected, one inference module intentionally skipped), **34 dedicated CPU engine tests**, **34 frontend tests**, **2 browser workflows**, production build, npm audit (**0 vulnerabilities**) and the complete tracked-file medical-data guard. The separate engine job executes the module skipped by the application environment. Earlier local full-suite and real-source checks are recorded above; these results establish software behavior, not clinical accuracy.

The deliverable is a working presentation prototype with real preliminary outputs, correction lineage and exports. It is **not a reliable or validated clinical counter**. The posterior-boundary failures and possible merged cells are known model limitations. Next action: collect independent manual 3D counts and expert cell/boundary review, then improve the model on development data and evaluate a new freeze. Do not present the older selected-slice annotations or model-assisted corrections as independent 3D ground truth.

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

## Remaining validation and excluded scope

- Manual 3D Slicer comparison.
- Clinical confirmation of ethmoid-region coverage and suitability.
- Clinical approval of Protocol v2, its anatomical rules, reviewer roles, and predeclared pilot thresholds.
- A clinically defined and validated complexity score; Protocol v2 deliberately omits one.
- Expert ground-truth annotations or an annotation-consistency pilot.
- Clinical validation of the implemented `uint16` cell instances and automatic inference. The older four-region `uint8` editor remains available separately; the inactive Protocol v2 draft is not the counting runtime contract.
- Training a bespoke supervised ethmoid-instance model and defining a complexity score are outside this prototype's scope.
- Bulk conversion beyond the verified ten-study cohort remains outside this prototype's scope.
- Remote hosting, authentication, multi-user assignment, and collaboration features are outside this prototype's scope.

## Next milestone

The six prototype engineering phases are complete. The next milestone is clinical reference review and improved localization, not additional SaaS features. The older inactive Protocol v2 documents retain their historical approval states. Reference counts will evaluate untouched automatic predictions when available; model-assisted corrections remain separate from independent ground truth. Training a bespoke supervised clinical model was not part of this first prototype.

## Start the application

From PowerShell at the repository root:

```powershell
.\scripts\bootstrap.ps1
.\scripts\bootstrap_inference.ps1
.\.venv\Scripts\Activate.ps1
ethmoid-app start
```

The command opens `http://127.0.0.1:8765`. Bootstrap is a one-time environment setup; on the already prepared machine, `.\.venv\Scripts\ethmoid-app.exe start` starts the app directly. The inference bootstrap uses uv and prefetches public model weights before patient processing. See [runtime setup](docs/INFERENCE_RUNTIME.md). The default app data directory is outside Git under `%LOCALAPPDATA%\EthmoidComplexity\work\webapp`.

Basic workflow:

1. Select `Import scan` or open an existing case.
2. Import a DICOM folder or `.nii`/`.nii.gz` file.
3. Select `Run automatic count`; no outlines or seed points are needed.
4. Review the preliminary left/right/total and numbered cells, or the reason a count was withheld. Switch between air and cavity overlays and click a cell to locate it.
5. Add, remove, split, merge or edit a boundary only after the first result. Each saved correction creates a revision and preserves the original automatic prediction.
6. Export the current or original revision as a ZIP containing count CSV, cell registry, native labels, PDF and PNG. Keep these medical outputs protected locally.

`Previous annotations` retains the earlier drawing, manual QC and approval workflow. It is not needed to start automatic inference. Its existing approval lock still protects the case from recounting or correction. A model-assisted correction is not an independent reference count.

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
