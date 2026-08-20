# Ethmoid Complexity Pipeline

This repository contains the engineering foundation for the UCSD ethmoid-complexity project:

- a fail-closed DICOM-to-NIfTI conversion pipeline with geometry and HU verification; and
- a function-first local web application for scan review and provisional manual annotation.

It does **not** yet contain a clinically approved annotation protocol, automated ethmoid segmentation, a complexity score, or a trained clinical model.

## Documentation rule

This README is the durable project handoff and must be updated after every meaningful project work session. Record completed work, important findings or decisions, current validation status, newly created outputs, unresolved questions, and the next action. Keep completed and pending work clearly separated so a future chat can resume without reconstructing prior context.

## Current project status (2026-08-20)

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

The existing verified `Case003` NIfTI and QC report are loaded into the local app as a clean, unannotated `VERIFIED` case. The import was repeated after tightening the QC-report contract and still passed. A real-browser smoke test confirmed CT rendering, multiplanar/3D switching, drawing, autosave, mask reload, viewer-state persistence, and measurement display. The deliberately drawn test annotation was removed afterward.

Application labels are explicitly `PROVISIONAL_NOT_FOR_TRAINING`:

1. Left anterior ethmoid
2. Right anterior ethmoid
3. Left posterior ethmoid
4. Right posterior ethmoid

These names are a temporary protocol configuration, not a clinical decision. The application deliberately displays no model prediction or complexity score.

### Validation evidence

- Python suite: **48 passed**; one third-party Starlette TestClient deprecation warning.
- Frontend suite: **17 passed** across five test files, including pending-change navigation and approved-case read-only behavior.
- Automated Microsoft Edge/WebGL workflow: **1 passed**. It imports a synthetic directional NIfTI, verifies orientation-aware 2D and 3D rendering, draws and autosaves a mask, reloads the annotation and viewer mode, proves browser-Back review persistence, completes QC, finalizes for review, and downloads the ZIP export.
- Production TypeScript/Vite build: **PASS**.
- Runtime/development dependency audit: **0 vulnerabilities**.
- The repository guard passed against an isolated index containing the complete would-be commit.
- Installed `ethmoid-app` command, health endpoint, static frontend, protocol endpoint, case API, and 117,440,864-byte pilot-volume delivery: **PASS**.
- Real browser: dashboard, NIfTI decoding, 2D/3D CT display, drawing, autosave, reload, coordinates, and metrics: **PASS**.

The current web-app implementation is in the local working tree and has not yet been committed or pushed. The private GitHub repository still has validated baseline commit `e95b1c9` on `main`.

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
- Generated patient-derived visual review material outside Git.
- Prepared a reviewer package asking for conversion approval and application requirements.
- Reviewed the published SinusSegment approach. Its released model predicts one combined sinonasal foreground rather than separate anterior/posterior or left/right ethmoid structures, and the publication does not report ethmoid-only performance. It has therefore not been adopted as the project solution.

## Not completed

- Manual 3D Slicer comparison.
- Clinical confirmation of ethmoid-region coverage and suitability.
- Final annotation protocol or definition of complexity.
- Expert ground-truth annotations or an annotation-consistency pilot.
- Automated ethmoid segmentation, model training, inference, or complexity scoring.
- Bulk conversion of the remaining studies.
- Remote hosting, authentication, multi-user assignment, or collaboration features.
- Visual-design refinement; functionality and correctness remain the priority.

## Next milestone

Collect reviewer answers and turn them into a versioned annotation and application specification. It must define:

- the exact anatomical structures and boundaries to label;
- whether labels are separated by left/right and anterior/posterior regions;
- the clinical or quantitative definition of complexity;
- required measurements, scores, overlays, reports, and exports;
- the complete user workflow from opening the application through reviewing and saving results;
- who creates annotations and who resolves disagreements; and
- what constitutes an acceptable pilot result.

After that specification and the conversion review are approved, run a small expert annotation-protocol pilot in this app. Adjudicate disagreements before annotating the remaining suitable scans. Model training comes only after approved ground truth exists.

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
