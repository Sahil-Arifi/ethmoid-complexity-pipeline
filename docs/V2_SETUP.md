# V2 software setup

Setup was completed on October 12, 2026 for local review tools and synthetic research comparators. No application or counting code, dependency locks, database contents, annotations or clinical outputs were changed. No application bootstrap, migration, training, model initialization or scan processing was run.

## Selected tools

| Tool | Fit and setup status | Verification |
| --- | --- | --- |
| 3D Slicer 5.12.3, revision 34627 | Reused the existing pinned installation for local CT review, named candidate locations and manual outlines. | Markups, Segment Editor and Volumes modules load. |
| SlicerNNInteractive 1.0.0 | Installed through Slicer's official extension manager, source revision `95a1bfae369806e0070a3887597eb92a1d0a2645`. Configured for local, in-process annotation assistance. | Extension and widget load; local backend is available. No model session or volume is initialized. |
| nnInteractive 2.6.0 | Installed in Slicer's embedded Python 3.12.10, separate from the application's counting runtime. Client 2.6.0 and nnUNet 2.8.1 are pinned. | Imports and dependency check pass. |
| PyTorch 2.14.0+cu130 / torchvision 0.29.0+cu130 | Installed in Slicer's Python for the RTX 5070 Ti. Slicer's NumPy remains 2.4.6. | CUDA detection passes; the wheel includes `sm_120`. GPU computation and inference were not run. |
| PoreSpy 3.1.1 | Installed in a separate Python 3.12.14 environment for a later synthetic SNOW2 comparison. It is not an anatomical counting model. | All 62 packages pass dependency checks; import and callable SNOW2 API checks pass. |
| pydicom skill 1.4 | Installed only this selected K-Dense skill. A separate utility environment contains pydicom 3.0.2, NumPy 2.5.3 and Pillow 12.3.0. | Installed source hashes, references, Python syntax and eight help commands pass; utility imports and dependency check pass. |
| scientific-critical-thinking skill 1.5 | Installed its offline evidence-appraisal guidance. No additional provider or schematic-generation service was configured. | Installed source hashes and local references pass. |

Both skills are pinned to K-Dense commit `a1043ef544e3a100c28e70a736b6cf32554d0677` and become available to Codex on the next turn. The existing council recommendation was reused; this setup did not repeat the same strategic decision.

The Windows Start menu contains **Ethmoid Review (3D Slicer)**. It launches the existing pinned Slicer installation without running a user startup script.

## Local model and settings

The official `nnInteractive_v1.0` model was downloaded from `MIC-DKFZ/nnInteractive` at revision `3f308d751c00644e4fde6f09c600264b393b21b5`. All five required file lengths were verified. The 411,387,150-byte checkpoint matches SHA256 `b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9`. Small files also match published Git blob hashes. The upstream model card, manifest and license are retained locally.

Slicer settings specify local mode, the full backend, `cuda:0` and the exact verified model folder. `torch.compile` is disabled for Windows; the server field is empty. An earlier settings copy is retained locally. The checkpoint was not deserialized or initialized. Initialization later uses upstream `torch.load(..., weights_only=False)`, so retain the verified source and hash.

The backend and extension code use Apache-2.0. The model weights use CC BY-NC-SA 4.0 and remain a noncommercial research resource. These assets provide annotation assistance; they have no established ethmoid-cell counting accuracy here.

## Not selected

- **PanSeg:** native Windows support exists, but the full microscopy application adds a separate environment and model downloads. Its boundary-partitioning methods remain a research lead; its pretrained microscopy workflow is not selected for sinus CT.
- **CT-SCOPE / Craneal_CT:** supplies surrounding-bone masks, not individual pocket identities. Zenodo restricts file access, but the authors' GitHub repository provides public DICOM files and PNG masks. Creating individual-cell references would need extra annotation work, so it is not downloaded or integrated. See the [dataset screening](V2_DATASET_RESEARCH.md).
- **SinusSegment:** supplies a broad combined sinonasal mask. Licensing and inference defects remain unresolved; it is not installed or integrated.
- **Scite:** optional public-literature research integration. The catalog still reports it uninstalled. Installation and account connection require the user's Codex app flow; the available integration tools cannot complete those steps. Existing web tools remain usable.

## Stack retained for V2

| Layer | Choice | Purpose |
| --- | --- | --- |
| Review website | React 19.2.8, TypeScript 5.9.3, Vite 8.2.2, ordinary CSS | Cases and Case Viewer with review controls; three primary views maximum. |
| Browser CT viewer | NiiVue 0.69.0 | CT slices, overlays, candidate labels and orientation. |
| Local API | Python 3.13.12, FastAPI 0.141.1, Uvicorn 0.52.4 | One lightweight review application and its analysis interface. |
| Conversion and verification | dcm2niix, pydicom, nibabel, NumPy | Preserve source identity, geometry and HU verification. The existing app's pydicom 3.0.1 lock was not changed; the new utility skill uses its isolated 3.0.2 environment. |
| Analysis package | Separate Python 3.12.14 runtime; NumPy, SciPy, scikit-image, PyTorch and existing TotalSegmentator landmarks | Preserve the native-grid baseline and later test cell separation. Landmark models do not supply individual ethmoid cells. |
| Persistence | SQLite and protected local files | Series/run/review identity, immutable histories and image artifacts. |
| Jobs | Existing single-worker local queue and isolated analysis subprocess | No new distributed service or queue. |
| Verification | pytest, Vitest, Testing Library, Playwright, GitHub Actions and the medical-data Git guard | Synthetic software checks and protected-source boundaries. |

The app environment, frontend dependencies, pinned conversion tool and counting runtime already exist. Installed versions were inspected without starting the application. Application startup can migrate state or prepare scans and is not a setup verification command.

## Local locations and limits

All added software, model files and verification records live outside Git under `%LOCALAPPDATA%/EthmoidComplexity/tools`:

- `review-setup`: setup manifest, probes, package freeze, saved settings and checks.
- `review-assist/models/nnInteractive_v1.0`: verified public model files.
- `research-comparators/porespy`: isolated comparator environment and dependency/import records.
- `setup-utilities`: isolated utility environment. Use its `Scripts/python.exe` for pydicom skill commands, not the application's environment or counting runtime.

The two skills live under `%USERPROFILE%/.codex/skills`. No patient data, private research context, credentials or new model-provider configuration were sent to these public downloads. Review-tool settings do not establish a network-isolated environment or approve cloud hosting.

The pydicom skill's anonymizer does not establish owner-only Windows key permissions. Do not run key generation or anonymization without separate authorization and Windows ACL verification. Optional pixel codecs, clinical performance, model inference and hosting were not tested.

The later authorized bounded synthetic counting experiment is complete; see its [frozen design](SYNTHETIC_COUNTING_EXPERIMENT.md) and the [current results](../README.md). This setup creates no new questionnaire and does not require exhaustive clinician annotations before engineering tests. Real anatomical accuracy still needs correctly matched expert evidence.

Primary sources: [Slicer extension installation API](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#download-and-install-extension), [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), [Slicer extension](https://github.com/coendevente/SlicerNNInteractive), [pinned model](https://huggingface.co/MIC-DKFZ/nnInteractive/tree/3f308d751c00644e4fde6f09c600264b393b21b5), [PoreSpy installation](https://porespy.org/installation.html), [selected skill source](https://github.com/K-Dense-AI/scientific-agent-skills/tree/a1043ef544e3a100c28e70a736b6cf32554d0677).
