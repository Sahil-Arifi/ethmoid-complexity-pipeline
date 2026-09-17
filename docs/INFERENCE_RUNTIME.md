# Local counting runtime and cohort

The acquisition application stays on Python 3.13. Counting runs in a separate
Windows x64 Python 3.12.14 environment under
`%LOCALAPPDATA%\EthmoidComplexity\tools\counting-runtime`. Source scans, converted
volumes, masks, logs, and cohort manifests remain in the protected local work
directory and must never be committed or uploaded. See [SECURITY.md](../SECURITY.md).

## Installation and offline operation

Run from a checkout containing the committed inference lock:

```powershell
.\scripts\bootstrap_inference.ps1
```

The script installs the hash-locked Windows dependencies, disables TotalSegmentator
usage telemetry **before inference**, downloads and hashes the four required model
sets, and runs real CUDA matrix-multiplication and 3D-convolution checks against
CPU results. The precision check explicitly disables TF32; the inference networks
retain their upstream inference behavior. `-RefreshLock` deliberately resolves a
new dependency lock and should be reviewed as a dependency change. Ordinary runs
use `uv pip sync --require-hashes`. `-SkipWeights` skips model download only.

Verified runtime: PyTorch 2.14.0+cu130, TotalSegmentator 2.18.0, nnUNet 2.8.1,
NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, nibabel 5.4.2, pytest 9.1.1.
The tested machine has a Ryzen 7 9800X3D, approximately 31 GiB RAM, and an RTX
5070 Ti with 16 GiB GPU memory. CUDA 13.0 and `sm_120` support were exercised.
No inference throughput claim applies to other hardware.

The environment installs a default-on `sitecustomize.py` Python audit hook that
blocks external DNS, socket connections, and datagrams while allowing loopback
for multiprocessing. Spawned Python workers inherit the guard. The bootstrap
temporarily sets `ETHMOID_INFERENCE_OFFLINE=0` only for public model downloads;
patient inference must use `ETHMOID_INFERENCE_OFFLINE=1`. This is a Python-process
guard, not an operating-system firewall for arbitrary native programs. TotalSeg's
separate `send_usage_stats=false` configuration is also mandatory.

The runtime Python is `counting-runtime\Scripts\python.exe`. The worker receives
the application's `src` directory on `PYTHONPATH`. Its local model/config paths are:

- `TOTALSEG_HOME_DIR`: `counting-runtime\totalsegmentator`
- `TOTALSEG_WEIGHTS_PATH`: `counting-runtime\totalsegmentator\nnunet\results`
- `counting-runtime\weights-manifest.json`: model source URLs, versions, and every file's SHA-256
- `counting-runtime\runtime-report.json`: actual package, hardware, CUDA, and smoke-check results

## Anatomical localizers

| Task | Model | Purpose and internal working resolution |
|---|---:|---|
| `craniofacial_structures` | 115 | Skull, maxillary/frontal sinus and other head landmarks; 0.5 mm |
| `head_glands_cavities` | 775 | Eyes, optic nerves, nasal cavities and other head landmarks; 0.75 × 0.75 × 1 mm |
| Robust crop and brain prior | 297 | 3 mm whole-body model locates the skull crop and supplies a brain mask, refined against native bone |
| Default crop | 298 | Upstream 6 mm crop dependency, prefetched for complete offline operation |

Run the two head localizers with `robust_crop=True`, `fast=False`, `device="gpu"`,
`nr_thr_resamp=2`, `nr_thr_saving=1`, `statistics=False`, and `preview=False`.
`nnUNet_compile=false` avoids compilation at first patient use. These networks
provide **anatomical landmarks, not ethmoid masks or individual air-cell labels**.
Their internal resampling does not change the counting algorithm's native grid.
Output masks must match the source shape and affine and contain only binary
values. The counting worker must verify the local model-file hashes and bind
cached landmarks to the source image hash.

The final counting worker additionally runs `task="total"`, `fast=True`,
`roi_subset=["brain"]` using task 297. This needs no extra model download.
GPU memory exhaustion permits one CPU fallback. Missing or clipped essential
landmarks permit one full-field retry with the same model settings. Neither
fallback introduces a manual ROI or changes counting thresholds. Fresh cache
manifests bind every task to the producer package and model-manifest hash;
historical partial manifests retain an explicit unverified-producer warning.

The verified Case003 pilot produced 7 craniofacial masks and 19 head-cavity masks
in 51.86 and 35.83 seconds respectively on the machine above. All 26 matched the
native 512 × 512 × 224 image grid and affine to `1e-5`; all were binary. Every
required counting landmark was nonempty; the unrelated hypopharynx class was
empty. This verifies execution and geometry, not anatomical accuracy. The masks
and their report are local under `work\counting\localizers\Case003`.

## Fixed cohort and conversion provenance

Run with the acquisition application's existing Python 3.13 environment and a
**clean, committed** converter checkout:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_counting_cohort.py `
  --pipeline-root $cleanConverterCheckout --dicom-root $env:ETHMOID_DICOM_ROOT --dry-run
.\.venv\Scripts\python.exe scripts\prepare_counting_cohort.py `
  --pipeline-root $cleanConverterCheckout --dicom-root $env:ETHMOID_DICOM_ROOT
```

Development selections are fixed: Case003/Series007, Case009/Series004,
Case015/Series001, Case021/Series002, Case025/Series001. Held-out selections are
the first five other eligible studies in neutral order: Case001/Series009,
Case002/Series009, Case004/Series005, Case005/Series009, Case006/Series007.
Cases017 and 022 remain excluded because of mixed-acquisition/spacing defects.
This is a split by study, never by slices or alternative reconstructions of the
same study. Eligibility currently reflects acquisition geometry; manual clinical
coverage and exclusion review remain separate pending states.

The script preflights the complete selection, reuses the existing pilot, converts
only missing outputs, and verifies full-volume HU equality and source/image
hashes. Existing outputs are never silently overwritten. Stored pilot clinical
and Slicer-review fields are preserved. A complete run freezes
`work\counting\cohort.v1.local.json`; any change to the frozen selections,
source image hashes, or converter provenance fails rather than overwriting it.
Each entry retains its actual original conversion provenance; the manifest also
records the clean converter revision used to verify and assemble the cohort.
All identity lookup remains in the existing protected UID map.

The prepared cohort now contains all ten verified volumes. Full-volume DICOM/HU
checks passed for every entry, with zero maximum HU error. The two decimal-spacing
reconversions use clean converter commit
`606fbea2946cd716353f3c0466121075cc94e31c`; previously verified images retain their
earlier recorded converter revisions. The original failed attempts were retained
separately. Clinical coverage and review states were not promoted by these checks.

After the affine precision fix, `--reconvert-unverified` preserves earlier
conversions that never obtained a QC report and writes a separate verified
attempt under `work\counting\cohort-affine-v1`. It does not overwrite the earlier
attempt or relax any check. Already verified images continue to be reused.
Subsequent runs use the frozen manifest's verified paths, including those separate
reconversions, without requiring the recovery flag again.

## Decimal-spacing affine reconciliation

The first additional conversions exposed a precision discrepancy in the pinned
converter: some regular 0.6 mm DICOM series retain accurate qform spacing while
their sform spacing is computed from rounded float32 positions. Over a volume,
this can violate the existing strict voxel-grid verification even though the
encoded image data are intact. For example, 0.600036621 mm versus 0.6 mm creates
about 0.01042 voxel of end-to-end disagreement in the 172-plane case.

The converter now derives a candidate affine from validated DICOM orientation,
spacing, and physical positions, preserving the converter's axis permutation
and flips. Reconciliation is permitted only when one existing form already
agrees with the source within the unchanged 0.001-voxel gate and neither differs
by more than 0.05 voxel across any volume corner. Larger discrepancies still fail.
Only NIfTI header geometry bytes are changed; encoded voxel bytes are hashed
before/after and HU slope/intercept are preserved. The conversion report records
whether reconciliation occurred. Full-volume HU and geometry verification still
run with their original tolerances. Correct existing geometry is not rewritten.

Synthetic regression tests cover the observed decimal-spacing failure, flipped
and permuted axes, an axial-oblique volume, unchanged correct images, rejection
when both original forms are wrong, and the actual pinned dcm2niix executable.

Sources: [TotalSegmentator installation and tasks](https://github.com/wasserth/TotalSegmentator),
[versioned task configuration](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/map_tasks_config.py),
[versioned telemetry configuration](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/config.py),
[official PyTorch installation](https://pytorch.org/get-started/locally/).
