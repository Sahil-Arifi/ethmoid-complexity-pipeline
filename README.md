# Ethmoid Complexity Pipeline

This repository contains the first engineering milestone for the UCSD ethmoid-complexity project: a local, fail-closed DICOM-to-NIfTI conversion pilot with independent geometry and HU verification.

It does **not** segment anatomy, run SinusSegment, train a model, define a clinical complexity score, or convert the full dataset.

## Privacy boundary

Raw DICOM, the local UID map, staged files, NIfTI volumes, converter logs, Slicer state, screenshots, and QC outputs are PHI-sensitive. They must stay outside this repository and must not be uploaded. The code reads source DICOMs but never renames, deletes, or writes them.

Use three physically separate roots:

- This Git repository: source code and synthetic tests only.
- A read-only DICOM root supplied at runtime.
- A local work root outside Git, defaulting to `%LOCALAPPDATA%\EthmoidComplexity\work`.

See [SECURITY.md](SECURITY.md) before handling real data.

## Reproduce the environment

From PowerShell:

```powershell
.\scripts\bootstrap.ps1 -InstallSlicer
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
git config core.hooksPath .hooks
```

The bootstrap verifies the pinned dcm2niix and Slicer release archives before extracting or installing them. Exact Python packages are recorded in `requirements.lock`.

## Commands

Set the source location only in the current shell; do not save it in tracked configuration:

```powershell
$env:ETHMOID_DICOM_ROOT = '<read-only DICOM root>'
ethmoid-pipeline inventory
ethmoid-pipeline convert --case Case003 --series Series007
ethmoid-pipeline verify --case Case003 --series Series007
```

`inventory` assigns neutral labels using case-insensitive sorted relative paths and saves the UID/path mapping only in the local work root. `convert` is intentionally restricted to the audited pilot fingerprint. `verify` compares DICOM physical coordinates and rescaled pixel values directly with the generated NIfTI.

The process is automated only through engineering QC. Completion still requires the manual [3D Slicer checklist](docs/SLICER_QC.md) and clinical confirmation that the chosen series covers the intended anatomy.

## Tests

```powershell
python -m pytest
python scripts/check_staged.py
```

All test DICOMs are generated synthetically. Real medical data is never copied into the test tree or CI.
