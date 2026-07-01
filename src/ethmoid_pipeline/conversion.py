from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .dicom import assert_pilot_fingerprint, preflight_series, sensitive_values
from .inventory import resolve_selection
from .privacy import redact_text, sanitize_sidecar, scrub_nifti_header
from .provenance import current_code_provenance
from .util import (
    PipelineError,
    atomic_write_json,
    atomic_write_text,
    repo_root,
    sha256_file,
    sha256_canonical_json,
    utc_now,
    validate_roots,
)


DCM2NIIX_VERSION = "v1.0.20260724"
DCM2NIIX_ARCHIVE_SHA256 = "8310549ef7340b1b169c26f4721aeac69423fd73643fbbd22eec47b36cc70ada"
DCM2NIIX_EXE_SHA256 = "de2466c1e31bb0114ad2c5a3fdfa08be62447b25beb4f4f6a426c314797e48d8"
SLICER_VERSION = "5.12.3"
SLICER_REVISION = 34627
SLICER_INSTALLER_SHA512 = (
    "109508d54b204f75371f6ae99b9951993a6b753a91c855cd223fcb8eeb65b37b"
    "ac4f1096d13c1cae89b4d525910b238239a5fb05c98d49874ff090315cd19315"
)


def _source_hash_manifest(records: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {"slice_index": index, "sha256": sha256_file(record.path)}
        for index, record in enumerate(records)
    ]


def _selection_identity_sha256(records: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for value in (
        records[0].study_uid,
        records[0].series_uid,
        *(record.sop_uid for record in records),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _verify_dcm2niix(dcm2niix: Path) -> dict[str, Any]:
    executable = dcm2niix.resolve(strict=True)
    if not executable.is_file():
        raise PipelineError("The pinned dcm2niix executable is unavailable.")
    archive = executable.parent.parent / "dcm2niix_win.zip"
    if not archive.is_file() or sha256_file(archive) != DCM2NIIX_ARCHIVE_SHA256:
        raise PipelineError("The pinned dcm2niix release archive is missing or has the wrong checksum.")
    executable_hash = sha256_file(executable)
    if executable_hash != DCM2NIIX_EXE_SHA256:
        raise PipelineError("The extracted dcm2niix executable has the wrong checksum.")
    completed = subprocess.run(
        [str(executable), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    version_text = completed.stdout + completed.stderr
    # This release intentionally returns 3 for --version on Windows; the
    # verified hashes and reported version string are the authoritative gates.
    if DCM2NIIX_VERSION not in version_text:
        raise PipelineError("The dcm2niix executable does not report the pinned version.")
    return {
        "version": DCM2NIIX_VERSION,
        "archive_sha256": DCM2NIIX_ARCHIVE_SHA256,
        "executable_sha256": executable_hash,
    }


def _slicer_provenance() -> dict[str, Any]:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return {"version": SLICER_VERSION, "revision": SLICER_REVISION, "installed": False}
    root = Path(local) / "EthmoidComplexity" / "tools" / "slicer" / "5.12.3-r34627"
    installer = root / "Slicer-5.12.3-win-amd64.exe"
    application = root / "app" / "Slicer.exe"
    installer_verified = False
    if installer.is_file():
        import hashlib

        digest = hashlib.sha512()
        with installer.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        installer_verified = digest.hexdigest() == SLICER_INSTALLER_SHA512
    return {
        "version": SLICER_VERSION,
        "revision": SLICER_REVISION,
        "installed": application.is_file(),
        "installer_sha512": SLICER_INSTALLER_SHA512,
        "installer_verified": installer_verified,
    }


def _dangerous_log_categories(text: str) -> list[str]:
    lowered = text.lower()
    categories: list[str] = []
    patterns = {
        "converter warning": "warning",
        "converter error": "error",
        "unequal spacing": "unequal",
        "gantry tilt handling": "gantry tilt",
        "inconsistent geometry": "inconsistent",
        "missing images": "missing image",
        "failed stacking": "not stacked",
    }
    for category, needle in patterns.items():
        if needle in lowered:
            categories.append(category)
    return categories


def build_dcm2niix_command(
    executable: Path, output_dir: Path, neutral_name: str, stage_dir: Path
) -> list[str]:
    """Return the complete pinned argv; paths are passed without shell interpolation."""
    # Deliberately omit -m: the pinned default auto mode preserves ordinary CT
    # stacking without forcing incompatible frames or splitting tube-current
    # modulation. Preflight independently requires one regular SeriesUID grid.
    return [
        str(executable.resolve()),
        "-g",
        "i",
        "-a",
        "y",
        "-b",
        "y",
        "-ba",
        "y",
        "-c",
        "",
        "-d",
        "0",
        "-e",
        "n",
        "-f",
        neutral_name,
        "-i",
        "n",
        "-l",
        "o",
        "-o",
        str(output_dir),
        "-p",
        "y",
        "-q",
        "n",
        "-r",
        "n",
        "-s",
        "n",
        "-v",
        "1",
        "-w",
        "0",
        "-x",
        "i",
        "-z",
        "n",
        str(stage_dir),
    ]


def _write_manual_checklist(path: Path, case_label: str, series_label: str) -> None:
    text = f"""# Manual Slicer QC: {case_label}/{series_label}

Status: PENDING
Engineer reviewer/date: ____________________
Clinical reviewer/date: ____________________
Help/About version and revision: ____________________
Installer SHA-512 verified: ____________________

- [ ] Use Slicer {SLICER_VERSION} revision {SLICER_REVISION}; record Help/About and verify installer SHA-512 `{SLICER_INSTALLER_SHA512}`.
- [ ] Record DICOM-loaded and NIfTI dimensions, spacing, directions, IJK-to-RAS, and physical bounds.
- [ ] Overlay/fade in axial, coronal, and sagittal views; check first, middle, and last slices.
- [ ] Confirm L/R, A/P, and S/I orientation with no translation, reflection, rotation, or truncation.
- [ ] Scroll the complete 224-slice stack and confirm no missing, duplicate, reordered, blank, localizer, or derived slices.
- [ ] Sanity-check representative air, soft-tissue/water, and bone HU values.
- [ ] Engineer confirms geometry, intensity, and provenance.
- [ ] Study clinician confirms the intended series and complete ethmoid-region coverage.

Do not save a Slicer scene bundle, screenshot, database, or log to Git. Any failure stops work before segmentation.
"""
    atomic_write_text(path, text)


def convert_pilot(
    dicom_root: Path,
    work_root: Path,
    case_label: str,
    series_label: str,
    dcm2niix: Path,
) -> dict[str, Any]:
    validate_roots(dicom_root, work_root, repo_root())
    work_root.mkdir(parents=True, exist_ok=True)
    validate_roots(dicom_root, work_root, repo_root())
    paths, study_uid, series_uid = resolve_selection(
        dicom_root, work_root, case_label, series_label
    )
    preflight = preflight_series(paths, study_uid, series_uid)
    assert_pilot_fingerprint(preflight, case_label, series_label)
    code = current_code_provenance(require_clean=True)
    tool = _verify_dcm2niix(dcm2niix)

    neutral_name = f"{case_label}_{series_label}"
    final_dir = work_root / "pilot" / neutral_name
    if final_dir.exists():
        raise PipelineError("The pilot output already exists; refusing to overwrite it.")
    run_parent = work_root / "_runs"
    run_parent.mkdir(parents=True, exist_ok=True)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{neutral_name}-", dir=run_parent))
    stage_dir = run_dir / "stage"
    output_dir = run_dir / "converted"
    stage_dir.mkdir()
    output_dir.mkdir()

    try:
        hashes_before = _source_hash_manifest(preflight.records)
        source_manifest_sha256 = sha256_canonical_json(hashes_before)
        selection_identity_sha256 = _selection_identity_sha256(preflight.records)
        inventory_map_sha256 = sha256_file(
            work_root / "inventory" / "uid_map.local.json"
        )
        for index, record in enumerate(preflight.records):
            shutil.copyfile(record.path, stage_dir / f"slice_{index:06d}.dcm")

        command = build_dcm2niix_command(
            dcm2niix, output_dir, neutral_name, stage_dir
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        combined_log = completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr
        safe_log = redact_text(
            combined_log,
            sensitive_values(record.path for record in preflight.records),
            (str(run_dir), str(dcm2niix.resolve())),
        )
        if completed.returncode != 0:
            raise PipelineError("dcm2niix returned a nonzero exit code.")
        dangerous = _dangerous_log_categories(combined_log)
        if dangerous:
            raise PipelineError(
                "dcm2niix reported a blocked condition: " + ", ".join(dangerous) + "."
            )

        nifti_files = list(output_dir.glob("*.nii"))
        sidecars = list(output_dir.glob("*.json"))
        other_files = [path for path in output_dir.iterdir() if path not in nifti_files + sidecars]
        if len(nifti_files) != 1 or len(sidecars) != 1 or other_files:
            raise PipelineError("dcm2niix did not produce exactly one NIfTI and one JSON sidecar.")
        nifti = nifti_files[0]
        sidecar = sidecars[0]
        if nifti.stem != neutral_name or sidecar.stem != neutral_name:
            raise PipelineError("dcm2niix added an unexpected output postfix or filename.")

        sanitize_sidecar(sidecar)
        scrub_nifti_header(nifti)
        hashes_after = _source_hash_manifest(preflight.records)
        if hashes_after != hashes_before:
            raise PipelineError("At least one source DICOM hash changed during conversion.")

        shutil.move(str(nifti), run_dir / nifti.name)
        shutil.move(str(sidecar), run_dir / sidecar.name)
        atomic_write_text(run_dir / "conversion.log", safe_log)
        atomic_write_json(
            run_dir / "source_hashes.json",
            {
                "ordered_manifest_sha256": source_manifest_sha256,
                "slices": hashes_before,
            },
        )
        atomic_write_json(
            run_dir / "preflight_report.json",
            preflight.safe_report(case_label, series_label),
        )
        atomic_write_json(
            run_dir / "tool_provenance.json",
            {
                "schema_version": 1,
                "generated_at_utc": utc_now(),
                "dcm2niix": tool,
                "slicer": _slicer_provenance(),
                "pipeline": code,
            },
        )
        atomic_write_json(
            run_dir / "conversion_report.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "generated_at_utc": utc_now(),
                "case": case_label,
                "series": series_label,
                "input_slices": len(preflight.records),
                "output_nifti": nifti.name,
                "output_sidecar": sidecar.name,
                "source_hashes_unchanged": True,
                "inventory_uid_map_sha256": inventory_map_sha256,
                "ordered_source_manifest_sha256": source_manifest_sha256,
                "selection_identity_sha256": selection_identity_sha256,
                "dcm2niix_options": [
                    "-g i",
                    "-a y",
                    "-b y",
                    "-ba y",
                    "-c <empty>",
                    "-d 0",
                    "-e n",
                    f"-f {neutral_name}",
                    "-i n",
                    "-l o",
                    "-p y",
                    "-q n",
                    "-r n",
                    "-s n",
                    "-v 1",
                    "-w 0",
                    "-x i",
                    "-z n",
                ],
                "warnings": list(preflight.warnings),
            },
        )
        _write_manual_checklist(run_dir / "SLICER_QC_CHECKLIST.md", case_label, series_label)
        shutil.rmtree(stage_dir)
        shutil.rmtree(output_dir)
        os.replace(run_dir, final_dir)
        return {
            "status": "PASS",
            "case": case_label,
            "series": series_label,
            "output_dir": final_dir,
            "nifti": final_dir / nifti.name,
        }
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("dcm2niix exceeded the 10-minute conversion timeout.") from exc
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
