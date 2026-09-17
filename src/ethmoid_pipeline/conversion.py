from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from .dicom import (
    PreflightResult,
    assert_pilot_fingerprint,
    preflight_series,
    sensitive_values,
)
from .inventory import resolve_selection
from .geometry import prove_native_voxel_bijection
from .privacy import redact_text, sanitize_sidecar, scrub_nifti_header
from .provenance import current_code_provenance
from .util import (
    PipelineError,
    atomic_write_json,
    atomic_write_text,
    is_within,
    repo_root,
    sha256_file,
    sha256_canonical_json,
    utc_now,
    validate_labels,
    validate_roots,
)


DCM2NIIX_VERSION = "v1.0.20260724"
DCM2NIIX_ARCHIVE_SHA256 = "8310549ef7340b1b169c26f4721aeac69423fd73643fbbd22eec47b36cc70ada"
DCM2NIIX_EXE_SHA256 = "de2466c1e31bb0114ad2c5a3fdfa08be62447b25beb4f4f6a426c314797e48d8"
SLICER_VERSION = "5.12.3"
SLICER_REVISION = 34627


class GeometryReconciliationFailure(PipelineError):
    """The selected series cannot meet the permitted physical geometry policy."""
    def __init__(self, message: str, *, diagnostics: dict | None = None):
        self.diagnostics = diagnostics
        super().__init__(message)
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


def _write_manual_checklist(
    path: Path, case_label: str, series_label: str, slice_count: int
) -> None:
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
- [ ] Scroll the complete {slice_count}-slice stack and confirm no missing, duplicate, reordered, blank, localizer, or derived slices.
- [ ] Sanity-check representative air, soft-tissue/water, and bone HU values.
- [ ] Engineer confirms geometry, intensity, and provenance.
- [ ] Study clinician confirms the intended series and complete ethmoid-region coverage.

Do not save a Slicer scene bundle, screenshot, database, or log to Git. Any failure stops work before segmentation.
"""
    atomic_write_text(path, text)


def _validate_selected_paths(dicom_root: Path, paths: Iterable[Path]) -> list[Path]:
    source = dicom_root.resolve(strict=True)
    selected: list[Path] = []
    for path in paths:
        try:
            candidate = Path(path).resolve(strict=True)
        except OSError as exc:
            raise PipelineError("A selected DICOM path is unavailable.") from exc
        if not candidate.is_file() or not is_within(candidate, source):
            raise PipelineError("A selected DICOM path is outside the source root.")
        selected.append(candidate)
    if not selected:
        raise PipelineError("No source files were selected.")
    return selected


def reconcile_nifti_affine(path: Path, preflight: PreflightResult) -> dict[str, Any]:
    """Reconcile qform/sform against separately validated source geometry.

    Some decimal-spacing acquisitions get an sform step computed from rounded
    float32 positions, while their qform retains the correct spacing. At least
    one original form must already match the source grid to the unchanged
    0.001-voxel QC tolerance. The other may differ by at most 0.05 voxel across
    the complete volume. Larger disagreement is a failure, never a repair.
    Only header bytes are written; encoded voxel bytes and HU scaling remain
    identical. Full source-to-image HU verification still runs afterward.

    Exploratory quantized MPR uses its recorded fitted-grid policy instead:
    original forms must preserve the identical discrete voxel assignment, and
    both serialized fitted forms must satisfy the 0.001 mm source-corner cap.
    This separate policy never resamples the pixel payload.
    """
    image = nib.load(path, mmap=True)
    if not isinstance(image, nib.Nifti1Image) or len(image.shape) != 3:
        raise PipelineError("Affine reconciliation requires one uncompressed 3D NIfTI-1 image.")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode or not scode:
        raise PipelineError("Affine reconciliation cannot replace missing physical geometry.")
    record = preflight.records[0]
    lps_to_ras = np.asarray([-1.0, -1.0, 1.0])
    row_direction = np.asarray(record.iop[:3])
    column_direction = np.asarray(record.iop[3:])
    source_affine = np.eye(4)
    source_affine[:3, 0] = lps_to_ras * column_direction * preflight.pixel_spacing[0]
    source_affine[:3, 1] = lps_to_ras * row_direction * preflight.pixel_spacing[1]
    normal = np.cross(row_direction, column_direction)
    if preflight.grid_policy is not None:
        normal /= np.linalg.norm(normal)
    source_affine[:3, 2] = lps_to_ras * normal * preflight.slice_spacing
    source_affine[:3, 3] = lps_to_ras * np.asarray(preflight.fitted_origin_lps if preflight.fitted_origin_lps is not None else record.ipp)
    mapping = np.linalg.inv(qform) @ source_affine
    axes = np.argmax(np.abs(mapping[:3, :3]), axis=0)
    if len(set(int(axis) for axis in axes)) != 3:
        raise PipelineError("Converted image does not have a lossless source-axis mapping.")
    index_mapping = np.zeros((4, 4))
    index_mapping[3, 3] = 1
    source_shape = (preflight.rows, preflight.columns, len(preflight.records))
    for source_axis, image_axis in enumerate(axes):
        if image.shape[image_axis] != source_shape[source_axis]:
            raise PipelineError("Converted image dimensions do not match the source-axis mapping.")
        direction = 1 if mapping[image_axis, source_axis] >= 0 else -1
        index_mapping[image_axis, source_axis] = direction
        index_mapping[image_axis, 3] = 0 if direction > 0 else image.shape[image_axis] - 1
    if np.max(np.abs(mapping[:3, :3] - index_mapping[:3, :3])) > 1e-3:
        raise PipelineError("Converted image axes disagree with validated source geometry.")
    canonical = source_affine @ np.linalg.inv(index_mapping)
    corners = np.asarray([(*corner, 1.0) for corner in product(*[(0, size - 1) for size in image.shape])]).T
    inverse_canonical = np.linalg.inv(canonical)
    residuals = [float(np.max(np.abs((inverse_canonical @ form @ corners - corners)[:3]))) for form in (qform, sform)]
    original_geometry = None
    if preflight.grid_policy is not None:
        canonical_proof = prove_native_voxel_bijection(image.shape, canonical, preflight.records, physical_limit_mm=0.001)
        original_proofs = [prove_native_voxel_bijection(image.shape, form, preflight.records) for form in (qform, sform)]
        if any(proof["assignment"] != canonical_proof["assignment"] for proof in original_proofs):
            raise PipelineError("Original image geometry does not preserve the fitted native voxel assignment.")
        original_geometry = {"qform_max_world_residual_mm": original_proofs[0]["max_world_residual_mm"],
                             "sform_max_world_residual_mm": original_proofs[1]["max_world_residual_mm"],
                             "discrete_native_assignment_unchanged": True}
    elif min(residuals) > 1e-3 or max(residuals) > 0.05:
        raise PipelineError("Converted physical geometry differs beyond bounded affine precision reconciliation.")
    result: dict[str, Any] = {
        "method": "exploratory-mpr-quantized-grid-v1" if preflight.grid_policy is not None else "validated-dicom-grid-v1",
        "applied": preflight.grid_policy is not None or max(residuals) > 1e-3,
        "original_qform_residual_voxel": residuals[0],
        "original_sform_residual_voxel": residuals[1],
        "voxel_payload_unchanged": True,
    }
    if original_geometry is not None:
        result["grid_policy"] = preflight.grid_policy
        result["original_geometry"] = original_geometry
    if not result["applied"]:
        return result
    # nib.load consumes scl_slope/scl_inter in its in-memory header. Read the
    # original on-disk header to preserve those bytes when writing geometry.
    with path.open("rb") as stream:
        header = nib.Nifti1Header.from_fileobj(stream)
    offset = int(header["vox_offset"])
    if offset < 352 or len(image.header.extensions):
        raise PipelineError("Unexpected NIfTI layout prevents header-only affine reconciliation.")

    def payload_hash() -> str:
        with path.open("rb") as stream:
            stream.seek(offset)
            return hashlib.file_digest(stream, "sha256").hexdigest()

    before = payload_hash()
    header.set_qform(canonical, int(qcode), strip_shears=False)
    header.set_sform(canonical, int(scode))
    with path.open("r+b") as stream:
        stream.write(header.binaryblock)
        stream.flush()
        os.fsync(stream.fileno())
    if payload_hash() != before:
        raise PipelineError("Voxel bytes changed during header-only affine reconciliation.")
    result["voxel_payload_sha256"] = before
    if preflight.grid_policy is not None:
        reloaded = nib.load(path, mmap=True)
        checks = []
        for name, form in (("qform", reloaded.get_qform()), ("sform", reloaded.get_sform())):
            try:
                checks.append(prove_native_voxel_bijection(reloaded.shape, form, preflight.records, physical_limit_mm=0.001))
            except PipelineError as error:
                if hasattr(error, "diagnostics"):
                    error.diagnostics.update(stage="reloaded_geometry", form=name)
                raise
        if any(check["assignment"] != canonical_proof["assignment"] for check in checks):
            raise PipelineError("Serialized fitted geometry changed the native voxel assignment.")
        result["reloaded_geometry"] = {"qform_max_world_residual_mm": checks[0]["max_world_residual_mm"],
                                       "sform_max_world_residual_mm": checks[1]["max_world_residual_mm"],
                                       "one_to_one_native_voxels": True}
    return result


def _convert_preflighted_series(
    *,
    work_root: Path,
    preflight: PreflightResult,
    case_label: str,
    series_label: str,
    dcm2niix: Path,
    final_dir: Path,
    inventory_map_path: Path | None,
    pipeline_provenance: dict[str, Any] | None,
    require_clean_provenance: bool,
) -> dict[str, Any]:
    code = (
        dict(pipeline_provenance)
        if pipeline_provenance is not None
        else current_code_provenance(require_clean=require_clean_provenance)
    )
    tool = _verify_dcm2niix(dcm2niix)

    neutral_name = f"{case_label}_{series_label}"
    final_dir = final_dir.resolve()
    work = work_root.resolve()
    if final_dir == work or not is_within(final_dir, work):
        raise PipelineError("The conversion artifact directory must be inside the work root.")
    if final_dir.exists():
        raise PipelineError("The conversion output already exists; refusing to overwrite it.")
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
        inventory_map_sha256 = (
            sha256_file(inventory_map_path) if inventory_map_path is not None else None
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

        try:
            affine_reconciliation = reconcile_nifti_affine(nifti, preflight)
        except PipelineError as error:
            raise GeometryReconciliationFailure(str(error), diagnostics=getattr(error, "diagnostics", None)) from error
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
        conversion_report = {
            "schema_version": 1,
            "status": "PASS",
            "generated_at_utc": utc_now(),
            "case": case_label,
            "series": series_label,
            "input_slices": len(preflight.records),
            "output_nifti": nifti.name,
            "output_sidecar": sidecar.name,
            "source_hashes_unchanged": True,
            "ordered_source_manifest_sha256": source_manifest_sha256,
            "selection_identity_sha256": selection_identity_sha256,
            "affine_reconciliation": affine_reconciliation,
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
        }
        if inventory_map_sha256 is not None:
            conversion_report["inventory_uid_map_sha256"] = inventory_map_sha256
        atomic_write_json(run_dir / "conversion_report.json", conversion_report)
        _write_manual_checklist(
            run_dir / "SLICER_QC_CHECKLIST.md",
            case_label,
            series_label,
            len(preflight.records),
        )
        shutil.rmtree(stage_dir)
        shutil.rmtree(output_dir)
        os.replace(run_dir, final_dir)
        return {
            "status": "PASS",
            "case": case_label,
            "series": series_label,
            "output_dir": final_dir,
            "nifti": final_dir / nifti.name,
            "ordered_source_manifest_sha256": source_manifest_sha256,
            "selection_identity_sha256": selection_identity_sha256,
        }
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("dcm2niix exceeded the 10-minute conversion timeout.") from exc
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def convert_series(
    dicom_root: Path,
    work_root: Path,
    paths: Iterable[Path],
    case_label: str,
    series_label: str,
    dcm2niix: Path,
    *,
    expected_study_uid: str | None = None,
    expected_series_uid: str | None = None,
    final_dir: Path | None = None,
    pipeline_provenance: dict[str, Any] | None = None,
    require_clean_provenance: bool = True,
    preflight_profile: str = "strict",
) -> dict[str, Any]:
    """Convert one caller-selected clean CT series without pilot fingerprinting."""
    validate_labels(case_label, series_label)
    validate_roots(dicom_root, work_root, repo_root())
    work_root.mkdir(parents=True, exist_ok=True)
    validate_roots(dicom_root, work_root, repo_root())
    selected = _validate_selected_paths(dicom_root, paths)
    preflight = preflight_series(selected, expected_study_uid, expected_series_uid, profile=preflight_profile)
    neutral_name = f"{case_label}_{series_label}"
    destination = final_dir or (work_root / "conversions" / neutral_name)
    return _convert_preflighted_series(
        work_root=work_root,
        preflight=preflight,
        case_label=case_label,
        series_label=series_label,
        dcm2niix=dcm2niix,
        final_dir=destination,
        inventory_map_path=None,
        pipeline_provenance=pipeline_provenance,
        require_clean_provenance=require_clean_provenance,
    )


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
    neutral_name = f"{case_label}_{series_label}"
    final_dir = work_root / "pilot" / neutral_name
    return _convert_preflighted_series(
        work_root=work_root,
        preflight=preflight,
        case_label=case_label,
        series_label=series_label,
        dcm2niix=dcm2niix,
        final_dir=final_dir,
        inventory_map_path=work_root / "inventory" / "uid_map.local.json",
        pipeline_provenance=None,
        require_clean_provenance=True,
    )
