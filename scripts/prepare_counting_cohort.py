"""Prepare the fixed local prototype cohort with the existing lossless converter.

Run with the acquisition application's Python 3.13 environment. A clean,
committed --pipeline-root is mandatory; inference dependencies are not needed.
Medical images and this script's manifest remain in the protected work root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


DEVELOPMENT = (
    ("Case003", "Series007"),
    ("Case009", "Series004"),
    ("Case015", "Series001"),
    ("Case021", "Series002"),
    ("Case025", "Series001"),
)
EXCLUDED_CASES = {"Case017", "Case022"}
PROTOCOL_VERSION = "ethmoid-count-prototype-v1"


def select_cohort(inventory: dict) -> list[tuple[str, str, str]]:
    """One sharp, thin, native axial series per held-out study, neutral order."""
    candidates = []
    for series in inventory["series"]:
        image_type = str(series.get("image_type", "")).upper()
        thickness = series.get("slice_thickness_mm")
        spacing = series.get("ipp_spacing_median_mm")
        if (
            series["case"] not in EXCLUDED_CASES
            and str(series.get("kernel", "")).upper() in {"BONEPLUS", "HR68S", "HR68F"}
            and all(term in image_type.split("\\") for term in ("ORIGINAL", "PRIMARY", "AXIAL"))
            and not any(term in image_type for term in ("MPR", "DERIVED", "REFORMATTED", "LOCALIZER"))
            and series.get("ipp_spacing_regular") is True
            and isinstance(thickness, (int, float)) and 0 < thickness <= 1.0
            and isinstance(spacing, (int, float)) and 0 < spacing <= 1.0
        ):
            candidates.append((series["case"], series["series"]))
    eligible = set(candidates)
    if not set(DEVELOPMENT).issubset(eligible):
        raise ValueError("The fixed development selections no longer qualify in the inventory.")
    used = {case for case, _ in DEVELOPMENT}
    held_out = []
    for case, series in sorted(eligible):
        if case not in used:
            held_out.append((case, series))
            used.add(case)
        if len(held_out) == 5:
            break
    if len(held_out) != 5:
        raise ValueError("Five separate eligible held-out studies are required.")
    return [(case, series, "development") for case, series in DEVELOPMENT] + [
        (case, series, "held_out") for case, series in held_out
    ]


def _verify_existing(directory, case, series, preflight):
    """Recheck existing data without changing conversion or clinical-QC files."""
    import nibabel as nib
    import numpy as np
    from ethmoid_pipeline.conversion import _selection_identity_sha256, _source_hash_manifest
    from ethmoid_pipeline.privacy import verify_sanitized_sidecar
    from ethmoid_pipeline.util import PipelineError, read_json, sha256_file
    from ethmoid_pipeline.verification import _full_volume_hu_metrics

    name = f"{case}_{series}"
    nifti = directory / f"{name}.nii"
    report = read_json(directory / "qc_report.json")
    conversion = read_json(directory / "conversion_report.json")
    hashes = read_json(directory / "source_hashes.json")
    if not report.get("checks") or not all(check.get("passed") is True for check in report["checks"]):
        raise PipelineError("Existing conversion lacks a fully passing automated QC report.")
    if (report.get("case"), report.get("series")) != (case, series):
        raise PipelineError("Existing QC does not match the selected neutral case and series.")
    if sha256_file(nifti) != report["metrics"]["nifti_sha256"]:
        raise PipelineError("Existing image changed after automated QC.")
    if _source_hash_manifest(preflight.records) != hashes["slices"]:
        raise PipelineError("Selected source files changed after conversion.")
    if _selection_identity_sha256(preflight.records) != conversion["selection_identity_sha256"]:
        raise PipelineError("Existing conversion belongs to a different source selection.")
    if verify_sanitized_sidecar(directory / f"{name}.json"):
        raise PipelineError("Existing sidecar failed its privacy allowlist.")
    image = nib.load(nifti, mmap=True)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode or not scode or not np.allclose(qform, sform, atol=1e-5, rtol=0):
        raise PipelineError("Existing image has invalid or inconsistent physical geometry.")
    expected_voxels = preflight.rows * preflight.columns * len(preflight.records)
    if len(image.shape) != 3 or int(np.prod(image.shape)) != expected_voxels:
        raise PipelineError("Existing image dimensions do not match the selected DICOM volume.")
    metrics = _full_volume_hu_metrics(image, preflight.records, np.linalg.inv(image.affine))
    if metrics["voxel_count"] != expected_voxels or metrics["nonfinite_count"] or metrics["max_absolute_error"] != 0:
        raise PipelineError("Existing image failed full-volume lossless HU verification.")
    return nifti, report, metrics, sha256_file(nifti)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, default=os.environ.get("ETHMOID_DICOM_ROOT"))
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("LOCALAPPDATA", "")) / "EthmoidComplexity" / "work")
    parser.add_argument("--pipeline-root", type=Path, required=True, help="Clean committed checkout of the acquisition converter.")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata selections without converting or writing a manifest.")
    parser.add_argument("--reconvert-unverified", action="store_true", help="Preserve prior unverified conversions and retry them in a separate affine-v1 directory.")
    args = parser.parse_args(argv)
    if args.dicom_root is None:
        parser.error("--dicom-root or ETHMOID_DICOM_ROOT is required")
    if not (args.pipeline_root / "src" / "ethmoid_pipeline" / "conversion.py").is_file():
        parser.error("--pipeline-root must contain the existing converter")
    sys.path.insert(0, str(args.pipeline_root.resolve() / "src"))

    from ethmoid_pipeline.conversion import convert_series
    from ethmoid_pipeline.dicom import preflight_series
    from ethmoid_pipeline.inventory import resolve_selection
    from ethmoid_pipeline.provenance import current_code_provenance
    from ethmoid_pipeline.util import atomic_write_json, default_dcm2niix, read_json, sha256_file, validate_roots
    from ethmoid_pipeline.verification import verify_conversion

    validate_roots(args.dicom_root, args.work_root, args.pipeline_root)
    provenance = current_code_provenance(require_clean=True)
    inventory_path = args.work_root / "inventory" / "inventory.redacted.json"
    cohort = select_cohort(read_json(inventory_path))
    selections = []
    # Preflight the complete cohort before starting any conversion.
    for case, series, split in cohort:
        paths, study_uid, series_uid = resolve_selection(args.dicom_root, args.work_root, case, series)
        preflight = preflight_series(paths, study_uid, series_uid)
        selections.append((case, series, split, paths, study_uid, series_uid, preflight))
    if args.dry_run:
        print(json.dumps({"status": "PASS", "dry_run": True, "development": 5, "held_out": 5, "selections": [{"case": c, "series": s, "split": split} for c, s, split in cohort]}))
        return 0

    manifest_path = args.work_root / "counting" / "cohort.v1.local.json"
    frozen = read_json(manifest_path) if manifest_path.exists() else None
    frozen_entries = {(entry["case"], entry["series"]): entry for entry in frozen["entries"]} if frozen else {}
    if frozen and (
        len(frozen_entries) != len(cohort)
        or {(entry["case"], entry["series"], entry["split"]) for entry in frozen["entries"]} != set(cohort)
    ):
        raise RuntimeError("The existing frozen cohort has different selections.")
    entries = []
    for case, series, split, paths, study_uid, series_uid, preflight in selections:
        name = f"{case}_{series}"
        if (case, series) in frozen_entries:
            frozen_image = (args.work_root / frozen_entries[(case, series)]["nifti"]).resolve(strict=True)
            frozen_image.relative_to(args.work_root.resolve())
            if frozen_image.name != name + ".nii":
                raise RuntimeError("The frozen cohort image has an unexpected neutral filename.")
            destination = frozen_image.parent
        elif (case, series) == ("Case003", "Series007"):
            destination = args.work_root / "pilot" / name
        else:
            destination = args.work_root / "counting" / "cohort" / name
            if args.reconvert_unverified and destination.exists() and not (destination / "qc_report.json").is_file():
                # Preserve the earlier converter output for investigation. The
                # new attempt must pass every existing geometry and HU gate.
                destination = args.work_root / "counting" / "cohort-affine-v1" / name
        reused = destination.exists()
        if not reused:
            convert_series(args.dicom_root, args.work_root, paths, case, series, default_dcm2niix(), expected_study_uid=study_uid, expected_series_uid=series_uid, final_dir=destination, pipeline_provenance=provenance, require_clean_provenance=True)
            report = verify_conversion(args.dicom_root, args.work_root, paths, case, series, expected_study_uid=study_uid, expected_series_uid=series_uid, artifact_dir=destination, pipeline_provenance=provenance, require_clean_provenance=True)
            if not all(check["passed"] for check in report["checks"]):
                raise RuntimeError("A new conversion failed automated QC.")
        nifti, report, metrics, image_hash = _verify_existing(destination, case, series, preflight)
        entries.append({"case": case, "series": series, "split": split, "nifti": nifti.relative_to(args.work_root).as_posix(), "nifti_sha256": image_hash, "native_spacing_mm": [*preflight.pixel_spacing, preflight.slice_spacing], "image_count": len(preflight.records), "kernel": preflight.kernel, "automated_qc": "PASS", "hu_max_absolute_error": metrics["max_absolute_error"], "clinical_review": report.get("clinical_series_and_coverage_review", "PENDING"), "manual_slicer_qc": report.get("manual_slicer_qc", "PENDING")})
        entries[-1]["conversion_provenance"] = read_json(destination / "tool_provenance.json")["pipeline"]
        print(json.dumps({"case": case, "series": series, "split": split, "conversion": "reused_verified" if reused else "created_verified", "full_volume_hu": "PASS"}), flush=True)

    manifest = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "development_count": 5, "held_out_count": 5, "selection_policy": "fixed five development studies; first five other eligible studies in neutral order", "excluded_cases": sorted(EXCLUDED_CASES), "converter_provenance": provenance, "inventory_sha256": sha256_file(inventory_path), "source_identity_mapping": "inventory/uid_map.local.json", "entries": entries}
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("The existing cohort manifest differs; do not overwrite a frozen cohort.")
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "development": 5, "held_out": 5, "full_volume_hu_checks": len(entries), "manifest": "counting/cohort.v1.local.json"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Detailed real-data exceptions can contain protected local paths.
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "message": "Local cohort preparation failed; no source details were printed."}), file=sys.stderr)
        raise SystemExit(2)
