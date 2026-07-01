from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .conversion import convert_pilot
from .inventory import build_inventory
from .util import PipelineError, default_dcm2niix, default_work_root, validate_roots
from .verification import verify_pilot


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=os.environ.get("ETHMOID_DICOM_ROOT"),
        help="Read-only source DICOM root; alternatively set ETHMOID_DICOM_ROOT.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=default_work_root(),
        help="PHI-sensitive local work root outside Git.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ethmoid-pipeline",
        description="Local fail-closed DICOM-to-NIfTI pilot; no segmentation or bulk conversion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Build local UID mapping and redacted inventory.")
    _add_common_arguments(inventory)

    convert = subparsers.add_parser("convert", help="Convert the one approved pilot series.")
    _add_common_arguments(convert)
    convert.add_argument("--case", required=True)
    convert.add_argument("--series", required=True)
    convert.add_argument("--dcm2niix", type=Path, default=default_dcm2niix())

    verify = subparsers.add_parser("verify", help="Verify physical geometry, HU, and source hashes.")
    _add_common_arguments(verify)
    verify.add_argument("--case", required=True)
    verify.add_argument("--series", required=True)
    return parser


def _source_root(value: Path | str | None) -> Path:
    if value is None or str(value).strip() == "":
        raise PipelineError("Pass --dicom-root or set ETHMOID_DICOM_ROOT.")
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dicom_root = _source_root(args.dicom_root)
        validate_roots(dicom_root, args.work_root)
        if args.command == "inventory":
            result = build_inventory(dicom_root, args.work_root)
            safe = {
                "status": "PASS",
                "dicom_file_count": result["dicom_file_count"],
                "study_count": result["study_count"],
                "series_count": result["series_count"],
                "local_outputs": ["uid_map.local.json", "inventory.redacted.json"],
            }
            print(json.dumps(safe, indent=2))
            return 0
        if args.command == "convert":
            result = convert_pilot(
                dicom_root,
                args.work_root,
                args.case,
                args.series,
                args.dcm2niix,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "case": result["case"],
                        "series": result["series"],
                        "local_artifact": f"pilot/{result['case']}_{result['series']}",
                    },
                    indent=2,
                )
            )
            return 0
        report = verify_pilot(
            dicom_root, args.work_root, args.case, args.series
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "case": report["case"],
                    "series": report["series"],
                    "automated_checks": len(report["checks"]),
                    "blocking_issues": len(report["issues"]),
                    "manual_slicer_qc": report["manual_slicer_qc"],
                    "clinical_review": report["clinical_series_and_coverage_review"],
                },
                indent=2,
            )
        )
        return 0 if report["status"] == "AUTOMATED_PASS_PENDING_MANUAL" else 2
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Fail closed without echoing source paths, DICOM values, or exception
        # representations that may contain PHI. Developers reproduce failures
        # with synthetic data rather than enabling verbose real-data traces.
        print(
            "ERROR: unexpected local processing failure; detailed source metadata is not displayed.",
            file=sys.stderr,
        )
        return 3
