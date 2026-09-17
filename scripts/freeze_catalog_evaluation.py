"""Snapshot every prepared local series for a separate protected evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3

from evaluate_catalog import REGRESSION_STUDIES, checks, normalized_acquisition, quality_helpers


def validate_current_source(row) -> None:
    """Read-only counterpart of the catalog's current-source countability guard."""
    if not row["present"] or row["linked_series_id"] != row["id"]:
        raise checks.EvaluationError("CATALOG_CURRENT_IDENTITY_INVALID")
    try:
        root = Path(row["root_path"]).resolve(strict=True)
        names = json.loads(row["files_json"])
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError
        paths = [(root / name).resolve(strict=True) for name in names]
        if len(set(paths)) != len(paths) or any(not path.is_file() or not path.is_relative_to(root) for path in paths):
            raise ValueError
        fingerprint = checks.canonical_sha256(sorted(checks.sha256(path) for path in paths))
    except (OSError, ValueError, TypeError):
        raise checks.EvaluationError("CATALOG_RAW_SOURCE_UNAVAILABLE") from None
    if fingerprint != row["source_fingerprint"]:
        raise checks.EvaluationError("CATALOG_RAW_SOURCE_CHANGED")


def freeze(work: Path, destination: Path) -> dict:
    protected = (Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work").resolve(strict=True)
    if work.resolve(strict=True) != protected:
        raise checks.EvaluationError("PROTECTED_WORK_ROOT_REQUIRED")
    destination = checks.contained(destination, protected, exists=False)
    if destination.exists():
        raise checks.EvaluationError("IMMUTABLE_OUTPUT_ALREADY_EXISTS")
    app = work / "webapp"
    with sqlite3.connect((app / "ethmoid-app.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        if db.execute("SELECT COUNT(*) FROM preparation_jobs WHERE status IN ('queued','preparing')").fetchone()[0]:
            raise checks.EvaluationError("PREPARATION_NOT_FINISHED")
        rows = db.execute("""SELECT s.*,t.study_label,c.volume_relpath,c.source_sha256,c.validation_status,
            l.acquisition_json,l.source_fingerprint AS linked_fingerprint,
            l.series_id AS linked_series_id,src.root_path
            FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id
            JOIN catalog_sources src ON src.id=t.source_id
            LEFT JOIN cases c ON c.id=s.case_id LEFT JOIN catalog_case_links l ON l.case_id=s.case_id
            ORDER BY t.study_label,s.series_label""").fetchall()
    if not rows or any(row["preparation_status"] not in {"ready", "unavailable"} for row in rows):
        raise checks.EvaluationError("PREPARATION_NOT_FINISHED")
    entries, unavailable = [], []
    cache_candidates = {}
    for parent in (work / "counting/evaluations", work / "counting/engine-validation", app / "cases"):
        if parent.is_dir():
            for cache_manifest in parent.rglob("localizers.local.json"):
                try:
                    value = checks.read_json(cache_manifest)
                    if value.get("weights_manifest_sha256") and value.get("totalsegmentator_version"):
                        cache_candidates.setdefault(value.get("source_sha256"), cache_manifest.parent)
                except checks.EvaluationError:
                    continue
    runtime_python = Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/tools/counting-runtime/Scripts/python.exe"
    runtime = checks._runtime_provenance(runtime_python) if cache_candidates and runtime_python.is_file() else None
    quality = quality_helpers(Path(__file__).resolve().parents[1])
    for row in rows:
        if row["preparation_status"] != "ready":
            unavailable.append({"study_label": row["study_label"], "series_label": row["series_label"],
                                "reason_code": row["reason_code"]})
            continue
        if row["validation_status"] != "VERIFIED" or not row["source_fingerprint"] or row["source_fingerprint"] != row["linked_fingerprint"]:
            raise checks.EvaluationError("CATALOG_QC_OR_IDENTITY_INVALID")
        validate_current_source(row)
        source = checks.contained(app / row["volume_relpath"], work)
        if checks.sha256(source) != row["source_sha256"]:
            raise checks.EvaluationError("CATALOG_SOURCE_CHANGED")
        try:
            acquisition = json.loads(row["acquisition_json"])
            if not isinstance(acquisition, dict):
                raise ValueError
        except (TypeError, ValueError):
            raise checks.EvaluationError("CATALOG_ACQUISITION_INVALID") from None
        acquisition.update(source_sha256=row["source_sha256"], study_label=row["study_label"], series_label=row["series_label"])
        acquisition.pop("warnings", None)
        entry = {"study_label": row["study_label"], "series_label": row["series_label"],
            "split": "regression" if row["study_label"] in REGRESSION_STUDIES else "acceptance",
            "nifti": source.relative_to(work).as_posix(), "source_sha256": row["source_sha256"],
            "dicom_fingerprint": row["source_fingerprint"], "automated_qc": "PASS", "acquisition": acquisition}
        entry["acquisition"] = normalized_acquisition(entry, quality)
        # Rejected historical caches remain untouched. Their absence from the
        # new manifest requests fresh inference, not repaired provenance.
        if runtime and row["source_sha256"] in cache_candidates:
            cache = checks.contained(cache_candidates[row["source_sha256"]], work)
            try:
                info = checks.validate_cache(cache, source, row["source_sha256"], work, runtime)
                if info and info["producer_provenance"] == "verified":
                    entry["localizer_cache"] = cache.relative_to(work).as_posix()
            except checks.EvaluationError:
                entry["historical_cache_rejected"] = True
        entries.append(entry)
    manifest = {"schema_version": 1, "studies": len({row["study_label"] for row in rows}),
                "series": len(rows), "entries": entries, "unavailable": unavailable,
                "accuracy_measured": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    checks.write_new_json(destination, manifest)
    return {"studies": manifest["studies"], "series": len(rows), "ready": len(entries),
            "unavailable": len(unavailable), "manifest_sha256": checks.sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = freeze(args.work_root, args.output)
    except checks.EvaluationError as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
