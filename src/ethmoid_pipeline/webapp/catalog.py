"""Stable local study catalog layered over immutable legacy volume identities."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import threading
import uuid

from ..util import sha256_canonical_json, sha256_file, utc_now
from .catalog_config import CatalogConfig, load_local_config
from .catalog_validation import verified_volume
from .service import WebAppError
from .store import decode_json, encode_json


REASONS = {
    "SCOUT": "This is a scout image, not a three-dimensional CT volume.",
    "PROJECTION": "Projection or averaged images cannot provide a three-dimensional cell count.",
    "GEOMETRY_INVALID": "The images do not form one consistent three-dimensional CT volume.",
    "GEOMETRY_RECONCILIATION_UNRESOLVED": "The reconstructed geometry cannot match all source planes within the permitted physical limit. This series remains unavailable.",
    "MIXED_ACQUISITION": "This series mixes acquisitions or reconstruction settings and cannot form one reliable volume.",
    "ORIENTATION_INVALID": "Image orientations are inconsistent, nonorthogonal, or sheared within this series.",
    "DUPLICATE_PLANES": "Duplicate images or overlapping slice positions prevent a reliable three-dimensional volume.",
    "MISSING_PLANES": "This series has missing planes or too few spatial planes for a complete volume.",
    "SPACING_IRREGULAR": "Slice spacing changes within this series, so a lossless regular volume cannot be prepared.",
    "HU_UNCALIBRATED": "The images do not have consistent, verified CT Hounsfield-unit calibration.",
    "ENCODING_UNSUPPORTED": "This image encoding or image type is not supported by the verified converter.",
    "METADATA_INVALID": "Required CT geometry metadata is missing, unreadable, or invalid.",
    "PRIVACY_FLAG": "Source images contain an annotation or visual-identity flag and require local review before preparation.",
    "SOURCE_CHANGED": "The local source changed. Its previous volume and review history were retained.",
    "SOURCE_MISSING": "This series is no longer present in the configured local collection.",
    "SOURCE_UNAVAILABLE": "The local source files are currently unavailable.",
    "DISK_SPACE": "Preparation paused because the protected drive has insufficient free space.",
    "CONVERSION_FAILED": "The volume could not be prepared and verified. The local diagnostic log is retained.",
    "INTERRUPTED": "Preparation was interrupted and can be retried.",
    "CONFIGURATION": "The saved local collection configuration is unavailable or invalid.",
    "INVENTORY_INVALID": "The saved local inventory is incomplete or inconsistent.",
    "WORKER_STORAGE": "Preparation is waiting for local storage to recover. Queued jobs are retained.",
}
RETRYABLE = {"SOURCE_CHANGED", "SOURCE_UNAVAILABLE", "DISK_SPACE", "CONVERSION_FAILED", "INTERRUPTED"}


def source_kind(image_type: str) -> str:
    terms = set(re.split(r"[\\ /]+", image_type.upper()))
    if terms & {"LOCALIZER", "SCOUT", "TOPOGRAM"}:
        return "scout"
    if terms & {"AVERAGE", "MIP", "MINIP"}:
        return "projection"
    if terms & {"MPR", "REFORMATTED"}:
        return "mpr"
    if {"ORIGINAL", "PRIMARY"}.issubset(terms) and "DERIVED" not in terms:
        return "native"
    return "unknown"


def acquisition(metadata: dict) -> dict:
    spacing = metadata.get("spacing_mm")
    thickness = metadata.get("slice_thickness_mm")
    valid = isinstance(spacing, list) and len(spacing) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in spacing)
    thickness_valid = type(thickness) in (int, float) and math.isfinite(thickness) and thickness > 0
    resolution = "unknown"
    warnings = list(metadata.get("warnings", []))
    if valid and thickness_valid:
        resolution = "within_protocol" if metadata.get("source_kind") == "native" and max(spacing) <= 1.0 and thickness <= 1.0 else "outside_protocol"
    if resolution == "outside_protocol":
        warnings.append("Outside the clinician's thin native CT protocol; any count is an exploratory result.")
    elif resolution == "unknown":
        warnings.append("Acquisition resolution has not been established.")
    return {"spacing_mm": spacing if valid else None, "slice_thickness_mm": thickness if thickness_valid else None, "resolution_class": resolution, "warnings": list(dict.fromkeys(warnings)),
            "reconstruction_kind": metadata.get("source_kind") if metadata.get("source_kind") in {"native", "mpr"} else "unknown",
            "kernel_class": "bone" if metadata.get("kernel") in {"BONEPLUS", "Hr68s", "Hr68f"} else "soft" if metadata.get("kernel") == "STANDARD" else "unknown"}


def safe_descriptor(summary: dict) -> dict:
    pixels = summary.get("pixel_spacing_mm")
    spacing = [*pixels, summary.get("ipp_spacing_median_mm")] if isinstance(pixels, list) and len(pixels) == 2 else None
    kernel = str(summary.get("kernel") or "")
    kernel = kernel if re.fullmatch(r"BONEPLUS|STANDARD|Hr[0-9]+[a-z]?|[A-Z][0-9]+[a-z]?", kernel) else ("Other" if kernel else None)
    result = {"spacing_mm": spacing, "slice_thickness_mm": summary.get("slice_thickness_mm"), "image_count": int(summary.get("image_count", 0)), "kernel": kernel, "source_kind": source_kind(str(summary.get("image_type", ""))), "warnings": [], "metadata_validated": False}
    result.update(acquisition(result))
    return result


def fingerprint_hashes(values) -> str:
    # A multiset preserves duplicate-file evidence while ignoring file names and
    # upload ordering. Matching this is stronger than matching a neutral label.
    return sha256_canonical_json(sorted(values))


class CatalogService:
    def __init__(self, app, config: CatalogConfig | None = None):
        self.app, self.store = app, app.store
        self.config = config
        self._explicit_config = config is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue = None
        self._status = "unconfigured"
        self._error = None
        self._sync_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._startup, name="ethmoid-catalog", daemon=True)
        self._thread.start()

    def _startup(self) -> None:
        enqueue_all = False
        try:
            if self.config is None:
                self.config = load_local_config(self.app.root)
            if self.config is not None:
                self.sync()
                enqueue_all = self.config.auto_prepare
            else:
                self._status = "ready"
        except Exception:
            self._status, self._error = "error", "CONFIGURATION"
        # An unavailable default collection must not disable valid new uploads.
        if not self._stop.is_set():
            from .catalog_queue import PreparationQueue
            self._queue = PreparationQueue(self)
            self._queue.start(enqueue_all=enqueue_all)

    def close(self) -> None:
        self._stop.set()
        if self._queue is not None:
            self._queue.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _source(self):
        if self.config is None:
            raise WebAppError(REASONS["CONFIGURATION"], status_code=503)
        self.config.validate(self.app.root)
        return self.config.source_root.resolve()

    @property
    def work_root(self) -> Path:
        return self.config.work_root if self.config else self.app.root

    @property
    def minimum_free_bytes(self) -> int:
        return self.config.minimum_free_bytes if self.config else 2 * 1024 ** 3

    def source_root(self, row) -> Path:
        with self.store.connect() as connection:
            source = connection.execute("SELECT src.root_path FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id LEFT JOIN catalog_series_sources ss ON ss.series_id=s.id JOIN catalog_sources src ON src.id=COALESCE(ss.source_id,t.source_id) WHERE s.id=?", (row["id"],)).fetchone()
        if source is None:
            raise ValueError("SOURCE_UNAVAILABLE")
        return Path(source[0]).resolve(strict=True)

    def files(self, row) -> list[Path]:
        root = self.source_root(row)
        values = decode_json(row["files_json"], [])
        if not values or len(values) != len(set(values)):
            raise ValueError("INVENTORY_INVALID")
        paths = []
        for relative in values:
            path = (root / relative).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(root):
                raise ValueError("SOURCE_UNAVAILABLE")
            paths.append(path)
        return paths

    def fingerprint(self, row) -> str:
        return fingerprint_hashes(sha256_file(path) for path in self.files(row))

    def sync(self) -> None:
        with self._sync_lock:
            self._status, self._error = "indexing", None
            root = self._source()
            inventory_dir = self.config.work_root / "inventory"
            mapping = json.loads((inventory_dir / "uid_map.local.json").read_text(encoding="utf-8"))
            inventory = json.loads((inventory_dir / "inventory.redacted.json").read_text(encoding="utf-8"))
            if Path(mapping["source_root"]).resolve() != root:
                raise ValueError("INVENTORY_INVALID")
            summaries = {(item["case"], item["series"]): item for item in inventory["series"]}
            expected = {(study["case"], series["series"]) for study in mapping["cases"] for series in study["series"]}
            if set(summaries) != expected or len(summaries) != inventory["series_count"] or len(mapping["cases"]) != inventory["study_count"]:
                raise ValueError("INVENTORY_INVALID")
            now = utc_now()
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                source = connection.execute("SELECT id FROM catalog_sources WHERE root_path=?", (str(root),)).fetchone()
                source_id = source[0] if source else uuid.uuid4().hex
                connection.execute("INSERT INTO catalog_sources(id,root_path,inventory_sha256,status,updated_at) VALUES(?,?,?,'indexing',?) ON CONFLICT(root_path) DO UPDATE SET inventory_sha256=excluded.inventory_sha256,status='indexing',error_code=NULL,updated_at=excluded.updated_at", (source_id, str(root), sha256_file(inventory_dir / "inventory.redacted.json"), now))
                connection.execute("UPDATE catalog_series SET present=0 WHERE study_id IN (SELECT id FROM catalog_studies WHERE source_id=?) AND id NOT IN (SELECT series_id FROM catalog_series_sources)", (source_id,))
                for study in mapping["cases"]:
                    row = connection.execute("SELECT * FROM catalog_studies WHERE source_id=? AND study_uid=?", (source_id, study["study_uid"])).fetchone()
                    study_id = row["id"] if row else uuid.uuid4().hex
                    if row is None:
                        used = {r[0] for r in connection.execute("SELECT study_label FROM catalog_studies")}
                        label = self._label(study["case"], used, "Case")
                        connection.execute("INSERT INTO catalog_studies(id,source_id,study_uid,study_label) VALUES(?,?,?,?)", (study_id, source_id, study["study_uid"], label))
                    for series in study["series"]:
                        metadata = safe_descriptor(summaries[(study["case"], series["series"])])
                        relative_files = series["relative_files"]
                        if len(relative_files) != metadata["image_count"]:
                            raise ValueError("INVENTORY_INVALID")
                        # Stat changes enqueue a full-hash verification. They are
                        # not themselves sufficient proof for reusing a volume.
                        stats = []
                        for name in relative_files:
                            path = (root / name).resolve()
                            if not path.is_relative_to(root):
                                raise ValueError("INVENTORY_INVALID")
                            try:
                                stat = path.stat()
                                stats.append((name, stat.st_size, stat.st_mtime_ns))
                            except OSError:
                                stats.append((name, None, None))
                        # The inventory signature is a frozen source-change contract.
                        # Additive display/profile fields must not invalidate old
                        # verified volumes when the underlying DICOM is unchanged.
                        signature_metadata = {key: value for key, value in metadata.items() if key not in {"reconstruction_kind", "kernel_class"}}
                        signature = sha256_canonical_json({"files": stats, "metadata": signature_metadata})
                        old = connection.execute("SELECT * FROM catalog_series WHERE study_id=? AND series_uid=?", (study_id, series["series_uid"])).fetchone()
                        if old is None:
                            used = {r[0] for r in connection.execute("SELECT series_label FROM catalog_series WHERE study_id=?", (study_id,))}
                            label = self._label(series["series"], used, "Series")
                            code = "SCOUT" if metadata["source_kind"] == "scout" else "PROJECTION" if metadata["source_kind"] == "projection" else None
                            connection.execute("INSERT INTO catalog_series(id,study_id,series_uid,series_label,descriptor_json,files_json,inventory_signature,preparation_status,reason_code) VALUES(?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, study_id, series["series_uid"], label, encode_json(metadata), encode_json(relative_files), signature, "unavailable" if code else "not_prepared", code))
                        elif old["inventory_signature"] != signature:
                            connection.execute("UPDATE catalog_series SET descriptor_json=?,files_json=?,inventory_signature=?,case_id=NULL,preparation_status='not_prepared',reason_code='SOURCE_CHANGED',present=1 WHERE id=?", (encode_json(metadata), encode_json(relative_files), signature, old["id"]))
                            connection.execute("UPDATE preparation_jobs SET status='obsolete' WHERE series_id=? AND status IN ('queued','preparing')", (old["id"],))
                        else:
                            connection.execute("UPDATE catalog_series SET present=1 WHERE id=?", (old["id"],))
                connection.execute("UPDATE catalog_series SET preparation_status='unavailable',reason_code='SOURCE_MISSING',case_id=NULL WHERE present=0 AND study_id IN (SELECT id FROM catalog_studies WHERE source_id=?)", (source_id,))
                connection.execute("UPDATE catalog_sources SET status='ready',updated_at=? WHERE id=?", (now, source_id))
            self.link_existing(mapping)
            self.refresh_recommendations()
            self._status = "ready"

    @staticmethod
    def _label(preferred: str, used: set[str], prefix: str) -> str:
        if re.fullmatch(prefix + r"[0-9]{3}", preferred) and preferred not in used:
            return preferred
        return next(f"{prefix}{number:03}" for number in range(1, 1000) if f"{prefix}{number:03}" not in used)

    def series_row(self, series_id: str, study_id: str | None = None):
        with self.store.connect() as connection:
            row = connection.execute("SELECT s.*,t.study_uid,t.study_label FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id WHERE s.id=?", (series_id,)).fetchone()
        if row is None or (study_id is not None and row["study_id"] != study_id):
            raise WebAppError("Series not found in this study.", status_code=404)
        return row

    def link(self, series_id: str, case_id: str, fingerprint: str, proof: str, metadata: dict | None = None) -> None:
        with self.store.connect() as connection:
            row = connection.execute("SELECT * FROM catalog_series WHERE id=?", (series_id,)).fetchone()
            descriptor = metadata or decode_json(row["descriptor_json"], {})
            old = connection.execute("SELECT * FROM catalog_case_links WHERE case_id=?", (case_id,)).fetchone()
            if old and (old["series_id"] != series_id or old["source_fingerprint"] != fingerprint):
                raise ValueError("IMMUTABLE_CASE_IDENTITY_CONFLICT")
            connection.execute("INSERT INTO catalog_case_links(case_id,series_id,source_fingerprint,proof,acquisition_json) VALUES(?,?,?,?,?) ON CONFLICT(case_id) DO UPDATE SET acquisition_json=excluded.acquisition_json", (case_id, series_id, fingerprint, proof, encode_json(acquisition(descriptor))))
            connection.execute("UPDATE catalog_series SET case_id=?,source_fingerprint=?,preparation_status='ready',reason_code=NULL,descriptor_json=? WHERE id=?", (case_id, fingerprint, encode_json(descriptor), series_id))

    def link_existing(self, mapping: dict) -> None:
        """Reuse only exact volume/file fingerprints, never displayed labels."""
        with self.store.connect() as connection:
            cases = connection.execute("SELECT * FROM cases WHERE id NOT IN (SELECT case_id FROM catalog_case_links)").fetchall()
            rows = connection.execute("SELECT s.*,t.study_uid FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id WHERE s.present=1").fetchall()
            imports = connection.execute("SELECT case_label,series_json FROM imports").fetchall()
        by_identity = {(row["study_uid"], row["series_uid"]): row for row in rows}
        by_inventory_label = {(study["case"], series["series"]): (study["study_uid"], series["series_uid"]) for study in mapping["cases"] for series in study["series"]}
        cohort_path = self.config.work_root / "counting/cohort.v1.local.json"
        cohort = json.loads(cohort_path.read_text(encoding="utf-8")) if cohort_path.exists() else {"entries": []}
        for case in cases:
            source = self.app._path(case["volume_relpath"])
            if not source.is_file() or sha256_file(source) != case["source_sha256"] or case["validation_status"] != "VERIFIED":
                continue
            manifest_path = source.parent / "source_hashes.json"
            candidates, expected, proof = [], None, "dicom_file_hashes"
            report_path, report_label = source.parent / "qc_report.json", case["case_label"]
            if manifest_path.is_file():
                expected = fingerprint_hashes(item["sha256"] for item in json.loads(manifest_path.read_text())['slices'])
                validation = decode_json(case["validation_details_json"], {})
                for imported in imports:
                    for item in decode_json(imported["series_json"], []):
                        if imported["case_label"] == case["case_label"] and item.get("series_label") == validation.get("series"):
                            candidate = by_identity.get((item.get("study_uid"), item.get("series_uid")))
                            if candidate is not None:
                                candidates.append(candidate)
                if not candidates:
                    candidates = rows
            else:
                for entry in cohort["entries"]:
                    if entry["nifti_sha256"] != case["source_sha256"]:
                        continue
                    volume = (self.config.work_root / entry["nifti"]).resolve()
                    if not volume.is_relative_to(self.config.work_root.resolve()) or not volume.is_file() or sha256_file(volume) != case["source_sha256"]:
                        continue
                    hashes_path = volume.parent / "source_hashes.json"
                    if hashes_path.is_file():
                        expected = fingerprint_hashes(item["sha256"] for item in json.loads(hashes_path.read_text())['slices'])
                        candidate = by_identity.get(by_inventory_label.get((entry["case"], entry["series"])))
                        if candidate is not None:
                            candidates.append(candidate)
                        proof = "verified_cohort_and_dicom_hashes"
                        report_path, report_label = volume.parent / "qc_report.json", entry["case"]
            matches = []
            for candidate in candidates:
                try:
                    if expected is not None and self.fingerprint(candidate) == expected:
                        matches.append(candidate)
                except (OSError, ValueError):
                    continue
            matches = {row["id"]: row for row in matches}
            if len(matches) == 1:
                row = next(iter(matches.values()))
                from ..dicom import preflight_series
                try:
                    preflight = preflight_series(self.files(row), row["study_uid"], row["series_uid"], profile="exploratory")
                    if not verified_volume(source, report_path, report_label, preflight):
                        continue
                    descriptor = decode_json(row["descriptor_json"], {})
                    observed_interval = preflight.grid_policy["original_median_interval_mm"] if preflight.grid_policy else preflight.slice_spacing
                    descriptor.update(image_count=len(preflight.records), spacing_mm=[*preflight.pixel_spacing, observed_interval], fitted_spacing_mm=[*preflight.pixel_spacing, preflight.slice_spacing], grid_policy=preflight.grid_policy, slice_thickness_mm=preflight.slice_thickness, source_kind=preflight.source_kind, metadata_validated=True, warnings=list(preflight.warnings))
                    descriptor.update(acquisition(descriptor))
                    self.link(row["id"], case["id"], expected, proof, descriptor)
                except Exception:
                    continue

    @staticmethod
    def rank(row) -> tuple:
        metadata = decode_json(row["descriptor_json"], {})
        available = row["preparation_status"] != "unavailable"
        thin = metadata.get("resolution_class") == "within_protocol"
        sharp = metadata.get("kernel") in {"BONEPLUS", "Hr68s", "Hr68f"}
        spacing = metadata.get("spacing_mm") or []
        interval = spacing[2] if len(spacing) == 3 and type(spacing[2]) in (int, float) and spacing[2] > 0 else 999
        coverage = max(0, metadata.get("image_count", 0) - 1) * interval if interval != 999 else 0
        native = metadata.get("source_kind") == "native"
        tier = 0 if native and thin and sharp else 1 if native and thin else 2
        return (not available, tier, interval, -coverage, row["series_label"])

    def refresh_recommendations(self) -> None:
        with self.store.connect() as connection:
            for study in connection.execute("SELECT * FROM catalog_studies").fetchall():
                rows = connection.execute("SELECT * FROM catalog_series WHERE study_id=? AND present=1", (study["id"],)).fetchall()
                eligible = [row for row in rows if row["preparation_status"] != "unavailable"]
                recommended = min(eligible, key=self.rank)["id"] if eligible else None
                explicit = connection.execute("SELECT 1 FROM metadata WHERE key=?", ("catalog_selection:" + study["id"],)).fetchone()
                selected = study["selected_series_id"] if explicit else recommended or (rows[0]["id"] if rows else None)
                connection.execute("UPDATE catalog_studies SET recommended_series_id=?,selected_series_id=? WHERE id=?", (recommended, selected, study["id"]))

    def _summary(self, row, recommended: str | None) -> dict:
        metadata = decode_json(row["descriptor_json"], {})
        count_status, counts, count_outcome = "not_run", None, None
        if row["case_id"]:
            with self.store.connect() as connection:
                run = connection.execute("SELECT * FROM count_runs WHERE case_id=? ORDER BY created_at DESC,id DESC LIMIT 1", (row["case_id"],)).fetchone()
                if run:
                    count_status = run["status"]
                    job = connection.execute("SELECT outcome FROM count_jobs WHERE run_id=?", (run["id"],)).fetchone()
                    count_outcome = job["outcome"] if job else None
                    if run["status"] == "completed":
                        result = decode_json(run["original_result_json"], {})
                        if run["latest_revision"]:
                            revision = connection.execute("SELECT result_json FROM cell_revisions WHERE run_id=? AND revision=?", (run["id"], run["latest_revision"])).fetchone()
                            result = decode_json(revision[0], {}) if revision else {}
                        counts = result.get("counts")
        return {"series_id": row["id"], "series_label": row["series_label"], "case_id": row["case_id"], "preparation_status": row["preparation_status"], "reason": REASONS.get(row["reason_code"]), "retryable": row["reason_code"] in RETRYABLE, **{key:metadata.get(key) for key in ("spacing_mm", "slice_thickness_mm", "image_count", "kernel", "source_kind", "resolution_class")}, "warnings": metadata.get("warnings", []), "recommended": row["id"] == recommended, "count_status": count_status, "count_outcome": count_outcome, "counts": counts}

    def study(self, study_id: str) -> dict:
        with self.store.connect() as connection:
            study = connection.execute("SELECT * FROM catalog_studies WHERE id=?", (study_id,)).fetchone()
            rows = connection.execute("SELECT * FROM catalog_series WHERE study_id=? ORDER BY series_label", (study_id,)).fetchall()
        if study is None:
            raise WebAppError("Study not found.", status_code=404)
        return {"study_id": study["id"], "study_label": study["study_label"], "series_count": len(rows), "selected_series_id": study["selected_series_id"], "recommended_series_id": study["recommended_series_id"], "series": [self._summary(row, study["recommended_series_id"]) for row in rows]}

    def status(self) -> dict:
        with self.store.connect() as connection:
            counts = dict(connection.execute("SELECT preparation_status,COUNT(*) FROM catalog_series GROUP BY preparation_status").fetchall())
            studies = connection.execute("SELECT COUNT(*) FROM catalog_studies").fetchone()[0]
        status = "preparing" if self._status != "error" and counts.get("queued", 0) + counts.get("preparing", 0) else self._status
        worker_error = self._queue.last_error if self._queue is not None else None
        return {"status": status, "reason": REASONS.get(self._error or worker_error), "worker_error": worker_error, "total_studies": studies, "total_series": sum(counts.values()), **{key:counts.get(key, 0) for key in ("not_prepared", "queued", "preparing", "ready", "unavailable")}}

    def list_studies(self) -> dict:
        with self.store.connect() as connection:
            ids = [row[0] for row in connection.execute("SELECT id FROM catalog_studies ORDER BY study_label")]
            linked = {row[0] for row in connection.execute("SELECT case_id FROM catalog_series WHERE case_id IS NOT NULL")}
        return {"studies": [self.study(identity) for identity in ids], "catalog": self.status(), "unlinked_cases": [case for case in self.app.list_cases() if case["case_id"] not in linked]}

    def select(self, study_id: str, series_id: str) -> dict:
        self.series_row(series_id, study_id)
        with self.store.connect() as connection:
            connection.execute("UPDATE catalog_studies SET selected_series_id=? WHERE id=?", (series_id, study_id))
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,'explicit')", ("catalog_selection:" + study_id,))
        return self.study(study_id)

    def prepare(self, study_id: str, series_id: str) -> dict:
        row = self.series_row(series_id, study_id)
        if row["preparation_status"] == "ready":
            return self._summary(row, self.study(study_id)["recommended_series_id"])
        if self._queue is None:
            raise WebAppError("The local preparation service is not ready.", status_code=503)
        self._queue.enqueue(series_id, explicit=True)
        return next(item for item in self.study(study_id)["series"] if item["series_id"] == series_id)
