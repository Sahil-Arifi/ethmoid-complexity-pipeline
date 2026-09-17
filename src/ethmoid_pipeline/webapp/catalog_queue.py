"""One durable local preparation worker. No inference is scheduled here."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import threading
import traceback
import uuid

from ..conversion import convert_series
from ..dicom import PreflightFailure, preflight_series
from ..provenance import current_code_provenance
from ..util import sha256_file, utc_now
from ..verification import verify_conversion
from .catalog import REASONS, RETRYABLE, acquisition, fingerprint_hashes
from .catalog_validation import verified_volume
from .protocol import protocol_version
from .service import WebAppError, _atomic_copy, _empty_checklist, inspect_nifti
from .store import decode_json, encode_json


class PreparationFailure(RuntimeError):
    pass


def preflight_issue_codes(issues) -> list[str]:
    """Translate diagnostics to fixed neutral categories, never raw tag values."""
    categories = (
        ("PRIVACY_FLAG", ("BurnedInAnnotation", "RecognizableVisualFeatures", "overlay data", "graphic annotation")),
        ("SCOUT", ("localizer/scout",)),
        ("PROJECTION", ("projection or slab-average",)),
        ("HU_UNCALIBRATED", ("Rescale", "HU rescale", "HU validation")),
        ("MIXED_ACQUISITION", ("AcquisitionNumber", "IrradiationEventUID", "FrameOfReferenceUID", "StudyInstanceUID", "SeriesInstanceUID", "inventoried", "ConvolutionKernel changes", "ImageType changes", "MultienergyCTAcquisition changes")),
        ("ORIENTATION_INVALID", ("ImageOrientationPatient", "IOP ", "orientation", "drift or shear")),
        ("DUPLICATE_PLANES", ("duplicate",)),
        ("MISSING_PLANES", ("missing-plane", "fewer than two spatial planes")),
        ("SPACING_IRREGULAR", ("spacing is nonuniform", "SpacingBetweenSlices", "PixelSpacing changes", "SliceThickness changes")),
        ("ENCODING_UNSUPPORTED", ("transfer syntax", "TransferSyntaxUID", "non-classic-CT", "multi-frame", "photometric", "PhotometricInterpretation", "pixel encoding", "neither native CT")),
    )
    codes = []
    for issue in issues:
        code = next((code for code, terms in categories if any(term in str(issue) for term in terms)), "METADATA_INVALID")
        if code not in codes:
            codes.append(code)
    return codes or ["GEOMETRY_INVALID"]


class PreparationQueue:
    def __init__(self, catalog):
        self.catalog, self.app, self.store = catalog, catalog.app, catalog.store
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        self._lock_file = None

    def _acquire(self) -> bool:
        path = self.app.root / "catalog-preparation.lock"
        self._lock_file = path.open("a+b")
        if path.stat().st_size == 0:
            self._lock_file.write(b"0")
            self._lock_file.flush()
        self._lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            self._lock_file = None
            return False
        return True

    def start(self, *, enqueue_all: bool = True) -> None:
        if self._thread is not None or not self._acquire():
            return
        with self.store.connect() as connection:
            connection.execute("UPDATE preparation_jobs SET status='queued',error_code='INTERRUPTED' WHERE status='preparing'")
            connection.execute("UPDATE catalog_series SET preparation_status='queued',reason_code='INTERRUPTED' WHERE preparation_status='preparing'")
        if enqueue_all:
            self.enqueue_all()
        self._thread = threading.Thread(target=self._loop, name="ethmoid-prepare", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # A conversion already running is allowed to finish verification. Its
        # thread retains the process lock until it actually exits.

    def enqueue_all(self) -> None:
        with self.store.connect() as connection:
            rows = connection.execute("SELECT s.*,t.recommended_series_id FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id WHERE present=1").fetchall()
        for row in sorted(rows, key=lambda item: (item["id"] != item["recommended_series_id"], self.catalog.rank(item))):
            if row["reason_code"] in {"SCOUT", "PROJECTION", "SOURCE_MISSING"}:
                continue
            self.enqueue(row["id"], priority=0 if row["id"] == row["recommended_series_id"] else 100, startup=True)

    def enqueue(self, series_id: str, *, explicit: bool = False, priority: int = 100, startup: bool = False) -> None:
        row = self.catalog.series_row(series_id)
        if not row["present"] or row["reason_code"] in {"SCOUT", "PROJECTION", "SOURCE_MISSING"}:
            raise WebAppError("This series cannot form a countable three-dimensional volume.", status_code=409)
        if row["preparation_status"] == "unavailable" and row["reason_code"] not in RETRYABLE and not startup:
            raise WebAppError("This series did not pass source geometry validation.", status_code=409)
        # A prior verified immutable volume stays usable across process restarts.
        # Source stat/inventory changes were already invalidated during sync;
        # exact raw source hashes are rechecked before each new counting run.
        if row["preparation_status"] == "ready" and row["case_id"]:
            try:
                self.app.volume_path(row["case_id"])
                return
            except WebAppError:
                with self.store.connect() as connection:
                    connection.execute("UPDATE catalog_series SET preparation_status='not_prepared',reason_code='CONVERSION_FAILED',case_id=NULL WHERE id=?", (series_id,))
        if explicit:
            priority = -100
        now = utc_now()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT * FROM catalog_series WHERE id=?", (series_id,)).fetchone()
            active = connection.execute("SELECT id,status FROM preparation_jobs WHERE series_id=? AND status IN ('queued','preparing')", (series_id,)).fetchone()
            if active:
                if explicit and active["status"] == "queued":
                    connection.execute("UPDATE preparation_jobs SET priority=?,updated_at=? WHERE id=?", (priority, now, active["id"]))
                return
            old = connection.execute("SELECT * FROM preparation_jobs WHERE series_id=? AND inventory_signature=?", (series_id, current["inventory_signature"])).fetchone()
            if old and old["status"] == "failed" and old["error_code"] not in RETRYABLE:
                return
            if old and old["status"] == "failed" and not explicit:
                # Explicit retry avoids endless disk/converter failure loops.
                return
            if old:
                if old["status"] == "completed" and current["preparation_status"] == "ready":
                    return
                connection.execute("UPDATE preparation_jobs SET status='queued',priority=?,error_code=NULL,updated_at=? WHERE id=?", (priority, now, old["id"]))
            else:
                connection.execute("INSERT INTO preparation_jobs(id,series_id,inventory_signature,status,priority,created_at,updated_at) VALUES(?,?,?,'queued',?,?,?)", (uuid.uuid4().hex, series_id, current["inventory_signature"], priority, now, now))
            connection.execute("UPDATE catalog_series SET preparation_status='queued',reason_code=NULL WHERE id=?", (series_id,))
        self._wake.set()

    def _claim(self):
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM preparation_jobs WHERE status='queued' ORDER BY priority,created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            connection.execute("UPDATE preparation_jobs SET status='preparing',attempt=attempt+1,updated_at=? WHERE id=?", (utc_now(), row["id"]))
            connection.execute("UPDATE catalog_series SET preparation_status='preparing' WHERE id=?", (row["series_id"],))
            return dict(row) | {"attempt": row["attempt"] + 1}

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                job = self._claim()
                if job is None:
                    self._wake.wait(timeout=1)
                    self._wake.clear()
                    continue
                self.process(job)
        finally:
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

    def _check_source(self, row, fingerprint: str) -> None:
        current = self.catalog.series_row(row["id"])
        if not current["present"] or current["inventory_signature"] != row["inventory_signature"] or self.catalog.fingerprint(current) != fingerprint:
            raise PreparationFailure("SOURCE_CHANGED")

    def _existing(self, row, fingerprint: str, preflight) -> str | None:
        with self.store.connect() as connection:
            rows = connection.execute("SELECT c.* FROM cases c JOIN catalog_case_links l ON l.case_id=c.id WHERE l.series_id=? AND l.source_fingerprint=?", (row["id"], fingerprint)).fetchall()
        for case in rows:
            path = self.app._path(case["volume_relpath"])
            if path.is_file() and sha256_file(path) == case["source_sha256"] and case["validation_status"] == "VERIFIED":
                report_path = path.parent / "qc_report.json"
                if report_path.is_file():
                    label = json.loads(report_path.read_text(encoding="utf-8")).get("case")
                    if verified_volume(path, report_path, label, preflight):
                        return case["id"]
                reused = self._cohort(fingerprint, preflight)
                if reused and sha256_file(reused[0]) == case["source_sha256"]:
                    return case["id"]
        return None

    def _cohort(self, fingerprint: str, preflight):
        path = self.catalog.config.work_root / "counting/cohort.v1.local.json"
        if not path.exists():
            return None
        cohort = json.loads(path.read_text(encoding="utf-8"))
        for entry in cohort["entries"]:
            source = (self.catalog.config.work_root / entry["nifti"]).resolve()
            if not source.is_relative_to(self.catalog.config.work_root.resolve()):
                continue
            hashes = source.parent / "source_hashes.json"
            if not hashes.is_file():
                continue
            stored = fingerprint_hashes(item["sha256"] for item in json.loads(hashes.read_text())['slices'])
            if stored != fingerprint or not source.is_file() or sha256_file(source) != entry["nifti_sha256"]:
                continue
            report_path = source.parent / "qc_report.json"
            if verified_volume(source, report_path, entry["case"], preflight):
                return source, json.loads(report_path.read_text(encoding="utf-8"))
        return None

    def _register(self, row, job, case_id: str, volume: Path, report: dict, fingerprint: str, descriptor: dict, proof: str) -> str:
        info, volume_hash, now = inspect_nifti(volume), sha256_file(volume), utc_now()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT * FROM catalog_series WHERE id=?", (row["id"],)).fetchone()
            if current["inventory_signature"] != job["inventory_signature"] or not current["present"]:
                raise PreparationFailure("SOURCE_CHANGED")
            used = {item[0] for item in connection.execute("SELECT case_label FROM cases UNION SELECT case_label FROM imports WHERE status!='CONVERTED'")}
            label = self.catalog._label(row["study_label"], used, "Case")
            connection.execute("""INSERT INTO cases(id,case_label,source_type,state,validation_status,
                volume_relpath,source_sha256,shape_json,spacing_json,affine_json,warnings_json,
                validation_details_json,qc_checklist_json,protocol_version,created_at,updated_at)
                VALUES(?,?,'DICOM','READY','VERIFIED',?,?,?,?,?,?,?,?,?,?,?)""",
                (case_id, label, self.app._relative(volume), volume_hash, encode_json(list(info.shape)), encode_json(list(info.spacing)), encode_json(info.affine), encode_json(descriptor["warnings"]), encode_json(report), encode_json(_empty_checklist()), protocol_version(), now, now))
            connection.execute("INSERT INTO catalog_case_links(case_id,series_id,source_fingerprint,proof,acquisition_json) VALUES(?,?,?,?,?)", (case_id, row["id"], fingerprint, proof, encode_json(acquisition(descriptor))))
            connection.execute("UPDATE catalog_series SET case_id=?,source_fingerprint=?,descriptor_json=?,preparation_status='ready',reason_code=NULL WHERE id=?", (case_id, fingerprint, encode_json(descriptor), row["id"]))
        return case_id

    def process(self, job: dict) -> None:
        directory = self.app.root / "preparation" / job["id"] / f"attempt-{job['attempt']:04}"
        error = None
        try:
            directory.mkdir(parents=True, exist_ok=False)
            row = self.catalog.series_row(job["series_id"])
            if row["inventory_signature"] != job["inventory_signature"]:
                raise PreparationFailure("SOURCE_CHANGED")
            paths = self.catalog.files(row)
            fingerprint = fingerprint_hashes(sha256_file(path) for path in paths)
            preflight = preflight_series(paths, row["study_uid"], row["series_uid"], profile="exploratory")
            descriptor = decode_json(row["descriptor_json"], {})
            descriptor.update(spacing_mm=[*preflight.pixel_spacing, preflight.slice_spacing], slice_thickness_mm=preflight.slice_thickness, source_kind=preflight.source_kind, metadata_validated=True, warnings=list(preflight.warnings))
            descriptor.update(acquisition(descriptor))
            (directory / "preflight.local.json").write_text(json.dumps(preflight.safe_report(row["study_label"], row["series_label"]), indent=2), encoding="utf-8")
            existing = self._existing(row, fingerprint, preflight)
            if existing is not None:
                self._check_source(row, fingerprint)
                self.catalog.link(row["id"], existing, fingerprint, "existing_verified_volume", descriptor)
            else:
                expected_bytes = sum(path.stat().st_size for path in paths) * 3 + preflight.rows * preflight.columns * len(paths) * 8
                if shutil.disk_usage(self.app.root).free < self.catalog.config.minimum_free_bytes + expected_bytes:
                    raise PreparationFailure("DISK_SPACE")
                case_id = uuid.uuid4().hex
                destination = self.app.root / "cases" / case_id / "volume"
                reused = self._cohort(fingerprint, preflight)
                if reused:
                    source, report = reused
                    destination.mkdir(parents=True, exist_ok=False)
                    volume = destination / ("source.nii.gz" if source.name.endswith(".gz") else "source.nii")
                    _atomic_copy(source, volume)
                    if sha256_file(volume) != sha256_file(source):
                        raise PreparationFailure("CONVERSION_FAILED")
                    (destination / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    (destination / "source_hashes.json").write_bytes((source.parent / "source_hashes.json").read_bytes())
                    proof = "verified_cohort_and_dicom_hashes"
                else:
                    provenance = current_code_provenance(require_clean=False)
                    result = convert_series(self.catalog.config.source_root, self.catalog.config.work_root, paths, row["study_label"], row["series_label"], self.app.dcm2niix, expected_study_uid=row["study_uid"], expected_series_uid=row["series_uid"], final_dir=destination, pipeline_provenance=provenance, require_clean_provenance=False, preflight_profile="exploratory")
                    report = verify_conversion(self.catalog.config.source_root, self.catalog.config.work_root, paths, row["study_label"], row["series_label"], expected_study_uid=row["study_uid"], expected_series_uid=row["series_uid"], artifact_dir=destination, pipeline_provenance=provenance, require_clean_provenance=False, preflight_profile="exploratory")
                    if report.get("status") != "AUTOMATED_PASS_PENDING_MANUAL" or not report.get("checks") or not all(check.get("passed") is True for check in report["checks"]):
                        raise PreparationFailure("CONVERSION_FAILED")
                    volume, proof = Path(result["nifti"]), "verified_exploratory_conversion"
                self._check_source(row, fingerprint)
                self._register(row, job, case_id, volume, report, fingerprint, descriptor, proof)
        except PreflightFailure as failure:
            codes = preflight_issue_codes(failure.issues)
            error = codes[0]
            try:
                (directory / "preflight-failure.local.json").write_text(json.dumps({"schema_version": 1, "profile": "exploratory", "issue_count": len(failure.issues), "issues": [{"code": code, "message": REASONS[code]} for code in codes]}, indent=2), encoding="utf-8")
            except OSError:
                error = "DISK_SPACE"
        except PreparationFailure as failure:
            error = str(failure)
        except (FileNotFoundError, PermissionError):
            error = "SOURCE_UNAVAILABLE"
        except OSError as failure:
            error = "DISK_SPACE" if failure.errno == errno.ENOSPC else "CONVERSION_FAILED"
        except Exception:
            error = "CONVERSION_FAILED"
            try:
                (directory / "error.local.txt").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
        with self.store.connect() as connection:
            connection.execute("UPDATE preparation_jobs SET status=?,error_code=?,updated_at=? WHERE id=? AND status='preparing'", ("failed" if error else "completed", error, utc_now(), job["id"]))
            if error:
                connection.execute("UPDATE catalog_series SET preparation_status='unavailable',reason_code=?,case_id=NULL WHERE id=? AND inventory_signature=?", (error, job["series_id"], job["inventory_signature"]))
        self.catalog.refresh_recommendations()
