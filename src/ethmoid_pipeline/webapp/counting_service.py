"""Local experimental count jobs and immutable, source-bound cell revisions."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import threading
import time
import errno
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np

from ..util import atomic_write_text, repo_root, sha256_canonical_json, sha256_file, utc_now
from .service import AppService, WebAppError, inspect_nifti, _atomic_save_nifti
from .store import decode_json, encode_json
from .counting_reports import build_count_reports, report_identity
from ..counting.protocol import PROTOCOL_VERSION
from ..counting.quality import acquisition_snapshot, resolution_quality
from .count_queue import CountQueue
from .owned_process import launch_owned_process


ARTIFACTS = {"cavities": "cavities.nii.gz", "air": "air.nii.gz", "region": "region.nii.gz"}
ELIGIBILITY = "EXPLORATORY_UNVALIDATED_ELIGIBILITY_UNCONFIRMED"
ENGINE_FAILURE_CODES = {
    "INVALID_CONFIG", "INVALID_GEOMETRY", "INVALID_VOLUME", "EMPTY_REGION",
    "NO_COMPARTMENTS", "NO_SUPPORTED_CELLS", "INSTANCE_LIMIT", "OUTPUT_EXISTS",
    "SOURCE_UNAVAILABLE", "RESOLUTION_INELIGIBLE", "HU_RANGE_INVALID",
    "VOLUME_OR_OUTPUT_INVALID", "ENGINE_ERROR", "LOCALIZER_NOT_READY",
    "LOCALIZER_INTEGRITY", "LOCALIZER_FAILED", "LANDMARKS_MISSING",
    "LANDMARK_GEOMETRY", "LANDMARK_VALUES", "LANDMARK_EMPTY", "LANDMARK_LATERALITY",
    "LANDMARK_SCALE", "ORBIT_WALL_UNRESOLVED", "ORBIT_FLOOR_UNRESOLVED",
    "ROOF_UNRESOLVED", "SPHENOID_UNRESOLVED", "SPHENOID_AMBIGUOUS", "TARGET_CLIPPED",
    "LOCALIZATION_ORIENTATION", "OBLIQUE_GRID_UNSUPPORTED",
    "LANDMARK_PROVENANCE", "LANDMARK_SOURCE", "LANDMARK_INTEGRITY",
    "INSUFFICIENT_COVERAGE", "INVALID_ACQUISITION", "RESOURCE_LIMIT",
    "MODEL_UNAVAILABLE", "MODEL_INTEGRITY", "ASSISTANCE_INVALID", "PROCESSING_PLAN_INVALID",
    "ASSISTANCE_INTEGRITY", "ASSISTANCE_GEOMETRY", "ASSISTANCE_VALUES", "ASSISTANCE_SOURCE",
    "ASSISTANCE_LINEAGE", "ASSISTANCE_LATERALITY", "PROCESSING_PLAN_MISMATCH", "SOURCE_CHANGED",
    "ARTIFACT_VERIFICATION", "DIRECT_MODEL_UNAVAILABLE", "DIRECT_MODEL_INTEGRITY", "DIRECT_MODEL_RUNTIME",
    "DIRECT_MODEL_GEOMETRY", "DIRECT_MODEL_VALUES", "DIRECT_MODEL_FAILED", "DIRECT_MODEL_LATERALITY",
}
REVIEW_CODES = {"EMPTY_REGION", "NO_COMPARTMENTS", "NO_SUPPORTED_CELLS", "INSTANCE_LIMIT",
    "LANDMARK_EMPTY", "LANDMARK_LATERALITY", "LANDMARK_SCALE", "ORBIT_WALL_UNRESOLVED",
    "ORBIT_FLOOR_UNRESOLVED", "ROOF_UNRESOLVED", "SPHENOID_UNRESOLVED", "SPHENOID_AMBIGUOUS",
    "TARGET_CLIPPED", "INSUFFICIENT_COVERAGE", "ASSISTANCE_LATERALITY", "DIRECT_MODEL_LATERALITY"}
Executor = Callable[[Path, Path, Path], None]


def _counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    left = sum(c["included"] and c["side"] == "left" for c in cells)
    right = sum(c["included"] and c["side"] == "right" for c in cells)
    return {"left": left, "right": right, "total": left + right}


def _error(message: str, code: int = 422) -> WebAppError:
    return WebAppError(message, status_code=code)


class CountingService:
    def __init__(
        self, app_service: AppService, *, executor: Executor | None = None,
        protocol: dict[str, Any] | None = None,
    ):
        self.app = app_service
        self.root = app_service.root
        self.store = app_service.store
        self._protocol = protocol
        self._executor = executor or self._execute_local
        self._process: subprocess.Popen[bytes] | None = None
        self._active_run_id: str | None = None
        self._process_lock = threading.Lock()
        self._closed = threading.Event()
        self._queue = CountQueue(self)

    def close(self) -> None:
        self._closed.set()
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                self._terminate_process(self._process)
        self._queue.close()

    def wait_for_idle(self, timeout: float = 30) -> None:
        self._queue.wait_idle(timeout)

    def protocol(self) -> dict[str, Any]:
        if self._protocol is not None:
            value = copy.deepcopy(self._protocol)
        else:
            try:
                from ..counting.protocol import load_counting_protocol, ADAPTIVE_PROTOCOL_VERSION
                value = load_counting_protocol(ADAPTIVE_PROTOCOL_VERSION)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                raise _error("The experimental counting configuration is unavailable.", 503) from exc
        if value.get("protocol_version") not in {PROTOCOL_VERSION, "ethmoid-count-prototype-v3"}:
            raise _error("The experimental counting configuration is unsupported.", 503)
        encode_json(value)
        return value

    def _source(self, case_id: str) -> tuple[Any, Path]:
        with self.store.connect() as connection:
            row = self.app._get_case_row(connection, case_id)
        if row["validation_status"] == "FAILED" or row["state"] == "VALIDATION_FAILED":
            raise _error("A failed source volume cannot be counted.", 409)
        source = self.app._path(row["volume_relpath"])
        if not source.is_file() or sha256_file(source) != row["source_sha256"]:
            raise _error("The source volume failed its integrity check.", 409)
        return row, source

    def _run(self, case_id: str, run_id: str, connection: Any | None = None) -> Any:
        if connection is None:
            with self.store.connect() as connection:
                return self._run(case_id, run_id, connection)
        row = connection.execute(
            "SELECT * FROM count_runs WHERE id=? AND case_id=?", (run_id, case_id)
        ).fetchone()
        if row is None:
            raise _error("Count run not found.", 404)
        return row

    def _assert_current_catalog_source(self, case_id: str) -> None:
        checker = getattr(self.app, "assert_catalog_countable", None)
        if checker is not None:
            checker(case_id)

    def _validate_automatic_retry_parent(self, case_id: str, parent_run_id: str,
                                         source_sha256: str, connection: Any | None = None) -> None:
        if connection is None:
            with self.store.connect() as connection:
                return self._validate_automatic_retry_parent(case_id, parent_run_id, source_sha256, connection)
        parent = self._run(case_id, parent_run_id, connection)
        job = connection.execute("SELECT assistance_id FROM count_jobs WHERE run_id=?", (parent_run_id,)).fetchone()
        parent_protocol = decode_json(parent["protocol_json"], {})
        if (parent["status"] != "failed" or parent["source_sha256"] != source_sha256 or
            (job is not None and job["assistance_id"] is not None) or parent_protocol.get("assistance") is not None):
            raise _error("An automatic retry requires a failed automatic parent from the same source.", 409)

    def create_run(self, case_id: str, idempotency_key: str | None = None, *, boundary_id: str | None = None,
                   parent_run_id: str | None = None) -> dict[str, Any]:
        if self._closed.is_set():
            raise _error("The counting worker is stopping.", 503)
        if idempotency_key is not None and (
            not 1 <= len(idempotency_key) <= 100 or
            not all(c.isascii() and (c.isalnum() or c in "-_") for c in idempotency_key)
        ):
            raise _error("The request identifier is invalid.")
        case, source = self._source(case_id)
        self._assert_current_catalog_source(case_id)
        if case["state"] == "APPROVED":
            raise _error("Approved cases cannot start new count runs.", 409)
        info = inspect_nifti(source)
        if info.warnings:
            raise _error("Resolve source orientation metadata before counting.", 409)
        protocol = self.protocol()
        context_reader = getattr(self.app, "catalog_context", None)
        context = context_reader(case_id) if context_reader is not None else None
        try:
            protocol["acquisition"] = acquisition_snapshot(context, case["source_sha256"])
        except ValueError as exc:
            raise _error("The trusted acquisition metadata is invalid.", 409) from exc
        protocol["source_quality"] = resolution_quality(protocol["acquisition"], np.asarray(info.affine))
        if protocol["protocol_version"] == "ethmoid-count-prototype-v3":
            from ..counting.profiles import build_scan_profile, build_processing_plan
            protocol["profile_characteristics"] = {key: (context or {}).get(key, "unknown") for key in ("reconstruction_kind", "kernel_class")}
            protocol["scan_profile"] = build_scan_profile(source, {**protocol["acquisition"], **protocol["profile_characteristics"]})
            protocol["processing_plan"] = build_processing_plan(protocol["scan_profile"])
        if boundary_id is not None:
            boundary = self.get_boundary(case_id, boundary_id)
            if boundary["parent_run_id"] != parent_run_id:
                raise _error("The boundary belongs to a different count run.", 409)
            protocol["assistance"] = {
                "schema_version": 1, "source_sha256": case["source_sha256"],
                "region_path": str(self.boundary_path(case_id, boundary_id)),
                "region_sha256": boundary["mask_sha256"],
                "revision_id": boundary_id, "parent_run_id": parent_run_id,
            }
            if protocol['protocol_version'] != 'ethmoid-count-prototype-v3':
                raise _error("Assisted boundaries require the adaptive v3 method.", 409)
            protocol['processing_plan'] = build_processing_plan(protocol['scan_profile'], assistance=protocol['assistance'])
        now = utc_now()
        run_id = uuid.uuid4().hex
        relative = f"cases/{case_id}/count-runs/{run_id}"
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_case = self.app._get_case_row(connection, case_id)
            if current_case["state"] == "APPROVED":
                raise _error("Approved cases cannot start new count runs.", 409)
            if boundary_id is None and parent_run_id is not None:
                self._validate_automatic_retry_parent(case_id, parent_run_id, case["source_sha256"], connection)
            existing = None
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT id FROM count_runs WHERE case_id=? AND idempotency_key=?",
                    (case_id, idempotency_key),
                ).fetchone()
            if existing is None:
                existing = connection.execute(
                    "SELECT id FROM count_runs WHERE case_id=? AND status IN ('queued','running')",
                    (case_id,),
                ).fetchone()
            if existing is not None:
                run_id = existing["id"]
                job = connection.execute("SELECT assistance_id,parent_run_id FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
                if ((job["assistance_id"] if job else None) != boundary_id or
                    (job["parent_run_id"] if job else None) != parent_run_id):
                    raise _error("Another counting request with different assistance or parent is already present.", 409)
            else:
                connection.execute(
                    "INSERT INTO count_runs(id,case_id,idempotency_key,status,source_sha256,"
                    "protocol_version,protocol_json,output_relpath,created_at,updated_at) "
                    "VALUES(?,?,?,'queued',?,?,?,?,?,?)",
                    (run_id, case_id, idempotency_key, case["source_sha256"], protocol["protocol_version"],
                     encode_json(protocol), relative, now, now),
                )
                connection.execute("""INSERT INTO count_jobs(run_id,parent_run_id,assistance_id,created_at,updated_at)
                    VALUES(?,?,?,?,?)""", (run_id, parent_run_id, boundary_id, now, now))
        if existing is None:
            self._queue.notify()
        return self.get_run(case_id, run_id)

    def _execute_local(self, source: Path, output: Path, config: Path) -> None:
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise _error("The local inference runtime is unavailable.", 503)
        runtime = Path(base) / "EthmoidComplexity" / "tools" / "counting-runtime"
        python = runtime / "Scripts" / "python.exe"
        if not python.is_file():
            raise _error("The local inference runtime is unavailable.", 503)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo_root() / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["ETHMOID_INFERENCE_OFFLINE"] = "1"
        environment["nnUNet_compile"] = "false"
        environment["TOTALSEG_HOME_DIR"] = str(runtime / "totalsegmentator")
        environment["TOTALSEG_WEIGHTS_PATH"] = str(runtime / "totalsegmentator" / "nnunet" / "results")
        command = [str(python), "-m", "ethmoid_pipeline.counting.worker", "--input", str(source),
                   "--output", str(output)]
        # The clinical contract is not an AlgorithmConfig. Only an explicit,
        # frozen algorithm section may override worker defaults; the worker also
        # records every effective default in its result config.
        snapshot = json.loads(config.read_text(encoding="utf-8"))
        if "algorithm" in snapshot:
            command.extend(["--config", str(config)])
        acquisition_path = config.with_name("acquisition.json")
        atomic_write_text(acquisition_path, encode_json({**snapshot.get("acquisition", acquisition_snapshot(None, sha256_file(source))), **snapshot.get("profile_characteristics", {})}))
        command.extend(["--acquisition", str(acquisition_path)])
        if snapshot["protocol_version"] == "ethmoid-count-prototype-v3":
            command.extend(["--method", "adaptive-v3", "--progress", str(config.with_name("progress.json"))])
            plan_path = config.with_name("processing-plan.json")
            atomic_write_text(plan_path, encode_json(snapshot["processing_plan"]))
            command.extend(["--processing-plan", str(plan_path)])
            if snapshot.get("assistance"):
                assistance_path = config.with_name("assistance.json")
                atomic_write_text(assistance_path, encode_json(snapshot["assistance"]))
                command.extend(["--assistance", str(assistance_path)])
        with self._process_lock:
            if self._closed.is_set():
                raise RuntimeError("Counting worker is stopping")
            self._process = launch_owned_process(
                command, env=environment, cwd=runtime, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            process = self._process
        try:
            deadline = time.monotonic() + 7200
            while process.poll() is None:
                if self._closed.is_set() or (self._active_run_id and self._cancelled(self._active_run_id)):
                    self._terminate_process(process)
                    raise InterruptedError("Counting stopped")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, 7200)
                try:
                    stage = json.loads(config.with_name("progress.json").read_text())["stage"]
                    if stage in {"profiling", "localizing", "counting", "verifying"} and self._active_run_id:
                        self._stage(self._active_run_id, stage)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode and not (output / "result.json").is_file():
                raise RuntimeError("Inference did not produce a result")
        except subprocess.TimeoutExpired:
            self._terminate_process(process)
            process.wait()
            raise
        finally:
            with self._process_lock:
                process.close()
                self._process = None

    @staticmethod
    def _terminate_process(process) -> None:
        if process.poll() is not None:
            return
        process.terminate()

    def _cancelled(self, run_id: str) -> bool:
        with self.store.connect() as db:
            row = db.execute("SELECT cancel_requested FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
        return bool(row and row[0])

    def _stage(self, run_id: str, stage: str) -> None:
        with self.store.connect() as db:
            db.execute("UPDATE count_jobs SET stage=?,updated_at=? WHERE run_id=?", (stage, utc_now(), run_id))

    def _fail(self, run_id: str, code: str) -> None:
        outcome = "needs_review" if code in REVIEW_CODES else "cancelled" if code == "CANCELLED" else "unsupported_input" if code in {"RESOLUTION_INELIGIBLE", "INVALID_ACQUISITION", "OBLIQUE_GRID_UNSUPPORTED"} else "technical_error"
        with self.store.connect() as connection:
            connection.execute(
                "UPDATE count_runs SET status='failed',failure_code=?,updated_at=? "
                "WHERE id=? AND status IN ('queued','running')", (code, utc_now(), run_id),
            )
            connection.execute("UPDATE count_jobs SET state='done',stage='finished',outcome=?,last_error=?,updated_at=? WHERE run_id=?",
                               (outcome, code, utc_now(), run_id))

    def _resume_output(self, row, directory: Path) -> Path | None:
        with self.store.connect() as db:
            job = db.execute("SELECT checkpoint_json FROM count_jobs WHERE run_id=?", (row["id"],)).fetchone()
        checkpoint = decode_json(job[0], {}) if job else {}
        if not checkpoint:
            return None
        if checkpoint.get("source_sha256") != row["source_sha256"] or checkpoint.get("protocol_sha256") != sha256_canonical_json(decode_json(row["protocol_json"], {})):
            raise _error("The checkpoint belongs to another source or processing plan.")
        output = self.app._path(checkpoint["output_relpath"])
        # A crash during publication can leave the verified directory at its
        # final name before the DB transaction commits.
        if not output.exists() and (directory / "original").exists():
            output = directory / "original"
        for name, expected in checkpoint["hashes"].items():
            if name not in {"result.json", *ARTIFACTS.values()} or sha256_file(self._safe_file(output, name)) != expected:
                raise _error("The inference checkpoint failed its integrity check.")
        if "result.json" not in checkpoint["hashes"]:
            raise _error("The inference checkpoint is incomplete.")
        return output

    def _checkpoint(self, row, output: Path) -> None:
        result = json.loads(self._safe_file(output, "result.json").read_text(encoding="utf-8"))
        names = ["result.json", *ARTIFACTS.values()] if result.get("status") == "completed" else ["result.json"]
        value = {"source_sha256": row["source_sha256"],
                 "protocol_sha256": sha256_canonical_json(decode_json(row["protocol_json"], {})),
                 "output_relpath": self.app._relative(output),
                 "hashes": {name: sha256_file(self._safe_file(output, name)) for name in names}}
        with self.store.connect() as db:
            db.execute("UPDATE count_jobs SET checkpoint_json=?,stage='verifying',updated_at=? WHERE run_id=?",
                       (encode_json(value), utc_now(), row["id"]))

    def _requeue(self, run_id: str, *, transient: bool = False) -> bool:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT retry_count,cancel_requested FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
            if not job or job['cancel_requested'] or (transient and job['retry_count'] >= 2):
                return False
            db.execute("UPDATE count_runs SET status='queued',failure_code=NULL,updated_at=? WHERE id=?", (utc_now(), run_id))
            db.execute("""UPDATE count_jobs SET state='queued',owner=NULL,retry_count=retry_count+?,
                updated_at=? WHERE run_id=?""", (int(transient), utc_now(), run_id))
        return True

    def _work(self, case_id: str, run_id: str) -> None:
        self._active_run_id = run_id
        try:
            if self._cancelled(run_id):
                self._fail(run_id, "CANCELLED")
                return
            with self.store.connect() as connection:
                cursor = connection.execute(
                    "UPDATE count_runs SET status='running',updated_at=? WHERE id=? AND status='queued'",
                    (utc_now(), run_id),
                )
                if cursor.rowcount != 1:
                    return
            row = self._run(case_id, run_id)
            _, source = self._source(case_id)
            self._assert_current_catalog_source(case_id)
            directory = self.app._path(row["output_relpath"])
            directory.mkdir(parents=True, exist_ok=True)
            protocol = decode_json(row["protocol_json"], {})
            output = self._resume_output(row, directory)
            attempt = directory / f"attempt-{uuid.uuid4().hex}"
            attempt.mkdir()
            config_path = attempt / "protocol.json"
            atomic_write_text(config_path, encode_json(protocol))
            if output is None:
                output = attempt / "original"
                self._stage(run_id, "localizing")
                self._executor(source, output, config_path)
                self._checkpoint(row, output)
            if self._closed.is_set():
                self._requeue(run_id)
                return
            if self._cancelled(run_id):
                self._fail(run_id, "CANCELLED")
                return
            self._stage(run_id, "verifying")
            self._source(case_id)
            self._assert_current_catalog_source(case_id)
            result_path = self._safe_file(output, "result.json")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            # Server-derived source facts remain available even when anatomical
            # localization abstains. Free-form worker text is never needed.
            if row["protocol_version"] in {PROTOCOL_VERSION, "ethmoid-count-prototype-v3"}:
                for key, expected in (("acquisition", protocol["acquisition"]), ("quality", protocol["source_quality"])):
                    if key in result and result[key] != expected:
                        raise _error("The counting acquisition facts do not match the frozen source snapshot.")
                    result[key] = copy.deepcopy(expected)
            if result.get("status") == "failed":
                code = result.get("failure_code")
                self._fail(run_id, code if isinstance(code, str) and code in ENGINE_FAILURE_CODES else "INFERENCE_FAILED")
                return
            if row["protocol_version"] == "ethmoid-count-prototype-v3":
                for key in ("scan_profile", "processing_plan"):
                    if result.get(key) != protocol.get(key):
                        raise _error("The adaptive processing facts do not match the source snapshot.")
                self._validate_adaptive_provenance(result, protocol, row, output)
            result = self._validate_result(result, source, output, row)
            quality = result.get("quality", {})
            result["warnings"] = list(dict.fromkeys([*quality.get("warnings", []), *result.get("warnings", [])]))
            if quality.get("flags"):
                for cell in result["cells"]:
                    cell["uncertain"] = True
                    cell["reasons"] = list(dict.fromkeys([*cell["reasons"], *quality["flags"]]))
            result["protocol_sha256"] = protocol.get("sha256", sha256_canonical_json(protocol))
            result["protocol_snapshot_sha256"] = sha256_canonical_json(protocol)
            result["revision"] = 0
            result["parent_revision"] = None
            result["lineage"] = []
            result["next_instance_id"] = max((c["id"] for c in result["cells"]), default=0) + 1
            atomic_write_text(result_path, encode_json(result))
            # Refresh the checkpoint after server validation adds frozen metadata.
            self._checkpoint(row, output)
            hashes = self._hashes(output)
            self._stage(run_id, "publishing")
            final = directory / "original"
            if output != final:
                if final.exists():
                    raise _error("A previous published result already exists.", 409)
                os.replace(output, final)
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                job = connection.execute("SELECT cancel_requested,assistance_id FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
                if job['cancel_requested']:
                    raise InterruptedError("Cancelled before publication")
                connection.execute(
                    "UPDATE count_runs SET status='completed',original_result_json=?,original_hashes_json=?,"
                    "latest_revision=0,updated_at=? WHERE id=? AND status='running'",
                    (encode_json(result), encode_json(hashes), utc_now(), run_id),
                )
                connection.execute("""UPDATE count_jobs SET state='done',stage='finished',outcome=?,updated_at=? WHERE run_id=?""",
                    ("assisted_candidate_count" if job['assistance_id'] else "automatic_candidate_count", utc_now(), run_id))
        except Exception as exc:
            # Neither exception strings nor subprocess output enter API responses:
            # model libraries may put protected filenames and metadata in errors.
            try:
                failure_dir = self.app._path(self._run(case_id, run_id)['output_relpath'])
                failure_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_text(failure_dir / f'error-{uuid.uuid4().hex}.local.txt', traceback.format_exc())
            except Exception:
                pass
            if self._cancelled(run_id):
                self._fail(run_id, "CANCELLED")
            elif self._closed.is_set():
                self._requeue(run_id)
            elif isinstance(exc, subprocess.TimeoutExpired) or (isinstance(exc, OSError) and exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT}):
                if not self._requeue(run_id, transient=True):
                    self._fail(run_id, "RETRY_EXHAUSTED")
            else:
                self._fail(run_id, "STORAGE_FULL" if isinstance(exc, OSError) and exc.errno == errno.ENOSPC else "INVALID_OR_UNAVAILABLE_RESULT")
        finally:
            self._active_run_id = None

    def _safe_file(self, directory: Path, filename: str) -> Path:
        path = directory / filename
        if not directory.resolve().is_relative_to(self.root) or path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise _error("A count artifact is unavailable or invalid.", 409)
        return path

    def _validate_adaptive_provenance(self, result: dict, protocol: dict, row: Any, output: Path) -> None:
        from ..counting.profiles import sealed
        assistance = protocol.get('assistance')
        mode = 'assisted' if assistance is not None else 'automatic'
        expected_model = {'localization': 'reviewer_native_region' if assistance is not None else 'v2_geometric_baseline',
                          'direct_ethmoid_model': 'unavailable_without_promoted_weights'}
        if (result.get('method_version') != 'scan-adaptive-native-0.3.0' or
            result.get('analysis_mode') != mode or result.get('outcome') != mode + '_candidate_count' or
            result.get('model_identity') != expected_model):
            raise _error('The adaptive result identity differs from its frozen processing authority.')
        with self.store.connect() as db:
            job = db.execute('SELECT assistance_id,parent_run_id FROM count_jobs WHERE run_id=?', (row['id'],)).fetchone()
        if job is None or (job['assistance_id'] is not None) != (assistance is not None):
            raise _error('The adaptive assistance authority differs from the queued job.')
        actual = result.get('assistance')
        if assistance is None:
            if actual is not None:
                raise _error('An automatic result cannot acquire assistance lineage.')
            # A failed automatic attempt is retry provenance, not assistance.
            if job['parent_run_id'] is not None:
                self._validate_automatic_retry_parent(row['case_id'], job['parent_run_id'], row['source_sha256'])
        else:
            fixed = {key: assistance[key] for key in ('schema_version', 'source_sha256', 'revision_id', 'parent_run_id', 'region_sha256')}
            fixed['exclusion_sha256'] = assistance.get('exclusion_sha256')
            if (job['assistance_id'] != fixed['revision_id'] or job['parent_run_id'] != fixed['parent_run_id'] or
                not isinstance(actual, dict) or set(actual) != {*fixed, 'midline_ras_x', 'sha256'} or
                any(actual.get(key) != value for key, value in fixed.items()) or sealed(actual) != actual):
                raise _error('The assisted result does not match its immutable boundary and parent.')
            midline = actual.get('midline_ras_x')
            if type(midline) not in (float, int) or not np.isfinite(midline):
                raise _error('The assisted result has invalid physical laterality metadata.')
        expected_verification = {'status': 'verified', 'source_sha256': row['source_sha256'],
            'sha256': {key: sha256_file(self._safe_file(output, name)) for key, name in ARTIFACTS.items()}}
        if result.get('artifact_verification') != expected_verification:
            raise _error('The adaptive artifact verification does not match the source and saved masks.')

    def _hashes(self, directory: Path) -> dict[str, str]:
        return {name: sha256_file(self._safe_file(directory, name))
                for name in ["result.json", *ARTIFACTS.values()]}

    def _verify_files(self, directory: Path, hashes: dict[str, str]) -> None:
        if set(hashes) != {"result.json", *ARTIFACTS.values()}:
            raise _error("The count artifact manifest is invalid.", 409)
        for name, expected in hashes.items():
            if sha256_file(self._safe_file(directory, name)) != expected:
                raise _error("A count artifact failed its integrity check.", 409)

    @staticmethod
    def _load_mask(path: Path, source: nib.Nifti1Image, *, binary: bool = False, region: bool = False) -> np.ndarray:
        try:
            image = nib.load(path)
            if image.shape != source.shape or not np.allclose(image.affine, source.affine, rtol=0, atol=1e-5):
                raise ValueError("geometry")
            values = np.asanyarray(image.dataobj)
            limit = 1 if binary else 2 if region else 65535
            if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > limit) or not np.array_equal(values, np.rint(values)):
                raise ValueError("values")
            if not binary and not region and image.get_data_dtype() != np.dtype("uint16"):
                raise ValueError("dtype")
            return values.astype(np.uint16)
        except (OSError, EOFError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise _error("A count mask has invalid values, type, or source geometry.") from exc

    def _validate_result(self, result: Any, source_path: Path, output: Path, row: Any) -> dict[str, Any]:
        if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("status") != "completed":
            raise _error("The counting result format is unsupported.")
        if result.get("source_sha256") != row["source_sha256"] or result.get("protocol_version") != row["protocol_version"]:
            raise _error("The counting result does not match its source or protocol.")
        expected_algorithm = decode_json(row["protocol_json"], {}).get("algorithm")
        if not isinstance(expected_algorithm, dict) or not expected_algorithm or result.get("config") != expected_algorithm:
            raise _error("The effective counting settings do not match the frozen protocol.")
        if result.get("artifacts") != ARTIFACTS or not isinstance(result.get("method_version"), str):
            raise _error("The counting artifact manifest is invalid.")
        source = nib.load(source_path)
        cavities = self._load_mask(self._safe_file(output, ARTIFACTS["cavities"]), source)
        air = self._load_mask(self._safe_file(output, ARTIFACTS["air"]), source)
        region = self._load_mask(self._safe_file(output, ARTIFACTS["region"]), source, region=True)
        # RAS +X is patient-right, independent of native voxel-axis direction.
        # Compare bilateral regions, never assume scanner X=0 is the midline.
        region_x: dict[int, float] = {}
        for label in (1, 2):
            locations = np.argwhere(region == label)
            if len(locations):
                region_x[label] = float(nib.affines.apply_affine(source.affine, locations.mean(axis=0))[0])
        if len(region_x) == 2 and region_x[1] >= region_x[2]:
            raise _error("The localized compartments have reversed physical laterality.")
        if np.any((air != 0) & (air != cavities)):
            raise _error("Air labels do not match the cavity instances.")
        expected_air = np.where(np.asanyarray(source.dataobj) < expected_algorithm["air_hu"], cavities, 0)
        if not np.array_equal(air, expected_air):
            raise _error("Air labels do not match calibrated source voxels.")
        cells = result.get("cells")
        if not isinstance(cells, list) or not cells:
            raise _error("The cell registry is invalid.")
        seen: set[int] = set()
        for cell in cells:
            self._validate_cell(cell, source)
            identity = cell["id"]
            if identity in seen or cell.get("origin") != "automatic":
                raise _error("The automatic cell registry is invalid.")
            seen.add(identity)
            selected = cavities == identity
            positions = np.argwhere(selected)
            if not len(positions) or not np.allclose(cell["centroid_ras_mm"], nib.affines.apply_affine(source.affine, positions.mean(axis=0)), rtol=0, atol=1e-4):
                raise _error("A cell location does not match its native compartment mask.")
            if np.any(region[selected] != (1 if cell["side"] == "left" else 2)):
                raise _error("A cell does not match the localized side of its compartment.")
            if cell.get("voxel_count") != int(np.count_nonzero(cavities == identity)) or cell.get("air_voxel_count") != int(np.count_nonzero(air == identity)):
                raise _error("Cell measurements do not match the instance masks.")
        if seen != set(map(int, np.unique(cavities))) - {0}:
            raise _error("The cell registry does not match the instance mask.")
        if result.get("counts") != _counts(cells):
            raise _error("The reported count does not match the cell registry.")
        if result["counts"]["total"] == 0:
            raise _error("Automatic inference did not resolve any countable cells.")
        if not isinstance(result.get("warnings", []), list) or not all(isinstance(w, str) for w in result.get("warnings", [])):
            raise _error("Count warnings are invalid.")
        encode_json(result)
        return result

    @staticmethod
    def _validate_cell(cell: Any, source: nib.Nifti1Image) -> None:
        if not isinstance(cell, dict) or type(cell.get("id")) is not int or not 1 <= cell["id"] <= 65535:
            raise _error("Cell identifiers must be integers from 1 through 65535.")
        if cell.get("side") not in {"left", "right"} or type(cell.get("included")) is not bool or type(cell.get("uncertain")) is not bool:
            raise _error("Cell side, inclusion, or uncertainty is invalid.")
        if not isinstance(cell.get("reasons"), list) or not all(isinstance(r, str) and len(r) <= 2000 for r in cell["reasons"]):
            raise _error("Cell reasons are invalid.")
        try:
            point = np.asarray(cell["centroid_ras_mm"], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("point")
            voxel = nib.affines.apply_affine(np.linalg.inv(source.affine), point)
            if np.any(voxel < -0.5) or np.any(voxel > np.asarray(source.shape) - 0.5):
                raise ValueError("bounds")
        except (KeyError, TypeError, ValueError) as exc:
            raise _error("Cell coordinates must identify a location inside the source volume.") from exc

    def _snapshot(self, row: Any, revision: int | None = None) -> tuple[dict[str, Any], Path, dict[str, str]]:
        if row["status"] != "completed":
            raise _error("This count run has no completed result.", 409)
        revision = row["latest_revision"] if revision is None else revision
        if revision == 0:
            result = decode_json(row["original_result_json"], {})
            directory = self.app._path(row["output_relpath"]) / "original"
            hashes = decode_json(row["original_hashes_json"], {})
        else:
            with self.store.connect() as connection:
                item = connection.execute("SELECT * FROM cell_revisions WHERE run_id=? AND revision=?", (row["id"], revision)).fetchone()
            if item is None:
                raise _error("Cell revision not found.", 404)
            result = decode_json(item["result_json"], {})
            directory = self.app._path(item["artifact_relpath"])
            hashes = decode_json(item["hashes_json"], {})
        self._verify_files(directory, hashes)
        if json.loads((directory / "result.json").read_text(encoding="utf-8")) != result:
            raise _error("The saved registry does not match its revision.", 409)
        return result, directory, hashes

    def list_runs(self, case_id: str) -> dict[str, Any]:
        self.app.get_case(case_id)
        with self.store.connect() as connection:
            rows = connection.execute("SELECT id FROM count_runs WHERE case_id=? ORDER BY created_at DESC,id", (case_id,)).fetchall()
        return {"runs": [self.get_run(case_id, row["id"]) for row in rows]}

    def get_run(self, case_id: str, run_id: str, *, revision: int | None = None) -> dict[str, Any]:
        self._source(case_id)
        row = self._run(case_id, run_id)
        result = None
        if row["status"] == "completed":
            result, _, _ = self._snapshot(row, revision)
        original = decode_json(row["original_result_json"], {})
        protocol = decode_json(row["protocol_json"], {})
        quality = result.get("quality", {}) if result else protocol.get("source_quality", {})
        with self.store.connect() as db:
            job = db.execute("SELECT * FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
        return {
            "run_id": row["id"], "case_id": case_id, "status": row["status"],
            "latest_revision": row["latest_revision"], "revision": result.get("revision") if result else None,
            "original_counts": original.get("counts") if row["status"] == "completed" else None,
            "counts": result.get("counts") if result else None,
            "cells": result.get("cells", []) if result else [],
            "warnings": result.get("warnings", []) if result else quality.get("warnings", []),
            "quality": quality,
            "acquisition": result.get("acquisition", {}) if result else protocol.get("acquisition", {}),
            "lineage": result.get("lineage", []) if result else [],
            "failure_code": row["failure_code"],
            "message": "The count did not complete. No anatomical count is available." if row["status"] == "failed" else None,
            "method_version": result.get("method_version") if result else None,
            "protocol_version": row["protocol_version"], "training_eligibility": ELIGIBILITY,
            "source_sha256": row["source_sha256"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "stage": job["stage"] if job else row["status"],
            "outcome": job["outcome"] if job else ("automatic_candidate_count" if result else "needs_review" if row["failure_code"] in REVIEW_CODES else "technical_error" if row["status"] == "failed" else None),
            "attempt": job["attempt"] if job else 1,
            "retry_count": job["retry_count"] if job else 0,
            "assistance": {"boundary_id": job["assistance_id"], "parent_run_id": job["parent_run_id"]} if job and job["assistance_id"] else None,
            "scan_profile": protocol.get("scan_profile"), "processing_plan": protocol.get("processing_plan"),
            "model_identity": result.get("model_identity") if result else None,
            "analysis_mode": "assisted" if job and job["assistance_id"] else "automatic",
        }

    def cancel(self, case_id: str, run_id: str) -> dict:
        row = self._run(case_id, run_id)
        if row['status'] not in {'queued', 'running'}:
            return self.get_run(case_id, run_id)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._run(case_id, run_id, db)
            if current['status'] in {'queued', 'running'}:
                db.execute("UPDATE count_jobs SET cancel_requested=1 WHERE run_id=?", (run_id,))
                if current['status'] == 'queued':
                    db.execute("UPDATE count_runs SET status='failed',failure_code='CANCELLED',updated_at=? WHERE id=?", (utc_now(), run_id))
                    db.execute("UPDATE count_jobs SET state='done',stage='finished',outcome='cancelled' WHERE run_id=?", (run_id,))
        self._queue.notify()
        return self.get_run(case_id, run_id)

    def retry(self, case_id: str, run_id: str, idempotency_key: str | None = None) -> dict:
        row = self._run(case_id, run_id)
        if row['status'] != 'failed':
            raise _error("Only an unsuccessful run can be retried.", 409)
        with self.store.connect() as db:
            job = db.execute("SELECT assistance_id,parent_run_id FROM count_jobs WHERE run_id=?", (run_id,)).fetchone()
        return self.create_run(case_id, idempotency_key, boundary_id=job['assistance_id'] if job else None,
                               parent_run_id=job['parent_run_id'] if job and job['assistance_id'] else run_id)

    def save_boundary(self, case_id: str, run_id: str, mask_path: Path, note: str = '') -> dict:
        row = self._run(case_id, run_id)
        case, source_path = self._source(case_id)
        if case['state'] == 'APPROVED' or row['status'] in {'queued', 'running'}:
            raise _error("Finish the active run and use an editable case before saving boundaries.", 409)
        if len(note) > 2000:
            raise _error("The boundary note is too long.")
        source = nib.load(source_path)
        values = self._load_mask(mask_path, source, region=True)
        if set(np.unique(values)) - {0} != {1, 2}:
            raise _error("Draw both left (1) and right (2) anatomical regions before assisted counting.")
        centers = [nib.affines.apply_affine(source.affine, np.argwhere(values == side).mean(axis=0))[0] for side in (1, 2)]
        if centers[0] >= centers[1]:
            raise _error("The anatomical boundaries have reversed left/right labels.")
        identity = uuid.uuid4().hex
        path = self.app._path(f"cases/{case_id}/boundaries/{identity}.nii.gz")
        image = nib.Nifti1Image(values.astype(np.uint8), source.affine)
        image.set_qform(source.affine, code=1)
        image.set_sform(source.affine, code=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save_nifti(image, path)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.app._get_case_row(db, case_id)
            if current['state'] == 'APPROVED' or current['source_sha256'] != row['source_sha256']:
                raise _error("The source or review state changed before saving boundaries.", 409)
            db.execute("INSERT INTO boundary_revisions VALUES(?,?,?,?,?,?,?,?)", (
                identity, case_id, run_id, row['source_sha256'], self.app._relative(path), sha256_file(path), note, utc_now()))
        return self.get_boundary(case_id, identity)

    @staticmethod
    def _boundary_metadata(row) -> dict:
        return {"boundary_id": row['id'], **{key: row[key] for key in ('case_id', 'parent_run_id', 'source_sha256', 'mask_sha256', 'note', 'created_at')}}

    def list_boundaries(self, case_id: str, run_id: str) -> dict:
        parent = self._run(case_id, run_id)
        case, _ = self._source(case_id)
        with self.store.connect() as db:
            rows = db.execute("""SELECT * FROM boundary_revisions
                WHERE case_id=? AND parent_run_id=? ORDER BY created_at DESC,rowid DESC""",
                (case_id, run_id)).fetchall()
        if parent['source_sha256'] != case['source_sha256'] or any(row['source_sha256'] != parent['source_sha256'] for row in rows):
            raise _error("The saved boundary history does not match the current source.", 409)
        return {"boundaries": [self._boundary_metadata(row) for row in rows]}

    def get_boundary(self, case_id: str, boundary_id: str) -> dict:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM boundary_revisions WHERE id=? AND case_id=?", (boundary_id, case_id)).fetchone()
        if row is None:
            raise _error("Anatomical boundary revision not found.", 404)
        parent = self._run(case_id, row['parent_run_id'])
        case, _ = self._source(case_id)
        if row['source_sha256'] != parent['source_sha256'] or row['source_sha256'] != case['source_sha256']:
            raise _error("The boundary revision does not match the current source and parent run.", 409)
        return self._boundary_metadata(row)

    def boundary_path(self, case_id: str, boundary_id: str) -> Path:
        boundary = self.get_boundary(case_id, boundary_id)
        case, _ = self._source(case_id)
        with self.store.connect() as db:
            row = db.execute("SELECT mask_relpath FROM boundary_revisions WHERE id=?", (boundary_id,)).fetchone()
        path = self.app._path(row[0])
        if boundary['source_sha256'] != case['source_sha256'] or path.is_symlink() or not path.is_file() or sha256_file(path) != boundary['mask_sha256']:
            raise _error("The boundary revision failed its source or integrity check.", 409)
        return path

    def artifact(self, case_id: str, run_id: str, artifact: str, revision: int | None = 0) -> Path:
        if artifact not in ARTIFACTS:
            raise _error("Count artifact not found.", 404)
        self._source(case_id)
        row = self._run(case_id, run_id)
        _, directory, _ = self._snapshot(row, revision)
        return self._safe_file(directory, ARTIFACTS[artifact])

    def revise(self, case_id: str, run_id: str, parent_revision: int, operation: dict[str, Any], mask_path: Path | None = None) -> dict[str, Any]:
        if type(parent_revision) is not int or parent_revision < 0:
            raise _error("The parent revision is invalid.")
        case, source_path = self._source(case_id)
        if case["state"] == "APPROVED":
            raise _error("Approved cases cannot be edited.", 409)
        row = self._run(case_id, run_id)
        if parent_revision != row["latest_revision"]:
            raise _error("This result changed. Reload before saving the correction.", 409)
        result, directory, _ = self._snapshot(row, parent_revision)
        result = copy.deepcopy(result)
        source = nib.load(source_path)
        cavities = self._load_mask(directory / ARTIFACTS["cavities"], source)
        air = self._load_mask(directory / ARTIFACTS["air"], source)
        region = self._load_mask(directory / ARTIFACTS["region"], source, region=True)
        selection = self._load_mask(mask_path, source, binary=True).astype(bool) if mask_path else None
        self._apply_operation(result, operation, cavities, air, source, selection)
        revision = parent_revision + 1
        result["revision"] = revision
        result["parent_revision"] = parent_revision
        result["counts"] = _counts(result["cells"])
        # A unique directory also permits retry after a disk/database failure;
        # an uncommitted orphan can never be mistaken for the saved revision.
        destination = self.app._path(row["output_relpath"]) / f"revision_{revision:06d}_{uuid.uuid4().hex}"
        # Serialize writes with a SQLite transaction and repeat the parent check.
        # No stale request may overwrite a revision's immutable files.
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._run(case_id, run_id, connection)
            if current["latest_revision"] != parent_revision:
                raise _error("This result changed. Reload before saving the correction.", 409)
            current_case = self.app._get_case_row(connection, case_id)
            if current_case["state"] == "APPROVED":
                raise _error("Approved cases cannot be edited.", 409)
            if sha256_file(source_path) != current["source_sha256"]:
                raise _error("The source volume failed its integrity check.", 409)
            destination.mkdir(parents=True)
            for key, values in (("cavities", cavities), ("air", air), ("region", region)):
                image = nib.Nifti1Image(values.astype(np.uint16), source.affine)
                image.set_qform(source.affine, code=1)
                image.set_sform(source.affine, code=1)
                image.header.set_xyzt_units("mm")
                _atomic_save_nifti(image, destination / ARTIFACTS[key])
            atomic_write_text(destination / "result.json", encode_json(result))
            hashes = self._hashes(destination)
            now = utc_now()
            connection.execute(
                "INSERT INTO cell_revisions(run_id,revision,parent_revision,result_json,artifact_relpath,"
                "hashes_json,operation_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, revision, parent_revision, encode_json(result), self.app._relative(destination),
                 encode_json(hashes), encode_json(operation), now),
            )
            connection.execute("UPDATE count_runs SET latest_revision=?,updated_at=? WHERE id=?", (revision, now, run_id))
        return self.get_run(case_id, run_id)

    def _apply_operation(self, result: dict[str, Any], operation: dict[str, Any], cavities: np.ndarray, air: np.ndarray, source: nib.Nifti1Image, selection: np.ndarray | None) -> None:
        allowed = {"type", "id", "ids", "side", "centroid_ras_mm", "included", "uncertain", "reasons"}
        if not isinstance(operation, dict) or set(operation) - allowed:
            raise _error("Correction fields are invalid.")
        kind = operation.get("type")
        cells = result["cells"]
        registry = {c["id"]: c for c in cells}
        affected: list[int] = []
        created: list[int] = []
        source_values: np.ndarray | None = None

        def classify_air(identity: int, selected: np.ndarray) -> None:
            nonlocal source_values
            parameters = result.get("config", {})
            if isinstance(parameters, dict) and "algorithm" in parameters:
                parameters = parameters["algorithm"]
            threshold = parameters.get("air_hu") if isinstance(parameters, dict) else None
            if type(threshold) not in {int, float} or not np.isfinite(threshold) or not -1000 < threshold < 0:
                raise _error("The saved run lacks a valid frozen air-content threshold.", 409)
            if source_values is None:
                source_values = np.asanyarray(source.dataobj)
            air[selected] = np.where(source_values[selected] < threshold, identity, 0)

        def current(identity: Any) -> dict[str, Any]:
            if type(identity) is not int or identity not in registry or registry[identity].get("retired", False):
                raise _error("Select an existing, non-retired cell.")
            return registry[identity]

        def allocate(side: str, selected: np.ndarray | None, parents: list[int], point: Any = None, *, observe_air: bool = False) -> dict[str, Any]:
            identity = result["next_instance_id"]
            if identity > 65535:
                raise _error("The cell identifier range is exhausted.", 409)
            result["next_instance_id"] += 1
            if selected is not None and np.any(selected):
                point = nib.affines.apply_affine(source.affine, np.argwhere(selected).mean(axis=0)).tolist()
            cell = {"id": identity, "side": side, "centroid_ras_mm": point,
                    "origin": "manual", "included": operation.get("included", True), "uncertain": operation.get("uncertain", True),
                    "reasons": operation.get("reasons", []), "voxel_count": 0, "air_voxel_count": 0,
                    "parent_ids": parents, "retired": False}
            self._validate_cell(cell, source)
            if selected is not None:
                cavities[selected] = identity
                if observe_air:
                    classify_air(identity, selected)
                else:
                    air[selected & (air != 0)] = identity
            cells.append(cell)
            created.append(identity)
            return cell

        if kind == "add":
            if selection is not None and (not np.any(selection) or np.any(cavities[selection] != 0)):
                raise _error("A new cell mask must be nonempty and cannot overwrite another cell.")
            allocate(operation.get("side"), selection, [], operation.get("centroid_ras_mm"), observe_air=True)
        elif kind == "remove":
            cell = current(operation.get("id"))
            if selection is not None:
                raise _error("Remove does not accept a mask.")
            cell["included"] = False
            cell["corrected"] = True
            if "reasons" in operation:
                cell["reasons"] = operation["reasons"]
                self._validate_cell(cell, source)
            affected = [cell["id"]]
        elif kind == "merge":
            ids = operation.get("ids")
            if not isinstance(ids, list) or len(ids) < 2 or not all(type(identity) is int for identity in ids) or len(set(ids)) != len(ids):
                raise _error("Merge requires at least two different cell identifiers.")
            parents = [current(identity) for identity in ids]
            if len({c["side"] for c in parents}) != 1 or selection is not None:
                raise _error("Merge requires cells on one side and does not accept a mask.")
            selected = np.isin(cavities, ids)
            point = np.mean([c["centroid_ras_mm"] for c in parents], axis=0).tolist()
            allocate(parents[0]["side"], selected, ids, point)
            for cell in parents:
                cell.update(included=False, retired=True)
            affected = ids
        elif kind == "split":
            cell = current(operation.get("id"))
            target = cavities == cell["id"]
            if selection is None or not np.any(selection) or np.any(selection & ~target) or not np.any(target & ~selection):
                raise _error("Split requires a nonempty proper subset of the selected cell.")
            other = target & ~selection
            allocate(cell["side"], selection, [cell["id"]])
            allocate(cell["side"], other, [cell["id"]])
            cell.update(included=False, retired=True)
            affected = [cell["id"]]
        elif kind == "update":
            cell = current(operation.get("id"))
            affected = [cell["id"]]
            if selection is not None:
                if not np.any(selection) or np.any((cavities[selection] != 0) & (cavities[selection] != cell["id"])):
                    raise _error("A boundary edit must be nonempty and cannot overwrite another cell.")
                removed = (cavities == cell["id"]) & ~selection
                cavities[removed] = 0
                air[removed] = 0
                cavities[selection] = cell["id"]
                classify_air(cell["id"], selection)
                cell["centroid_ras_mm"] = nib.affines.apply_affine(source.affine, np.argwhere(selection).mean(axis=0)).tolist()
            for field in ("side", "centroid_ras_mm", "included", "uncertain", "reasons"):
                if field in operation:
                    cell[field] = operation[field]
            cell["corrected"] = True
            self._validate_cell(cell, source)
        else:
            raise _error("The correction operation is unsupported.")
        for cell in cells:
            # A removed/retired cell must disappear from the current overlays.
            # Prior revisions retain its original geometry and air content.
            if not cell["included"]:
                cavities[cavities == cell["id"]] = 0
                air[air == cell["id"]] = 0
            cell["voxel_count"] = int(np.count_nonzero(cavities == cell["id"]))
            cell["air_voxel_count"] = int(np.count_nonzero(air == cell["id"]))
        result["lineage"].append({"revision": result["revision"] + 1, "operation": kind,
                                  "parent_ids": affected, "created_ids": created})

    def export(self, case_id: str, run_id: str, revision: int | None = None) -> Path:
        case, source = self._source(case_id)
        row = self._run(case_id, run_id)
        original, original_dir, original_hashes = self._snapshot(row, 0)
        corrected, corrected_dir, corrected_hashes = self._snapshot(row, revision)
        identity = report_identity(original, case["case_label"])
        native_spacing = nib.affines.voxel_sizes(nib.load(source).affine).tolist()
        reports = build_count_reports(
            source, corrected_dir / ARTIFACTS["cavities"], corrected_dir / ARTIFACTS["air"],
            {**corrected, "original_counts": original["counts"], "acquisition": original.get("acquisition"), "quality": original.get("quality")}, case["case_label"],
        )
        manifest = {
            "schema_version": 1, "case_label": case["case_label"], "run_id": run_id,
            "identity": identity,
            "source_sha256": row["source_sha256"], "protocol_version": row["protocol_version"],
            "protocol_sha256": original["protocol_sha256"], "method_version": original["method_version"],
            "original_counts": original["counts"], "corrected_counts": corrected["counts"],
            "revision": corrected["revision"], "original_artifact_sha256": original_hashes,
            "corrected_artifact_sha256": corrected_hashes, "training_eligibility": ELIGIBILITY,
            "clinical_accuracy": "UNMEASURED", "warnings": corrected.get("warnings", []),
            "quality": original.get("quality", {}), "acquisition": original.get("acquisition", {}),
            "config": original.get("config", {}),
            **({key: original.get(key) for key in ("analysis_mode", "assistance", "scan_profile", "processing_plan", "model_identity")} if row['protocol_version'] == "ethmoid-count-prototype-v3" else {}),
            "report_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in reports.items()},
        }
        # A second request must never replace a ZIP that FileResponse is still
        # streaming. Windows locks open download files against replacement.
        export_dir = self.app._path(row["output_relpath"]) / f"export_{uuid.uuid4().hex}"
        export_dir.mkdir()
        target = export_dir / f"count_revision_{corrected['revision']:06d}.zip"
        temporary = target.with_name(f".{uuid.uuid4().hex}.zip")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", encode_json(manifest))
                exported_protocol = decode_json(row['protocol_json'], {})
                if exported_protocol.get('assistance'):
                    exported_protocol['assistance'] = {key: value for key, value in exported_protocol['assistance'].items() if key not in {'region_path', 'exclusion_path'}}
                archive.writestr("protocol.json", encode_json(exported_protocol))
                for name, data in reports.items():
                    archive.writestr(name, data)
                count_stream = io.StringIO(newline="")
                count_writer = csv.writer(count_stream)
                count_writer.writerow(["result", "revision", "left", "right", "total", "resolution_class", "slice_thickness_mm", "slice_interval_mm", "quality_flags",
                                       "study_label", "series_label", "legacy_case_label", "source_sha256",
                                       "acquisition_spacing_x_mm", "acquisition_spacing_y_mm", "acquisition_spacing_z_mm",
                                       "native_spacing_x_mm", "native_spacing_y_mm", "native_spacing_z_mm", "analysis_mode", "method_version", "model_identity", "processing_plan_sha256"])
                acquisition = original.get("acquisition") or {}
                acquired_spacing = acquisition.get("spacing_mm") or [None, None, None]
                for name, snapshot in (("original", original), ("reviewed", corrected)):
                    quality = snapshot.get("quality", {})
                    count_writer.writerow([name, snapshot["revision"], *[snapshot["counts"][side] for side in ("left", "right", "total")],
                                           quality.get("resolution_class", "unknown"), acquisition.get("slice_thickness_mm"), acquired_spacing[2],
                                           " ".join(quality.get("flags", [])), identity["study_label"], identity["series_label"], identity["legacy_case_label"], row["source_sha256"],
                                           *acquired_spacing, *native_spacing, snapshot.get('analysis_mode', 'automatic'), snapshot.get('method_version'),
                                           encode_json(snapshot.get('model_identity')), (snapshot.get('processing_plan') or {}).get('sha256')])
                archive.writestr("counts.csv", count_stream.getvalue())
                for prefix, result, directory in (("original", original, original_dir), ("reviewed", corrected, corrected_dir)):
                    for name in ["result.json", *ARTIFACTS.values()]:
                        archive.write(self._safe_file(directory, name), f"{prefix}/{name}")
                    stream = io.StringIO(newline="")
                    writer = csv.writer(stream)
                    writer.writerow(["id", "side", "included", "origin", "uncertain", "x_ras_mm", "y_ras_mm", "z_ras_mm", "voxel_count", "air_voxel_count", "parent_ids"])
                    for cell in result["cells"]:
                        writer.writerow([cell["id"], cell["side"], cell["included"], cell["origin"], cell["uncertain"],
                                         *cell["centroid_ras_mm"], cell["voxel_count"], cell["air_voxel_count"],
                                         " ".join(map(str, cell.get("parent_ids", [])))])
                    archive.writestr(f"{prefix}/cells.csv", stream.getvalue())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
            if not target.exists():
                export_dir.rmdir()
        return target
