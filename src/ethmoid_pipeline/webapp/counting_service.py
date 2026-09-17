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
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np

from ..util import atomic_write_text, repo_root, sha256_canonical_json, sha256_file, utc_now
from .service import AppService, WebAppError, inspect_nifti, _atomic_save_nifti
from .store import decode_json, encode_json
from .counting_reports import build_count_reports


ARTIFACTS = {"cavities": "cavities.nii.gz", "air": "air.nii.gz", "region": "region.nii.gz"}
PROTOCOL_VERSION = "ethmoid-count-prototype-v1"
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
    "INSUFFICIENT_COVERAGE",
}
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
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ethmoid-count")
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()
        self._closed = threading.Event()
        # App startup never treats an unfinished job from an earlier process as
        # completed or silently starts a second inference against the same case.
        with self.store.connect() as connection:
            connection.execute(
                "UPDATE count_runs SET status='failed', failure_code='INTERRUPTED', "
                "updated_at=? WHERE status IN ('queued','running')", (utc_now(),),
            )

    def close(self) -> None:
        self._closed.set()
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
        self._pool.shutdown(wait=False, cancel_futures=True)
        with self.store.connect() as connection:
            connection.execute(
                "UPDATE count_runs SET status='failed', failure_code='INTERRUPTED', "
                "updated_at=? WHERE status IN ('queued','running')", (utc_now(),),
            )

    def protocol(self) -> dict[str, Any]:
        if self._protocol is not None:
            value = copy.deepcopy(self._protocol)
        else:
            try:
                from ..counting.protocol import load_counting_protocol
                value = load_counting_protocol()
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                raise _error("The experimental counting configuration is unavailable.", 503) from exc
        if value.get("protocol_version") != PROTOCOL_VERSION:
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

    def create_run(self, case_id: str, idempotency_key: str | None = None) -> dict[str, Any]:
        if self._closed.is_set():
            raise _error("The counting worker is stopping.", 503)
        if idempotency_key is not None and (
            not 1 <= len(idempotency_key) <= 100 or
            not all(c.isascii() and (c.isalnum() or c in "-_") for c in idempotency_key)
        ):
            raise _error("The request identifier is invalid.")
        case, source = self._source(case_id)
        if case["state"] == "APPROVED":
            raise _error("Approved cases cannot start new count runs.", 409)
        info = inspect_nifti(source)
        if info.warnings:
            raise _error("Resolve source orientation metadata before counting.", 409)
        protocol = self.protocol()
        now = utc_now()
        run_id = uuid.uuid4().hex
        relative = f"cases/{case_id}/count-runs/{run_id}"
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_case = self.app._get_case_row(connection, case_id)
            if current_case["state"] == "APPROVED":
                raise _error("Approved cases cannot start new count runs.", 409)
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
            else:
                connection.execute(
                    "INSERT INTO count_runs(id,case_id,idempotency_key,status,source_sha256,"
                    "protocol_version,protocol_json,output_relpath,created_at,updated_at) "
                    "VALUES(?,?,?,'queued',?,?,?,?,?,?)",
                    (run_id, case_id, idempotency_key, case["source_sha256"], PROTOCOL_VERSION,
                     encode_json(protocol), relative, now, now),
                )
        if existing is None:
            try:
                self._pool.submit(self._work, case_id, run_id)
            except RuntimeError:
                self._fail(run_id, "INTERRUPTED")
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
        if "algorithm" in json.loads(config.read_text(encoding="utf-8")):
            command.extend(["--config", str(config)])
        with self._process_lock:
            if self._closed.is_set():
                raise RuntimeError("Counting worker is stopping")
            self._process = subprocess.Popen(
                command, env=environment, cwd=runtime, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            process = self._process
        try:
            process.wait(timeout=7200)
            if process.returncode and not (output / "result.json").is_file():
                raise RuntimeError("Inference did not produce a result")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            with self._process_lock:
                self._process = None

    def _fail(self, run_id: str, code: str) -> None:
        with self.store.connect() as connection:
            connection.execute(
                "UPDATE count_runs SET status='failed',failure_code=?,updated_at=? "
                "WHERE id=? AND status IN ('queued','running')", (code, utc_now(), run_id),
            )

    def _work(self, case_id: str, run_id: str) -> None:
        try:
            with self.store.connect() as connection:
                cursor = connection.execute(
                    "UPDATE count_runs SET status='running',updated_at=? WHERE id=? AND status='queued'",
                    (utc_now(), run_id),
                )
                if cursor.rowcount != 1:
                    return
            row = self._run(case_id, run_id)
            _, source = self._source(case_id)
            directory = self.app._path(row["output_relpath"])
            directory.mkdir(parents=True, exist_ok=True)
            protocol = decode_json(row["protocol_json"], {})
            config_path = directory / "protocol.json"
            atomic_write_text(config_path, encode_json(protocol))
            output = directory / "original"
            self._executor(source, output, config_path)
            self._source(case_id)
            result_path = self._safe_file(output, "result.json")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "failed":
                code = result.get("failure_code")
                self._fail(run_id, code if isinstance(code, str) and code in ENGINE_FAILURE_CODES else "INFERENCE_FAILED")
                return
            result = self._validate_result(result, source, output, row)
            result["protocol_sha256"] = protocol.get("sha256", sha256_canonical_json(protocol))
            result["protocol_snapshot_sha256"] = sha256_canonical_json(protocol)
            result["revision"] = 0
            result["parent_revision"] = None
            result["lineage"] = []
            result["next_instance_id"] = max((c["id"] for c in result["cells"]), default=0) + 1
            atomic_write_text(result_path, encode_json(result))
            hashes = self._hashes(output)
            with self.store.connect() as connection:
                connection.execute(
                    "UPDATE count_runs SET status='completed',original_result_json=?,original_hashes_json=?,"
                    "latest_revision=0,updated_at=? WHERE id=? AND status='running'",
                    (encode_json(result), encode_json(hashes), utc_now(), run_id),
                )
        except Exception:
            # Neither exception strings nor subprocess output enter API responses:
            # model libraries may put protected filenames and metadata in errors.
            self._fail(run_id, "INTERRUPTED" if self._closed.is_set() else "INVALID_OR_UNAVAILABLE_RESULT")

    def _safe_file(self, directory: Path, filename: str) -> Path:
        path = directory / filename
        if not directory.resolve().is_relative_to(self.root) or path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise _error("A count artifact is unavailable or invalid.", 409)
        return path

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
        return {
            "run_id": row["id"], "case_id": case_id, "status": row["status"],
            "latest_revision": row["latest_revision"], "revision": result.get("revision") if result else None,
            "original_counts": original.get("counts") if row["status"] == "completed" else None,
            "counts": result.get("counts") if result else None,
            "cells": result.get("cells", []) if result else [],
            "warnings": result.get("warnings", []) if result else [],
            "lineage": result.get("lineage", []) if result else [],
            "failure_code": row["failure_code"],
            "message": "The count did not complete. No anatomical count is available." if row["status"] == "failed" else None,
            "method_version": result.get("method_version") if result else None,
            "protocol_version": row["protocol_version"], "training_eligibility": ELIGIBILITY,
            "source_sha256": row["source_sha256"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

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
        reports = build_count_reports(
            source, corrected_dir / ARTIFACTS["cavities"], corrected_dir / ARTIFACTS["air"],
            {**corrected, "original_counts": original["counts"]}, case["case_label"],
        )
        manifest = {
            "schema_version": 1, "case_label": case["case_label"], "run_id": run_id,
            "source_sha256": row["source_sha256"], "protocol_version": row["protocol_version"],
            "protocol_sha256": original["protocol_sha256"], "method_version": original["method_version"],
            "original_counts": original["counts"], "corrected_counts": corrected["counts"],
            "revision": corrected["revision"], "original_artifact_sha256": original_hashes,
            "corrected_artifact_sha256": corrected_hashes, "training_eligibility": ELIGIBILITY,
            "clinical_accuracy": "UNMEASURED", "warnings": corrected.get("warnings", []),
            "config": original.get("config", {}),
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
                archive.writestr("protocol.json", row["protocol_json"])
                for name, data in reports.items():
                    archive.writestr(name, data)
                count_stream = io.StringIO(newline="")
                count_writer = csv.writer(count_stream)
                count_writer.writerow(["result", "revision", "left", "right", "total"])
                for name, snapshot in (("original", original), ("reviewed", corrected)):
                    count_writer.writerow([name, snapshot["revision"], *[snapshot["counts"][side] for side in ("left", "right", "total")]])
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
