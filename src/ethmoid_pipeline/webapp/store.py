from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 4


class Store:
    """Small SQLite store for workflow state; image bytes stay on disk."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "ethmoid-app.sqlite3"
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    case_label TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL CHECK(source_type IN ('DICOM', 'NIFTI')),
                    state TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    volume_relpath TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    spacing_json TEXT NOT NULL,
                    affine_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    validation_details_json TEXT NOT NULL,
                    qc_checklist_json TEXT NOT NULL,
                    reviewer_notes TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'Draft',
                    viewer_state_json TEXT NOT NULL DEFAULT '{}',
                    protocol_version TEXT NOT NULL,
                    latest_revision INTEGER,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotation_revisions (
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    mask_relpath TEXT NOT NULL,
                    mask_sha256 TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    viewer_state_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(case_id, revision)
                );
                CREATE TABLE IF NOT EXISTS imports (
                    id TEXT PRIMARY KEY,
                    case_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    staging_relpath TEXT NOT NULL,
                    series_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_updated_at ON cases(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_imports_updated_at ON imports(updated_at DESC);
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            previous_version = int(row["value"]) if row is not None else None
            if previous_version not in {None, 1, 2, 3, SCHEMA_VERSION}:
                raise RuntimeError("The web application database schema is unsupported.")
            if previous_version in {1, 2, 3}:
                # SQLite's backup API includes committed WAL pages. Keep the backup
                # beside the protected database, never in the source checkout.
                backup_path = self.root / f"schema-v{previous_version}-{uuid.uuid4().hex}.sqlite3"
                with sqlite3.connect(backup_path) as backup:
                    connection.backup(backup)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS count_runs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    idempotency_key TEXT,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    source_sha256 TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    protocol_json TEXT NOT NULL,
                    output_relpath TEXT NOT NULL,
                    original_result_json TEXT,
                    original_hashes_json TEXT,
                    latest_revision INTEGER,
                    failure_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(case_id, idempotency_key)
                )"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_count_run_active
                    ON count_runs(case_id) WHERE status IN ('queued','running')"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cell_revisions (
                    run_id TEXT NOT NULL REFERENCES count_runs(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    parent_revision INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    artifact_relpath TEXT NOT NULL,
                    hashes_json TEXT NOT NULL,
                    operation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, revision)
                )"""
            )
            # Add a study/series catalog around legacy volume IDs. Existing case,
            # annotation and count-run rows are never renamed or reconstructed.
            connection.execute("""CREATE TABLE IF NOT EXISTS catalog_sources (
                id TEXT PRIMARY KEY, root_path TEXT NOT NULL UNIQUE,
                inventory_sha256 TEXT, status TEXT NOT NULL DEFAULT 'indexing',
                error_code TEXT, updated_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS catalog_studies (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES catalog_sources(id),
                study_uid TEXT NOT NULL, study_label TEXT NOT NULL,
                selected_series_id TEXT, recommended_series_id TEXT,
                UNIQUE(source_id,study_uid), UNIQUE(source_id,study_label))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS catalog_series (
                id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES catalog_studies(id),
                series_uid TEXT NOT NULL, series_label TEXT NOT NULL,
                descriptor_json TEXT NOT NULL, files_json TEXT NOT NULL,
                inventory_signature TEXT NOT NULL, source_fingerprint TEXT,
                case_id TEXT REFERENCES cases(id) ON DELETE SET NULL,
                preparation_status TEXT NOT NULL DEFAULT 'not_prepared',
                reason_code TEXT, present INTEGER NOT NULL DEFAULT 1,
                UNIQUE(study_id,series_uid), UNIQUE(study_id,series_label))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS catalog_case_links (
                case_id TEXT PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE,
                series_id TEXT NOT NULL REFERENCES catalog_series(id),
                source_fingerprint TEXT NOT NULL, proof TEXT NOT NULL,
                acquisition_json TEXT NOT NULL DEFAULT '{}')""")
            connection.execute("""CREATE TABLE IF NOT EXISTS preparation_jobs (
                id TEXT PRIMARY KEY, series_id TEXT NOT NULL REFERENCES catalog_series(id),
                inventory_signature TEXT NOT NULL, status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100, attempt INTEGER NOT NULL DEFAULT 0,
                error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(series_id,inventory_signature))""")
            connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_preparation_active
                ON preparation_jobs(series_id) WHERE status IN ('queued','preparing')""")
            connection.execute("""CREATE TABLE IF NOT EXISTS count_jobs (
                run_id TEXT PRIMARY KEY REFERENCES count_runs(id) ON DELETE CASCADE,
                state TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'queued',
                outcome TEXT, owner TEXT, heartbeat_at TEXT,
                attempt INTEGER NOT NULL DEFAULT 0, retry_count INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                checkpoint_json TEXT, last_error TEXT, parent_run_id TEXT,
                assistance_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS boundary_revisions (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                parent_run_id TEXT NOT NULL REFERENCES count_runs(id) ON DELETE CASCADE,
                source_sha256 TEXT NOT NULL, mask_relpath TEXT NOT NULL,
                mask_sha256 TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS catalog_series_sources (
                series_id TEXT PRIMARY KEY REFERENCES catalog_series(id),
                source_id TEXT NOT NULL REFERENCES catalog_sources(id))""")
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )


def encode_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)
