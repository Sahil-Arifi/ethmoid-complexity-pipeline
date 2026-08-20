from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1


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
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError("The web application database schema is unsupported.")


def encode_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)
