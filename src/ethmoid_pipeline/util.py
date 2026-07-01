from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^Case\d{3}$")
SERIES_RE = re.compile(r"^Series\d{3}$")


class PipelineError(RuntimeError):
    """Expected fail-closed pipeline error with PHI-safe text."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_work_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise PipelineError("LOCALAPPDATA is unavailable; pass --work-root explicitly.")
    return Path(base) / "EthmoidComplexity" / "work"


def default_dcm2niix() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise PipelineError("LOCALAPPDATA is unavailable; pass --dcm2niix explicitly.")
    return (
        Path(base)
        / "EthmoidComplexity"
        / "tools"
        / "dcm2niix"
        / "v1.0.20260724"
        / "bin"
        / "dcm2niix.exe"
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
        return True
    except ValueError:
        return False


def validate_roots(dicom_root: Path, work_root: Path, code_root: Path | None = None) -> None:
    source = dicom_root.resolve(strict=True)
    work = work_root.resolve()
    code = (code_root or repo_root()).resolve(strict=True)
    if not source.is_dir():
        raise PipelineError("The DICOM root is not a directory.")
    if is_within(work, source) or is_within(source, work):
        raise PipelineError("The work root and DICOM root must be physically separate.")
    if is_within(work, code) or is_within(code, work):
        raise PipelineError("The PHI-sensitive work root must be outside the Git repository.")
    if is_within(source, code) or is_within(code, source):
        raise PipelineError("The DICOM root must be outside the Git repository.")


def validate_labels(case_label: str, series_label: str) -> None:
    if not CASE_RE.fullmatch(case_label) or not SERIES_RE.fullmatch(series_label):
        raise PipelineError("Case and series labels must use Case### and Series###.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Required local state is unreadable: {path.name}.") from exc
