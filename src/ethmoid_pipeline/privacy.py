from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib

from .util import PipelineError, atomic_write_json


UID_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3,}[0-9]+(?![0-9])")
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_BACKSLASH = chr(92)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:["
    + re.escape(_BACKSLASH)
    + r"/]|"
    + re.escape(_BACKSLASH * 2)
    + r")[^\r\n\x00]+"
)
SAFE_SIDECAR_KEYS = {
    "Modality",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatientDICOM",
    "ConversionSoftware",
    "ConversionSoftwareVersion",
}


def _sidecar_semantic_issues(value: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if value.get("Modality") != "CT":
        issues.append("sidecar Modality is not exactly CT")
    if value.get("ConversionSoftware") != "dcm2niix":
        issues.append("sidecar converter is not exactly dcm2niix")
    if value.get("ConversionSoftwareVersion") != "v1.0.20260724":
        issues.append("sidecar converter version is not the pinned release")
    orientation = value.get("ImageOrientationPatientDICOM")
    if not (
        isinstance(orientation, list)
        and len(orientation) == 6
        and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in orientation
        )
    ):
        issues.append("sidecar orientation is not six finite numeric values")
    for key in ("SliceThickness", "SpacingBetweenSlices"):
        if key in value and not (
            isinstance(value[key], (int, float))
            and math.isfinite(float(value[key]))
            and float(value[key]) > 0
        ):
            issues.append(f"sidecar {key} is not finite and positive")
    return issues


def redact_text(text: str, sensitive_values: Iterable[str], replacements: Iterable[str] = ()) -> str:
    redacted = text
    for value in sorted({item for item in sensitive_values if item}, key=len, reverse=True):
        redacted = redacted.replace(value, "<REDACTED>")
    for value in sorted({item for item in replacements if item}, key=len, reverse=True):
        redacted = redacted.replace(value, "<LOCAL_PATH>")
    redacted = WINDOWS_ABSOLUTE_PATH_PATTERN.sub("<LOCAL_PATH>", redacted)
    redacted = EMAIL_PATTERN.sub("<EMAIL_REDACTED>", redacted)
    return UID_PATTERN.sub("<UID_REDACTED>", redacted)


def sanitize_sidecar(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            original = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("dcm2niix produced an unreadable JSON sidecar.") from exc
    sanitized: dict[str, Any] = {"schema_version": 1}
    for key in sorted(SAFE_SIDECAR_KEYS):
        if key not in original:
            continue
        value = original.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            sanitized[key] = value
    semantic_issues = _sidecar_semantic_issues(sanitized)
    if semantic_issues:
        raise PipelineError("dcm2niix sidecar failed semantic validation: " + "; ".join(semantic_issues))
    atomic_write_json(path, sanitized)
    return sanitized


def scrub_nifti_header(path: Path) -> None:
    try:
        image = nib.load(path, mmap=True)
    except Exception as exc:
        raise PipelineError("The generated NIfTI cannot be opened.") from exc
    if len(image.header.extensions) != 0:
        raise PipelineError("The generated NIfTI contains unexpected header extensions.")

    neutral_description = b"dcm2niix v1.0.20260724; neutral pilot"
    with path.open("r+b") as stream:
        stream.seek(4)
        stream.write(b"\x00" * 10)
        stream.seek(14)
        stream.write(b"\x00" * 18)
        stream.seek(148)
        stream.write(neutral_description.ljust(80, b"\x00"))
        stream.seek(228)
        stream.write(b"\x00" * 24)
        stream.seek(328)
        stream.write(b"\x00" * 16)
        stream.flush()

    reloaded = nib.load(path, mmap=True)
    description = bytes(reloaded.header["descrip"]).rstrip(b"\x00")
    legacy_data_type = bytes(reloaded.header["data_type"]).rstrip(b"\x00")
    legacy_database_name = bytes(reloaded.header["db_name"]).rstrip(b"\x00")
    auxiliary = bytes(reloaded.header["aux_file"]).rstrip(b"\x00")
    intent = bytes(reloaded.header["intent_name"]).rstrip(b"\x00")
    if (
        description != neutral_description
        or legacy_data_type
        or legacy_database_name
        or auxiliary
        or intent
        or reloaded.header.extensions
    ):
        raise PipelineError("NIfTI header privacy scrubbing did not verify.")


def verify_sanitized_sidecar(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return ["sanitized conversion sidecar is unreadable"]
    unexpected = set(value) - SAFE_SIDECAR_KEYS - {"schema_version"}
    if unexpected:
        issues.append("sanitized conversion sidecar has unexpected fields")
    if UID_PATTERN.search(json.dumps(value, sort_keys=True)):
        issues.append("sanitized conversion sidecar contains a UID-like value")
    issues.extend(_sidecar_semantic_issues(value))
    return issues
