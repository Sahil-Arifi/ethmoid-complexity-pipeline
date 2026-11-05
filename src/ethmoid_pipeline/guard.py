from __future__ import annotations

import gzip
import io
import re
import subprocess
import sys
import warnings
from pathlib import Path, PurePosixPath

import pydicom


MAX_TRACKED_BYTES = 5 * 1024 * 1024
EXACT_ALLOWED = {
    ".github/workflows/verify.yml",
    ".gitattributes",
    ".gitignore",
    ".hooks/pre-commit",
    "LICENSE.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "requirements.lock",
    "requirements-build.in",
    "inference-requirements.in",
    "inference-requirements.lock",
    "counting-tests.in",
    "counting-tests.lock",
    "boundary-partition-conda.lock.json",
    "boundary-partition-source.lock.json",
    "boundary-partition-requirements.in",
    "boundary-partition-win-64.explicit.txt",
    "src/ethmoid_pipeline/counting/protocol.json",
    "src/ethmoid_pipeline/counting/protocol.v1.json",
    "src/ethmoid_pipeline/counting/protocol.v3.json",
    "src/ethmoid_pipeline/counting/protocol.research.v4.json",
    "src/ethmoid_pipeline/counting/protocol.research.v5.json",
    "src/ethmoid_pipeline/webapp/protocol.v1.json",
    "src/ethmoid_pipeline/webapp/protocol.v2.draft.json",
    "tools.lock.json",
    "web/.gitignore",
}
PREFIX_SUFFIX_ALLOWED = {
    "src/": {".py"},
    "tests/": {".py"},
    "scripts/": {".py", ".ps1"},
    "docs/": {".md"},
    "web/": {".css", ".html", ".js", ".json", ".ts", ".tsx"},
}
FORBIDDEN_SUFFIXES = {
    ".dcm",
    ".dicom",
    ".nii",
    ".nrrd",
    ".mha",
    ".mhd",
    ".mrb",
    ".mrml",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".csv",
    ".xls",
    ".xlsx",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".log",
}
UID_TEXT_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3,}[0-9]+(?![0-9])")
_BACKSLASH = chr(92)
ABSOLUTE_WINDOWS_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9:])(?:[A-Z]:["
    + re.escape(_BACKSLASH)
    + r"/]|"
    + re.escape(_BACKSLASH * 2)
    + r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}"
    + re.escape(_BACKSLASH)
    + r"|//[A-Za-z0-9][A-Za-z0-9._-]{0,62}/"
    + r"|"
    + re.escape(_BACKSLASH * 2)
    + r"[.?]"
    + re.escape(_BACKSLASH)
    + r")[^\r\n\x00]+"
)
SAFE_DOTTED_VALUES = {"9.9.9.9.9", "127.0.0.1"}


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise RuntimeError("Git index inspection failed.")
    return completed.stdout


def _allowed_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in EXACT_ALLOWED:
        return True
    pure = PurePosixPath(normalized)
    lowered = normalized.casefold()
    if pure.name.casefold() == "dicomdir":
        return False
    if any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return False
    return any(
        normalized.startswith(prefix) and pure.suffix.casefold() in suffixes
        for prefix, suffixes in PREFIX_SUFFIX_ALLOWED.items()
    )


def _medical_magic(blob: bytes) -> str | None:
    if len(blob) >= 132 and blob[128:132] == b"DICM":
        return "DICOM content"
    if len(blob) >= 348 and blob[344:348] in {b"n+1\x00", b"ni1\x00"}:
        return "NIfTI content"
    if len(blob) >= 12 and blob[4:12] in {b"n+2\x00\r\n\x1a\n", b"ni2\x00\r\n\x1a\n"}:
        return "NIfTI-2 content"
    if blob.startswith(b"\x1f\x8b"):
        try:
            header = gzip.decompress(blob[: 1024 * 1024])[:352]
        except (OSError, EOFError):
            return "compressed binary content"
        if len(header) >= 348 and header[344:348] in {b"n+1\x00", b"ni1\x00"}:
            return "compressed NIfTI content"
        return "compressed binary content"
    signatures = (
        (b"SQLite format 3\x00", "SQLite database"),
        (b"PK\x03\x04", "archive content"),
        (b"\x89PNG\r\n\x1a\n", "image content"),
        (b"\xff\xd8\xff", "image content"),
        (b"II*\x00", "image content"),
        (b"MM\x00*", "image content"),
        (b"version https://git-lfs.github.com/spec/v1", "Git LFS pointer"),
    )
    for signature, reason in signatures:
        if blob.startswith(signature):
            return reason
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = pydicom.dcmread(
                io.BytesIO(blob),
                force=True,
                stop_before_pixels=True,
                specific_tags=[
                    "SOPClassUID",
                    "SOPInstanceUID",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "Modality",
                ],
            )
        if hasattr(dataset, "SOPClassUID") and (
            hasattr(dataset, "SOPInstanceUID")
            or hasattr(dataset, "StudyInstanceUID")
            or hasattr(dataset, "SeriesInstanceUID")
            or hasattr(dataset, "Modality")
        ):
            return "DICOM content without a Part-10 preamble"
    except Exception:
        pass
    return None


def _blocked_text_content(blob: bytes) -> str | None:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if ABSOLUTE_WINDOWS_PATH_PATTERN.search(text):
        return "an absolute Windows/UNC path"
    uids = {match.group(0) for match in UID_TEXT_PATTERN.finditer(text)}
    if uids - SAFE_DOTTED_VALUES:
        return "a non-allowlisted UID-like value"
    return None


def scan_staged(repo: Path, *, all_tracked: bool = False) -> list[str]:
    repo = repo.resolve(strict=True)
    if _git(repo, "ls-files", "-u", "-z"):
        return ["the index contains unresolved conflict stages"]
    raw_names = (
        _git(repo, "ls-files", "-z")
        if all_tracked
        else _git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    )
    paths = [item.decode("utf-8", errors="strict") for item in raw_names.split(b"\x00") if item]
    failures: list[str] = []
    for path in paths:
        if not _allowed_path(path):
            failures.append("a staged path is outside the source/documentation allowlist")
            continue
        index_entry = _git(repo, "ls-files", "-s", "-z", "--", path)
        mode = index_entry.split(maxsplit=1)[0] if index_entry else b""
        if mode == b"120000":
            failures.append("a staged symbolic link is not allowed")
            continue
        blob = _git(repo, "show", f":{path}")
        if len(blob) > MAX_TRACKED_BYTES:
            failures.append("a staged blob exceeds the five-megabyte source limit")
        magic = _medical_magic(blob)
        if magic:
            failures.append(f"a staged blob contains blocked {magic}")
        blocked_text = _blocked_text_content(blob)
        if blocked_text:
            failures.append(f"a staged blob contains blocked {blocked_text}")
    return list(dict.fromkeys(failures))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Git index blobs for medical data.")
    parser.add_argument("--all-tracked", action="store_true", help="Inspect the entire index, including committed files.")
    args = parser.parse_args()
    try:
        failures = scan_staged(Path.cwd(), all_tracked=args.all_tracked)
    except Exception:
        print("PHI guard failed closed because the Git index could not be inspected.", file=sys.stderr)
        return 2
    if failures:
        print("PHI guard blocked the staged commit:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        print("No staged path or matched content is displayed.", file=sys.stderr)
        return 1
    print("PHI guard: staged blobs are within the source/documentation allowlist.")
    return 0
