from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .util import PipelineError, repo_root, sha256_file


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise PipelineError("The pipeline Git revision cannot be verified.")
    return completed.stdout.strip()


def current_code_provenance(*, require_clean: bool = True) -> dict[str, Any]:
    root = repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if require_clean and dirty:
        raise PipelineError("The pipeline repository must be committed and clean before conversion or QC.")
    dependencies = {
        name: importlib.metadata.version(name)
        for name in (
            "fastapi",
            "nibabel",
            "numpy",
            "pydicom",
            "pylibjpeg",
            "pylibjpeg-libjpeg",
            "pylibjpeg-openjpeg",
            "pyjpegls",
            "python-multipart",
            "pytest",
            "setuptools",
            "uvicorn",
        )
    }
    # uv can install the locked environment without installing pip itself.
    # Missing installer metadata must not block an otherwise verified converter;
    # processing-library versions above remain mandatory and are never invented.
    try:
        dependencies["pip"] = importlib.metadata.version("pip")
    except importlib.metadata.PackageNotFoundError:
        dependencies["pip"] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "requirements_lock_sha256": sha256_file(root / "requirements.lock"),
        "pyproject_sha256": sha256_file(root / "pyproject.toml"),
        "tools_lock_sha256": sha256_file(root / "tools.lock.json"),
        "web_package_lock_sha256": sha256_file(root / "web" / "package-lock.json"),
        "executable_bits": 64 if sys.maxsize > 2**32 else 32,
    }
