"""Local web application for reviewing and annotating ethmoid CT volumes."""

from __future__ import annotations

from pathlib import Path


def create_app(*, work_root: Path | None = None, dcm2niix: Path | None = None):
    """Create the FastAPI application.

    FastAPI is imported lazily so the conversion-only command line tools remain
    usable when the optional web application dependencies are not installed.
    """

    from .app import create_app as _create_app

    return _create_app(work_root=work_root, dcm2niix=dcm2niix)


__all__ = ["create_app"]
