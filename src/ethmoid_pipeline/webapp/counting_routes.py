"""Narrow API surface for local exploratory counting and review corrections."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from .counting_service import CountingService
from .service import WebAppError


MAX_MASK_BYTES = 512 * 1024 * 1024


class CreateCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=100)


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    parent_revision: StrictInt = Field(ge=0)
    operation: dict[str, Any]


class AssistedRequest(CreateCountRequest):
    boundary_id: str = Field(min_length=1, max_length=64)


def _revision(value: str | None, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if value == "latest":
        return None
    if not value.isascii() or not value.isdecimal() or len(value) > 9:
        raise WebAppError("The revision selector is invalid.", status_code=422)
    return int(value)


def _cleanup_export(path: Path) -> None:
    # Each response owns this unique directory. Cleanup happens after streaming;
    # interrupted downloads can leave only a protected, case-scoped export.
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        pass


def counting_router(service: CountingService) -> APIRouter:
    router = APIRouter(prefix="/api/cases/{case_id}/count-runs", tags=["experimental-counting"])

    @router.post("", status_code=202)
    def create_run(case_id: str, payload: CreateCountRequest | None = None) -> dict[str, Any]:
        return service.create_run(case_id, payload.idempotency_key if payload else None)

    @router.get("")
    def list_runs(case_id: str) -> dict[str, Any]:
        return service.list_runs(case_id)

    @router.get("/{run_id}")
    def get_run(case_id: str, run_id: str) -> dict[str, Any]:
        return service.get_run(case_id, run_id)

    @router.get("/{run_id}/revisions/latest")
    def latest_revision(case_id: str, run_id: str) -> dict[str, Any]:
        return service.get_run(case_id, run_id)

    @router.post("/{run_id}/cancel")
    def cancel(case_id: str, run_id: str):
        return service.cancel(case_id, run_id)

    @router.post("/{run_id}/retry", status_code=202)
    def retry(case_id: str, run_id: str, payload: CreateCountRequest | None = None):
        return service.retry(case_id, run_id, payload.idempotency_key if payload else None)

    @router.post("/{run_id}/boundaries", status_code=201)
    async def save_boundary(case_id: str, run_id: str, request: Request):
        incoming = Path(tempfile.mkdtemp(prefix="boundary-", dir=service.root / "incoming"))
        try:
            async with request.form(max_files=1, max_fields=1, max_part_size=64 * 1024) as form:
                if set(form.keys()) - {"mask", "note"}:
                    raise WebAppError("Unexpected boundary fields.", status_code=422)
                upload = form.get("mask")
                note = form.get("note", "")
                if not isinstance(upload, UploadFile) or not isinstance(note, str):
                    raise WebAppError("A boundary mask and optional note are required.", status_code=422)
                filename = (upload.filename or '').lower()
                if not filename.endswith(('.nii', '.nii.gz')):
                    raise WebAppError("Boundary masks must be NIfTI files.", status_code=422)
                path = incoming / ('boundary.nii.gz' if filename.endswith('.gz') else 'boundary.nii')
                size = 0
                with path.open('wb') as stream:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_MASK_BYTES:
                            raise WebAppError("The boundary mask exceeds the upload limit.", status_code=413)
                        stream.write(chunk)
                return await run_in_threadpool(service.save_boundary, case_id, run_id, path, note)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

    @router.get("/{run_id}/boundaries/{boundary_id}")
    def boundary(case_id: str, run_id: str, boundary_id: str):
        value = service.get_boundary(case_id, boundary_id)
        if value['parent_run_id'] != run_id:
            raise WebAppError("Boundary revision not found for this run.", status_code=404)
        return value

    @router.get("/{run_id}/boundaries/{boundary_id}/mask")
    def boundary_mask(case_id: str, run_id: str, boundary_id: str):
        boundary(case_id, run_id, boundary_id)
        return FileResponse(service.boundary_path(case_id, boundary_id), media_type="application/gzip",
                            filename="boundary.nii.gz", content_disposition_type="inline")

    @router.post("/{run_id}/assisted-runs", status_code=202)
    def assisted(case_id: str, run_id: str, payload: AssistedRequest):
        return service.create_run(case_id, payload.idempotency_key, boundary_id=payload.boundary_id, parent_run_id=run_id)

    @router.get("/{run_id}/artifacts/{artifact}")
    def artifact(case_id: str, run_id: str, artifact: str, revision: str | None = None) -> FileResponse:
        path = service.artifact(case_id, run_id, artifact, _revision(revision, default=0))
        return FileResponse(path, media_type="application/gzip", filename=path.name, content_disposition_type="inline")

    @router.post("/{run_id}/revisions")
    async def revise(case_id: str, run_id: str, request: Request) -> dict[str, Any]:
        incoming: Path | None = None
        try:
            content_type = request.headers.get("content-type", "").lower()
            mask_path = None
            if content_type.startswith("application/json"):
                value = await request.json()
            elif content_type.startswith("multipart/form-data"):
                async with request.form(max_files=1, max_fields=1, max_part_size=64 * 1024) as form:
                    if set(form.keys()) - {"metadata", "mask"}:
                        raise ValueError("unexpected fields")
                    metadata = form.get("metadata")
                    if not isinstance(metadata, str):
                        raise ValueError("missing metadata")
                    value = json.loads(metadata)
                    upload = form.get("mask")
                    if upload is not None:
                        if not isinstance(upload, UploadFile):
                            raise ValueError("invalid mask")
                        incoming = Path(tempfile.mkdtemp(prefix="cell-edit-", dir=service.root / "incoming"))
                        suffix = ".nii.gz" if (upload.filename or "").lower().endswith(".nii.gz") else ".nii"
                        mask_path = incoming / f"selection{suffix}"
                        size = 0
                        with mask_path.open("wb") as stream:
                            while chunk := await upload.read(1024 * 1024):
                                size += len(chunk)
                                if size > MAX_MASK_BYTES:
                                    raise WebAppError("The selection mask exceeds the upload limit.", status_code=413)
                                stream.write(chunk)
                        if not size:
                            raise ValueError("empty mask")
            else:
                raise WebAppError("Use JSON or a multipart selection mask for corrections.", status_code=415)
            payload = CorrectionRequest.model_validate(value)
            return await run_in_threadpool(
                service.revise, case_id, run_id, payload.parent_revision, payload.operation, mask_path,
            )
        except (ValueError, TypeError, ValidationError) as exc:
            raise WebAppError("The cell correction request is invalid.", status_code=422) from exc
        finally:
            if incoming is not None:
                shutil.rmtree(incoming, ignore_errors=True)

    @router.get("/{run_id}/export")
    def export(case_id: str, run_id: str, revision: str | None = None) -> FileResponse:
        path = service.export(case_id, run_id, _revision(revision))
        return FileResponse(path, media_type="application/zip", filename=path.name,
                            background=BackgroundTask(_cleanup_export, path))

    return router
