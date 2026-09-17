from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from ..util import repo_root
from .protocol import load_protocol
from .service import AppService, WebAppError
from .counting_service import CountingService
from .counting_routes import counting_router


MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024


class ReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["Draft", "Ready for review", "Needs changes", "Approved"] | None = None
    qc_checklist: dict[str, bool] | None = None
    reviewer_notes: str | None = None


class ViewerStateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    crosshair: tuple[float, float, float] | None = None
    pan_zoom: tuple[float, float, float, float] | None = None
    view_mode: Literal["multiplanar", "render"] | None = None
    selected_label: int | None = None
    brush_size: int | None = None
    overlay_opacity: float | None = None
    active_tool: Literal["navigate", "window", "draw", "erase", "fill"] | None = None
    window_center: float | None = None
    window_width: float | None = None
    render_azimuth: float | None = None
    render_elevation: float | None = None
    volume_scale: float | None = None


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise WebAppError("An uploaded file exceeds the 4 GiB limit.", status_code=413)
                stream.write(chunk)
    finally:
        await upload.close()
    if size == 0:
        raise WebAppError("An uploaded file is empty.", status_code=422)


def _nifti_suffix(filename: str | None) -> str:
    lowered = (filename or "").lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    if lowered.endswith(".nii"):
        return ".nii"
    raise WebAppError("Only .nii and .nii.gz files can be imported.", status_code=422)


def create_app(*, work_root: Path | None = None, dcm2niix: Path | None = None) -> FastAPI:
    service = AppService(work_root=work_root, dcm2niix=dcm2niix)
    counting = CountingService(service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        counting.close()

    app = FastAPI(
        title="Ethmoid Complexity App",
        version="0.1.0",
        description="Local function-first CT review and provisional annotation API.",
        lifespan=lifespan,
    )
    app.state.ethmoid_service = service
    app.state.counting_service = counting
    app.include_router(counting_router(counting))

    @app.exception_handler(WebAppError)
    async def webapp_error_handler(_request: Request, exc: WebAppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/protocol")
    def protocol() -> dict[str, object]:
        return load_protocol()

    @app.get("/api/cases")
    def list_cases() -> dict[str, object]:
        return {"cases": service.list_cases()}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, object]:
        return service.get_case(case_id)

    @app.get("/api/cases/{case_id}/volume")
    def get_volume(case_id: str) -> FileResponse:
        path = service.volume_path(case_id)
        media_type = "application/gzip" if path.name.endswith(".gz") else "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="inline",
        )

    @app.patch("/api/cases/{case_id}/viewer-state")
    def update_viewer_state(
        case_id: str, update: ViewerStateUpdate
    ) -> dict[str, object]:
        return service.update_viewer_state(
            case_id, update.model_dump(exclude_none=True)
        )

    @app.post("/api/imports/nifti")
    async def import_nifti(
        file: Annotated[UploadFile, File(...)],
        case_label: Annotated[str | None, Form()] = None,
        qc_report: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, object]:
        incoming = Path(tempfile.mkdtemp(prefix="nifti-", dir=service.root / "incoming"))
        try:
            source = incoming / f"source{_nifti_suffix(file.filename)}"
            await _save_upload(file, source)
            report_path: Path | None = None
            if qc_report is not None:
                report_path = incoming / "qc_report.json"
                await _save_upload(qc_report, report_path)
            case = await run_in_threadpool(
                service.import_nifti,
                source,
                case_label=case_label,
                qc_report=report_path,
            )
            return {"case": case}
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

    @app.post("/api/imports/dicom")
    async def import_dicom(
        files: Annotated[list[UploadFile], File(...)],
        case_label: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        incoming = Path(tempfile.mkdtemp(prefix="dicom-", dir=service.root / "incoming"))
        try:
            staged: list[Path] = []
            for index, upload in enumerate(files):
                destination = incoming / f"upload_{index:06d}.dcm"
                await _save_upload(upload, destination)
                staged.append(destination)
            return await run_in_threadpool(
                service.import_dicom, staged, case_label=case_label
            )
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

    @app.get("/api/imports/{import_id}/series")
    def get_import_series(import_id: str) -> dict[str, object]:
        result = service.get_import(import_id)
        return {"series": result["series"]}

    @app.delete("/api/imports/{import_id}", status_code=204)
    def delete_import(import_id: str) -> Response:
        service.delete_import(import_id)
        return Response(status_code=204)

    @app.post("/api/imports/{import_id}/series/{series_label}/convert")
    def convert_series(import_id: str, series_label: str) -> dict[str, object]:
        return {"case": service.convert_dicom(import_id, series_label)}

    @app.put("/api/cases/{case_id}/annotations")
    async def save_annotation(
        case_id: str,
        mask: Annotated[UploadFile, File(...)],
        viewer_state: Annotated[str, Form()] = "{}",
    ) -> dict[str, object]:
        try:
            parsed_state = json.loads(viewer_state)
        except json.JSONDecodeError as exc:
            raise WebAppError("Viewer state is not valid JSON.", status_code=422) from exc
        incoming = Path(tempfile.mkdtemp(prefix="mask-", dir=service.root / "incoming"))
        try:
            path = incoming / f"annotation{_nifti_suffix(mask.filename)}"
            await _save_upload(mask, path)
            result = await run_in_threadpool(
                service.save_annotation, case_id, path, parsed_state
            )
            result["voxel_counts"] = {
                str(item["value"]): item["voxel_count"]
                for item in result["metrics"]["labels"]
            }
            return result
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

    @app.get("/api/cases/{case_id}/annotations/latest.nii.gz")
    @app.get("/api/cases/{case_id}/annotations/latest")
    def latest_annotation(case_id: str) -> FileResponse:
        path = service.latest_annotation_path(case_id)
        return FileResponse(
            path,
            media_type="application/gzip",
            filename="annotation.nii.gz",
            content_disposition_type="inline",
        )

    @app.patch("/api/cases/{case_id}/review")
    def update_review(case_id: str, update: ReviewUpdate) -> dict[str, object]:
        return service.update_review(
            case_id,
            status=update.status,
            qc_checklist=update.qc_checklist,
            reviewer_notes=update.reviewer_notes,
        )

    @app.post("/api/cases/{case_id}/finalize")
    def finalize_case(case_id: str) -> dict[str, object]:
        return service.finalize_case(case_id)

    @app.get("/api/cases/{case_id}/export")
    def export_case(case_id: str) -> FileResponse:
        path = service.export_case(case_id)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.delete("/api/cases/{case_id}", status_code=204)
    def delete_case(case_id: str) -> Response:
        service.delete_draft(case_id)
        return Response(status_code=204)

    distribution = repo_root() / "web" / "dist"
    assets = distribution / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_index() -> Response:
        index = distribution / "index.html"
        if index.is_file():
            return FileResponse(index, media_type="text/html")
        return HTMLResponse(
            "<h1>Ethmoid application frontend is not built</h1>"
            "<p>Run the documented frontend build, then restart ethmoid-app.</p>",
            status_code=503,
        )

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend_fallback(frontend_path: str) -> Response:
        # API routes are declared above and win FastAPI's ordered route match.
        if frontend_path == "api" or frontend_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "API route not found."})
        candidate = (distribution / frontend_path).resolve()
        if distribution.is_dir():
            try:
                candidate.relative_to(distribution.resolve())
            except ValueError:
                candidate = distribution / "__invalid__"
            if candidate.is_file():
                return FileResponse(candidate)
            index = distribution / "index.html"
            if index.is_file():
                return FileResponse(index, media_type="text/html")
        return HTMLResponse(
            "<h1>Ethmoid application frontend is not built</h1>", status_code=503
        )

    return app
