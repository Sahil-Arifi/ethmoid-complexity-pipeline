"""Study navigation exposes neutral summaries, never local source identities."""
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from .catalog import CatalogService


class StudySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_series_id: str = Field(min_length=1, max_length=64)


def catalog_router(service: CatalogService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["local-study-catalog"])

    @router.get("/studies")
    def studies():
        return service.list_studies()

    @router.get("/studies/{study_id}/series")
    def series(study_id: str):
        return {"series": service.study(study_id)["series"]}

    @router.patch("/studies/{study_id}")
    def select(study_id: str, value: StudySelection):
        return {"study": service.select(study_id, value.selected_series_id)}

    @router.post("/studies/{study_id}/series/{series_id}/prepare")
    def prepare(study_id: str, series_id: str):
        return {"series": service.prepare(study_id, series_id)}

    @router.get("/catalog/status")
    def status():
        return service.status()

    return router
