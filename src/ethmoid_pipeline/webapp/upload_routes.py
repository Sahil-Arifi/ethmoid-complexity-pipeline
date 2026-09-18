"""Upload bytes never become client-supplied filesystem paths."""
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from .service import WebAppError
from .upload_service import MAX_CHUNK, UploadService


class UploadFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1,max_length=1024)
    size: int = Field(ge=0,strict=True)


class UploadManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    files: list[UploadFile] = Field(min_length=1,max_length=50000)


def upload_router(service: UploadService):
    router=APIRouter(prefix='/api/uploads',tags=['local-uploads'])

    @router.post('',status_code=201)
    def create(body:UploadManifest):
        return service.create([f.model_dump() for f in body.files])

    @router.put('/{upload_id}/files/{index}')
    async def put(upload_id:str,index:int,request:Request,offset:int=0):
        body=bytearray()
        async for chunk in request.stream():
            if len(body)+len(chunk)>MAX_CHUNK:
                raise WebAppError('Upload chunk exceeds 8 MiB.',status_code=413)
            body.extend(chunk)
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(service.put,upload_id,index,offset,bytes(body))

    @router.post('/{upload_id}/finalize',status_code=202)
    def finalize(upload_id:str):
        return service.finalize(upload_id)

    @router.get('/{upload_id}')
    def status(upload_id:str):
        return service.status(upload_id)

    return router
