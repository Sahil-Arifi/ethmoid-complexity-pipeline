import io
import re
import shutil
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pydicom
import pytest

from ethmoid_pipeline.util import default_dcm2niix
from ethmoid_pipeline.webapp.catalog import CatalogService
from ethmoid_pipeline.webapp.catalog_queue import PreparationQueue
from ethmoid_pipeline.webapp.service import AppService, WebAppError
from ethmoid_pipeline.webapp.upload_service import UploadService
from ethmoid_pipeline.webapp.upload_routes import upload_router
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from tests.synthetic import write_ct_series
from tests.test_catalog import catalog


@pytest.fixture
def uploads(tmp_path):
    app = AppService(work_root=tmp_path/'app', dcm2niix=default_dcm2niix())
    catalog = CatalogService(app)
    catalog._queue = PreparationQueue(catalog)
    calls=[]
    def create(case_id, idempotency_key):
        assert re.fullmatch(r'[A-Za-z0-9_-]{1,100}',idempotency_key)
        if not any(c[1]==idempotency_key for c in calls):
            calls.append((case_id,idempotency_key))
        return {'run_id':'synthetic-'+str(next(i for i,c in enumerate(calls) if c[1]==idempotency_key))}
    service = UploadService(app,catalog,create)
    return service,calls


def submit(service, files):
    session=service.create([{'name':name,'size':len(data)} for name,data in files])
    identity=session['upload_id']
    for index,(_,data) in enumerate(files):
        service.put(identity,index,0,data)
    service.finalize(identity)
    service.process(identity)
    return identity


def payload(paths):
    return [(p.name,p.read_bytes()) for p in paths]


def archive(entries):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in entries:
            z.writestr(name,data)
    return stream.getvalue()


def test_resume_offset_replay_and_crash_tail(uploads):
    service,calls=uploads
    identity=service.create([{'name':'synthetic.dcm','size':6}])['upload_id']
    service.put(identity,0,0,b'abc')
    path=service._directory(identity)/'files/000000.source'
    with path.open('ab') as out:
        out.write(b'uncommitted')
    resumed=UploadService(service.app,service.catalog,service.create_count_run)
    assert resumed.status(identity)['files'][0]['received_bytes']==3
    assert resumed.put(identity,0,0,b'abc')['received_bytes']==3
    with pytest.raises(WebAppError,match='differs'):
        resumed.put(identity,0,0,b'xyz')
    with pytest.raises(WebAppError,match='offset'):
        resumed.put(identity,0,4,b'ef')
    resumed.put(identity,0,3,b'def')
    assert path.read_bytes()==b'abcdef'
    resumed.finalize(identity)
    assert resumed.finalize(identity)['status']=='queued'
    assert calls==[]


def test_empty_ancillary_and_corrupt_files_accounted(uploads):
    service,_=uploads
    identity=submit(service,[('readme.txt',b'notes'),('broken.dcm',b'bad'),('empty',b'')])
    result=service.status(identity)
    assert result['status']=='completed' and result['reason_code']=='NO_DICOM'
    assert [f['status'] for f in result['files']]==['ancillary','unreadable','ancillary']
    assert len(result['issues'])==3
    assert all((service._directory(identity)/'files'/f'{i:06d}.source').is_file() for i in range(3))


def test_every_study_series_and_duplicate_image_retained(uploads,tmp_path):
    service,calls=uploads
    a=write_ct_series(tmp_path/'source-a')
    ds=pydicom.dcmread(a[0],stop_before_pixels=True)
    b=write_ct_series(tmp_path/'source-b',study_uid=ds.StudyInstanceUID,image_type=['ORIGINAL','PRIMARY','LOCALIZER'])
    c=write_ct_series(tmp_path/'source-c')
    identity=submit(service,payload(a+b+c)+[('duplicate.dcm',a[0].read_bytes()),('notes.txt',b'text')])
    result=service.status(identity)
    assert result['status']=='preparing'
    assert len(result['studies'])==2 and sum(s['series_count'] for s in result['studies'])==3
    assert service.catalog.status()['total_series']==3 and calls==[]
    assert result['files'][-2]['status']=='duplicate'
    with service.store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM upload_items').fetchone()[0]==20
        assert db.execute("SELECT COUNT(*) FROM preparation_jobs WHERE status='queued'").fetchone()[0]==2


def test_repeat_upload_global_identity_and_conflict_preserves_work(uploads,tmp_path):
    service,calls=uploads
    files=payload(write_ct_series(tmp_path/'source'))
    first=submit(service,files)
    second=submit(service,list(reversed(files)))
    assert service.catalog.status()['total_studies']==1
    assert service.catalog.status()['total_series']==1
    assert service.status(second)['status']=='completed'
    assert service.status(second)['studies'][0]['analysis_status']=='duplicate'
    changed=files.copy(); changed[0]=(changed[0][0],changed[0][1]+b'changed')
    conflict=submit(service,changed)
    assert service.status(conflict)['reason_code']=='SOURCE_CONFLICT'
    assert all(f['status']=='conflict' for f in service.status(conflict)['files'])
    assert service.catalog.status()['total_series']==1
    assert service.status(first)['studies'][0]['series_count']==1 and calls==[]


def test_added_series_same_study_uses_its_own_source(uploads,tmp_path):
    service,_=uploads
    a=write_ct_series(tmp_path/'a')
    uid=str(pydicom.dcmread(a[0]).StudyInstanceUID)
    b=write_ct_series(tmp_path/'b',study_uid=uid)
    first=submit(service,payload(a)); second=submit(service,payload(b))
    catalog=service.catalog.list_studies()
    assert len(catalog['studies'])==1 and catalog['studies'][0]['series_count']==2
    roots=[]
    for s in catalog['studies'][0]['series']:
        row=service.catalog.series_row(s['series_id'])
        roots.append(service.catalog.source_root(row))
        assert len(service.catalog.files(row))==6
    assert set(roots)=={service._directory(first),service._directory(second)}


def test_default_source_resync_preserves_uploaded_series_in_existing_study(catalog,tmp_path):
    service,native,_=catalog
    service.sync()
    service._queue=PreparationQueue(service)
    uploads=UploadService(service.app,service,lambda *a,**k:pytest.fail('Unverified series cannot count'))
    uid=str(pydicom.dcmread(native[0]).StudyInstanceUID)
    extra=write_ct_series(tmp_path/'extra',study_uid=uid)
    identity=submit(uploads,payload(extra))
    assert service.status()['total_series']==4
    service.sync(); service.sync()
    assert service.status()['total_series']==4
    study=service.study(uploads.status(identity)['studies'][0]['study_id'])
    assert study['series_count']==3
    assert all(s['reason'] is None or 'scout' in s['reason'] for s in study['series'])


def test_additive_profile_fields_preserve_legacy_catalog_signature(catalog):
    import json
    from ethmoid_pipeline.util import sha256_canonical_json
    from ethmoid_pipeline.webapp.catalog import safe_descriptor
    service,_,_=catalog
    service.sync()
    with service.store.connect() as db:
        rows=db.execute('SELECT * FROM catalog_series').fetchall()
    summaries=json.loads((service.config.work_root/'inventory/inventory.redacted.json').read_text())['series']
    for row in rows:
        with service.store.connect() as db:
            study=db.execute('SELECT study_label FROM catalog_studies WHERE id=?',(row['study_id'],)).fetchone()[0]
        summary=next(s for s in summaries if s['case']==study and s['series']==row['series_label'])
        metadata=safe_descriptor(summary)
        metadata.pop('reconstruction_kind'); metadata.pop('kernel_class')
        stats=[]
        for name in json.loads(row['files_json']):
            value=(service.config.source_root/name).stat()
            stats.append((name,value.st_size,value.st_mtime_ns))
        legacy=sha256_canonical_json({'files':stats,'metadata':metadata})
        assert row['inventory_signature']==legacy
    service.sync()
    with service.store.connect() as db:
        assert [dict(r) for r in db.execute('SELECT * FROM catalog_series')]==[dict(r) for r in rows]


@pytest.mark.parametrize('name',['../outside.dcm','/absolute.dcm','C:' + '/outside.dcm','a/../../b','a\\..\\b'])
def test_archive_path_traversal_is_terminal_and_original_retained(uploads,name):
    service,_=uploads
    data=archive([(name,b'bad')])
    identity=submit(service,[('study.zip',data)])
    result=service.status(identity)
    assert result['status']=='failed' and result['reason_code']=='ARCHIVE_UNSAFE'
    assert (service._directory(identity)/'files/000000.source').read_bytes()==data
    assert service.finalize(identity)['status']=='failed'


def test_archive_link_and_malformed_and_expansion_quota(uploads):
    service,_=uploads
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        info=zipfile.ZipInfo('link'); info.create_system=3
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(info,'target')
    link=submit(service,[('link.zip',stream.getvalue())])
    assert service.status(link)['reason_code']=='ARCHIVE_UNSAFE'
    bad=submit(service,[('bad.zip',b'invalid')])
    assert service.status(bad)['reason_code']=='ARCHIVE_INVALID'
    service.max_expanded_bytes=1024
    quota=submit(service,[('large.zip',archive([('big',b'0'*2048)]))])
    assert service.status(quota)['reason_code']=='UPLOAD_QUOTA'


def test_disk_full_write_keeps_committed_offset(uploads,monkeypatch):
    service,_=uploads
    identity=service.create([{'name':'x','size':6}])['upload_id']
    service.put(identity,0,0,b'abc')
    monkeypatch.setattr(shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(1,1,0))
    with pytest.raises(WebAppError) as error:
        service.put(identity,0,3,b'def')
    assert error.value.status_code==507
    assert service.status(identity)['files'][0]['received_bytes']==3


def test_archive_mixed_studies_and_restart_indexing_idempotent(uploads,tmp_path):
    service,calls=uploads
    paths=write_ct_series(tmp_path/'a')+write_ct_series(tmp_path/'b')
    data=archive([(f'{i}/{p.name}',p.read_bytes()) for i,p in enumerate(paths)])
    identity=submit(service,[('studies.zip',data)])
    service.process(identity)  # Simulate crash after catalog commit but before status polling.
    assert service.catalog.status()['total_studies']==2 and service.catalog.status()['total_series']==2
    assert len(service.status(identity)['studies'])==2 and calls==[]


@pytest.mark.integration
def test_verified_upload_schedules_one_recommended_run_once(uploads,tmp_path):
    if not default_dcm2niix().is_file():
        pytest.skip('pinned converter unavailable')
    service,calls=uploads
    paths=write_ct_series(tmp_path/'a')
    uid=str(pydicom.dcmread(paths[0]).StudyInstanceUID)
    other=write_ct_series(tmp_path/'b',study_uid=uid)
    identity=submit(service,payload(paths+other))
    queue=service.catalog._queue
    while (job:=queue._claim()) is not None:
        queue.process(job)
    assert service.catalog.status()['ready']==2
    service.schedule_ready(); service.schedule_ready()
    resumed=UploadService(service.app,service.catalog,service.create_count_run)
    resumed.schedule_ready()
    assert len(calls)==1
    result=resumed.status(identity)
    assert result['status']=='completed'
    assert result['studies'][0]['count_run_id']=='synthetic-0'
    service.app.assert_catalog_countable(calls[0][0])
    repeat=submit(resumed,payload(paths+other))
    assert resumed.status(repeat)['studies'][0]['analysis_status']=='duplicate'
    assert len(calls)==1


def test_concurrent_chunk_retries_and_finalize_publish_once(uploads):
    service,_=uploads
    identity=service.create([{'name':'notes.txt','size':3}])['upload_id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:service.put(identity,0,0,b'abc'),range(4)))
        finalized=list(pool.map(lambda _:service.finalize(identity),range(4)))
        list(pool.map(lambda _:service.process(identity),range(4)))
    assert all(r['received_bytes']==3 for r in results)
    assert all(r['status']=='queued' for r in finalized)
    with service.store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM upload_items').fetchone()[0]==1
    assert service.status(identity)['status']=='completed'


def test_restart_recovers_indexing_and_worker_lock_prevents_second_owner(uploads):
    service,_=uploads
    identity=service.create([{'name':'notes.txt','size':3}])['upload_id']
    service.put(identity,0,0,b'abc');service.finalize(identity)
    with service.store.connect() as db:
        db.execute("UPDATE upload_sessions SET status='indexing' WHERE id=?",(identity,))
    service.start()
    other=UploadService(service.app,service.catalog,service.create_count_run)
    try:
        other.start()
        assert other._thread is None
        deadline=time.monotonic()+5
        while service.status(identity)['status']!='completed' and time.monotonic()<deadline:
            time.sleep(0.01)
        assert service.status(identity)['status']=='completed'
    finally:
        other.close();service.close()


def test_same_series_uid_in_two_studies_is_an_explicit_conflict(uploads,tmp_path):
    service,_=uploads
    a=write_ct_series(tmp_path/'a')
    uid=str(pydicom.dcmread(a[0]).SeriesInstanceUID)
    b=write_ct_series(tmp_path/'b',series_uid=uid)
    identity=submit(service,payload(a+b))
    assert service.status(identity)['reason_code']=='SOURCE_CONFLICT'
    assert service.catalog.status()['total_series']==0
    assert len(service.status(identity)['issues'])==12


def test_upload_http_resumability_streaming_and_validation(uploads):
    service,_=uploads
    app=FastAPI()
    app.include_router(upload_router(service))
    @app.exception_handler(WebAppError)
    async def handle(request,error):
        return JSONResponse(status_code=error.status_code,content={'detail':str(error)})
    with TestClient(app) as client:
        assert client.post('/api/uploads',json={'files':[{'name':'x','size':-1}]}).status_code==422
        created=client.post('/api/uploads',json={'files':[{'name':'notes.txt','size':6}]})
        assert created.status_code==201
        identity=created.json()['upload_id']
        url=f'/api/uploads/{identity}'
        assert client.put(url+'/files/0?offset=0',content=b'abc').status_code==200
        assert client.post(url+'/finalize').status_code==409
        assert client.get(url).json()['files'][0]['received_bytes']==3
        assert client.put(url+'/files/0?offset=0',content=b'xyz').status_code==409
        assert client.put(url+'/files/0?offset=3',content=b'def').status_code==200
        assert client.post(url+'/finalize').status_code==202
        service.process(identity)
        assert client.get(url).json()['status']=='completed'
        assert client.get('/api/uploads/missing').status_code==404


@pytest.mark.integration
def test_recover_count_creation_before_intent_commit(uploads,tmp_path):
    if not default_dcm2niix().is_file():
        pytest.skip('pinned converter unavailable')
    service,calls=uploads
    identity=submit(service,payload(write_ct_series(tmp_path/'source')))
    queue=service.catalog._queue
    while (job:=queue._claim()) is not None:
        queue.process(job)
    original=service.create_count_run
    def crash_after_create(*args,**kwargs):
        original(*args,**kwargs)
        raise KeyboardInterrupt('synthetic crash before upload intent commit')
    service.create_count_run=crash_after_create
    with pytest.raises(KeyboardInterrupt):
        service.schedule_ready()
    assert len(calls)==1
    resumed=UploadService(service.app,service.catalog,original)
    resumed.schedule_ready()
    assert len(calls)==1
    assert resumed.status(identity)['status']=='completed'


@pytest.mark.integration
def test_count_scheduling_error_has_terminal_actionable_outcome(uploads,tmp_path):
    if not default_dcm2niix().is_file():
        pytest.skip('pinned converter unavailable')
    service,_=uploads
    identity=submit(service,payload(write_ct_series(tmp_path/'source')))
    queue=service.catalog._queue
    while (job:=queue._claim()) is not None:
        queue.process(job)
    service.create_count_run=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('synthetic'))
    service.schedule_ready()
    result=service.status(identity)
    assert result['status']=='completed'
    assert result['studies'][0]['analysis_status']=='needs_run'
    assert result['studies'][0]['reason_code']=='COUNT_SCHEDULING'


def test_unavailable_default_collection_does_not_disable_upload_worker(tmp_path,monkeypatch):
    import ethmoid_pipeline.webapp.catalog_config as config_module
    import ethmoid_pipeline.webapp.catalog as catalog_module
    def unavailable(*args):
        raise ValueError('synthetic unavailable source')
    monkeypatch.setattr(config_module,'load_local_config',unavailable)
    monkeypatch.setattr(catalog_module,'load_local_config',unavailable)
    app=AppService(work_root=tmp_path/'app',dcm2niix=default_dcm2niix())
    catalog=CatalogService(app)
    service=UploadService(app,catalog,lambda *a,**k:None)
    try:
        catalog._startup()
        assert catalog.status()['status']=='error'
        assert catalog._queue is not None and catalog._queue._thread.is_alive()
        identity=submit(service,[('notes.txt',b'local upload still accepted')])
        assert service.status(identity)['status']=='completed'
    finally:
        catalog.close()


def test_transient_upload_database_failure_recovers_without_lost_jobs(uploads,monkeypatch):
    from contextlib import contextmanager
    import sqlite3
    service,_=uploads
    identity=service.create([{'name':'notes.txt','size':3}])['upload_id']
    service.put(identity,0,0,b'abc'); service.finalize(identity)
    original=service.store.connect
    failures=[]
    @contextmanager
    def flaky():
        if threading.current_thread().name=='ethmoid-upload' and not failures:
            failures.append(True)
            raise sqlite3.OperationalError('synthetic temporary lock')
        with original() as db:
            yield db
    monkeypatch.setattr(service.store,'connect',flaky)
    try:
        service.start()
        deadline=time.monotonic()+5
        while service.status(identity)['status']!='completed' and time.monotonic()<deadline:
            time.sleep(0.01)
        assert failures and service._thread.is_alive()
        assert service.status(identity)['status']=='completed'
    finally:
        service.close()
