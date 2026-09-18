"""Real catalog conversion must work in app runtimes created without pip."""
import importlib.metadata
import json
from pathlib import Path

import pydicom
import pytest

from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.webapp.catalog import CatalogService
from ethmoid_pipeline.webapp.catalog_queue import PreparationQueue
from ethmoid_pipeline.webapp.service import AppService
from ethmoid_pipeline.webapp.upload_service import UploadService
from tests.synthetic import write_ct_series
from tests.test_dicom_frames import enhanced_fixture


@pytest.mark.integration
@pytest.mark.parametrize('enhanced',[False,True],ids=['classic','enhanced'])
def test_actual_upload_preparation_provenance_and_hu(tmp_path,enhanced):
    executable=default_dcm2niix()
    if not executable.is_file():
        pytest.skip('pinned dcm2niix unavailable')
    source=tmp_path/'input';source.mkdir()
    paths=[enhanced_fixture(source)[0]] if enhanced else write_ct_series(source/'native',rows=32,columns=32)
    study_uid=str(pydicom.dcmread(paths[0],stop_before_pixels=True).StudyInstanceUID)
    scouts=write_ct_series(source/'scout',study_uid=study_uid,image_type=['ORIGINAL','PRIMARY','LOCALIZER'])
    originals={path:sha256_file(path) for path in paths+scouts}
    app=AppService(work_root=tmp_path/'app',dcm2niix=executable)
    catalog=CatalogService(app)
    queue=PreparationQueue(catalog)
    catalog._queue=queue
    callbacks=[]
    def count_only_verified(case_id,*,idempotency_key):
        app.assert_catalog_countable(case_id)
        case=app.get_case(case_id)
        assert case['validation_status']=='VERIFIED'
        directory=app.volume_path(case_id).parent
        report=json.loads((directory/'qc_report.json').read_text())
        assert report['status']=='AUTOMATED_PASS_PENDING_MANUAL'
        assert all(check['passed'] for check in report['checks'])
        assert report['metrics']['max_absolute_hu_error']==0
        callbacks.append((case_id,idempotency_key))
        # The test validates scheduling without starting GPU inference.
        return {'run_id':'synthetic-verified-provenance'}
    uploads=UploadService(app,catalog,count_only_verified)
    selected=paths+scouts
    identity=uploads.create([{'name':p.name,'size':p.stat().st_size} for p in selected])['upload_id']
    for index,path in enumerate(selected):
        uploads.put(identity,index,0,path.read_bytes())
    uploads.finalize(identity)
    uploads.process(identity)
    uploads.schedule_ready()
    assert callbacks==[]
    assert catalog.status()['ready']==0
    assert catalog.status()['unavailable']==1
    job=queue._claim()
    assert job is not None
    # No monkeypatch and no supplied pipeline_provenance: the real dependency
    # metadata, Git state and lock hashes are collected inside PreparationQueue.
    queue.process(job)
    assert queue._claim() is None
    assert callbacks==[]
    assert catalog.status()['ready']==1
    uploads.schedule_ready()
    assert len(callbacks)==1
    assert uploads.status(identity)['status']=='completed'
    provenance_path=app.volume_path(callbacks[0][0]).parent/'tool_provenance.json'
    provenance=json.loads(provenance_path.read_text())['pipeline']
    try:
        expected_pip=importlib.metadata.version('pip')
    except importlib.metadata.PackageNotFoundError:
        expected_pip=None
    assert provenance['dependencies']['pip']==expected_pip
    for package in ('pydicom','nibabel','numpy','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg','pyjpegls'):
        assert provenance['dependencies'][package]==importlib.metadata.version(package)
    repository=Path(__file__).resolve().parents[1]
    assert provenance['requirements_lock_sha256']==sha256_file(repository/'requirements.lock')
    assert len(provenance['git_commit'])==40
    assert all(sha256_file(path)==originals[path] for path in selected)
