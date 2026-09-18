"""Real CPU compartment worker through the application API on synthetic CT."""
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import nibabel as nib

import pytest
from fastapi.testclient import TestClient

from ethmoid_pipeline.util import repo_root
from ethmoid_pipeline.webapp.app import create_app
from tests.test_counting_adaptive import phantom
from tests.test_counting_revisions import finish


@pytest.mark.integration
def test_real_assisted_worker_through_api_and_export(tmp_path):
    python = os.environ.get('ETHMOID_ENGINE_TEST_PYTHON')
    if not python or not Path(python).is_file():
        pytest.skip('Set ETHMOID_ENGINE_TEST_PYTHON to the locked CPU counting runtime')
    source, mask, _, _ = phantom(tmp_path, oblique=True, flipped=True)
    image = nib.load(source)
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 1)
    image.header.set_xyzt_units('mm')
    nib.save(image, source)
    app = create_app(work_root=tmp_path/'work', dcm2niix=tmp_path/'unused')
    counter = app.state.counting_service
    def execute(source, output, config):
        snapshot = json.loads(config.read_text())
        if not snapshot.get('assistance'):
            output.mkdir()
            (output/'result.json').write_text(json.dumps({'status':'failed','failure_code':'SPHENOID_UNRESOLVED'}))
            return
        assistance = config.with_name('assistance.json')
        acquisition = config.with_name('acquisition.json')
        plan = config.with_name('plan.json')
        assistance.write_text(json.dumps(snapshot['assistance']))
        acquisition.write_text(json.dumps({**snapshot['acquisition'], **snapshot['profile_characteristics']}))
        plan.write_text(json.dumps(snapshot['processing_plan']))
        environment = {**os.environ, 'PYTHONPATH': str(repo_root()/'src')}
        completed = subprocess.run([python,'-m','ethmoid_pipeline.counting.worker','--input',str(source),
            '--output',str(output),'--method','adaptive-v3','--config',str(config),'--assistance',str(assistance),
            '--acquisition',str(acquisition),'--processing-plan',str(plan)],env=environment,capture_output=True,timeout=30)
        assert completed.returncode == 0, completed.stdout.decode()+completed.stderr.decode()
    counter._executor = execute
    case = app.state.ethmoid_service.import_nifti(source,case_label='Case910')
    base = f"/api/cases/{case['case_id']}/count-runs"
    with TestClient(app) as client:
        response = client.post(base)
        assert response.status_code == 202, response.text
        initial = response.json()
        assert finish(counter,case['case_id'],initial['run_id'])['outcome'] == 'needs_review'
        url = base+'/'+initial['run_id']
        boundary = client.post(url+'/boundaries',files={'mask':('region.nii.gz',mask.read_bytes())},data={'note':'Synthetic expert fixture'})
        assert boundary.status_code == 201, boundary.text
        revision = boundary.json()
        created = client.post(url+'/assisted-runs',json={'boundary_id':revision['boundary_id'],'idempotency_key':'assisted-api-test'})
        assert created.status_code == 202, created.text
        run = finish(counter,case['case_id'],created.json()['run_id'])
        assert run['status'] == 'completed', run
        assert run['outcome'] == 'assisted_candidate_count' and run['counts'] == {'left':1,'right':1,'total':2}
        assert run['assistance']['parent_run_id'] == initial['run_id']
        assert run['processing_plan']['localization_backend'] == 'reviewer_native_region'
        assert client.get(url).json()['counts'] is None
        exported = client.get(base+'/'+run['run_id']+'/export')
        assert exported.status_code == 200
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['analysis_mode'] == 'assisted'
            assert manifest['assistance']['revision_id'] == revision['boundary_id']
            protocol = json.loads(archive.read('protocol.json'))
            assert 'region_path' not in protocol['assistance']
            assert str(tmp_path).encode() not in archive.read('protocol.json')
