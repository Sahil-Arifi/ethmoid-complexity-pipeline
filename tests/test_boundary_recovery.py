"""Saved anatomical work remains discoverable without a successful count."""
import json

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from ethmoid_pipeline.webapp.app import create_app
from tests.test_counting_revisions import PROTOCOL, finish
from tests.test_webapp_service import write_nifti


def failed_executor(source, output, config):
    output.mkdir()
    (output / 'result.json').write_text(json.dumps({'status': 'failed', 'failure_code': 'SPHENOID_UNRESOLVED'}))


def configure(tmp_path):
    app = create_app(work_root=tmp_path / 'work', dcm2niix=tmp_path / 'unused')
    service = app.state.counting_service
    service._protocol = PROTOCOL
    service._executor = failed_executor
    source = write_nifti(tmp_path / 'source.nii.gz')
    case = app.state.ethmoid_service.import_nifti(source, case_label='Case902')
    mask = np.zeros((8, 9, 6), np.uint8)
    mask[1:3, 1:6, 1:5] = 1
    mask[5:7, 1:6, 1:5] = 2
    path = tmp_path / 'boundary.nii.gz'
    nib.save(nib.Nifti1Image(mask, nib.load(source).affine), path)
    return app, case['case_id'], path


def failed_parent(client, app, case_id):
    run = client.post(f'/api/cases/{case_id}/count-runs').json()
    assert finish(app.state.counting_service, case_id, run['run_id'])['outcome'] == 'needs_review'
    return f"/api/cases/{case_id}/count-runs/{run['run_id']}"


def save(client, url, path, note):
    response = client.post(url + '/boundaries', data={'note': note},
        files={'mask': ('boundary.nii.gz', path.read_bytes(), 'application/gzip')})
    assert response.status_code == 201, response.text
    return response.json()


def test_failed_run_saved_boundaries_survive_reload_with_stable_order_and_isolation(tmp_path):
    app, case_id, path = configure(tmp_path)
    with TestClient(app) as client:
        url = failed_parent(client, app, case_id)
        first = save(client, url, path, 'First saved review')
        second = save(client, url, path, 'Second saved review')
        other = failed_parent(client, app, case_id)
        assert client.get(other + '/boundaries').json() == {'boundaries': []}
        assert client.get(other + '/boundaries/' + first['boundary_id']).status_code == 404
        assert client.get(other + '/boundaries/' + first['boundary_id'] + '/mask').status_code == 404
        assert client.get(url.replace(case_id, 'missing-case') + '/boundaries').status_code == 404
    reloaded = create_app(work_root=tmp_path / 'work', dcm2niix=tmp_path / 'unused')
    with TestClient(reloaded) as client:
        response = client.get(url + '/boundaries')
        assert response.status_code == 200
        assert response.json() == {'boundaries': [second, first]}
        assert 'mask_relpath' not in response.text
        assert client.get(url + '/boundaries/' + second['boundary_id'] + '/mask').status_code == 200
        assert client.get(url).json()['counts'] is None
        assert len(client.get(f'/api/cases/{case_id}/count-runs').json()['runs']) == 2


def test_boundary_discovery_rejects_changed_source_and_parent_binding(tmp_path):
    app, case_id, path = configure(tmp_path)
    with TestClient(app) as client:
        url = failed_parent(client, app, case_id)
        boundary = save(client, url, path, 'Saved before source drift')
        with app.state.ethmoid_service.store.connect() as db:
            db.execute('UPDATE boundary_revisions SET source_sha256=? WHERE id=?', ('0' * 64, boundary['boundary_id']))
        assert client.get(url + '/boundaries').status_code == 409
        assert client.get(url + '/boundaries/' + boundary['boundary_id']).status_code == 409
        assert client.get(url + '/boundaries/' + boundary['boundary_id'] + '/mask').status_code == 409
        with app.state.ethmoid_service.store.connect() as db:
            db.execute('UPDATE boundary_revisions SET source_sha256=? WHERE id=?', (boundary['source_sha256'], boundary['boundary_id']))
        app.state.ethmoid_service.volume_path(case_id).write_bytes(b'changed synthetic source')
        assert client.get(url + '/boundaries').status_code == 409
        assert client.get(url + '/boundaries/' + boundary['boundary_id']).status_code == 409
