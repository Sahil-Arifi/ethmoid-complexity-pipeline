"""Recovery tests use deterministic synthetic masks, never accuracy claims."""
import errno
import json
import sqlite3
import subprocess
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.webapp.counting_service import CountingService
from ethmoid_pipeline.webapp.service import AppService, WebAppError
from tests.test_counting_revisions import PROTOCOL, finish, setup_service, synthetic_executor


def test_verified_checkpoint_survives_graceful_worker_restart(tmp_path):
    started, release = threading.Event(), threading.Event()
    calls = []
    def execute(source, output, config):
        calls.append(1)
        started.set()
        assert release.wait(5)
        synthetic_executor(source, output, config)
    app, case, first = setup_service(tmp_path, execute)
    run = first.create_run(case['case_id'])
    assert started.wait(5)
    closing = threading.Thread(target=first.close)
    closing.start()
    assert first._closed.wait(2)
    release.set()
    closing.join(5)
    assert not closing.is_alive()
    assert first.get_run(case['case_id'], run['run_id'])['status'] == 'queued'
    def must_not_reinfer(*args):
        pytest.fail('Verified inference checkpoint was recomputed')
    second = CountingService(app, executor=must_not_reinfer, protocol=PROTOCOL)
    try:
        done = finish(second, case['case_id'], run['run_id'])
        assert done['status'] == 'completed'
        assert done['outcome'] == 'automatic_candidate_count'
        assert len(calls) == 1
    finally:
        second.close()


def test_two_service_instances_do_not_steal_live_work(tmp_path):
    started, release = threading.Event(), threading.Event()
    def execute(source, output, config):
        started.set()
        assert release.wait(5)
        synthetic_executor(source, output, config)
    app, case, first = setup_service(tmp_path, execute)
    run = first.create_run(case['case_id'])
    assert started.wait(5)
    second = CountingService(app, executor=lambda *a: pytest.fail('Duplicate inference'), protocol=PROTOCOL)
    try:
        assert second.create_run(case['case_id'])['run_id'] == run['run_id']
        assert second.get_run(case['case_id'], run['run_id'])['status'] == 'running'
        release.set()
        assert finish(first, case['case_id'], run['run_id'])['status'] == 'completed'
    finally:
        release.set()
        second.close()
        first.close()


@pytest.mark.parametrize('failures,expected', [(2, 'completed'), (3, 'failed')])
def test_transient_retry_budget(tmp_path, failures, expected):
    calls = []
    def execute(source, output, config):
        calls.append(1)
        if len(calls) <= failures:
            raise subprocess.TimeoutExpired('synthetic-worker', 1)
        synthetic_executor(source, output, config)
    _, case, service = setup_service(tmp_path, execute)
    try:
        run = service.create_run(case['case_id'])
        done = finish(service, case['case_id'], run['run_id'])
        assert done['status'] == expected and len(calls) == 3
        assert done['retry_count'] == 2
        if expected == 'failed':
            assert done['counts'] is None and done['failure_code'] == 'RETRY_EXHAUSTED'
    finally:
        service.close()


def test_anatomical_abstention_is_not_automatically_retried(tmp_path):
    calls = []
    def execute(source, output, config):
        calls.append(1)
        output.mkdir()
        (output / 'result.json').write_text(json.dumps({'status': 'failed', 'failure_code': 'SPHENOID_UNRESOLVED'}))
    _, case, service = setup_service(tmp_path, execute)
    try:
        run = service.create_run(case['case_id'])
        done = finish(service, case['case_id'], run['run_id'])
        assert done['outcome'] == 'needs_review' and done['counts'] is None
        assert len(calls) == 1
    finally:
        service.close()


def test_disk_full_has_actionable_outcome_without_retry_loop(tmp_path):
    def execute(*args):
        raise OSError(errno.ENOSPC, 'synthetic disk full')
    _, case, service = setup_service(tmp_path, execute)
    try:
        run = service.create_run(case['case_id'])
        done = finish(service, case['case_id'], run['run_id'])
        assert done['failure_code'] == 'STORAGE_FULL'
        assert done['outcome'] == 'technical_error' and done['attempt'] == 1
    finally:
        service.close()


def test_cancel_queued_run_never_launches_inference(tmp_path):
    started, release = threading.Event(), threading.Event()
    calls = []
    def execute(source, output, config):
        calls.append(1)
        started.set()
        assert release.wait(5)
        synthetic_executor(source, output, config)
    app, case, service = setup_service(tmp_path, execute)
    try:
        other = app.import_nifti(app.volume_path(case['case_id']), case_label='Case902')
        first = service.create_run(case['case_id'])
        assert started.wait(5)
        queued = service.create_run(other['case_id'])
        assert service.cancel(other['case_id'], queued['run_id'])['outcome'] == 'cancelled'
        release.set()
        finish(service, case['case_id'], first['run_id'])
        assert len(calls) == 1
    finally:
        release.set()
        service.close()


def test_boundary_revision_is_native_source_bound_and_immutable(tmp_path):
    app, case, service = setup_service(tmp_path)
    try:
        run = service.create_run(case['case_id'])
        done = finish(service, case['case_id'], run['run_id'])
        source = nib.load(app.volume_path(case['case_id']))
        region = np.zeros(source.shape, np.uint8)
        region[1:3, 1:6, 1:5] = 1
        region[5:7, 1:6, 1:5] = 2
        path = tmp_path / 'boundary.nii.gz'
        nib.save(nib.Nifti1Image(region, source.affine), path)
        boundary = service.save_boundary(case['case_id'], run['run_id'], path, 'Synthetic review')
        assert service.boundary_path(case['case_id'], boundary['boundary_id']).is_file()
        assert service.get_run(case['case_id'], run['run_id'])['counts'] == done['counts']
        broken = source.affine.copy()
        broken[0, 3] += 1
        nib.save(nib.Nifti1Image(region, broken), path)
        with pytest.raises(WebAppError, match='geometry'):
            service.save_boundary(case['case_id'], run['run_id'], path)
        stored = service.boundary_path(case['case_id'], boundary['boundary_id'])
        stored.write_bytes(b'corrupt synthetic artifact')
        with pytest.raises(WebAppError, match='integrity'):
            service.boundary_path(case['case_id'], boundary['boundary_id'])
    finally:
        service.close()


def test_schema_three_backup_precedes_additive_migration(tmp_path):
    app, case, service = setup_service(tmp_path)
    service.close()
    with app.store.connect() as db:
        db.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")
        before = dict(db.execute('SELECT * FROM cases WHERE id=?', (case['case_id'],)).fetchone())
    again = AppService(work_root=app.root, dcm2niix=tmp_path/'unused')
    backups = list(app.root.glob('schema-v3-*.sqlite3'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == '3'
    with again.store.connect() as db:
        assert dict(db.execute('SELECT * FROM cases WHERE id=?', (case['case_id'],)).fetchone()) == before
