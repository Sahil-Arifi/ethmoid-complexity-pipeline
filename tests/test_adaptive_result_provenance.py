"""Reject successful worker outputs whose assistance or authority was replayed."""
import copy
import json

import pytest

from ethmoid_pipeline.counting.profiles import sealed
from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.counting_service import ARTIFACTS, CountingService
from ethmoid_pipeline.webapp.service import WebAppError
from tests.test_counting_revisions import finish, setup_service, synthetic_executor


def adaptive_executor(source, output, config):
    synthetic_executor(source, output, config)
    protocol = json.loads(config.read_text())
    path = output / 'result.json'
    result = json.loads(path.read_text())
    result.update(method_version='scan-adaptive-native-0.3.0', protocol_version=protocol['protocol_version'], analysis_mode='automatic',
        outcome='automatic_candidate_count', assistance=None,
        model_identity={'localization': 'v2_geometric_baseline', 'direct_ethmoid_model': 'unavailable_without_promoted_weights'},
        scan_profile=protocol['scan_profile'], processing_plan=protocol['processing_plan'],
        artifact_verification={'status': 'verified', 'source_sha256': sha256_file(source),
            'sha256': {key: sha256_file(output / name) for key, name in ARTIFACTS.items()}})
    path.write_text(json.dumps(result))


@pytest.mark.parametrize('field,value', [
    ('method_version', 'another-method'), ('analysis_mode', 'assisted'),
    ('outcome', 'assisted_candidate_count'), ('model_identity', {'localization': 'invented_model'}),
    ('assistance', {'revision_id': 'wrong-parent'}), ('artifact_verification', {'status': 'verified'}),
])
def test_worker_cannot_change_frozen_authority_or_claim_unverified_masks(tmp_path, field, value):
    app, case, old = setup_service(tmp_path)
    old.close()
    def changed(source, output, config):
        adaptive_executor(source, output, config)
        path = output / 'result.json'
        result = json.loads(path.read_text())
        result[field] = value
        path.write_text(json.dumps(result))
    service = CountingService(app, executor=changed)
    try:
        run = service.create_run(case['case_id'])
        result = finish(service, case['case_id'], run['run_id'])
        assert result['status'] == 'failed' and result['counts'] is None
        assert result['outcome'] == 'technical_error'
    finally:
        service.close()


@pytest.mark.parametrize('field', ['source_sha256', 'revision_id', 'parent_run_id', 'region_sha256', 'private_path'])
def test_assisted_lineage_must_match_job_and_frozen_boundary(tmp_path, field):
    app, case, old = setup_service(tmp_path)
    old.close()
    service = CountingService(app, executor=adaptive_executor)
    try:
        run = service.create_run(case['case_id'])
        assert finish(service, case['case_id'], run['run_id'])['status'] == 'completed'
        row = service._run(case['case_id'], run['run_id'])
        result, output, _ = service._snapshot(row, 0)
        assistance = {'schema_version': 1, 'source_sha256': row['source_sha256'],
            'revision_id': 'boundary-fixture', 'parent_run_id': 'parent-fixture', 'region_sha256': 'a' * 64}
        protocol = json.loads(row['protocol_json'])
        protocol['assistance'] = assistance
        with app.store.connect() as db:
            db.execute('UPDATE count_jobs SET assistance_id=?,parent_run_id=? WHERE run_id=?',
                       ('boundary-fixture', 'parent-fixture', run['run_id']))
        result.update(analysis_mode='assisted', outcome='assisted_candidate_count',
            model_identity={'localization': 'reviewer_native_region', 'direct_ethmoid_model': 'unavailable_without_promoted_weights'},
            assistance=sealed({**assistance, 'exclusion_sha256': None, 'midline_ras_x': 0.0}))
        service._validate_adaptive_provenance(result, protocol, row, output)
        altered = copy.deepcopy(result)
        altered['assistance'][field] = 'different-value'
        altered['assistance'] = sealed(altered['assistance'])
        with pytest.raises(WebAppError, match='immutable boundary'):
            service._validate_adaptive_provenance(altered, protocol, row, output)
    finally:
        service.close()
