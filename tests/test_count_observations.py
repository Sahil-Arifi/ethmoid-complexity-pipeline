from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.app import create_app
from tests.test_counting_revisions import PROTOCOL, finish, synthetic_executor
from tests.test_webapp_service import write_nifti


@pytest.fixture
def observation_run(tmp_path: Path):
    app = create_app(work_root=tmp_path / "work", dcm2niix=tmp_path / "unused")
    counting = app.state.counting_service
    counting._executor = synthetic_executor
    counting._protocol = PROTOCOL
    source_path = write_nifti(tmp_path / "source.nii.gz")
    source = nib.load(source_path)
    affine = source.affine.copy()
    affine[1, 1] *= -1
    image = nib.Nifti1Image(np.asarray(source.dataobj), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, source_path)
    case = app.state.ethmoid_service.import_nifti(source_path, case_label="Case904")
    with TestClient(app) as client:
        run = counting.create_run(case["case_id"], "observations")
        run = finish(counting, case["case_id"], run["run_id"])
        url = f"/api/cases/{case['case_id']}/count-runs/{run['run_id']}/observations"
        yield app, client, case, run, url


def _database_rows(store):
    with store.connect() as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return {row["name"]: [tuple(value) for value in connection.execute(f'SELECT * FROM "{row["name"]}" ORDER BY rowid')]
                for row in tables}


def test_observation_notes_round_trip_without_changing_counts_or_outputs(observation_run):
    app, client, case, run, url = observation_run
    counting = app.state.counting_service
    row = counting._run(case["case_id"], run["run_id"])
    _, original, hashes = counting._snapshot(row, 0)
    before = _database_rows(counting.store)
    assert client.get(url).json() == {"observations": []}
    point = run["cells"][0]["centroid_ras_mm"]
    first = client.post(url, json={"revision": 0, "kind": "merged", "candidate_ids": [1], "note": "  Review possible septum.  "})
    assert first.status_code == 201, first.text
    observation = first.json()
    assert observation["note"] == "Review possible septum."
    assert observation["source_sha256"] == run["source_sha256"]
    assert observation["artifact_sha256"] == hashes
    assert observation["method_version"] == "synthetic-fixture"
    assert observation["protocol_snapshot_sha256"] == run["protocol_snapshot_sha256"]
    second = client.post(url, json={"revision": 0, "kind": "missed", "point_ras_mm": point, "note": "Physician-marked location needs comparison."})
    assert second.status_code == 201, second.text
    notes = client.get(url).json()["observations"]
    assert {value["id"] for value in notes} == {observation["id"], second.json()["id"]}
    exported = client.get(url + "/export")
    assert exported.status_code == 200
    assert exported.headers["content-disposition"] == 'attachment; filename="count-observations.json"'
    assert exported.json() == {"observations": notes}
    assert "volume_relpath" not in exported.text and "source.nii.gz" not in exported.text
    assert _database_rows(counting.store) == before
    assert counting.get_run(case["case_id"], run["run_id"]) == run
    assert {path.name: sha256_file(path) for path in original.iterdir() if path.is_file()} == hashes
    sidecars = list((counting.root / "diagnostic-observations").glob("*/*.json"))
    assert len(sidecars) == 2 and all(not path.is_relative_to(original) for path in sidecars)


def test_observations_remain_bound_to_historical_revision(observation_run):
    app, client, case, run, url = observation_run
    service = app.state.counting_service
    selection = np.zeros((8, 9, 6), dtype=np.uint8)
    selection[5, 2:4, 2:4] = 1
    source = nib.load(app.state.ethmoid_service.volume_path(case["case_id"]))
    mask_path = service.root / "synthetic-split.nii.gz"
    nib.save(nib.Nifti1Image(selection, source.affine), mask_path)
    service.revise(case["case_id"], run["run_id"], 0, {"type": "split", "id": 300}, mask_path)
    result = client.post(url, json={"revision": 0, "kind": "extra", "candidate_ids": [300], "note": "Review the original candidate."})
    assert result.status_code == 201
    assert result.json()["revision"] == 0
    assert result.json()["artifact_sha256"] == service._snapshot(service._run(case["case_id"], run["run_id"]), 0)[2]
    stale_candidate = client.post(url, json={"revision": 1, "kind": "extra", "candidate_ids": [300], "note": "Retired candidate."})
    assert stale_candidate.status_code == 422
    assert client.get(url).json()["observations"] == [result.json()]
    assert service.get_run(case["case_id"], run["run_id"])["latest_revision"] == 1


@pytest.mark.parametrize("changes", [
    {"revision": "0"}, {"revision": True}, {"revision": -1}, {"kind": "confirmed"},
    {"candidate_ids": [True]}, {"candidate_ids": ["1"]}, {"candidate_ids": [1, 1]},
    {"candidate_ids": [2]}, {"candidate_ids": [0]}, {"candidate_ids": [1] * 65},
    {"candidate_ids": [], "point_ras_mm": None}, {"point_ras_mm": [0, 1]},
    {"point_ras_mm": ["0", 1, 2]}, {"point_ras_mm": [True, 1, 2]},
    {"point_ras_mm": [99999, 0, 0]}, {"note": "   "}, {"note": 123}, {"note": "x" * 2001},
    {"source_sha256": "forged"},
])
def test_invalid_observations_are_rejected_without_writing(observation_run, changes):
    app, client, _, _, url = observation_run
    payload = {"revision": 0, "kind": "merged", "candidate_ids": [1], "note": "Review this candidate."}
    response = client.post(url, json={**payload, **changes})
    assert response.status_code == 422, response.text
    assert not (app.state.counting_service.root / "diagnostic-observations").exists()


def test_foreign_run_revision_approved_and_changed_source_are_rejected(observation_run):
    app, client, case, run, url = observation_run
    payload = {"revision": 0, "kind": "side", "candidate_ids": [1], "note": "Review assignment."}
    assert client.post(url.replace(case["case_id"], "missing"), json=payload).status_code == 404
    other = app.state.ethmoid_service.import_nifti(app.state.ethmoid_service.volume_path(case["case_id"]), case_label="Case905")
    assert client.post(url.replace(case["case_id"], other["case_id"]), json=payload).status_code == 404
    assert client.post(url, json={**payload, "revision": 77}).status_code == 404
    with app.state.ethmoid_service.store.connect() as connection:
        connection.execute("UPDATE cases SET state='APPROVED',review_status='Approved' WHERE id=?", (case["case_id"],))
    assert client.post(url, json=payload).status_code == 409
    with app.state.ethmoid_service.store.connect() as connection:
        connection.execute("UPDATE cases SET state='IMPORTED',review_status='Draft' WHERE id=?", (case["case_id"],))
    source = app.state.ethmoid_service.volume_path(case["case_id"])
    source.write_bytes(b"changed synthetic source")
    assert client.post(url, json=payload).status_code == 409
    assert client.get(url).status_code == 409
    assert client.get(url + "/export").status_code == 409
    # Updating the current case's hash cannot rebind a historical run or notes.
    with app.state.ethmoid_service.store.connect() as connection:
        connection.execute("UPDATE cases SET source_sha256=? WHERE id=?", (sha256_file(source), case["case_id"]))
    assert client.post(url, json=payload).status_code == 409
    assert client.get(url + "/export").status_code == 409


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_coordinates_return_safe_validation_error(observation_run, number):
    _, client, _, _, url = observation_run
    response = client.post(url, content='{"revision":0,"kind":"missed","point_ras_mm":[' + number + ',0,0],"note":"Review."}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json() == {"detail": "The diagnostic observation request is invalid."}


def test_sidecar_provenance_corruption_is_rejected(observation_run):
    app, client, _, _, url = observation_run
    assert client.post(url, json={"revision": 0, "kind": "merged", "candidate_ids": [1], "note": "Review."}).status_code == 201
    path = next((app.state.counting_service.root / "diagnostic-observations").glob("*/*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["artifact_sha256"]["cavities.nii.gz"] = "forged"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert client.get(url).status_code == 409
    assert client.get(url + "/export").status_code == 409


@pytest.mark.parametrize("changes", [
    {"candidate_ids": [65000]}, {"candidate_ids": [1, 1]},
    {"point_ras_mm": [99999, 0, 0]}, {"note": "   "},
    {"candidate_ids": [], "point_ras_mm": None}, {"revision": 77},
])
def test_invalid_saved_observation_semantics_block_list_and_export(observation_run, changes):
    app, client, _, _, url = observation_run
    payload = {"revision": 0, "kind": "merged", "candidate_ids": [1], "note": "Review."}
    assert client.post(url, json=payload).status_code == 201
    path = next((app.state.counting_service.root / "diagnostic-observations").glob("*/*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**value, **changes}), encoding="utf-8")
    for target in (url, url + "/export"):
        response = client.get(target)
        assert response.status_code == 409
        assert response.json() == {"detail": "Saved diagnostic notes failed their integrity check."}


def test_native_bounds_use_physical_affine_not_coordinate_sign(observation_run):
    app, client, _, run, url = observation_run
    source = nib.load(app.state.ethmoid_service.volume_path(run["case_id"]))
    # Fixture affine includes reflected y, unequal spacing, and translation.
    assert np.linalg.det(source.affine[:3, :3]) < 0
    inside = nib.affines.apply_affine(source.affine, [0, 0, 0]).tolist()
    outside = nib.affines.apply_affine(source.affine, [-1, 0, 0]).tolist()
    payload = {"revision": 0, "kind": "missed", "note": "Physical location."}
    assert client.post(url, json={**payload, "point_ras_mm": inside}).status_code == 201
    assert client.post(url, json={**payload, "point_ras_mm": outside}).status_code == 422
