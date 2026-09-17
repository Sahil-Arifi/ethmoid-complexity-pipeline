"""Freezing checks synthetic local files and never changes catalog state."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import nibabel as nib
import numpy as np
import pytest

from tests.test_evaluate_catalog import evaluation


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "evaluate_catalog", evaluation)
    spec = importlib.util.spec_from_file_location("freeze_catalog_evaluation", Path(__file__).resolve().parents[1] / "scripts/freeze_catalog_evaluation.py")
    freeze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    work = tmp_path / "local/EthmoidComplexity/work"
    app = work / "webapp"
    app.mkdir(parents=True)
    raw = tmp_path / "synthetic-source"
    raw.mkdir()
    (raw / "slice1").write_bytes(b"synthetic source one")
    (raw / "slice2").write_bytes(b"synthetic source two")
    fingerprint = evaluation.checks.canonical_sha256(sorted(evaluation.checks.sha256(p) for p in raw.iterdir()))
    volume = app / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.int16), np.diag([0.4, 0.4, 2.5, 1.])), volume)
    database = app / "ethmoid-app.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript("""
        CREATE TABLE preparation_jobs(status TEXT);
        CREATE TABLE catalog_sources(id TEXT,root_path TEXT);
        CREATE TABLE catalog_studies(id TEXT,source_id TEXT,study_label TEXT);
        CREATE TABLE catalog_series(id TEXT,study_id TEXT,series_label TEXT,case_id TEXT,
            source_fingerprint TEXT,preparation_status TEXT,reason_code TEXT,files_json TEXT,present INTEGER);
        CREATE TABLE cases(id TEXT,volume_relpath TEXT,source_sha256 TEXT,validation_status TEXT);
        CREATE TABLE catalog_case_links(case_id TEXT,series_id TEXT,acquisition_json TEXT,source_fingerprint TEXT);
        """)
        db.execute("INSERT INTO catalog_sources VALUES('source',?)", (str(raw),))
        db.execute("INSERT INTO catalog_studies VALUES('study','source','Case003')")
        db.execute("INSERT INTO catalog_series VALUES('series','study','Series001','case',?,'ready',NULL,?,1)", (fingerprint, json.dumps(["slice1", "slice2"])))
        db.execute("INSERT INTO catalog_series VALUES('scout','study','Series002',NULL,NULL,'unavailable','SCOUT','[]',1)")
        db.execute("INSERT INTO cases VALUES('case','synthetic.nii.gz',?,'VERIFIED')", (evaluation.checks.sha256(volume),))
        db.execute("INSERT INTO catalog_case_links VALUES('case','series',?,?)", (json.dumps({"slice_thickness_mm": 2.5, "spacing_mm": [0.4, 0.4, 2.5], "resolution_class": "outside_protocol"}), fingerprint))
    return freeze, work, work / "counting/frozen.local.json", database, raw, volume


def test_freeze_current_source_and_normalized_acquisition_are_read_only(catalog):
    freeze, work, destination, database, raw, _ = catalog
    before = evaluation.checks.sha256(database)
    result = freeze.freeze(work, destination)
    manifest = evaluation.checks.read_json(destination)
    assert result["ready"] == 1 and result["unavailable"] == 1 and result["series"] == 2
    entry = manifest["entries"][0]
    assert entry["dicom_fingerprint"] == evaluation.checks.canonical_sha256(sorted(evaluation.checks.sha256(p) for p in raw.iterdir()))
    assert entry["acquisition"]["study_label"] == "Case003"
    assert entry["acquisition"]["series_label"] == "Series001"
    assert entry["acquisition"]["slice_thickness_mm"] == 2.5
    assert entry["acquisition"]["spacing_mm"] == [0.4, 0.4, 2.5]
    assert entry["acquisition"]["resolution_class"] == "outside_protocol"
    assert entry["acquisition"]["study_id"] is None
    assert evaluation.checks.sha256(database) == before


@pytest.mark.parametrize("fault,code", [
    ("changed", "CATALOG_RAW_SOURCE_CHANGED"),
    ("missing", "CATALOG_RAW_SOURCE_UNAVAILABLE"),
    ("duplicate", "CATALOG_RAW_SOURCE_UNAVAILABLE"),
    ("escape", "CATALOG_RAW_SOURCE_UNAVAILABLE"),
    ("absent", "CATALOG_CURRENT_IDENTITY_INVALID"),
    ("wrong_link", "CATALOG_CURRENT_IDENTITY_INVALID"),
    ("changed_volume", "CATALOG_SOURCE_CHANGED"),
])
def test_freeze_rejects_stale_current_sources_without_updating_catalog(catalog, fault, code):
    freeze, work, destination, database, raw, volume = catalog
    if fault == "changed":
        (raw / "slice1").write_bytes(b"changed synthetic source")
    elif fault == "missing":
        (raw / "slice1").unlink()
    elif fault == "changed_volume":
        volume.write_bytes(volume.read_bytes() + b"changed")
    else:
        with sqlite3.connect(database) as db:
            if fault == "duplicate":
                db.execute("UPDATE catalog_series SET files_json=? WHERE id='series'", (json.dumps(["slice1", "slice1"]),))
            elif fault == "escape":
                (raw.parent / "outside").write_bytes(b"synthetic outside")
                db.execute("UPDATE catalog_series SET files_json=? WHERE id='series'", (json.dumps(["../outside"]),))
            elif fault == "absent":
                db.execute("UPDATE catalog_series SET present=0 WHERE id='series'")
            else:
                db.execute("UPDATE catalog_case_links SET series_id='other'")
    before = evaluation.checks.sha256(database)
    with pytest.raises(evaluation.checks.EvaluationError, match=code):
        freeze.freeze(work, destination)
    assert not destination.exists()
    assert evaluation.checks.sha256(database) == before
