import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ridge_counting_experiment as experiment
from synthetic_blur_experiment import ImagingCase
from synthetic_counting_experiment import Fixture
from ethmoid_pipeline.counting.engine import CountingError


@pytest.fixture
def unit_case():
    labels = np.zeros((12, 10, 8), dtype=np.uint16)
    labels[2:5, 2:8, 2:6], labels[6:10, 2:8, 2:6] = 1, 2
    affine = np.eye(4)
    affine[0, 3] = -5.5
    fixture = Fixture("unit_only", "Analytic harness test outside evaluation.",
                      np.where(labels, -900, 800).astype(np.float32), labels,
                      np.ones_like(labels, bool), affine, components=2)
    return ImagingCase(fixture, labels.astype(np.uint8), labels.copy(), 1, 0, 0)


def prediction(case, status="accepted"):
    f = case.fixture
    cells = []
    for identifier in (1, 2):
        mask = f.labels == identifier
        centroid = f.affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + f.affine[:3, 3]
        cells.append({"id": identifier, "side": "left" if centroid[0] <= 0 else "right",
                      "voxel_count": int(mask.sum()), "air_voxel_count": int(mask.sum()),
                      "centroid_ras_mm": centroid.tolist(), "uncertain": False, "reasons": []})
    return {"cavities": f.labels.copy(), "air": f.labels.copy(),
            "partitions": f.labels.astype(np.int32), "cells": cells, "warnings": [],
            "diagnostics": {"method": "unit", "status": status,
                            "counts": {"left": 1, "right": 1, "total": 2} if status == "accepted"
                            else {"left": None, "right": None, "total": None},
                            "withheld_reasons": [] if status == "accepted" else ["ambiguous_material"],
                            "initial_basin_count": 2, "final_basin_count": 2, "passage_merges": [],
                            "experimental_count_available": status == "accepted",
                            "clinical_accuracy": "NOT_MEASURED"}}


def test_withheld_perfect_proposals_are_not_accepted_recovery(unit_case, monkeypatch):
    monkeypatch.setattr(experiment, "count_with_ridges", lambda *args: prediction(unit_case, "withheld"))
    result, arrays = experiment.measure(unit_case)
    assert result["proposal_passed"] and not result["accepted_recovery"] and not result["passed"]
    assert result["candidate_counts"] == {"left": None, "right": None, "total": None}
    assert result["proposal_counts"]["total"] == 2 and len(result["matches"]) == 2
    assert np.array_equal(arrays["cavities"], unit_case.fixture.labels)


@pytest.mark.parametrize("unexpected", [False, True])
def test_failures_keep_null_counts_and_every_zero_match(unit_case, monkeypatch, unexpected):
    def fail(*args):
        if unexpected:
            raise RuntimeError("Unit-only injected error")
        raise CountingError("NO_SUPPORTED_CELLS", "Unit-only expected rejection")
    monkeypatch.setattr(experiment, "count_with_ridges", fail)
    result, arrays = experiment.measure(unit_case)
    assert result["failure_code"] == ("UNEXPECTED_RuntimeError" if unexpected else "NO_SUPPORTED_CELLS")
    assert result["candidate_counts"]["total"] is None and not result["accepted_recovery"]
    assert len(result["matches"]) == 2
    assert all(m["dice"] == m["core_coverage"] == 0 for m in result["matches"])
    assert all(not array.any() for array in arrays.values())


def test_detector_mutation_is_isolated_and_registered(unit_case, monkeypatch):
    before = unit_case.fixture.hu.copy()
    def corrupt(hu, roi, affine):
        assert not hu.flags.writeable and not roi.flags.writeable and not affine.flags.writeable
        hu.flags.writeable = True
        hu.fill(0)
        return prediction(unit_case)
    monkeypatch.setattr(experiment, "count_with_ridges", corrupt)
    result, _ = experiment.measure(unit_case)
    assert not result["inputs_preserved"] and not result["accepted_recovery"]
    assert result["failure_code"] == "UNEXPECTED_ValueError"
    assert np.array_equal(unit_case.fixture.hu, before)


@pytest.mark.parametrize("defect", ["registry", "accepted_null", "withheld_count", "air", "partition_dtype", "boolean_count", "bad_warnings"])
def test_bad_output_is_an_execution_error(unit_case, monkeypatch, defect):
    value = prediction(unit_case)
    if defect == "registry":
        value["cells"][0]["id"] = 1.5
    elif defect == "accepted_null":
        value["diagnostics"]["counts"]["total"] = None
    elif defect == "withheld_count":
        value["diagnostics"]["status"] = "withheld"
    elif defect == "air":
        value["air"].fill(0)
    elif defect == "partition_dtype":
        value["partitions"] = value["partitions"].astype(np.float32)
    elif defect == "boolean_count":
        value["diagnostics"]["counts"]["left"] = True
    else:
        value["warnings"] = {"unexpected": "object"}
    monkeypatch.setattr(experiment, "count_with_ridges", lambda *args: value)
    result, _ = experiment.measure(unit_case)
    assert result["failure_code"].startswith("UNEXPECTED_")
    assert result["candidate_counts"]["total"] is None and len(result["matches"]) == 2


def summary_rows():
    rows = []
    for kind, count in (("pair", 34), ("control", 22)):
        for index in range(count):
            passed = kind == "control" or index < 15
            candidate = {"accepted_recovery": passed, "proposal_passed": passed,
                         "status": "accepted", "candidate_counts": {"total": 2 if kind == "pair" else 1},
                         "failure_code": None, "inputs_preserved": True,
                         "output_fingerprint": "same"}
            rows.append({"name": f"known_{kind}_{index}", "group": "known_development",
                         "case_kind": kind, "expected_abstention": False, "equivalence_group": None,
                         "saved_faint_wall": {"passed": kind == "control" or index < 14},
                         "candidate": candidate})
    for index in range(12):
        ambiguous = index < 8
        rows.append({"name": f"fresh_{index}", "group": "fresh_stress",
                     "case_kind": "material_ambiguity" if ambiguous else "control",
                     "expected_abstention": ambiguous,
                     "equivalence_group": f"twins_{index // 2}" if ambiguous else None,
                     "input_hashes": {name: "same" for name in ("hu", "roi", "affine")},
                     "candidate": {"accepted_recovery": not ambiguous, "proposal_passed": True,
                                   "status": "withheld" if ambiguous else "accepted",
                                   "candidate_counts": {"total": None if ambiguous else 1},
                                   "failure_code": None, "inputs_preserved": True,
                                   "output_fingerprint": "same"}})
    return rows


def test_admission_requires_all_68_cases_and_strict_improvement():
    rows = summary_rows()
    assert experiment.admission(rows)["admitted"]
    assert not experiment.admission(rows[:-1])["admitted"]
    rows[14]["candidate"]["accepted_recovery"] = False
    assert not experiment.admission(rows)["gates"]["known_pair_improvement"]


@pytest.mark.parametrize("defect", ["duplicate", "regression", "control_withheld", "fresh_failure",
                                     "twins_differ", "twins_accepted", "inputs_differ", "exception", "mutation"])
def test_admission_rejects_independent_failures(defect):
    rows = summary_rows()
    if defect == "duplicate":
        rows[-1]["name"] = rows[0]["name"]
    elif defect == "regression":
        rows[0]["candidate"]["accepted_recovery"] = False
    elif defect == "control_withheld":
        rows[34]["candidate"].update(accepted_recovery=False, status="withheld", candidate_counts={"total": None})
    elif defect == "fresh_failure":
        rows[-1]["candidate"]["accepted_recovery"] = False
    elif defect == "twins_differ":
        rows[56]["candidate"]["output_fingerprint"] = "different"
    elif defect == "twins_accepted":
        rows[56]["candidate"]["status"] = "accepted"
    elif defect == "inputs_differ":
        rows[56]["input_hashes"]["hu"] = "different"
    elif defect == "exception":
        rows[0]["candidate"]["failure_code"] = "UNEXPECTED_RuntimeError"
    else:
        rows[0]["candidate"]["inputs_preserved"] = False
    assert not experiment.admission(rows)["admitted"]


def test_freeze_does_not_measure_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_items", lambda known: [])
    monkeypatch.setattr(experiment, "protocol", lambda known, items: {"known_run": str(known), "source": "fixed"})
    monkeypatch.setattr(experiment, "count_with_ridges", lambda *args: pytest.fail("Detector called during freeze"))
    target = tmp_path / "plan.json"
    experiment.freeze_plan(target, tmp_path / "known")
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.freeze_plan(target, tmp_path / "known")
    assert target.read_bytes() == original


def test_drift_rejected_before_outputs_or_measurement(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"known_run": str(tmp_path / "known"), "source": "old"}))
    monkeypatch.setattr(experiment, "load_items", lambda known: [])
    monkeypatch.setattr(experiment, "protocol", lambda known, items: {"known_run": str(known), "source": "new"})
    monkeypatch.setattr(experiment, "count_with_ridges", lambda *args: pytest.fail("Detector reached after drift"))
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="frozen plan"):
        experiment.run(plan, destination)
    assert not destination.exists()


def test_existing_outputs_and_git_checkouts_are_protected(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        experiment.run(tmp_path / "missing-plan", existing)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="outside source"):
        experiment.freeze_plan(checkout / "plan.json", tmp_path / "known")


def test_run_saves_complete_artifacts_with_mock_detector(unit_case, tmp_path, monkeypatch):
    known = tmp_path / "known"
    snapshot = {"known_run": str(known), "unit_only": True}
    metadata = {"name": unit_case.fixture.name, "group": "known_development", "case_kind": "pair",
                "expected_abstention": False, "equivalence_group": None, "saved_faint_wall": {"passed": False}}
    monkeypatch.setattr(experiment, "load_items", lambda path: [(unit_case, metadata)])
    monkeypatch.setattr(experiment, "protocol", lambda path, items: snapshot)
    monkeypatch.setattr(experiment, "count_with_ridges", lambda *args: prediction(unit_case))
    plan = tmp_path / "plan.json"
    experiment.freeze_plan(plan, known)
    destination = tmp_path / "run"
    result = experiment.run(plan, destination)
    assert result["results"][0]["candidate"]["accepted_recovery"]
    assert result["fixture_count"] == 1 and not result["admitted"]
    assert result["gates"]["frozen_inputs_preserved"]
    manifest = json.loads((destination / "sha256.json").read_text())
    assert set(manifest) == {"protocol.json", "results.json", "report.md",
                             "unit_only/input.npz", "unit_only/proposal.npz", "unit_only/result.json"}
    assert all(experiment._sha(destination / name) == digest for name, digest in manifest.items())
    with np.load(destination / "unit_only/proposal.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"cavities", "air", "partitions"}
        assert np.array_equal(saved["cavities"], unit_case.fixture.labels)
