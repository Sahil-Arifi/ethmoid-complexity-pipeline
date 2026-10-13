import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthetic_faint_wall_experiment as experiment
from synthetic_blur_experiment import ImagingCase
from synthetic_counting_experiment import Fixture
from ethmoid_pipeline.counting.engine import CountingError


def summary_rows():
    rows = []
    for group, kind, n, successes in (("development", "pair", 18, 6), ("development", "control", 6, 6),
                                      ("challenge", "pair", 16, 4), ("challenge", "control", 16, 16)):
        for i in range(n):
            method = {"passed": i < successes, "candidate_counts": {"total": 2 if kind == "pair" else 1},
                      "failure_code": None, "inputs_preserved": True}
            rows.append({"name": f"{group}_{kind}_{i}", "group": group, "case_kind": kind,
                         "baseline": method, "candidate": copy.deepcopy(method)})
    for group in ("development", "challenge"):
        next(r for r in rows if r["group"] == group and not r["baseline"]["passed"])["candidate"]["passed"] = True
    return rows


def test_admission_requires_both_improvements_and_every_gate():
    rows = summary_rows()
    assert experiment.admission(rows)["admitted"]
    assert not experiment.admission(rows[:-1])["admitted"]
    for group in ("development", "challenge"):
        altered = copy.deepcopy(rows)
        for r in altered:
            if r["group"] == group and not r["baseline"]["passed"]:
                r["candidate"]["passed"] = False
        assert not experiment.admission(altered)["gates"][f"{group}_pair_improvement"]


@pytest.mark.parametrize("defect", ["regression", "false_split", "missing", "exception", "mutation", "duplicate", "extra_unknown"])
def test_admission_retains_independent_failures(defect):
    rows = summary_rows()
    target = next(r for r in rows if r["case_kind"] == "control")
    if defect == "extra_unknown":
        rows.append(dict(rows[0], group="unknown", case_kind="unknown"))
    elif defect == "duplicate":
        rows[-1]["name"] = rows[0]["name"]
    elif defect in ("regression", "false_split", "missing"):
        target["candidate"]["passed"] = False
        if defect == "false_split":
            target["candidate"]["candidate_counts"]["total"] = 2
        if defect == "missing":
            target["candidate"]["candidate_counts"]["total"] = None
    elif defect == "exception":
        target["candidate"]["failure_code"] = "UNEXPECTED_ValueError"
    else:
        target["candidate"]["inputs_preserved"] = False
    result = experiment.admission(rows)
    assert not result["admitted"]
    if defect == "false_split":
        assert result["control_false_splits"] == [target["name"]]
    if defect in ("exception", "mutation"):
        assert result["execution_errors"] == [target["name"] + ":candidate"]


@pytest.fixture
def unit_case():
    labels = np.zeros((12, 10, 8), dtype=np.uint16)
    labels[2:5, 2:8, 2:6], labels[6:10, 2:8, 2:6] = 1, 2
    affine = np.eye(4)
    affine[0, 3] = -5.5
    f = Fixture("unit_only", "Analytic harness test, outside evaluation design.",
                np.where(labels, -900, 800).astype(np.float32), labels, np.ones_like(labels, bool), affine, components=2)
    return ImagingCase(f, labels.astype(np.uint8), labels.copy(), 1, 0, 0)


def test_expected_rejection_keeps_null_count_and_all_zero_matches(unit_case, monkeypatch):
    monkeypatch.setattr(experiment, "case_record", lambda c: {"name": c.fixture.name, "hu": c.fixture.hu.tobytes().hex()})
    def reject(*args, **kwargs):
        raise CountingError("NO_SUPPORTED_CELLS", "No supported cells")
    monkeypatch.setattr(experiment, "compartment_stage", reject)
    row, cavities, air, exclusion = experiment.measure(unit_case, "baseline")
    assert row["failure_code"] == "NO_SUPPORTED_CELLS"
    assert row["candidate_counts"] == {"left": None, "right": None, "total": None}
    assert not row["passed"] and row["inputs_preserved"]
    assert not cavities.any() and not air.any() and not exclusion.any()
    assert len(row["matches"]) == 2 and all(m["dice"] == m["core_coverage"] == 0 for m in row["matches"])


def test_bad_registry_is_execution_error_with_preserved_denominator(unit_case, monkeypatch):
    monkeypatch.setattr(experiment, "case_record", lambda c: {"name": c.fixture.name})
    labels = unit_case.fixture.labels
    monkeypatch.setattr(experiment, "compartment_stage", lambda *a, **k: (labels, labels, [{"id": 1.5, "side": "left"}], []))
    row, *_ = experiment.measure(unit_case, "baseline")
    assert row["failure_code"] == "UNEXPECTED_ValueError"
    assert row["candidate_counts"]["total"] is None and not row["passed"]
    assert len(row["matches"]) == 2


def test_candidate_rejection_keeps_its_actual_inferred_mask(unit_case, monkeypatch):
    hu = unit_case.fixture.hu.copy()
    hu[5, 2:8, 2:6] = 75
    case = replace(unit_case, fixture=replace(unit_case.fixture, hu=hu))
    monkeypatch.setattr(experiment, "case_record", lambda c: {"name": c.fixture.name})
    def reject(*args):
        raise CountingError("NO_SUPPORTED_CELLS", "No supported cells")
    monkeypatch.setattr(experiment, "count_with_faint_wall", reject)
    row, cavities, air, mask = experiment.measure(case, "candidate")
    assert row["failure_code"] == "NO_SUPPORTED_CELLS"
    assert row["weak_evidence_used"] and row["inferred_exclusion_voxels"] > 0
    assert mask.any() and not cavities.any() and not air.any()


def test_mutated_scoring_truth_keeps_original_failure_denominator(unit_case, monkeypatch):
    monkeypatch.setattr(experiment, "case_record", lambda c: {"cores": c.cores.tobytes().hex()})
    def corrupt(*args, **kwargs):
        unit_case.cores.fill(0)
        raise RuntimeError("Injected unit-only input corruption")
    monkeypatch.setattr(experiment, "compartment_stage", corrupt)
    row, *_ = experiment.measure(unit_case, "baseline")
    assert row["failure_code"] == "UNEXPECTED_RuntimeError"
    assert not row["inputs_preserved"] and not row["passed"]
    assert row["candidate_counts"]["total"] is None
    assert len(row["matches"]) == 2 and all(m["dice"] == m["core_coverage"] == 0 for m in row["matches"])


def test_run_isolates_methods_from_input_mutation(unit_case, tmp_path, monkeypatch):
    from PIL import Image
    import synthetic_counting_experiment

    original = unit_case.fixture.hu.copy()
    for array in (unit_case.fixture.hu, unit_case.fixture.labels, unit_case.fixture.roi,
                  unit_case.fixture.affine, unit_case.highres_labels, unit_case.cores):
        array.flags.writeable = False
    repo = tmp_path / "source"
    (repo / ".local").mkdir(parents=True)
    (repo / ".local/frozen-faint-wall-plan.json").write_text('{}')
    monkeypatch.setattr(experiment, "REPO", repo)
    monkeypatch.setattr(experiment, "cases", lambda: [unit_case])
    monkeypatch.setattr(experiment, "protocol", lambda items: {})
    monkeypatch.setattr(experiment, "case_record", lambda c: {
        "name": c.fixture.name, "group": "development", "case_kind": "pair",
        "construction_counts": {"total": 2}, "hu": c.fixture.hu.tobytes().hex()})
    monkeypatch.setattr(synthetic_counting_experiment, "draw_case", lambda *a: Image.new("RGB", (1560, 1060)))
    def corrupt(hu, *args):
        assert not hu.flags.writeable
        hu.flags.writeable = True
        hu[0, 0, 0] = 0
        raise RuntimeError("Injected unit-only input corruption")
    def reject_fresh(hu, *args):
        assert not hu.flags.writeable and np.array_equal(hu, original)
        raise CountingError("NO_SUPPORTED_CELLS", "Injected unit-only rejection")
    monkeypatch.setattr(experiment, "compartment_stage", corrupt)
    monkeypatch.setattr(experiment, "count_with_faint_wall", reject_fresh)
    result = experiment.run(tmp_path / "output")
    row = result["results"][0]
    assert row["baseline"]["failure_code"] == "UNEXPECTED_RuntimeError"
    assert row["candidate"]["failure_code"] == "NO_SUPPORTED_CELLS"
    assert np.array_equal(unit_case.fixture.hu, original)
    assert result["execution_errors"] == [unit_case.fixture.name + ":baseline"]


def test_protocol_is_json_stable_and_includes_rules_and_candidate_sources():
    snapshot = experiment.protocol([])
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    assert "scripts/synthetic_faint_wall_candidate.py" in snapshot["source_hashes"]
    assert "scripts/synthetic_faint_wall_cases.py" in snapshot["source_hashes"]
    assert snapshot["config"]["bone_hu"] == 200 and snapshot["candidate_parameters"]["weak_hu_inclusive"] == 50


def test_freeze_has_no_measurement_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "REPO", tmp_path)
    monkeypatch.setattr(experiment, "cases", lambda: [])
    monkeypatch.setattr(experiment, "protocol", lambda items: {"frozen": "source-and-inputs"})
    monkeypatch.setattr(experiment, "measure", lambda *a: pytest.fail("Measurement during freeze"))
    path = experiment.freeze_plan()
    assert json.loads(path.read_text()) == {"frozen": "source-and-inputs"}
    with pytest.raises(FileExistsError):
        experiment.freeze_plan()
    assert json.loads(path.read_text()) == {"frozen": "source-and-inputs"}


def test_source_drift_is_rejected_before_output_and_counts(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    (repo / ".local").mkdir(parents=True)
    (repo / ".local/frozen-faint-wall-plan.json").write_text(json.dumps({"source": "old"}))
    monkeypatch.setattr(experiment, "REPO", repo)
    monkeypatch.setattr(experiment, "cases", lambda: [])
    monkeypatch.setattr(experiment, "protocol", lambda items: {"source": "new"})
    monkeypatch.setattr(experiment, "measure", lambda *a: pytest.fail("Detector reached after drift"))
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="frozen plan"):
        experiment.run(destination)
    assert not destination.exists()


def test_existing_output_and_any_git_checkout_are_protected(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("unchanged")
    monkeypatch.setattr(experiment, "cases", lambda: pytest.fail("Constructed inputs before output validation"))
    with pytest.raises(FileExistsError):
        experiment.run(existing)
    assert sentinel.read_text() == "unchanged"
    other_checkout = tmp_path / "other_source"
    other_checkout.mkdir()
    (other_checkout / ".git").write_text("gitdir: another-checkout")
    with pytest.raises(ValueError, match="outside source"):
        experiment.run(other_checkout / "output")
    assert not (other_checkout / "output").exists()
