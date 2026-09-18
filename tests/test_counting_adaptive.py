"""Adaptive engineering checks, using synthetic data; no clinical accuracy claim."""
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from ethmoid_pipeline.counting.profiles import build_scan_profile, build_processing_plan, sealed
from ethmoid_pipeline.counting.protocol import load_counting_protocol, ADAPTIVE_PROTOCOL_VERSION


def phantom(tmp_path, *, oblique=False, flipped=False):
    hu = np.full((52, 44, 40), 800, np.float32)
    hu[7:22, 10:32, 8:30] = -900
    hu[29:44, 10:32, 8:30] = 35
    labels = np.zeros(hu.shape, np.uint8)
    labels[4:25, 7:35, 5:33] = 1
    labels[26:47, 7:35, 5:33] = 2
    affine = np.diag([0.6, 0.6, 0.6, 1])
    affine[:3, 3] = [-15, -12, -10]
    if oblique:
        angle = np.deg2rad(15)
        affine[:3, :3] = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]) @ affine[:3, :3]
    if flipped:
        transform = np.eye(4)
        transform[0, 0], transform[0, 3] = -1, hu.shape[0] - 1
        affine = affine @ transform
        hu, labels = hu[::-1].copy(), labels[::-1].copy()
    source, mask = tmp_path / "source.nii.gz", tmp_path / "region.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), source)
    nib.save(nib.Nifti1Image(labels, affine), mask)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assistance = {"schema_version": 1, "source_sha256": digest, "region_path": str(mask),
        "region_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(), "revision_id": "boundary_test", "parent_run_id": "run_failed"}
    acquisition = {"source_sha256": digest, "slice_thickness_mm": 2.5, "spacing_mm": [0.6, 0.6, 0.6]}
    return source, mask, assistance, acquisition


def test_profile_is_source_bound_deterministic_and_not_identity_dependent(tmp_path):
    source, _, _, acquisition = phantom(tmp_path)
    first = build_scan_profile(source, acquisition)
    changed = {**acquisition, "study_label": "Case999", "series_label": "Series001"}
    assert first == build_scan_profile(source, changed)
    assert first["slice_thickness_mm"] == 2.5 and first["slice_interval_mm"] == 0.6
    assert first["image_diagnostics"]["sample_voxels"] <= 64 ** 3
    plan = build_processing_plan(first)
    assert plan["acquisition_policy"] == "experimental_resolution"
    assert plan["counting_grid"] == "original_native"
    assert plan["resolution_recovery_claim"] is False


def test_profile_unknown_thickness_is_never_inferred(tmp_path):
    source, *_ = phantom(tmp_path)
    profile = build_scan_profile(source)
    assert profile["slice_thickness_mm"] is None
    assert profile["slice_interval_mm"] is None
    assert build_processing_plan(profile)["acquisition_policy"] == "unknown_resolution"


def test_profile_mutation_invalidates_digest(tmp_path):
    source, *_ = phantom(tmp_path)
    profile = build_scan_profile(source)
    profile["slice_thickness_mm"] = 0.6
    with pytest.raises(ValueError, match="digest"):
        build_processing_plan(profile)


def test_v3_keeps_v2_cell_definition_and_parameters():
    v2 = load_counting_protocol()
    v3 = load_counting_protocol(ADAPTIVE_PROTOCOL_VERSION)
    for key in ("algorithm", "included_variants", "assumptions", "count_opacified_cells", "maximum_slice_thickness_mm"):
        assert v3[key] == v2[key]
    assert v3["sha256"] != v2["sha256"]


@pytest.mark.parametrize("changed", [{"source_sha256": "0" * 64}, {"kernel_class": "patient free text"}])
def test_profile_rejects_unbound_or_uncontrolled_acquisition(tmp_path, changed):
    source, _, _, acquisition = phantom(tmp_path)
    with pytest.raises(ValueError):
        build_scan_profile(source, {**acquisition, **changed})


def require_engine():
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    from ethmoid_pipeline.counting.adaptive import analyze_adaptive_volume
    return analyze_adaptive_volume


@pytest.mark.parametrize("oblique,flipped", [(False, False), (True, False), (True, True)])
def test_assisted_end_to_end_after_failed_automatic_localization(tmp_path, monkeypatch, oblique, flipped):
    analyze = require_engine()
    from ethmoid_pipeline.counting.engine import analyze_volume
    import ethmoid_pipeline.counting.localization as localization
    source, _, assistance, acquisition = phantom(tmp_path, oblique=oblique, flipped=flipped)
    original = analyze_volume(source, tmp_path / "baseline", tmp_path / "missing", acquisition=acquisition)
    assert original["status"] == "failed" and original["counts"]["total"] is None
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **k: pytest.fail("Assisted boundaries must not require failed automatic localization"))
    output = tmp_path / "assisted"
    profile = build_scan_profile(source, acquisition)
    plan = build_processing_plan(profile, assistance=assistance)
    result = analyze(source, output, config=load_counting_protocol(ADAPTIVE_PROTOCOL_VERSION), acquisition=acquisition,
        assistance=assistance, processing_plan=plan, progress_path=tmp_path / "progress.json")
    assert result["status"] == "completed", result
    assert result["outcome"] == "assisted_candidate_count" and result["analysis_mode"] == "assisted"
    assert result["counts"]["total"] == 2
    assert result["processing_plan"] == plan
    assert result["scan_profile"] == profile
    assert result["assistance"]["parent_run_id"] == "run_failed"
    assert "region_path" not in result["assistance"]
    assert result["artifact_verification"]["status"] == "verified"
    raw = nib.load(source)
    cavity = nib.load(output / result["artifacts"]["cavities"])
    assert cavity.shape == raw.shape and np.allclose(cavity.affine, raw.affine)
    assert np.array_equal(np.asanyarray(cavity.dataobj) > 0, np.asanyarray(raw.dataobj) < 200)
    assert all(cell["origin"] == "automatic" for cell in result["cells"])
    assert json.loads((tmp_path / "progress.json").read_text())["stage"] == "completed"
    assert json.loads((tmp_path / "baseline" / "result.json").read_text()) == original


@pytest.mark.parametrize("field,value,code", [
    ("source_sha256", "0" * 64, "ASSISTANCE_SOURCE"),
    ("region_sha256", "0" * 64, "ASSISTANCE_INTEGRITY"),
    ("parent_run_id", None, "ASSISTANCE_LINEAGE"),
    ("midline_ras_x", float("nan"), "ASSISTANCE_LATERALITY"),
])
def test_assistance_invalid_provenance_never_publishes_count(tmp_path, field, value, code):
    analyze = require_engine()
    source, _, assistance, acquisition = phantom(tmp_path)
    assistance[field] = value
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["failure_code"] == code
    assert result["counts"]["total"] is None and result["cells"] == [] and result["artifacts"] == {}


@pytest.mark.parametrize("mutation,code", [("affine", "ASSISTANCE_GEOMETRY"), ("values", "ASSISTANCE_VALUES"), ("dtype", "ASSISTANCE_VALUES"), ("edge", "TARGET_CLIPPED")])
def test_assistance_geometry_values_and_source_coverage(tmp_path, mutation, code):
    analyze = require_engine()
    source, mask, assistance, acquisition = phantom(tmp_path)
    image = nib.load(mask)
    data, affine = np.asarray(image.dataobj).copy(), image.affine.copy()
    if mutation == "affine":
        affine[0, 3] += 1
    elif mutation == "values":
        data[5, 8, 6] = 8
    elif mutation == "dtype":
        data = data.astype(np.float32)
    else:
        data[0, 8, 6] = 1
    nib.save(nib.Nifti1Image(data, affine), mask)
    assistance["region_sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["failure_code"] == code and result["counts"]["total"] is None


def test_assistance_does_not_bypass_bone_evidence(tmp_path):
    analyze = require_engine()
    source, _, assistance, acquisition = phantom(tmp_path)
    raw = nib.load(source)
    hu = np.full(raw.shape, -900, np.float32)
    hu[0:3] = 800  # HU range is plausible, but no walls enclose the assisted region.
    nib.save(nib.Nifti1Image(hu, raw.affine), source)
    assistance["source_sha256"] = acquisition["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["failure_code"] == "NO_SUPPORTED_CELLS"
    assert result["outcome"] == "needs_review" and result["counts"]["total"] is None


def test_altered_processing_plan_is_rejected(tmp_path):
    analyze = require_engine()
    source, _, assistance, acquisition = phantom(tmp_path)
    plan = build_processing_plan(build_scan_profile(source, acquisition))
    plan["proxy_max_voxels"] *= 2
    plan = sealed(plan)
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition, processing_plan=plan)
    assert result["failure_code"] == "PROCESSING_PLAN_MISMATCH"
    assert result["counts"]["total"] is None


def test_verification_failure_clears_success_fields(tmp_path, monkeypatch):
    analyze = require_engine()
    from ethmoid_pipeline.counting.adaptive import AdaptiveStrategy
    from ethmoid_pipeline.counting.engine import CountingError
    source, _, assistance, acquisition = phantom(tmp_path)
    def invalid(*args):
        raise CountingError("ARTIFACT_VERIFICATION", "Injected artifact integrity failure")
    monkeypatch.setattr(AdaptiveStrategy, "verify_artifacts", invalid)
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["status"] == "failed" and result["outcome"] == "technical_error"
    assert result["counts"]["total"] is None and result["cells"] == [] and result["artifacts"] == {}


def test_worker_v2_rejects_assisted_options(tmp_path, capsys):
    require_engine()
    from ethmoid_pipeline.counting.worker import main
    code = main(["--input", str(tmp_path / "absent"), "--output", str(tmp_path / "out"), "--assistance", str(tmp_path / "anything")])
    assert code == 2 and "INVALID_CONFIG" in capsys.readouterr().out


def test_worker_adaptive_executes_assisted_contract(tmp_path, capsys):
    require_engine()
    from ethmoid_pipeline.counting.worker import main
    source, _, assistance, acquisition = phantom(tmp_path)
    for name, value in (("assistance", assistance), ("acquisition", acquisition)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    code = main(["--input", str(source), "--output", str(tmp_path / "out"), "--method", "adaptive-v3",
        "--assistance", str(tmp_path / "assistance.json"), "--acquisition", str(tmp_path / "acquisition.json")])
    assert code == 0
    assert json.loads((tmp_path / "out" / "result.json").read_text())["outcome"] == "assisted_candidate_count"


def test_reviewer_side_labels_are_used_instead_of_midline_guess(tmp_path):
    analyze = require_engine()
    source, mask, assistance, acquisition = phantom(tmp_path)
    image = nib.load(mask)
    labels = np.asarray(image.dataobj).copy()
    labels[labels > 0] = 2
    nib.save(nib.Nifti1Image(labels, image.affine), mask)
    assistance["region_sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["status"] == "completed", result
    assert result["counts"] == {"left": 0, "right": 2, "total": 2}
    assert all(cell["side"] == "right" for cell in result["cells"])


def test_conflicting_reviewer_side_labels_require_review(tmp_path):
    analyze = require_engine()
    source, mask, assistance, acquisition = phantom(tmp_path)
    image = nib.load(mask)
    labels = np.asarray(image.dataobj).copy()
    labels[10:15, 15:20, 12:20] = 2  # One enclosed cavity now has conflicting side labels.
    nib.save(nib.Nifti1Image(labels, image.affine), mask)
    assistance["region_sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    result = analyze(source, tmp_path / "result", assistance=assistance, acquisition=acquisition)
    assert result["failure_code"] == "ASSISTANCE_LATERALITY"
    assert result["outcome"] == "needs_review" and result["counts"]["total"] is None
