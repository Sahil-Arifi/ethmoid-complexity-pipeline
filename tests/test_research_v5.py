"""CPU engineering regression controls; no expert-referenced accuracy claims."""
from copy import deepcopy
from dataclasses import asdict
from importlib.resources import files
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting import grids, localization, research, research_v5, worker
from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, compartment_stage
from ethmoid_pipeline.counting.profiles import build_scan_profile, file_sha256, sealed
from ethmoid_pipeline.counting.protocol import load_counting_protocol
from ethmoid_pipeline.counting.research_policy import LOCALIZATION_POLICY as V4_POLICY, build_research_plan as v4_plan
from ethmoid_pipeline.counting.research_v5_policy import (
    LOCALIZATION_POLICY, METHOD_VERSION, PLAN_VERSION, PROTOCOL_VERSION, build_research_plan,
)
from tests.test_counting_research_localization import sinus_phantom
from tests.test_research_proxy import affine_for, landmarks, synthetic_source


def width_phantom(spacing, lower=28, upper=32):
    args = list(sinus_phantom(spacing))
    x, y, z = np.ogrid[tuple(slice(0, n) for n in args[0].shape)]
    x, y, z = x * spacing[0], y * spacing[1], z * spacing[2]
    inside = (x >= lower) & (x < upper) & (y >= 10) & (y < 20) & (z >= 8) & (z < 24)
    args[0] = np.where(inside, -900, 800).astype(np.float32)
    args[1], args[2], args[3] = args[0] >= 200, inside, inside.copy()
    return args


@pytest.mark.parametrize("interval", [0.5, 0.625, 1.0, 1.25, 2.0, 2.5])
def test_existing_supported_seed_preserves_exact_legacy_wall_and_region(interval):
    args = width_phantom((0.5, 0.5, interval), 24, 36)
    original = args[0].copy()
    baseline_diagnostics, selected_diagnostics = {}, {}
    baseline = localization._sphenoid_boundary(*args, diagnostics=baseline_diagnostics)
    selected = localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=selected_diagnostics)
    assert all(np.array_equal(before, after) for before, after in zip(baseline, selected))
    assert np.array_equal(args[0], original)
    old, new = baseline_diagnostics["sphenoid"], selected_diagnostics["sphenoid"]
    assert new["initial_interior_seed_voxels"] == new["interior_seed_voxels"] == old["interior_seed_voxels"] > 0
    assert new["initial_seed_clearance_mm"] == new["chosen_seed_clearance_mm"] == old["seed_clearance_mm"]
    assert new["seed_fallback_used"] is False
    assert new["seed_fallback_reason"] == "legacy_seed_supported"
    assert new["seed_spacing_reference"] == "current_anatomical_localization_grid"


@pytest.mark.parametrize("interval", [0.625, 1.0])
def test_empty_thin_seed_remains_unavailable_even_when_v4_accepts_it(interval):
    # A 2.2mm-wide lumen on a 0.2mm X grid has >=3 supported columns
    # at 1mm clearance, while no voxel reaches the larger legacy clearance.
    args = width_phantom((0.2, 0.5, interval), 28.8, 31.0)
    v4_diagnostics = {}
    localization._sphenoid_boundary(*args, localization_policy=V4_POLICY, diagnostics=v4_diagnostics)
    assert v4_diagnostics["sphenoid"]["interior_seed_voxels"] > 0
    diagnostic = {}
    with pytest.raises(CountingError, match="no spatially resolved interior") as failure:
        localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=diagnostic)
    assert failure.value.code == "SPHENOID_UNRESOLVED"
    value = diagnostic["sphenoid"]
    assert value["initial_interior_seed_voxels"] == value["interior_seed_voxels"] == 0
    assert value["initial_seed_clearance_mm"] == value["chosen_seed_clearance_mm"] == 2 * interval
    assert value["seed_fallback_used"] is False and value["seed_fallback_reason"] == "legacy_seed_empty_thin_grid"


@pytest.mark.parametrize("interval", [1.01, 1.25, 2.0, 2.5])
def test_empty_thick_seed_uses_the_same_physical_seed_wall_and_region_as_v4(interval):
    args = sinus_phantom((0.5, 0.5, interval))
    with pytest.raises(CountingError, match="no spatially resolved interior"):
        localization._sphenoid_boundary(*args)
    before, after = {}, {}
    v4 = localization._sphenoid_boundary(*args, localization_policy=V4_POLICY, diagnostics=before)
    v5 = localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=after)
    assert all(np.array_equal(a, b) for a, b in zip(v4, v5))
    value = after["sphenoid"]
    assert value["initial_interior_seed_voxels"] == 0
    assert value["initial_seed_clearance_mm"] == 2 * interval
    assert value["chosen_seed_clearance_mm"] == value["seed_clearance_mm"] == 1.0
    assert value["interior_seed_voxels"] == before["sphenoid"]["interior_seed_voxels"] > 0
    assert value["seed_fallback_used"] is True
    assert value["seed_fallback_reason"] == "legacy_seed_empty_thick_grid"


def test_fallback_still_fails_when_even_one_mm_interior_is_absent():
    args = width_phantom((0.5, 0.5, 2.5), 29.5, 30.5)
    diagnostic = {}
    with pytest.raises(CountingError, match="no spatially resolved interior"):
        localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=diagnostic)
    assert diagnostic["sphenoid"]["seed_fallback_used"] is True
    assert diagnostic["sphenoid"]["interior_seed_voxels"] == 0


@pytest.mark.parametrize("interval", [1.25, 2.0, 2.5])
def test_fallback_does_not_fabricate_missing_anterior_bone(interval):
    args = list(sinus_phantom((0.5, 0.5, interval)))
    args[0][:, 40:, :] = -900
    args[1] = args[0] >= 200
    diagnostic = {}
    with pytest.raises(CountingError, match="supported anterior bony wall") as failure:
        localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=diagnostic)
    assert failure.value.code == "SPHENOID_UNRESOLVED"
    assert diagnostic["sphenoid"]["seed_fallback_used"] is True
    assert diagnostic["sphenoid"]["supported_wall_rays"] == 0


@pytest.mark.parametrize("key,value", [
    ("sphenoid_fallback_clearance_mm", 0.5), ("sphenoid_fallback_clearance_mm", True),
    ("fallback_requires_max_spacing_gt_mm", 0), ("fallback_requires_max_spacing_gt_mm", True),
    ("physical_ray_extent", False), ("sphenoid_seed_mode", "always_one_mm"), ("extra", True),
])
def test_unversioned_policy_changes_are_rejected(key, value):
    policy = {**LOCALIZATION_POLICY, key: value}
    with pytest.raises(CountingError) as failure:
        localization._sphenoid_boundary(*sinus_phantom(), localization_policy=policy)
    assert failure.value.code == "INVALID_CONFIG"


def test_v5_plan_is_distinct_sealed_and_leaves_v4_and_default_protocol_unchanged(tmp_path):
    source, _, _ = synthetic_source(tmp_path)
    profile = build_scan_profile(source)
    baseline = v4_plan(profile)
    before = deepcopy(baseline)
    plan = build_research_plan(profile)
    assert sealed(plan) == plan and plan["version"] == PLAN_VERSION
    assert plan["sha256"] != baseline["sha256"]
    assert plan["source_sha256"] == profile["source_sha256"]
    assert plan["scan_profile_sha256"] == profile["sha256"]
    assert plan["localization_parameters"] == LOCALIZATION_POLICY
    assert plan["counting_grid"] == "original_native"
    assert plan["bone_support_max_mm"] == 1.01
    assert plan["anatomy_proxy_fallback"] == baseline["anatomy_proxy_fallback"]
    assert plan["cell_separation_parameters"] == "frozen_v2_unchanged"
    assert plan["automatic_completion_guaranteed"] is False
    plan["localization_parameters"]["sphenoid_fallback_clearance_mm"] = 99
    assert LOCALIZATION_POLICY["sphenoid_fallback_clearance_mm"] == 1.0
    assert v4_plan(profile) == before
    root = files("ethmoid_pipeline.counting")
    v4 = json.loads(root.joinpath("protocol.research.v4.json").read_text())
    v5 = json.loads(root.joinpath("protocol.research.v5.json").read_text())
    assert v5["protocol_version"] == PROTOCOL_VERSION
    assert v5["scan_adaptation"]["localization_parameters"] == LOCALIZATION_POLICY
    assert v4["scan_adaptation"]["localization_parameters"] == V4_POLICY
    assert v5["algorithm"] == v4["algorithm"] == asdict(CountingConfig())
    assert load_counting_protocol()["protocol_version"] == "ethmoid-count-prototype-v2"
    with pytest.raises(ValueError, match="Unsupported"):
        load_counting_protocol(PROTOCOL_VERSION)


@pytest.mark.parametrize("flipped", [False, True])
def test_v5_actual_engine_preserves_native_ids_hu_and_measurements_through_proxy(tmp_path, monkeypatch, flipped):
    source, hu, affine = synthetic_source(tmp_path, flipped=flipped)
    source_hash = file_sha256(source)
    canonical = nib.as_closest_canonical(nib.load(source))
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.io_orientation(canonical.affine))
    canonical_masks = {name: nib.orientations.apply_orientation(mask, orientation) for name, mask in landmarks(hu.shape).items()}
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **kw: canonical_masks)
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **kw: pytest.fail("Unexpected GPU call"))
    policies = []
    def synthetic_region(values, transform, masks, config, *, localization_policy, diagnostics):
        positions = np.moveaxis(np.indices(values.shape), 0, -1)
        native_positions = nib.affines.apply_affine(np.linalg.inv(affine) @ transform, positions)
        region = np.all((native_positions > [4, 6, 4]) & (native_positions < [43, 33, 27]), axis=-1)
        policies.append(deepcopy(localization_policy))
        return localization.Localization(region, np.zeros(values.shape, bool), {}, 0, [])
    monkeypatch.setattr(localization, "automatic_region", synthetic_region)
    real_grid = research.localization_grid
    def oversized(shape, transform):
        grid = real_grid(shape, transform)
        grid["shape"] = [128_000_001, 1, 1]
        return grid
    monkeypatch.setattr(research, "localization_grid", oversized)
    output = tmp_path / "v5-output"
    plan = build_research_plan(build_scan_profile(source))
    result = research_v5.analyze_research_volume(source, output, tmp_path / "synthetic-cache", processing_plan=plan)
    assert result["status"] == "completed", result
    assert result["method_version"] == METHOD_VERSION and result["protocol_version"] == PROTOCOL_VERSION
    assert result["processing_plan"] == plan
    assert result["analysis_mode"] == "automatic" and result["outcome"] == "automatic_candidate_count"
    assert result["research_status"] == "NOT_ACCURACY_VALIDATED"
    assert result["artifact_verification"]["status"] == "verified"
    assert result["localization_geometry"]["mode"] == "research_bounded_resolution_anatomy"
    assert policies == [LOCALIZATION_POLICY]
    assert file_sha256(source) == source_hash
    expected_cavities, expected_air, expected_cells, _ = compartment_stage(hu, np.ones_like(hu, bool), affine)
    for name, expected in (("cavities", expected_cavities), ("air", expected_air)):
        actual = nib.load(output / result["artifacts"][name])
        assert actual.shape == hu.shape and actual.get_data_dtype() == np.dtype("uint16")
        assert np.array_equal(actual.affine, affine)
        assert np.array_equal(np.asanyarray(actual.dataobj), expected)
    assert result["counts"] == {"left": 1, "right": 1, "total": 2}
    for actual, expected in zip(result["cells"], expected_cells):
        for key in ("id", "side", "voxel_count", "air_voxel_count"):
            assert actual[key] == expected[key]
        assert actual["centroid_ras_mm"] == pytest.approx(expected["centroid_ras_mm"])


def test_v5_padding_is_excluded_before_membership_mapping(monkeypatch):
    hu = np.zeros((18, 14, 10), np.float32)
    affine = affine_for((0.4, 0.7, 0.9), 29)
    def padding_only(values, transform, source_masks, cfg, **options):
        assert options["localization_policy"] == LOCALIZATION_POLICY
        return localization.Localization(values == -1024, np.zeros_like(values, bool), {}, 0, [])
    monkeypatch.setattr(localization, "automatic_region", padding_only)
    with pytest.raises(CountingError) as failure:
        localization.localize_native(hu, affine, landmarks(hu.shape), CountingConfig(), research.research_anatomy_grid(hu.shape, affine),
            localization_policy=LOCALIZATION_POLICY)
    assert failure.value.code == "TARGET_CLIPPED"


@pytest.mark.parametrize("wrong", ["v4_plan", "source", "v4_protocol"])
def test_v5_refuses_cross_version_or_changed_source_plan_before_inference(tmp_path, monkeypatch, wrong):
    source, _, _ = synthetic_source(tmp_path)
    profile = build_scan_profile(source)
    options = {"processing_plan": build_research_plan(profile)}
    if wrong == "v4_plan":
        options["processing_plan"] = v4_plan(profile)
    elif wrong == "source":
        options["processing_plan"] = sealed({**options["processing_plan"], "source_sha256": "0" * 64})
    else:
        options["config"] = json.loads(files("ethmoid_pipeline.counting").joinpath("protocol.research.v4.json").read_text())
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **kw: pytest.fail("Loaded landmarks before verification"))
    monkeypatch.setattr(localization, "run_localizers", lambda *a, **kw: pytest.fail("Inferred before verification"))
    result = research_v5.analyze_research_volume(source, tmp_path / "output", **options)
    assert result["status"] == "failed"
    assert result["failure_code"] == ("INVALID_CONFIG" if wrong == "v4_protocol" else "PROCESSING_PLAN_MISMATCH")
    assert result["counts"] == {"left": None, "right": None, "total": None}
    assert result["artifacts"] == {} and result["cells"] == []


def test_worker_v5_is_explicit_and_disallows_assistance(tmp_path, monkeypatch, capsys):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "failed", "failure_code": "SPHENOID_UNRESOLVED"}
    monkeypatch.setattr(research_v5, "analyze_research_volume", run)
    args = ["--input", str(tmp_path / "source"), "--output", str(tmp_path / "output"), "--method", "research-v5"]
    assert worker.main(args) == 2 and len(calls) == 1
    for flag in ("--assistance", "--model-manifest"):
        assert worker.main(args + [flag, str(tmp_path / "unavailable")]) == 2
    assert len(calls) == 1
    assert "INVALID_CONFIG" in capsys.readouterr().out


def test_localization_spacing_alone_controls_fallback_and_thin_proxy_does_not_inherit_source_thickness():
    args = width_phantom((0.5, 0.5, 1.0), 29, 31)
    diagnostic = {}
    # This represents a <=1mm anatomy proxy of any acquisition. The function
    # deliberately receives no nominal slice thickness or patient identity.
    with pytest.raises(CountingError, match="no spatially resolved interior"):
        localization._sphenoid_boundary(*args, localization_policy=LOCALIZATION_POLICY, diagnostics=diagnostic)
    assert max(diagnostic["sphenoid"]["spacing_mm"]) == 1.0
    assert diagnostic["sphenoid"]["seed_fallback_used"] is False
