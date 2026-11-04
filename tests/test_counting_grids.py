"""Synthetic geometry tests; no patient images or learned inference."""
from itertools import product
import hashlib

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting.engine import CountingError, analyze_volume, compartment_stage
from ethmoid_pipeline.counting.grids import anatomy_proxy, check_proxy_limit, localization_grid
from ethmoid_pipeline.counting.localization import Localization, MASK_TASK, TASKS, load_landmarks, localize_native


def rotated_affine(degrees, spacing=(0.4, 0.6, 2.5)):
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = [-15, -10, -20]
    return affine


@pytest.mark.parametrize("degrees", [0.1, 5, 15, 30])
@pytest.mark.parametrize("interval", [0.625, 1.25, 2.0, 2.5])
def test_proxy_covers_full_half_voxel_extent_without_coarsening(degrees, interval):
    shape = (20, 22, 14)
    affine = rotated_affine(degrees, (0.4, 0.6, interval))
    plan = localization_grid(shape, affine)
    assert plan["mode"] == "anatomy_only_ras_proxy"
    assert np.allclose(plan["spacing_mm"], [0.4] * 3)
    corners = nib.affines.apply_affine(affine, list(product(*[(-0.5, n - 0.5) for n in shape])))
    target_indices = nib.affines.apply_affine(np.linalg.inv(plan["affine"]), corners)
    assert np.all(target_indices >= -0.500001)
    assert np.all(target_indices <= np.asarray(plan["shape"]) - 0.5 + 1e-6)
    assert np.allclose(np.asarray(plan["affine"])[:3, :3], np.diag([0.4] * 3))


def test_proxy_limit_fails_before_array_allocation_without_changing_resolution():
    plan = localization_grid((500, 500, 2000), rotated_affine(30))
    original_step = plan["spacing_mm"][:]
    with pytest.raises(CountingError) as failure:
        check_proxy_limit(plan)
    assert failure.value.code == "RESOURCE_LIMIT"
    assert plan["spacing_mm"] == original_step


def test_linear_proxy_preserves_physical_ramp_and_binary_semantics():
    shape = (24, 20, 14)
    affine = rotated_affine(15, (0.5, 0.6, 2.5))
    ijk = np.indices(shape).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, ijk)
    hu = (3 * world[:, 0] + 4 * world[:, 1] - 2 * world[:, 2] + 100).reshape(shape).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[5:18, 5:15, 4:10] = True
    plan = localization_grid(shape, affine)
    proxy, target, masks, valid = anatomy_proxy(hu, affine, {"example": mask}, plan)
    locations = np.argwhere(valid)
    physical = nib.affines.apply_affine(target, locations)
    native = nib.affines.apply_affine(np.linalg.inv(affine), physical)
    interior = np.all((native >= 1) & (native <= np.asarray(shape) - 2), axis=1)
    positions = locations[interior]
    expected = 3 * physical[interior, 0] + 4 * physical[interior, 1] - 2 * physical[interior, 2] + 100
    assert np.allclose(proxy[tuple(positions.T)], expected, atol=1e-4)
    assert masks["example"].dtype == bool and not masks["example"][~valid].any()
    assert np.all(proxy[~valid] == -1024)


@pytest.mark.parametrize("interval", [1.25, 2.0, 2.5])
def test_thick_series_reaches_localization_and_failure_retains_quality(tmp_path, interval):
    hu = np.full((32, 32, 20), 800, np.float32)
    hu[4:28, 4:28, 4:16] = -900
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(hu, np.diag([0.5, 0.5, interval, 1])), source)
    result = analyze_volume(source, tmp_path / "result", tmp_path / "missing", acquisition={"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "slice_thickness_mm": interval, "spacing_mm": [0.5, 0.5, interval]})
    assert result["failure_code"] == "LANDMARKS_MISSING"
    assert result["quality"]["resolution_class"] == "outside_protocol"
    assert "out_of_protocol_resolution" in result["quality"]["flags"]
    assert result["warnings"] and result["counts"]["total"] is None


@pytest.mark.parametrize("interval", [0.625, 1.25, 2.5])
def test_resolved_native_compartments_keep_identity_when_source_grid_is_rotated(interval):
    # Deliberately large fully enclosed chambers remain spatially resolved at
    # every tested interval; this checks geometry, not small-cell clinical recall.
    hu = np.full((64, 48, 40), 800, np.float32)
    hu[7:29, 8:40, 5:35] = -900
    hu[35:57, 8:40, 5:35] = 35
    roi = np.ones(hu.shape, bool)
    results = []
    for angle in (0, 15):
        affine = rotated_affine(angle, (0.6, 0.6, interval))
        results.append(compartment_stage(hu, roi, affine))
    assert len(results[0][2]) == len(results[1][2]) == 2
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])
    for cell in results[1][2]:
        positions = np.argwhere(results[1][0] == cell["id"])
        assert np.allclose(cell["centroid_ras_mm"], nib.affines.apply_affine(affine, positions.mean(axis=0)))


@pytest.mark.parametrize("flip_x", [False, True])
def test_oblique_native_artifacts_are_never_resampled_instance_ids(tmp_path, monkeypatch, flip_x):
    import ethmoid_pipeline.counting.localization as localization
    hu = np.full((64, 48, 40), 800, np.float32)
    hu[7:29, 8:40, 5:35] = -900
    hu[35:57, 8:40, 5:35] = 35
    affine = rotated_affine(15, (0.6, 0.6, 2.5))
    if flip_x:
        hu = hu[::-1].copy()
        flip = np.eye(4)
        flip[0, 0], flip[0, 3] = -1, hu.shape[0] - 1
        affine = affine @ flip
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), source)
    masks = {name: np.zeros(hu.shape, bool) for name in ("brain", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right")}
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **k: masks)
    def synthetic_anatomy(values, aff, masks, cfg):
        # Membership is a physical box contained well inside the acquired FOV.
        coordinates = np.indices(values.shape).reshape(3, -1).T
        source_ijk = nib.affines.apply_affine(np.linalg.inv(affine) @ aff, coordinates).reshape((*values.shape, 3))
        region = np.all((source_ijk >= [3, 4, 3]) & (source_ijk <= [60, 43, 37]), axis=-1)
        return Localization(region, np.zeros_like(region), {}, 15.0, ["Synthetic anatomical membership only."])
    monkeypatch.setattr(localization, "automatic_region", synthetic_anatomy)
    result = analyze_volume(source, tmp_path / "result", tmp_path / "synthetic", acquisition={"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "slice_thickness_mm": 2.5, "spacing_mm": [0.6, 0.6, 2.5]})
    assert result["status"] == "completed", result
    assert result["counts"]["total"] == 2
    cavity_image = nib.load(tmp_path / "result" / "cavities.nii.gz")
    assert cavity_image.shape == hu.shape and np.allclose(cavity_image.affine, affine)
    cavities = np.asarray(cavity_image.dataobj)
    assert np.array_equal(cavities > 0, hu < 200)
    assert all("out_of_protocol_resolution" in cell["reasons"] for cell in result["cells"])
    assert result["localization_geometry"]["mode"] == "anatomy_only_ras_proxy"


def test_proxy_padding_cannot_become_counting_membership(monkeypatch):
    import ethmoid_pipeline.counting.localization as localization
    from ethmoid_pipeline.counting.engine import CountingConfig
    hu = np.full((12, 12, 12), -900, np.float32)
    affine = rotated_affine(30, (0.6, 0.6, 1.25))
    plan = localization_grid(hu.shape, affine)
    monkeypatch.setattr(localization, "automatic_region", lambda values, *args: Localization(np.ones(values.shape, bool), np.zeros(values.shape, bool), {}, 0, []))
    with pytest.raises(CountingError) as failure:
        localize_native(hu, affine, {}, CountingConfig(), plan)
    assert failure.value.code == "TARGET_CLIPPED"


def test_coarse_spacing_cannot_certify_bone_several_millimeters_beyond_roi():
    hu = np.full((60, 60, 60), 800, np.float32)
    hu[5:55, 5:55, 5:55] = 35
    # Artificial membership cuts through tissue 2mm from the true lateral bone.
    # A 2.5mm sampling interval must not make those cuts supported cavity walls.
    region = np.zeros_like(hu, bool)
    region[8:52, 8:52, 5:55] = True
    with pytest.raises(CountingError) as failure:
        compartment_stage(hu, region, np.diag([0.5, 0.5, 2.5, 1]))
    assert failure.value.code == "NO_SUPPORTED_CELLS"


def test_slice_averaging_cannot_recover_a_septum_missing_from_coarse_samples():
    fine = np.full((52, 52, 80), 800, np.float32)
    fine[6:46, 6:46, 5:75] = -900
    fine[6:46, 6:46, 40] = 800
    first = compartment_stage(fine, np.ones_like(fine, bool), np.diag([0.5, 0.5, 0.5, 1]))
    assert len(first[2]) == 2
    coarse = fine.reshape(52, 52, 16, 5).mean(axis=3)
    assert coarse[20, 20, 8] < 200  # Five-slice averaging removes wall evidence.
    try:
        observed = compartment_stage(coarse, np.ones_like(coarse, bool), np.diag([0.5, 0.5, 2.5, 1]))
    except CountingError as failure:
        assert failure.code == "NO_SUPPORTED_CELLS"
    else:
        assert len(observed[2]) == 1


@pytest.mark.parametrize("acquisition", [{"slice_thickness_mm": 2.5}, {"source_sha256": "b" * 64, "slice_thickness_mm": 2.5}])
def test_worker_refuses_unbound_or_wrong_source_acquisition(tmp_path, acquisition):
    hu = np.full((24, 24, 24), 800, np.float32)
    hu[4:20, 4:20, 4:20] = -900
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(hu, np.eye(4)), source)
    result = analyze_volume(source, tmp_path / "result", tmp_path / "missing", acquisition=acquisition)
    assert result["failure_code"] == "INVALID_ACQUISITION"
    assert result["counts"]["total"] is None


@pytest.mark.parametrize("origin,translation_error_mm,accepted", [
    (0.0, 0.00005, True), (0.0, 0.0002, False),
    (1000.0, 0.005, False), (1_000_000.0, 0.0, True), (1_000_000.0, 2.0, False),
])
def test_bound_landmark_grid_tolerance_is_absolute(tmp_path, origin, translation_error_mm, accepted):
    import json

    shape = (8, 8, 8)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [origin, -origin, origin]
    source_path = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), source_path)
    source = nib.load(source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    mask_affine = source.affine.copy()
    mask_affine[0, 3] += translation_error_mm
    mask = np.zeros(shape, np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    entries = {task: [] for task in TASKS}
    for name, task in MASK_TASK.items():
        folder = tmp_path / task
        folder.mkdir(exist_ok=True)
        path = folder / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask, mask_affine), path)
        entries[task].append({"class": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"source_sha256": source_sha256,
        "tasks": [{"task": task, "masks": entries[task]} for task in TASKS]}
    (tmp_path / "localizers.local.json").write_text(json.dumps(manifest), encoding="utf-8")
    if accepted:
        loaded = load_landmarks(tmp_path, source, source_sha256=source_sha256)
        assert set(loaded) == set(MASK_TASK)
        assert all(np.array_equal(value, mask.astype(bool)) for value in loaded.values())
    else:
        with pytest.raises(CountingError) as failure:
            load_landmarks(tmp_path, source, source_sha256=source_sha256)
        assert failure.value.code == "LANDMARK_GEOMETRY"
