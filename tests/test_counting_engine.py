"""Geometric algorithm validation; these phantoms are not clinical accuracy tests."""

from dataclasses import asdict
import json

import nibabel as nib
import numpy as np
import pytest

# Application Python deliberately excludes the separate inference stack.
# These tests are executed, without skips, in the pinned inference runtime.
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting.engine import CountingConfig, CountingError, analyze_volume, compartment_stage
from ethmoid_pipeline.counting.localization import Localization, TASKS, _closed_nasal_pockets, _continuous_wall, _intracranial_roof, automatic_region, cache_producer_warnings, load_landmarks


def volume(shape=(52, 44, 40), spacing=(0.6, 0.6, 0.6)):
    hu = np.full(shape, 800.0, dtype=np.float32)
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) * np.array(spacing) / 2
    return hu, np.ones(shape, dtype=bool), affine


def run(hu, roi, affine, **kwargs):
    return compartment_stage(hu, roi, affine, **kwargs)


def assert_consistent(cavities, air, cells):
    ids = set(np.unique(cavities)) - {0}
    assert ids == {cell["id"] for cell in cells}
    assert cavities.dtype == air.dtype == np.uint16
    assert np.all((air == 0) | (air == cavities))
    for cell in cells:
        assert cell["voxel_count"] == np.count_nonzero(cavities == cell["id"])
        assert cell["air_voxel_count"] == np.count_nonzero(air == cell["id"])


def test_one_cell_is_counted_once_across_many_planes():
    hu, roi, affine = volume()
    hu[12:36, 10:32, 8:30] = -900
    cavities, air, cells, _ = run(hu, roi, affine)
    assert len(cells) == 1
    assert cells[0]["voxel_count"] == 24 * 22 * 22
    assert_consistent(cavities, air, cells)


def test_two_bone_separated_chambers_are_two_cells():
    hu, roi, affine = volume()
    hu[6:24, 10:32, 8:30] = -900
    hu[25:45, 10:32, 8:30] = -900
    cavities, air, cells, _ = run(hu, roi, affine)
    assert len(cells) == 2
    assert {c["side"] for c in cells} == {"left", "right"}
    assert cells[0]["centroid_ras_mm"][0] < 0 and cells[0]["side"] == "left"
    assert cells[1]["centroid_ras_mm"][0] > 0 and cells[1]["side"] == "right"
    assert_consistent(cavities, air, cells)


def test_communicating_lobes_without_a_septum_merge():
    hu, roi, affine = volume()
    grid = np.indices(hu.shape)
    left = sum((grid[i] - v) ** 2 for i, v in enumerate((18, 22, 20))) < 11 ** 2
    right = sum((grid[i] - v) ** 2 for i, v in enumerate((33, 22, 20))) < 11 ** 2
    hu[left | right] = -900
    cavities, air, cells, _ = run(hu, roi, affine)
    assert len(cells) == 1
    assert_consistent(cavities, air, cells)


def test_incomplete_thin_septum_preserves_two_uncertain_cells():
    hu, roi, affine = volume()
    hu[6:45, 9:35, 7:33] = -900
    hu[25, 9:35, 7:33] = 800
    hu[25, 20:23, 18:21] = -900  # A 1.8 mm opening in a 0.6 mm plate.
    cavities, air, cells, _ = run(hu, roi, affine)
    assert len(cells) == 2
    assert all("incomplete_septum_inferred_from_bone_plate" in c["reasons"] for c in cells)
    assert_consistent(cavities, air, cells)


@pytest.mark.parametrize("opacity", ["partial", "full"])
def test_opacity_does_not_remove_a_bone_enclosed_cell(opacity):
    hu, roi, affine = volume()
    hu[12:36, 10:32, 8:30] = 35
    if opacity == "partial":
        hu[12:20, 10:32, 8:30] = -900
        hu[28:36, 10:32, 8:30] = -900
    cavities, air, cells, _ = run(hu, roi, affine)
    assert len(cells) == 1
    assert cells[0]["voxel_count"] == 24 * 22 * 22
    if opacity == "full":
        assert not air.any()
        assert "fully_opacified_candidate_requires_review" in cells[0]["reasons"]
    else:
        assert cells[0]["air_voxel_count"] == 16 * 22 * 22
    assert_consistent(cavities, air, cells)


def test_variant_extension_can_count_a_separate_compartment():
    hu, _, affine = volume()
    hu[10:22, 10:24, 8:22] = -900
    hu[30:42, 26:38, 24:35] = -900
    core = np.zeros(hu.shape, dtype=bool)
    core[8:24, 8:26, 6:24] = True
    extension = np.zeros_like(core)
    extension[28:44, 24:40, 22:37] = True
    cavities, air, cells, _ = run(hu, core | extension, affine, variant_regions={"haller": extension})
    assert len(cells) == 2
    assert sum("possible_haller_variant_requires_review" in c["reasons"] for c in cells) == 1
    assert_consistent(cavities, air, cells)


def test_roi_face_cannot_invent_an_opaque_cell():
    hu = np.full((24, 24, 24), 35, dtype=np.float32)
    with pytest.raises(CountingError, match="not a zero-cell count"):
        run(hu, np.ones_like(hu, dtype=bool), np.eye(4))


def test_physical_ids_laterality_and_centroids_survive_axis_flip():
    hu, roi, affine = volume(spacing=(0.4, 0.6, 0.9))
    hu[6:20, 8:30, 6:28] = -900
    hu[29:44, 10:32, 8:30] = -900
    cavities, _, cells, _ = run(hu, roi, affine)
    transform = np.eye(4)
    transform[0, 0] = -1
    transform[0, 3] = hu.shape[0] - 1
    flipped, _, flipped_cells, _ = run(hu[::-1], roi[::-1], affine @ transform)
    assert np.array_equal(cavities, flipped[::-1])
    assert len(cells) == len(flipped_cells) == 2
    assert [c["side"] for c in cells] == [c["side"] for c in flipped_cells]
    assert np.allclose([c["centroid_ras_mm"] for c in cells], [c["centroid_ras_mm"] for c in flipped_cells])


def test_air_threshold_changes_content_not_bone_compartment_identity():
    hu, roi, affine = volume()
    hu[12:36, 10:32, 8:30] = -550
    low = run(hu, roi, affine, config={"air_hu": -600})
    high = run(hu, roi, affine, config={"air_hu": -400})
    assert np.array_equal(low[0], high[0])
    assert not low[1].any()
    assert high[1].any()


def test_missing_localizers_fail_without_count_or_artifacts(tmp_path):
    hu, _, affine = volume()
    hu[12:36, 10:32, 8:30] = -900
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), source)
    destination = tmp_path / "result"
    result = analyze_volume(source, destination, tmp_path / "absent_landmarks")
    assert result["status"] == "failed"
    assert result["failure_code"] == "LANDMARKS_MISSING"
    assert result["counts"] == {"left": None, "right": None, "total": None}
    assert result["cells"] == []
    assert result["artifacts"] == {}
    assert json.loads((destination / "result.json").read_text()) == result


def test_existing_result_is_never_overwritten(tmp_path):
    destination = tmp_path / "result"
    destination.mkdir()
    original = destination / "result.json"
    original.write_text("original", encoding="utf-8")
    with pytest.raises(CountingError, match="fresh or empty"):
        analyze_volume(tmp_path / "absent.nii.gz", destination)
    assert original.read_text() == "original"


@pytest.mark.parametrize("configuration", [{"bone_hu": float("nan")}, {"air_hu": 30}, {"minimum_voxels": 1.2}, {"unknown": 1}])
def test_invalid_fixed_config_rejected(configuration):
    with pytest.raises(CountingError):
        CountingConfig.parse(configuration)


def test_default_config_is_serializable_and_round_trips():
    cfg = CountingConfig()
    assert CountingConfig.parse(json.loads(json.dumps(asdict(cfg)))) == cfg


def test_automatic_localization_requires_real_landmark_evidence():
    hu, _, affine = volume()
    with pytest.raises(CountingError, match="required anatomical structure"):
        automatic_region(hu, affine, {}, CountingConfig())


def test_landmark_loader_rejects_a_different_physical_grid(tmp_path):
    source = nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.uint8), np.eye(4))
    translated = np.eye(4)
    translated[0, 3] = 10
    nib.save(nib.Nifti1Image(np.ones(source.shape, dtype=np.uint8), translated), tmp_path / "skull.nii.gz")
    with pytest.raises(CountingError, match="does not match"):
        load_landmarks(tmp_path, source)


def test_cache_from_another_source_is_rejected_even_with_same_geometry(tmp_path):
    source = nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.uint8), np.eye(4))
    (tmp_path / "localizers.local.json").write_text(json.dumps({"source_sha256": "a" * 64, "tasks": []}), encoding="utf-8")
    with pytest.raises(CountingError) as failure:
        load_landmarks(tmp_path, source, source_sha256="b" * 64)
    assert failure.value.code == "LANDMARK_SOURCE"


def test_single_voxel_thick_opaque_pore_is_not_a_resolved_cell():
    hu, roi, affine = volume()
    hu[12:30, 20, 20] = 35
    with pytest.raises(CountingError, match="not a zero-cell count"):
        run(hu, roi, affine)


def test_real_algorithm_artifacts_return_to_original_flipped_grid(tmp_path, monkeypatch):
    """Inject only synthetic anatomical bounds; exercise actual counting and I/O."""
    import ethmoid_pipeline.counting.localization as localization

    hu, roi, affine = volume(spacing=(0.4, 0.6, 0.9))
    hu[6:20, 8:30, 6:28] = -900
    hu[29:44, 10:32, 8:30] = 35
    transform = np.eye(4)
    transform[0, 0] = -1
    transform[0, 3] = hu.shape[0] - 1
    native_affine = affine @ transform
    source = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(hu[::-1], native_affine), source)
    monkeypatch.setattr(localization, "load_landmarks", lambda *args, **kwargs: {})
    monkeypatch.setattr(localization, "automatic_region", lambda values, aff, masks, cfg: Localization(np.ones(values.shape, bool), np.zeros(values.shape, bool), {}, 0.0, ["Synthetic anatomical bounds for I/O regression only."]))
    destination = tmp_path / "result"
    result = analyze_volume(source, destination, landmarks_dir=tmp_path / "synthetic_bounds")
    assert result["status"] == "completed"
    assert result["counts"] == {"left": 1, "right": 1, "total": 2}
    loaded = {}
    for name in ("cavities", "air", "region"):
        artifact = nib.load(destination / result["artifacts"][name])
        assert artifact.shape == hu.shape
        assert np.allclose(artifact.affine, native_affine)
        assert artifact.get_data_dtype() == (np.dtype("uint8") if name == "region" else np.dtype("uint16"))
        loaded[name] = np.asanyarray(artifact.dataobj)
    assert_consistent(loaded["cavities"], loaded["air"], result["cells"])
    for cell in result["cells"]:
        expected = 1 if cell["side"] == "left" else 2
        assert np.all(loaded["region"][loaded["cavities"] == cell["id"]] == expected)
    assert set(np.unique(loaded["air"])) == {0, 1}


@pytest.mark.parametrize("degrees", [0.1, 15])
def test_oblique_localization_fails_before_assigning_anatomical_bounds(degrees):
    hu, _, affine = volume()
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine[:3, :3] = rotation @ affine[:3, :3]
    with pytest.raises(CountingError) as failure:
        automatic_region(hu, affine, {}, CountingConfig())
    assert failure.value.code == "OBLIQUE_GRID_UNSUPPORTED"


def test_low_coverage_withholds_count_and_registry(tmp_path, monkeypatch):
    import ethmoid_pipeline.counting.localization as localization

    hu = np.full((60, 60, 60), -900, dtype=np.float32)
    hu[12:30, 12:30, 12:30] = 800
    hu[14:28, 14:28, 14:28] = -900
    source = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(hu, np.diag([0.6, 0.6, 0.6, 1])), source)
    monkeypatch.setattr(localization, "load_landmarks", lambda *args, **kwargs: {})
    monkeypatch.setattr(localization, "automatic_region", lambda values, aff, masks, cfg: Localization(np.ones(values.shape, bool), np.zeros(values.shape, bool), {}, 0.0, []))
    result = analyze_volume(source, tmp_path / "result", landmarks_dir=tmp_path / "synthetic_bounds")
    assert result["status"] == "failed"
    assert result["failure_code"] == "INSUFFICIENT_COVERAGE"
    assert result["counts"] == {"left": None, "right": None, "total": None}
    assert result["cells"] == [] and result["artifacts"] == {}
    assert (tmp_path / "result" / "diagnostic-candidates.local.json").is_file()
    diagnostics = result["diagnostics"]
    assert diagnostics["air_coverage_fraction"] == diagnostics["assigned_air_voxels"] / diagnostics["observed_air_voxels"]
    assert diagnostics["air_coverage_fraction"] < 0.5


def test_gpu_out_of_memory_retries_once_on_cpu(tmp_path, monkeypatch):
    """Control-path test; this does not claim a real GPU exhaustion occurred."""
    import sys
    from types import SimpleNamespace
    from ethmoid_pipeline.counting.localization import _predict_task

    devices = []

    class SyntheticOutOfMemory(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=SyntheticOutOfMemory, is_available=lambda: False)))

    def inference(**kwargs):
        devices.append(kwargs["device"])
        if kwargs["device"] == "gpu":
            raise SyntheticOutOfMemory("synthetic out of memory")

    api = SimpleNamespace(totalsegmentator=inference, get_task_config=lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", api)
    result = _predict_task(tmp_path / "source", tmp_path / "outputs", "craniofacial_structures")
    assert devices == ["gpu", "cpu"]
    assert result == {"device": "cpu", "cpu_fallback": True, "full_extent": False}


def test_non_memory_localizer_error_does_not_silently_retry(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from ethmoid_pipeline.counting.localization import _predict_task

    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError, is_available=lambda: False)))

    def inference(**kwargs):
        calls.append(kwargs["device"])
        raise RuntimeError("synthetic invalid model")

    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", SimpleNamespace(totalsegmentator=inference, get_task_config=lambda *args, **kwargs: {}))
    with pytest.raises(RuntimeError, match="invalid model"):
        _predict_task(tmp_path / "source", tmp_path / "outputs", "craniofacial_structures")
    assert calls == ["gpu"]


def test_missing_essential_anatomy_triggers_only_one_full_extent_retry(tmp_path, monkeypatch):
    import ethmoid_pipeline.counting.localization as localization

    hu, _, affine = volume()
    hu[12:36, 10:32, 8:30] = -900
    source = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), source)
    calls = []
    monkeypatch.setattr(localization, "load_landmarks", lambda *args, **kwargs: {})
    monkeypatch.setattr(localization, "run_localizers", lambda source, destination, **kwargs: calls.append(kwargs))

    def absent(*args):
        raise CountingError("LANDMARK_EMPTY", "A required anatomical structure was not detected.")

    monkeypatch.setattr(localization, "automatic_region", absent)
    result = analyze_volume(source, tmp_path / "result", landmarks_dir=tmp_path / "synthetic_bounds")
    assert calls == [{"full_extent": True}]
    assert result["status"] == "failed" and result["failure_code"] == "LANDMARK_EMPTY"


def test_curved_bony_wall_is_not_rejected_for_global_anterior_variation():
    xx, zz = np.indices((50, 50))
    truth = np.rint(35 + 0.025 * (xx - 25) ** 2 + 0.4 * zz).astype(np.int32)
    noisy = truth.copy()
    noisy[::6, ::5] += 40  # Sparse irrelevant bony hits must not dictate the surface.
    fit = _continuous_wall(noisy, np.ones_like(truth, dtype=bool), np.array([0.6, 0.6, 0.6]))
    assert np.mean(np.abs(fit - truth)) < 1
    assert np.ptp(fit) > 20


def test_spatially_incoherent_wall_evidence_abstains():
    rng = np.random.default_rng(55)
    wall = rng.integers(5, 100, (40, 40))
    with pytest.raises(CountingError) as failure:
        _continuous_wall(wall, np.ones(wall.shape, bool), np.array([0.6, 0.6, 0.6]))
    assert failure.value.code == "SPHENOID_AMBIGUOUS"


def test_membership_plane_does_not_truncate_reconstructed_bony_cavity():
    hu, _, affine = volume()
    hu[12:36, 10:32, 8:30] = -900
    hu[40:48, 12:30, 10:28] = -900  # Outside the target anatomy membership.
    membership = np.zeros(hu.shape, dtype=bool)
    membership[:33] = True
    cavities, air, cells, _ = run(hu, membership, affine, reconstruction_region=np.ones(hu.shape, dtype=bool))
    assert len(cells) == 1
    assert cells[0]["voxel_count"] == 24 * 22 * 22
    assert np.all(cavities[12:36, 10:32, 8:30] == 1)
    assert not cavities[40:48].any()
    assert "cavity_reconstructed_beyond_membership_boundary" in cells[0]["reasons"]
    assert_consistent(cavities, air, cells)


def test_brain_prior_roof_is_refined_to_native_bone():
    hu = np.full((40, 40, 50), 35, dtype=np.float32)
    hu[:, :, 30] = 800
    hu[:, :, 45] = 800  # A higher bone plane must not become the skull base.
    brain = np.zeros(hu.shape, dtype=bool)
    brain[:, :, 33:] = True  # Coarse segmentation is 1.8 mm above the true bone.
    nasal = np.zeros_like(brain)
    nasal[10:30, 10:30, 10:25] = True
    roof, returned_brain = _intracranial_roof(hu, hu >= 200, brain, nasal, np.array([0.6, 0.6, 0.6]))
    assert np.all(roof == 30)
    assert np.array_equal(returned_brain, brain)


def test_concha_candidate_requires_separation_from_nasal_parent():
    nasal = np.zeros((40, 40, 40), dtype=bool)
    nasal[2:38, 2:38, 2:38] = True
    bone = np.zeros_like(nasal)
    bone[12:25, 12:25, 12:25] = True
    bone[13:24, 13:24, 13:24] = False
    closed = _closed_nasal_pockets(nasal, bone, np.zeros_like(nasal))
    assert closed.sum() == 11 ** 3
    assert np.all(closed[13:24, 13:24, 13:24])
    bone[12, 18, 18] = False  # Unresolved communication: cannot call it separate.
    assert not _closed_nasal_pockets(nasal, bone, np.zeros_like(nasal)).any()


def test_cache_producer_binding_does_not_attribute_current_weights_to_legacy_masks(tmp_path, monkeypatch):
    import hashlib
    import ethmoid_pipeline.counting.localization as localization

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    manifest = runtime / "weights-manifest.json"
    manifest.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(localization.sys, "prefix", str(runtime))
    monkeypatch.setattr(localization.metadata, "version", lambda name: "2.18.0")
    cache = tmp_path / "cache"
    cache.mkdir()
    provenance = {"totalsegmentator_version": "2.18.0", "tasks": [{"task": task} for task in TASKS]}
    path = cache / "localizers.local.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    assert "Legacy" in cache_producer_warnings(cache)[0]
    provenance["weights_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    path.write_text(json.dumps(provenance), encoding="utf-8")
    assert cache_producer_warnings(cache) == []
    provenance["weights_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(CountingError) as failure:
        cache_producer_warnings(cache)
    assert failure.value.code == "LANDMARK_PROVENANCE"
