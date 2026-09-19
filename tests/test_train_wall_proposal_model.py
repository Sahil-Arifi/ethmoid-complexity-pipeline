"""Scientific split/provenance and bounded preparation acceptance tests."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_spec = importlib.util.spec_from_file_location("_wall_train_command",
    Path(__file__).resolve().parents[1] / "scripts/train_wall_proposal_model.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_preparation_preserves_labels_determinism_and_distinct_voxels():
    geometry = runner.make_geometry(3, "smooth_bowl", seed=34)
    config = runner.TrainingConfig(seed=65)
    first, records = runner.prepare_partition([geometry], config, candidates_per_scene=16)
    second, other = runner.prepare_partition([geometry], config, candidates_per_scene=16)
    assert runner.patch_set_sha256(*first) == runner.patch_set_sha256(*second)
    assert records == other
    assert set(np.unique(first[1])) == {0, 1}
    offset = 0
    for row, augmentation in zip(records, runner.augmentations(config.seed)):
        assert row["status"] == "sampled"
        chosen = row["selected_candidate_indices"]
        assert len(set(chosen)) == len(chosen)
        scene = runner.render_scene(geometry, augmentation)
        candidates = runner.extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
        independent = runner.candidate_targets(scene, candidates.positions_ijk)[chosen]
        np.testing.assert_array_equal(first[1][offset:offset + len(chosen)], independent)
        offset += len(chosen)
    assert first[2] == [geometry.geometry_id] * offset
    assert first[0].shape == (offset, 3, 17, 17, 17)
    assert np.isin(first[0][:, 1:], (0, 1)).all()
    assert np.all(first[0][:, 1] <= first[0][:, 2])


def test_dataset_budget_checks_before_any_scene_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A rejected allocation must not render a scene")
    monkeypatch.setattr(runner, "render_scene", forbidden)
    geometry = runner.make_geometry(0)
    with pytest.raises(ValueError, match="budget"):
        runner.prepare_partition([geometry] * 1000, runner.TrainingConfig(), candidates_per_scene=256)
    with pytest.raises(ValueError, match="budget"):
        runner.prepare_partition([geometry] * 50, runner.TrainingConfig(patch_size=33), candidates_per_scene=96)


@pytest.mark.parametrize("size", [0, 7, 257, 1.5, True])
def test_candidate_sample_bound(size):
    with pytest.raises(ValueError):
        runner.prepare_partition([], runner.TrainingConfig(), candidates_per_scene=size)


def test_declared_experiment_families_are_disjoint_before_augmentation():
    geometries = [runner.make_geometry(i, family, seed=20260918)
        for family in runner.GEOMETRY_FAMILIES for i in range(4)]
    split = runner.split_geometry_families(geometries, seed=20260918)
    runner.require_disjoint_geometries(*[[g.geometry_id for g in values] for values in split.values()])
    runner.require_disjoint_geometries(*[[g.family for g in values] for values in split.values()])
    assert sum(map(len, split.values())) == 32
    assert all(split.values())


def _stub_clean_git(monkeypatch):
    def git(command, **kwargs):
        if command[-2:] == ["status", "--porcelain"]:
            return ""
        assert command[-2:] == ["rev-parse", "HEAD"]
        return "a" * 40 + "\n"
    monkeypatch.setattr(runner.subprocess, "check_output", git)


@pytest.mark.parametrize("name", [
    "wall_learning", "synthetic_walls", "wall_proposals", "wall_benchmark", "localization",
])
def test_frozen_runner_rejects_same_named_module_from_another_checkout(tmp_path, monkeypatch, name):
    _stub_clean_git(monkeypatch)
    repo = Path(runner.__file__).resolve().parents[1]
    module = runner.importlib.import_module(f"ethmoid_pipeline.counting.{name}")
    # An inherited editable-install path can resolve the same import name to a
    # different checkout even while the runner's own Git tree is clean.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "other-checkout" / f"{name}.py"))
    def forbidden_hash(*args, **kwargs):
        pytest.fail("Incorrect module origins must be rejected before attesting file hashes")
    monkeypatch.setattr(runner, "file_sha256", forbidden_hash)
    with pytest.raises(ValueError, match="Imported experiment module"):
        runner.freeze_sources(repo)


@pytest.mark.parametrize("dirty", [" M src/ethmoid_pipeline/counting/wall_learning.py\n", "?? untracked.py\n"])
def test_frozen_runner_rejects_modified_and_untracked_source_before_hashing(monkeypatch, dirty):
    def git(command, **kwargs):
        assert command[-2:] == ["status", "--porcelain"]
        return dirty
    def forbidden_hash(*args, **kwargs):
        pytest.fail("Dirty source must never receive a frozen-source attestation")
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    monkeypatch.setattr(runner, "file_sha256", forbidden_hash)
    with pytest.raises(ValueError, match="clean committed"):
        runner.freeze_sources(Path(runner.__file__).resolve().parents[1])


def test_frozen_source_attestation_binds_actual_loaded_modules_and_rechecks_changes(monkeypatch):
    import hashlib

    _stub_clean_git(monkeypatch)
    repo = Path(runner.__file__).resolve().parents[1]
    frozen = runner.freeze_sources(repo)
    expected_files = ["scripts/train_wall_proposal_model.py"] + [
        f"src/ethmoid_pipeline/counting/{name}.py" for name in (
            "wall_learning", "synthetic_walls", "wall_proposals", "wall_benchmark", "localization",
        )
    ]
    assert frozen["git_commit"] == "a" * 40
    assert frozen["source_sha256"] == {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in expected_files
    }
    runner.verify_sources(repo, frozen)
    real_hash = runner.file_sha256
    monkeypatch.setattr(runner, "file_sha256", lambda path:
        "0" * 64 if path.name == "wall_learning.py" else real_hash(path))
    with pytest.raises(RuntimeError, match="Source changed"):
        runner.verify_sources(repo, frozen)
