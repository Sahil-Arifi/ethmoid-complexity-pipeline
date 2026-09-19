"""Independent contracts for quarantined, synthetic-only wall training."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ethmoid_pipeline.counting import wall_learning as learning


def _patches(prefix: str, count: int = 8, *, seed: int = 17):
    rng = np.random.default_rng(seed)
    labels = (np.arange(count) % 2).astype(np.float32)
    values = np.zeros((count, 3, 9, 9, 9), dtype=np.float32)
    values[:, 0] = (labels[:, None, None, None] * 1.6 - 0.8)
    values[:, 0] += rng.normal(0, 0.01, (count, 9, 9, 9)).astype(np.float32)
    values[:, 1, 4, 4, 4] = 1
    values[:, 2] = 1
    return values, labels, [f"{prefix}-geometry-{i // 2}" for i in range(count)]


def _new_release(tmp_path):
    work = tmp_path / "protected"
    work.mkdir()
    return work, learning.protected_new_directory(work / "experiment", work)


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", -1), ("seed", 2**32), ("epochs", 0),
    ("epochs", 1001), ("epochs", 1.5), ("batch_size", 0), ("batch_size", 257),
    ("patch_size", 8), ("patch_size", 10), ("patch_size", 35),
    ("cpu_threads", 0), ("cpu_threads", 9), ("learning_rate", 0),
    ("learning_rate", float("nan")), ("learning_rate", float("inf")),
    ("learning_rate", True), ("weight_decay", -0.1), ("weight_decay", 1.1),
    ("patch_spacing_mm", 0.1), ("patch_spacing_mm", 2.1),
])
def test_training_config_rejects_unbounded_or_ambiguous_values(field, value):
    with pytest.raises(ValueError):
        replace(learning.TrainingConfig(), **{field: value}).validate()


def test_physical_patch_spacing_is_preserved_in_valid_config():
    config = learning.TrainingConfig(patch_size=9, patch_spacing_mm=0.625)
    config.validate()
    assert config.patch_spacing_mm == 0.625


def test_json_fingerprints_ignore_key_order_but_reject_nonfinite_evidence():
    left = {"spacing": [0.625, 1.25], "policy": {"b": 2, "a": 1}}
    reordered = {"policy": {"a": 1, "b": 2}, "spacing": [0.625, 1.25]}
    assert learning.canonical_sha256(left) == learning.canonical_sha256(reordered)
    assert learning.canonical_sha256(left) != learning.canonical_sha256({**left, "spacing": [1.25, 0.625]})
    with pytest.raises(ValueError):
        learning.canonical_sha256({"loss": float("nan")})


def test_file_hash_detects_a_changed_byte_across_read_blocks(tmp_path):
    path = tmp_path / "weights.bin"
    data = b"a" * (1024 * 1024) + b"b"
    path.write_bytes(data)
    original = learning.file_sha256(path)
    assert original == hashlib.sha256(data).hexdigest()
    path.write_bytes(data[:-1] + b"c")
    assert learning.file_sha256(path) != original


def test_patch_hash_is_layout_independent_but_binds_values_labels_and_geometry_order():
    patches, labels, identities = _patches("train")
    fingerprint = learning.patch_set_sha256(patches, labels, identities)
    assert learning.patch_set_sha256(np.asfortranarray(patches), labels.copy(), list(identities)) == fingerprint
    altered = patches.copy()
    altered[0, 0, 0, 0, 0] += 0.01
    assert learning.patch_set_sha256(altered, labels, identities) != fingerprint
    assert learning.patch_set_sha256(patches.astype(np.float64), labels, identities) != fingerprint
    assert learning.patch_set_sha256(patches, 1 - labels, identities) != fingerprint
    assert learning.patch_set_sha256(patches, labels, identities[::-1]) != fingerprint


def test_normalization_preserves_source_and_separates_hu_from_binary_channels():
    raw = np.zeros((3, 9, 9, 9), dtype=np.float64)
    raw[0, 0, 0, :5] = [-2000, -1000, 0, 1000, 3000]
    raw[1, 4, 4, 4] = 1
    raw[2] = 1
    original = raw.copy()
    result = learning.normalize_patch(raw)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0, 0, :5], [-1, -1, 0, 1, 2])
    np.testing.assert_array_equal(result[1:], raw[1:])
    np.testing.assert_array_equal(raw, original)
    assert not np.shares_memory(raw, result)


@pytest.mark.parametrize("channel,value", [(0, np.nan), (1, 0.5), (2, -1)])
def test_normalization_rejects_invalid_hu_or_mask_channels(channel, value):
    raw = np.zeros((3, 9, 9, 9), dtype=np.float32)
    raw[channel, 0, 0, 0] = value
    with pytest.raises(ValueError):
        learning.normalize_patch(raw)


def test_new_json_preserves_existing_evidence_and_rejects_nan_before_creation(tmp_path):
    path = tmp_path / "report.json"
    learning.write_new_json(path, {"clinical_accuracy_measured": False, "values": [1, 2]})
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        learning.write_new_json(path, {"overwritten": True})
    assert path.read_bytes() == first
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        learning.write_new_json(invalid, {"metric": float("inf")})
    assert not invalid.exists()


def test_output_directory_cannot_escape_or_overwrite_protected_root(tmp_path):
    work, output = _new_release(tmp_path)
    assert output.is_dir()
    for invalid in (work, output, work / ".." / "escaped"):
        with pytest.raises(ValueError):
            learning.protected_new_directory(invalid, work)
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("git_marker", ["directory", "worktree_file"])
def test_output_rejects_both_git_repositories_and_linked_worktrees(tmp_path, git_marker):
    work = tmp_path / "protected"
    checkout = work / "checkout"
    checkout.mkdir(parents=True)
    marker = checkout / ".git"
    if git_marker == "directory":
        marker.mkdir()
    else:
        marker.write_text("gitdir: somewhere-else\n")
    output = checkout / "releases" / "model"
    with pytest.raises(ValueError, match="Git"):
        learning.protected_new_directory(output, work)
    assert not output.exists()


def test_output_symlink_cannot_redirect_training_outside_work_root(tmp_path):
    work = tmp_path / "protected"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    link = work / "redirect"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not allow creation of directory symlinks")
    with pytest.raises(ValueError, match="protected"):
        learning.protected_new_directory(link / "model", work)
    assert not (outside / "model").exists()


def test_partitions_cannot_share_a_geometry_through_augmentation():
    # Different patches from one procedural anatomy must stay in one partition.
    learning.require_disjoint_geometries(["geometry-a"] * 3, ["geometry-b"] * 2)
    with pytest.raises(ValueError, match="partitions"):
        learning.require_disjoint_geometries(["geometry-a"], ["geometry-b"], ["geometry-a"])


@pytest.mark.parametrize("defect", ["channel_count", "nonfinite", "integer_hu", "no_negative", "invalid_target", "missing_identity", "empty_identity"])
def test_patch_validation_rejects_unusable_supervision(defect):
    patches, labels, identities = _patches("train")
    if defect == "channel_count":
        patches = patches[:, :2]
    elif defect == "nonfinite":
        patches[0, 0, 0, 0, 0] = np.nan
    elif defect == "integer_hu":
        patches = patches.astype(np.int16)
    elif defect == "no_negative":
        labels[:] = 1
    elif defect == "invalid_target":
        labels[0] = 0.5
    elif defect == "missing_identity":
        identities = identities[:-1]
    elif defect == "empty_identity":
        identities[0] = ""
    with pytest.raises(ValueError):
        learning.validate_patch_set(patches, labels, identities, patch_size=9)


def test_valid_patches_keep_noncontiguous_native_values_unchanged():
    patches, labels, identities = _patches("train")
    patches = patches[..., ::-1]
    snapshot = patches.copy()
    learning.validate_patch_set(patches, labels, identities, patch_size=9)
    np.testing.assert_array_equal(patches, snapshot)


@pytest.mark.parametrize("defect", ["raw_hu_scale", "nonbinary_seed", "nonbinary_validity", "seed_in_padding"])
def test_training_boundary_rejects_false_channel_semantics(defect):
    patches, labels, identities = _patches("train")
    if defect == "raw_hu_scale":
        patches[0, 0, 0, 0, 0] = -1000
    elif defect == "nonbinary_seed":
        patches[0, 1, 0, 0, 0] = 0.3
    elif defect == "nonbinary_validity":
        patches[0, 2, 0, 0, 0] = 0.3
    else:
        patches[0, 2, 4, 4, 4] = 0
    with pytest.raises(ValueError):
        learning.validate_patch_set(patches, labels, identities, patch_size=9)


def test_metrics_use_actual_confusion_counts_and_keep_clinical_claim_false():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.5, 0.2, 0.3])
    metrics = learning.binary_metrics(labels, scores)
    assert {k: metrics[k] for k in ("true_positive", "false_negative", "false_positive", "true_negative")} == {
        "true_positive": 1, "false_negative": 1, "false_positive": 1, "true_negative": 2,
    }
    assert metrics["precision"] == metrics["recall"] == 0.5
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["brier_score"] == pytest.approx(0.24)
    assert metrics["binary_cross_entropy"] == pytest.approx(-np.log([0.9, 0.1, 0.5, 0.8, 0.7]).mean())
    assert metrics["clinical_accuracy_measured"] is False
    assert metrics["scope"] == "procedural_synthetic_candidate_classification_only"


def test_undefined_metrics_are_null_not_success_or_zero():
    metrics = learning.binary_metrics([0, 0], [0.0, 0.0])
    assert metrics["precision"] is None and metrics["recall"] is None
    assert metrics["specificity"] == 1
    assert np.isfinite(metrics["binary_cross_entropy"])


@pytest.mark.parametrize("labels,scores", [([], []), ([1], [0.2, 0.4]), ([2], [0.5]), ([0], [np.nan]), ([1], [np.inf]), ([1], [-0.1]), ([0], [1.1])])
def test_invalid_metrics_cannot_be_reported_as_success(labels, scores):
    with pytest.raises(ValueError):
        learning.binary_metrics(labels, scores)


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_tiny_optimization_and_hashed_reload(tmp_path, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; the CPU integration still exercises actual optimization")
    work, output = _new_release(tmp_path)
    train, validation = _patches("train"), _patches("validation", 4, seed=38)
    config = learning.TrainingConfig(seed=318, epochs=8, batch_size=4, learning_rate=0.005, patch_size=9, cpu_threads=1)
    torch.manual_seed(config.seed)
    initial_weights = {name: value.clone() for name, value in learning.build_ranker().state_dict().items()}
    provenance = {"supervision": learning.SUPERVISION, "clinical_accuracy_measured": False, "fixture": "synthetic-test-only"}
    report = learning.train_ranker(train, validation, output, work_root=work, config=config, provenance=provenance, device=device)
    assert report["status"] == "trained_research_only" and report["promoted"] is False
    assert report["clinical_accuracy_measured"] is False
    assert len(report["history"]) == config.epochs
    best_loss = min(row["validation"]["binary_cross_entropy"] for row in report["history"])
    assert best_loss < report["initial_validation"]["binary_cross_entropy"]
    assert report["history"][report["best_epoch"] - 1]["validation"]["binary_cross_entropy"] == best_loss
    checkpoint = output / "checkpoint-best.local.pt"
    assert learning.file_sha256(checkpoint) == report["checkpoint_sha256"]
    model, restored_config = learning.load_research_ranker(checkpoint, report["checkpoint_sha256"], device=device)
    assert restored_config == config
    assert any(not torch.equal(value.detach().cpu(), initial_weights[name]) for name, value in model.state_dict().items())
    scores = learning.predict_scores(model, validation[0], device=device, batch_size=2)
    assert scores.shape == validation[1].shape and np.isfinite(scores).all()
    assert scores[validation[1] == 1].mean() > scores[validation[1] == 0].mean()
    recomputed = learning.binary_metrics(validation[1], scores)
    assert recomputed["binary_cross_entropy"] == pytest.approx(best_loss, abs=1e-6)
    assert json.loads((output / "training-report.local.json").read_text()) == report
    assert report["data_sha256"] == {
        "train": learning.patch_set_sha256(*train),
        "validation": learning.patch_set_sha256(*validation),
    }
    training_input = json.loads((output / "training-input.local.json").read_text())
    assert training_input["data_sha256"] == report["data_sha256"]
    assert report["exact_training_resume_supported"] is False
    before = (output / "training-report.local.json").read_bytes()
    with pytest.raises(ValueError, match="empty"):
        learning.train_ranker(train, validation, output, work_root=work,
            config=config, provenance=provenance, device=device)
    assert (output / "training-report.local.json").read_bytes() == before
    assert not list(output.glob("*writing*"))
    if device == "cuda":
        assert report["peak_gpu_bytes"] > 0
        del model
        torch.cuda.empty_cache()


@pytest.mark.integration
def test_checkpoint_digest_rejected_before_deserialization(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not even a serialized model")
    def forbidden_load(*args, **kwargs):
        pytest.fail("A changed checkpoint must never reach deserialization")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="changed"):
        learning.load_research_ranker(checkpoint, "0" * 64)


@pytest.mark.integration
@pytest.mark.parametrize("architecture", ["another-model", "observed-wall-patch-cnn-1"])
def test_matching_hash_does_not_override_wrong_architecture(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "wrong-model.pt"
    torch.save({"architecture": architecture, "supervision": learning.SUPERVISION}, checkpoint)
    with pytest.raises(ValueError, match="declared synthetic"):
        learning.load_research_ranker(checkpoint, learning.file_sha256(checkpoint))


@pytest.mark.integration
@pytest.mark.parametrize("location", ["outside_root", "git_checkout"])
def test_training_api_cannot_bypass_release_quarantine(tmp_path, location):
    pytest.importorskip("torch")
    work = tmp_path / "protected"
    work.mkdir()
    output = (tmp_path if location == "outside_root" else work) / "release"
    output.mkdir()
    if location == "git_checkout":
        (work / ".git").write_text("gitdir: elsewhere\n")
    with pytest.raises(ValueError, match="protected|Git"):
        learning.train_ranker(_patches("train"), _patches("validation"), output,
            work_root=work, config=learning.TrainingConfig(epochs=1, patch_size=9),
            provenance={"supervision": learning.SUPERVISION, "clinical_accuracy_measured": False}, device="cpu")
    assert not list(output.iterdir())


@pytest.mark.integration
def test_training_array_mutation_prevents_completed_release(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    work, output = _new_release(tmp_path)
    train, validation = _patches("train"), _patches("validation")
    real_step = torch.optim.AdamW.step
    def mutate_after_real_optimization(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        train[0][0, 0, 0, 0, 0] += 0.001
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", mutate_after_real_optimization)
    with pytest.raises(RuntimeError, match="arrays changed"):
        learning.train_ranker(train, validation, output, work_root=work,
            config=learning.TrainingConfig(epochs=1, patch_size=9, cpu_threads=1),
            provenance={"supervision": learning.SUPERVISION, "clinical_accuracy_measured": False}, device="cpu")
    assert (output / "training-input.local.json").exists()
    assert not (output / "training-report.local.json").exists()


@pytest.mark.integration
def test_checkpoint_write_failure_never_publishes_a_completed_release(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    work, output = _new_release(tmp_path)
    def disk_full(bundle, handle, *args, **kwargs):
        handle.write(b"partial checkpoint")
        raise OSError("simulated disk full")
    monkeypatch.setattr(torch, "save", disk_full)
    with pytest.raises(OSError, match="disk full"):
        learning.train_ranker(_patches("train"), _patches("validation"), output, work_root=work,
            config=learning.TrainingConfig(epochs=1, patch_size=9, cpu_threads=1),
            provenance={"supervision": learning.SUPERVISION, "clinical_accuracy_measured": False}, device="cpu")
    assert (output / "training-input.local.json").exists()
    assert not (output / "training-report.local.json").exists()
    assert not (output / "checkpoint-best.local.pt").exists()
    assert not (output / "checkpoint-last.local.pt").exists()


def _center_task(prefix, count, *, seed):
    """Equal HU histograms; only the location of each intensity carries labels."""
    assert count % 2 == 0
    rng = np.random.default_rng(seed)
    patches = np.zeros((count, 3, 9, 9, 9), np.float32)
    labels = (np.arange(count) % 2).astype(np.float32)
    values = np.array([-0.8] * 364 + [0.8] * 364 + [0.0], np.float32)
    center = np.ravel_multi_index((4, 4, 4), (9, 9, 9))
    for index in range(0, count, 2):
        arranged = rng.permutation(values)
        other = int(rng.choice(np.flatnonzero(arranged == np.float32(-0.8))))
        arranged[center], arranged[other] = arranged[other], arranged[center]
        patches[index, 0] = arranged.reshape(9, 9, 9)
        # Pair the negative with the same background and a positive center,
        # compensating at a uniformly chosen remote voxel to preserve counts.
        other = int(rng.choice(np.flatnonzero(arranged == np.float32(0.8))))
        arranged[center], arranged[other] = arranged[other], arranged[center]
        patches[index + 1, 0] = arranged.reshape(9, 9, 9)
    patches[:, 1, 4, 0, 4] = 1
    patches[:, 2] = 1
    return patches, labels, [f"{prefix}-geometry-{index // 2}" for index in range(count)]


def test_center_task_has_identical_global_histograms_without_mask_label_leakage():
    patches, labels, _ = _center_task("synthetic", 12, seed=170)
    histograms = np.sort(patches[:, 0].reshape(12, -1), axis=1)
    np.testing.assert_array_equal(histograms, np.repeat(histograms[:1], 12, axis=0))
    np.testing.assert_array_equal(patches[:, 1:], np.repeat(patches[:1, 1:], 12, axis=0))
    np.testing.assert_array_equal(patches[:, 0, 4, 4, 4] > 0, labels.astype(bool))
    assert len({p[0].tobytes() for p in patches}) == 12
    assert np.all(np.count_nonzero(patches[::2, 0] != patches[1::2, 0], axis=(1, 2, 3)) == 2)


@pytest.mark.integration
@pytest.mark.parametrize("size", range(9, 34, 2))
def test_center_features_remain_source_aligned_for_every_supported_odd_size(size):
    torch = pytest.importorskip("torch")
    model = learning.build_ranker()

    class CoordinateFeatures(torch.nn.Module):
        def forward(self, inputs):
            # The existing stride-4 stem's physical lattice, with a linear
            # coordinate field whose value is independently known at center.
            coordinates = torch.arange(0, size, 4, dtype=inputs.dtype)
            x, y, z = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
            field = x + 2 * y + 3 * z
            return field[None, None].expand(len(inputs), 32, -1, -1, -1)

    model.stem = CoordinateFeatures()
    model.head = torch.nn.Identity()
    inputs = torch.zeros((2, 3, size, size, size))
    center = size // 2
    inputs[0, 0, center, center, center] = -0.8
    inputs[1, 0, center, center, center] = 0.9
    with torch.inference_mode():
        features = model(inputs)
    assert features.shape == (2, 65)
    np.testing.assert_array_equal(features[:, 32:64].numpy(), np.full((2, 32), 6 * center))
    np.testing.assert_array_equal(features[:, -1].numpy(), np.array([-0.8, 0.9], np.float32))


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_center_specific_task_learns_and_reloads_without_global_histogram_signal(tmp_path, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; actual CPU optimization remains covered")
    work, output = _new_release(tmp_path)
    train = _center_task("train", 256, seed=1729)
    validation = _center_task("validation", 64, seed=2718)
    held_out = _center_task("test", 128, seed=3141)
    learning.require_disjoint_geometries(train[2], validation[2], held_out[2])
    config = learning.TrainingConfig(seed=1729, epochs=16, batch_size=32,
        learning_rate=0.004, patch_size=9, cpu_threads=1)
    report = learning.train_ranker(train, validation, output, work_root=work, config=config,
        provenance={"supervision": learning.SUPERVISION, "clinical_accuracy_measured": False}, device=device)
    assert report["architecture"] == "observed-wall-centered-cnn-2"
    best = report["history"][report["best_epoch"] - 1]["validation"]
    assert best["binary_cross_entropy"] < report["initial_validation"]["binary_cross_entropy"] * 0.25
    checkpoint = output / "checkpoint-best.local.pt"
    restored, restored_config = learning.load_research_ranker(checkpoint, report["checkpoint_sha256"], device=device)
    assert restored_config == config
    validation_scores = learning.predict_scores(restored, validation[0], device=device, batch_size=8)
    assert learning.binary_metrics(validation[1], validation_scores)["binary_cross_entropy"] == pytest.approx(best["binary_cross_entropy"], abs=1e-6)
    scores = learning.predict_scores(restored, held_out[0], device=device, batch_size=8)
    # Every independent matched background must respond to the changed center.
    # This verifies learned discrimination, without treating tiny-training
    # scores as perfectly calibrated probabilities at an arbitrary threshold.
    assert np.all(scores[1::2] > scores[::2])
    assert np.mean(scores[1::2] - scores[::2]) > 0.75
    assert learning.binary_metrics(held_out[1], scores)["binary_cross_entropy"] < 0.15
    assert report["data_sha256"] == {"train": learning.patch_set_sha256(*train), "validation": learning.patch_set_sha256(*validation)}
    with pytest.raises(ValueError, match="changed"):
        learning.load_research_ranker(checkpoint, "0" * 64, device=device)
    if device == "cuda":
        del restored
        torch.cuda.empty_cache()
