"""Quarantined learning tools for procedurally supervised wall proposals.

This is not an ethmoid segmentation model, expert supervision, or a promoted
counting method. PyTorch is imported only by functions that execute learning.
No application route or historical method selects these models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np


ARCHITECTURE = "observed-wall-patch-cnn-1"
SUPERVISION = "procedural_synthetic_visible_outer_wall"
INPUT_CHANNELS = ("normalized_hu", "posterior_seed", "valid_source_extent")
HU_NORMALIZATION = {"clip_hu": [-1000, 2000], "divisor": 1000.0}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260918
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patch_size: int = 17
    patch_spacing_mm: float = 1.0
    cpu_threads: int = 2

    def validate(self) -> None:
        for name in ("seed", "epochs", "batch_size", "patch_size", "cpu_threads"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32 or not 1 <= self.epochs <= 1000:
            raise ValueError("Invalid seed or bounded epoch count")
        if not 1 <= self.batch_size <= 256 or not 1 <= self.cpu_threads <= 8:
            raise ValueError("Training resource bounds exceeded")
        if not 9 <= self.patch_size <= 33 or self.patch_size % 2 != 1:
            raise ValueError("An odd, bounded patch size is required")
        for name in ("learning_rate", "weight_decay", "patch_spacing_mm"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Training values must be finite numbers")
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Invalid optimizer configuration")
        if not 0.25 <= self.patch_spacing_mm <= 2.0:
            raise ValueError("Unsupported model-only sampling spacing")


def canonical_sha256(value: dict | list) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_set_sha256(patches, labels, geometry_ids) -> str:
    digest = hashlib.sha256()
    for array in (patches, labels):
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(json.dumps(values.shape).encode())
        digest.update(memoryview(values).cast("B"))
    digest.update(json.dumps(list(geometry_ids), separators=(",", ":")).encode())
    return digest.hexdigest()


def _protected_directory(path: str | Path, work_root: str | Path) -> Path:
    root = Path(work_root).resolve(strict=True)
    destination = Path(path).resolve()
    if not root.is_dir() or destination == root or not destination.is_relative_to(root):
        raise ValueError("Research training outputs must stay inside the protected work root")
    if any((parent / ".git").exists() for parent in (destination, *destination.parents)):
        raise ValueError("Training outputs cannot be created inside a Git checkout")
    return destination


def protected_new_directory(path: str | Path, work_root: str | Path) -> Path:
    destination = _protected_directory(path, work_root)
    if destination.exists():
        raise ValueError("Training releases cannot overwrite an existing directory")
    destination.mkdir(parents=True)
    return destination


def write_new_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_ranker():
    """A small 3D context classifier; its scores are not calibrated clinical risk."""
    from torch import nn

    return nn.Sequential(
        nn.Conv3d(3, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.SiLU(),
        nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
        nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
        nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, 1),
    )


def normalize_patch(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 4 or raw.shape[0] != 3 or not np.isfinite(raw).all():
        raise ValueError("A finite HU/seed/validity patch is required")
    if not np.isin(raw[1:], (0, 1)).all():
        raise ValueError("Seed and source validity channels must be binary")
    result = np.array(raw, dtype=np.float32, copy=True)
    result[0] = np.clip(result[0], -1000, 2000) / 1000.0
    return result


def validate_patch_set(patches, labels, geometry_ids, *, patch_size: int) -> None:
    expected = (3, patch_size, patch_size, patch_size)
    if patches.ndim != 5 or patches.shape[1:] != expected or len(patches) < 2:
        raise ValueError("Training requires bounded physical three-channel patches")
    if patches.dtype.kind != "f" or not np.isfinite(patches).all():
        raise ValueError("Training patches must be finite floating point values")
    if np.any((patches[:, 0] < -1) | (patches[:, 0] > 2)):
        raise ValueError("HU must use the declared bounded normalization")
    if not np.isin(patches[:, 1:], (0, 1)).all() or np.any(patches[:, 1] > patches[:, 2]):
        raise ValueError("Seed and validity channels must be binary and source-contained")
    if labels.shape != (len(patches),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Candidate targets must be explicit binary synthetic labels")
    if len(geometry_ids) != len(patches) or any(not isinstance(g, str) or not g for g in geometry_ids):
        raise ValueError("Every patch must retain its complete procedural geometry identity")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("A training/evaluation split must contain positive and negative candidates")


def require_disjoint_geometries(*groups) -> None:
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            if set(left) & set(right):
                raise ValueError("A procedural geometry or its augmentations cross data partitions")


def binary_metrics(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Metrics require matched nonempty candidate arrays")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Metrics require binary labels and finite probability-like scores")
    predicted = scores >= 0.5
    positives, negatives = labels == 1, labels == 0
    tp, fp = int(np.count_nonzero(predicted & positives)), int(np.count_nonzero(predicted & negatives))
    fn, tn = int(np.count_nonzero(~predicted & positives)), int(np.count_nonzero(~predicted & negatives))
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return {"scope": "procedural_synthetic_candidate_classification_only", "threshold": 0.5,
        "samples": len(labels), "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "binary_cross_entropy": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "clinical_accuracy_measured": False}


def predict_scores(model, patches, *, device="cpu", batch_size=64) -> np.ndarray:
    import torch

    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("Prediction batch size is outside the resource bound")
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            inputs = torch.as_tensor(np.asarray(patches[start:start + batch_size]), dtype=torch.float32, device=device)
            logits = model(inputs).reshape(-1)
            values.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(values) if values else np.empty(0, np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("The research model produced nonfinite scores")
    return result


def train_ranker(train, validation, output: Path, *, work_root: Path,
                 config: TrainingConfig, provenance: dict, device="cuda") -> dict:
    """Train a quarantined model. Selection uses validation loss, never test data."""
    import torch

    config.validate()
    x_train, y_train, ids_train = train
    x_val, y_val, ids_val = validation
    validate_patch_set(x_train, y_train, ids_train, patch_size=config.patch_size)
    validate_patch_set(x_val, y_val, ids_val, patch_size=config.patch_size)
    require_disjoint_geometries(ids_train, ids_val)
    if provenance.get("supervision") != SUPERVISION or provenance.get("clinical_accuracy_measured") is not False:
        raise ValueError("Training provenance must declare procedural targets and absent clinical evidence")
    output = _protected_directory(output, work_root)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Training needs a new empty protected release directory")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested training device is unavailable")
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_ranker().to(device)
    initial = binary_metrics(y_val, predict_scores(model, x_val, device=device, batch_size=config.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_epoch = 0
    started = time.monotonic()
    history = []
    data_hashes = {"train": patch_set_sha256(*train), "validation": patch_set_sha256(*validation)}
    write_new_json(output / "training-input.local.json", {"schema_version": 1,
        "architecture": ARCHITECTURE, "config": asdict(config), "provenance": provenance,
        "train_geometries": sorted(set(ids_train)), "validation_geometries": sorted(set(ids_val)),
        "train_candidates": len(y_train), "validation_candidates": len(y_val),
        "data_sha256": data_hashes,
        "torch_version": str(torch.__version__), "device": device, "initial_validation": initial})
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = generator.permutation(len(x_train))
        total_loss = 0.0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            inputs = torch.as_tensor(np.asarray(x_train[indices]), dtype=torch.float32, device=device)
            targets = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs).reshape(-1), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        metrics = binary_metrics(y_val, predict_scores(model, x_val, device=device, batch_size=config.batch_size))
        row = {"epoch": epoch, "training_loss": total_loss / len(x_train), "validation": metrics}
        history.append(row)
        checkpoint = {"architecture": ARCHITECTURE, "supervision": SUPERVISION,
            "epoch": epoch, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config), "provenance_sha256": canonical_sha256(provenance),
            "data_sha256": data_hashes}
        temporary = output / "checkpoint-writing.local.pt"
        with temporary.open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "checkpoint-last.local.pt")
        if metrics["binary_cross_entropy"] < best_loss:
            best_loss, best_epoch = metrics["binary_cross_entropy"], epoch
            temporary = output / "best-writing.local.pt"
            with temporary.open("xb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output / "checkpoint-best.local.pt")
        row["checkpoint_last_sha256"] = file_sha256(output / "checkpoint-last.local.pt")
        write_new_json(output / f"epoch-{epoch:04d}.local.json", row)
        print(json.dumps({"epoch": epoch, "epochs": config.epochs, "train_loss": round(row["training_loss"], 6),
            "best_validation_loss": round(best_loss, 6), "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
    checkpoint_path = output / "checkpoint-best.local.pt"
    if {"train": patch_set_sha256(*train), "validation": patch_set_sha256(*validation)} != data_hashes:
        raise RuntimeError("Training arrays changed during the release")
    report = {"schema_version": 1, "status": "trained_research_only", "architecture": ARCHITECTURE,
        "supervision": SUPERVISION, "clinical_accuracy_measured": False, "promoted": False,
        "config": asdict(config), "provenance_sha256": canonical_sha256(provenance), "data_sha256": data_hashes,
        "initial_validation": initial, "best_epoch": best_epoch, "history": history,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "checkpoint_sha256": file_sha256(checkpoint_path), "torch_version": str(torch.__version__),
        "exact_training_resume_supported": False,
        "research_only_reason": "Synthetic visible-wall targets do not establish sinus identity or cell-count accuracy"}
    write_new_json(output / "training-report.local.json", report)
    return report


def load_research_ranker(checkpoint: Path, expected_sha256: str, *, device="cpu"):
    import torch

    if file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Research checkpoint changed")
    bundle = torch.load(checkpoint, map_location=device, weights_only=True)
    if bundle.get("architecture") != ARCHITECTURE or bundle.get("supervision") != SUPERVISION:
        raise ValueError("Checkpoint is not the declared synthetic proposal model")
    config = TrainingConfig(**bundle["config"])
    config.validate()
    model = build_ranker().to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, config
