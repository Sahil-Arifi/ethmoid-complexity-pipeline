"""Local direct-anatomy model contract. No trained ethmoid weights are bundled.

Only operator-installed models with hash-bound expert evaluation evidence may be
selected. A missing model is explicit; it must never masquerade as a prediction.
"""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np

from .engine import CountingError
from .profiles import file_sha256

LABELS = {"background": 0, "ethmoid_left": 1, "ethmoid_right": 2, "sphenoid": 3}


def load_model_manifest(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    if not path.is_file():
        raise CountingError("DIRECT_MODEL_UNAVAILABLE", "No trained and evaluated direct anatomy model is installed.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest["schema_version"] != 1 or manifest["backend"] != "nnunet_v2" or
            manifest["output_labels"] != LABELS or manifest["status"] != "expert_reference_review_passed" or
            not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", manifest["model_id"])):
            raise ValueError
        folds = manifest["folds"]
        if not isinstance(folds, list) or not folds or len(set(folds)) != len(folds) or any(type(f) is not int or not 0 <= f <= 9 for f in folds):
            raise ValueError
        root = (path.parent / manifest["model_directory"]).resolve()
        if not root.is_relative_to(path.parent) or not root.is_dir():
            raise ValueError
        entries = manifest["files"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError
        needed = {"dataset.json", "plans.json", *[f"fold_{f}/checkpoint_final.pth" for f in folds]}
        if needed - entries.keys():
            raise ValueError
        for relative, digest in entries.items():
            item = (root / relative).resolve()
            if not item.is_relative_to(root) or not item.is_file() or file_sha256(item) != digest:
                raise ValueError
        dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        if dataset.get("labels") != LABELS or dataset.get("channel_names") != {"0": "CT"} or dataset.get("file_ending") != ".nii.gz":
            raise ValueError
        evidence = (path.parent / manifest["evaluation_report"]).resolve()
        if not evidence.is_relative_to(path.parent) or file_sha256(evidence) != manifest["evaluation_sha256"]:
            raise ValueError
        report = json.loads(evidence.read_text(encoding="utf-8"))
        if (report.get("reference_source") != "independent_expert" or report.get("promotion_approved") is not True or
            not report.get("acceptance_thresholds") or report.get("patient_group_disjoint") is not True or
            report.get("clinical_readiness_claim") is not False):
            raise ValueError
        if not manifest.get("nnunet_version"):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise CountingError("DIRECT_MODEL_INTEGRITY", "The direct anatomy model, labels, or expert evaluation evidence is incomplete or changed.") from None
    return manifest, root


def _require_offline_runtime():
    if os.environ.get("ETHMOID_INFERENCE_OFFLINE", "1") != "1":
        raise CountingError("DIRECT_MODEL_RUNTIME", "Direct anatomy inference requires the protected offline runtime.")
    try:
        import sitecustomize  # type: ignore[import-not-found]
        location = Path(sitecustomize.__file__).resolve()
        if "counting-runtime" not in str(location).lower() or not location.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError
    except (ImportError, AttributeError, ValueError):
        raise CountingError("DIRECT_MODEL_RUNTIME", "The protected offline runtime guard is unavailable.") from None


def run_direct_model(manifest_path, source, canonical, destination):
    manifest, root = load_model_manifest(manifest_path)
    _require_offline_runtime()
    try:
        if metadata.version("nnunetv2") != manifest["nnunet_version"]:
            raise CountingError("DIRECT_MODEL_RUNTIME", "The installed nnU-Net version differs from the promoted model.")
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from .localization import Localization
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(root), use_folds=tuple(manifest["folds"]), checkpoint_name="checkpoint_final.pth")
        predictor.predict_from_files([[str(source)]], [str(destination / "anatomy")],
            save_probabilities=False, overwrite=False, num_processes_preprocessing=1, num_processes_segmentation_export=1)
        raw = nib.load(destination / "anatomy.nii.gz")
        original = nib.load(source)
        if raw.shape != original.shape or not np.allclose(raw.affine, original.affine, atol=1e-5, rtol=0):
            raise CountingError("DIRECT_MODEL_GEOMETRY", "The direct anatomy output does not match the original source grid.")
        output = nib.as_closest_canonical(raw)
        if output.shape != canonical.shape or not np.allclose(output.affine, canonical.affine, atol=1e-5, rtol=0):
            raise CountingError("DIRECT_MODEL_GEOMETRY", "The direct anatomy output does not match the verified canonical orientation.")
        labels = np.asanyarray(output.dataobj)
        if not np.isfinite(labels).all() or not np.isin(labels, tuple(LABELS.values())).all():
            raise CountingError("DIRECT_MODEL_VALUES", "Direct anatomy output contains unsupported labels.")
        if any(not np.any(labels == value) for value in (1, 2, 3)):
            raise CountingError("EMPTY_REGION", "Direct anatomy inference did not resolve both ethmoid regions and the sphenoid.")
        region = np.isin(labels, (1, 2))
        indices = np.where(region)
        if any(axis.min() == 0 or axis.max() == region.shape[i] - 1 for i, axis in enumerate(indices)):
            raise CountingError("TARGET_CLIPPED", "Direct ethmoid anatomy reaches the acquired source boundary.")
        centers = [nib.affines.apply_affine(canonical.affine, np.argwhere(labels == value).mean(axis=0))[0] for value in (1, 2)]
        local = Localization(region, labels == 3, {}, float(np.mean(centers)), [
            "Direct model anatomical membership; individual cells still require native bone-supported separation and clinical review.",
        ])
        return local, {"localization": "nnunet_v2_direct_anatomy", "model_id": manifest["model_id"],
            "model_manifest_sha256": file_sha256(manifest_path), "evaluation_sha256": manifest["evaluation_sha256"],
            "nnunet_version": manifest["nnunet_version"], "clinical_accuracy": "requires_current_case_review"}, np.where(region, labels, 0).astype(np.uint8)
    except CountingError:
        raise
    except Exception:
        raise CountingError("DIRECT_MODEL_FAILED", "The offline direct anatomy model could not complete; no anatomical count was produced.") from None
