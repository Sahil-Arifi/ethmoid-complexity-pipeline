"""Expert annotation contracts and patient-grouped evaluation for future models.

These functions do not turn automatic proposals into training truth. Dataset
export is available only after explicit expert provenance and geometry checks.
All produced data remain in an operator-selected protected local directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .profiles import file_sha256, sealed

LABELS = {"background": 0, "ethmoid_left": 1, "ethmoid_right": 2, "sphenoid": 3}


def validate_annotation(record: dict) -> dict:
    required = {"sample_id", "patient_group", "source_path", "source_sha256", "anatomy_path", "anatomy_sha256", "expert_review", "protocol_sha256"}
    if not isinstance(record, dict) or required - record.keys():
        raise ValueError("Expert annotation record is incomplete")
    for key in ("sample_id", "patient_group"):
        if not isinstance(record[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", record[key]):
            raise ValueError("Training identity must use local neutral group tokens")
    review = record["expert_review"]
    if not isinstance(review, dict) or review.get("status") != "adjudicated" or review.get("source") != "independent_expert" or not review.get("reviewer_token"):
        raise ValueError("Unreviewed or automatic proposals cannot become training reference labels")
    if not re.fullmatch(r"[a-f0-9]{64}", str(record["protocol_sha256"])):
        raise ValueError("Annotation must bind the clinical definition protocol")
    images = {}
    for name in ("source", "anatomy", "instances"):
        if name == "instances" and not record.get("instances_path"):
            continue
        path = Path(record[name + "_path"])
        if not path.is_file() or file_sha256(path) != record.get(name + "_sha256"):
            raise ValueError("An expert source or annotation artifact changed")
        images[name] = nib.load(path)
    source = images["source"]
    if len(source.shape) != 3 or not np.isfinite(source.affine).all():
        raise ValueError("The training source geometry is invalid")
    for name, image in images.items():
        if image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-5, rtol=0):
            raise ValueError("Expert labels must match native source geometry")
        if name == "source":
            continue
        values = np.asanyarray(image.dataobj)
        if values.dtype.kind not in "ui" or not np.isfinite(values).all():
            raise ValueError("Expert masks require nonnegative integer labels")
        if name == "anatomy" and (not np.isin(values, tuple(LABELS.values())).all() or any(not np.any(values == v) for v in (1, 2, 3))):
            raise ValueError("Expert anatomy labels must identify both ethmoid regions and sphenoid")
    counts = record.get("reference_counts")
    if counts is not None:
        if not isinstance(counts, dict) or any(type(counts.get(side)) is not int or counts[side] < 0 for side in ("left", "right", "total")) or counts["left"] + counts["right"] != counts["total"]:
            raise ValueError("Independent reference counts are inconsistent")
        if "instances" in images and len(set(np.unique(np.asanyarray(images["instances"].dataobj))) - {0}) != counts["total"]:
            raise ValueError("Expert instance count disagrees with its independent reference count")
    return json.loads(json.dumps(record))


def patient_grouped_splits(records: list[dict], *, folds: int = 5, seed: str = "ethmoid-development-v1") -> list[dict]:
    """nnU-Net compatible train/val lists, disjoint by patient across every fold."""
    if type(folds) is not int or folds < 2:
        raise ValueError("At least two patient folds are required")
    seen = {}
    for record in records:
        sample, group = record["sample_id"], record["patient_group"]
        if sample in seen:
            raise ValueError("Duplicate samples cannot enter a split")
        seen[sample] = group
    groups = sorted(set(seen.values()), key=lambda value: hashlib.sha256((seed + ":" + value).encode()).hexdigest())
    if len(groups) < folds:
        raise ValueError("There are fewer independent patient groups than folds")
    assignments = {group: index % folds for index, group in enumerate(groups)}
    return [{"train": sorted(sample for sample, group in seen.items() if assignments[group] != fold),
             "val": sorted(sample for sample, group in seen.items() if assignments[group] == fold)} for fold in range(folds)]


def export_nnunet_dataset(records: list[dict], destination: str | Path, *, folds: int = 5) -> dict:
    """Export reviewed CT/anatomy pairs and grouped splits; never starts training."""
    validated = [validate_annotation(record) for record in records]
    if not validated or len({record["protocol_sha256"] for record in validated}) != 1:
        raise ValueError("A training release needs one adjudicated definition protocol")
    splits = patient_grouped_splits(validated, folds=folds)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("A dataset release must use a new directory")
    destination.mkdir(parents=True)
    (destination / "imagesTr").mkdir()
    (destination / "labelsTr").mkdir()
    for record in validated:
        # Re-save so .nii and .nii.gz sources both match the dataset file ending.
        nib.save(nib.load(record["source_path"]), destination / "imagesTr" / f"{record['sample_id']}_0000.nii.gz")
        nib.save(nib.load(record["anatomy_path"]), destination / "labelsTr" / f"{record['sample_id']}.nii.gz")
    dataset = {"channel_names": {"0": "CT"}, "labels": LABELS, "numTraining": len(validated), "file_ending": ".nii.gz"}
    (destination / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (destination / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    manifest = sealed({"schema_version": 1, "state": "expert_dataset_exported_not_trained", "protocol_sha256": validated[0]["protocol_sha256"],
        "patient_groups": len({record["patient_group"] for record in validated}), "samples": [{key: record[key] for key in ("sample_id", "patient_group", "source_sha256", "anatomy_sha256")} for record in validated],
        "split_policy": "patient_group_disjoint_development_cross_validation_not_external_validation"})
    (destination / "dataset-release.local.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def instance_agreement(predicted: np.ndarray, reference: np.ndarray, *, iou_threshold: float = 0.5) -> dict:
    """One-to-one IoU matching with explicit missed, false, split and merge counts."""
    from scipy.optimize import linear_sum_assignment
    if predicted.shape != reference.shape or predicted.ndim != 3 or not 0 < iou_threshold <= 1:
        raise ValueError("Instance evaluation needs aligned 3D labels and a fixed IoU threshold")
    if any(array.dtype.kind not in "ui" for array in (predicted, reference)):
        raise ValueError("Instance IDs must be nonnegative integers")
    pids = np.unique(predicted)
    rids = np.unique(reference)
    pids, rids = pids[pids > 0], rids[rids > 0]
    overlap = np.zeros((len(pids), len(rids)), dtype=np.int64)
    for index, pid in enumerate(pids):
        values, counts = np.unique(reference[predicted == pid], return_counts=True)
        nonzero = values > 0
        overlap[index, np.searchsorted(rids, values[nonzero])] = counts[nonzero]
    psizes = np.array([np.count_nonzero(predicted == pid) for pid in pids])
    rsizes = np.array([np.count_nonzero(reference == rid) for rid in rids])
    union = psizes[:, None] + rsizes[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
    # Prioritize the number of qualifying matches before secondary IoU quality.
    score = (iou >= iou_threshold).astype(float) * (min(len(pids), len(rids)) + 1) + iou
    rows, cols = linear_sum_assignment(-score)
    matched = [(int(pids[p]), int(rids[r]), float(iou[p, r])) for p, r in zip(rows, cols) if iou[p, r] >= iou_threshold]
    return {"iou_threshold": iou_threshold, "predicted_count": len(pids), "reference_count": len(rids),
        "count_error": len(pids) - len(rids), "absolute_count_error": abs(len(pids) - len(rids)),
        "matched_instances": len(matched), "missed_instances": len(rids) - len(matched), "false_instances": len(pids) - len(matched),
        "split_reference_instances": int(np.count_nonzero(np.sum(overlap > 0, axis=0) > 1)),
        "merged_predicted_instances": int(np.count_nonzero(np.sum(overlap > 0, axis=1) > 1)),
        "matches": [{"prediction": p, "reference": r, "iou": value} for p, r, value in matched],
        "clinical_accuracy_scope": "only_supplied_expert_reference_instances"}
