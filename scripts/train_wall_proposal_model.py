"""Train/evaluate an isolated synthetic wall experiment; never import patient data.

Run in the pinned offline GPU runtime from a clean source checkout. This command
has no application/database/catalog argument and cannot publish a counting model.
Each run is a new protected release; interrupted artifacts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ethmoid_pipeline.counting.wall_learning import (
    ARCHITECTURE, HU_NORMALIZATION, SUPERVISION, TrainingConfig, binary_metrics,
    canonical_sha256, file_sha256, load_research_ranker, normalize_patch,
    patch_set_sha256, predict_scores, protected_new_directory,
    require_disjoint_geometries, train_ranker, write_new_json,
)
from ethmoid_pipeline.counting.synthetic_walls import (
    GEOMETRY_FAMILIES, VERSION, SceneAugmentation, candidate_targets,
    make_geometry, render_scene, split_geometry_families,
)
from ethmoid_pipeline.counting.wall_proposals import (
    SOLVER_VERSION, SUPPORT_POLICY_VERSION, WallProposalError, candidate_patch, extract_candidates,
)


EXPERIMENT = "synthetic-observed-wall-learning-3"
MAX_PATCHES_PER_SPLIT = 20_000
MAX_PARTITION_BYTES = 768 * 1024**2


def augmentations(seed):
    return (
        SceneAugmentation(seed=seed),
        SceneAugmentation(opacity="partial", noise_std_hu=60, blur_sigma_mm=0.2, seed=seed + 1),
        SceneAugmentation(opacity="full", noise_std_hu=35, blur_sigma_mm=0.35, seed=seed + 2),
        SceneAugmentation(spacing_mm=(0.625, 2.5, 0.625), opacity="partial",
                          noise_std_hu=25, blur_sigma_mm=0.3, seed=seed + 3),
    )


def patches_for(scene, candidates, indices, config):
    # Extent is the voxel footprint: this produces exactly patch_spacing_mm
    # between adjacent model samples, independently of native spacing.
    extent = config.patch_size * config.patch_spacing_mm
    result = np.empty((len(indices), 3, config.patch_size, config.patch_size, config.patch_size), np.float16)
    for row, index in enumerate(indices):
        raw = candidate_patch(scene.hu, scene.seed, candidates, int(index),
            physical_size_mm=extent, output_shape=(config.patch_size,) * 3)
        result[row] = normalize_patch(raw)
    return result


def prepare_partition(geometries, config, *, candidates_per_scene=96, hard_negative_mining=False):
    if type(candidates_per_scene) is not int or not 8 <= candidates_per_scene <= 256:
        raise ValueError("Candidate sampling exceeds the bounded experiment configuration")
    if type(hard_negative_mining) is not bool:
        raise ValueError("Hard-negative sampling must be an explicit boolean policy")
    total = len(geometries) * len(augmentations(config.seed)) * candidates_per_scene
    config.validate()
    if total > MAX_PATCHES_PER_SPLIT or total * 3 * config.patch_size**3 * 2 > MAX_PARTITION_BYTES:
        raise ValueError("Synthetic split exceeds its patch budget before allocation")
    rng = np.random.default_rng(config.seed)
    patches, labels, identities, scenes = [], [], [], []
    for geometry in geometries:
        for augmentation in augmentations(config.seed):
            scene = render_scene(geometry, augmentation)
            record = {"geometry_id": geometry.geometry_id, "family": geometry.family,
                "augmentation": asdict(augmentation), "augmentation_id": scene.augmentation_id,
                "observable": scene.observable, "support_fraction": scene.support_fraction}
            try:
                candidates = extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
                targets = candidate_targets(scene, candidates.positions_ijk)
                positive, negative = np.flatnonzero(targets == 1), np.flatnonzero(targets == 0)
                material = scene.outer_wall[tuple(candidates.positions_ijk.T)]
                hard_negative = np.flatnonzero((targets == 0) & material)
                negative_budget = candidates_per_scene // 2
                reserved = rng.permutation(hard_negative)[:negative_budget // 2] if hard_negative_mining else np.empty(0, np.int64)
                remaining = np.setdiff1d(negative, reserved, assume_unique=True)
                negative_selection = np.concatenate((reserved, rng.permutation(remaining)[:negative_budget - len(reserved)]))
                # Equal per-class cap; never repeat a voxel or turn ignored targets positive.
                selected = np.concatenate((rng.permutation(positive)[:candidates_per_scene // 2],
                                           negative_selection))
                selected = rng.permutation(selected)
                record.update(status="sampled", candidates=len(targets),
                    positives_available=len(positive), negatives_available=len(negative),
                    hard_negatives_available=len(hard_negative), hard_negative_mining=hard_negative_mining,
                    hard_negatives_sampled=int(np.isin(selected, hard_negative).sum()),
                    selected_candidate_indices=selected.tolist(), sampled=len(selected))
                if len(selected):
                    patches.append(patches_for(scene, candidates, selected, config))
                    labels.append(targets[selected].astype(np.float32))
                    identities.extend([geometry.geometry_id] * len(selected))
            except WallProposalError as exc:
                record.update(status="unavailable", reason=exc.code, sampled=0)
            scenes.append(record)
    if not patches:
        raise ValueError("The procedural partition has no eligible observed candidates")
    values = (np.concatenate(patches), np.concatenate(labels), identities)
    return values, scenes


def freeze_sources(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8").strip()
    if git("status", "--porcelain"):
        raise ValueError("Training/evaluation requires a clean committed source checkout")
    names = ["scripts/train_wall_proposal_model.py"] + [
        f"src/ethmoid_pipeline/counting/{name}.py" for name in
        ("wall_learning", "synthetic_walls", "wall_proposals", "wall_benchmark", "localization")]
    for relative in names[1:]:
        module_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != (repo / relative).resolve():
            raise ValueError("Imported experiment module does not match the frozen checkout")
    return {"git_commit": git("rev-parse", "HEAD"),
            "source_sha256": {name: file_sha256(repo / name) for name in names}}


def verify_sources(repo, frozen):
    if freeze_sources(repo) != frozen:
        raise RuntimeError("Source changed after the experiment was frozen")


def evaluate_partition(geometries, model, config, device, *, controls=False, evidence_directory=None):
    from ethmoid_pipeline.counting.wall_benchmark import evaluate_scene, extraction_failure_row

    rows = []
    for geometry in geometries:
        settings = augmentations(config.seed)
        if controls:
            settings = tuple(SceneAugmentation(wall_state=state, seed=config.seed + 101)
                for state in ("missing", "averaged_away", "cropped", "ambiguous"))
        for augmentation in settings:
            scene = render_scene(geometry, augmentation)
            evidence = {} if evidence_directory is not None else None
            try:
                candidates = extract_candidates(scene.hu, scene.seed, scene.spacing_mm, scene.hu.shape[1] - 1)
                scores = np.empty(len(candidates.positions_ijk), np.float32)
                for start in range(0, len(scores), config.batch_size):
                    indices = np.arange(start, min(len(scores), start + config.batch_size))
                    patches = patches_for(scene, candidates, indices, config)
                    scores[start:start + len(indices)] = predict_scores(model, patches,
                        device=device, batch_size=config.batch_size)
                row = evaluate_scene(scene, candidates, scores, evidence=evidence)
            except WallProposalError as exc:
                row = extraction_failure_row(scene, exc.code)
            if evidence is not None:
                evidence["source_hu_and_seed_sha256"] = np.asarray(patch_set_sha256(scene.hu, scene.seed, []))
                evidence_name = f"{geometry.geometry_id}-{scene.augmentation_id}.local.npz"
                evidence_path = evidence_directory / evidence_name
                with evidence_path.open("xb") as handle:
                    np.savez(handle, **evidence)
                row["evidence"] = {"filename": evidence_name, "sha256": file_sha256(evidence_path)}
            rows.append(row)
            print(json.dumps({"stage": "negative_controls" if controls else "surface_evaluation",
                              "scenes_completed": len(rows), "family": geometry.family}), flush=True)
    return rows


def run(args):
    from ethmoid_pipeline.counting.models import _require_offline_runtime
    from ethmoid_pipeline.counting.wall_benchmark import summarize
    import torch

    config = TrainingConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
    config.validate()
    if not 1 <= args.geometries_per_family <= 8:
        raise ValueError("Use between one and eight complete geometries per family")
    _require_offline_runtime()
    if not torch.cuda.is_available():
        raise ValueError("This local experiment command requires the verified GPU runtime")
    repo = Path(__file__).resolve().parents[1]
    frozen = freeze_sources(repo)
    release = protected_new_directory(args.output, args.work_root)
    started = time.monotonic()
    geometries = [make_geometry(i, family, seed=config.seed)
                  for family in GEOMETRY_FAMILIES for i in range(args.geometries_per_family)]
    split = split_geometry_families(geometries, seed=config.seed)
    require_disjoint_geometries(*[[g.geometry_id for g in values] for values in split.values()])
    require_disjoint_geometries(*[[g.family for g in values] for values in split.values()])
    provenance = {"schema_version": 1, "experiment": EXPERIMENT, **frozen,
        "supervision": SUPERVISION, "clinical_accuracy_measured": False,
        "patient_images_used": 0, "expert_masks_used": 0, "training_target": "noiseless_wall_attributed_observed_leading_face",
        "generator": VERSION, "decoder": SOLVER_VERSION, "support_policy": SUPPORT_POLICY_VERSION,
        "architecture": ARCHITECTURE,
        "hu_normalization": HU_NORMALIZATION, "config": asdict(config),
        "geometry_split": {name: [asdict(g) for g in values] for name, values in split.items()},
        "split_families": {name: sorted({g.family for g in values}) for name, values in split.items()},
        "augmentations": [asdict(a) for a in augmentations(config.seed)],
        "candidates_per_scene": args.candidates_per_scene,
        "training_sampling": "half_negative_budget_reserved_for_material_candidates_without_observable_target; no_target_relabeling",
        "validation_test_sampling": "uniform_within_class_cap; no_hard_negative_reservation",
        "selection": "lowest_validation_binary_cross_entropy; fixed_threshold_0.5; test_not_used_for_selection",
        "evaluation_scope": "procedural_development; prior_experiment_informed_engineering; not_independent_clinical_validation",
        "offline_guard_sha256": file_sha256(Path(sys.modules["sitecustomize"].__file__)),
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "gpu_name": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "promotion": "quarantined_synthetic_research_only; clinical model interface cannot load this checkpoint"}
    write_new_json(release / "provenance.local.json", provenance)
    try:
        datasets = {}
        inventory = {}
        # Geometry and complete families were frozen before any image is rendered.
        for name in ("train", "validation", "test"):
            datasets[name], rows = prepare_partition(split[name], config, candidates_per_scene=args.candidates_per_scene,
                                                    hard_negative_mining=name == "train")
            x, y, ids = datasets[name]
            np.savez(release / f"{name}-patches.local.npz", patches=x, targets=y, geometry_ids=np.asarray(ids))
            inventory[name] = {"scenes": rows, "samples": len(y), "class_counts": dict(Counter(map(int, y))),
                "dataset_sha256": patch_set_sha256(x, y, ids),
                "file_sha256": file_sha256(release / f"{name}-patches.local.npz")}
            print(json.dumps({"stage": "dataset_prepared", "partition": name, "samples": len(y)}), flush=True)
        write_new_json(release / "dataset-inventory.local.json", inventory)
        verify_sources(repo, frozen)
        model_output = protected_new_directory(release / "training", args.work_root)
        trained = train_ranker(datasets["train"], datasets["validation"], model_output,
            work_root=Path(args.work_root), config=config, provenance=provenance, device="cuda")
        model, loaded_config = load_research_ranker(model_output / "checkpoint-best.local.pt",
                                                    trained["checkpoint_sha256"], device="cuda")
        if loaded_config != config:
            raise RuntimeError("Checkpoint processing configuration differs from frozen experiment")
        # Untouched family test is read only after checkpoint selection is complete.
        x_test, y_test, _ = datasets["test"]
        classification = binary_metrics(y_test, predict_scores(model, x_test, device="cuda", batch_size=config.batch_size))
        classification["sampling"] = "class_capped_observed_candidates_not_natural_prevalence"
        classification["calibrated_clinical_probabilities"] = False
        del datasets
        evidence_directory = protected_new_directory(release / "scene-evidence.local", args.work_root)
        rows = evaluate_partition(split["test"], model, config, "cuda", evidence_directory=evidence_directory)
        rows += evaluate_partition(split["test"], model, config, "cuda", controls=True, evidence_directory=evidence_directory)
        summary = summarize(rows)
        verify_sources(repo, frozen)
        report = {"schema_version": 1, "status": "research_evaluated", "experiment": EXPERIMENT,
            "clinical_accuracy_measured": False, "promoted": False, "automatic_counts_produced": 0,
            "provenance_sha256": canonical_sha256(provenance), "checkpoint_sha256": trained["checkpoint_sha256"],
            "classification": classification, "surface_benchmark": summary, "scenes": rows,
            "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_new_json(release / "evaluation.local.json", report)
        print(json.dumps({"stage": "research_evaluated", "classification": classification,
                          "surface_benchmark": summary, "promoted": False}), flush=True)
    except Exception as exc:
        write_new_json(release / "interrupted.local.json", {"status": "technical_error",
            "exception_type": type(exc).__name__, "elapsed_seconds": round(time.monotonic() - started, 3),
            "published_count": False, "resume": "Artifacts preserved; exact training resume unsupported; use a new release"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--geometries-per-family", type=int, default=4)
    parser.add_argument("--candidates-per-scene", type=int, default=96)
    run(parser.parse_args())
