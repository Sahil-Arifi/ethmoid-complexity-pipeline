"""Isolated partial-supervision research fork; no production integration."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import re
import time

import numpy as np

from . import instance_experiment as ex
from . import instance_learning as core
from .instance_connectivity import select_constraints

VERSION = "video-guided-connectivity-prototype-2"
CONFIG = {
    "spacing_mm": 0.5, "patch_size": 64, "inference_stride": 32,
    "threshold": 0.5, "minimum_native_voxels": 4, "channels": [8, 16, 32, 64],
    "steps": 128, "maximum_wall_seconds": 3600, "seed": 20261102,
    "learning_rate": 0.001, "weight_decay": 0.0001, "noise_std_hu": 20.0,
    "left_right_reflection": True, "same_margin": 0.7, "different_margin": 0.3,
    "local_loss_weight": 1.0, "connectivity_loss_weight": 1.0,
    "coverage_loss_weight": 10.0, "coverage_positive_margin": 0.9,
    "coverage_negative_margin": 0.1, "maximum_gradient_norm": 5.0,
    "initialization": "fork_preserved_closed_checkpoint",
    "graph_refresh": "full_six_head_foreground_capacity_every_step",
    "gradient": "stream_original_tiles_from_current_full_field",
    "checkpoint_rule": "final_only",
}


def _shared_geometry():
    keys = ("spacing_mm", "patch_size", "inference_stride", "threshold",
            "minimum_native_voxels", "channels", "seed")
    ex.require(all(CONFIG[k] == ex.CONFIG[k] for k in keys), "SHARED_GEOMETRY_CHANGED")


def tile_plan(shape):
    """Original tile order and per-head averaging denominators."""
    _shared_geometry()
    shape = tuple(int(n) for n in shape)
    ex.require(len(shape) == 3 and all(n > 0 for n in shape), "TILE_SHAPE_INVALID")
    weights = np.zeros((7, *shape), np.float32)
    tiles = []
    for start in itertools.product(*(ex.starts(n) for n in shape)):
        low = np.array(start); high = np.minimum(low + 64, shape)
        bounds = [(low, high)] + [(low, high - np.array(off)) for off in core.OFFSETS]
        entries = []
        for channel, (a, b) in enumerate(bounds):
            if np.any(b <= a):
                continue
            box = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
            weights[(channel, *box)] += 1
            entries.append((channel, box))
        tiles.append((low, high, entries))
    return tiles, weights


def probability_loss(probabilities, targets, known, constraints, torch):
    """Local observations plus explicitly declared pair-supervised gradients."""
    leaf = torch.tensor(np.asarray(probabilities), dtype=torch.float32, requires_grad=True)
    ex.require(targets.shape == known.shape == tuple(leaf.shape), "LOSS_SHAPE_INVALID")
    heads = []
    for channel in range(7):
        classes = []
        for label in (0, 1):
            selected = known[channel] & (targets[channel] == label)
            if np.any(selected):
                values = leaf[channel][torch.from_numpy(selected)]
                classes.append(torch.nn.functional.binary_cross_entropy(values, torch.full_like(values, label)))
        if classes:
            heads.append(torch.stack(classes).mean())
    ex.require(bool(heads), "NO_LOCAL_SUPERVISION")
    local = torch.stack(heads).mean()
    relation_losses = []
    for kind, margin in (("same", CONFIG["same_margin"]), ("different", CONFIG["different_margin"])):
        rows = [row for row in constraints if row["kind"] == kind]
        ex.require(bool(rows), "CONNECTIVITY_CLASS_MISSING")
        indices = [row["cache_flat_index"] for row in rows]
        ex.require(all(float(probabilities.reshape(-1)[i]) == row["capacity"] for i, row in zip(indices, rows)),
                   "CONSTRAINT_SCALAR_MISMATCH")
        values = leaf.reshape(-1)[indices]
        excess = margin-values if kind == "same" else values-margin
        relation_losses.append(torch.relu(excess).square().mean())
    connectivity = torch.stack(relation_losses).mean()
    pos = known[0] & (targets[0] == 1)
    neg = known[0] & (targets[0] == 0)
    ex.require(np.any(pos) and np.any(neg), "FOREGROUND_CLASSES_MISSING")
    # A single disappearing labeled voxel must not be cheap relative to a pair.
    coverage = (torch.relu(CONFIG["coverage_positive_margin"] - leaf[0][torch.from_numpy(pos)].min()).square()
                + torch.relu(leaf[0][torch.from_numpy(neg)].max() - CONFIG["coverage_negative_margin"]).square())
    loss = (CONFIG["local_loss_weight"]*local + CONFIG["connectivity_loss_weight"]*connectivity
            + CONFIG["coverage_loss_weight"]*coverage)
    loss.backward()
    ex.require(torch.isfinite(loss) and torch.isfinite(leaf.grad).all(), "LOSS_NONFINITE")
    return leaf.grad.numpy(), {"loss": float(loss.detach()), "local_loss": float(local.detach()),
                              "connectivity_loss": float(connectivity.detach()), "coverage_loss": float(coverage.detach())}


def stream_backward(hu, valid, model, torch, gradient, tiles, weights):
    """Chain the current full-field derivative through every contributing tile."""
    gradient = np.array(gradient, copy=True)
    gradient[0, ~valid] = 0  # Forward explicitly fixes unacquired foreground to zero.
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    used = 0
    for low, high, entries in tiles:
        index_parts, coefficient_parts = [], []
        for channel, box in entries:
            local_gradient = gradient[(channel, *box)]
            where = np.nonzero(local_gradient)
            if not len(where[0]):
                continue
            indices = channel*64**3 + np.ravel_multi_index(where, (64,)*3)
            coefficients = local_gradient[where] / weights[(channel, *box)][where]
            index_parts.append(indices); coefficient_parts.append(coefficients)
        if not index_parts:
            continue
        h = np.full((64,)*3, -1000, np.float32)
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        sub = tuple(slice(0, int(b-a)) for a, b in zip(low, high))
        h[sub] = hu[box]
        output = torch.sigmoid(model(torch.from_numpy(core.normalize_hu(h)[None]).to(device)))
        indices = torch.from_numpy(np.concatenate(index_parts)).to(device)
        coefficients = torch.from_numpy(np.concatenate(coefficient_parts)).to(device)
        (output.reshape(-1)[indices] * coefficients).sum().backward()
        used += 1
    ex.require(used > 0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
               "NETWORK_GRADIENT_INVALID")
    return used


def _context(manifest_path):
    manifest = ex.read_json(manifest_path)
    ex.require(set(manifest) == {"schema_version", "annotation_manifest", "annotation_sha256", "initial_checkpoint",
                                "initial_checkpoint_sha256", "protocol", "protocol_sha256"}
               and manifest["schema_version"] == 1, "TRAIN_MANIFEST_INVALID")
    for key, hash_key in (("annotation_manifest", "annotation_sha256"), ("initial_checkpoint", "initial_checkpoint_sha256"),
                          ("protocol", "protocol_sha256")):
        ex.require(ex.sha256(ex.external(manifest[key])) == manifest[hash_key], "TRAIN_INPUT_HASH_MISMATCH")
    protocol = ex.read_json(manifest["protocol"])
    ex.require(protocol.get("version") == VERSION and protocol.get("config") == CONFIG
               and protocol.get("annotation_manifest_sha256") == manifest["annotation_sha256"]
               and protocol.get("initial_checkpoint_sha256") == manifest["initial_checkpoint_sha256"], "PROTOCOL_MISMATCH")
    _preserved(protocol)
    annotation = ex.read_json(manifest["annotation_manifest"])
    old_ready = ex.inspect_readiness(annotation)
    ex.require(old_ready["training_allowed"], "PARTIAL_ANNOTATIONS_NOT_READY")
    source = ex.source_record(annotation["source"])
    _, _, hu, affine = ex.load_crop(source)
    ids, fg, _ = ex._annotation(annotation, source, hu, affine)
    grid, valid, ids, fg = ex.model_grid(hu, affine, ids, fg)
    ex.require(10 not in ids and set(np.unique(ids[ids > 0])) == {1, 2, 3, 5, 6, 7, 8}, "ANNOTATION_RELEASE_CHANGED")
    pairs = annotation["distinct_pairs"]
    ex.require({tuple(sorted(p)) for p in pairs} == {(1, 5), (2, 3), (6, 7)}, "DISTINCTIONS_CHANGED")
    readiness = {"version": VERSION, "training_allowed": True, "annotation_manifest_sha256": manifest["annotation_sha256"],
                 "protocol_sha256": manifest["protocol_sha256"], "evolving_full_graph": True,
                 "partial_supervision_only": True, "long_continuation_unmeasured": True,
                 "source_sha256": source["source_sha256"], "source_match_uncertainty": old_ready["source_match_uncertainty"],
                 "boundary_uncertainty": old_ready["boundary_uncertainty"]}
    return manifest, protocol, grid, valid, ids, fg, pairs, readiness


def _preserved(protocol):
    ex.require(isinstance(protocol.get("input_hashes"), dict) and bool(protocol["input_hashes"]), "PROTOCOL_INPUTS_MISSING")
    # Integrity inputs include committed code; artifact readers separately enforce outside-Git storage.
    ex.require(all(Path(path).is_absolute() and Path(path).is_file() and ex.sha256(path) == expected
                   for path, expected in protocol["input_hashes"].items()),
               "FROZEN_INPUT_CHANGED")


def _initial_model(manifest, torch):
    saved = torch.load(ex.external(manifest["initial_checkpoint"]), map_location="cpu", weights_only=True)
    ex.require(saved.get("version") == ex.VERSION and saved.get("model_version") == core.MODEL_VERSION
               and saved.get("config") == ex.CONFIG and saved.get("steps_completed") == 2000, "INITIAL_CHECKPOINT_INVALID")
    model = core.build_model().to("cuda"); model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model


def load_checkpoint(path, torch):
    saved = torch.load(ex.external(path), map_location="cpu", weights_only=True)
    ex.require(saved.get("version") == VERSION and saved.get("model_version") == core.MODEL_VERSION
               and saved.get("config") == CONFIG and saved.get("steps_completed") == CONFIG["steps"]
               and saved.get("complete") is True
               and all(isinstance(saved.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", saved[k]) for k in
                       ("annotation_manifest_sha256", "protocol_sha256", "initial_checkpoint_sha256")),
               "CHECKPOINT_PROTOCOL_MISMATCH")
    model = core.build_model(); model.load_state_dict(saved["state_dict"], strict=True)
    return model, saved


def _step(grid, valid, ids, fg, pairs, model, torch, tiles, weights):
    probabilities = ex.tiled_probabilities(grid, model, torch)
    probabilities[0, ~valid] = 0
    constraints = select_constraints(probabilities, ids, pairs)
    targets, known = core.build_targets(ids, distinct_pairs=pairs, foreground_labels=fg)
    gradient, losses = probability_loss(probabilities, targets, known, constraints, torch)
    used = stream_backward(grid, valid, model, torch, gradient, tiles, weights)
    return {**losses, "gradient_tiles": used, "constraints": constraints}


def preflight(manifest_path, output_path):
    manifest, protocol, grid, valid, ids, fg, pairs, readiness = _context(manifest_path)
    ex.require(ex.external(output_path, exists=False) == ex.external(protocol.get("preflight_output"), exists=False),
               "PREFLIGHT_OUTPUT_MISMATCH")
    output = ex.new_output(output_path); ex.write_json(output/"readiness.local.json", readiness)
    torch = ex.deterministic_torch(); torch.set_num_threads(1)
    model = _initial_model(manifest, torch)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tiles, weights = tile_plan(grid.shape)
    started = time.monotonic()
    step = _step(grid, valid, ids, fg, pairs, model, torch, tiles, weights)
    torch.cuda.synchronize(); elapsed = time.monotonic()-started
    ex.require(all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()), "PREFLIGHT_MODEL_CHANGED")
    _preserved(protocol)
    result = {"version": VERSION, "status": "PASS" if elapsed*CONFIG["steps"] < .8*CONFIG["maximum_wall_seconds"] else "FAIL_BUDGET",
              "step_seconds": elapsed, "projected_training_seconds": elapsed*CONFIG["steps"], "step": step,
              "weights_unchanged": True, "optimizer_ran": False, "protocol_sha256": manifest["protocol_sha256"]}
    ex.write_json(output/"preflight.local.json", result)
    return result


def train(manifest_path, output_path):
    manifest, protocol, grid, valid, ids, fg, pairs, readiness = _context(manifest_path)
    preflight_record = ex.read_json(ex.external(protocol.get("preflight_output"))/"preflight.local.json")
    ex.require(preflight_record.get("version") == VERSION and preflight_record.get("status") == "PASS"
               and preflight_record.get("weights_unchanged") is True and preflight_record.get("optimizer_ran") is False
               and preflight_record.get("protocol_sha256") == manifest["protocol_sha256"], "PREFLIGHT_GATE_FAILED")
    output = ex.new_output(output_path); ex.write_json(output/"readiness.local.json", readiness)
    torch = ex.deterministic_torch(); torch.set_num_threads(1)
    model = _initial_model(manifest, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    rng = np.random.default_rng(CONFIG["seed"]); tiles, weights = tile_plan(grid.shape)
    started = time.monotonic(); completed = 0
    try:
        for step in range(CONFIG["steps"]):
            if time.monotonic()-started >= CONFIG["maximum_wall_seconds"]:
                break
            reflected = bool(rng.random() < .5)
            transform = lambda a: a[::-1].copy() if reflected else a
            h = transform(grid) + rng.normal(0, CONFIG["noise_std_hu"], grid.shape).astype(np.float32)
            v = transform(valid); h[~v] = -1000
            row = _step(h, v, transform(ids), transform(fg), pairs, model, torch, tiles, weights)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["maximum_gradient_norm"], error_if_nonfinite=True)
            if time.monotonic()-started >= CONFIG["maximum_wall_seconds"]:
                break
            optimizer.step(); completed += 1
            row.update({"step": completed, "reflection": reflected, "unclipped_gradient_norm": float(norm),
                        "elapsed_seconds": time.monotonic()-started})
            with (output/"progress.local.jsonl").open("a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"step={completed}/{CONFIG['steps']} loss={row['loss']:.6f} elapsed={row['elapsed_seconds']:.1f}", flush=True)
        torch.cuda.synchronize()
        complete = completed == CONFIG["steps"] and time.monotonic()-started <= CONFIG["maximum_wall_seconds"]
        _preserved(protocol)
        saved = {"version": VERSION, "model_version": core.MODEL_VERSION, "config": CONFIG,
                 "steps_completed": completed, "complete": complete,
                 "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "annotation_manifest_sha256": manifest["annotation_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                 "initial_checkpoint_sha256": manifest["initial_checkpoint_sha256"]}
        checkpoint = output/("final-checkpoint.local.pt" if complete else "incomplete-checkpoint.local.pt")
        torch.save(saved, checkpoint)
        result = {"version": VERSION, "status": "COMPLETE" if complete else "INCOMPLETE_BUDGET",
                  "steps_completed": completed, "checkpoint": str(checkpoint), "checkpoint_sha256": ex.sha256(checkpoint),
                  "elapsed_seconds": time.monotonic()-started, "inputs_unchanged": True,
                  "protocol_sha256": manifest["protocol_sha256"], "config": CONFIG}
        ex.write_json(output/"training.local.json", result)
        return result
    except Exception as error:
        ex.write_json(output/"failure.local.json", {"version": VERSION, "status": "FAILED", "steps_completed": completed,
                                                   "error_code": ex.failure_code(error), "no_inference_allowed": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "infer", "mechanism", "report"))
    parser.add_argument("manifest"); parser.add_argument("output")
    args = parser.parse_args()
    if args.command in {"preflight", "train"}:
        function = preflight if args.command == "preflight" else train
    else:
        from . import connectivity_evaluation
        function = getattr(connectivity_evaluation, args.command)
    function(args.manifest, args.output)


if __name__ == "__main__":
    main()
