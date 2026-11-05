"""Generate fresh geometries and exercise the complete isolated research CLI.

Uses no patient volumes, old benchmarks, saved weights or reference-driven tuning.
Run with the existing CUDA Python and a sealed, isolated partition runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ethmoid_pipeline.counting import instance_experiment as safe
from ethmoid_pipeline.counting.boundary_learning import VERSION
from ethmoid_pipeline.counting.boundary_io import new_output


def run(output, runtime, steps=200):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
    from boundary_synthetic_fixture import create_release
    root = new_output(output)
    release_path = create_release(root/"generated")
    cli = Path(__file__).with_name("boundary_cell_pipeline.py")
    transcript = []

    def command(name, manifest, destination):
        completed = subprocess.run([sys.executable, str(cli), name, "--manifest", str(manifest),
                                    "--output", str(destination)], capture_output=True, text=True,
                                   timeout=3700 if name == "train" else 1100)
        transcript.append({"command": name, "exit_code": completed.returncode})
        safe.require(completed.returncode == 0, "SYNTHETIC_COMMAND_FAILED")

    command("export", release_path, root/"export")
    exported = root/"export"/"release.local.json"
    command("readiness", exported, root/"readiness")
    train_request = root/"train-request.local.json"
    safe.write_json(train_request, {"schema_version": 1, "version": VERSION,
        "dataset": {"path": str(exported), "sha256": safe.sha256(exported)},
        "purpose": "synthetic_engineering", "synthetic_steps": steps})
    command("train", train_request, root/"train")
    training = safe.read_json(root/"train"/"training.local.json")
    checkpoint = root/"train"/"selected-checkpoint.local.pt"
    release = safe.read_json(exported)
    from ethmoid_pipeline.counting.boundary_data import load_sample
    entries = []
    # Freeze the checkpoint before any reference masks/counts are read for scoring.
    for index, sample in enumerate(release["samples"]):
        request = root/f"inference-{index}.local.json"
        safe.write_json(request, {"schema_version": 1, "version": VERSION, "case_id": sample["sample_id"],
            "source": sample["source"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": safe.sha256(checkpoint),
            "partition_runtime": {"path": str(runtime), "sha256": safe.sha256(runtime)}})
        predictions = []
        for repeat in range(2):
            destination = root/f"prediction-{index}-{repeat}"
            command("infer", request, destination)
            result = destination/"prediction.local.json"
            predictions.append({"path": str(result), "sha256": safe.sha256(result)})
        hu, affine, truth = load_sample(sample)
        _, reference_counts = safe.label_registry(truth["instance_ids"], hu, affine, sample["source"]["midline_ras_x"])
        entries.append({"case_id": sample["sample_id"], "source_sha256": sample["source"]["source_sha256"],
                        "predictions": predictions, "sample": sample, "reference_counts": reference_counts})
    evaluation_request = root/"evaluation-request.local.json"
    safe.write_json(evaluation_request, {"schema_version": 1, "version": VERSION, "entries": entries})
    command("evaluate", evaluation_request, root/"evaluation")
    evaluation = safe.read_json(root/"evaluation"/"evaluation.local.json")
    accuracy = all(c.get("instances", {}).get("matched_instances") == 2
                   and c["instances"].get("missed_instances") == 0
                   and c["instances"].get("false_instances") == 0
                   and c["instances"].get("merged_predicted_instances") == 0
                   and c["instances"].get("split_reference_instances") == 0
                   and all(match["iou"] >= .9 for match in c["instances"].get("matches", []))
                   for c in evaluation["cases"])
    counts_pass = evaluation["exact_bilateral_studies"] == len(entries)
    repeat_pass = evaluation["repeatable_studies"] == len(entries)
    gates = {"two_fresh_process_repeatability": repeat_pass,
             "two_instances_each_iou_at_least_0_9_no_merge_split_miss_extra": accuracy,
             "exact_generated_bilateral_counts": counts_pass}
    result = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
              "scope": "synthetic_engineering_only", "version": VERSION,
              "training_steps": training["steps_completed"], "selected_step": training["selected_step"],
              "checkpoint_sha256": safe.sha256(checkpoint), "commands": transcript,
              "repeatable_studies": evaluation["repeatable_studies"], "cases": evaluation["cases"],
              "exact_sides": evaluation["exact_sides"], "exact_bilateral_studies": evaluation["exact_bilateral_studies"],
              "anatomical_accuracy": "NOT_ESTABLISHED"}
    safe.write_json(root/"engineering.local.json", result)
    safe.require(all(gates.values()), "SYNTHETIC_INSTANCE_RECOVERY_FAILED")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    try:
        result = run(args.output, safe.external(args.runtime), args.steps)
        print(json.dumps({k: result[k] for k in ("status", "training_steps", "selected_step", "repeatable_studies")}))
    except Exception as exc:
        print(safe.failure_code(exc), file=sys.stderr)
        raise SystemExit(2)
