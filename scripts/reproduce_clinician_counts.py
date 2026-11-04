"""Repeat source-bound automatic counts; references are used only after inference.

All manifests, images, logs and results stay outside the checkout. These are
development reproduction checks, not an independent clinical evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NULL_COUNTS = {"left": None, "right": None, "total": None}
ARTIFACTS = {name: name + ".nii.gz" for name in ("cavities", "air", "region")}
METHODS = {"v2": ("native-bone-watershed-0.2.1", "ethmoid-count-prototype-v2"),
           "research-v5": ("research-native-0.5.0", "ethmoid-count-research-v5")}
_spec = importlib.util.spec_from_file_location("_reproduction_native_checks", ROOT / "scripts/evaluate_counting_cohort.py")
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)


class ReproductionError(ValueError):
    """Fixed diagnostic code without protected file contents."""


def require(condition, code):
    if not condition:
        raise ReproductionError(code)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def external(path, *, exists=True):
    path = Path(path).resolve(strict=exists)
    require(path != ROOT and not path.is_relative_to(ROOT), "EXTERNAL_DATA_PATH_REQUIRED")
    require(not any((parent / ".git").exists() for parent in (path, *path.parents)), "EXTERNAL_DATA_PATH_REQUIRED")
    return path


def code_hashes():
    files = sorted((ROOT / "src/ethmoid_pipeline/counting").glob("*.py"))
    files += sorted((ROOT / "src/ethmoid_pipeline/counting").glob("*.json"))
    files.append(Path(__file__).resolve())
    files.append(ROOT / "scripts/evaluate_counting_cohort.py")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def preflight(manifest):
    manifest = external(manifest)
    data = read_json(manifest)
    require(isinstance(data, dict) and isinstance(data.get("entries"), list) and data["entries"], "MANIFEST_INVALID")
    seen = set()
    sources = set()
    protected = {str(manifest): sha256(manifest)}
    for entry in data["entries"]:
        require(isinstance(entry, dict) and {"case_id", "input", "source_sha256", "landmarks", "acquisition", "reference"} <= set(entry)
                and isinstance(entry["input"], str) and isinstance(entry["landmarks"], str), "MANIFEST_INVALID")
        case = entry.get("case_id")
        require(isinstance(case, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", case)
                and case.upper() not in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}, "CASE_ID_INVALID")
        require(case.casefold() not in seen, "DUPLICATE_CASE")
        seen.add(case.casefold())
        reference = entry.get("reference")
        require(isinstance(reference, dict) and set(reference) == {"left", "right"}
                and all(type(v) is int and v >= 0 for v in reference.values()), "REFERENCE_INVALID")
        digest = entry.get("source_sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest), "SOURCE_HASH_INVALID")
        source = external(entry["input"])
        require(source.is_file() and sha256(source) == digest, "SOURCE_HASH_MISMATCH")
        require(source not in sources, "DUPLICATE_SOURCE")
        sources.add(source)
        acquisition = entry.get("acquisition")
        require(isinstance(acquisition, dict) and acquisition.get("source_sha256") == digest, "ACQUISITION_SOURCE_MISMATCH")
        cache = external(entry["landmarks"])
        require(cache.is_dir(), "LANDMARKS_UNAVAILABLE")
        for path in [source, *sorted(p for p in cache.rglob("*") if p.is_file())]:
            require(not path.resolve().is_relative_to(ROOT), "EXTERNAL_DATA_PATH_REQUIRED")
            protected[str(path)] = sha256(path)
        entry["input"], entry["landmarks"] = str(source), str(cache)
    return manifest, data["entries"], protected


def run_worker(command, environment):
    return subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                          check=False, timeout=3600)


def validated_prediction(result, directory, source_hash, source, method, algorithm):
    require(isinstance(result, dict) and result.get("source_sha256") == source_hash, "RESULT_SOURCE_MISMATCH")
    require((result.get("method_version"), result.get("protocol_version")) == METHODS[method], "WORKER_METHOD_MISMATCH")
    require(result.get("config") == algorithm, "WORKER_CONFIG_MISMATCH")
    if result.get("status") == "failed":
        require(result.get("counts") == NULL_COUNTS and result.get("cells") == [] and result.get("artifacts") == {}, "INVALID_WITHHELD_RESULT")
        return "withheld", {}, None
    require(result.get("status") == "completed", "WORKER_STATUS_INVALID")
    counts, cells = result.get("counts"), result.get("cells")
    require(isinstance(counts, dict) and set(counts) == set(NULL_COUNTS)
            and all(type(v) is int and v >= 0 for v in counts.values())
            and counts["total"] == counts["left"] + counts["right"], "COUNTS_INVALID")
    require(isinstance(cells, list) and all(isinstance(c, dict) for c in cells), "COUNT_REGISTRY_MISMATCH")
    ids = [c.get("id") for c in cells]
    sides = Counter(c.get("side") for c in cells)
    require(all(type(i) is int and i > 0 for i in ids) and len(set(ids)) == len(ids) == counts["total"]
            and set(sides) <= {"left", "right"} and all(c.get("included") is True for c in cells)
            and all(sides[side] == counts[side] for side in ("left", "right")), "COUNT_REGISTRY_MISMATCH")
    require(result.get("artifacts") == ARTIFACTS, "ARTIFACT_SET_INVALID")
    hashes = {}
    for name, filename in ARTIFACTS.items():
        path = directory / filename
        require(path.is_file() and not path.is_symlink(), "ARTIFACT_UNAVAILABLE")
        artifact = nib.load(path)
        require(np.issubdtype(artifact.get_data_dtype(), np.integer), "ARTIFACT_DTYPE_INVALID")
        hashes[name] = sha256(path)
    checks.validate_completed(result, directory, Path(source), algorithm)
    cavities = np.asanyarray(nib.load(directory / ARTIFACTS["cavities"]).dataobj)
    air = np.asanyarray(nib.load(directory / ARTIFACTS["air"]).dataobj)
    require(np.array_equal(air, np.where(nib.load(source).get_fdata(dtype=np.float32) < algorithm["air_hu"], cavities, 0)), "NATIVE_HU_AIR_MISMATCH")
    return "available", hashes, canonical_hash({"counts": counts, "cells": cells, "artifacts": hashes})


def unchanged_inputs(protected, initial_code):
    try:
        return all(Path(path).is_file() and sha256(path) == digest for path, digest in protected.items()) and code_hashes() == initial_code
    except OSError:
        return False


def run(manifest, output, *, repeat=3, method="research-v5", config=None, worker=run_worker):
    require(type(repeat) is int and repeat >= 1, "REPEAT_INVALID")
    require(method in METHODS, "METHOD_INVALID")
    output = external(output, exists=False)
    require(not output.exists(), "OUTPUT_EXISTS")
    manifest, entries, protected = preflight(manifest)
    config_value = read_json(config) if config else None
    require(config_value is None or isinstance(config_value, dict), "CONFIG_INVALID")
    if config_value is not None and "protocol_version" in config_value:
        require(config_value["protocol_version"] == METHODS[method][1], "CONFIG_PROTOCOL_MISMATCH")
    from ethmoid_pipeline.counting.engine import CountingConfig, CountingError
    try:
        algorithm = asdict(CountingConfig.parse(config_value))
    except CountingError:
        raise ReproductionError("CONFIG_INVALID") from None
    if config:
        protected[str(Path(config).resolve())] = sha256(config)
    initial_code = code_hashes()
    output.mkdir(parents=True)
    config_snapshot = output / "config.local.json"
    write_json(config_snapshot, algorithm)
    protected[str(config_snapshot)] = sha256(config_snapshot)
    write_json(output / "inputs.local.json", {"entries": entries})
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1",
                   "ETHMOID_INFERENCE_OFFLINE": "1", "nnUNet_compile": "false"}
    records = []
    for repetition in range(1, repeat + 1):
        for entry in entries:
            directory = output / f"repeat-{repetition:03}" / entry["case_id"]
            directory.mkdir(parents=True)
            acquisition = directory / "acquisition.local.json"
            write_json(acquisition, entry["acquisition"])
            protected[str(acquisition)] = sha256(acquisition)
            reference = {**entry["reference"], "total": sum(entry["reference"].values())}
            row = {"case_id": entry["case_id"], "repeat": repetition, "reference": reference,
                   "prediction_state": "technical_failure", "counts": dict(NULL_COUNTS), "exact": False,
                   "deltas": dict(NULL_COUNTS), "signature": None, "artifact_sha256": {}}
            command = [sys.executable, "-m", "ethmoid_pipeline.counting.worker", "--input", entry["input"],
                       "--output", str(directory / "worker"), "--method", method,
                       "--landmarks", entry["landmarks"], "--acquisition", str(acquisition)]
            command.extend(["--config", str(config_snapshot)])
            try:
                require(sha256(entry["input"]) == entry["source_sha256"], "SOURCE_HASH_MISMATCH")
                completed = worker(command, environment)
                (directory / "stdout.local.log").write_text(completed.stdout or "", encoding="utf-8")
                (directory / "stderr.local.log").write_text(completed.stderr or "", encoding="utf-8")
                row["exit_code"] = completed.returncode
                result_path = directory / "worker/result.json"
                require(result_path.is_file(), "WORKER_RESULT_MISSING")
                result = read_json(result_path)
                require(sha256(entry["input"]) == entry["source_sha256"], "SOURCE_CHANGED_DURING_RUN")
                state, hashes, signature = validated_prediction(result, result_path.parent, entry["source_sha256"], entry["input"], method, algorithm)
                require(completed.returncode == (0 if state == "available" else 2), "WORKER_EXIT_MISMATCH")
                require(sha256(entry["input"]) == entry["source_sha256"], "SOURCE_CHANGED_DURING_RUN")
                row.update(prediction_state=state, counts=result["counts"], signature=signature,
                           artifact_sha256=hashes, result_sha256=sha256(result_path), failure_code=result.get("failure_code"))
                if state == "available":
                    row.update(exact=result["counts"] == reference,
                               deltas={k: result["counts"][k] - reference[k] for k in reference})
            except Exception as error:
                (directory / "runner-error.local.log").write_text(traceback.format_exc(), encoding="utf-8")
                row["failure_code"] = str(error) if isinstance(error, (ReproductionError, checks.EvaluationError)) else "RUNNER_TECHNICAL_FAILURE"
            write_json(directory / "comparison.local.json", row)
            records.append(row)
    unchanged = unchanged_inputs(protected, initial_code)
    cases = []
    for entry in entries:
        rows = [row for row in records if row["case_id"] == entry["case_id"]]
        available = all(row["prediction_state"] == "available" for row in rows)
        cases.append({"case_id": entry["case_id"], "exact_runs": sum(row["exact"] for row in rows),
                      "repeatable": None if repeat == 1 else available and len({row["signature"] for row in rows}) == 1})
    summary = {"schema_version": 1, "purpose": "development_reference_reproduction", "method": method,
               "repeats": repeat, "scheduled_runs": len(entries) * repeat, "exact_runs": sum(row["exact"] for row in records),
               "withheld_runs": sum(row["prediction_state"] == "withheld" for row in records),
               "technical_failures": sum(row["prediction_state"] == "technical_failure" for row in records),
               "inputs_unchanged": unchanged, "cases": cases, "records": records}
    summary["success"] = unchanged and summary["exact_runs"] == summary["scheduled_runs"] and all(c["repeatable"] is True for c in cases)
    write_json(output / "provenance.local.json", {"manifest_sha256": protected[str(manifest)],
        "config_sha256": sha256(config_snapshot), "algorithm": algorithm, "code_sha256": initial_code,
        "sources": {e["case_id"]: e["source_sha256"] for e in entries}, "protected_inputs": protected,
        "python": sys.version, "executable": sys.executable, "inputs_unchanged": unchanged})
    write_json(output / "summary.local.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--method", choices=tuple(METHODS), default="research-v5")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = run(args.manifest, args.output, repeat=args.repeat, method=args.method, config=args.config)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure_code": str(error) if isinstance(error, ReproductionError) else "PREFLIGHT_FAILED"}))
        return 2
    print(json.dumps({k: summary[k] for k in ("scheduled_runs", "exact_runs", "withheld_runs", "technical_failures", "inputs_unchanged", "success")}))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
