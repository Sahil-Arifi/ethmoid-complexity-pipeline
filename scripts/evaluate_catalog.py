"""Frozen all-series evaluation; protected local artifacts, no accuracy claims."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from collections import Counter

import nibabel as nib
import numpy as np

_spec = importlib.util.spec_from_file_location("_cohort_checks", Path(__file__).with_name("evaluate_counting_cohort.py"))
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)
REGRESSION_STUDIES = {"Case001", "Case002", "Case003", "Case004", "Case005", "Case006", "Case009", "Case015", "Case021", "Case025"}


def quality_helpers(pipeline: Path):
    """Load only the acquisition helpers from the verified counting checkout."""
    path = checks.contained(pipeline / "src/ethmoid_pipeline/counting/quality.py", pipeline)
    spec = importlib.util.spec_from_file_location("_frozen_counting_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_acquisition(entry: dict, helpers) -> dict:
    try:
        value = helpers.acquisition_snapshot(entry["acquisition"], entry["source_sha256"])
    except (ValueError, TypeError, KeyError):
        raise checks.EvaluationError("CATALOG_ACQUISITION_INVALID") from None
    if any(value[name] != entry[name] for name in ("study_label", "series_label")):
        raise checks.EvaluationError("CATALOG_ACQUISITION_IDENTITY_MISMATCH")
    return value


def load_manifest(path: Path, work: Path, split: str) -> list[dict]:
    value = checks.read_json(path)
    entries = value.get("entries")
    if value.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise checks.EvaluationError("CATALOG_MANIFEST_INVALID")
    identities, sources = set(), set()
    for entry in entries:
        key = (entry.get("study_label"), entry.get("series_label"))
        if not re.fullmatch(r"Case\d{3}", str(key[0])) or not re.fullmatch(r"Series\d{3}", str(key[1])) or key in identities:
            raise checks.EvaluationError("CATALOG_IDENTITY_INVALID")
        identities.add(key)
        source = checks.contained(work / entry["nifti"], work, exists=False)
        if source in sources or not re.fullmatch("[a-f0-9]{64}", entry.get("source_sha256", "")):
            raise checks.EvaluationError("CATALOG_SOURCE_INVALID")
        sources.add(source)
        expected_split = "regression" if key[0] in REGRESSION_STUDIES else "acceptance"
        if entry.get("split") != expected_split or entry.get("automated_qc") != "PASS":
            raise checks.EvaluationError("CATALOG_SPLIT_OR_QC_INVALID")
        if entry.get("acquisition", {}).get("source_sha256") != entry["source_sha256"]:
            raise checks.EvaluationError("ACQUISITION_SOURCE_MISMATCH")
    selected = [e for e in entries if split == "all" or e["split"] == split]
    if not selected:
        raise checks.EvaluationError("EMPTY_EVALUATION_SPLIT")
    for entry in selected:
        if checks.sha256(checks.contained(work / entry["nifti"], work)) != entry["source_sha256"]:
            raise checks.EvaluationError("CATALOG_SOURCE_CHANGED")
    return selected


def worker(runtime, pipeline, source, output, protocol, acquisition, cache, log, timeout):
    command = [str(runtime), "-m", "ethmoid_pipeline.counting.worker", "--input", str(source),
               "--output", str(output), "--config", str(protocol), "--acquisition", str(acquisition)]
    if cache:
        command.extend(["--landmarks", str(cache)])
    environment = {**os.environ, "PYTHONPATH": str(pipeline / "src"), "PYTHONNOUSERSITE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "ETHMOID_INFERENCE_OFFLINE": "1", "nnUNet_compile": "false"}
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with log.open("xb") as stream, subprocess.Popen(command, cwd=pipeline, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **options) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            raise checks.EvaluationError("WORKER_TIMEOUT") from None


def summarize(attempts):
    def group(items):
        return {"attempted": len(items), "completed": sum(a["status"] == "completed" for a in items),
                "withheld": sum(a["status"] != "completed" for a in items),
                "failure_codes": dict(Counter(a["failure_code"] for a in items if a["status"] != "completed")),
                "elapsed_seconds": round(sum(a["elapsed_seconds"] for a in items), 3)}
    return {**group(attempts), "studies": len({a["study_label"] for a in attempts}),
            "by_thickness_mm": {str(k): group([a for a in attempts if a["slice_thickness_mm"] == k])
                                for k in sorted({a["slice_thickness_mm"] for a in attempts}, key=str)},
            "uncertain_candidates": sum(a.get("diagnostics", {}).get("uncertain_candidates", 0) for a in attempts if a["status"] == "completed"),
            "accuracy_measured": False, "interpretation": "Execution and consistency, not anatomical accuracy."}


def evaluate(*, work_root, pipeline_root, runtime_python, manifest, output, split,
             expected_code_sha, expected_manifest_sha256, expected_protocol_sha256, timeout_seconds=1800):
    work = work_root.resolve(strict=True)
    protected = (Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work").resolve(strict=True)
    if work != protected or timeout_seconds < 1:
        raise checks.EvaluationError("PROTECTED_WORK_ROOT_REQUIRED")
    pipeline = pipeline_root.resolve(strict=True)
    output = checks.contained(output, work, exists=False)
    manifest = checks.contained(manifest, work)
    if output.exists() or pipeline.is_relative_to(work) or work.is_relative_to(pipeline):
        raise checks.EvaluationError("IMMUTABLE_SEPARATE_OUTPUT_REQUIRED")
    code = checks.code_provenance(pipeline, require_clean=True)
    protocol_path = pipeline / "src/ethmoid_pipeline/counting/protocol.json"
    protocol = checks.read_json(protocol_path)
    if code["git_commit"] != expected_code_sha or checks.sha256(manifest) != expected_manifest_sha256 or checks.sha256(protocol_path) != expected_protocol_sha256:
        raise checks.EvaluationError("FREEZE_MISMATCH")
    if protocol.get("protocol_version") != "ethmoid-count-prototype-v2" or not protocol.get("algorithm"):
        raise checks.EvaluationError("V2_PROTOCOL_REQUIRED")
    tree = ast.parse((pipeline / "src/ethmoid_pipeline/counting/engine.py").read_text(encoding="utf-8"))
    config = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CountingConfig"), None)
    fields = {n.target.id for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)} if config else set()
    if set(protocol["algorithm"]) != fields or any(type(v) not in (int, float) or not np.isfinite(v) for v in protocol["algorithm"].values()):
        raise checks.EvaluationError("COMPLETE_FROZEN_ALGORITHM_REQUIRED")
    entries = load_manifest(manifest, work, split)
    quality = quality_helpers(pipeline)
    for entry in entries:
        normalized_acquisition(entry, quality)
    runtime = checks._runtime_provenance(runtime_python)
    output.mkdir(parents=True)
    checks.write_new_json(output / "protocol.local.json", protocol)
    checks.write_new_json(output / "provenance.local.json", {"code": code, "runtime": runtime,
            "manifest_sha256": expected_manifest_sha256, "protocol_sha256": expected_protocol_sha256,
            "evaluator_sha256": checks.sha256(Path(__file__)), "split": split, "accuracy_measured": False})
    attempts = []
    for entry in entries:
        name = entry["study_label"] + "_" + entry["series_label"]
        directory = output / name
        directory.mkdir()
        acquisition = normalized_acquisition(entry, quality)
        attempt = {"study_label": entry["study_label"], "series_label": entry["series_label"],
                   "slice_thickness_mm": acquisition.get("slice_thickness_mm"), "split": entry["split"],
                   "status": "failed", "counts": dict(checks.NULL_COUNTS), "source_sha256": entry["source_sha256"],
                   "acquisition": acquisition, "quality": quality.resolution_quality(acquisition)}
        started = time.monotonic()
        try:
            # A later source may disappear while earlier workers run. Keep its
            # withheld attempt and the remaining series in the final accounting.
            source = checks.contained(work / entry["nifti"], work)
            checks.write_new_json(directory / "acquisition.local.json", acquisition)
            expected_quality = quality.resolution_quality(acquisition, nib.load(source).affine)
            attempt["quality"] = expected_quality
            def unchanged():
                if checks.code_provenance(pipeline, require_clean=True) != code or checks.sha256(manifest) != expected_manifest_sha256 or checks.sha256(protocol_path) != expected_protocol_sha256 or checks.sha256(source) != entry["source_sha256"]:
                    raise checks.EvaluationError("PROVENANCE_CHANGED")
            unchanged()
            cache = checks.contained(work / entry["localizer_cache"], work) if entry.get("localizer_cache") else None
            cache_info = checks.validate_cache(cache, source, entry["source_sha256"], work, runtime) if cache else None
            if cache_info and cache_info["producer_provenance"] != "verified":
                raise checks.EvaluationError("UNVERIFIED_CACHE_PRODUCER")
            attempt["cached_localizers"] = cache_info
            code_return = worker(runtime_python, pipeline, source, directory / "worker", output / "protocol.local.json",
                    directory / "acquisition.local.json", cache, directory / "worker.local.log", timeout_seconds)
            result_path = directory / "worker/result.json"
            result = checks.read_json(result_path)
            attempt["result_sha256"] = checks.sha256(result_path)
            unchanged()
            if checks._runtime_provenance(runtime_python) != runtime:
                raise checks.EvaluationError("RUNTIME_CHANGED")
            if cache and checks.validate_cache(cache, source, entry["source_sha256"], work, runtime) != cache_info:
                raise checks.EvaluationError("CACHE_CHANGED")
            generated = {}
            for cache_name in ("localizers", "localizers-full-extent"):
                cache_path = directory / "worker" / cache_name
                if (cache_path / "localizers.local.json").is_file():
                    generated[cache_name] = checks.validate_cache(cache_path, source, entry["source_sha256"], work, runtime)
                    if generated[cache_name]["producer_provenance"] != "verified":
                        raise checks.EvaluationError("UNVERIFIED_CACHE_PRODUCER")
            attempt["generated_localizers"] = generated
            if result.get("source_sha256") != entry["source_sha256"] or result.get("protocol_version") != protocol["protocol_version"] or result.get("config") != protocol["algorithm"]:
                raise checks.EvaluationError("WORKER_PROVENANCE_MISMATCH")
            if result.get("acquisition") != acquisition:
                raise checks.EvaluationError("WORKER_ACQUISITION_MISMATCH")
            if result.get("quality") != expected_quality:
                raise checks.EvaluationError("WORKER_QUALITY_MISMATCH")
            attempt["method_version"] = result.get("method_version")
            if code_return == 0 and result.get("status") == "completed":
                validated = checks.validate_completed(result, directory / "worker", source, protocol["algorithm"])
                for kind in ("cavities", "air"):
                    if nib.load(directory / "worker" / result["artifacts"][kind]).get_data_dtype() != np.dtype("uint16"):
                        raise checks.EvaluationError("NATIVE_DTYPE_INVALID")
                attempt.update(validated, status="completed", counts=result["counts"])
            elif code_return != 0 and result.get("status") == "failed" and result.get("counts") == checks.NULL_COUNTS and result.get("cells") == [] and result.get("artifacts") == {} and re.fullmatch(r"[A-Z][A-Z0-9_]{1,79}", str(result.get("failure_code", ""))):
                attempt["failure_code"] = result["failure_code"]
                attempt["diagnostics"] = checks.structured_diagnostics(result)
            else:
                raise checks.EvaluationError("WORKER_RESULT_INVALID")
        except checks.EvaluationError as exc:
            attempt["failure_code"] = str(exc)
        except Exception:
            attempt["failure_code"] = "LOCAL_EVALUATION_ERROR"
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if (directory / "worker.local.log").exists():
            attempt["worker_log_sha256"] = checks.sha256(directory / "worker.local.log")
        checks.write_new_json(directory / "attempt.local.json", attempt)
        attempts.append(attempt)
        print(json.dumps({"attempted": len(attempts), "planned": len(entries), "completed": sum(a["status"] == "completed" for a in attempts)}), flush=True)
    summary = summarize(attempts)
    checks.write_new_json(output / "summary.local.json", summary)
    return summary


def main():
    base = Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=base / "work")
    parser.add_argument("--runtime-python", type=Path, default=base / "tools/counting-runtime/Scripts/python.exe")
    for name in ("pipeline-root", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-code-sha", "expected-manifest-sha256", "expected-protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("regression", "acceptance", "all"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    try:
        summary = evaluate(**vars(parser.parse_args()))
    except checks.EvaluationError as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc)}))
        return 2
    print(json.dumps(summary))
    return 0 if summary["withheld"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
