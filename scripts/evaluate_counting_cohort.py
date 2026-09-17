"""Evaluate one frozen local cohort split without uploading images or claiming accuracy.

Use the acquisition application's environment to run this script. Each volume
is handled serially in a separate Python 3.12 inference worker. A new output
directory is mandatory. Held-out runs additionally require an explicitly named
clean commit and the SHA-256 of the complete committed protocol.algorithm.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


PROTOCOL_VERSION = "ethmoid-count-prototype-v1"
NULL_COUNTS = {"left": None, "right": None, "total": None}


class EvaluationError(RuntimeError):
    """A fixed diagnostic code, never an underlying protected path or header."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (OSError, ValueError):
        raise EvaluationError("LOCAL_JSON_INVALID") from None


def write_new_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contained(path: Path, root: Path, *, exists: bool = True) -> Path:
    try:
        value = path.resolve(strict=exists)
        if value == root.resolve() or not value.is_relative_to(root.resolve()):
            raise ValueError
        return value
    except (OSError, ValueError):
        raise EvaluationError("PROTECTED_PATH_REQUIRED") from None


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)
    if completed.returncode:
        raise EvaluationError("CODE_PROVENANCE_UNAVAILABLE")
    return completed.stdout.strip()


def code_provenance(pipeline_root: Path, *, require_clean: bool) -> dict:
    if Path(_git(pipeline_root, "rev-parse", "--show-toplevel")).resolve() != pipeline_root.resolve():
        raise EvaluationError("PIPELINE_ROOT_MUST_BE_CHECKOUT")
    dirty = bool(_git(pipeline_root, "status", "--porcelain=v1", "--untracked-files=all"))
    if require_clean and dirty:
        raise EvaluationError("HELD_OUT_REQUIRES_CLEAN_COMMIT")
    package = pipeline_root / "src/ethmoid_pipeline/counting"
    if not all((package / filename).is_file() for filename in ("engine.py", "worker.py", "localization.py", "protocol.json")):
        raise EvaluationError("COUNTING_ENGINE_UNAVAILABLE")
    files = {path.relative_to(pipeline_root).as_posix(): sha256(contained(path, pipeline_root)) for path in sorted(package.rglob("*.py"))}
    return {"git_commit": _git(pipeline_root, "rev-parse", "HEAD"), "git_dirty": dirty, "module_sha256": files, "modules_sha256": canonical_sha256(files)}


def protocol_snapshot(pipeline_root: Path) -> tuple[Path, dict, dict]:
    path = contained(pipeline_root / "src/ethmoid_pipeline/counting/protocol.json", pipeline_root)
    value = read_json(path)
    algorithm = value.get("algorithm") if isinstance(value, dict) else None
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL_VERSION or not isinstance(algorithm, dict) or not algorithm:
        raise EvaluationError("COMPLETE_FROZEN_ALGORITHM_REQUIRED")
    # Inspect only field names, without importing CUDA/scipy into the app runtime.
    tree = ast.parse((pipeline_root / "src/ethmoid_pipeline/counting/engine.py").read_text(encoding="utf-8"))
    config = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CountingConfig"), None)
    fields = {node.target.id for node in config.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)} if config else set()
    if set(algorithm) != fields or any(type(v) not in (int, float) or not np.isfinite(v) for v in algorithm.values()):
        raise EvaluationError("COMPLETE_FROZEN_ALGORITHM_REQUIRED")
    return path, value, {"protocol_sha256": sha256(path), "algorithm_sha256": canonical_sha256(algorithm), "algorithm": algorithm}


def _runtime_provenance(runtime_python: Path) -> dict:
    runtime = runtime_python.resolve(strict=True).parent.parent
    manifest_path = runtime / "weights-manifest.json"
    manifest = read_json(manifest_path)
    models = {item["task_id"]: item for item in manifest["models"]}
    if not {115, 775, 297, 298}.issubset(models) or manifest.get("telemetry_enabled") is not False:
        raise EvaluationError("LOCALIZER_PROVENANCE_INVALID")
    weights = runtime / "totalsegmentator/nnunet/results"
    for model in models.values():
        if not model.get("files"):
            raise EvaluationError("LOCALIZER_PROVENANCE_INVALID")
        for entry in model["files"]:
            path = contained(weights / entry["path"], weights)
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise EvaluationError("LOCALIZER_WEIGHT_CHANGED")
    probe = """import hashlib,importlib.metadata,json,os,sys
from pathlib import Path
import sitecustomize
runtime=Path(sys.prefix).resolve()
guard=Path(sitecustomize.__file__).resolve()
assert sys.version_info[:2]==(3,12)
assert guard.is_relative_to(runtime) and 'counting-runtime' in str(runtime).lower()
assert os.environ.get('ETHMOID_INFERENCE_OFFLINE')=='1'
assert json.loads((runtime/'totalsegmentator/config.json').read_text())['send_usage_stats'] is False
import socket
try:
    socket.getaddrinfo('example.com',443)
except PermissionError:
    pass
else:
    raise RuntimeError('OFFLINE_GUARD_FAILED')
print(json.dumps({'python':sys.version.split()[0],'offline_guard_sha256':hashlib.sha256(guard.read_bytes()).hexdigest(),'packages':{d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}))
"""
    environment = {**os.environ, "ETHMOID_INFERENCE_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
    environment.pop("PYTHONPATH", None)
    checked = subprocess.run([str(runtime_python), "-c", probe], capture_output=True, text=True, env=environment, check=False, timeout=60)
    if checked.returncode:
        raise EvaluationError("OFFLINE_RUNTIME_NOT_READY")
    try:
        info = json.loads(checked.stdout)
    except ValueError:
        raise EvaluationError("OFFLINE_RUNTIME_NOT_READY") from None
    return {**info, "weights_manifest_sha256": sha256(manifest_path), "weights_manifest": manifest, "python_executable_sha256": sha256(runtime_python)}


def load_cohort(manifest_path: Path, work_root: Path, split: str) -> tuple[dict, list[dict]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise EvaluationError("COHORT_INVALID")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 10:
        raise EvaluationError("FROZEN_TEN_STUDY_COHORT_REQUIRED")
    seen, source_paths, source_hashes, counts = set(), set(), set(), Counter()
    for entry in entries:
        if not re.fullmatch(r"Case[0-9]{3}", str(entry.get("case", ""))) or not re.fullmatch(r"Series[0-9]{3}", str(entry.get("series", ""))):
            raise EvaluationError("COHORT_INVALID")
        if entry["case"] in seen or entry.get("split") not in {"development", "held_out"}:
            raise EvaluationError("COHORT_SPLIT_INVALID")
        seen.add(entry["case"])
        counts[entry["split"]] += 1
        if entry.get("automated_qc") != "PASS" or entry.get("hu_max_absolute_error") != 0 or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("nifti_sha256", ""))):
            raise EvaluationError("COHORT_QC_INVALID")
        # Resolve manifest paths without opening held-out image content.
        source_path = contained(work_root / entry["nifti"], work_root, exists=False)
        if source_path in source_paths or entry["nifti_sha256"] in source_hashes:
            raise EvaluationError("COHORT_DUPLICATE_SOURCE")
        source_paths.add(source_path)
        source_hashes.add(entry["nifti_sha256"])
    if counts != {"development": 5, "held_out": 5}:
        raise EvaluationError("COHORT_SPLIT_INVALID")
    selected = [entry for entry in entries if entry["split"] == split]
    # Verify just the requested split. Held-out voxels are not opened during
    # development runs; filenames in the already-frozen manifest are sufficient.
    for entry in selected:
        source = contained(work_root / entry["nifti"], work_root)
        if not source.is_file() or sha256(source) != entry["nifti_sha256"]:
            raise EvaluationError("COHORT_SOURCE_CHANGED")
    return manifest, selected


def validate_cache(cache: Path, source: Path, source_sha256: str, work_root: Path, runtime_info: dict | None = None) -> dict | None:
    if not cache.exists():
        return None
    cache = contained(cache, work_root)
    path = cache / "localizers.local.json"
    value = read_json(path)
    if value.get("source_sha256") != source_sha256 or not value.get("tasks"):
        raise EvaluationError("LANDMARK_SOURCE_MISMATCH")
    image = nib.load(source, mmap=True)
    hashes = {}
    for task in value["tasks"]:
        if not re.fullmatch(r"[a-z_]+", task["task"]) or not task.get("masks"):
            raise EvaluationError("LANDMARK_MANIFEST_INVALID")
        for mask in task["masks"]:
            if not re.fullmatch(r"[a-z0-9_]+\.nii\.gz", mask["filename"]):
                raise EvaluationError("LANDMARK_MANIFEST_INVALID")
            mask_path = contained(cache / task["task"] / mask["filename"], cache)
            if sha256(mask_path) != mask["sha256"]:
                raise EvaluationError("LANDMARK_HASH_MISMATCH")
            localizer = nib.load(mask_path, mmap=True)
            data = np.asanyarray(localizer.dataobj)
            if localizer.shape != image.shape or not np.allclose(localizer.affine, image.affine, rtol=0, atol=1e-5) or not np.isin(data, (0, 1)).all():
                raise EvaluationError("LANDMARK_GEOMETRY_OR_VALUES_INVALID")
            hashes[mask_path.relative_to(cache).as_posix()] = mask["sha256"]
    producer_verified = bool(runtime_info and value.get("weights_manifest_sha256") == runtime_info.get("weights_manifest_sha256") and value.get("totalsegmentator_version") == runtime_info.get("packages", {}).get("totalsegmentator"))
    return {"manifest_sha256": sha256(path), "mask_sha256": hashes, "source_sha256": source_sha256, "producer_provenance": "verified" if producer_verified else "unverified", "producer_weights_manifest_sha256": value.get("weights_manifest_sha256"), "producer_totalsegmentator_version": value.get("totalsegmentator_version")}


def structured_diagnostics(result: dict) -> dict:
    values = result.get("diagnostics", {})
    if not isinstance(values, dict):
        raise EvaluationError("STRUCTURED_DIAGNOSTIC_INVALID")
    output = {}
    for name in ("air_coverage_fraction", "observed_air_voxels", "assigned_air_voxels"):
        if name not in values:
            continue
        value = values[name]
        if name == "air_coverage_fraction" and value is None:
            output[name] = None
        elif type(value) in (int, float) and np.isfinite(value) and value >= 0 and (not name.endswith("fraction") or value <= 1):
            output[name] = value
        else:
            raise EvaluationError("STRUCTURED_DIAGNOSTIC_INVALID")
    if "assigned_air_voxels" in output and "observed_air_voxels" in output and output["assigned_air_voxels"] > output["observed_air_voxels"]:
        raise EvaluationError("STRUCTURED_DIAGNOSTIC_INVALID")
    return output


def validate_completed(result: dict, directory: Path, source: Path, algorithm: dict) -> dict:
    if result.get("config") != algorithm:
        raise EvaluationError("EFFECTIVE_ALGORITHM_MISMATCH")
    cells = result.get("cells")
    counts = result.get("counts")
    if not isinstance(cells, list) or not isinstance(counts, dict):
        raise EvaluationError("COUNT_REGISTRY_INVALID")
    if not cells:
        raise EvaluationError("EMPTY_COMPLETED_COUNT_UNSUPPORTED")
    ids, sides = set(), Counter()
    for cell in cells:
        identity = cell.get("id")
        if type(identity) is not int or not 1 <= identity <= 65535 or identity in ids or cell.get("side") not in {"left", "right"} or cell.get("included") is not True or cell.get("origin") != "automatic" or type(cell.get("uncertain")) is not bool:
            raise EvaluationError("COUNT_REGISTRY_INVALID")
        ids.add(identity)
        sides[cell["side"]] += 1
    expected = {"left": sides["left"], "right": sides["right"], "total": len(cells)}
    if counts != expected or any(type(counts.get(key)) is not int for key in expected):
        raise EvaluationError("COUNT_IDENTITY_MISMATCH")
    image = nib.load(source, mmap=True)
    arrays, artifacts = {}, {}
    for key in ("cavities", "air", "region"):
        name = result.get("artifacts", {}).get(key)
        if name != f"{key}.nii.gz":
            raise EvaluationError("NATIVE_ARTIFACT_MISSING")
        path = contained(directory / name, directory)
        mask = nib.load(path, mmap=True)
        if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, rtol=0, atol=1e-5):
            raise EvaluationError("NATIVE_ARTIFACT_GEOMETRY_INVALID")
        array = np.asanyarray(mask.dataobj)
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 65535) or not np.array_equal(array, np.rint(array)):
            raise EvaluationError("NATIVE_ARTIFACT_VALUES_INVALID")
        arrays[key] = array
        artifacts[key] = {"filename": name, "sha256": sha256(path)}
    cavities, air, region = arrays["cavities"], arrays["air"], arrays["region"]
    if set(np.unique(cavities)) - {0} != ids or np.any((air != 0) & (air != cavities)) or not np.isin(region, (0, 1, 2)).all():
        raise EvaluationError("NATIVE_ARTIFACT_IDENTITY_MISMATCH")
    # Scanner X=0 is not necessarily the anatomical midline. Bilateral region
    # centers must still have left X < right X in physical RAS coordinates.
    region_x = {}
    for label in (1, 2):
        locations = np.argwhere(region == label)
        if len(locations):
            region_x[label] = float(nib.affines.apply_affine(image.affine, locations.mean(axis=0))[0])
    if len(region_x) == 2 and region_x[1] >= region_x[2]:
        raise EvaluationError("NATIVE_ARTIFACT_LATERALITY_MISMATCH")
    cavity_counts = np.bincount(cavities.ravel().astype(np.int64), minlength=65536)
    air_counts = np.bincount(air.ravel().astype(np.int64), minlength=65536)
    opaque, partial, uncertain = 0, 0, 0
    for cell in cells:
        identity = cell["id"]
        if cell.get("voxel_count") != cavity_counts[identity] or cell.get("air_voxel_count") != air_counts[identity]:
            raise EvaluationError("NATIVE_ARTIFACT_IDENTITY_MISMATCH")
        if not np.all(region[cavities == identity] == (1 if cell["side"] == "left" else 2)):
            raise EvaluationError("NATIVE_ARTIFACT_LATERALITY_MISMATCH")
        point = np.asarray(cell.get("centroid_ras_mm"), dtype=float)
        index = np.mean(np.argwhere(cavities == identity), axis=0)
        if point.shape != (3,) or not np.isfinite(point).all() or not np.allclose(point, nib.affines.apply_affine(image.affine, index), atol=1e-4, rtol=0):
            raise EvaluationError("NATIVE_ARTIFACT_CENTROID_MISMATCH")
        opaque += int(air_counts[identity] == 0)
        partial += int(0 < air_counts[identity] < cavity_counts[identity])
        uncertain += int(cell["uncertain"])
    diagnostics = {"fully_opaque_candidates": opaque, "partly_opaque_candidates": partial, "uncertain_candidates": uncertain}
    # Only structured diagnostics are copied. Never parse prose warnings as
    # measurements or convert geometric coverage into a clinical accuracy claim.
    diagnostics.update(structured_diagnostics(result))
    return {"artifacts": artifacts, "native_geometry": "PASS", "count_identity": "PASS", "diagnostics": diagnostics}


def _worker(runtime_python: Path, pipeline_root: Path, source: Path, output: Path, config: Path, cache: Path | None, log: Path, timeout: int) -> int:
    command = [str(runtime_python), "-m", "ethmoid_pipeline.counting.worker", "--input", str(source), "--output", str(output), "--config", str(config)]
    if cache is not None:
        command.extend(("--landmarks", str(cache)))
    environment = {**os.environ, "PYTHONPATH": str(pipeline_root / "src"), "PYTHONNOUSERSITE": "1", "ETHMOID_INFERENCE_OFFLINE": "1", "nnUNet_compile": "false"}
    with log.open("xb") as stream:
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with subprocess.Popen(command, cwd=pipeline_root, env=environment, stdout=stream, stderr=subprocess.STDOUT, **options) as process:
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
                raise EvaluationError("WORKER_TIMEOUT") from None


def evaluate(*, work_root: Path, pipeline_root: Path, runtime_python: Path, output: Path, split: str, expected_code_sha: str | None = None, expected_algorithm_sha256: str | None = None, expected_cohort_sha256: str | None = None, timeout_seconds: int = 1800) -> dict:
    protected = Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work"
    work_root = work_root.resolve(strict=True)
    if work_root != protected.resolve(strict=True):
        raise EvaluationError("PROTECTED_WORK_ROOT_REQUIRED")
    pipeline_root = pipeline_root.resolve(strict=True)
    if work_root.is_relative_to(pipeline_root) or pipeline_root.is_relative_to(work_root):
        raise EvaluationError("CODE_AND_MEDICAL_OUTPUT_MUST_BE_SEPARATE")
    if split not in {"development", "held_out"} or type(timeout_seconds) is not int or timeout_seconds < 1:
        raise EvaluationError("EVALUATION_OPTIONS_INVALID")
    output = contained(output, work_root, exists=False)
    if output.exists():
        raise EvaluationError("IMMUTABLE_OUTPUT_ALREADY_EXISTS")
    for parent in (work_root, *work_root.parents, *output.parents):
        if (parent / ".git").exists():
            raise EvaluationError("MEDICAL_OUTPUT_INSIDE_GIT")
    code = code_provenance(pipeline_root, require_clean=split == "held_out")
    protocol_path, protocol, protocol_info = protocol_snapshot(pipeline_root)
    if split == "held_out" and (not expected_code_sha or not expected_algorithm_sha256 or not expected_cohort_sha256):
        raise EvaluationError("EXPLICIT_HELD_OUT_FREEZE_REQUIRED")
    if expected_code_sha is not None and expected_code_sha != code["git_commit"]:
        raise EvaluationError("FROZEN_CODE_SHA_MISMATCH")
    if expected_algorithm_sha256 is not None and expected_algorithm_sha256 != protocol_info["algorithm_sha256"]:
        raise EvaluationError("FROZEN_ALGORITHM_SHA_MISMATCH")
    cohort_path = contained(work_root / "counting/cohort.v1.local.json", work_root)
    if expected_cohort_sha256 is not None and sha256(cohort_path) != expected_cohort_sha256:
        raise EvaluationError("FROZEN_COHORT_SHA_MISMATCH")
    _, entries = load_cohort(cohort_path, work_root, split)
    runtime_info = _runtime_provenance(runtime_python)
    provenance = {"schema_version": 1, "split": split, "protocol_version": PROTOCOL_VERSION, **protocol_info, "code": code, "cohort_sha256": sha256(cohort_path), "runtime": runtime_info, "evaluator_sha256": sha256(Path(__file__)), "selection_count": len(entries), "started_utc": datetime.now(timezone.utc).isoformat(), "accuracy_measured": False, "clinical_review_status": "Not established by this engineering evaluation"}
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "provenance.local.json", provenance)
    with (output / "protocol.local.json").open("xb") as stream:
        stream.write(protocol_path.read_bytes())
    attempts = []
    for entry in entries:
        case = entry["case"]
        destination = output / case
        destination.mkdir(exist_ok=False)
        attempt = {"case": case, "series": entry["series"], "split": split, "status": "failed", "counts": dict(NULL_COUNTS), "input_sha256": entry["nifti_sha256"], "clinical_review": entry.get("clinical_review", "PENDING"), "manual_slicer_qc": entry.get("manual_slicer_qc", "PENDING"), "elapsed_seconds": 0.0}
        start = time.monotonic()
        try:
            source = contained(work_root / entry["nifti"], work_root)
            if sha256(source) != entry["nifti_sha256"] or code_provenance(pipeline_root, require_clean=split == "held_out") != code or sha256(protocol_path) != protocol_info["protocol_sha256"] or sha256(cohort_path) != provenance["cohort_sha256"]:
                raise EvaluationError("PROVENANCE_CHANGED")
            cache = work_root / "counting/localizers" / case
            cache_info = validate_cache(cache, source, entry["nifti_sha256"], work_root, runtime_info)
            attempt["cached_localizers"] = cache_info
            if split == "held_out" and cache_info is not None and cache_info["producer_provenance"] != "verified":
                raise EvaluationError("HELD_OUT_CACHE_PRODUCER_UNVERIFIED")
            if cache_info is not None and cache_info["producer_provenance"] != "verified":
                attempt["warnings"] = ["Cached landmarks have verified source and mask hashes, but their producer model and software provenance was not recorded at generation."]
            returncode = _worker(runtime_python, pipeline_root, source, destination / "worker", output / "protocol.local.json", cache if cache_info else None, destination / "worker.local.log", timeout_seconds)
            attempt["worker_returncode"] = returncode
            result_path = destination / "worker/result.json"
            result = read_json(result_path)
            attempt["worker_result_sha256"] = sha256(result_path)
            if sha256(source) != entry["nifti_sha256"] or result.get("source_sha256") != entry["nifti_sha256"] or code_provenance(pipeline_root, require_clean=split == "held_out") != code or sha256(protocol_path) != protocol_info["protocol_sha256"] or sha256(cohort_path) != provenance["cohort_sha256"] or _runtime_provenance(runtime_python) != runtime_info:
                raise EvaluationError("PROVENANCE_CHANGED")
            if cache_info is not None and validate_cache(cache, source, entry["nifti_sha256"], work_root, runtime_info) != cache_info:
                raise EvaluationError("LANDMARK_PROVENANCE_CHANGED")
            generated_caches = {}
            for cache_name in ("localizers", "localizers-full-extent"):
                generated = destination / "worker" / cache_name
                if (generated / "localizers.local.json").is_file():
                    info = validate_cache(generated, source, entry["nifti_sha256"], work_root, runtime_info)
                    if split == "held_out" and info["producer_provenance"] != "verified":
                        raise EvaluationError("HELD_OUT_CACHE_PRODUCER_UNVERIFIED")
                    generated_caches[cache_name] = info
            attempt["generated_localizers"] = generated_caches
            attempt["method_version"] = result.get("method_version")
            if result.get("protocol_version") != PROTOCOL_VERSION or result.get("config") != protocol["algorithm"]:
                raise EvaluationError("EFFECTIVE_ALGORITHM_MISMATCH")
            if result.get("status") == "completed" and returncode == 0:
                attempt.update(validate_completed(result, destination / "worker", source, protocol["algorithm"]))
                attempt.update(status="completed", counts=result["counts"])
            elif result.get("status") == "failed" and returncode != 0 and result.get("counts") == NULL_COUNTS and result.get("cells") == [] and result.get("artifacts") == {}:
                failure_code = result.get("failure_code")
                if not isinstance(failure_code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,79}", failure_code):
                    raise EvaluationError("WORKER_FAILURE_INVALID")
                attempt["failure_code"] = failure_code
                attempt["diagnostics"] = structured_diagnostics(result)
            else:
                raise EvaluationError("WORKER_RESULT_INVALID")
        except EvaluationError as error:
            attempt["failure_code"] = str(error)
        except Exception:
            # Full worker stdout/stderr remains local. Never echo exceptions,
            # source paths, arbitrary warnings or patient headers to the console.
            attempt["failure_code"] = "LOCAL_EVALUATION_ERROR"
        attempt["elapsed_seconds"] = round(time.monotonic() - start, 3)
        if (destination / "worker.local.log").is_file():
            attempt["worker_log_sha256"] = sha256(destination / "worker.local.log")
        write_new_json(destination / "attempt.local.json", attempt)
        attempts.append(attempt)
    completed = [item for item in attempts if item["status"] == "completed"]
    diagnostic_keys = ("fully_opaque_candidates", "partly_opaque_candidates", "uncertain_candidates")
    summary = {"schema_version": 1, "split": split, "attempted": len(attempts), "completed": len(completed), "failed": len(attempts) - len(completed), "failure_codes": dict(sorted(Counter(item["failure_code"] for item in attempts if item["status"] != "completed").items())), "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in attempts), 3), "counts_from_completed_only": {key: sum(item["counts"][key] for item in completed) if completed else None for key in NULL_COUNTS}, "diagnostics_from_completed_only": {key: sum(item["diagnostics"][key] for item in completed) if completed else None for key in diagnostic_keys}, "accuracy_measured": False, "interpretation": "Engineering execution and consistency checks only; no reference counts or clinical accuracy measurement."}
    summary["producer_unverified_cached_cases"] = sum(bool(item.get("cached_localizers")) and item["cached_localizers"]["producer_provenance"] != "verified" for item in attempts)
    structured_coverage = [item["diagnostics"]["air_coverage_fraction"] for item in attempts if item.get("diagnostics", {}).get("air_coverage_fraction") is not None]
    if structured_coverage:
        summary["geometric_air_coverage"] = {"scope": "all attempts with structured diagnostics, including failures", "available_cases": len(structured_coverage), "minimum": min(structured_coverage), "maximum": max(structured_coverage), "mean": sum(structured_coverage) / len(structured_coverage), "is_clinical_accuracy": False}
    write_new_json(output / "summary.local.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "EthmoidComplexity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=base / "work")
    parser.add_argument("--pipeline-root", type=Path, required=True, help="Engine checkout; a clean detached worktree can freeze code while UI work continues.")
    parser.add_argument("--runtime-python", type=Path, default=base / "tools/counting-runtime/Scripts/python.exe")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory inside the protected work root; never reused.")
    parser.add_argument("--split", choices=("development", "held_out"), required=True)
    parser.add_argument("--expected-code-sha")
    parser.add_argument("--expected-algorithm-sha256")
    parser.add_argument("--expected-cohort-sha256")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(**vars(args))
    except Exception as error:
        code = str(error) if isinstance(error, EvaluationError) else "LOCAL_EVALUATION_ERROR"
        print(json.dumps({"status": "failed", "failure_code": code}), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
