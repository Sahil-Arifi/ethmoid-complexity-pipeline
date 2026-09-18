"""Frozen v4 development experiments; original scans and prior results are read-only.

Selections describe archived diagnostic outcomes, never patient-specific policies.
Run serially from a clean frozen checkout. Outputs, including all failed attempts,
remain under the protected work directory. This does not measure clinical accuracy.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib.util
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

_spec = importlib.util.spec_from_file_location("_research_adaptive_checks", Path(__file__).with_name("evaluate_adaptive_catalog.py"))
adaptive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive)
checks = adaptive.checks
EXPECTED_CATALOG = (26, 230, 98)
METHOD = "research-native-0.4.0"
PROTOCOL = "ethmoid-count-research-v4"
SEED_MESSAGE = "The posterior sinus candidate has no spatially resolved interior."
SELECTIONS = ("all", "seed_failures", "resource_limits", "baseline_completed")
ARTIFACTS = {name: name + ".nii.gz" for name in ("cavities", "air", "region")}


def require(condition, code):
    if not condition:
        raise checks.EvaluationError(code)


def identity(entry):
    return entry["study_label"], entry["series_label"]


def load_catalog(manifest, work):
    entries = adaptive.load_manifest(manifest, work, "all")
    catalog = checks.read_json(manifest)
    unavailable = catalog.get("unavailable")
    require(isinstance(unavailable, list) and catalog.get("accuracy_measured") is False, "CATALOG_ACCOUNTING_INVALID")
    combined = entries + unavailable
    keys = [identity(entry) for entry in combined]
    studies, series, ready = EXPECTED_CATALOG
    require(all(re.fullmatch(r"Case\d{3}", str(key[0])) and re.fullmatch(r"Series\d{3}", str(key[1])) for key in keys)
            and len(set(keys)) == len(keys) == series and len(entries) == ready
            and len({key[0] for key in keys}) == studies
            and catalog.get("studies") == studies and catalog.get("series") == series, "CATALOG_COVERAGE_MISMATCH")
    return catalog, entries


def research_helpers(pipeline):
    # Only fixed modules in the already verified clean checkout are imported.
    profiles = adaptive.profile_helpers(pipeline)
    name = profiles.__package__ + ".research_policy"
    sys.modules[profiles.__package__ + ".profiles"] = profiles
    path = checks.contained(pipeline / "src/ethmoid_pipeline/counting/research_policy.py", pipeline)
    spec = importlib.util.spec_from_file_location(name, path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    require(policy.METHOD_VERSION == METHOD and policy.PROTOCOL_VERSION == PROTOCOL, "RESEARCH_POLICY_VERSION_MISMATCH")
    return profiles, policy


def script_bindings(pipeline):
    bindings = {}
    for name in ("evaluate_research_catalog.py", "evaluate_adaptive_catalog.py", "evaluate_counting_cohort.py"):
        digest = checks.sha256(Path(__file__).with_name(name))
        require(checks.sha256(pipeline / "scripts" / name) == digest, "EVALUATOR_FREEZE_MISMATCH")
        bindings[name] = digest
    return bindings


def load_baseline(directory, entries, work, manifest_hash, pipeline):
    if directory is None:
        return {}, None
    directory = checks.contained(directory, work)
    provenance = checks.read_json(directory / "provenance.local.json")
    summary = checks.read_json(directory / "summary.local.json")
    protocol = checks.read_json(directory / "protocol.local.json")
    require(provenance.get("manifest_sha256") == manifest_hash
            and provenance.get("accuracy_measured") is False
            and re.fullmatch(r"[a-f0-9]{40}", str(provenance.get("code", {}).get("git_commit", "")))
            and provenance.get("code", {}).get("git_dirty") is False
            and protocol == checks.read_json(pipeline / "src/ethmoid_pipeline/counting/protocol.v3.json")
            and protocol.get("protocol_version") == "ethmoid-count-prototype-v3", "BASELINE_PROVENANCE_INVALID")
    expected_paths = {directory / ("_".join(identity(entry))) / "attempt.local.json" for entry in entries}
    require(set(directory.glob("*/attempt.local.json")) == expected_paths, "BASELINE_COVERAGE_MISMATCH")
    records, attempts, hashes = {}, [], {}
    for entry in entries:
        folder = checks.contained(directory / "_".join(identity(entry)), work)
        attempt_path, result_path = folder / "attempt.local.json", folder / "worker/result.json"
        attempt, result = checks.read_json(attempt_path), checks.read_json(result_path)
        require(identity(attempt) == identity(entry) and attempt.get("source_sha256") == result.get("source_sha256") == entry["source_sha256"]
                and attempt.get("result_sha256") == checks.sha256(result_path)
                and attempt.get("worker_log_sha256") == checks.sha256(folder / "worker.local.log")
                and result.get("protocol_version") == protocol["protocol_version"]
                and result.get("config") == protocol["algorithm"]
                and result.get("method_version") == "scan-adaptive-native-0.3.0"
                and result.get("analysis_mode") == "automatic" and result.get("assistance") is None
                and result.get("acquisition") == attempt.get("acquisition") == entry["acquisition"], "BASELINE_RESULT_BINDING_INVALID")
        for field, digest_field in (("scan_profile", "scan_profile_sha256"), ("processing_plan", "processing_plan_sha256")):
            snapshot = result.get(field, {})
            require(snapshot.get("sha256") == attempt.get(digest_field) == checks.canonical_sha256({k: v for k, v in snapshot.items() if k != "sha256"}), "BASELINE_ADAPTATION_INVALID")
        require(result["scan_profile"].get("source_sha256") == entry["source_sha256"], "BASELINE_ADAPTATION_INVALID")
        require(attempt.get("status") == result.get("status") and attempt.get("counts") == result.get("counts")
                and attempt.get("outcome") == result.get("outcome"), "BASELINE_OUTCOME_INVALID")
        if result["status"] == "completed":
            require(result.get("outcome") == "automatic_candidate_count" and result.get("counts", {}).get("total", 0) > 0
                    and result.get("artifact_verification", {}).get("status") == "verified", "BASELINE_OUTCOME_INVALID")
        else:
            require(result.get("status") == "failed" and result.get("counts") == checks.NULL_COUNTS
                    and result.get("cells") == [] and result.get("artifacts") == {}
                    and result.get("failure_code") == attempt.get("failure_code")
                    and result.get("outcome") in {"needs_review", "unsupported_input", "technical_error"}, "BASELINE_OUTCOME_INVALID")
        records[identity(entry)] = {"directory": folder, "attempt": attempt, "result": result}
        attempts.append(attempt)
        for path in (attempt_path, result_path, folder / "worker.local.log"):
            hashes[path.relative_to(directory).as_posix()] = checks.sha256(path)
    recomputed = adaptive.summarize(attempts)
    require(all(summary.get(key) == value for key, value in recomputed.items()), "BASELINE_SUMMARY_INVALID")
    for name in ("provenance.local.json", "summary.local.json", "protocol.local.json"):
        hashes[name] = checks.sha256(directory / name)
    return records, {"directory": directory.relative_to(work).as_posix(), "sha256": hashes,
                     "code": provenance["code"], "runtime": provenance["runtime"]}


def select_entries(entries, baseline, selection):
    require(selection in SELECTIONS, "SELECTION_INVALID")
    require(selection == "all" or baseline, "BASELINE_REQUIRED_FOR_SELECTION")
    selected = []
    for entry in entries:
        result = baseline.get(identity(entry), {}).get("result", {})
        if (selection == "all" or
            selection == "seed_failures" and result.get("failure_code") == "SPHENOID_UNRESOLVED" and result.get("message") == SEED_MESSAGE or
            selection == "resource_limits" and result.get("failure_code") == "RESOURCE_LIMIT" or
            selection == "baseline_completed" and result.get("status") == "completed"):
            selected.append(entry)
    require(selected, "EMPTY_EVALUATION_SELECTION")
    return selected


def cache_for(entry, baseline, source, work, runtime):
    reference = None
    if entry.get("localizer_cache"):
        cache = checks.contained(work / entry["localizer_cache"], work)
        origin = "frozen_manifest"
    else:
        previous = baseline.get(identity(entry))
        generated = previous["attempt"].get("generated_localizers", {}) if previous else {}
        name = next((name for name in ("localizers-full-extent", "localizers") if name in generated), None)
        if name is None:
            return None, None, "fresh_inference"
        cache = checks.contained(previous["directory"] / "worker" / name, work)
        reference = generated[name]
        origin = "verified_v3_generated"
    info = checks.validate_cache(cache, source, entry["source_sha256"], work, runtime)
    require(info and info["producer_provenance"] == "verified", "UNVERIFIED_CACHE_PRODUCER")
    require(reference is None or info == reference, "BASELINE_CACHE_CHANGED")
    return cache, info, origin


def worker(runtime, pipeline, source, output, protocol, acquisition, plan, cache, log, timeout):
    command = [str(runtime), "-m", "ethmoid_pipeline.counting.worker", "--input", str(source),
               "--output", str(output), "--config", str(protocol), "--acquisition", str(acquisition),
               "--method", "research-v4", "--processing-plan", str(plan), "--progress", str(log.with_name("progress.local.json"))]
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


def validate_success(result, directory, source, algorithm):
    require(result.get("outcome") == "automatic_candidate_count" and "failure_code" not in result
            and result.get("artifacts") == ARTIFACTS, "WORKER_SUCCESS_INVALID")
    verification = result.get("artifact_verification", {})
    require(verification.get("status") == "verified" and verification.get("source_sha256") == result["source_sha256"]
            and set(verification.get("sha256", {})) == set(ARTIFACTS), "WORKER_VERIFICATION_MISSING")
    for name, filename in ARTIFACTS.items():
        require(checks.sha256(checks.contained(directory / filename, directory)) == verification["sha256"][name], "WORKER_ARTIFACT_HASH_CHANGED")
    validated = checks.validate_completed(result, directory, source, algorithm)
    arrays = {}
    for name, filename in ARTIFACTS.items():
        image = nib.load(directory / filename)
        require(image.get_data_dtype() == np.dtype("uint8" if name == "region" else "uint16"), "NATIVE_DTYPE_INVALID")
        arrays[name] = np.asanyarray(image.dataobj)
    image = nib.load(source)
    hu = image.get_fdata(dtype=np.float32)
    require(np.isfinite(hu).all() and np.array_equal(arrays["air"], np.where(hu < algorithm["air_hu"], arrays["cavities"], 0)), "NATIVE_HU_AIR_MISMATCH")
    volume = abs(float(np.linalg.det(image.affine[:3, :3])))
    require(np.isfinite(volume) and volume > 0, "NATIVE_GEOMETRY_INVALID")
    return {**validated, "native_voxel_volume_mm3": volume, "source_hu_air_consistency": "PASS"}


def evaluate(*, work_root, pipeline_root, runtime_python, manifest, output, expected_code_sha,
             expected_manifest_sha256, expected_protocol_sha256, selection="all", baseline_evaluation=None, timeout_seconds=1800):
    work = work_root.resolve(strict=True)
    protected = (Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work").resolve(strict=True)
    require(work == protected and type(timeout_seconds) is int and timeout_seconds > 0, "PROTECTED_WORK_ROOT_REQUIRED")
    pipeline = pipeline_root.resolve(strict=True)
    output, manifest = checks.contained(output, work, exists=False), checks.contained(manifest, work)
    require(not output.exists() and not pipeline.is_relative_to(work) and not work.is_relative_to(pipeline), "IMMUTABLE_SEPARATE_OUTPUT_REQUIRED")
    require(not any((parent / ".git").exists() for parent in (work, *output.parents)), "MEDICAL_OUTPUT_INSIDE_GIT")
    code = checks.code_provenance(pipeline, require_clean=True)
    protocol_path = pipeline / "src/ethmoid_pipeline/counting/protocol.research.v4.json"
    protocol = checks.read_json(protocol_path)
    require(code["git_commit"] == expected_code_sha and checks.sha256(manifest) == expected_manifest_sha256
            and checks.sha256(protocol_path) == expected_protocol_sha256, "FREEZE_MISMATCH")
    require(protocol.get("protocol_version") == PROTOCOL, "RESEARCH_PROTOCOL_REQUIRED")
    tree = ast.parse((pipeline / "src/ethmoid_pipeline/counting/engine.py").read_text(encoding="utf-8"))
    config = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CountingConfig"), None)
    fields = {n.target.id for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)} if config else set()
    require(fields and set(protocol.get("algorithm", {})) == fields and all(type(v) in (int, float) and np.isfinite(v) for v in protocol["algorithm"].values()), "COMPLETE_FROZEN_ALGORITHM_REQUIRED")
    scripts = script_bindings(pipeline)
    catalog, entries = load_catalog(manifest, work)
    profiles, policy = research_helpers(pipeline)
    quality = adaptive.quality_helpers(pipeline)
    baseline, baseline_info = load_baseline(baseline_evaluation, entries, work, expected_manifest_sha256, pipeline)
    selected = select_entries(entries, baseline, selection)
    runtime = checks._runtime_provenance(runtime_python)
    if baseline_info:
        require(baseline_info["runtime"] == runtime, "BASELINE_RUNTIME_MISMATCH")
    output.mkdir(parents=True)
    checks.write_new_json(output / "protocol.local.json", protocol)
    protocol_copy_hash = checks.sha256(output / "protocol.local.json")
    provenance = {"schema_version": 1, "code": code, "runtime": runtime, "manifest_sha256": expected_manifest_sha256,
        "protocol_sha256": expected_protocol_sha256, "evaluator_sha256": scripts["evaluate_research_catalog.py"], "evaluator_dependencies": scripts,
        "selection": selection, "selection_count": len(selected), "selected_sources_sha256": checks.canonical_sha256([{**dict(zip(("study_label", "series_label"), identity(e))), "source_sha256": e["source_sha256"]} for e in selected]),
        "baseline": baseline_info, "accuracy_measured": False, "runtime_verification_policy": "full_fingerprint_before_and_after_batch",
        "raw_dicom_verification": "frozen_manifest_fingerprints_not_rehashed_by_this_runner",
        "evaluation_scope": "existing_development_collection_not_independent_clinical_validation"}
    checks.write_new_json(output / "provenance.local.json", provenance)

    def frozen_unchanged():
        require(checks.code_provenance(pipeline, require_clean=True) == code and script_bindings(pipeline) == scripts
                and checks.sha256(manifest) == expected_manifest_sha256 and checks.sha256(protocol_path) == expected_protocol_sha256
                and checks.sha256(output / "protocol.local.json") == protocol_copy_hash, "PROVENANCE_CHANGED")
        if baseline_info:
            for relative, digest in baseline_info["sha256"].items():
                require(checks.sha256(checks.contained(work / baseline_info["directory"] / relative, work)) == digest, "BASELINE_CHANGED")

    attempts = []
    for entry in selected:
        directory = output / "_".join(identity(entry))
        directory.mkdir()
        attempt = {"study_label": entry["study_label"], "series_label": entry["series_label"], "split": entry["split"],
                   "slice_thickness_mm": entry["acquisition"].get("slice_thickness_mm"), "source_sha256": entry["source_sha256"],
                   "status": "failed", "outcome": "technical_error", "counts": dict(checks.NULL_COUNTS)}
        started = time.monotonic()
        try:
            frozen_unchanged()
            source = checks.contained(work / entry["nifti"], work)
            require(checks.sha256(source) == entry["source_sha256"], "CATALOG_SOURCE_CHANGED")
            acquisition = adaptive.normalized_acquisition(entry, quality)
            profile = profiles.build_scan_profile(source, acquisition)
            plan = policy.build_research_plan(profile)
            expected_quality = quality.resolution_quality(acquisition, nib.load(source).affine)
            attempt.update(acquisition=acquisition, quality=expected_quality, scan_profile_sha256=profile["sha256"], processing_plan_sha256=plan["sha256"])
            for name, value in (("acquisition", acquisition), ("scan-profile", profile), ("processing-plan", plan)):
                checks.write_new_json(directory / (name + ".local.json"), value)
            inputs = {name + ".local.json": checks.sha256(directory / (name + ".local.json"))
                      for name in ("acquisition", "scan-profile", "processing-plan")}
            attempt["input_snapshot_sha256"] = inputs
            cache, cache_info, cache_origin = cache_for(entry, baseline, source, work, runtime)
            attempt.update(cached_localizers=cache_info, cache_origin=cache_origin)
            if cache:
                attempt["localizer_cache"] = cache.relative_to(work).as_posix()
            returncode = worker(runtime_python, pipeline, source, directory / "worker", output / "protocol.local.json",
                directory / "acquisition.local.json", directory / "processing-plan.local.json", cache, directory / "worker.local.log", timeout_seconds)
            attempt["worker_returncode"] = returncode
            result_path = directory / "worker/result.json"
            result = checks.read_json(result_path)
            attempt["result_sha256"] = checks.sha256(result_path)
            frozen_unchanged()
            require(checks.sha256(source) == entry["source_sha256"], "CATALOG_SOURCE_CHANGED")
            require(all(checks.sha256(directory / name) == digest for name, digest in inputs.items()), "WORKER_INPUT_SNAPSHOT_CHANGED")
            if cache:
                require(checks.validate_cache(cache, source, entry["source_sha256"], work, runtime) == cache_info, "CACHE_CHANGED")
            generated = {}
            for cache_name in ("localizers", "localizers-full-extent"):
                cache_path = directory / "worker" / cache_name
                if (cache_path / "localizers.local.json").is_file():
                    generated[cache_name] = checks.validate_cache(cache_path, source, entry["source_sha256"], work, runtime)
                    require(generated[cache_name] and generated[cache_name]["producer_provenance"] == "verified", "UNVERIFIED_CACHE_PRODUCER")
            attempt["generated_localizers"] = generated
            require(result.get("source_sha256") == entry["source_sha256"] and result.get("protocol_version") == PROTOCOL
                    and result.get("method_version") == METHOD and result.get("config") == protocol["algorithm"], "WORKER_PROVENANCE_MISMATCH")
            require(result.get("acquisition") == acquisition and result.get("quality") == expected_quality, "WORKER_ACQUISITION_MISMATCH")
            require(result.get("analysis_mode") == "automatic" and result.get("assistance") is None
                    and result.get("scan_profile") == profile and result.get("processing_plan") == plan
                    and result.get("model_identity") == {"localization": "research_geometric_baseline", "direct_ethmoid_model": "not_used"}
                    and result.get("research_status") == "NOT_ACCURACY_VALIDATED", "WORKER_ADAPTATION_MISMATCH")
            attempt.update(method_version=METHOD, localization_geometry=result.get("localization_geometry", {}))
            if returncode == 0 and result.get("status") == "completed":
                validated = validate_success(result, directory / "worker", source, protocol["algorithm"])
                attempt.update(validated, status="completed", counts=result["counts"], outcome="automatic_candidate_count")
            else:
                require(returncode != 0 and result.get("status") == "failed" and result.get("counts") == checks.NULL_COUNTS
                        and result.get("cells") == [] and result.get("artifacts") == {}
                        and re.fullmatch(r"[A-Z][A-Z0-9_]{1,79}", str(result.get("failure_code", "")))
                        and result.get("outcome") in {"needs_review", "unsupported_input", "technical_error"}, "WORKER_FAILURE_INVALID")
                attempt.update(failure_code=result["failure_code"], outcome=result["outcome"], diagnostics=checks.structured_diagnostics(result))
        except checks.EvaluationError as exc:
            attempt.update(status="failed", counts=dict(checks.NULL_COUNTS), outcome="technical_error", failure_code=str(exc))
        except Exception:
            attempt.update(status="failed", counts=dict(checks.NULL_COUNTS), outcome="technical_error", failure_code="LOCAL_EVALUATION_ERROR")
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if (directory / "worker.local.log").is_file():
            attempt["worker_log_sha256"] = checks.sha256(directory / "worker.local.log")
        checks.write_new_json(directory / "attempt.local.json", attempt)
        attempts.append(attempt)
        print(json.dumps({"attempted": len(attempts), "planned": len(selected), "completed": sum(a["status"] == "completed" for a in attempts)}), flush=True)
    verification = {"status": "verified", "runtime_before_sha256": checks.canonical_sha256(runtime),
                    "attempt_sha256": {"_".join(identity(a)): checks.sha256(output / "_".join(identity(a)) / "attempt.local.json") for a in attempts}}
    try:
        frozen_unchanged()
        final_runtime = checks._runtime_provenance(runtime_python)
        verification["runtime_after_sha256"] = checks.canonical_sha256(final_runtime)
        require(final_runtime == runtime, "RUNTIME_CHANGED")
    except checks.EvaluationError as exc:
        verification.update(status="failed", failure_code=str(exc))
    except Exception:
        verification.update(status="failed", failure_code="FINAL_VERIFICATION_ERROR")
    checks.write_new_json(output / "batch-verification.local.json", verification)
    summary = adaptive.summarize(attempts)
    summary.update(selection=selection, catalog_studies=catalog["studies"], catalog_series=catalog["series"], catalog_ready=len(entries),
        catalog_unavailable=len(catalog["unavailable"]), selected=len(selected), unselected=len(entries) - len(selected),
        verification_status=verification["status"], batch_verification_sha256=checks.sha256(output / "batch-verification.local.json"),
        verified_completed=summary["completed"] if verification["status"] == "verified" else 0,
        outcomes=dict(Counter(a["outcome"] for a in attempts)), evaluation_scope=provenance["evaluation_scope"])
    checks.write_new_json(output / "summary.local.json", summary)
    return summary


def main(argv=None):
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "EthmoidComplexity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=base / "work")
    parser.add_argument("--runtime-python", type=Path, default=base / "tools/counting-runtime/Scripts/python.exe")
    for name in ("pipeline-root", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path)
    for name in ("expected-code-sha", "expected-manifest-sha256", "expected-protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--selection", choices=SELECTIONS, default="all")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    try:
        summary = evaluate(**vars(parser.parse_args(argv)))
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc) if isinstance(exc, checks.EvaluationError) else "LOCAL_EVALUATION_ERROR"}))
        return 2
    print(json.dumps(summary))
    return 0 if summary["withheld"] == 0 and summary["verification_status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
