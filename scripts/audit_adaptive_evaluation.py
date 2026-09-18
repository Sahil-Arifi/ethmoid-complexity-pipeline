"""Read-only integrity audit of a completed frozen v3 development evaluation.

No workers, GPU inference, database mutations, or source modifications occur.
An optional report is written once to a new protected local path. Existing
results and diagnostic proposals remain unchanged. Software consistency is
not anatomical accuracy, and historical v2 outputs are not reference labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import math
import os
from pathlib import Path
import re
import statistics

import nibabel as nib
import numpy as np

_spec = importlib.util.spec_from_file_location("_adaptive_evaluation_audit_helpers", Path(__file__).with_name("evaluate_adaptive_catalog.py"))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
checks = helpers.checks

ARTIFACTS = {"cavities": "cavities.nii.gz", "air": "air.nii.gz", "region": "region.nii.gz"}
SCOPE = "existing_development_collection_not_independent_clinical_validation"
HASH = re.compile(r"[0-9a-f]{64}")


def require(condition, code):
    if not condition:
        raise checks.EvaluationError(code)


def identity(value):
    study, series = value.get("study_label"), value.get("series_label")
    require(bool(re.fullmatch(r"Case[0-9]{3,}", str(study))) and bool(re.fullmatch(r"Series[0-9]{3,}", str(series))), "AUDIT_IDENTITY_INVALID")
    return study, series


def bound_hash(path, digest, *, root):
    require(isinstance(digest, str) and bool(HASH.fullmatch(digest)), "AUDIT_HASH_BINDING_MISSING")
    path = checks.contained(path, root)
    require(path.is_file() and checks.sha256(path) == digest, "AUDIT_ARTIFACT_HASH_CHANGED")
    return path


def audit_catalog(catalog, expected_studies, expected_series, expected_ready):
    require(catalog.get("schema_version") == 1 and catalog.get("accuracy_measured") is False, "AUDIT_CATALOG_INVALID")
    ready, unavailable = catalog.get("entries"), catalog.get("unavailable")
    require(isinstance(ready, list) and isinstance(unavailable, list), "AUDIT_CATALOG_INVALID")
    all_entries = ready + unavailable
    keys = [identity(entry) for entry in all_entries]
    require(len(set(keys)) == len(keys), "AUDIT_CATALOG_DUPLICATE_IDENTITY")
    require(len(ready) == expected_ready and len(all_entries) == expected_series and len({key[0] for key in keys}) == expected_studies,
            "AUDIT_CATALOG_COVERAGE_INCOMPLETE")
    require(catalog.get("series") == expected_series and catalog.get("studies") == expected_studies, "AUDIT_CATALOG_TOTALS_INVALID")
    require(all(isinstance(entry.get("reason_code"), str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", entry["reason_code"]) for entry in unavailable),
            "AUDIT_UNAVAILABLE_REASON_MISSING")
    paths = [entry.get("nifti") for entry in ready]
    require(all(isinstance(path, str) for path in paths) and len(set(paths)) == len(paths), "AUDIT_SOURCE_DUPLICATE")
    return ready, unavailable


def audit_cache(directory, expected, source, source_hash, work, runtime):
    """Hashes bind previously validated masks; no mask pixel arrays are allocated."""
    require(isinstance(expected, dict) and expected.get("producer_provenance") == "verified", "AUDIT_CACHE_BINDING_MISSING")
    manifest_path = bound_hash(directory / "localizers.local.json", expected.get("manifest_sha256"), root=work)
    manifest = checks.read_json(manifest_path)
    require(manifest.get("source_sha256") == source_hash == expected.get("source_sha256"), "AUDIT_CACHE_SOURCE_MISMATCH")
    require(manifest.get("weights_manifest_sha256") == runtime.get("weights_manifest_sha256") == expected.get("producer_weights_manifest_sha256") and
            manifest.get("totalsegmentator_version") == runtime.get("packages", {}).get("totalsegmentator") == expected.get("producer_totalsegmentator_version"),
            "AUDIT_CACHE_RUNTIME_MISMATCH")
    source_image = nib.load(source)
    actual = {}
    for task in manifest.get("tasks", []):
        require(bool(re.fullmatch(r"[a-z_]+", str(task.get("task")))), "AUDIT_CACHE_PATH_INVALID")
        for mask in task.get("masks", []):
            filename = mask.get("filename")
            require(bool(re.fullmatch(r"[a-z0-9_]+\.nii\.gz", str(filename))), "AUDIT_CACHE_PATH_INVALID")
            relative = task["task"] + "/" + filename
            require(relative not in actual, "AUDIT_CACHE_DUPLICATE")
            path = bound_hash(directory / relative, mask.get("sha256"), root=directory)
            image = nib.load(path)
            require(image.shape == source_image.shape and np.allclose(image.affine, source_image.affine, atol=1e-5, rtol=0), "AUDIT_CACHE_GEOMETRY_INVALID")
            actual[relative] = mask["sha256"]
    require(actual and actual == expected.get("mask_sha256"), "AUDIT_CACHE_ARTIFACT_SET_CHANGED")


def audit_success(result, attempt, worker, source, protocol):
    require(result.get("outcome") == attempt.get("outcome") == "automatic_candidate_count", "AUDIT_SUCCESS_OUTCOME_INVALID")
    require(result.get("artifacts") == ARTIFACTS and "failure_code" not in result, "AUDIT_SUCCESS_MANIFEST_INVALID")
    verification = result.get("artifact_verification", {})
    require(verification.get("status") == "verified" and verification.get("source_sha256") == result["source_sha256"], "AUDIT_ENGINE_VERIFICATION_MISSING")
    require(set(verification.get("sha256", {})) == set(ARTIFACTS), "AUDIT_ENGINE_HASH_SET_INVALID")
    for kind, filename in ARTIFACTS.items():
        bound_hash(worker / filename, verification["sha256"].get(kind), root=worker)
    validated = checks.validate_completed(result, worker, source, protocol["algorithm"])
    for key in ("artifacts", "native_geometry", "count_identity", "diagnostics"):
        require(attempt.get(key) == validated[key], "AUDIT_RECORDED_VALIDATION_MISMATCH")
    require(attempt.get("counts") == result.get("counts"), "AUDIT_ATTEMPT_COUNT_MISMATCH")
    source_image = nib.load(source)
    arrays = {}
    for kind, filename in ARTIFACTS.items():
        path = bound_hash(worker / filename, verification["sha256"].get(kind), root=worker)
        image = nib.load(path)
        require(image.get_data_dtype() == np.dtype("uint8" if kind == "region" else "uint16"), "AUDIT_NATIVE_DTYPE_INVALID")
        arrays[kind] = np.asanyarray(image.dataobj)
    native_hu = source_image.get_fdata(dtype=np.float32)
    require(np.isfinite(native_hu).all(), "AUDIT_SOURCE_HU_INVALID")
    expected_air = np.where(native_hu < protocol["algorithm"]["air_hu"], arrays["cavities"], 0)
    require(np.array_equal(arrays["air"], expected_air), "AUDIT_NATIVE_HU_AIR_MISMATCH")
    voxel_volume = abs(float(np.linalg.det(source_image.affine[:3, :3])))
    require(math.isfinite(voxel_volume) and voxel_volume > 0, "AUDIT_SOURCE_GEOMETRY_INVALID")
    return {"uncertain_candidates": sum(cell["uncertain"] for cell in result["cells"]),
            "native_voxel_volume_mm3": voxel_volume,
            "cavity_voxels": sum(cell["voxel_count"] for cell in result["cells"]),
            "air_voxels": sum(cell["air_voxel_count"] for cell in result["cells"])}


def audit_failure(result, attempt, worker):
    require(result.get("counts") == attempt.get("counts") == checks.NULL_COUNTS and result.get("cells") == [] and result.get("artifacts") == {},
            "AUDIT_FAILED_COUNT_EXPOSED")
    code = result.get("failure_code")
    require(isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", code) and code == attempt.get("failure_code"), "AUDIT_FAILURE_CODE_MISMATCH")
    require(result.get("outcome") in {"needs_review", "technical_error", "unsupported_input"} and result["outcome"] == attempt.get("outcome"),
            "AUDIT_FAILURE_OUTCOME_INVALID")
    require(attempt.get("diagnostics", {}) == checks.structured_diagnostics(result), "AUDIT_FAILURE_DIAGNOSTICS_MISMATCH")
    diagnostic = worker / "diagnostic-candidates.local.json"
    hashes = {}
    if diagnostic.exists():
        value = checks.read_json(diagnostic)
        require(value.get("clinical_count_available") is False and isinstance(value.get("cells"), list), "AUDIT_REJECTED_PROPOSAL_POLICY_INVALID")
        hashes[diagnostic.name] = checks.sha256(diagnostic)
        for filename in ARTIFACTS.values():
            path = worker / filename
            if path.exists():
                hashes[filename] = checks.sha256(checks.contained(path, worker))
    elif any((worker / name).exists() for name in ARTIFACTS.values()):
        # Some integrity failures occur after artifact creation; retain the bytes
        # as diagnostics without interpreting them as accepted cells.
        for filename in ARTIFACTS.values():
            path = worker / filename
            if path.exists():
                hashes[filename] = checks.sha256(checks.contained(path, worker))
    return hashes


def independently_summarize(attempts, *, studies, series, unavailable):
    def group(items):
        return {"attempted": len(items), "completed": sum(item["status"] == "completed" for item in items),
            "withheld": sum(item["status"] == "failed" for item in items),
            "failure_codes": dict(Counter(item["failure_code"] for item in items if item["status"] == "failed")),
            "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in items), 3)}
    return {**group(attempts), "studies": len({item["study_label"] for item in attempts}),
        "by_thickness_mm": {str(thickness): group([item for item in attempts if item["slice_thickness_mm"] == thickness])
            for thickness in sorted({item["slice_thickness_mm"] for item in attempts}, key=str)},
        "uncertain_candidates": sum(item.get("diagnostics", {}).get("uncertain_candidates", 0) for item in attempts if item["status"] == "completed"),
        "accuracy_measured": False, "interpretation": "Execution and consistency, not anatomical accuracy.",
        "catalog_studies": studies, "catalog_series": series, "catalog_unavailable": unavailable,
        "outcomes": dict(Counter(item.get("outcome", "technical_error") for item in attempts)), "evaluation_scope": SCOPE}


def runtime_summary(attempts):
    result = {}
    for thickness in sorted({item["slice_thickness_mm"] for item in attempts}, key=str):
        selected = [item for item in attempts if item["slice_thickness_mm"] == thickness]
        seconds = [item["elapsed_seconds"] for item in selected]
        result[str(thickness)] = {"attempted": len(selected), "completed": sum(item["status"] == "completed" for item in selected),
            "withheld": sum(item["status"] == "failed" for item in selected),
            "mean_seconds": round(statistics.mean(seconds), 3), "median_seconds": round(statistics.median(seconds), 3),
            "minimum_seconds": min(seconds), "maximum_seconds": max(seconds), "sum_seconds": round(sum(seconds), 3)}
    return result


def compare_v2(attempts, directories, work):
    if not directories:
        return None
    old = {}
    bindings = {}
    for directory in directories:
        directory = checks.contained(directory, work)
        protocol = checks.read_json(directory / "protocol.local.json")
        require(protocol.get("protocol_version") == "ethmoid-count-prototype-v2", "AUDIT_COMPARISON_NOT_V2")
        provenance = checks.read_json(directory / "provenance.local.json")
        # v2 serialized the protocol, so compare semantic contents to its frozen
        # raw hash only through archived provenance; do not invent a new freeze.
        bindings[directory.relative_to(work).as_posix()] = {"provenance_sha256": checks.sha256(directory / "provenance.local.json"),
            "recorded_code_sha": provenance.get("code", {}).get("git_commit"), "protocol_sha256": provenance.get("protocol_sha256")}
        for path in sorted(directory.glob("*/attempt.local.json")):
            entry = checks.read_json(path)
            key = identity(entry)
            require(key not in old, "AUDIT_COMPARISON_DUPLICATE")
            archived_result = checks.read_json(bound_hash(path.parent / "worker/result.json", entry.get("result_sha256"), root=directory))
            bound_hash(path.parent / "worker.local.log", entry.get("worker_log_sha256"), root=directory)
            require(archived_result.get("protocol_version") == "ethmoid-count-prototype-v2" and
                all(archived_result.get(key) == entry.get(key) for key in ("source_sha256", "status", "counts", "failure_code")),
                "AUDIT_COMPARISON_RESULT_MISMATCH")
            old[key] = entry
    totals = Counter()
    transitions = Counter()
    for attempt in attempts:
        prior = old.get(identity(attempt))
        if prior is None:
            totals["missing_v2_pair"] += 1
            continue
        if prior.get("source_sha256") != attempt.get("source_sha256"):
            totals["source_changed_not_comparable"] += 1
            continue
        totals["same_source_pairs"] += 1
        same = prior.get("status") == attempt["status"] and prior.get("counts") == attempt["counts"] and prior.get("failure_code") == attempt.get("failure_code")
        totals["same_status_counts_and_failure_code" if same else "changed_status_counts_or_failure_code"] += 1
        before = "completed" if prior.get("status") == "completed" else str(prior.get("failure_code"))
        after = "completed" if attempt["status"] == "completed" else str(attempt.get("failure_code"))
        transitions[before + " -> " + after] += 1
    return {"interpretation": "method_repeatability_only_not_ground_truth_or_accuracy", "totals": dict(totals),
            "transitions": dict(transitions), "archive_bindings": bindings}


def audit(*, work_root, pipeline_root, runtime_python, manifest, evaluation, expected_code_sha,
          expected_manifest_sha256, expected_protocol_sha256, expected_studies=26, expected_series=230,
          expected_ready=98, v2_evaluations=()):
    work = work_root.resolve(strict=True)
    protected = (Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work").resolve(strict=True)
    require(work == protected, "PROTECTED_WORK_ROOT_REQUIRED")
    directory = checks.contained(evaluation, work)
    # Do this before opening any large source or runtime artifact.
    require((directory / "summary.local.json").is_file(), "AUDIT_EVALUATION_NOT_FINISHED")
    stored_summary = checks.read_json(directory / "summary.local.json")
    pipeline = pipeline_root.resolve(strict=True)
    manifest = checks.contained(manifest, work)
    provenance = checks.read_json(directory / "provenance.local.json")
    require(provenance.get("split") == "all" and provenance.get("evaluation_scope") == SCOPE and provenance.get("accuracy_measured") is False,
            "AUDIT_FULL_DEVELOPMENT_EVALUATION_REQUIRED")
    require(checks.sha256(manifest) == provenance.get("manifest_sha256") == expected_manifest_sha256, "AUDIT_MANIFEST_CHANGED")
    protocol_path = checks.contained(pipeline / "src/ethmoid_pipeline/counting/protocol.v3.json", pipeline)
    require(checks.sha256(protocol_path) == provenance.get("protocol_sha256") == expected_protocol_sha256, "AUDIT_PROTOCOL_CHANGED")
    protocol = checks.read_json(protocol_path)
    require(protocol.get("protocol_version") == "ethmoid-count-prototype-v3" and protocol == checks.read_json(directory / "protocol.local.json"), "AUDIT_PROTOCOL_SNAPSHOT_CHANGED")
    code = checks.code_provenance(pipeline, require_clean=True)
    require(code == provenance.get("code") and code["git_commit"] == expected_code_sha, "AUDIT_FROZEN_CODE_CHANGED")
    require(checks.sha256(pipeline / "scripts/evaluate_adaptive_catalog.py") == provenance.get("evaluator_sha256"), "AUDIT_EVALUATOR_CHANGED")
    runtime = checks._runtime_provenance(runtime_python)
    require(runtime == provenance.get("runtime"), "AUDIT_RUNTIME_CHANGED")
    catalog = checks.read_json(manifest)
    entries, unavailable = audit_catalog(catalog, expected_studies, expected_series, expected_ready)
    expected_paths = {directory / (entry["study_label"] + "_" + entry["series_label"]) / "attempt.local.json" for entry in entries}
    found = set(directory.rglob("attempt.local.json"))
    require(found == expected_paths, "AUDIT_ATTEMPT_COVERAGE_INCOMPLETE_OR_DUPLICATE")
    profiles = helpers.profile_helpers(pipeline)
    quality = helpers.quality_helpers(pipeline)
    attempts, bindings = [], []
    seen_sources = set()
    successes_verified = diagnostic_failures = 0
    for entry in entries:
        name = entry["study_label"] + "_" + entry["series_label"]
        folder = directory / name
        worker = folder / "worker"
        path = folder / "attempt.local.json"
        attempt = checks.read_json(path)
        require(identity(attempt) == identity(entry) and attempt.get("source_sha256") == entry.get("source_sha256"), "AUDIT_ATTEMPT_SOURCE_MISMATCH")
        source = bound_hash(work / entry["nifti"], entry.get("source_sha256"), root=work)
        require(source not in seen_sources, "AUDIT_SOURCE_DUPLICATE")
        seen_sources.add(source)
        require(entry.get("automated_qc") == "PASS", "AUDIT_SOURCE_NOT_VERIFIED")
        acquisition = helpers.normalized_acquisition(entry, quality)
        require(attempt.get("acquisition") == acquisition and attempt.get("slice_thickness_mm") == acquisition.get("slice_thickness_mm") and attempt.get("split") == entry.get("split"),
                "AUDIT_ATTEMPT_ACQUISITION_MISMATCH")
        require(checks.read_json(folder / "acquisition.local.json") == acquisition, "AUDIT_ACQUISITION_FILE_CHANGED")
        expected_quality = quality.resolution_quality(acquisition, nib.load(source).affine)
        require(attempt.get("quality") == expected_quality, "AUDIT_ATTEMPT_QUALITY_MISMATCH")
        duration = attempt.get("elapsed_seconds")
        require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, "AUDIT_RUNTIME_MEASUREMENT_INVALID")
        bound_hash(folder / "worker.local.log", attempt.get("worker_log_sha256"), root=folder)
        result_path = bound_hash(worker / "result.json", attempt.get("result_sha256"), root=worker)
        result = checks.read_json(result_path)
        require(result.get("schema_version") == 1, "AUDIT_WORKER_SCHEMA_INVALID")
        require(result.get("status") == attempt.get("status") and result.get("status") in {"completed", "failed"}, "AUDIT_WORKER_STATUS_MISMATCH")
        require(result.get("source_sha256") == entry["source_sha256"] and result.get("protocol_version") == protocol["protocol_version"] and result.get("config") == protocol["algorithm"], "AUDIT_WORKER_SOURCE_OR_METHOD_MISMATCH")
        require(result.get("method_version") == attempt.get("method_version") == "scan-adaptive-native-0.3.0", "AUDIT_WORKER_METHOD_MISMATCH")
        require(result.get("analysis_mode") == "automatic" and result.get("assistance") is None, "AUDIT_UNEXPECTED_ASSISTANCE")
        require(result.get("model_identity") == {"localization": "v2_geometric_baseline", "direct_ethmoid_model": "unavailable_without_promoted_weights"}, "AUDIT_MODEL_IDENTITY_MISMATCH")
        require(result.get("acquisition") == acquisition and result.get("quality") == expected_quality, "AUDIT_WORKER_ACQUISITION_MISMATCH")
        profile = profiles.build_scan_profile(source, acquisition)
        plan = profiles.build_processing_plan(profile)
        require(result.get("scan_profile") == profile and attempt.get("scan_profile_sha256") == profile["sha256"], "AUDIT_PROFILE_MISMATCH")
        require(result.get("processing_plan") == plan and attempt.get("processing_plan_sha256") == plan["sha256"], "AUDIT_PLAN_MISMATCH")
        require(result.get("localization_geometry", {}) == attempt.get("localization_geometry", {}), "AUDIT_LOCALIZATION_RECORD_MISMATCH")
        if entry.get("localizer_cache"):
            audit_cache(checks.contained(work / entry["localizer_cache"], work), attempt.get("cached_localizers"), source, entry["source_sha256"], work, runtime)
        else:
            require(attempt.get("cached_localizers") is None, "AUDIT_UNEXPECTED_CACHE")
        generated = attempt.get("generated_localizers", {})
        require(isinstance(generated, dict) and not (set(generated) - {"localizers", "localizers-full-extent"}), "AUDIT_GENERATED_CACHE_INVALID")
        for cache_name in ("localizers", "localizers-full-extent"):
            exists = (worker / cache_name / "localizers.local.json").is_file()
            require(exists == (cache_name in generated), "AUDIT_GENERATED_CACHE_UNRECORDED")
            if exists:
                audit_cache(worker / cache_name, generated[cache_name], source, entry["source_sha256"], work, runtime)
        binding = {"study_label": entry["study_label"], "series_label": entry["series_label"], "source_sha256": entry["source_sha256"],
            "attempt_sha256": checks.sha256(path), "result_sha256": attempt["result_sha256"], "worker_log_sha256": attempt["worker_log_sha256"]}
        if result["status"] == "completed":
            binding["native_measurements"] = audit_success(result, attempt, worker, source, protocol)
            binding["artifact_sha256"] = result["artifact_verification"]["sha256"]
            successes_verified += 1
        else:
            diagnostic = audit_failure(result, attempt, worker)
            if diagnostic:
                diagnostic_failures += 1
                binding["protected_rejected_proposal_hashes"] = diagnostic
        attempts.append(attempt)
        bindings.append(binding)
    recomputed = independently_summarize(attempts, studies=expected_studies, series=expected_series, unavailable=len(unavailable))
    require(recomputed == stored_summary, "AUDIT_AGGREGATE_SUMMARY_MISMATCH")
    require(checks.code_provenance(pipeline, require_clean=True) == code and checks.sha256(manifest) == expected_manifest_sha256 and checks.sha256(protocol_path) == expected_protocol_sha256,
            "AUDIT_INPUT_CHANGED_DURING_AUDIT")
    return {"schema_version": 1, "status": "verified", "accuracy_measured": False, "clinical_validation": "pending_independent_expert_references",
        "coverage": {"studies": expected_studies, "series": expected_series, "unique_attempts": len(attempts), "unavailable": len(unavailable)},
        "bindings": {"code_sha": expected_code_sha, "manifest_sha256": expected_manifest_sha256, "protocol_sha256": expected_protocol_sha256,
            "runtime_snapshot_sha256": checks.canonical_sha256(runtime), "evaluation_provenance_sha256": checks.sha256(directory / "provenance.local.json"),
            "summary_sha256": checks.sha256(directory / "summary.local.json"), "auditor_sha256": checks.sha256(Path(__file__))},
        "summary": recomputed, "runtime_by_nominal_thickness_mm": runtime_summary(attempts),
        "runtime_interpretation": "serial_local_elapsed_time_including_cache_validation_and_abstentions_not_a_general_latency_guarantee",
        "successful_native_results_verified": successes_verified, "failed_attempts_with_protected_proposals": diagnostic_failures,
        "unavailable_reasons": dict(Counter(entry["reason_code"] for entry in unavailable)),
        "attempt_bindings": bindings, "v2_comparison": compare_v2(attempts, v2_evaluations, work)}


def main():
    base = Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=base / "work")
    parser.add_argument("--runtime-python", type=Path, default=base / "tools/counting-runtime/Scripts/python.exe")
    for name in ("pipeline-root", "manifest", "evaluation"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-code-sha", "expected-manifest-sha256", "expected-protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-studies", type=int, default=26)
    parser.add_argument("--expected-series", type=int, default=230)
    parser.add_argument("--expected-ready", type=int, default=98)
    parser.add_argument("--v2-evaluation", dest="v2_evaluations", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        if output is not None:
            output = checks.contained(output, args["work_root"].resolve(strict=True), exists=False)
            require(not output.exists(), "AUDIT_OUTPUT_ALREADY_EXISTS")
        report = audit(**args)
        if output is not None:
            checks.write_new_json(output, report)
        # Only aggregate information is printed; source paths/identities stay local.
        import json
        print(json.dumps({"status": report["status"], "coverage": report["coverage"], "summary": report["summary"],
            "runtime_by_nominal_thickness_mm": report["runtime_by_nominal_thickness_mm"], "accuracy_measured": False}))
        return 0
    except checks.EvaluationError as error:
        import json
        print(json.dumps({"status": "failed", "failure_code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print('{"status":"failed","failure_code":"AUDIT_INPUT_UNREADABLE_OR_INVALID"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
