"""Independent read-only audit of a complete research-v4 or opt-in v5 batch.

Never invokes inference or alters evaluations. The optional new protected report
contains aggregate measurements and hash bindings, not patient-level results.
Incomplete batches return explicitly before source or runtime reads. Integrity
verification is not anatomical accuracy or independent clinical validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path

import nibabel as nib


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = _module("_research_audit_native_checks", "audit_adaptive_evaluation.py")
runner = _module("_research_audit_policy_helpers", "evaluate_research_catalog.py")
runner_v5 = _module("_research_v5_audit_policy_helpers", "evaluate_research_v5_catalog.py")
RUNNERS = {4: runner, 5: runner_v5}
checks, require, identity, bound_hash = prior.checks, prior.require, prior.identity, prior.bound_hash
SCOPE = prior.SCOPE


def _duration(attempt):
    value = attempt.get("elapsed_seconds")
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "AUDIT_RUNTIME_MEASUREMENT_INVALID")


def _summary(attempts, studies, series, ready, unavailable, batch_hash):
    result = prior.independently_summarize(attempts, studies=studies, series=series, unavailable=unavailable)
    result.update(selection="all", catalog_ready=ready, selected=ready, unselected=0,
                  verification_status="verified", batch_verification_sha256=batch_hash,
                  verified_completed=result["completed"])
    return result


def _comparison(attempts, baseline):
    transitions, outcomes, totals, deltas = Counter(), Counter(), Counter(), Counter()
    by_thickness = {}
    for attempt in attempts:
        before = baseline[identity(attempt)]["attempt"]
        require(before["source_sha256"] == attempt["source_sha256"], "AUDIT_COMPARISON_SOURCE_CHANGED")
        old = "completed" if before["status"] == "completed" else before["failure_code"]
        new = "completed" if attempt["status"] == "completed" else attempt["failure_code"]
        transition = old + " -> " + new
        transitions[transition] += 1
        outcomes[before["outcome"] + " -> " + attempt["outcome"]] += 1
        by_thickness.setdefault(str(attempt["slice_thickness_mm"]), Counter())[transition] += 1
        totals["same_source_pairs"] += 1
        totals["previously_completed"] += before["status"] == "completed"
        totals["now_completed"] += attempt["status"] == "completed"
        if old == new == "completed":
            totals["completed_in_both"] += 1
            totals["identical_per_side_counts" if before["counts"] == attempt["counts"] else "changed_per_side_counts"] += 1
            deltas[str(attempt["counts"]["total"] - before["counts"]["total"])] += 1
        if old != "completed" and new == "completed":
            totals["newly_completed"] += 1
        if old == "completed" and new != "completed":
            totals["newly_withheld"] += 1
    return {"interpretation": "paired_development_method_comparison_not_accuracy_or_ground_truth",
            "totals": dict(totals), "failure_transitions": dict(transitions),
            "outcome_transitions": dict(outcomes), "transitions_by_nominal_thickness_mm": by_thickness,
            "completed_pair_total_count_delta_distribution": dict(deltas)}


def _completed_groups(records):
    """Publish aggregates only; study labels are used internally for grouping."""
    def group(selected):
        complete = [a for a, _ in selected if a["status"] == "completed"]
        return {"attempted": len(selected), "completed_volumes": len(complete),
                "studies_with_completed_volume": len({a["study_label"] for a in complete}),
                "candidate_instances": sum(a["counts"]["total"] for a in complete),
                "uncertain_instances": sum(a["diagnostics"]["uncertain_candidates"] for a in complete)}
    thickness, spacing, studies = {}, {}, {}
    for attempt, profile in records:
        thickness.setdefault(str(attempt["slice_thickness_mm"]), []).append((attempt, profile))
        key = ",".join(str(float(value)) for value in profile["native_spacing_mm"])
        spacing.setdefault(key, []).append((attempt, profile))
        if attempt["status"] == "completed":
            studies.setdefault(attempt["study_label"], []).append(attempt["counts"]["total"])
    multiple = [values for values in studies.values() if len(values) > 1]
    return {"overall": group(records), "by_nominal_thickness_mm": {key: group(values) for key, values in sorted(thickness.items())},
            "by_native_spacing_mm": {key: group(values) for key, values in sorted(spacing.items())},
            "within_study_reconstruction_counts": {
                "interpretation": "reconstruction_repeatability_not_reference_agreement_or_accuracy",
                "studies_with_multiple_completed_reconstructions": len(multiple),
                "total_count_range_width_distribution": dict(Counter(str(max(values) - min(values)) for values in multiple)),
                "completed_reconstructions_per_study_distribution": dict(Counter(str(len(values)) for values in studies.values()))}}


def _cache_bindings(entry, attempt, folder, baseline, source, work, runtime):
    expected_path, origin, reference = None, "fresh_inference", None
    if entry.get("localizer_cache"):
        expected_path, origin = work / entry["localizer_cache"], "frozen_manifest"
    else:
        previous = baseline.get(identity(entry))
        generated = previous["attempt"].get("generated_localizers", {}) if previous else {}
        name = next((n for n in ("localizers-full-extent", "localizers") if n in generated), None)
        if name:
            expected_path = previous["directory"] / "worker" / name
            origin, reference = "verified_v3_generated", generated[name]
    require(attempt.get("cache_origin") == origin, "AUDIT_CACHE_ORIGIN_MISMATCH")
    if expected_path is None:
        require(attempt.get("cached_localizers") is None and not attempt.get("localizer_cache"), "AUDIT_UNEXPECTED_CACHE")
    else:
        expected_path = checks.contained(expected_path, work)
        require(isinstance(attempt.get("localizer_cache"), str)
                and checks.contained(work / attempt["localizer_cache"], work) == expected_path, "AUDIT_CACHE_PATH_MISMATCH")
        if reference is not None:
            require(attempt.get("cached_localizers") == reference, "AUDIT_BASELINE_CACHE_CHANGED")
        prior.audit_cache(expected_path, attempt.get("cached_localizers"), source, entry["source_sha256"], work, runtime)
    generated = attempt.get("generated_localizers", {})
    require(isinstance(generated, dict) and not set(generated) - {"localizers", "localizers-full-extent"}, "AUDIT_GENERATED_CACHE_INVALID")
    for name in ("localizers", "localizers-full-extent"):
        cache = folder / "worker" / name
        require((cache / "localizers.local.json").is_file() == (name in generated), "AUDIT_GENERATED_CACHE_UNRECORDED")
        if name in generated:
            prior.audit_cache(cache, generated[name], source, entry["source_sha256"], work, runtime)


def audit(*, work_root, pipeline_root, runtime_python, manifest, evaluation, expected_code_sha,
          expected_manifest_sha256, expected_protocol_sha256, expected_studies=26, expected_series=230,
          expected_ready=98, generation=4):
    require(type(generation) is int and generation in RUNNERS, "AUDIT_RESEARCH_GENERATION_UNSUPPORTED")
    runner = RUNNERS[generation]
    runner_name = "evaluate_research_catalog.py" if generation == 4 else "evaluate_research_v5_catalog.py"
    work = work_root.resolve(strict=True)
    protected = (Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity/work").resolve(strict=True)
    require(work == protected, "PROTECTED_WORK_ROOT_REQUIRED")
    directory = checks.contained(evaluation, work)
    if not (directory / "summary.local.json").is_file():
        return {"schema_version": 1, "status": "incomplete", "research_generation": generation, "failure_code": "AUDIT_EVALUATION_NOT_FINISHED",
                "finished_attempt_records": sum(1 for _ in directory.glob("*/attempt.local.json")),
                "expected_attempts": expected_ready, "accuracy_measured": False}
    pipeline = pipeline_root.resolve(strict=True)
    require(not pipeline.is_relative_to(work) and not work.is_relative_to(pipeline), "AUDIT_CODE_AND_DATA_OVERLAP")
    manifest = bound_hash(manifest, expected_manifest_sha256, root=work)
    provenance_path = directory / "provenance.local.json"
    provenance = checks.read_json(provenance_path)
    summary_path = directory / "summary.local.json"
    stored_summary = checks.read_json(summary_path)
    summary_hash, provenance_hash = checks.sha256(summary_path), checks.sha256(provenance_path)
    require(provenance.get("selection") == "all" and provenance.get("selection_count") == expected_ready
            and provenance.get("evaluation_scope") == SCOPE and provenance.get("accuracy_measured") is False,
            "AUDIT_FULL_DEVELOPMENT_EVALUATION_REQUIRED")
    require(provenance.get("manifest_sha256") == expected_manifest_sha256, "AUDIT_MANIFEST_CHANGED")
    code = checks.code_provenance(pipeline, require_clean=True)
    require(code == provenance.get("code") and code["git_commit"] == expected_code_sha, "AUDIT_FROZEN_CODE_CHANGED")
    protocol_path = bound_hash(pipeline / f"src/ethmoid_pipeline/counting/protocol.research.v{generation}.json", expected_protocol_sha256, root=pipeline)
    protocol = checks.read_json(protocol_path)
    require(protocol.get("protocol_version") == runner.PROTOCOL and provenance.get("protocol_sha256") == expected_protocol_sha256
            and protocol == checks.read_json(directory / "protocol.local.json"), "AUDIT_PROTOCOL_CHANGED")
    script_hashes = {name: checks.sha256(pipeline / "scripts" / name) for name in
                     (runner_name, "evaluate_adaptive_catalog.py", "evaluate_counting_cohort.py")}
    require(script_hashes == provenance.get("evaluator_dependencies") and
            script_hashes[runner_name] == provenance.get("evaluator_sha256"), "AUDIT_EVALUATOR_CHANGED")
    # Low-level audit helpers must match the frozen runner helpers they check.
    for name in script_hashes:
        require(checks.sha256(Path(__file__).with_name(name)) == script_hashes[name], "AUDIT_HELPER_VERSION_MISMATCH")
    runtime = checks._runtime_provenance(runtime_python)
    require(runtime == provenance.get("runtime"), "AUDIT_RUNTIME_CHANGED")
    batch_path = bound_hash(directory / "batch-verification.local.json", stored_summary.get("batch_verification_sha256"), root=directory)
    batch = checks.read_json(batch_path)
    runtime_hash = checks.canonical_sha256(runtime)
    require(batch.get("status") == "verified" and batch.get("runtime_before_sha256") == batch.get("runtime_after_sha256") == runtime_hash,
            "AUDIT_BATCH_VERIFICATION_FAILED")
    catalog = checks.read_json(manifest)
    entries, unavailable = prior.audit_catalog(catalog, expected_studies, expected_series, expected_ready)
    require(len({entry["source_sha256"] for entry in entries}) == expected_ready, "AUDIT_SOURCE_DUPLICATE")
    require(checks.canonical_sha256([{**dict(zip(("study_label", "series_label"), identity(e))), "source_sha256": e["source_sha256"]} for e in entries])
            == provenance.get("selected_sources_sha256"), "AUDIT_SELECTED_SOURCE_SET_CHANGED")
    expected_paths = {directory / "_".join(identity(e)) / "attempt.local.json" for e in entries}
    require(set(directory.rglob("attempt.local.json")) == expected_paths, "AUDIT_ATTEMPT_COVERAGE_INCOMPLETE_OR_DUPLICATE")
    attempts_hashes = {p.parent.name: checks.sha256(p) for p in expected_paths}
    require(batch.get("attempt_sha256") == attempts_hashes, "AUDIT_ATTEMPT_SEAL_CHANGED")
    baseline_info = provenance.get("baseline")
    require(isinstance(baseline_info, dict) and isinstance(baseline_info.get("directory"), str), "AUDIT_BOUND_V3_BASELINE_REQUIRED")
    baseline_dir = checks.contained(work / baseline_info["directory"], work)
    baseline, current_baseline_info = runner.load_baseline(baseline_dir, entries, work, expected_manifest_sha256, pipeline)
    require(current_baseline_info == baseline_info and baseline_info.get("runtime") == runtime, "AUDIT_BASELINE_BINDINGS_CHANGED")
    profiles, policy = runner.research_helpers(pipeline)
    quality = runner.adaptive.quality_helpers(pipeline)
    v3_protocol = checks.read_json(baseline_dir / "protocol.local.json")
    attempts, all_bindings, seen_paths, records = [], [], set(), []
    successes = baseline_successes = 0
    for entry in entries:
        folder = checks.contained(directory / "_".join(identity(entry)), directory)
        attempt = checks.read_json(folder / "attempt.local.json")
        require(identity(attempt) == identity(entry) and attempt.get("source_sha256") == entry["source_sha256"], "AUDIT_ATTEMPT_SOURCE_MISMATCH")
        source = bound_hash(work / entry["nifti"], entry["source_sha256"], root=work)
        require(source not in seen_paths and entry.get("automated_qc") == "PASS", "AUDIT_SOURCE_DUPLICATE_OR_UNVERIFIED")
        seen_paths.add(source)
        acquisition = runner.adaptive.normalized_acquisition(entry, quality)
        profile = profiles.build_scan_profile(source, acquisition)
        plan = policy.build_research_plan(profile)
        expected_quality = quality.resolution_quality(acquisition, nib.load(source).affine)
        require(attempt.get("acquisition") == acquisition and attempt.get("quality") == expected_quality
                and attempt.get("slice_thickness_mm") == acquisition.get("slice_thickness_mm")
                and attempt.get("split") == entry.get("split"), "AUDIT_ATTEMPT_ACQUISITION_MISMATCH")
        _duration(attempt)
        snapshots = {"acquisition.local.json": acquisition, "scan-profile.local.json": profile, "processing-plan.local.json": plan}
        require(set(attempt.get("input_snapshot_sha256", {})) == set(snapshots), "AUDIT_INPUT_SNAPSHOT_SET_CHANGED")
        for name, value in snapshots.items():
            path = bound_hash(folder / name, attempt["input_snapshot_sha256"][name], root=folder)
            require(checks.read_json(path) == value, "AUDIT_INPUT_SNAPSHOT_CHANGED")
        bound_hash(folder / "worker.local.log", attempt.get("worker_log_sha256"), root=folder)
        worker = checks.contained(folder / "worker", folder)
        result = checks.read_json(bound_hash(worker / "result.json", attempt.get("result_sha256"), root=worker))
        require(result.get("schema_version") == 1 and result.get("status") == attempt.get("status")
                and result.get("status") in {"completed", "failed"}, "AUDIT_WORKER_STATUS_MISMATCH")
        require(result.get("source_sha256") == entry["source_sha256"] and result.get("protocol_version") == runner.PROTOCOL
                and result.get("method_version") == attempt.get("method_version") == runner.METHOD
                and result.get("config") == protocol["algorithm"], "AUDIT_WORKER_SOURCE_OR_METHOD_MISMATCH")
        require(result.get("analysis_mode") == "automatic" and result.get("assistance") is None
                and result.get("model_identity") == {"localization": "research_geometric_baseline", "direct_ethmoid_model": "not_used"}
                and result.get("research_status") == "NOT_ACCURACY_VALIDATED", "AUDIT_MODEL_OR_ASSISTANCE_MISMATCH")
        require(result.get("acquisition") == acquisition and result.get("quality") == expected_quality, "AUDIT_WORKER_ACQUISITION_MISMATCH")
        require(result.get("scan_profile") == profile and attempt.get("scan_profile_sha256") == profile["sha256"], "AUDIT_PROFILE_MISMATCH")
        require(result.get("processing_plan") == plan and attempt.get("processing_plan_sha256") == plan["sha256"], "AUDIT_PLAN_MISMATCH")
        require(result.get("localization_geometry", {}) == attempt.get("localization_geometry", {}), "AUDIT_LOCALIZATION_RECORD_MISMATCH")
        _cache_bindings(entry, attempt, folder, baseline, source, work, runtime)
        rejected_artifacts = {}
        if result["status"] == "completed":
            require(type(attempt.get("worker_returncode")) is int and attempt["worker_returncode"] == 0, "AUDIT_WORKER_EXIT_MISMATCH")
            metrics = prior.audit_success(result, attempt, worker, source, protocol)
            require(attempt.get("source_hu_air_consistency") == "PASS"
                    and attempt.get("native_voxel_volume_mm3") == metrics["native_voxel_volume_mm3"], "AUDIT_RECORDED_NATIVE_VALIDATION_MISMATCH")
            successes += 1
        else:
            require(type(attempt.get("worker_returncode")) is int and attempt["worker_returncode"] != 0, "AUDIT_WORKER_EXIT_MISMATCH")
            rejected_artifacts = prior.audit_failure(result, attempt, worker)
        previous = baseline[identity(entry)]
        old_result, old_attempt = previous["result"], previous["attempt"]
        require(old_result.get("scan_profile") == profile and old_result.get("processing_plan") == profiles.build_processing_plan(profile), "AUDIT_BASELINE_PROFILE_OR_PLAN_MISMATCH")
        _duration(old_attempt)
        if old_result["status"] == "completed":
            prior.audit_success(old_result, old_attempt, previous["directory"] / "worker", source, v3_protocol)
            baseline_successes += 1
        else:
            prior.audit_failure(old_result, old_attempt, previous["directory"] / "worker")
        attempts.append(attempt)
        records.append((attempt, profile))
        all_bindings.append({"source_sha256": entry["source_sha256"], "attempt_sha256": checks.sha256(folder / "attempt.local.json"),
                             "result_sha256": attempt["result_sha256"], "worker_log_sha256": attempt["worker_log_sha256"],
                             "artifact_sha256": result.get("artifact_verification", {}).get("sha256", {}),
                             "protected_rejected_artifact_sha256": rejected_artifacts,
                             "input_snapshot_sha256": attempt["input_snapshot_sha256"]})
    recomputed = _summary(attempts, expected_studies, expected_series, expected_ready, len(unavailable), checks.sha256(batch_path))
    require(recomputed == stored_summary, "AUDIT_AGGREGATE_SUMMARY_MISMATCH")
    require(checks.code_provenance(pipeline, require_clean=True) == code and checks.sha256(manifest) == expected_manifest_sha256
            and checks.sha256(protocol_path) == expected_protocol_sha256 and checks.sha256(summary_path) == summary_hash
            and checks.sha256(provenance_path) == provenance_hash
            and checks.sha256(batch_path) == stored_summary["batch_verification_sha256"]
            and all(checks.sha256(path) == attempts_hashes[path.parent.name] for path in expected_paths), "AUDIT_INPUT_CHANGED_DURING_AUDIT")
    for relative, digest in baseline_info["sha256"].items():
        bound_hash(baseline_dir / relative, digest, root=baseline_dir)
    return {"schema_version": 1, "status": "verified", "research_generation": generation, "accuracy_measured": False,
            "clinical_validation": "pending_independent_expert_references",
            "coverage": {"studies": expected_studies, "series": expected_series, "unique_sources": len(seen_paths), "unique_attempts": len(attempts), "unavailable": len(unavailable)},
            "bindings": {"code_sha": expected_code_sha, "manifest_sha256": expected_manifest_sha256,
                "protocol_sha256": expected_protocol_sha256, "runtime_sha256": runtime_hash, "provenance_sha256": provenance_hash,
                "summary_sha256": summary_hash, "batch_verification_sha256": checks.sha256(batch_path),
                "all_attempt_evidence_sha256": checks.canonical_sha256(all_bindings), "baseline_binding_sha256": checks.canonical_sha256(baseline_info),
                "auditor_sha256": checks.sha256(Path(__file__)), "audit_helper_sha256": checks.sha256(Path(__file__).with_name("audit_adaptive_evaluation.py"))},
            "summary": recomputed, "runtime_by_nominal_thickness_mm": prior.runtime_summary(attempts),
            "runtime_interpretation": "serial_local_elapsed_time_including_cache_checks_not_general_latency",
            "successful_native_results_verified": successes, "baseline_native_results_verified": baseline_successes,
            "candidate_aggregates": _completed_groups(records),
            "unavailable_reasons": dict(Counter(e["reason_code"] for e in unavailable)),
            "v3_comparison": _comparison(attempts, baseline)}


def main(argv=None):
    base = Path(os.environ["LOCALAPPDATA"]) / "EthmoidComplexity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=base / "work")
    parser.add_argument("--runtime-python", type=Path, default=base / "tools/counting-runtime/Scripts/python.exe")
    for name in ("pipeline-root", "manifest", "evaluation"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-code-sha", "expected-manifest-sha256", "expected-protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("studies", 26), ("series", 230), ("ready", 98)):
        parser.add_argument("--expected-" + name, type=int, default=default)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generation", type=int, choices=(4, 5), default=4)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    try:
        if output is not None:
            work = args["work_root"].resolve(strict=True)
            output = checks.contained(output, work, exists=False)
            require(not output.exists(), "AUDIT_OUTPUT_ALREADY_EXISTS")
            require(not output.is_relative_to(args["evaluation"].resolve(strict=True)), "AUDIT_OUTPUT_INSIDE_EVALUATION")
            require(not any((p / "provenance.local.json").is_file() for p in output.parents), "AUDIT_OUTPUT_INSIDE_EVALUATION")
            require(not any((p / ".git").exists() for p in output.parents), "AUDIT_OUTPUT_INSIDE_GIT")
        report = audit(**args)
        if output is not None:
            checks.write_new_json(output, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report["status"] == "verified" else 3
    except (checks.EvaluationError, runner.checks.EvaluationError, runner_v5.checks.EvaluationError) as error:
        print(json.dumps({"status": "failed", "failure_code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError, nib.filebasedimages.ImageFileError):
        print('{"status":"failed","failure_code":"AUDIT_INPUT_UNREADABLE_OR_INVALID"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
