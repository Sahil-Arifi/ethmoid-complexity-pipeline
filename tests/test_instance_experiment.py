"""Synthetic-only contracts for the isolated instance research runner."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")  # The pinned counting runtime executes this numerical suite.

from ethmoid_pipeline.counting import instance_experiment as experiment


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def source(tmp_path, *, shape=(9, 10, 11), affine=None):
    affine = np.diag([.5, .75, 1., 1.]) if affine is None else affine
    path = tmp_path/"synthetic.nii"
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)-1000
    nib.save(nib.Nifti1Image(values, affine), path)
    return {"input": str(path), "source_sha256": experiment.sha256(path),
            "crop_bounds": [[0, n] for n in shape], "midline_ras_x": 2.}, values


def evidence(tmp_path, kind, ids):
    frames = []
    for i in range(2):
        p = tmp_path/f"{kind}-{i}.txt"
        p.write_text("synthetic view", encoding="utf-8")
        frames.append({"path": str(p), "sha256": experiment.sha256(p)})
    return {"kind": kind, "source_match": "provisional", "boundary_support": "supported",
            "point_ids": [1], "timestamps_seconds": [1, 2], "frames": frames, "label_ids": ids}


def annotation_manifest(tmp_path):
    record, _ = source(tmp_path, shape=(8, 8, 8), affine=np.diag([.5, .5, .5, 1.]))
    labels = np.full((8,)*3, -1, np.int32)
    labels[1:4, 1:6, 1:6] = 1
    labels[4:7, 1:6, 1:6] = 2
    foreground = np.full_like(labels, -1, dtype=np.int8)
    foreground[labels > 0] = 1
    foreground[0] = 0
    path = tmp_path/"annotations.npz"
    np.savez_compressed(path, instance_ids=labels, foreground_labels=foreground, affine=np.diag([.5, .5, .5, 1.]))
    return {"schema_version": 1, "source": record,
            "annotation": {"path": str(path), "sha256": experiment.sha256(path)},
            "distinct_pairs": [[1, 2]], "evidence": [evidence(tmp_path, "same_cell", [1]),
                evidence(tmp_path, "different_cell", [1, 2]), evidence(tmp_path, "background", []),
                evidence(tmp_path, "same_cell", [2])]}


def test_source_hash_and_strict_inference_schema_prevent_reference_leakage(tmp_path):
    record, _ = source(tmp_path)
    assert experiment.source_record(record) == record
    with pytest.raises(experiment.ExperimentError, match="SOURCE_SCHEMA_INVALID"):
        experiment.source_record({**record, "reference": {"left": 3}})
    with pytest.raises(experiment.ExperimentError, match="SOURCE_HASH_MISMATCH"):
        experiment.source_record({**record, "source_sha256": "0"*64})
    manifest = write(tmp_path/"bad-infer.json", {"schema_version":1,"case_id":"Synthetic", "source":record,
        "checkpoint":"unused", "checkpoint_sha256":"unused", "annotations":"must not enter"})
    with pytest.raises(experiment.ExperimentError, match="INFERENCE_MANIFEST_INVALID"):
        experiment.infer(manifest, tmp_path/"never-created")
    assert not (tmp_path/"never-created").exists()


def test_artifacts_require_absolute_non_git_fresh_directories(tmp_path):
    with pytest.raises(experiment.ExperimentError):
        experiment.external("relative.json", exists=False)
    (tmp_path/".git").mkdir()
    with pytest.raises(experiment.ExperimentError):
        experiment.external(tmp_path/"inside", exists=False)
    (tmp_path/".git").rmdir()
    out = experiment.new_output(tmp_path/"one")
    assert out.is_dir()
    with pytest.raises(experiment.ExperimentError, match="FRESH_OUTPUT_REQUIRED"):
        experiment.new_output(out)


def test_failure_metadata_is_fixed_code_external_and_never_overwrites(tmp_path):
    output=experiment.new_output(tmp_path/"failed-run")
    secret="synthetic-private-path-and-header"
    error=RuntimeError("kernel does not have a deterministic implementation "+secret)
    assert experiment.record_failure("train",output,error)=="DETERMINISTIC_KERNEL_UNAVAILABLE"
    path=output/"failure.local.json"
    text=path.read_text()
    record=json.loads(text)
    assert secret not in text and "RuntimeError"==record["exception_class"]
    assert record["last_recorded_completed_step"]==0 and not record["exact_completed_steps_known"]
    assert record["input_preservation"]=="NOT_CHECKED_AFTER_FAILURE"
    experiment.record_failure("train",output,ValueError(secret))
    assert path.read_text()==text
    missing=tmp_path/"never-created"
    assert experiment.record_failure("train",missing,ValueError(secret))=="EXPERIMENT_EXECUTION_FAILED"
    assert not missing.exists()


def test_failure_progress_is_explicit_lower_bound_and_oom_has_fixed_code(tmp_path):
    output=experiment.new_output(tmp_path/"failed-run")
    (output/"progress.local.jsonl").write_text(json.dumps({"step":100,"loss":.1})+"\n")
    experiment.record_failure("train",output,experiment.ExperimentError("SYNTHETIC_FAILURE"))
    record=json.loads((output/"failure.local.json").read_text())
    assert record["last_recorded_completed_step"]==100
    torch=pytest.importorskip("torch")
    assert experiment.failure_code(torch.cuda.OutOfMemoryError("private"))=="CUDA_OUT_OF_MEMORY"


def test_points_alone_produce_not_ready_and_training_never_initializes_torch(tmp_path, monkeypatch):
    record, _ = source(tmp_path)
    manifest = {"schema_version":1,"source":record,"annotation":None,"evidence":[],"distinct_pairs":[]}
    path = write(tmp_path/"release.json", manifest)
    result = experiment.readiness(path, tmp_path/"readiness")
    assert result["status"] == "NOT_READY" and result["training_allowed"] is False
    assert "SUPPORTED_VOXEL_ANNOTATIONS_MISSING" in result["reasons"]
    monkeypatch.setattr(experiment, "deterministic_torch", lambda: pytest.fail("must not initialize training"))
    with pytest.raises(experiment.ExperimentError, match="READINESS_FAILED_NO_TRAINING"):
        experiment.train(path, tmp_path/"training")
    assert (tmp_path/"training/readiness.local.json").is_file()
    assert not list((tmp_path/"training").glob("*.pt"))


def test_supported_partial_regions_pass_but_unknowns_remain_ignored(tmp_path):
    manifest = annotation_manifest(tmp_path)
    result = experiment.inspect_readiness(manifest)
    assert result["training_allowed"]
    assert result["label_counts"]["same_cell_edges"] > 0
    assert result["label_counts"]["different_cell_edges"] > 0
    assert result["label_counts"]["unknown_foreground_voxels"] > 0
    assert result["source_match_uncertainty"] == ["provisional"]*4
    assert result["boundary_uncertainty"] == ["supported"]*4


def test_unsupported_pair_and_background_are_rejected(tmp_path):
    manifest = annotation_manifest(tmp_path)
    manifest["distinct_pairs"] = [[1, 3]]
    with pytest.raises(experiment.ExperimentError, match="UNSUPPORTED_DISTINCT_PAIR"):
        experiment.inspect_readiness(manifest)
    manifest["distinct_pairs"] = [[1, 2]]
    manifest["evidence"] = [r for r in manifest["evidence"] if r["kind"] != "background"]
    with pytest.raises(experiment.ExperimentError, match="BACKGROUND_EVIDENCE_REQUIRED"):
        experiment.inspect_readiness(manifest)


@pytest.mark.parametrize("field,value", [("point_ids", [10]), ("source_match", "ambiguous"),
                                         ("source_match", "mismatch"), ("boundary_support", "ambiguous")])
def test_review_only_or_ambiguous_relations_cannot_make_readiness(tmp_path, field, value):
    manifest = annotation_manifest(tmp_path)
    manifest["evidence"][0][field] = value
    with pytest.raises(experiment.ExperimentError, match="UNSUPPORTED_INSTANCE_VOXELS"):
        experiment.inspect_readiness(manifest)


def test_one_video_time_with_two_native_views_is_traceable_without_invented_timestamp(tmp_path):
    manifest = annotation_manifest(tmp_path)
    for record in manifest["evidence"]:
        record["timestamps_seconds"] = [1]
        record["native_views"] = [{"axis":0,"index":2},{"axis":1,"index":3}]
    assert experiment.inspect_readiness(manifest)["training_allowed"]
    manifest["evidence"][0]["native_views"][1]["index"] = 100
    with pytest.raises(experiment.ExperimentError,match="NATIVE_VIEW_OUTSIDE_SOURCE_CROP"):
        experiment.inspect_readiness(manifest)


def test_instance_zero_background_cannot_bypass_evidence_and_extra_instance_is_rejected(tmp_path):
    manifest = annotation_manifest(tmp_path)
    path = Path(manifest["annotation"]["path"])
    with np.load(path) as archive:
        arrays = {key:archive[key] for key in archive.files}
    arrays["foreground_labels"].fill(-1)
    arrays["instance_ids"][0] = 0
    np.savez_compressed(path,**arrays)
    manifest["annotation"]["sha256"] = experiment.sha256(path)
    manifest["evidence"] = [r for r in manifest["evidence"] if r["kind"] != "background"]
    with pytest.raises(experiment.ExperimentError,match="BACKGROUND_EVIDENCE_REQUIRED"):
        experiment.inspect_readiness(manifest)
    manifest["evidence"].append(evidence(tmp_path,"background",[]))
    assert experiment.inspect_readiness(manifest)["training_allowed"]
    arrays["instance_ids"][7,7,7] = 10
    np.savez_compressed(path,**arrays)
    manifest["annotation"]["sha256"] = experiment.sha256(path)
    with pytest.raises(experiment.ExperimentError,match="UNSUPPORTED_INSTANCE_VOXELS"):
        experiment.inspect_readiness(manifest)


def test_annotation_affine_mismatch_and_input_mutation_fail_closed(tmp_path):
    manifest = annotation_manifest(tmp_path)
    path = Path(manifest["annotation"]["path"])
    with np.load(path) as data:
        arrays = {k:data[k] for k in data.files}
    arrays["affine"][0,3] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(experiment.ExperimentError, match="ANNOTATION_HASH_MISMATCH"):
        experiment.inspect_readiness(manifest)
    manifest["annotation"]["sha256"] = experiment.sha256(path)
    with pytest.raises(experiment.ExperimentError, match="ANNOTATION_VALUES_INVALID"):
        experiment.inspect_readiness(manifest)


def test_model_grid_and_native_projection_preserve_physical_coordinates():
    hu = np.arange(3*4*5, dtype=np.float32).reshape(3,4,5)
    affine = np.diag([1., 1.5, .5, 1.])
    grid, valid = experiment.model_grid(hu, affine)
    assert grid.shape == (5,10,5) and valid.all()
    assert np.array_equal(grid[::2,::3,:], hu)
    labels = np.ones(grid.shape, np.int32)
    labels[2:] = 2
    native = experiment.native_labels(labels, hu.shape, affine)
    assert native.shape == hu.shape
    assert np.all(native[0] == 1) and np.all(native[1:] == 2)


def test_regridding_preserves_known_and_unknown_native_half_voxel_footprints():
    hu = np.zeros((3,3,3), np.float32)
    ids = np.full_like(hu,-1,dtype=np.int32)
    ids[1] = 7
    fg = np.where(ids>0,1,-1).astype(np.int8)
    _, _, labels, foreground = experiment.model_grid(hu,np.diag([1.,1.,1.,1.]),ids,fg)
    # Native index1 occupies [0.5,1.5). Model coordinates are0,0.5,1,1.5,2.
    # Both0.5and1 sample index1. The1.5tie samples unknown index2, not index1.
    assert np.all(labels[[1,2]] == 7)
    assert np.all(labels[[0,3,4]] == -1)
    assert np.all(foreground[[1,2]] == 1)
    assert np.all(foreground[[0,3,4]] == -1)


def test_categorical_half_index_ties_and_background_use_same_policy():
    hu = np.zeros((3,3,3),np.float32)
    ids = np.broadcast_to(np.array([0,4,9],np.int32)[:,None,None],hu.shape).copy()
    fg = np.where(ids > 0,1,0).astype(np.int8)
    _,valid,labels,foreground = experiment.model_grid(hu,np.diag([1.,1.,1.,1.]),ids,fg)
    assert valid.all()
    assert labels[:,0,0].tolist() == [0,4,4,9,9]
    assert foreground[:,0,0].tolist() == [0,1,1,1,1]


def test_categorical_sampling_keeps_outside_acquired_voxel_extent_unknown():
    hu = np.zeros((2,2,2),np.float32)
    ids = np.full(hu.shape,4,np.int32)
    fg = np.ones(hu.shape,np.int8)
    _,valid,labels,foreground = experiment.model_grid(hu,np.diag([.625,.625,.625,1.]),ids,fg)
    assert np.any(~valid)
    assert np.all(labels[valid] == 4) and np.all(foreground[valid] == 1)
    assert np.all(labels[~valid] == -1) and np.all(foreground[~valid] == -1)


def test_point625_grid_last_native_plane_roundtrip_uses_acquired_voxel_footprint():
    hu = np.ones((3,3,3),np.float32)
    affine = np.diag([.625,.625,.625,1.])
    _,valid = experiment.model_grid(hu,affine)
    # The last model center maps to native2.4, within native voxel2's footprint.
    # Native voxel2 maps to model2.5, whose order0 tie selects model index3.
    assert valid.shape == (4,4,4) and valid.all()
    model_labels = valid.astype(np.int32)
    projected = experiment.native_labels(model_labels,hu.shape,affine)
    assert np.all(projected == 1), "An acquired native boundary plane must not become forced background"


def test_native_volume_filter_and_ras_laterality():
    labels = np.zeros((7,3,3),np.int32)
    labels[0,0,0] = 3  # Native one-voxel object is removed.
    labels[1:3] = 5
    labels[4:6] = 8
    projected = experiment.native_labels(labels,labels.shape,np.diag([.5,.5,.5,1.]))
    assert set(np.unique(projected)) == {0,1,2}
    registry, counts = experiment.label_registry(projected,np.zeros_like(labels,dtype=float),np.diag([.5,.5,.5,1.]),1.6)
    assert counts == {"left":1,"right":1,"total":2}
    assert registry[0]["centroid_ras_mm"][0] < 1.6 < registry[1]["centroid_ras_mm"][0]


def test_patch_padding_is_unknown_and_tiling_covers_remainders():
    values = np.ones((3,4,5),np.int32)
    patch = experiment.patch(values,[0,0,0],8,-1)
    assert patch.shape == (8,)*3 and patch[0,0,0] == -1 and patch[4,4,4] == 1
    for size in (1,63,64,65,97,128):
        hits = np.zeros(size,int)
        for start in experiment.starts(size):
            hits[start:start+64] += 1
        assert np.all(hits > 0)


def test_tiling_does_not_average_untrained_affinities_at_patch_edges():
    from contextlib import nullcontext
    class Tensor:
        def __init__(self,array): self.array=array
        def to(self,_device): return self
        def cpu(self): return self
        def numpy(self): return self.array
        def __getitem__(self,index): return Tensor(self.array[index])
    class Torch:
        @staticmethod
        def from_numpy(array): return Tensor(array)
        @staticmethod
        def sigmoid(tensor): return tensor
        @staticmethod
        def inference_mode(): return nullcontext()
    class Model:
        def eval(self): pass
        def __call__(self,tensor):
            values=np.full((1,7,64,64,64),.25,np.float32)
            values[:,0]=.7
            for head,offset in enumerate(experiment.core.OFFSETS,start=1):
                axis=next(i for i,v in enumerate(offset) if v)
                boundary=[slice(None)]*5
                boundary[1]=head
                boundary[axis+2]=slice(64-offset[axis],64)
                values[tuple(boundary)]=1
            return Tensor(values)
    probabilities=experiment.tiled_probabilities(np.zeros((70,9,10),np.float32),Model(),Torch())
    assert np.allclose(probabilities[0],.7)
    for head,offset in enumerate(experiment.core.OFFSETS,start=1):
        source,_=experiment.core.edge_slices((70,9,10),offset)
        assert np.allclose(probabilities[(head,*source)],.25)
        expected=np.zeros((70,9,10),bool); expected[source]=True
        assert np.all(probabilities[head,~expected]==0)


def test_mechanism_scores_only_supported_regions_and_detects_trivial_merge(tmp_path):
    annotation = annotation_manifest(tmp_path)
    annotation_path = write(tmp_path/"release.json",annotation)
    with np.load(annotation["annotation"]["path"]) as archive:
        values = np.maximum(archive["instance_ids"],0).astype(np.uint16)
    values[7,7,7] = 99  # Unknown area is not a false-positive reference.
    prediction_dir = tmp_path/"prediction"
    prediction_dir.mkdir()
    artifact = prediction_dir/"instances.local.nii.gz"
    nib.save(nib.Nifti1Image(values,np.diag([.5,.5,.5,1.])),artifact)
    record = {"status":"COMPLETE","source_sha256":annotation["source"]["source_sha256"],
              "annotation_manifest_sha256":experiment.sha256(annotation_path),
              "artifact_sha256":experiment.sha256(artifact)}
    prediction_path = write(prediction_dir/"prediction.local.json",record)
    manifest = write(tmp_path/"mechanism.json",{"schema_version":1,"annotation_manifest":str(annotation_path),
                                              "prediction":str(prediction_path)})
    result = experiment.mechanism(manifest,tmp_path/"mechanism-out")
    assert result["unknown_voxels_unscored"] > 0
    assert result["false_positive_known_background_voxels"] == 0
    assert result["gates"]["annotated_within_cell_connections"]
    assert result["distinctions"][0]["separate"]
    assert not result["gates"]["three_supported_distinctions"]
    values[values == 2] = 1
    nib.save(nib.Nifti1Image(values,np.diag([.5,.5,.5,1.])),artifact)
    record["artifact_sha256"] = experiment.sha256(artifact)
    write(prediction_path,record)
    result = experiment.mechanism(manifest,tmp_path/"merged-out")
    assert not result["distinctions"][0]["separate"] and result["status"] == "FAIL"


def report_entries():
    # Artificial cases retain the required aggregate baseline error without clinical data.
    cases = ["Case024_Series002","Case008_Series004",*[f"Synthetic{i}" for i in range(6)]]
    result = []
    for index, case in enumerate(cases):
        reference = {"left":10,"right":10}
        baseline = {"left":10,"right":10}
        if index == 0: baseline["right"] = 8
        if index == 2: baseline["left"] = 45
        result.append({"case_id":case,"reference":reference,"baseline":baseline,
                       "source_sha256":str(index)*64,"predictions":[]})
    return result


def test_report_keeps_all_eight_not_run_and_never_substitutes_baseline(tmp_path):
    manifest = write(tmp_path/"report.json",{"schema_version":1,"entries":report_entries()})
    result = experiment.report(manifest,tmp_path/"out")
    assert result["status"] == "NOT_RUN" and result["scheduled_cases"] == 8
    assert result["scheduled_sides"] == 16 and result["complete_cases"] == 0
    assert result["absolute_side_error"] is None
    assert all(r["prototype"] is None for r in result["cases"])


def test_report_requires_two_distinct_workers_and_one_checkpoint(tmp_path):
    entries = report_entries()
    for index, entry in enumerate(entries):
        for repeat in range(2):
            directory = tmp_path/f"prediction-{index}-{repeat}"
            directory.mkdir()
            values = np.arange(1,21,dtype=np.uint16).reshape(20,1,1)
            artifact = directory/"instances.local.nii.gz"
            nib.save(nib.Nifti1Image(values,np.eye(4)),artifact)
            record = {"status":"COMPLETE","version":experiment.VERSION,"case_id":entry["case_id"],
                      "counts":{**entry["reference"],"total":20},"source_sha256":entry["source_sha256"],
                      "inputs_unchanged":True,"native_array_sha256":experiment.hashlib.sha256(values.tobytes()).hexdigest(),
                      "native_shape":[20,1,1],"native_affine":np.eye(4).tolist(),
                      "artifact_sha256":experiment.sha256(artifact),
                      "registry":[{"id":i,"side":"left" if i <= 10 else "right"} for i in range(1,21)],
                      "checkpoint_sha256":"b"*64,
                      "config_sha256":experiment.fingerprint(experiment.CONFIG),
                      "worker_identity":f"{index}-{repeat}","worker_pid":100+index*2+repeat}
            entry["predictions"].append(str(write(directory/"prediction.local.json",record)))
    manifest = write(tmp_path/"report.json",{"schema_version":1,"entries":entries})
    result = experiment.report(manifest,tmp_path/"out")
    assert result["status"] == "FAIL" and result["count_status"] == "PASS_DEVELOPMENT_COUNTS"
    assert result["mechanism_status"] == "NOT_EVALUATED"
    assert result["bilateral_exact"] == 8 and result["absolute_side_error"] == 0
    entries[0]["predictions"][1] = entries[0]["predictions"][0]
    write(tmp_path/"report2.json",{"schema_version":1,"entries":entries})
    result = experiment.report(tmp_path/"report2.json",tmp_path/"out2")
    assert result["status"] == "FAIL" and not result["gates"]["two_repeat_identical_native_masks_and_counts"]
    Path(entries[1]["predictions"][0]).unlink()
    write(tmp_path/"report3.json",{"schema_version":1,"entries":entries})
    result = experiment.report(tmp_path/"report3.json",tmp_path/"out3")
    assert result["complete_cases"] == 7 and result["cases"][1]["status"] == "FAILED"
    assert result["absolute_side_error"] is None
