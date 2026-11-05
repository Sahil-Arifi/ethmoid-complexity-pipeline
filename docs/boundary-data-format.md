# Boundary and instance labels

`ethmoid_pipeline.counting.boundary_data` keeps source volumes, annotations and
review releases in absolute paths outside Git workspaces. All examples below
refer to synthetic data. No function uploads images or initializes a model.

A dataset JSON has exactly these fields:

```json
{"schema_version": 1, "version": "boundary-instance-v1", "samples": []}
```

Each sample has exactly `sample_id`, `patient_group`, `scope`, `source`,
`annotation`, `protocol_sha256` and `provenance`. Sample and patient-group IDs are
opaque tokens of at most 64 ASCII letters, digits, underscores or hyphens.
`protocol_sha256` is the lowercase SHA-256 digest of the annotation protocol.
The caller must group all reconstructions and scans of the same patient
together. Conflicting patient groups for the same source digest are rejected;
different reconstructions cannot be linked automatically.

`source` uses the existing instance-experiment contract without additions:
`input`, `source_sha256`, `crop_bounds`, `midline_ras_x`. Its bounds are expressed
in the closest-canonical native source grid. `annotation` is an absolute NPZ
`path` and its `sha256`. Count references and extra manifest fields are rejected.

The annotation archive contains exactly:

| Array | Meaning |
| --- | --- |
| `foreground_labels` | Signed integer: -1 unknown, 0 explicit negative, 1 explicit positive. |
| `separator_labels` | Signed integer: -1 unknown, 0 explicit negative, 1 explicit positive. |
| `instance_ids` | Signed integer: -1 unknown, 0 known background, positive local instance ID. IDs must fit int32. |
| `evaluation_region` | Boolean region explicitly designated for evaluation. Every included voxel must have a known instance/background ID and known foreground label. It does not establish anatomical completeness or clinical validity. |
| `affine` | Finite 4 by 4 matrix matching the canonical native crop affine within absolute tolerance 1e-6. |

Separator labels describe voxel-centered boundary evidence. They are not
directional face-affinity labels. Partition adapter v3 converts each valid
face to `1 - max(boundary_at_low_endpoint, boundary_at_high_endpoint)` so a
separator contributes whichever adjacent watershed region owns that voxel.
Background remains excluded from the graph. This conversion changes no
annotation values and does not infer a complete separator from sparse labels.

All voxel arrays match that native crop, with no resizing. Instance foreground
and explicit foreground/background labels may not contradict each other.
Unknown voxels remain unknown; an evaluation region containing unknown instance
or foreground voxels is rejected. Separator completeness is independent of
instance evaluation. Separator labels are never manufactured from
instance-core edges. Headers are inspected before loading arrays. Limits are
64 million crop voxels, 128 million summed release voxels and 128 million source
voxels. The source limit also bounds full-source canonical reorientation in the
existing loader.

`scope` and `provenance` are paired as follows:

```json
{"scope": "synthetic_truth", "provenance": {"kind": "synthetic_truth", "generator": "synthetic-generator-v1"}}
{"scope": "conditional_native", "provenance": {"kind": "conditional_native", "review_status": "conditional"}}
{"scope": "adjudicated_expert", "provenance": {"kind": "adjudicated_expert", "annotator": "rater-a", "reviewer": "rater-b", "review_status": "adjudicated", "independent_review": true}}
```

Expert annotator/reviewer tokens must differ. These declarations record the
caller's provenance; validation cannot verify that a clinical review occurred.

## Python API

```python
from ethmoid_pipeline.counting import boundary_data

validated = boundary_data.validate_release(manifest_dict)
hu, crop_affine, arrays = boundary_data.load_sample(validated["samples"][0])
report = boundary_data.readiness(manifest_path, fresh_external_output_directory)
summary = boundary_data.export(manifest_path, another_fresh_external_directory)
```

`validate_release` returns `samples` and `readiness`. Reports contain per-head
positive, negative and unknown voxel counts, per-sample counts, scope counts and
patient groups. Both classes of both heads must occur in the training
partition. With multiple groups, the last sorted group is held out and all
others train. The held-out group must have at least some known supervision in
each head; otherwise readiness is `NOT_READY` with a specific request for the
missing held-out head. The holdout is never silently folded into training.
One group is `MECHANISM_ONLY`; it cannot provide held-out validation.
`validation_available` requires both a held-out group and supervision in both
heads, while `holdout_group_available` reports group availability alone.
`clinical_validation_established` stays false.

`export` creates `release.local.json` with the exact dataset schema and copied
NPZ paths, plus `readiness.local.json`. Each sample folder contains the exact
NPZ copy, native `source_volume.nii.gz`, signed int8 foreground/separator label
files, int32 instance labels, uint8 `*_known.nii.gz` masks, a uint8 evaluation
region and `review.local.json`. The source volume is a native crop, not a full
source-series copy. Review metadata binds the original source hash, annotation
hash, patient group, protocol and provenance to the review layers.

## Return edited Slicer layers to NPZ

```python
annotation = boundary_data.import_review(
    sample, exported_sample_directory, fresh_external_npz_path
)
edited_sample = {**sample, "annotation": annotation}
```

Use the original sample or its exported equivalent with the same source and
annotation digest. Keep filenames and geometry intact. In pinned Slicer 5.12.3
revision 34627, load the source volume, import each label layer into a separate
segmentation, and retain each corresponding uint8 known mask as a separate
editable layer. Keep foreground, separator and instance segmentations separate.
Use the source volume's exact geometry for export; do not resample, crop or
smooth the source. Export labels and known masks under their original filenames.

Add or retract supervision by editing both the label layer and its uint8 known
mask. A mask value of 1 means the label is known and nonnegative. A mask value
of 0 restores label -1, accepting either a signed -1 or Slicer's exported
background 0. Positive painted labels outside the known mask are rejected so
they cannot silently disappear. This preserves untouched unknown voxels,
including unknown instances, even when Segment Editor exports background as 0.
The source volume cannot be edited. This interchange is numerically tested;
the workflow still requires a human editing and round-trip check in Slicer.

The helper returns a new annotation path/hash record and never overwrites an
existing NPZ. It does not change the sample's scope or promote a conditional
annotation to adjudicated expert evidence. Any later provenance change needs
the corresponding review to have actually occurred.
