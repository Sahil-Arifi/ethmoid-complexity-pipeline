# Video-guided instance prototype

This isolated research prototype learns foreground and same-cell relationships
from provisional video-guided annotations. It does not replace the production
counter, change the application, or supply a clinically validated cell model.
The existing repeated-count workflow remains the baseline.

## Current status

The implementation and 62 combined synthetic contracts passed in the existing
counting runtime. They cover label masking, geometry, graph decoding, loss,
reference isolation, evidence rejection, repeatability reporting and the model's
CPU forward pass. These checks establish engineering behavior, not anatomical
accuracy. Training and the eight-case prediction comparison are pending.

The first GPU attempt stopped because the installed native max-pool backward
has no deterministic CUDA implementation. It produced no checkpoint. An
equivalent disjoint two-cubed maximum now preserves the same forward values
and first-argmax gradients with determinism enabled. CPU equivalence controls
and two sequential, fresh-process CUDA forward/backward probes passed with
identical hashes. The failed attempt and its original software freeze are
retained; a new software freeze keeps the same annotations and configuration.

The private annotation release contains small interior cores and explicitly
observed boundary material. It does not trace complete cell extents and has no
clinician approval. Source reconstruction correspondence remains conditional.
The tenth historical selection is review-only; other unresolved selections stay
unknown. No unselected algorithm object becomes a negative example.

The release has limited directional supervision: different-cell targets occur
only at the four-voxel offset along the third axis. The first two axes have no
four-voxel targets of either class. One of the three demonstrated distinctions
has no direct training edge and remains a mechanism evaluation. Readiness does
not imply complete training coverage.

## Frozen model and supervision

The four-level 3D U-Net uses channels 8/16/32/64, two convolutions per level,
GroupNorm and SiLU. Two CT channels map HU [-1000,1000] and [-1000,200] to
[-1,1]. Model-only sampling uses 0.5 mm spacing and 64-cubed patches.
The source volume is never resampled or overwritten.

Seven logits predict foreground and six affinities. The affinities describe an
edge from the current voxel to a positive offset of one or four voxels along
each canonical array axis. Training uses masked, class-balanced binary
cross-entropy. Unknown targets have zero loss and gradient. Equal supported
instance IDs provide same-cell targets; different IDs provide different-cell
targets only when their relationship is explicitly supported. Missing classes
are not filled with invented observations.

Annotations describe native-voxel occupancy. Nearest-neighbor categorical
sampling transfers each native voxel's half-voxel footprint to model centers;
exact positive half-index ties select the higher native index. Unknown native
voxels remain unknown, and model centers outside the acquired extent are
ignored. This convention does not certify the underlying manual annotation.
HU sampling is linear. Left/right reflection is applied to labels before fresh
affinity targets are generated, preserving the offset direction convention.

Training is one configuration: AdamW, learning rate 0.001, weight decay 0.0001,
batch two, seed 20261102, 2,000 steps and a maximum 3,600-second GPU session.
Augmentations are left/right reflection and Gaussian HU noise with standard
deviation 20. Only a completed final checkpoint is usable; an interrupted or
budget-limited attempt cannot produce a substitute checkpoint.

Inference uses 64-cubed tiles with stride 32. Affinity averages include only
tiles containing both edge endpoints. Foreground and affinity thresholds are
fixed at 0.5, inclusively. Graph components become instance labels; long edges
require foreground endpoints but do not assert intervening voxel membership.
Labels return to the original grid with nearest sampling. Components smaller
than four native voxels are removed. The source-bound physical midline assigns
laterality, and crossing objects are flagged.

## External inputs and commands

Use the existing isolated GPU counting Python with this checkout's `src` on
`PYTHONPATH`. In the commands below, replace every angle-bracket placeholder
with an absolute local path. All manifests, annotations, frames, checkpoints,
logs, images and results must remain outside Git. Every output directory must
be new; previous attempts are preserved.

```text
python scripts/prototype_cell_instances.py readiness --manifest <annotation-manifest.json> --output <fresh-readiness-directory>
python scripts/prototype_cell_instances.py train --manifest <annotation-manifest.json> --output <fresh-training-directory>
python scripts/prototype_cell_instances.py infer --manifest <prediction-manifest.json> --output <fresh-prediction-directory>
python scripts/prototype_cell_instances.py mechanism --manifest <mechanism-manifest.json> --output <fresh-mechanism-directory>
python scripts/prototype_cell_instances.py report --manifest <comparison-manifest.json> --output <fresh-comparison-directory>
```

An annotation manifest binds the source hash, counting crop, physical midline,
annotation archive, frame hashes, timestamps, supported relationships and
separate source/boundary uncertainty. The annotation archive contains native
crop `instance_ids` (-1 unknown, 0 explicit background, positive supported IDs),
`foreground_labels` (-1 unknown, 0 non-cell, 1 cell) and its exact affine.
Ambiguous, mismatched and review-only evidence cannot authorize training labels.

A prediction manifest accepts only schema version, case ID, source record,
checkpoint path and checkpoint hash. Its source record accepts only the input
volume, source hash, preserved crop bounds and physical midline. It accepts no
reference counts, reference points or annotations. Inference uses the broad
preserved crop without the old membership/search masks. Reference comparison
happens after prediction has closed.

Run each of the eight preserved cases in two fresh processes using the same
completed checkpoint. Native masks, registry, counts and provenance are saved
per attempt. Bind the comparison manifest to the preserved baseline sources and
references; do not choose a reconstruction or checkpoint by count agreement.
Keep source and artifact hashes unchanged through closeout.

## Gates and interpretation

- Readiness requires traceable multi-slice same-cell and different-cell
  evidence, supported voxel annotations, and both foreground and affinity
  classes. Points alone fail. If readiness fails, stop before fitting.
- Mechanism checks require all three supported distinctions, covered
  within-cell connections and preservation of explicit exclusions. Unknown
  voxels remain unscored. All-split and all-merged results fail.
- The eight-case count target requires all eight outputs, at least two exact
  bilateral matches, total absolute side-count error below 37, and preservation
  of the previously exact case. The seven cases outside training must improve
  from combined error 35; otherwise the result is a training-case fit.
- Two fresh processes must produce identical native masks and counts using one
  checkpoint. Failures remain in the eight-case denominator. Missing results
  are not replaced by baseline predictions. The combined development verdict
  also requires mechanism evidence bound to that checkpoint and prediction.

Report side-specific errors, regressions, unavailable predictions, known-region
coverage and every observed merge or split. Equal totals can hide compensating
errors and do not establish correct cell identities. All eight cases were
previously inspected development data; they are not an independent holdout.
Provisional annotations and incomplete extents cannot establish clinical
accuracy or generalization. A failed gate closes this attempt without changing
thresholds or trying another checkpoint against the supplied totals.
