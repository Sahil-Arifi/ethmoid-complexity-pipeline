# Synthetic observed-wall learning experiment

This is a quarantined training experiment for one boundary-selection component.
It is **not an ethmoid/sphenoid segmenter, a cell counter, expert supervision, or
an application model**. The latest audited patient result remains research v5:
15/98 candidate counts, 83 review outcomes, zero technical errors. This experiment
cannot establish the requested 98/98 anatomical completion or accuracy.

## Why this experiment exists

Of the 83 remaining v5 abstentions, 66 concern sphenoid localization. Diagnostics
separate absent/competing parents, insufficient seed support, first-hit wall
selection and a restrictive whole-surface fit. The existing surrounding-anatomy
models do not directly label the ethmoid and sphenoid. This experiment addresses
observed wall selection after a posterior seed is already available; it cannot
resolve every preceding localization failure.

A new public-source search still did not verify an openly downloadable conventional
CT corpus with the required target regions or individual-cell references. The
[source review](RESEARCH_DATA_SOURCES.md) records additional atlas/access leads.
Synthetic training permits known geometric targets without relabeling existing
predictions as truth. Its simplified equations do not cover real sinus anatomy.

## Frozen engineering design

- Eight equation families supply planes, bowls, steep, asymmetric and curved
  walls. Every geometry and all its augmentations belong to exactly one partition;
  entire equation families are held out for validation and testing.
- Rendered HU examples vary opacity, noise, blur and physical spacing, including
  2.5 mm sampling. The renderer uses a documented local-plane partial-volume
  approximation, not a CT reconstruction simulator. Native bone erased by
  sampling does not become a positive visible-wall target.
- Both internal and external partial septa are distractors. Missing, cropped,
  averaged-away and competing complete walls are negative controls. Targets come
  from analytic material provenance intersected with observed leading bone faces.
  Target and distractor thicknesses are independently drawn from the same
  Uniform(1.2, 2.0) mm distribution. Opaque chamber and surrounding tissue share a
  per-augmentation Uniform(-20, 80) HU draw, preventing class-specific intensity
  and thickness cues from trivially revealing the target.
- All observed bone runs are enumerated within the supplied native seed columns
  and bounded search extent. A model never creates an observed candidate.
- A small three-dimensional convolutional classifier sees HU, posterior-seed and
  acquired-extent channels. Only model-input patches are sampled on a physical
  grid; source images and final cell instances are not resampled or modified.
  HU is clipped to [-1000, 2000] and divided by 1000 for model input only.
- A deterministic, bounded multi-start local surface search combines observed
  candidates. It is approximate and may miss a feasible surface. It rejects
  insufficient, disconnected, collinear, cropped and unsupported-hole evidence.
  A second coherent observed sheet remains ambiguity even if the model scores it
  poorly. Returned support points are native observed coordinates; unsupported
  locations remain unavailable.
  Native depth quantization uses simultaneous graph-consistent voxel intervals;
  independent edge slack cannot accumulate into a physically too-steep ramp.

Default experiment: four complete geometries per family, four acquisition
augmentations, at most 96 balanced candidate patches per scene, 17-cubed physical
patches at 1 mm model spacing, batch 64, 15 epochs. The classifier is selected by
validation binary cross-entropy only; the fixed score threshold is 0.5. Test
families are used only after selection. Configurations and source hashes are
recorded before rendering. These are development experiments, not clinical
thresholds or claims that coarse acquisitions recover missing septa.

## Verification and promotion boundary

Classification precision/recall and loss are reported separately from full-scene
surface behavior. Classification uses class-capped samples; its precision does
not estimate natural candidate prevalence or calibrated clinical probabilities.
The full-scene benchmark scores every extracted candidate. With eight total
families the fixed split contains five training, two validation and one test
family, so this is a limited withheld-family experiment. The scene benchmark compares the learned decoder, image-only
decoder and a boundary-only reproduction of the v5 first-hit/gradient/quadratic
procedure. This benchmark does not run the full anatomical localizer or cell
counter. It records every abstention and false acceptance.

The initial synthetic gate requires zero accepted negative controls, at least
one accepted observable scene, observed-target precision of at least 0.98 on
every accepted scene and mean leading-surface position error no larger than
max(1 mm, native Y spacing). A zero-completion method cannot pass vacuously.
Passing this small engineering benchmark alone does not authorize clinical-model
promotion. The existing expert-reference model interface cannot load these
synthetic research checkpoints, and no counting route selects them.

Execution checks cover immutable split identity, no repeated sampled voxel,
finite normalized channels, bounded memory allocations, unsafe output paths,
source-checkout/import matching, checkpoint hashes, real optimization and
reloaded prediction agreement. Exact totals and measured experiment results are
reported after commands finish; unexecuted tests are not claimed as passed.

At the source-freeze checkpoint, **391 CPU checks** passed, including the prior
220 engine tests. **Eight real PyTorch integrations** passed on the local CPU/GPU
runtime. One Windows directory-symlink-privilege test was skipped. GPU optimization
is locally tested, not part of the CPU-only GitHub job. The complete training run
and held-out scene outcomes remain pending at this checkpoint.

## Local execution and artifacts

Use the existing pinned Python 3.12 GPU runtime with its outbound-network guard
and `PYTHONPATH` set to the **same clean committed checkout's** `src` directory:

```powershell
& $InferencePython scripts/train_wall_proposal_model.py `
  --work-root $ProtectedWorkRoot `
  --output (Join-Path $ProtectedWorkRoot 'wall-learning/experiment-001')
```

`$InferencePython` and `$ProtectedWorkRoot` are installation-local values. The
command takes no patient source or application database argument. Generated
arrays, individual geometry inventories, checkpoints and reports remain outside
Git. Existing destinations and Git-contained destinations are rejected. Dataset
arrays, geometry splits, source files and final weights have recorded hashes.

Training writes fsynced atomic best/last checkpoints and immutable epoch reports.
It records technical interruptions without publishing a count. **Exact training
resume is not implemented**; interrupted releases are retained, and a new release
is required to restart. This limitation does not alter the application's durable
count-job recovery. No generated reference label is presented as an expert label.

The application default, previous protocols, original DICOMs, production database
and all previous v3/v4/v5 evidence remain untouched by this training command.
