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

## Experiment 001: completed, not promoted

Frozen source: [`1368662`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/1368662e9ff7af7377e57edf6b06ccc8a3106282).
The actual run used **eight geometries per family**, for 64 geometries total:
40 training, 16 validation and eight test geometries. Families were disjoint
(five/two/one). The test family was asymmetric walls. All 256 present-wall
augmentations were inventoried; sampled candidates were 15,360 training, 6,144
validation and 3,072 test. No patient images or expert masks were used.

Fifteen epochs ran on the local RTX 5070 Ti. Validation loss decreased from
0.694589 to a best 0.037030 at epoch 12, which selected the saved checkpoint.
Optimization/checkpointing took **12.880 seconds**, with peak PyTorch-allocated
GPU memory **129,949,184 bytes**. Complete generation, training and evaluation took
**44.485 seconds**. These are measurements of this small synthetic model on this
workstation, not clinical inference or nnU-Net training estimates.

On the class-capped test patches, precision was **95.55%**, recall **99.28%** and
binary cross-entropy **0.074373**. These are synthetic candidate-classification
metrics, **not ethmoid-cell accuracy**.

The full-scene benchmark retained 32 present-wall scenes and 32 negative controls
from the eight test geometries. Of those 64 scenes, 26 met the independent
observability definition and 38 did not.

| Boundary method | Accepted / 64 | Abstained | Accepted observable scenes | False acceptances | Fixed engineering gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Image-only approximate decoder | 2 | 62 | 0 | 2 | Failed |
| Learned scores with approximate decoder | 21 | 43 | 17 | 4 | Failed |
| First-hit/gradient/quadratic boundary proxy | 64 | 0 | 26 | 38 | Failed |

The learned decoder's four false acceptances were **four of eight competing-wall
controls**. It abstained on every missing, averaged-away and cropped control.
Three accepted observable scenes had observed-target precision below 0.98
(minimum 0.973684). Mean accepted-scene position error was 0.310958 mm, which
does not excuse accepting an ambiguous wall. The first-hit comparison is a
boundary-only proxy; these results do not mean the full v5 patient counter would
accept those 38 negative cases, because its preceding/following checks are absent.

**No model promotion, new patient count, application setting change or production
history import occurred. The requested patient completion remains 15/98.**

An independent integrity audit checked the frozen source and imports, every
geometry split, all NPZ file/array fingerprints, all 24,576 sampled target labels,
all 15 epoch records, validation-only best-checkpoint selection and complete
64-scene accounting. Reaggregation reproduced the failed engineering gate. The
initial audit did not numerically remeasure unsaved inference score/point arrays;
a separate frozen-checkout numerical rerun is pending at this report checkpoint.

Protected evidence SHA-256 values:

- Checkpoint: `c501d7291f81eb2a2d08e3bb750ba38b4fc14480d7dd464e55feb8b43228315c`
- Evaluation file: `39606123338c54f0deffbf410f1741c923d5defd83bc7ffc17097665b3560617`
- Provenance file: `4f233efb66ed38e6777cc050aff38de980e12cd1e070f8abafdfb34944f1c4f8`
- Initial independent audit: `65995a9b2b9ed6466d6dc619e9442d20bf736c48933ea58191eeecd3623ba030`

The next engineering investigation is a reproduced alternative-search gap:
partial septa change candidate depth ordinals, so approximate multi-start search
can miss a second complete observed wall. Separately, the generator's observed
target definition is under review at blurred wall edges. Any correction must
receive a new version and preserve this failed experiment. These inspected test
scenes have now informed development and cannot become a new independent test
set by rerunning them after a fix.
