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
initial audit did not numerically remeasure unsaved inference score/point arrays.
A second frozen-checkout audit subsequently reproduced validation/test inference
and all 64 scene/control results exactly: 782 compared floating values, maximum
difference 0.0. It saved 66 hash-bound evidence files and independently remeasured
selected-point targets and physical errors. It took 24.797 seconds and preserved
every original release file.

Protected evidence SHA-256 values:

- Checkpoint: `c501d7291f81eb2a2d08e3bb750ba38b4fc14480d7dd464e55feb8b43228315c`
- Evaluation file: `39606123338c54f0deffbf410f1741c923d5defd83bc7ffc17097665b3560617`
- Provenance file: `4f233efb66ed38e6777cc050aff38de980e12cd1e070f8abafdfb34944f1c4f8`
- Initial independent audit: `65995a9b2b9ed6466d6dc619e9442d20bf736c48933ea58191eeecd3623ba030`
- Numerical audit: `4424c3822a1c51f8967b7ffec1a1db9ef105b569d6092f6600732abf8ad5d314`

The next engineering investigation is a reproduced alternative-search gap:
partial septa change candidate depth ordinals, so approximate multi-start search
can miss a second complete observed wall. Separately, the generator's observed
target definition is under review at blurred wall edges. Any correction must
receive a new version and preserve this failed experiment. These inspected test
scenes have now informed development and cannot become a new independent test
set by rerunning them after a fix.

## Experiment 002: completed, not promoted

The first experiment exposed two distinct limitations. Changing candidate depth
order could hide a complete competing wall from approximate search. Separately,
21 selected voxels in three noisy scenes crossed the observed HU threshold only
because of added noise. Those voxels remain negative examples; their labels were
not changed to improve the first experiment's score.

The second version makes the following bounded changes:

- A score-independent interval-domain solver computes least and greatest
  feasible candidate sheets. If enough columns differ, they witness a competing
  sheet. If fewer differ, no pair of feasible sheets can differ on that many
  columns under the declared slope/voxel-interval assumptions. A finite-work
  exhaustion produces an explicit resource abstention. Neural energy ranking is
  still approximate and does not certify anatomical identity.
- The classifier now combines global context, features at the exact candidate
  center and that source-center HU value. Training reserves part of its negative
  sample budget for physical-wall candidates lacking the declared observable
  support. Validation/test sampling stays uniformly capped within each class;
  no targets are relabeled and no analytic mask enters the neural input.
- A separately reproduced noise-free blur defect receives a generator-version
  correction. A wall's measured signal may extend outside its original material
  footprint. Targets now require threshold-level **noiseless** signal attributed
  to that wall, dominance over other materials, and loss of threshold support
  when its contribution is removed. Noise-only hits stay negative.
- Every scene now retains hashed candidate scores, source coordinates, targets,
  selected observed points and the baseline's separately identified fitted
  surface. This enables numerical audits without relying only on summary rows.

The renderer, architecture, solver, benchmark and experiment identity are
versioned. Generator versioning creates new deterministic draws; the planned
eight-geometries-per-family split is 40/16/8 with smooth bowls as the test family.
These are **development experiments informed by earlier failures**, not a new
independent clinical test or a directly paired comparison with experiment 001.
Epochs, batch size, physical patch size and the original scene acceptance
criteria remain fixed before this run.

Verification before the second source freeze: **423 CPU checks passed**, including
the existing 220 counting regressions and independent linear-programming oracles
for the new feasibility solver. **24 real PyTorch integrations passed**, including
CPU/CUDA optimization, exact center alignment for every supported odd patch size,
checkpoint integrity and all 64 held-out equal-histogram pair rankings. That
functional task's fixed-threshold classification was 126/128; it is not a perfect
calibration or clinical result. One Windows symlink-privilege test was skipped.

A dedicated negative control records the unavoidable distinction between two
separately observed walls and two walls whose sampled HU joins into one bone run.
The decoder must not invent a second observed face in the latter. If it accepts
that apparent single wall, the independent benchmark still records a false
acceptance and fails its gate. This limitation remains visible.

Frozen source: [`d64111d`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/d64111d47e0382532550071531d1cfe702b732c1).
The completed run used 64 geometries, with disjoint 40/16/8 geometry and 5/2/1
family splits. Smooth bowls formed the test family. Sampled candidates numbered
15,328 training, 6,144 validation and 3,072 test. All targets remained procedural;
no patient images or expert annotations entered this experiment.

Fifteen epochs ran on the same local GPU. Validation binary cross-entropy
decreased from 0.703422 to 0.035166; epoch 12 selected the checkpoint.
Training/checkpointing took **16.172 seconds**, with peak PyTorch-allocated GPU
memory **129,986,048 bytes**. Complete generation, training and evaluation took
**47.797 seconds**. These timings describe this small synthetic experiment only.

Class-capped test precision was **98.13%**, recall **95.83%** and binary
cross-entropy **0.081597** (1,472 true positives, 28 false positives, 1,508 true
negatives and 64 false negatives). These are candidate-classification results,
not cell-count accuracy or calibrated clinical probabilities.

All 64 scenes were retained: 26 observable and 38 unobservable, with no extraction
failures. The engineering criteria remained unchanged.

| Boundary method | Accepted / 64 | Abstained | Accepted observable scenes | False acceptances | Fixed engineering gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Image-only interval-domain decoder | 28 | 36 | 22 | 6 | Failed |
| Learned scores with interval-domain decoder | 26 | 38 | 22 | 4 | Failed |
| First-hit/gradient/quadratic boundary proxy | 64 | 0 | 26 | 38 | Failed |

The learned decoder accepted four ambiguous controls. Four accepted observable
scenes fell below the 0.98 observed-target precision requirement; the minimum
was **0.961818**. Mean accepted-scene position error was 0.405746 mm. The proxy
remains a boundary-only comparison, not the complete patient counter. Generator
versioning changed the draws and test family, so these numbers are not a paired
improvement estimate against experiment 001.

Independent CPU audit verified all **24,544 sampled labels**, exact seeded
sampling, geometry/family separation, all 15 epochs and validation-only checkpoint
selection. It re-decoded all 64 saved proofs and independently remeasured 1,262
floating values with maximum difference **0.0**. Reconstructing all 3,072 test
probabilities from scene evidence reproduced the confusion counts exactly and
loss/Brier metrics within 1e-9. This confirms the recorded failed result; it does
not establish anatomical validity.

Protected evidence SHA-256 values:

- Checkpoint: `fcb7feb80a1ea9c39dd005ad6931d1f5feb19dd2da83c35e72f9126cb93149d4`
- Evaluation file: `84812f0057600343cc1820627be5e7b2c343642cf807234bd9868759d18fb6fd`
- Provenance file: `d51d66649fe8a1cd2d28c38429bf9b3c0578c6452d6c0f9b6113833bdc69e722`
- Independent scene audit: `cf65f03c7415cfc4cc3ff944ba4e98d099c62d4134a457f7146a83512b76f30f`

The four ambiguous controls are only partly merged: separate observed runs
remain in 37–85% of their sampled columns. Transitions between merged and separate
runs make each mapped rival surface violate the decoder's full-grid slope
assumptions. The solver correctly certifies its declared geometry model, but
that model cannot certify anatomical uniqueness. Its variable-column counts
were 147, 2, 63 and 45 against required 149, 169, 136 and 143 respectively.
Changing those cutoffs to fit these cases would not resolve the missing evidence.

A separate acceptance weakness is that a mean confidence can conceal individual
unsupported points. In the worst accepted thick scene, all 21 target-negative
points scored below 0.0015; no target-positive alternative existed in their
columns. The next bounded check requires support for each selected point and
reports abstention when it is absent. It must not drop uncertain points and
present the remainder as a complete wall. Reusing these inspected scenes is a
development regression, not a fresh independent evaluation.

**Neither trained experiment is promoted. No patient count or application setting
changed. The audited patient result remains 15/98, and the 98/98 target is unmet.**

## Per-point support correction

The separate `all_selected_points_at_least_0_5_v1` policy requires every selected
native observed point to meet the existing fixed 0.5 classification threshold.
The interval-domain geometry solver remains version 2. Geometry ambiguity checks
still run before score acceptance. A failed point withholds the entire proposal
as `WALL_POINT_UNSUPPORTED`, including selected/supported/unsupported counts and
minimum/mean score. No point is deleted, replaced, filled or reclassified to pass.
Lowering the optional historical mean threshold cannot bypass the point floor.

Benchmark version 3 preserves these diagnostics in failures and completed
proposals. Future training manifests separately record the support policy under
experiment version 3; no third training run has occurred. Classifier weights,
renderer and fixed benchmark precision/error/negative-control criteria remain
unchanged. A classifier score is still not a calibrated clinical probability.

**433 CPU checks passed**, comprising 213 synthetic research checks and 220
existing counting regressions. New controls cover a high mean concealing one
unsupported point, the exact threshold, immutable inputs, no partial output or
substitution, and unchanged image-only/geometry behavior. One Windows
directory-symlink-privilege test remains skipped; the unchanged real training
implementation previously passed 24 CPU/GPU integration checks.

The completed replay at frozen source
[`01f5e21`](https://github.com/Sahil-Arifi/ethmoid-complexity-pipeline/commit/01f5e21822542da0042f85089c59fe6d93e076d4)
reused all 64 source-bound experiment-002 scene scores. The learned decoder
accepted **2/64 scenes**, abstained on 62, and accepted **zero negative controls**.
Both accepted scenes had identical points to their previous acceptances. Minimum
observed-target precision was 0.996667 and mean position error 0.267494 mm.
The fixed engineering gate passes on these reused scenes, but the very low
coverage and development reuse prevent treating this as independent accuracy
evidence or a model-promotion result.

Twenty-four previous acceptances became abstentions; no new scene was accepted.
Reasons were 26 point-support failures, 29 incoherent surfaces, four ambiguous
surfaces, two insufficient-support outcomes and one unsupported hole. Every
image-only and boundary-proxy outcome and point artifact remained unchanged.
All 91 original release files were rehashed unchanged, and 128 new scene proof
files retain the replay evidence. Runtime was 9.922 seconds on CPU, with no model
inference, optimization or patient processing.

Replay evaluation SHA-256:
`6e4e72685b2433136479da656d6f44df70aa788fa1f4c2e2f02484389c7255a5`.
The separate classification audit of experiment 002 has SHA-256
`211ae18f443c89817a01ce40860c338867c994fc0cb4abc3897ffb2849554188`.

The 15/98 patient result remains unchanged. Meeting a small synthetic gate with
two accepted scenes does not demonstrate reliable patient boundaries, and it
cannot supply missing expert region masks or individual-cell reference counts.

## Native-boundary diagnostic pilot: six retained failures

A bounded diagnostic used six preselected source-verified, native-axis-aligned
inputs from distinct studies. Selection took the first three eligible studies
per prior wall-support/curved-fit message in frozen manifest order, before
scoring. It did not select new inputs after seeing results or attempt another
98-volume count evaluation.

Frozen v5 localization reproduced every original failure and diagnostic record
exactly. Its exception frame supplied the actual HU array, interior seed,
physical spacing and anatomical search limit. A synthetic control checked this
capture mechanism. No seed, coordinate convention or boundary was guessed.
Existing cached priors were hash-verified; no new localizer ran. Canonicalization
used native axis permutations/flips only. Every observed candidate mapped back
to an integer original voxel with exactly matching HU. This verifies current
converted-volume bindings and candidate correspondence, not another raw-DICOM
pixel-decode pass.

The trained checkpoint scored **7,467 observed candidates** in serial batches of
64 on the offline GPU. Both image-only and learned decoders withheld all six:

| Outcome | Inputs |
| --- | ---: |
| Insufficient observed support | 1 |
| Disconnected observed support | 4 |
| Incoherent surface | 1 |
| Accepted proposal | 0 |
| Technical error | 0 |

All model patches were within the acquired source extent. Some real patches
contained no seed context; per-input seed-present patch fractions ranged from
0 to 0.416. These are descriptive domain-transfer findings, not ground-truth
anatomical measurements. The geometry vetoes precede neural score ranking, so
changing a classification threshold cannot resolve these six outcomes.

The initial pilot took **149.295 seconds**. Root review found that it scored
normalized float32 patches directly, whereas frozen training/evaluation first
stored them as float16 and then computed in float32. That initial release remains
immutable and is explicitly recorded as having a preprocessing limitation.

A separate corrected scoring replay used the exact frozen `patches_for` helper
on the same six sources/seeds/candidates. Every patch matched an independently
computed float16 round trip, and regenerating the original float32 patch hashes
confirmed identical sampling. It made no new selection, localization attempt,
optimization or parameter adjustment. Runtime was **16.716 seconds**. Of 7,467
probabilities, 7,180 changed and 287 were identical; maximum absolute difference
was 0.000208855 and mean difference 0.000001926. There were **zero 0.5-threshold
crossings**; 807 scores reached that threshold in both runs. These scores are
not calibrated clinical probabilities. All six geometry outcomes stayed the same.

The initial independent CPU audit reconstructed all candidates and float32
patches, re-decoded both methods for all six inputs, and verified all 104 bound
original files and 20 production database tables unchanged. It explicitly did
not independently reproduce neural inference or attest exact training
preprocessing for that first pilot.

An independent audit of the corrected release then reconstructed all 7,467
float16 training tensors and their float32 computation inputs from source HU and
saved seeds, checked the frozen helper against the original training code, and
repeated 18 CPU decoder checks. All 123 bound input/runtime files and 20 database
tables remained unchanged. It independently recomputed the score-comparison
arithmetic, but did not rerun the neural network, anatomical localizer or raw
DICOM decoder. These are integrity and reproducibility checks, not clinical
validation.

Protected evidence SHA-256 values:

- Initial pilot: `9e1c9e02559f665e86ddb94309c5e5ff5c957c23a2bcb740394b35bfdb40c9f8`
- Initial scoped audit: `02393c0bfe6ed304e136bfdddd33f9b1dd47a9c165ad316246a49298e96efde1`
- Corrected scoring replay: `f5420870959837ec25e8a920ed3e0bd899fe02061d47af47be19dea360901c04`
- Corrected replay audit: `eaf7aae8ab27c4e1e4573e98afcccf79cd36861316c58211ac847f5b79f81077`

**No posterior surface, region mask or cell count was produced by the pilot.**
There is no evidence here for integrating this component into patient counting
or promoting it as an improved anatomical model. The current component returns
only supported observed points; the existing localizer also needs a complete
posterior boundary. Further extrapolation would introduce an unvalidated
anatomical assumption. The bounded pilot is complete; it does not justify
repeated threshold changes or a 98/98 claim. Verified target-region supervision
and separate independent cell references remain the next substantive dependency.
