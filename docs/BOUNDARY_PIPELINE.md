# Native boundary and instance research CLI

This separate research path learns cell membership and separator probabilities
on the native CT grid, then partitions foreground with PanSeg distance-transform
watershed and ELF GASP average linkage. It produces native instance masks,
registries, provisional counts and source-bound evaluation. It does not replace
the app, revive a closed experiment or establish anatomical accuracy.

## Conditional membership experiment

The latest annotation-only experiment retained the fixed model, training budget
and v3 partition settings. Source review froze 102 new membership targets and
24 separate QC voxels before predictions were inspected. The new labels cover
22 material voxels, 72 clear cavity-interior voxels and eight exterior-air
negatives. All previous labels and QC files were preserved. Unknown margins
remain unknown; no complete cell or separator was inferred.

The fresh 2,000-step run took 200.375 seconds and peaked at 336,035,840 allocated
GPU bytes. Two fresh-process predictions matched in probabilities, native masks,
registries and counts. Comparison used the previous frozen probabilities decoded
with the same v3 adapter.

| Conditional check | Previous v3 control | New model |
| --- | ---: | ---: |
| Original negative QC false positives | 0/896 | 186/896 |
| New material QC false positives | 3/4 | 0/4 |
| New interior QC retained | 12/12 | 12/12 |
| New exterior-air QC false positives | 8/8 | 8/8 |
| Original core voxels retained | 324/324 | 324/324 |
| Original material exclusions violated | 0/1,692 | 0/1,692 |
| Declared cell pairs separated | 0/3 | 0/3 |

The model also failed all eight exterior-air training negatives under tiled
inference. This candidate failed the frozen preservation gate and was not
promoted. A smaller foreground mask did not establish improvement. Counts remain
withheld for crop contact and midline crossing. Instance scoring reports
`NOT_SCORED_NO_COMPLETE_REGION`; the release contains no complete native
evaluation region. These same-patient checks do not establish patient transfer
or eight-study accuracy. Protected sampling and context audits investigate the
remaining fit failure before another training change.

## Commands and artifacts

Use the existing CUDA counting Python with `PYTHONPATH=src`:

```powershell
& $CountingPython scripts/boundary_cell_pipeline.py readiness --manifest $Release --output $FreshReadiness
& $CountingPython scripts/boundary_cell_pipeline.py export --manifest $Release --output $FreshReview
& $CountingPython scripts/boundary_cell_pipeline.py train --manifest $TrainingRequest --output $FreshTraining
& $CountingPython scripts/boundary_cell_pipeline.py infer --manifest $SourceRequest --output $FreshPrediction
& $CountingPython scripts/boundary_cell_pipeline.py evaluate --manifest $Comparison --output $FreshEvaluation
& $CountingPython scripts/boundary_cell_pipeline.py diagnose --manifest $LegacyDiagnostic --output $FreshDiagnosis
```

All manifest paths, input files and fresh output directories must be absolute
and outside Git workspaces. Each output directory receives a private ACL for
the current Windows account and SYSTEM before data is written (0700 on POSIX).
Existing outputs are never overwritten. CLI errors return fixed codes and a
`failure.local.json` when a fresh protected run can be created. Raw exceptions,
source headers and private paths are not printed.

The [annotation contract](boundary-data-format.md) defines signed foreground,
separator and instance labels, evaluation coverage, hashes, provenance and
patient groups. Unknown labels contribute zero loss. Separators are explicit
annotations; instance-core edges do not supply them. Automatic proposals never
become expert references through export or import.

Training requests contain `schema_version: 1`, `version: boundary-instance-v1`,
`dataset: {path, sha256}` and `purpose: native_development` or
`synthetic_engineering`. Only entirely synthetic releases may specify
`synthetic_steps` for a shorter engineering run.

Inference requests contain exactly `schema_version`, `version`, `case_id`,
`source`, `checkpoint`, `checkpoint_sha256` and
`partition_runtime: {path, sha256}`. `source` is the existing source hash,
canonical native crop bounds and physical midline record. Reference cores,
pair labels, annotations and counts are rejected as inference fields.

Prediction outputs include `probabilities.local.npz`, full-source native
`instances.local.nii.gz`, and `prediction.local.json`. The latter binds original
geometry, source/model/runtime/producer hashes, provisional registry, quality
flags and counts. Empty, crop-truncated or cross-midline candidates withhold
counts; candidate totals remain separately inspectable. A complete numerical
prediction does not establish full anatomical coverage.

## Fixed model and training

The `boundary-instance-v1` adapter initializes a new two-logit output layer on
the existing 8/16/32/64 U-Net and deterministic pooling. It uses the same two
HU-normalized channels. It does not load the failed affinity checkpoint.

Training uses native 64-cubed patches, batch size 2, FP32, independently masked
class-balanced binary cross-entropy, seeded sampling/flips/HU noise and AdamW
(learning rate 0.001, weight decay 0.0001). Initial native runs stop at 2,000
steps or one hour, including loading and validation under the CLI supervisor.
Each head requires positive and negative training supervision. With multiple
patient groups, the last sorted group is held out; every reconstruction and
patch from a patient must share its group. Each held-out head needs known
supervision. Validation runs every 100 steps and at the final requested step.
The lowest annotated validation loss selects the checkpoint, with earlier
ties retained. Counts never select weights. A single group is explicitly a
mechanism experiment and saves the last scheduled checkpoint without claiming
patient transfer.

Model revision `native-membership-separator-unet-2` corrects patch sampling.
Each selected label lies inside a uniformly sampled valid native window,
rather than always at its centre. Axes shorter than the patch use the same
high-side-only padding as inference before augmentation. Existing whole-patch
flips transform HU, targets and unknown masks together; augmented padding may
move. HU and both label arrays share the window; padded labels remain unknown. A separate seeded origin generator preserves the
existing anchor, flip and noise random sequence. The two sampling configuration
fields and helper source are bound into checkpoints. The backbone, loss,
normalization, inference tiling and partition settings are unchanged. Revision-1
checkpoints require the preserved source at commit `9e57405`; strict CPU reload
of both historical native checkpoints passed in a separate source checkout.

Inference reloads weights with `weights_only=True`, verifies the complete
configuration, model/geometry helper hashes and Python/Torch/CUDA runtime
identity, and averages overlapping native patches at stride 32. Reorientation
only permutes/flips axes; there is no CT resampling or native label interpolation.
CUDA uses a 12 GiB allocator cap, available-memory preflight and peak tracking.
The supervisor bounds inference to 1,020 seconds; patch prediction also checks
its 900-second deadline. Allocation failures produce explicit records.

## Isolated partition runtime

```powershell
& scripts/bootstrap_boundary_partition.ps1
```

This bootstraps a separate Python 3.12 CPU environment under the local tools
directory using the SHA-256-pinned official micromamba binary. Four public lock
files pin source/archive hashes and all 189 conda package builds; bootstrap
audits downloaded package hashes. It leaves the working CUDA environment
unchanged. The resulting `runtime-manifest.local.json` binds the executable,
adapter, PanSeg source tree, dependency metadata and locks. `-RefreshSeal`
preserves the previous seal when intentionally resealing a verified adapter.
Cached bootstrap and inference work offline; first bootstrap requires the
pinned public downloads. Partition workers explicitly deny network connections.

The CPU worker receives only hashed probabilities, affine and runtime identity.
Membership and watershed thresholds are 0.5. Watershed uses physical voxel pitch,
seed smoothing 1 voxel, weight smoothing 0 and no minimum-size merging. GASP
uses average linkage, merge bias 0.5, one thread and no minimum-size merging.
ELF's foreground mask excludes background from graph construction; disconnected
regions are relabeled separately. Components below 4 voxels are filtered after
partitioning and reported separately, without forcing a merge.

The partition preflight estimates `128 MiB + 384 bytes per crop voxel` against
a 2 GiB budget and available memory, before CUDA inference. The subprocess has
a 120-second timeout. This is a conservative estimate and preflight, not an
OS-enforced CPU memory cap. No resizing, alternate decoder or silent fallback
is used when limits or dependencies fail.

Average GASP is a measured baseline, not a hard separator constraint. A retained
real-library fixture creates two watershed regions across a one-voxel boundary
plane, then merges them when membership remains connected through that plane.
The corresponding excluded-membership plane yields two cells. Background
isolation and separator guarantees are different properties.

## Evaluation and gates

Comparison manifests contain `schema_version`, `version` and `entries`. Each
entry includes a neutral `case_id`, `source_sha256` and either two hashed
`predictions` from distinct processes, or an empty list. Optional `failure`
binds a fixed-code failed-inference record. Optional `reference_counts` may
provide known left/right/total values; absent or null targets stay unavailable.
Optional `sample` supplies a source-bound annotation record for instance scoring.
Keep every scheduled study, including failed, withheld and not-run entries.

Evaluation verifies mask geometry against the actual source and recomputes the
registry, physical laterality, quality flags and counts from the mask. Repeats
must agree on native arrays, geometry, crop, registry, flags, candidate counts
and reported count status. Frozen checkpoint/configuration/producer/runtime
identities must agree across the cohort. Repeat disagreements suppress scoring.
Partial references score only their declared completely annotated regions;
unknown instance/foreground targets are forbidden inside such a region.

Reports include coverage, voxel Dice and reference coverage, matched/missed/
extra instances, merges, splits, individual side errors, exact sides and exact
bilateral studies. Scheduled denominators remain explicit. `COMPLETE` means
evaluation executed; `validation_passes` separately records repeatability.
Reference quality and anatomical validity remain independent of either field.

For an entirely new synthetic engineering run:

```powershell
& $CountingPython scripts/validate_boundary_pipeline.py --output $FreshEngineeringRun --runtime $RuntimeManifest --steps 200
```

The driver generates two separate synthetic patient groups and new complete
chamber geometries, exports them, checks readiness, trains, reloads the frozen
checkpoint in two fresh inference processes per source, partitions and scores.
Its predetermined gates require exact repeat masks, two matched instances per
source at IoU at least 0.9 with no miss/extra/merge/split, and exact generated
bilateral counts. It preserves failed gates. Existing 132-case and untouched
48-case benchmarks are not used by this driver.

Before an eight-study native comparison, pass mechanism and resource checks,
freeze weights and partition settings, and obtain sufficient source-bound
annotations. The authoritative counting protocol covers unique cells over the
full volume, including opacified cells and specified variants. A cropped toy
experiment or correct totals alone cannot establish correct native recovery.

## Measured results

The complete generated run passed after 200 steps in 74.7 seconds, selecting
step 200 by held-out annotated loss (0.01654, compared with 0.03976 at step 100).
Peak PyTorch allocated memory was 336,035,840 bytes. Both generated sources had
two matched cells with IoU 1.0 and foreground Dice 1.0, no missed/extra/merged/
split instances, exact counts on all four sides, and identical native masks,
registries and counts across their two fresh inference processes. These are
simple chamber geometries from two generated groups, not anatomical references
or evidence of patient transfer. An initial export invocation produced correct
files but an incorrect nonzero CLI exit status; that mapping was fixed and a
regression test added before the completed training run.

The original saved affinity checkpoint was reproduced on its original crop.
Its native array matched the saved result exactly. Every supervised edge
prediction was correct, but deterministic positive-edge paths still joined all
three declared distinct pairs. These witnesses used alternative paths through
unknown predicted foreground, including long edges; they did not cross known
or predicted background. This diagnosis supports replacing that representation
without treating unknown space as anatomical evidence.

The initial conditional native release preserves the original sparse labels,
keeps all separator voxels unknown and declares no complete evaluation region.
It is `NOT_READY`: supported separator positives and negatives are missing.
The protected Slicer package retains those unknowns and an exact annotation
request. That release remains unchanged. A separate conditional mechanism
release and native experiment are described below. No new 8/8 result or native
count improvement is established.

The installed pinned Slicer core and nnInteractive extension were reused. A
local annotation-assistance trial on a generated source completed in 54.4
seconds and produced a 2,310-voxel automatic proposal. Peak PyTorch reserved
memory was 5,479,858,176 bytes; whole-process GPU memory was unavailable under
WDDM. The existing checkpoint digest and CC BY-NC-SA 4.0 license are retained
in the protected inventory/trial metadata. Its proposal remains separate from
training truth and expert annotations. No patient volume was used by the trial.

### First native mechanism experiment: failed

A separate source-bound release retained all 324 membership-positive core
voxels, 17 membership exclusions and seven partial instance IDs. Orthogonal
native CT review supported two conditional separator positives on a short
partial septum and two negatives in airspace interiors. All other separator
voxels stayed unknown, and the complete-evaluation mask stayed empty. These
four labels establish neither a complete separating surface nor expert cell
identity. The imported video-to-native correspondence remains provisional.
Three available independent advisors and anonymous peer review checked this
limited design; the council was reduced to fit four concurrent agent slots.

Readiness was `MECHANISM_ONLY`, with no separate annotated patient group.
The fixed model trained for 2,000 steps in 135.25 seconds, using a peak
336,035,840 bytes of PyTorch-allocated GPU memory. The selected checkpoint was
the last scheduled checkpoint, step 2,000. No counts or reference pairs selected
the checkpoint. The Slicer export/import round trip preserved every array,
including unknown labels and conditional provenance.

Two fresh inference processes produced identical probabilities, native arrays,
registries, candidate counts and quality flags. Independent source/affine,
checkpoint and full-volume placement checks passed. Both predictions assigned
every voxel of all seven reference cores to the same output instance: none of
the three declared distinct pairs separated. Membership was positive in 78.97%
of the crop; the largest instance occupied 75.58% of it and included extensive
surrounding tissue. Both runs produced 66 candidates, with counts withheld for
crop contact and midline crossing. This is a failed mechanism experiment, not
an eight-study count result.

All 17 membership exclusions and all known labels for both model heads were
classified correctly. This local fit did not constrain the surrounding unknown
volume. A predeclared CPU control held membership and partition settings fixed
and replaced the learned separator map with zero. It produced six regions
instead of 66, while still merging all three declared pairs. Thus the separator
head changed partitioning without recovering the required distinctions; it
cannot be credited with successful separation. The control did not select a
checkpoint or replacement configuration. Six small scorer self-checks passed;
diagnostic failures and the completed audit are retained locally.

Seven local nnInteractive proposals were also generated for review. They were
mutually disjoint and covered 323 of the 324 existing core voxels; three touched
the crop boundary. No proposal became a training target. The existing weight
license and artifact hashes remain recorded in protected local storage.

The native failure gate prevents an eight-study replay with this checkpoint.
The next annotation batch is a fixed enclosed-pair review, including reviewed
outside background, complete extents or explicit abstention, and separate
evidence for walls and open communications. Its coverage requirements are
independent of predicted counts. Sparse point fitting and repeatability do not
establish correct foreground extent, individual cell recovery or patient transfer.

### Background repair and patch-context diagnosis

Contiguous native CT and orthogonal review supported 1,664 additional
membership-negative voxels outside the cavities. All original known labels,
partial instance IDs and four separator labels were retained. The pair review
abstained from complete masks because of an open communication; no proposal
contour or closing plane became a training target. Another 896 background
voxels and two air-interior points were reserved for post-training QC, unknown
in every training field and at least three source voxels from supervision.
These are same-patient conditional observations, not independent cell truth.

With the original sampler and the same 2,000-step budget, the background
intervention reduced held-out output false positives from 828/896 to 117/896
(85.9%). Membership false positives fell from 828 to 120. All 324 core voxels,
17 original exclusions and both positive QC points were preserved. The largest
output instance shrank from 75.58% to 48.76% of the crop. Both fresh predictions
matched exactly, but all seven cores still shared one instance and none of the
three declared pairs separated. Counts remained withheld. Thirty expanded
training negatives also exceeded the membership threshold during tiled inference.

A five-point diagnostic then exposed a training/inference context mismatch.
For three erroneous training negatives, centred-patch probabilities were about
0.000001 to 0.000017; tiled probabilities were 0.5070 to 0.5084. Their training
patches contained 26.6–28.1% artificial padding, while contributing native
inference windows contained none. Recomputing those windows reproduced saved
tiled averages within 0.00000003. The probe demonstrates context dependence;
it does not separately identify padding, position and normalization effects.
It motivated the versioned valid-window sampler above, with the background
release and QC held byte-identical for the next controlled run.

The sampler revision passed 331 boundary/partition/evaluation and counting
regression checks, including its four new sampling tests and actual isolated
PanSeg workers. A fresh 200-step synthetic end-to-end run selected step 100 by
annotated validation loss, matched all four generated instances at IoU 1.0,
and produced identical fresh-process predictions for both generated studies.
Those generated results do not measure native accuracy.

### Controlled sampler run

The frozen `0c7c934` sampler then trained from new weights for 2,000 native
steps in 126.8 seconds, with 336,035,840 peak allocated GPU bytes. The release,
source, annotation archive and held-out QC were bound by hashes. Compared with
the background-supervised parent, annotation and QC bytes, protocol, patient
group, architecture, loss, optimizer and partition settings were unchanged.
Model settings changed only through the declared sampling fields, with the
corresponding model revision and source hashes. The single-patient run selected its last scheduled checkpoint; neither QC nor
reference counts selected weights. A strict CPU comparison validated this
transition and both fresh inference processes.

| Partial native check | Original | Background supervision | Valid-window sampler |
| --- | ---: | ---: | ---: |
| Held-out background mask false positives / 896 | 828 | 117 | 15 |
| Held-out background membership false positives / 896 | 828 | 120 | 18 |
| Conditional core voxels covered / 324 | 324 | 324 | 324 |
| Held-out airspace points covered / 2 | 2 | 2 | 2 |
| Declared pairs separated / 3 | 0 | 0 | 0 |

The final mask reduction is 98.2% against the original and 87.2% against the
background-supervised parent on these fixed regions. Membership errors on the
1,681 shared known background labels fell from 30 to zero; final-mask errors
fell from 23 to zero. All four separator labels fit in tiled inference. Masks,
probabilities, geometry, registries and counts matched across two fresh
processes. Root and independent scoring used the stored native outputs.

The whole conditional mechanism gate still failed. All seven partial cores
remain in one decoded instance, and the membership head still selects 45.1%
of the crop. Crop contact and midline flags keep counts withheld. These checks
cover two negative regions and two positive points from the same patient,
not full anatomical membership, opacified cells or complete cell instances.
There is no complete-instance IoU claim or new eight-study replay.

An additional padding restriction was investigated and rejected before native
training. Keeping padding fixed after every short-axis flip was not required:
aligned whole-patch transforms are valid augmentation. Two 200-step variants
and one fixed 2,000-step follow-up merged the second generated validation pair;
all failed artifacts and their candidate source patch remain protected. The
extra restriction also has no effect on this native crop, whose valid windows
are unpadded. The retained source exactly matches the earlier passing synthetic
run and its 331 regression checks; the rejected variants' 332 passing unit
checks do not override their failed instance-recovery gates.

The protected final evidence is in
`boundary-native-sampling-20261105T144558Z`, including the strict comparison,
source transition, root three-run scoring and independent same-plane figure.
Remaining work is anatomical membership beyond the sampled regions and
separator/communication recovery. More background fitting alone has not solved
those failures. Native source correspondence and unresolved counting boundaries
remain conditional review questions; the algorithm is not yet validated for
complete cell recovery.

### Join-stage diagnosis and graph isolation repair

The frozen native stage audit places pair 1/5 in different watershed fragments
(410/378) that GASP subsequently merges. Pairs 2/3 and 6/7 already occupy single
watershed fragments (418 and 377). All pairs remain connected after excluding
every membership voxel with separator probability at least 0.5. The native
paths use physical face neighbors; there is no long-edge or background jump.
Source review retains a partial septal sheet, an open route and an unresolved
mixed-density ridge rather than inventing complete separating surfaces.

The audit also exposed a separate graph correctness problem. Masking watershed
output can leave disconnected pieces with one superpixel ID. GASP then treats
them as one node, allowing disconnected regions to share graph costs before
the final connected-component split. An actual pinned-GASP fixture reproduces
the failure: changing only a remote disconnected region changes the local
split/join decision.

Partition adapter `native-boundary-panseg-average-2` assigns every six-connected
watershed piece its own graph node before GASP. Coverage, original watershed
boundaries, physical spacing, average linkage, merge bias and small-component
policy are preserved. The configuration records
`connected_watershed_superpixels=true`; diagnostics report fragment counts
before and after splitting. The isolated CPU environment and package locks are
unchanged; a separate v2 manifest seals the new adapter and private job path.
Historical source and runtime records remain available for old checkpoints.

The fix removes seven aliased native watershed IDs, increasing fragments from
551 to 647. On this frozen crop the graph edge costs and final 24 output masks
are unchanged. Two fresh official v2 workers reproduce both old NPY files byte
for byte. This repairs a general graph invariant without explaining away the
current anatomical merges. The expanded regression selection passed 333 tests,
including the remote-cost fixture, and fresh synthetic export/train/reload/
infer/partition/evaluate validation passed with exact repeated predictions.

The next bounded native intervention added only reviewed material observations:
two exact bone membership exclusions on the joining route and nine foreground
exclusions/positive separator samples on the visible partial septum. Old labels
and QC remain fixed; two additional bone points stay outside loss as adaptive
same-patient QC, including an off-path wall control. A single 2,000-step run
used the same learning configuration and no count/QC checkpoint selection.
Open and unresolved regions remain unknown.

### Native partial-wall follow-up: foreground improvement, repair gate failed

The frozen run completed 2,000 steps in 125.3 seconds with 336,035,840 peak
allocated GPU bytes. The last scheduled checkpoint was selected for this
single-patient mechanism experiment. The CUDA learning environment, sampler,
loss, seed and other learning settings match the parent. Only the reviewed
annotation delta and the isolated graph adapter changed; the decoder-only
control reproduced the parent native masks byte for byte.

| Fixed observation | Parent | Partial-wall follow-up |
| --- | ---: | ---: |
| Original background QC output errors / 896 | 15 | 0 |
| Original background QC membership-head errors / 896 | 18 | 0 |
| Core voxels covered / 324 | 324 | 324 |
| Positive QC points covered / 2 | 2 | 2 |
| Original known background errors / 1,681 | 0 | 0 |
| Added membership exclusion errors / 11 | 6 | 0 |
| Added partial-wall separator errors / 9 | 3 | 0 |
| Adaptive bone-control errors / 2 | 1 | 1 |
| Declared pairs separated / 3 | 0 | 0 |

The additional bone control on the joining route remains wrongly included:
membership probability fell from 0.560 to 0.545, above the fixed 0.5 threshold.
The off-path control remains excluded, although its probability rose from
0.00057 to 0.231. Both controls remain unknown in every training-loss field and
outside the evaluation-region mask. They were selected after the parent
failure, so they are adaptive development QC rather than independent
validation. The original 896 background voxels are also reused development
QC. Correct results on these samples do not establish full-volume membership.

The nine partial-wall observations fit, but they do not establish a closed
cell boundary. All seven conditional cores still share decoded instance 1.
The stage replay shows a narrower change: pair 2/3 now has distinct watershed
fragments (706/713), but GASP merges them. Their direct graph edge still has
mean affinity 0.597 over 46 contacts, favoring a merge under the fixed average
linkage. Pair 1/5 also merges after watershed; pair 6/7 still shares its initial
watershed fragment. The official CPU replay matches both saved native masks,
and no disconnected watershed aliases remain. Better initial proposals for
one pair therefore do not establish final cell recovery.
The membership head selects 417,032 crop voxels (44.5%), compared with 422,924
(45.1%) in the parent. After the unchanged small-component filter, the native
instance masks contain 416,746 and 422,823 nonzero voxels respectively. The
largest decoded instance still occupies 414,914 voxels (44.3% of the crop).
Crop contact and midline flags keep counts withheld. Broad connected
membership, incomplete separators and open-communication interpretation
remain unresolved; the graph correctness repair did not solve these errors.

Strict source, affine, annotation-delta, runtime and checkpoint checks passed.
Unknown voxels outside the eleven exclusions/nine separator additions were
unchanged. Masks, probabilities, geometry, registries and count status match
across two fresh inference processes. The source-only inference manifest
contains no cores, pairs, QC labels or reference counts. Independent native
scoring confirmed the result.

The predefined repair checks failed on the joining-route control, and the
whole mechanism gate still fails on all three pairs. The candidate is retained
as a failed repair with a measured foreground-QC improvement; it is not
promoted to application counting. No eight-study replay was run while this
mechanism gate remains unmet. Protected evidence is in
`boundary-native-wall-repair-20261105T164051Z`, including the frozen inputs,
Slicer review export, strict comparison and independent native review. Expert
review still needs to resolve conditional cell correspondence and boundaries
around communications; algorithmic membership and separation remain work too.

### Voxel-boundary projection repair

The next source review rejected a two-point material correction as a separation
repair. Excluding the two reviewed material voxels in a frozen diagnostic
removed three graph contacts but left all three pairs merged. Pair 1/5 used
another positive path, and pair 2/3 retained 45 contacts with a net attractive
edge. Three
additional lateral ridge observations did not establish closure of the thin
central band for pair 6/7. These observations remain proposals; no annotation
or training run was added from this rejected intervention.

The supervision audit also found all eleven known separator positives outside
the predicted foreground. Setting their separator probabilities to either
zero or one changes no effective decoder-boundary voxels. This is a direct
observability gap, not a loading or masked-loss defect; training can still
change other predictions indirectly. Unknown regions remain unknown.

A separate synthetic counterexample exposed a projection failure. With a
perfect one-voxel boundary plane, watershed produces two regions, but the
original scalar-to-affinity conversion produces one attractive graph edge
over all 256 contacts. Every contact samples the zero-boundary endpoint and
ignores the adjacent endpoint whose boundary probability is one. There is no
competing negative edge in this fixture, so the merge does not demonstrate
an average-linkage limitation. The earlier test attributed this failure to
averaging incorrectly.

The old conversion faithfully follows the [PanSeg wrapper](https://kreshuklab.github.io/panseg/latest/chapters/python_api/functionals/segmentation/).
Adapter `native-boundary-panseg-average-3` makes this pipeline's voxel-centered
separator contract explicit: each face affinity is
`1 - max(boundary_low, boundary_high)`. This gives a boundary voxel influence
on either incident face regardless of watershed ownership. No labels are
dilated or inferred. Background stays excluded, disconnected watershed pieces
retain separate graph nodes, and spacing, seed smoothing, thresholds, average
linkage, merge bias and component filtering are unchanged. A separate v3
runtime manifest seals the adapter; installed packages and locks are unchanged.
The previous adapter, source checkout and checkpoint records are preserved.

Twelve fixed-fragment ownership/reflection cases now retain a uniformly
repulsive edge and two regions; six full watershed/GASP cases retain both
interiors across all axes and reflections. Watershed tie ownership of the wall
voxels themselves need not reflect exactly. A genuine interrupted-wall control
has 64 repulsive and 192 attractive contacts and still merges under average
linkage. This correction does not introduce hard cannot-link constraints.

The change passed 355 checks across the existing numerical and application
test environments, including 18 actual partition tests. Initial collection
attempts used the numerical environment for application-dependent tests;
the missing FastAPI/ReportLab dependencies were resolved by running those
checks in the existing locked application environment. Its four SciPy skips
were covered in the numerical environment. Neither environment was modified.
A fresh 200-step synthetic export/train/reload/infer/partition/evaluate run
passed, recovering all four generated instances with exact repeated results.
Two fresh native CPU workers then partitioned the exact frozen D3 probability
bytes using v3, before loading any annotations or QC references. Repeated
masks and registries match exactly. All 324 core voxels remain covered and
unfragmented, all 1,692 exclusions stay clear, and original QC remains zero
errors on 896 negative voxels with both positive points covered. The adaptive
joining-route bone point remains wrongly included. All three declared pairs
still merge into instance 1; counts remain withheld. The partition changes
beyond label permutation, but foreground support remains 416,746 voxels.

This is a decoder-only causal replay, not a new model or training result.
The unchanged learning code, annotations and fixed predictions isolate the
projection repair. Because it does not pass the native separation gate, no
redundant native retraining or eight-study replay was run. The repaired
voxel-boundary contract is engineering progress; anatomical membership and
separator recovery still fail. Expert review must resolve the conditional
cell correspondence and boundaries around communications. The remaining
algorithm work also requires broader source-supported membership and usable
boundary supervision; sparse material fit alone does not supply it.

Protected evidence is in `boundary-native-separation-review-20261105T171519Z`,
including the supervision audit and strict decoder comparison. The independent
synthetic counterexample and source-contact review packages remain separate
and unchanged. Patient artifacts, proposed labels and model files stay outside
Git; application counting is unchanged.

## Verification

The preserved counting regression selection passed 472 tests. Existing wall
research controls passed 213 tests, with one Windows symlink-permission skip
and 24 integration tests deselected. The boundary/data/evaluation/partition and
medical-data guard selection passed 133 checks, including actual isolated
PanSeg workers and the supervised CLI timeout path.

The broad application selection passed 699 tests in a fresh Python 3.13
environment installed from the unchanged hash-locked application requirements;
61 numerical/host-capability checks were skipped and 74 integration checks
deselected. Numerical boundary and counting checks ran separately in the
existing CUDA counting runtime. The original application test environment was
missing four mandatory JPEG decoder packages; its 14 decoder/provenance
failures were reproduced, then all 82 affected tests passed in the complete
locked test environment before the broad successful run. The CUDA environment
was not reinstalled or upgraded.

Whitespace checks and the staged/all-tracked medical-data guards passed.
Only source, public dependency locks, tests and documentation enter Git.
Patient inputs, annotations, model weights, tool installations, review packages
and experiment results remain in protected local storage.
