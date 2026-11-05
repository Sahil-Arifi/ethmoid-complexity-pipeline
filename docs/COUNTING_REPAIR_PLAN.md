# Plan: diagnose and correct cell partition failures

Research reviewed 2026-11-04. This is an implementation plan, not a validated replacement algorithm or a promise of 8/8. Tasks are tracked in [the task checklist](COUNTING_REPAIR_TASKS.md).

The subsequent [native diagnosis](COUNTING_NATIVE_DIAGNOSIS.md) reproduced four
current baseline runs exactly and found two conditional raw-basin collisions.
The procedural audit's representability result therefore does not justify a
merge-only native repair. A finer target-free proposal diagnostic removed those
collisions, but its fragmentation, grouping and acceptance remain unresolved.
Treat proposal generation, separator evidence and localization/acceptance as
separate dependencies before task 8; the current clinical result remains 0/8.

## Starting evidence and scope

The current engineering branch is `f0a6749`, with numerical replay source `8689516` and engine `native-bone-watershed-0.2.3`. The eight-study development comparison fell from 1/8 to 0/8 exact bilateral agreement. Repeat artifacts passed independent technical verification. The existing 96-scene geometry screen retains eight failures; 36 additional counterexamples pass. Current source already exposes markers, raw basins, merged labels, interface evidence and proposal rejection traces. Reuse those traces rather than adding another parallel pipeline.

The demonstrated failures are extra same-side groups near a perforated sheet, exterior-neck bone acting as separator evidence, and open folds acting as separators. The withholding study has insufficient assignment within its automatic search region. Existing evidence does not establish missing acquisition coverage or a remaining arithmetic defect. Software test success does not settle these geometry failures.

Earlier connectivity, video-guided instance and synthetic wall-learning experiments failed their gates. Public data/model limitations are recorded in [the source review](RESEARCH_DATA_SOURCES.md). Do not restart those experiments under a new name.

## Research findings and tool choices

| Resource | Verified capability | Decision and limit |
| --- | --- | --- |
| Existing NumPy/SciPy/scikit-image stack | Stage arrays, connected components, contingency tables, variation of information and physical shape measurements | First choice for the causal audit. Reuse the existing scorer and spatial oracles; no dependency needed just to compare partitions. [Metrics documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html). |
| [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) | Generates property-based cases and reduces counterexamples | Optional isolated test dependency for shrinking and representation invariants. Generated tests cannot supply anatomical truth. |
| [ELF](https://github.com/constantinpape/elf) / [lifted multicut API](https://constantinpape.github.io/elf/elf/segmentation/lifted_multicut.html) | Graph partitioning with local and longer-range costs | Diagnostic alternative to the current greedy merge, only after graph representability is established. Current API uses bioimage-cpp; its `time_limit` argument is ignored. Use an external process timeout and memory cap. No claim of globally optimal output. |
| [Mutex Watershed](https://github.com/sciai-lab/mutex-watershed) | Partitions attractive/repulsive affinities | Research comparator, not a second implementation to build immediately. Wrong boundary evidence remains wrong input to this solver. |
| [ITK Hessian objectness](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1HessianToObjectnessMeasureImageFilter.html) | `ObjectDimension=2` enhances plate-like structures in 3D | Conditional source of observed thin-wall evidence. It is not a cell classifier. This is supported by the sinus-specific [bone-enhancement paper](https://cim.mcgill.ca/~shape/publications/miccai05b.pdf), not evidence of individual-cell counting accuracy. |
| [Panoptica](https://github.com/BrainLesion/panoptica) | Matches and evaluates 2D/3D instances | Optional independent scorer, after verifying the existing scorer against small hand-computed cases. Its instance approximation is not a source of reference labels. |
| [3D Slicer](https://slicer.readthedocs.io/en/5.8/user_guide/modules/segmenteditor.html) | Native image/segmentation review and editing | Prepare exact linked views and source-bound reference instances for clinician adjudication. Thin-wall resolution must be retained. |
| [CT-SCOPE / Craneal_CT](https://github.com/BYO-UPM/Craneal_CT), [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/) | Sinus-region bone segmentation resources | Already reviewed, not a newly discovered cell counter. Bone masks are not cell instances; access/license discrepancies and training-set overlap remain adoption checks. No weights or data adopted here. |
| [micro-sam model registry](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/util.py) | Lists medical-imaging encoder and instance-decoder weights with download URLs and hashes | A verified capability update worth a label/rights/3D-consistency audit if proposal representation is the bottleneck. No ethmoid-cell performance was verified. Do not describe the current registry as microscopy-only, or adopt its predictions as references. |
| [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) / [MONAI Label](https://github.com/Project-MONAI/MONAILabel) | Prompted 3D segmentation / annotation and training infrastructure | Potential assistance for reference preparation. Neither establishes automatic ethmoid-cell instance counting in this review. Prompted and unprompted results must be reported separately. |

The plugin catalog search returned general research/media/analytics tools, including Scite. No dedicated DICOM ethmoid-cell counter was identified in those results. Scite may assist literature review; it supplies neither native boundaries nor a counting algorithm. The installed planning, debugging, council and Graphify skills improve investigation and review, not anatomical evidence. No new plugin/provider is needed for the first diagnostic tasks.

## Architecture decision

Do not select whole-sheet reconstruction or a graph solver as the presumed fix. First distinguish four causes: an inadequate initial partition, incorrect observed separator evidence, incorrect optimization, and localization/acceptance loss. Use controlled interventions on procedural truth to identify which cause is active. These interventions are diagnostics only and must never be reported as automatic predictions.

A merging-only graph cannot recover a cell boundary that lies inside one of its nodes. Conversely, a solver with perfect consistency cannot correct false input evidence that says a neck is a wall. These two checks must precede an expensive model or solver replacement.

## Ordered execution and decisions

1. **Freeze the audit inputs and test contract.** Hash existing procedural sources, renderer parameters, arrays, spatial-oracle definitions, source/runtime and old outputs. Preserve the 96+36 controls and the 864 representation checks. Keep the eight clinical studies as an already exposed development set, not an independent holdout. Define source-only external working storage; retain all failures.

2. **Build an error ledger from existing traces.** For each declared procedural group, record foreground coverage, basin overlap, merges/splits, interface support provenance and rejection reason. Verify results with hand-computed tiny partitions. Use existing spatial criteria: exact foreground coverage, declared counts and core relationships in the 96 scenes; whole-group correspondence in the 36 controls. Do not invent an exact divider position inside an aperture. Report excluded/missed foreground separately so background handling cannot hide failures.

3. **Test representability before changing merging.** Check whether existing basins mix incompatible declared cores or omit required foreground. If so, a merge-only repair is ruled out for that case. A synthetic-only intervention using deterministic group-contained seeds tests whether proposal generation is responsible. A successful oracle intervention identifies the stage to repair; it is not an algorithm success.

4. **Separate evidence from optimization.** Where the partition is representable, hold the basins fixed and compare current merge decisions with diagnostic group-derived edge costs. On reduced graphs of at most eight nodes, enumerate partitions as an exact reference. Compare a candidate ELF solution against that reference before trusting it on larger graphs. Oracle-cost success with observed-cost failure directs work to boundary evidence; oracle-cost failure directs work to graph construction, sign conventions, constraints or optimization. Multiple independent failures may require separate fixes.

5. **Make one cause-specific numerical change.** Select the change only from tasks 2-4. If evidence is deficient, test observed plate orientation, thickness, coherent patch identity and side attribution, beginning with existing features and adding ITK sheetness only if justified. The perforated-wall case must merge same-side fragments while retaining opposite-side cores. The same rule must reject exterior neck and open-fold evidence while retaining the passing cantilever/foreign-wall controls. If initial basins cross cores, repair proposal granularity first. If only optimization fails, replace that component without changing evidence. Do not fit by case name, orientation, target count or hidden generator wall masks.

6. **Investigate coverage independently.** On procedural ROIs, hold final partitions fixed and vary search extent, membership and exclusion masks one at a time. Account for every rejected voxel and proposal. Then use preserved source-bound native traces to locate analogous losses; anatomical correctness of those exclusions requires references. A higher assigned fraction alone is not improvement: background or excluded structures must not be admitted to raise it. Keep the coverage and acceptance thresholds fixed unless a separately justified new specification is reviewed.

7. **Run an independent synthetic gate.** Freeze at least 48 new geometries from at least eight structural families before candidate execution, separate by family from development scenes. Include curved/branching sheets, multiple apertures, nearby unrelated walls, folds, necks, truncated acquisition and information-losing sampling. Publish the renderer assumptions and expected relationships before inference. Ambiguous/unobservable examples require a predefined uncertainty outcome, not a forced count. A failed holdout is retained; after using it for development, replace the independent test partition before another generalization claim. Require no regressions in the 96+36 controls and 864 representation checks, expected resolution of all eight known failures, opacity checks, source immutability and bounded execution. Use a 120-second/2-GiB cap for tiny synthetic processes; predeclare representative-volume limits from the preserved runtime measurements before that phase.

8. **Freeze one candidate for clinical development comparison.** Only after task 7 passes, create a new component version and source-bound plan; run all eight studies twice in fresh processes with fixed original localizers, targets outside inference and no per-case settings. Compare native partitions, foreground coverage, complete registries, exact bilateral counts and signed side errors against both preserved baselines. Any input/localizer change is a separate intervention, never silently included in the separator comparison. A localizer-specific candidate gets its own frozen comparison. 8/8 is the requested development outcome, not independent clinical validation. Individual reference instances are needed to rule out compensating misses/extras even if totals match.

## Failure-to-test map

| Failure | Diagnostic contrast | Required outcome before promotion |
| --- | --- | --- |
| Perforated sheet produces three groups instead of two | Fixed basins; group-derived edges versus current wall evidence | Same-side cores join; opposite-side cores remain separate under the original spatial oracle. |
| Exterior neck creates a split | Observed plate support versus procedural internal/exterior attribution | One declared group retained without erasing true thin-septum controls. |
| Open fold creates a split | Coherent local two-sidedness versus actual whole-group relationships | An open fold stays one group; supported perforated walls retain their required split. |
| Assignment withholding | Fixed partitions with independently controlled membership, exclusions and search bounds | Recover justified foreground without admitting known excluded foreground or relaxing coverage gates. |

## What the plan can and cannot establish

The 48-geometry/eight-family minimum and tiny-process resource caps are operational planning choices, not statistical guarantees. The exhaustive reference proves optimality only for its explicitly defined small objective and constraints. Reduced examples must preserve the tested property. Check oracle feasibility, cost conventions and graph construction before attributing an oracle-cost failure to optimization. Unassigned basins and unknown group relations remain unknown; exclude them from hard oracle edges instead of fabricating labels. Existing core relations, rather than a new arbitrary aperture divider, determine diagnostic feasibility.

Plans are stored in `docs/` because the repository's source/documentation guard does not permit a top-level `tasks/` directory. The guard is unchanged.

The plan replaces uncontrolled changes with causal tests and explicit rejection conditions. It does not establish in advance that a whole-wall model, ITK filter, learned affinity model or graph solver will work. Some image observations can admit different interpretations; no code can recover unavailable construction provenance. An observable geometric convention and clinical references must resolve that ambiguity. The existing eight synthetic failures remain engineering obligations, not clinician questions.

Rachel's later review should identify source-bound native cell instances and disputed anatomical membership, with clear inclusion/exclusion and communication conventions. Do not repeat resolved questions about revised recounts or repeated PDF levels. Work on synthetic causality can proceed without that review; claims of clinical correctness cannot.

Partial reference relations have three states: same cell, different cells, and unknown. Unreviewed pairs remain unknown. Never derive missing edge labels from a total count or complete an instance mask from an unverified pointer. Record the gap between oracle-supported and image-feature-supported performance; success with extra oracle information does not establish that production features can supply it.

## Review and handoff

Three real advisors on the existing model connection reviewed geometry/solver assumptions, public models and stage isolation. This adapts the council's five-advisor default to four concurrent slots. Anonymous peer review challenged the assumption that whole-sheet extraction is already the answer. The adopted sequence puts representability and evidence tests first. Existing Graphify output was queried but is stale; current source was checked for the actual call path. No patient inputs entered a graph or online query.

This phase changes planning documents only. No candidate was installed, trained, run on patient data or promoted. Every implementation task must produce its own evidence artifact and preserve preceding results.
