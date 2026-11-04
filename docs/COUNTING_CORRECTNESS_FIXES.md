# Counting correctness fixes

Engine component `native-bone-watershed-0.2.3` corrects reproducible software defects. It uses the same global thresholds and the same research-v5 strategy family. Previous attempts remain unchanged. New counts and identifiers belong to the new component and source build; annotations from an earlier build require correspondence checks before reuse.

## Independent component seeds

Peak reconstruction previously used 26-neighbor connectivity across the whole distance field, while component identity, seed labeling and watershed use face connectivity. Adding a separate component with only diagonal contact could therefore remove a seed in an unchanged original component. A synthetic anisotropic example lost one accepted split even though its original HU values and distance field were identical.

The correction reconstructs peaks separately in each face-connected component. It retains the existing 26-neighbor peak footprint inside that component. Free-space and aerated-lumen fields are handled independently. Each cropped field masks other components to zero and includes a zero halo. This matters when component bounding boxes overlap. Components whose maximum height is below prominence have no peaks; the existing component fallback still handles them.

This is narrower than changing the global footprint to face connectivity. Tests distinguish those implementations and exercise opaque, aerated and mixed contents, physical axis flips/permutations, overlapping bounds and the fallback for very small peaks.

## Separator tie ordering

Two independent reproductions exposed ordering effects after watershed. Equal-sized interfaces could select different support evidence after an axis permutation. Separately, identical watershed basins with exchanged numeric IDs could absorb a neighboring basin differently when support scores tied. The latter changed three accepted voxels without changing the premerge partition.

Tied dominant axes now follow their physical directions. Equally supported merge edges follow the physical positions of their original markers. On positive RAS-aligned grids these retain the existing ordering; support thresholds and cannot-merge constraints are unchanged. Numeric union representatives remain internal identifiers. The direct private helper retains its legacy ordering when physical metadata is omitted; the counting entry point supplies the affine and markers.

## Distance and watershed execution

Further tests isolated an upstream traversal effect. Equivalent array representations could change distance values by `2.22e-16`, enough to resolve watershed ties differently. A consistent watershed order alone left differences. Free-space and air-lumen distance transforms now use the same physical axis/sign order as watershed. These operations transpose and flip arrays, use matching spacing and C-contiguous kernel inputs, then inverse-map their outputs. They do not interpolate, coarsen or move native voxels onto another grid.

Positive RAS-aligned execution remains unchanged. Some oblique frames can select a different traversal than the earlier implementation; this is part of the versioned correction. A fallback seed on a flat maximum can still move within its component under reindexing. The tested fallback-only components retain identical final partitions; complete diagnostic-marker invariance is not claimed.

## Input binding and stable identifiers

- Configuration dataclass instances now receive the same validation as dictionaries.
- Direct array calls reject non-finite or nonscalar midlines and nonhomogeneous affines with an expected error code. These are API robustness corrections; the reviewed assisted midline and NIfTI paths already constrain those inputs.
- Auxiliary exclusion and variant masks must match the complete source shape. Finite numeric masks use nonzero boolean semantics before exclusion or overlap operations; text, complex and non-finite masks are rejected.
- Source-bound landmark masks use an absolute affine tolerance of `1e-4`, with zero relative tolerance. A large physical origin can no longer make a shifted mask appear to match the source grid.
- When rounded physical centroids coincide, the smallest actual physical voxel-center tuple determines proposal order. Rounded coordinates precede exact coordinates, which break remaining ties on exceptionally fine grids. Ordinary distinct-centroid proposals retain their ordering. This changes identifiers for affected ties without changing their geometry or counts.
- Low-coverage warnings now use the configured coverage gate, matching the result-withholding decision.

## Verification and limits

The locked counting suite passed **388 tests**, with **12 optional Torch tests skipped**. After repairing a pre-existing CI collection issue and adding the omitted connectivity module to the numerical job, the expanded suite passed **417 tests**, with **16 optional Torch tests skipped**. The lightweight application job now skips that SciPy-dependent research module; the locked numerical job executes it. Application/API checks passed **63 tests**, including the real CPU worker integrations. Focused tests reproduced the old failures before the corrections. Independent synthetic checks covered 43 helper contracts and 48 reorientations of the component witness. The seed-only comparison retained exactly the same markers and partitions in the earlier 96-scene screen: both implementations passed 88/96. All 36 forms of two additional counterexamples also retained their original outputs.

A fresh, locked application environment confirmed that isolated decoder and worker subprocesses import this checkout. Its full non-integration run recorded 666 passes, 50 dependency or host skips, 74 deselected tests and one failure because the frontend had not been built locally. After the required production build, that API test passed separately. The existing shared virtual environment and its older editable checkout were preserved; results from that misbound environment are not used as validation of this source.

The required dependency audit also found five vulnerable frontend build/test packages. Compatible patches update Vite to `8.2.4`, Vitest to `4.1.12`, and their affected transitive locks. The locked install, production build, 70 frontend tests and all five synthetic browser workflows passed; the dependency audit reported zero vulnerabilities. No frontend application code changed.

The final physical-execution comparison preserved all 96 canonical scene arrays and all 36 additional canonical counterexamples. Its public witness passed all 48 signed permutations. All 192 archived X/Y-permutation and X-flip variants retained their canonical markers, watershed basins and accepted partitions, replacing 22 earlier final-partition differences. Another 864 variants covered unequal dimensions, anisotropic spacing, edge contact, oblique grids including 45-degree ties, and read-only inputs; all final partitions and distance bits matched. Small measured resource controls found comparable runtime and peak memory, but nontrivial orientations can allocate extra contiguous kernel inputs. These bounded checks do not establish whole-dataset performance.

Three real independent reviewers on the existing model connection supplied specialized audits and anonymous proposal review. This adapts the council's default five-advisor format to the four-slot concurrency limit. Their challenges exposed the separator and numerical-execution defects before source freeze.

The eight failures in that older screen remain failures. Fresh traces isolate extra divisions near a perforated sheet, exterior neck bone mistaken for a septum, and open folds treated as separating walls. Both air and opaque partners fail alike. These are unresolved engineering method failures under declared synthetic expectations; they cannot be reassigned to Rachel as clinical-definition questions. No supported replacement rule has passed the preserved controls without regressions.

## Eight-study comparison: failed count gate

The all-eight comparison completed two fresh-process runs per study from clean source commit `8689516`, with the same manifest, fixed settings, supplied targets and upstream localizers. Fourteen attempts produced counts; both attempts on one study withheld them for insufficient geometric coverage. There were zero technical failures. Independent verification passed with actual exit code zero: all seven available pairs have identical native/canonical arrays, affines, artifact bytes and complete registries. Original inputs, all earlier baselines and all 150 new output files remained unchanged.

Exact bilateral agreement **fell from 1/8 to 0/8** against engine `0.2.2`. Exact available sides fell from 3/14 to 2/14, and absolute side-count error rose from 27 to 28 on the same fourteen available sides. Two left-side counts changed; the other five available count pairs stayed the same. This is an observed agreement regression. The candidate failed the count gate and is not an accuracy improvement.

The audit corrected demonstrated implementation defects but did not reach the requested end state of only clinician clarifications remaining. The separator model still needs an independently supported correction, and the insufficient-coverage study still lacks a complete automatic count. Native cell identities and unresolved reference-to-series correspondence need expert review, separately from those algorithm failures. Private probes and immutable logs remain under `algorithm-defect-audit-04`; the independently closed replay is under `component-isolation-replay-01`.
