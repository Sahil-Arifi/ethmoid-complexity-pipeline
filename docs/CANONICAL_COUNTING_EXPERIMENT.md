# Canonical counting and communication withholding: fixed design

The user authorized starting the two repairs identified in the [native-passage results](NATIVE_PASSAGE_RESULTS.md). This phase creates one isolated synthetic successor, fixes exact-reindexing inconsistency and removes unsupported acceptance of disconnected spaces. Production code, original candidate source, patient scans and prior results remain unchanged. No website or clinical outreach is included.

Two independent advisors and one anonymous peer reviewer used the existing authorized connection. Both advisors recommended a lossless orientation wrapper and an explicit communication abstention, with the resulting loss of correct counts reported. The peer required a precise orientation rule, complete inverse mapping and an ROI-boundary flag for the diagnostic witness. The reduced council follows the user's usage preference. A forward-model classifier and automatic merges from residual HU are outside this candidate.

## One fixed candidate

`count_with_canonical_ridges(hu, roi, affine)` receives only image observations. Validate finite real HU, a nonempty matching boolean ROI, orthogonal nondegenerate geometry and a homogeneous affine final row.

1. Normalize each affine direction column by its spacing. Give it the sign that makes its largest absolute component positive; ties use the first RAS coordinate. Sort the signed columns by descending absolute X/Y/Z components, followed by descending signed X/Y/Z components. Signed zero has the same key as zero. Do not round small components or interpolate data. Orthogonal columns cannot share the complete signed key.
2. Permute and reflect HU and ROI to this order in contiguous storage. Update the affine with the exact index mapping, including the translation for reflected voxel centers. Run the complete frozen ridge pipeline on that grid, including smoothing, marker creation, fallback seeding, watershed and existing checks.
3. Count face-connected components of the original `ROI & (HU < 200)` on the canonical grid. If there is more than one, append `unresolved_subvoxel_communication`, withhold all three counts and retain the proposed masks and cells. Apply this independently of watershed adjacency or whether any wall measurement is available. Preserve all existing abstentions. This rule deliberately sacrifices correct sealed-pair counts; withholding is not a recovered count.
4. Measure a diagnostic bridge only for exactly two native free components. In each component, terminals are its lowest-20th-percentile HU voxels farther than 1.01 times the largest spacing from the original free-space boundary. The search domain is the entire supplied ROI, including voxels classified as bone. Find the lowest native HU threshold that permits face connectivity between terminal sets. Save the union of all qualifying components as the witness, and report whether it touches the ROI boundary. An unrelated route may contribute; neither its bottleneck nor its location is interpreted as an anatomical opening.
5. Return every spatial output to the original index grid, including the bridge witness. Cell coordinates retain physical RAS meaning. Explicitly mark unavailable bridge evidence for other component counts, missing terminals or a disconnected ROI. None of these statuses proves a sealed wall.

The invariance claim concerns exact index permutations/reflections of the same sampled volume. Resampling, acquisition rotation and scanner differences are not covered. The pair diagnostic is limited to exactly two components; larger disconnected sets remain withheld until a separately validated multi-component analysis exists.

## Validation before measurement

First capture failing regressions against a passthrough to the old detector. Require all 48 signed permutations for each of three grids: the failing axis-aligned plateau, an anisotropic oblique plateau and an anisotropic oblique strong wall. Check exact corresponding partitions, cavities, air, bridge masks, decisions and physical cell registries (centroid tolerance 1e-8 mm). Reject invalid affines. The original narrow-opening abstention and accepted one-component controls must remain intact.

Analytic bridge tests cover uniform strong bone, attenuated bone, excluded ROI paths, missing terminals, more than two components and disconnected ROI islands. Independently validate minimum-threshold connectivity and witness membership. Runner tests must establish observation-only inputs, mutation detection, refusal of changed freezes or existing destinations, and complete saved failures.

## Frozen comparison and stopping rule

Load all 100 saved records from the preceding native-passage experiment. They are known development data, including 16 new physical constructions and their repeated reindexings; none is an independent clinical evaluation set. Verify exact input/output manifests and hashes and equality of aggregate/per-case records. Reuse the saved baseline decisions and partitions without rerunning an old method as a baseline.

Freeze source commit and hashes, runtime, this design, metadata and inputs before execution. Protect the full preceding native-passage directory and the earlier ridge input run. Plans and destinations must remain outside Git and both protected trees. Execute once and repeat in a separate process, preserving every result, failure, input, output and registry.

Report baseline and successor correct accepted counts, wrong accepted counts, withheld counts and execution errors separately for both existing groups. Also report each lost correct acceptance. All eight baseline wrong acceptances must become withheld; no new wrong acceptance is allowed. Every previously correct accepted single-component count must remain available. All exact-input material twins must retain matching withheld decisions. Check all 22 saved base/reindex pairs for exact corresponding masks and consistent physical cells, retaining tolerance failures rather than widening tolerances.

An independent saved-artifact audit must verify the census, references, hashes, truth isolation, bridge minimality, witness domain/terminal/boundary checks, exact reindexing, registry measurements, repeatability and historical preservation. Software checks and these repair requirements are distinct from the project's unmet automatic-counting accuracy requirements. Stop after this one fixed candidate, two process executions and audit; do not adjust a threshold after seeing results or promote the candidate from synthetic evidence.
