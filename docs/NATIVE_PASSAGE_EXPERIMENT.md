# Native passage evidence: fixed diagnostic experiment

The user authorized resuming counting-algorithm work after confirming that the existing navigation markers cannot be exported. The clinician rules, existing reference counts and screenshot remain useful evidence; exports and repeat answers are not prerequisites for this engineering experiment. No patient inference or website work is part of this step.

## Decision and scope

Measure whether original-voxel paths offer information that the current smoothed, minimum-clearance passage check loses. Keep the ridge candidate and its acceptance rules unchanged. This is an observable-discriminability experiment, not another claimed automatic counter or a search for a favorable threshold. It does not replace the failed count-admission result.

Two independent advisors and one anonymous reviewer used the existing authorized model connection. One favored a bounded forward-model comparison; the other identified native opening evidence as the smallest target. The peer identified model incompleteness as a risk: an apparently unique answer can result from omitting a plausible opening. The selected step measures the proposed observables before implementing an inverse model or conditional count rule. This reduced council respects the user's usage preference.

## Fixed measurements

Only original HU, supplied ROI, orthogonal affine and detector-generated partitions enter `passage_evidence`. Each partition must be face-connected and remain inside eligible non-bone space. For each contacting pair:

1. Define terminals within each partition from its lowest 20th-percentile native HU, farther than 1.01 times the largest voxel spacing from the original non-bone-region boundary. Missing terminals are explicit unavailable measurements.
2. Search sorted unique native intensities for the lowest threshold permitting a face-connected path between terminal sets inside the pair's union. This is the minimax native-HU bottleneck. Other basins and the ROI exterior cannot provide shortcuts.
3. At that bottleneck, find the largest voxel-center distance-to-background for which a path still exists. Save the union of all components that each touch both terminal sets as a witness mask; equally qualifying corridors can be disconnected from each other. This quantity is **not** an anatomical opening radius or diameter: voxel-center sampling and image blur limit its interpretation.
4. Record contact-HU quantiles and the fraction of contact voxels close to original strong bone. Artificial watershed boundaries never become bone evidence.

Connectivity uses SciPy's explicitly symmetric face structure ([label documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)); physical sampling uses [distance-transform spacing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html). The installed locked runtime is checked in the saved plan. Threshold connectivity and distance clearance are monotone searches, checked against analytic constructions before measurement.

No measured path overrides `material_ambiguity` or `unresolved_communication`. Low-HU connectivity can reflect a blurred sealed wall. No per-case construction information, desired count, clinician total, label or core enters the extractor.

## Frozen census

Read all 68 existing ridge cases and saved partitions from their protected `run-01`, checking their input/output hashes. These are known development observations. Reuse their saved detector diagnostics; do not rerun the historical detector or alter prior outputs.

Add exactly 16 physical constructions and their exact X/Z swap plus Y reflection, making 32 perturbation records. Native spacing is 0.5 mm and fine spacing is 0.125 mm. Bone is 800 HU, air -900 HU, and the cavity occupies X [-5,5), Y [-4,4), Z [-3,3) mm. Use physical half-open wall and aperture predicates, constant 800 HU exterior, Gaussian truncate 4 and block averaging:

- Sealed walls 0.125, 0.5 and 1 mm thick, each with blur sigma 0 and 0.3 mm: six.
- A 1 mm wall with square apertures 0.25, 0.5 and 1.5 mm wide, each with those same two blurs: six.
- Sealed 0.5 mm wall and 1 mm wall/0.5 mm aperture, each with 0.125 mm X phase and 0.3 mm blur: two.
- That same closed/open pair at zero phase and 0.3 mm blur with fixed 25 HU Gaussian noise: two.

Physical references come from geometric predicates before imaging. The fine-grid connectivity must be two for sealed walls and one for openings; native center labels can miss subvoxel structure. These are idealized constructions, not calibrated scanner simulations. Exact reindexings are consistency checks, not independent observations. The unchanged ridge detector supplies proposals for the 32 new records.

## Evaluation and stopping rule

Execute this fixed 100-case census once and repeat in a separate process. Save all inputs, measurements, witness masks, detector diagnostics, failures and complete denominators outside Git. If extraction fails after detection, retain the completed partitions and detector diagnostics with the failure stage. Freeze source, runtime, design and input hashes before measurement; refuse changed plans, existing output destinations and writes inside historical inputs. Preserve and rehash the complete historical run.

For records with exactly one measured contacting pair, tabulate the counterfactual rule `native bottleneck <= threshold` at exactly -900, -800, -700, -500, 0 and 199 HU. Count true one-space detections and sealed-wall false connections separately for historical and new data. This finite diagnostic table selects **no** operating threshold and computes no clinical accuracy. Multi-pair and unavailable cases remain in the full census, outside that explicitly limited table denominator.

Independently inspect saved witness connectivity, terminal contact, HU bounds, source preservation, exact-input twin consistency and transformed measurement equality. Use 1e-5 HU and 1e-8 mm only for numerical comparison under reindexing; these are not acquisition-error allowances. Explain overlaps and missing evidence. Stop after this batch and repeatability audit, with no threshold retuning, detector promotion or fabricated accuracy target.
