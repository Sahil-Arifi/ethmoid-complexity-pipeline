# Frozen faint-wall comparison

Authorized October 13, 2026. Test one opt-in synthetic candidate against the unchanged `native-bone-watershed-0.2.0` compartment stage. Keep its production source, defaults, localizers, application, old inputs and old results unchanged. No patient processing, training, deployment, additional providers or new dependencies. The known 24 thin-wall cases are development evidence. This protocol is frozen before any candidate measurement; no threshold sweep or second candidate is part of this experiment.

## Candidate

Inside the supplied ROI, strong voxels have original HU >= 200. Permitted propagation voxels have original HU >= 50. Face-connected binary propagation starts from strong voxels and remains within the permitted mask and ROI. The inferred exclusion mask contains reached voxels with HU < 200. There is no interpolation, dilation outside permitted intensities, hole filling, closing, reference-label access, expected-count access or special divider location. Pass this mask through the existing `excluded` argument to `compartment_stage`, with the complete original default configuration. Original HU still determines strong-bone support/enclosure and air content. Mark all candidates and the result as a weak-evidence experiment whenever the inferred mask is nonempty; retain all original reasons.

The 50 HU value is one fixed engineering convention for this test, not a calibrated bone cutoff or the declared material midpoint (-50 HU). Attachment to strong bone rejects detached weak fragments; it does not distinguish bone from soft tissue or an artifact. Complete exclusion barriers can manufacture a division that the existing merger cannot undo. This risk is part of the acceptance test. Binary propagation follows [SciPy's documented masked propagation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_propagation.html), with explicit 3D face connectivity.

## Cases and independent truth

Evaluate all 24 previously defined development cases using the unchanged generator. Add exactly 32 new challenge cases, created before detector calls and reported separately. They are a deterministic, predeclared synthetic challenge, not a clinical or statistical holdout.

The challenge field, fine/coarse spacing, declared materials, half-open enclosure and blur-before-block-mean conventions match the prior protocol: field edges (-12 + phase, -10, -8) mm, shape (48, 40, 32), fine spacing 0.125 mm, coarse spacing 0.5 mm, air -900 HU, bone 800 HU, Gaussian constant extension 800 HU and truncate 4. No fine-label resampling defines coarse truth: evaluate the physical predicates directly at each grid's centers. Reference cores use a conservative 1 mm interior margin and must be nonempty, contained in their coarse reference label, and face connected. Fine truth must have the declared unique object count and face-connected labels.

Sixteen complete-divider pairs cross these four geometries with four imaging rows:

| Geometry | Divider signed coordinate d |
| --- | --- |
| Off-center flat | x - 1 |
| Oblique | x + 0.75 - 0.25*y |
| Curved | x - 0.5 - 0.04*(y*y - 12) |
| Axis permutation | Off-center flat construction, then swap array axes 0 and 1 with the matching affine |

The divider is -t/2 <= d < t/2; reference pockets lie on either side. The oblique/curved t is signed-coordinate width, not a claimed normal physical thickness. Their conservative core margin uses 1 mm times sqrt(1 + slope squared); the curved slope bound is 0.48 within |y| <= 6. This preserves the specified physical interior margin without using the detector.

| Imaging row | t (mm) | Gaussian sigma (mm) | X grid phase (mm) |
| --- | --- | --- | --- |
| 1 | 0.375 | 0.125 | 0.125 |
| 2 | 0.375 | 0.375 | 0.125 |
| 3 | 0.75 | 0.375 | 0.375 |
| 4 | 0.75 | 0.625 | 0.375 |

Sixteen one-space controls cross eight constructions with two imaging rows (sigma/phase 0.125/0.125 and 0.375/0.375 mm). Their reference is the entire single enclosure, including its soft-tissue content; bone fragments are excluded from that reference. A material aperture must leave one face-connected physical pocket. Cores exclude bone and its 1 mm physical neighborhood.

1. No divider.
2. A 0.75 mm central bone divider with a 3 by 3 mm opening: remove bone where |y| < 1.5 and |z| < 1.5.
3. A boundary-attached 0.75 mm central bone ridge only where y < -2.
4. An isolated 0.75 mm central bone fragment only where |y| < 2 and |z| < 2.
5. A smooth content ramp from -900 to -600 HU across x in the enclosure.
6. No divider plus detector-grid Gaussian noise with standard deviation 75 HU, applied to the whole imaged field after averaging. Use NumPy PCG64 seeds 2026101301 and 2026101302 for the two rows. Noise is an arbitrary declared challenge, not a scanner calibration.
7. A complete 0.75 mm central soft-tissue sheet at 35 HU, attached to the surrounding enclosure.
8. The same soft-tissue sheet at 100 HU. It remains content of one physical cavity and directly challenges false exclusion barriers.

These control references follow physical connected-compartment identity as an engineering convention. A communicating bone-divider construction is not a clinical cell-count oracle: the current counter deliberately retains some supported incomplete septa as two uncertain candidates, and the prior experiment treated those examples as clinically unsettled. It may therefore already fail the one-connected-space control here. Preserve that disagreement, describe it separately, and do not reinterpret it as proof of a clinical counting error. The strict synthetic admission gate still requires all declared controls to pass; it may reject both methods' topology conventions.

Before measurement, assert construction topology, half-open identities, nonempty connected cores and predicate-to-coarse identity; validate image formation with analytic occupancy/convolution oracles. Save fine labels, coarse labels, cores, original image, ROI and affine; hash all six arrays. Keep physical truth independent of imaged non-bone connectivity. Record that connectivity at 200 HU and under the candidate exclusion mask, without changing truth or scoring.

## Scoring, admission and stop

Reuse the frozen prior one-to-one scorer: exact count, complete bijection, Dice >= 0.90, core coverage >= 0.95, connected predicted instances and physical centroid-side checks for pairs. Single-space controls require the original crossing-midline reason; missing matches score zero. Correct totals cannot excuse incorrect shapes. Record expected failures with null counts and zero matches. Unexpected exceptions, malformed outputs or mutated inputs are execution errors, separately retained.

Admit the candidate only if every condition holds: strict compound-recovery improvement on development pairs; strict compound-recovery improvement on challenge pairs; every baseline compound success across both groups remains a candidate success; all 6 development and all 16 challenge controls meet compound recovery; zero extra divisions in any control; zero execution errors. A case can improve the count yet fail overlap; report both measures. Every row remains in the denominator, including unresolved image evidence. Any failed condition rejects admission.

Stop after this one candidate comparison. Do not change its threshold, attachment, fixtures, scoring or gates after seeing results. Passing permits only further synthetic research; it does not promote a production or clinical method. A rejected candidate remains a completed research result with preserved failures.

## Validation, provenance and outputs

Before measurement, test the empty-mask baseline identity, exact threshold/connectivity/ROI rules, original input preservation, original-HU air bookkeeping, uncertainty reasons, axis permutation/flip, opacity and strong-enclosure abstention. Tests may exercise separate analytic unit fixtures; they must not execute candidate counts on the 56 evaluation cases before freezing.

Freeze complete source/rules/configuration/runtime hashes and all 56 case manifests before detector calls. Each method receives a separate copy of all six arrays with read-only flags, so a detected input mutation cannot contaminate the other method or saved original. Preserve the original empty-prediction scores before counting so a failure still retains every reference object. Run the two methods on identical inputs in two fresh processes, preserving a fresh output directory for each. The repeated process is repeatability evidence, not another 56 observations. Compare every protocol/result byte and saved array value; allow only PDF creation timestamps to differ. Save per-method inferred masks, cavity/air labels, registries, continuous scores, count changes, all failures, numbered comparison images, a complete Markdown table and PDF outside Git.

The source-only existing Graphify map was queried, then the full counter and callers were read. Its truncated graph traversal and previously documented dangling-edge limits were resolved for this decision by direct source inspection; no graph input included patient data. Five independent council advisors and five anonymous peer reviewers ran on the existing connection in batches. They agreed on frozen evaluation and no false splits; they disagreed on whole-threshold versus exclusion intervention. Peer review rejected a wrong material premise and identified irreversible false barriers. The chair selected one helper-only exclusion candidate; localization is already bypassed, so a lower counting config would not itself execute a localizer. No new provider was configured.
