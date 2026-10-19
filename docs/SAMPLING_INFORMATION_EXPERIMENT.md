# New samples versus image enlargement: fixed experiment

Authorized October 19, 2026, for algorithm research and a Google Doc for the clinician meeting. This is a measurement-information experiment. It does not run the clinical counter, recover historical withheld counts, calibrate a scanner, or request another patient scan. The two physical explanations and their imaging parameters are supplied to the inference function; discovering anatomy is outside this test.

## Question and decision rule

Can an additional, genuinely sampled view distinguish a through-opening from opposed blind recesses when the original sampled volume cannot? Compare original samples, linear interpolation of those samples, and original samples combined with a second independently forward-rendered grid shifted 0.125 mm along X. A shifted reconstruction of existing CT data is not assumed to supply independent measurements.

The already tested exact Gaussian-blur/voxel-average integrator renders a 4 mm cubic field as 8 x 8 x 8 voxels, spacing 0.5 mm, first center (-1.75, -1.75, -1.75) mm. A cubic air chamber has half extent 1 mm, a central X wall of thickness 1 mm, air -900 HU and bone 800 HU. The through-opening has square Y/Z width a. Blind recesses extend depth d from each wall face, have Y width a x (1 mm)/(2d) and Z width a. Their removed bone volumes are equal; the retained septum is 1-2d mm. The physical reference is one connected space for the through-opening and two for the blind recesses. These are synthetic connectivity references, not clinician cell annotations.

Compatibility means maximum absolute voxel residual at most 5 HU, an illustrative bounded error rather than measured scanner noise. Exactly two hypotheses are supplied. Accept their conditional count only when one is compatible; withhold if both or neither fit. Each pair shares a deterministic uniform error field in [-5,5] HU for its two possible truths; the second acquisition uses another fixed seed. Both truths are always evaluated. Interpolation applies scipy.ndimage.zoom with factor 2 and order 1 to the original observation and both hypotheses using the same operator. Joint comparison concatenates the original and shifted arrays without resampling.

## Frozen census and separation

Development: four pairs, a in {0.125, 0.25} mm, d=0.25 mm, blur sigma in {0,0.15} mm. Evaluation: sixteen new parameter pairs, a in {0.0625,0.09375,0.15625,0.21875} mm, d in {0.125,0.375} mm, sigma in {0,0.15} mm. All openings are centered. Both truths produce eight development and thirty-two evaluation observations per method. Parameter combinations are synthetic variations; they are not independent patients or different anatomical families. Test data are not used to choose the shift, error bound, model, or target.

The seeded field uses NumPy default_rng with seed 20261019+pair_index*10+acquisition_index. A fresh process repeats the frozen run. Save all prediction arrays, observations, residuals, decisions, pair parameters, source hashes, runtime and per-file hashes outside Git. Any execution failure stays in the census and fails the experiment; it is never counted as a successful abstention.

## Gates and limits

Software gates: all forty main observations retained per method; no execution errors; true generating explanation remains compatible; inputs unchanged; exact reindexing leaves decisions invariant; interpolation cannot increase maximum hypothesis separation; identical-input controls withhold; no-fit controls withhold. For each acquisition, use the hypothesis midpoint as an additional ambiguity control when separation is at most 10 HU. Controls and reindexings stay outside the main denominator. Verify two executions and preserve the prior aperture experiment byte for byte.

The investigator-selected exploratory utility target is zero incorrect accepted evaluation counts and at least 80% conditional coverage (26/32 observations) for the combined measurements. Report correct, wrong, withheld, errors, coverage and separation for all three methods, separately for development and evaluation. Report paired changes and every pair, including cases that remain unresolved. The 80% target has no clinical approval or clinical-performance meaning. A pass only supports these supplied geometries with exact calibration and bounded error. A failure is retained without retuning or removing cases.

The exact prior ambiguity controls remain regression requirements. Known shape alternatives, exact blur/material/grid parameters, ideal error bounds and simulated additional observations make this an optimistic information test. It cannot establish a complete hypothesis search, scanner feasibility, patient-count accuracy, or improvement in the unchanged historical 22 correct / 0 wrong / 78 withheld counter baseline. Finish one frozen evaluation, audit it, and deliver the meeting document with these limits.

## Decision review

Two independent advisors on the existing authorized model connection considered fitted 3D wall context and additional measurement information. An anonymous peer preferred the latter for this bounded meeting deliverable, because fitting uncertain geometry could hide the assumptions responsible for apparent count recovery. The review also required explicit disclosure that the correct shape alternatives are supplied. This compact council adapts the installed five-advisor workflow to the time and usage constraints; it is not a clinical review.
