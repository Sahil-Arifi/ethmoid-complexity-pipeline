# Opening detectability before count recovery: results

The new image-formation and compatibility components passed their software checks. The experiment rejects using a restricted wall/opening library to certify recovered counts: omitting small openings can make an open wall appear to have only a sealed explanation. A separate geometric control also produces identical voxel averages from one connected space and two separated spaces, even without noise. No accepted counts were changed.

## Fixed experiment

The [design](APERTURE_IDENTIFIABILITY_EXPERIMENT.md) was frozen at source `716e56d88a160c2f664b050ae48b41dfb691e233`, before either execution. Twenty paired imaging scenarios combine five opening widths, two locations and two blur settings. Their twenty exact reindexings give forty stored records. There are ten opening geometries; changing blur or indexing does not create an independent clinical example. The sealed reference is shared within each acquisition context.

The renderer integrates continuous Gaussian blur and voxel averaging analytically. The compatibility calculation compares full HU arrays using maximum absolute error. Geometry, materials and blur are known in this diagnostic, making it a best-case model test; it does not infer them from a scan. No threshold, hypothesis grid or scenario changed after measurement.

| Illustrative maximum voxel error (HU) | Overlapping sealed/open explanations, out of 20 | Misleading sealed-only results from the restricted library, out of 20 | Matching opening retained after adding it, out of 20 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 20 |
| 1 | 3 | 2 | 20 |
| 5 | 5 | 4 | 20 |
| 25 | 8 | 7 | 20 |

The first result tests whether a common observation can fit both the sealed and open predictions. The second tests the exact opening observation against a library containing a sealed wall and only 0.25, 0.5 and 1 mm opening alternatives. These are different questions and need not have equal totals. The final column is an oracle completeness check: the generating opening was deliberately added to the library, so its retained fit is not recovered counting accuracy.

The error bounds are illustrative deterministic limits on each voxel, not measured CT noise, Gaussian standard deviations or clinical tolerances. The results establish possible ambiguity under those assumptions, without estimating its frequency in real scans.

## Why a finite search can mislead

For a small square aperture, the maximum image difference is bounded by its fractional volume in a voxel. The independently verified bound approaches zero with aperture area, including after normalized Gaussian blur. Under any positive observation-error tolerance, sufficiently small nonzero openings can therefore remain compatible with a sealed wall. Finding no fitting opening on a coarse search grid does not exclude openings omitted from that grid.

A separate unit control demonstrates a shape ambiguity at zero observation error. One construction has a 0.125 mm through-opening. The other has wider blind recesses on either side with a 0.5 mm central bone septum still separating the spaces. Independent physical predicates on a 0.03125 mm grid produce one and two connected air regions respectively. Exact averaging to 0.5 mm voxels gives **byte-identical HU arrays**. Both explanations remain compatible at zero error. This control lies outside the forty-record full-through-aperture family and is not added to that denominator.

This construction proves that this supplied observation alone cannot select the correct count for both geometries. It does not prove that most clinical CT images are ambiguous or that a validated statistical model could not perform well. Additional assumptions and information must be stated and tested rather than hidden in the shape library.

## Verification and preserved evidence

All **575 dedicated counting tests passed without skips**, and the tracked-file medical-data guard passed. The new model and runner tests failed against their initial stubs before implementation. Tests cover exact volumes, independent Gaussian quadrature, shifted sampling, identical hypotheses, error-bound equality, all 48 signed index permutations, input isolation, changed historical evidence, immutable plans and complete retained failures.

Both frozen executions completed all forty records without execution errors. The independent saved-artifact audit passed on its first attempt. It reconstructed the predictions using numerical quadrature rather than importing the renderer; the maximum discrepancy was **1.03 x 10^-12 HU**. It verified all twenty exact reindex pairs, all compatibility decisions, the aperture bound, physical geometry and finite-field signal volumes. All **84 files per execution** matched byte for byte. All thirteen frozen source hashes, the commit/runtime, **1,432 protected files**, one hundred historical records and one thousand historical arrays were verified.

Plans, per-record arrays, changed-voxel masks, measurements, the two runs and independent audit remain outside Git at `%LOCALAPPDATA%/EthmoidComplexity/synthetic-experiments/aperture-identifiability-v1/2026-10-13-01`. The canonical artifacts are `plan.json`, `run-01`, `run-02`, `audit_saved.py` and `audit.json`. The blind-recess control is reproducible in `tests/test_aperture_observation_model.py` at the frozen source commit.

## Consequence for the counter

The saved counter baseline remains **22 correct accepted counts, zero wrong accepted counts and 78 withheld counts** across one hundred development records. It was verified from saved evidence without rerunning the detector. This experiment adds a tested image-formation component and a concrete rejection test for incomplete hypothesis searches; it does not recover any of those seventy-eight counts.

A future recovery model must account for alternative wall shapes and acquisition uncertainty, and be evaluated against independent reference counts with a defined accuracy and coverage target. Adding smaller templates alone does not address the exact geometric twin. The frequency and practical importance of these ambiguities in clinical scans remain unmeasured. Automatic counting and website readiness remain unmet; patient processing and production acceptance were not changed.
