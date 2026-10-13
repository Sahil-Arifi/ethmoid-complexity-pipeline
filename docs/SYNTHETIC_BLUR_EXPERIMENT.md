# Synthetic thin-wall and blur experiment

Authorized October 12, 2026. This experiment tests the unchanged `native-bone-watershed-0.2.0` compartment stage with its complete default configuration and supplied synthetic ROIs. It uses no patient images, localizers, training, detector tuning, review-app changes or new dependencies. Its results describe constructed objects and a simplified imaging model; clinical accuracy is not measured.

## Frozen construction and imaging

There are **24 unique cases**, not 48 observations when repeated:

- Eighteen sealed pairs: wall thickness 0.25, 0.5 or 1.0 mm, Gaussian blur standard deviation 0, 0.25 or 0.5 mm, and X grid phase 0 or 0.25 mm.
- Six absent-wall controls: blur standard deviation 0, 0.25 or 0.5 mm crossed with the same two phases. Each control contains one connected physical space.

The physical air enclosure occupies X [-8, 8), Y [-6, 6), Z [-4, 4) mm. A sealed pair has air boxes X [-8, -t/2) and [t/2, 8), where t is wall thickness; the divider occupies [-t/2, t/2). The control removes only that divider. All material and reference boundaries are half-open. Air is -900 HU and all other material is bone at 800 HU. The total physical air volume is (16-t) x 12 x 8 cubic millimeters. Thickness changes that volume; phase and blur do not change the physical construction.

The detector grid is 48 x 40 x 32 at 0.5 mm spacing. Its lower field edges are (-12+phase, -10, -8) mm; voxel centers are those edges plus 0.25 mm plus the voxel index times 0.5 mm. The affine maps those centers directly to RAS coordinates. Phase translates the sampling grid relative to the same physical objects. It does not move, resize or rotate the objects. The supplied ROI covers the entire field.

Construct material labels at centers of a 0.125 mm grid covering the same field (192 x 160 x 128). These declared thicknesses and phases are represented exactly by the fine-grid material cells. Validate fine-grid air topology, per-label connectivity and physical volume independently before detection. Retain the fine labels, detector-center reference labels and analytically drawn interior cores. Each core is inset 1 mm from every face of its physical box and must be nonempty. Reference identities come from the physical boxes, never from post-imaging connected components. The two sampled reference labels can touch when no detector center lands inside a thin wall.

Convert fine material labels to float64 HU. Apply SciPy `ndimage.gaussian_filter` with sigma = sigma_mm / 0.125 in all axes, constant extension at 800 HU and truncation at four standard deviations. Sigma zero preserves the unblurred HU array. The smallest exterior bone margin is 3.75 mm, exceeding the largest 2 mm kernel radius; extending the field cannot expose the enclosure to a different material. Blur precedes averaging, with no intermediate thresholding. Average aligned, nonoverlapping 4 x 4 x 4 fine cells into each detector voxel, then convert to float32. There is no noise model, scanner calibration, measured point-spread function or reconstruction-kernel claim. Sigma describes this constructed Gaussian kernel; it is not a clinical CT resolution specification. [SciPy defines sigma as the kernel standard deviation and documents constant extension and truncation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

## Premeasurement checks

Before the first detector invocation, verify the generator and scorer using tests independent of detector outputs:

- Check physical volumes, topology, grid-to-world alignment, nonempty cores and deterministic input hashes.
- Compare unblurred wall occupancy with analytic voxel fractions for both phases. For a 0.25 mm wall the central peak is -475 HU at phase 0 and -50 HU at phase 0.25; for a 0.5 mm wall it is -50 and 800 HU respectively.
- Compare central blurred profiles with independently normalized one-dimensional discrete Gaussian convolution followed by aligned averaging, using a fixed 0.001 HU absolute tolerance. Check constant bone remains 800 HU. This checks implementation and physical units, not continuous-phantom convergence or scanner calibration.
- Verify the scorer against deliberately perfect, relabeled, empty, merged, split, extra, disconnected and side-swapped predictions, including a compensating merge/split and a single-space control with and without its required reason.

Freeze the rules hash, all construction/input hashes, complete detector configuration, source hashes, package versions and Python version before measurement. The executable frozen plan includes source hashes as well as inputs and rules. A source change after freezing requires an explicitly retained new premeasurement plan and a new run; it must not overwrite an earlier result.

## Recovery criteria

Report physical count recovery separately from overlap and core coverage. The scorer maximizes Dice correspondence through a one-to-one assignment. Every physical reference label receives a row; a missing match has Dice and core coverage zero. Report unmatched prediction IDs and connected-component counts. Do not omit failures from denominators or use a correct total to excuse a wrong partition.

A case meets the compound engineering recovery requirement only when all of the following hold:

1. Exact physical total: two for a sealed pair, one for an absent-wall control.
2. A complete bijection between reference and detected IDs, with no extra labels.
3. Dice at least 0.90 and interior-core coverage at least 0.95 for every matched reference label. Report numerical scores even when a case fails. These are predeclared engineering conventions against center labels, not accuracy estimates for partly occupied boundary voxels.
4. Each predicted instance is connected under face connectivity.
5. For paired spaces, every matched instance has the correct physical centroid-X side relative to zero, checked both in its registry and from its mask/affine. This is a synthetic coordinate check, not anatomical laterality validation.
6. For the single-space controls, disregard centroid-side assignment and require the existing `crosses_midline_centroid_laterality_requires_review` reason. A report-added caution cannot replace a missing engine reason.

A `CountingError` yields null candidate counts, no instances and zero overlap/core coverage; retain its code and fail recovery. Unexpected exceptions and output/registry inconsistencies are execution errors, distinct from recovery failures. Validate saved label IDs against registries, cavity/air consistency, ROI containment, voxel counts and unchanged input hashes. Recovery failures complete the experiment honestly; they do not authorize detector changes or promotion to clinical use.

## Independent input visibility and reporting

Using the unchanged 200 HU bone threshold, label face-connected non-bone regions in the imaged input. A sealed divider is reported as a threshold-visible barrier only when each nonempty reference core lies entirely in a non-bone component and the two component IDs differ. Retain the component count and the largest central-profile HU within |X| <= 1.5 mm. Wall visibility is inapplicable to absent-wall controls. This diagnostic is a threshold-specific explanation, not proof of information loss at all thresholds or a new ground truth. It never changes recovery expectations or detector settings.

Save a fresh, non-overwriting artifact directory outside the source repository for each process: protocol before detector calls, every generated input and fine truth, detected cavity/air labels, registries, scores, exceptions, input visibility, numbered comparison views, complete table and PDF. Input arrays are read-only. Compare canonical protocol/results and array hashes across two fresh processes. Report their 24 shared cases once and report repeatability separately. Preserve earlier experiments unchanged.

Interpret failed physical recovery with and without a threshold-visible divider separately. A disappeared threshold barrier can explain why recovery is difficult; it does not convert the failure into success. A remaining threshold barrier does not alone identify a detector bug. Irregular anatomy, oblique partitions, opacity, noise, scanner response, automatic anatomical localization, full-volume gates and patient accuracy remain outside this run.

## Design review and source scope

Five independent LLM Council advisors and five fresh anonymous peer reviewers reviewed this design on the existing connection in batches because of the agent limit. They agreed to retain physical identity, preserve failures and avoid tuning. Peer review added independent occupancy/convolution checks and adversarial scorer self-checks. The chair combined execution provenance with a separate input-visibility diagnostic. No additional provider was configured.

The existing source-only Graphify map was queried for `compartment_stage` and its separator-merging relationship, then the configuration and acceptance code were read directly. Only the new experiment script, its tests, this design, README and the dedicated counting CI test list need change. The original counting source, original frozen experiment, patient data and application behavior remain unchanged.
