# Synthetic counting experiment

The user authorized this counting experiment on October 12, 2026 and excluded review-app work. It uses generated arrays only: no patient scans, anatomical localizers, training, parameter search, database access or application startup. The current `native-bone-watershed-0.2.0` compartment stage and its default configuration remain unchanged.

## Frozen design

Twenty fixtures use bone at 800 HU, air at -900 HU and soft tissue at 35 HU. The ordinary grid is 64 x 48 x 40 at 0.5 mm spacing with a centered RAS affine. Construction labels are drawn directly from boxes or spheres, independently of detection. Every label must be connected in the sampled grid; non-bone connectivity and HU assignments are checked before execution.

| Category | Fixtures | Predeclared requirement |
| --- | --- | --- |
| Exact construction recovery (14) | One, two and four sealed pockets; connected lobes without a wall; continuous one- and two-voxel walls; partial, full and mixed opacity; diagonal spherical pockets; anisotropic spacing; flipped axes; permuted axes; oblique grid | Correct left/right/total counts, one-to-one full voxel equality allowing label renumbering, correct physical side for each matched label, and any required opacity or crossing-midline reason. A correct total with wrong boundaries fails. |
| Unsettled wall interpretation (2) | A 1.5 mm or 4.5 mm opening in the same one-voxel wall | Show alternatives of one or two pockets. No exact-count score. Require an existing engine reason for inferred separation/unresolved internal walls, or explicit `NO_SUPPORTED_CELLS` abstention. A report-added caution cannot replace a missing engine flag. |
| Boundary control (1) | ROI cuts through the single pocket | Explicit `NO_SUPPORTED_CELLS` abstention or the existing `touches_inferred_anatomical_boundary` reason on an uncertain candidate. No exact-count score or complete anatomical total. |
| Rejection controls (3) | Unbounded soft tissue; one-voxel pore; sheared affine | Respectively `NO_SUPPORTED_CELLS`, `NO_SUPPORTED_CELLS` and `INVALID_GEOMETRY`, with null counts and no accepted instances. |

The anisotropic grid is 0.4 x 0.7 x 0.9 mm. A flip and an axis permutation reindex arrays and affines together, preserving physical geometry. The oblique fixture rigidly rotates the construction 15 degrees; it does not resample voxels or simulate partial-volume recovery. Left/right here means the sign of a construction's physical centroid X relative to zero, not independently validated anatomical laterality.

## Execution and interpretation

Before detector execution, save the fixture hashes, rules, complete default configuration, runtime versions and counting-source hashes. Keep every row, exception, null and unmet flag requirement. Run the same frozen experiment in two separate processes and compare protocol/results hashes. Preserve both executions; do not replace failures or change expectations after seeing outputs.

The runner accepts only a fresh output directory outside its source repository. It saves generated HU, construction labels, ROI, affine, detected cavity/air labels, registries and numbered comparison images. These local artifacts are synthetic; binary image artifacts still remain outside Git under the existing medical-data guard.

Every exact-recovery and control requirement must pass before promoting this baseline on synthetic evidence. Completing the experiment and honestly reporting a failure still completes the authorized task. No threshold or engine improvement is part of this run.

The supplied ROI bypasses anatomical localization. This low-level compartment stage warns about low air coverage; the full-volume entry point has additional coverage gates. These fixtures do not test those entry-point gates, realistic image formation, disease, clinical inclusion rules or patient accuracy. No overlap metric from these arrays may be described as clinical segmentation accuracy.

## Design appraisal

Five independent council advisors and five anonymous peer reviewers ran on the existing authorized connection in batches because of the agent limit. They agreed to preserve settings, score spatial correspondence, retain ambiguity and omit the optional PoreSpy comparator. There was no substantive disagreement. Peer review added explicit sampled-geometry assertions and executable flag/matching rules. No new provider was configured.

The source-only Graphify map covers seven counting Python files. It reported 38 dangling endpoints, one self-loop and 37 collapsed edge variants; relevant calls and acceptance logic were checked directly in source. The map is an aid to understanding, not evidence of complete execution coverage, and stays outside tracked files.
