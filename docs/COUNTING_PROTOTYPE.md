# Full-volume 3D counting prototype

Status: experimental prototype implemented, not clinically validated. Protocol: ethmoid-count-prototype-v1. See [executed evaluation findings](COUNTING_EVALUATION.md) for completion rates and known limitations.

## Source of truth

The answered Ethmoid Protocol v2 questionnaire and the accompanying clinician email supersede the earlier selected-plane counting workflow. The first output is one count per anatomical cell across the full 3D volume, with left and right subtotals. The original application questionnaire still supplies the local, one-scan, review/correct/export workflow. The nine older selected-slice examples are not 3D ground truth.

Experimental inference is authorized before reference counts are available. This does not approve existing annotations for training, establish clinical accuracy, or approve every scan. Existing v1 region masks and the inactive v2 annotation draft retain their identities.

## Requirements and prototype assumptions

| Requirement | Implementation contract |
| --- | --- |
| Automatic first count | No manual ROI, seeds, or per-patient parameter selection |
| Unique 3D cells | One stable identifier across all slices; left/right/total counts |
| Partial and complete opacification | Infer cavity identity separately from the air-only display; uncertain cavities remain identifiable |
| Communicating spaces and incomplete walls | Provisional precedence: supported septum separates cells; free communication without septal evidence merges them |
| Included variants | Agger nasi, Haller, Onodi, frontal-region, supraorbital ethmoid, and concha bullosa |
| Frontal-region ambiguity | Means separately partitioned frontal-region cells, not the main frontal sinus |
| Boundary ambiguity | Explicit included variants may extend beyond the nominal core envelope |
| Inferior boundary | Bony orbital floor, not the bottom of the eye mask |
| Clinical exclusions | Previous surgery, severe disease, distorted/pediatric anatomy, artifacts, incomplete coverage |
| Unknown suitability | Exploratory output only; not silently marked clinically eligible |
| Geometry | At most 1 mm source thickness, existing preflight and HU/affine checks |
| Review | Preserve automatic predictions; corrections create revisions with lineage |
| Exports | Native-coordinate NIfTI, count CSV, local representative summary |
| Omitted | 2D landmark counts, anterior/posterior classification, composite score, extra measurements, hosting/accounts/billing |

The machine-readable contract is packaged independently of the existing annotation protocol. Run outputs must bind its version/hash and the exact input hash.

## Architecture decision

Use pretrained anatomical localizers followed by native-resolution compartment analysis. TotalSegmentator craniofacial_structures supplies skull/head and main maxillary/frontal sinus evidence; head_glands_cavities supplies eyes, optic nerves, nasal cavities and palate; the already downloaded task 297 supplies a coarse brain prior. None is an ethmoid instance model. Recover actual bony surfaces and automatically estimate the sphenoid boundary; do not count everything inside a central box or blindly subtract all semantic sinus voxels. Native bone evidence refines the brain exclusion and orbital floor. The posterior boundary follows supported anterior sphenoid-wall evidence; ambiguous evidence withholds the count.

Cell proposals use bone boundaries, physical-spacing-aware distance transforms, and constrained watershed. Air components and unconstrained watershed are diagnostic baselines, not interchangeable product modes. Multiple visible air pockets may share one cavity identity, and a completely opaque cavity may have no air voxels. Bone-only inference is an experimental hypothesis, not established anatomical accuracy.

Failed essential localization must return a reasoned failure, never zero cells or a hidden manual fallback. The posterior boundary, included variants, and opacified cavities are explicit research risks. Parameters are frozen before held-out evaluation; no expected normal count is used as a tuning objective.

### Implemented method and limits

The frozen `protocol.json` records all twelve numerical settings. Analysis keeps the native image grid, labels cavities and air separately with `uint16` IDs, and assigns physical RAS laterality. Automatic localization currently supports near-axis-aligned acquisitions only; it explicitly rejects oblique grids rather than estimating bounds on incorrect axes. Oblique import, conversion verification, review and reports remain supported independently.

Development inspection rejected earlier proposals that leaked into brain or central nasal tissue. The final development method adds brain exclusion, distinguishes bone-partitioned concha proposals from open nasal spaces, and preserves uncertainty when internal plates suggest an unresolved merge. It does not reliably identify or classify every named variant. A broad candidate region is not evidence of variant sensitivity. Fully opaque pockets are exercised synthetically; their detection in clinical scans remains unvalidated.

At least half of the air in the inferred region must be assigned to accepted proposals or the run withholds its count. This is a fixed engineering completeness check, **not** 50% clinical sensitivity or an accuracy threshold. It does not measure missing opaque cells or errors in the inferred region. Passing it still produces an exploratory candidate count requiring review. The worker records the actual observed/assigned air and coverage fraction. Failed outputs retain protected diagnostics locally while exposing no numeric count or selectable masks.

Localizers use hashed prefetched weights with outbound Python networking blocked and telemetry disabled. Fresh outputs bind source, package version, task IDs and model-manifest hashes. Historical development caches have source/mask checks but incomplete original producer binding; the result explicitly labels that limitation. Held-out evaluation requires complete producer binding and rejects legacy caches.

### Research sources

- [TotalSegmentator tasks and licenses](https://github.com/wasserth/TotalSegmentator): available local landmark models; no ethmoid-cell labels.
- [Task configurations](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_tasks_config.py): task-specific model spacing and cropping.
- [SinusSegment](https://github.com/rheadkaul/SinusSegment): combined sinonasal target, not individual ethmoid cells; not adopted as the counter.
- [PARASIDE](https://github.com/Hendrik-code/paraside): MRI modality mismatch; not adopted for CT.
- [Watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) and [physical distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html): inspectable algorithm building blocks, not clinical counting validation.

## Delivery phases

1. Freeze this contract, stabilize verification, and establish synthetic CI.
2. Lock isolated inference dependencies, prefetch weights with telemetry disabled, prepare a protected ten-study cohort, and verify actual GPU inference.
3. Implement automatic localization and counting; evaluate five development scans, record failures, and freeze settings.
4. Add persistent local jobs, native-coordinate outputs, and immutable original/corrected revisions.
5. Integrate numbered overlays, corrections, and exports in the existing application.
6. Run the five held-out studies, prepare the presentation, and document all results and limits.

At each completed phase: update README with changes and executed evidence, test, inspect staged blobs, commit, push the private feature branch, verify local/remote commit equality and a clean worktree. Medical images, per-case outputs, screenshots, and reference annotations never enter Git or CI.

## Validation boundaries

Synthetic tests require correct 3D identity, communication/septa behavior, partial/complete opacity, variant configurations, native spacing/laterality, failures, correction lineage, and identifiers above 255. Integration tests require original-count preservation, stale-edit rejection, restart behavior, and matching exports. Browser tests use synthetic data.

Real-scan feasibility records every attempted scan and its automatic result or failure. Runtime, memory, parameter sensitivity, and software integrity can be measured without reference counts. Clinical sensitivity and anatomical accuracy cannot. Independent manual 3D counts, when supplied, are compared with untouched predictions; model-assisted corrections are not independent ground truth.
