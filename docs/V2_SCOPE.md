# V2 scope and implementation plan

V2 reports left, right and total ethmoid-cell counts throughout a 3D CT volume, with inspectable candidate evidence. Clinicians determine anatomical definitions and reference measurements. Engineering checks establish provenance and software behavior; they do not establish cell accuracy.

Reuse the verified conversion, native geometry, protected storage, counting package, viewer and revision machinery where suitable. Keep one analysis package and one lightweight review application. Preserve the existing implementation and frozen research evidence. A separate V2 branch or worktree is appropriate before application or engine changes; no migration is part of the current step.

The website has Cases and a Case Viewer with review controls. A separate Review view is optional; three primary views are the maximum. It needs exact series/run identity, masks and candidate labels, reference counts, notes/error flags, reviewer attribution, history and clear save states. Shared access requires approved infrastructure and access controls. Task boards, research hubs, new scoring systems and general sinus analysis are outside scope.

## Current authorization

After public tool research, the user authorized compatibility checks, installation and configuration of suitable tools on October 12, 2026, while explicitly withholding application and engine implementation. This setup is recorded in [V2 software setup](V2_SETUP.md). It does not authorize scan processing, model initialization, database migration or deployment.

On October 12, 2026, the user approved a local evidence packet for one existing development-series counting disagreement. The packet uses saved sources, outputs and diagnostic views. It provides the full candidate registry, focused questions and a separate feedback sheet. Fresh file, geometry, registry and rendering checks accompany it. It does not run inference, tune thresholds, train models, edit clinical annotations, migrate the database or publish anything.

The packet is held outside Git in the protected local review-packet directory. The historical reference and automated output remain separate. The exact reconstruction reviewed clinically remains unresolved, even where the documented case/series correspondence exists. Same-source application histories may use a different algorithm version and must not be substituted for a frozen research run.

The local packaging step is complete. Its byte/grid/registry/figure checks and independent read-only source audit passed; scans, saved results and database contents retained their hashes. The next evidence is clinical confirmation of the reviewed reconstruction and classification of the highlighted structures.

## Sequence and advancement evidence

| Milestone | Deliverable and check | Evidence needed before advancing | Clinical dependency |
|---|---|---|---|
| Source-bound development pilot | Verified inventory and a small series selection, with nominal thickness distinct from measured spacing; preserve failed/excluded cases. | Source identity, native geometry/HU records and a frozen selection rule; group patient reconstructions together. | Confirm which reconstruction supplied a reference when uncertain. |
| Inspectable baseline | Preserve a reproducible localization/counting baseline with native labels, warnings, exact configuration and source/run hashes. | Repeatability and complete candidate registry; missing counts stay unavailable, not zero. | Interpret boundary/inclusion ambiguities only where they affect concrete examples. |
| Targeted comparison | Review existing matched references and specific broad or discarded regions using saved CT evidence. The approved packet starts here using the existing baseline. | Source-aligned clinical classification of affected structures; synthetic false-split and opacity controls before promoting a behavior change. | Determine whether highlighted regions are separate cells, communicating anatomy or excluded structures. |
| Minimal review app | Adapt the existing viewer and review storage to exact series/run identity, reference counts, attribution and immutable review history. | Software, accessibility and security checks; a new run cannot inherit approval silently. | Check that the review flow records the intended measurements. Shared hosting requires institutional approval. |
| Pilot evaluation | Freeze method and selection, report count errors separately from coverage and publish only safe aggregate engineering documentation. | Correctly matched references, all failures retained, development and untouched evaluation cases separated. | Supply expert references; voxel overlap needs reference masks. |

Communicating spaces versus incomplete walls, fully opaque cells and the named inclusion list remain clinical rules to reconcile through concrete source-aligned examples. Count agreement alone cannot establish segmentation accuracy. Previously reviewed cases remain development data. December 2026 is a prototype/evaluation target, subject to evidence and clinical input.

## Preservation and validation

Keep scans, screenshots, metadata, linked case records and review notes local in protected storage. Do not add them to Git or send them to external services/providers. Public checks use synthetic data. No clinical outreach, shared-document edits or deployment are authorized by the packet step.

For this milestone, validate the packet's source/result/artifact hashes, native grids, all candidate IDs and measurements, figure coordinates, PDF text/fields and protected-file preservation. Prior clinical executions and full pixel/HU comparisons remain historical evidence. Source documentation changes require the existing medical-data guard and a reviewed diff; unrelated application suites do not establish the packet's correctness.
