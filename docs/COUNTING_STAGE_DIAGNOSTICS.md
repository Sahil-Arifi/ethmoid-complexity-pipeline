# Counting-stage diagnostics

Use stage evidence to distinguish a missing starting point from a later merge,
ROI exclusion, or rejected proposal. Final cavity masks alone cannot show which
decision caused a count discrepancy.

`analyze_volume(..., diagnostic_stages=True)` writes two protected sidecars in
its fresh output directory. The application and workers do not enable this
option. A clinical replay still requires the operator's approval; synthetic
fixtures are used for implementation checks.

- `diagnostic-stages.local.npz`: original membership, exclusion and reconstruction
  masks; seed markers; watershed basins before merging; labels after merging;
  the crop's physical RAS affine.
- `diagnostic-stages.local.json`: source and engine hashes, method/protocol and
  parameters, crop bounds, estimated midline, seed counts, dominant interface
  direction and bone support, merge decisions and constraints, rejected proposals
  with the check that rejected them, and accepted proposal-to-cell ID mappings.

The arrays use the cropped canonical grid. They must be placed using the saved
affine or crop bounds, never treated as native scan indices. Basin and merged
labels are intermediate IDs; only `cell_id` identifies a final candidate. A
rejected proposal or disconnected air component is not an expert-defined cell.
Trace files are excluded from public result links and exports. Keep them beside
the other medical outputs outside Git. Do not upload them externally.

Synthetic tests compare ordinary and traced results and native output bytes,
including a flipped input grid and rejected thin pore. The algorithm's thresholds,
side assignment and acceptance rules are unchanged.

A failed run may leave a partial trace, such as markers and initial basins without
merged labels. Check the adjacent `result.json` and available arrays before
interpreting it. These files do not make a withheld clinical count available.

The current investigation reproduced a narrow chamber losing its seed beside an
incomplete wall. More permissive seed selection recovered it, but caused extra
splits when isolated bone inclusions were added to a single chamber. Connected
bone size alone also failed those controls. These alternatives are diagnostic
experiments, with no justification for deployment or a clinical accuracy claim.

For an approved replay, preserve the selected reconstruction, frozen parameters
and cached anatomical masks. Write a new directory, compare its ordinary outputs
with the baseline, and inspect stage evidence at disputed locations. Freeze any
subsequent algorithm change before evaluating additional scans. Keep development
results separate from independent clinical validation.

An approved replay reproduced its frozen ordinary outputs byte for byte and
showed two distinct mechanisms: broad single-seed basins before merging, and
otherwise eligible basins rejected after merging with mostly outside-region
geometry. Eligibility under these gates does not establish a true clinical cell.

A synthetic outside-region merge veto recovered an interior target, but split
one communicating cavity into two candidates. Checking final merged membership
before preserving basins moved the error to a larger communicating cavity.
The veto also changed counts when a narrow connection became opaque with bone
geometry held constant. Neither rule was promoted. A justified change needs
bone or anatomical evidence that distinguishes a cell wall from communication,
with opaque-cavity and single-cavity controls; matching a reference total is
insufficient.

These controls use full reconstruction boxes to isolate the shared stage. With
the same phantom ROIs and production's exact 3 mm growth policy, the outside
connections disappear and neither guard fires. The baseline itself splits the
U-shaped control after search clipping. Those experiments establish shared-stage
limitations, not guard-induced failures in the complete production path. A
reachable target/control pair is still needed. Private replay arrays, coordinates
and reports remain outside Git.
