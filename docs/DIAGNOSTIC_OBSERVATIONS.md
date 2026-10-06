# Review observations for saved cell counts

The counting workspace can highlight an individual saved candidate and record
an observation about a possible missed cell, merged cells, extra detection, or
side assignment. This supports a focused review with physicians while keeping
automatic predictions separate from review evidence.

## Use the review view

1. Open the case and the reconstruction being reviewed. Check the series,
   thickness, run history, and result revision before comparing counts.
2. Click a candidate in **Review cells**. The viewer opens its compartment mask
   in linked slice views. With **Highlight one selected cell** enabled, yellow
   identifies that candidate and the other visible candidates are dimmed.
   Swatches match the overlay. Colors distinguish IDs, can repeat, and do not
   represent confidence, correctness, or anatomical side.
3. Select the relevant candidate IDs, or move the crosshair to a suspected
   missed location and choose **Use crosshair location**. A captured location
   uses the CT's physical RAS coordinates. Photographs do not supply exact
   coordinates and are not automatically registered to the volume.
4. Choose an observation type and describe the discrepancy. **Save observation**
   records the note without changing any counts, masks, or cell revisions.
   Unsaved observations are guarded when navigating or switching runs. Save
   or discard a note before applying a count correction.
5. **Locate observation** opens the recorded point in slice views. The list
   shows observations for the displayed revision. **Export observations JSON**
   downloads all saved observations for the selected run, including their
   source and revision bindings.

Count-only candidates may have no visible mask. The air-filled overlay can also
omit locations within a compartment, so use **Cell compartments** to compare
physician marks. Original predictions remain available through the existing
revision-0 overlay and export. Diagnostic entry is shown alongside the latest
cell list; the API can bind a note to an exact historical revision.

## Storage and interpretation

Notes are append-only files under the protected app work root's
`diagnostic-observations/<hashed-run-id>/` directory, separate from count
artifacts and the SQLite database. Each note records its run, revision, source
hash, method and protocol versions, protocol snapshot hash, artifact hashes,
candidate IDs, optional physical point, text, and creation time. Reads and
exports verify the source and artifact bindings. Approved cases reject new
notes. No source paths or patient identifiers are added automatically.

These are observations made while viewing model output. They are not independent
reference annotations or proof of clinical accuracy. A possible merge still
needs expert confirmation of the individual cells and excluded neighboring
anatomy. The feature does not change segmentation, side assignment, candidate
filtering, reconstruction selection, or inference parameters, and does not
start a scan run.

All new automated checks use synthetic inputs. They cover high-ID highlighting,
background transparency, source and historical-revision binding, native affine
bounds, approved locks, invalid requests, note persistence, export, and
preservation of original counts, masks, and database rows. The browser test
exercises the viewer and note workflow in an isolated synthetic server.

Verification on 2026-10-05: 70 frontend tests, the production frontend build,
36 observation/counting API tests, and the synthetic browser workflow passed.
The broader app-environment Python run passed 709 tests and skipped 67 because
of optional runtime dependencies, external test-runtime configuration, or host
symlink support. Separately, 167 engine, grid, localization, proxy, and synthetic
wall checks passed in the existing CPU counting-test runtime with no skips.
The staged-source privacy guard passed. No clinical inference was run.
