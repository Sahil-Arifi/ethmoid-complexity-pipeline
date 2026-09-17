# Local study catalog

The Cases screen groups the configured local DICOM collection by study. Every
series remains selectable, including scouts and sources that cannot form a valid
3D volume. Supported series are prepared in the background, with the recommended
series first. Preparation converts and verifies images; counting begins only
after the user clicks Run.

## Storage and identity

Schema 3 adds catalog sources, studies, series, immutable case links, and durable
preparation jobs. Existing case IDs, annotation revisions, count runs, cell
corrections, reviewer notes, and saved viewer state are retained. Before migrating
an existing schema 1 or 2 database, SQLite writes a consistent protected backup
named `schema-v<version>-<random>.sqlite3` beside the original database. The backup
is local medical application state and must never enter Git.

The protected inventory UID map identifies studies and series. Neutral labels
remain stable after the initial assignment, even if subsequent inventory discovery
orders change. Legacy display labels are never identity evidence. An old imported
volume becomes the current prepared series only after its immutable stored NIfTI
hash, paired QC report, exact raw DICOM SHA256 multiset, geometry, and every voxel's
HU mapping are verified. The ten frozen cohort conversions can be reused using the
same checks; their files are copied into protected application storage. Clinical
review states are not promoted by these engineering checks.

Source changes create a new current-volume identity after preparation. Old case
IDs and their histories remain accessible in the retained imports area. Starting
a new count requires the linked case to be the current ready series, the raw
source file hashes to match, and the stored volume hash to match. Historical result
access and export do not require the old source collection to remain current.

## Configuration and startup

The default application root is `%LOCALAPPDATA%\EthmoidComplexity\work\webapp`.
Its source configuration lives in the protected sibling `catalog.local.json`.
On first startup, the application derives this configuration from the existing
protected `inventory/uid_map.local.json`. No source path or UID is sent to the
browser. Custom application roots never implicitly load the production catalog.
Source, work, and Git directories must be physically separate.

Startup first serves the saved catalog, then synchronizes inventory in a background
thread. A stat/inventory signature change invalidates active reuse. An unchanged
ready volume retains its state after its cached NIfTI hash is checked; startup
does not reconvert it or repeat its full HU comparison. Raw DICOM hashes are
checked again before a new counting run. Pending and interrupted jobs resume.

Exactly one preparation process owns the protected file lock. SQLite job claims
are transactional, and repeated requests cannot create duplicate active jobs.
An explicit request promotes an already queued series; it does not restart a
running conversion. Completed outputs use immutable per-case and per-attempt
directories. The worker verifies source identity again before promoting a result.
Disk checks reserve 2 GiB plus a conservative conversion estimate. Transient
failures require an explicit retry, preventing an endless background retry loop.
Preflight failures retain a protected `preflight-failure.local.json` with fixed
issue categories. Browser reasons distinguish mixed acquisitions, inconsistent
orientation, duplicate/missing planes, irregular spacing, HU calibration, encoding,
and missing metadata; raw diagnostic strings and tag values are not exposed.

The command below performs the same catalog preparation without a browser. It
prints aggregate status only and waits until queued work is finished. Run it from
the repository with the existing application Python environment:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_local_catalog.py
```

Use `--catalog-only` to register/link existing verified images without preparing
remaining series. An explicit `--work-root` supports an isolated synthetic work
root with its own inventory. `--timeout-seconds` stops waiting after that duration;
an interrupted preparation is retried in a new immutable attempt on next startup.
Stop the application and confirm no active jobs before a production database
migration or restoring a backup.

## Series eligibility and recommendation

Preparation uses the converter's `exploratory` preflight profile. A consistent,
calibrated native CT or pure MPR stack may be prepared, including an orthogonal
oblique acquisition. Scouts, projection/averaged images, mixed planes/acquisitions,
shear, irregular spacing, unsupported decoding, and failed voxel/affine checks
remain unavailable with no numeric count. Broader preparation does not relax the
strict profile used by the original frozen evaluation.

`within_protocol` requires a native source, source SliceThickness at most 1 mm,
and all three native voxel intervals at most 1 mm. SliceThickness and distance
between slice positions remain distinct fields. Larger or reformatted acquisitions
are `outside_protocol` and carry an exploratory warning. Missing resolution is
`unknown`, never silently within protocol. Engineering preparation is not evidence
of anatomical coverage, correct cell counts, or clinical suitability.

### Bounded MPR coordinate quantization

The explicit `exploratory-mpr-quantized-grid-v1` policy applies only to exploratory
pure MPR sources whose IPP coordinate differences lie on a 0.001 mm lattice
(maximum lattice residual 0.0000001 mm). Some translated reconstructions have
many decimal places in their DICOM strings despite an earlier 0.001 mm rounding
step; this policy records measured differences rather than claiming decimal-string
precision. The original source records and pixel values remain unchanged.

The policy fits the positive interval along the fixed unit IOP normal and an
origin. In-plane axes and pixel spacing remain fixed; no shear or resampling is
introduced. Every original DICOM corner must be within **0.001 mm Euclidean**
distance of the fitted grid. A source above the ceiling remains unavailable.
Median adjacent IPP interval, fitted interval, measured lattice residual, physical
ceiling, and observed maximum error are recorded separately in protected QC and
conversion provenance. Catalog acquisition spacing retains the observed median;
the fitted volume spacing and policy are separate descriptor fields.

Both original NIfTI forms must describe the same discrete signed-axis permutation,
dimensions, corner assignments, and complete consecutive planes as the fitted
grid. The converter changes header geometry only and checks encoded pixel bytes
remain identical. Both reloaded forms must still meet the 0.001 mm ceiling.
Verification and catalog reuse prove every source voxel maps exactly once to a
native voxel, then compare all calibrated HU values. Duplicate planes cannot pass
by having identical intensities. The strict profile and original frozen v1
tolerances do not use this policy. A physical-limit failure has a specific
nonretryable geometry reason, not a generic transient converter error.

Recommendation tiers are thin native sharp-kernel images, then other thin native
images, then broader exploratory sources. Within a tier, smaller slice interval,
larger physical coverage, and stable series label determine the order. Ready status
does not outrank a better source; unavailable sources are not recommended.
Automatic selection follows the current best usable recommendation; an explicit
user selection is retained.

## API contract

- `GET /api/studies`: grouped studies, aggregate catalog status, and retained or
  unlinked legacy cases.
- `GET /api/studies/{study_id}/series`: all series in the study.
- `PATCH /api/studies/{study_id}` with `selected_series_id`: save explicit choice.
- `POST /api/studies/{study_id}/series/{series_id}/prepare`: queue or promote an
  eligible preparation; `retryable` distinguishes transient failures.
- `GET /api/catalog/status`: aggregate indexing/preparation status.

Existing `/api/cases` routes remain supported. Case detail includes a `catalog`
acquisition snapshot when a verified identity link exists. Study/series IDs are
opaque random hex strings. Responses contain neutral labels, acquisition quality,
preparation/count states, and fixed human-readable failure messages, never UIDs,
patient names, dates, raw file names, protected paths, or converter tracebacks.

## Verification

`tests/test_catalog.py` checks additive migration backup, identity versus labels,
source-change history preservation, stable grouping/selection, safe API output,
and rejection of unrelated voxels with copied source hashes.
`tests/test_catalog_queue.py` checks durable retry, explicit priority promotion,
single worker ownership, interrupted recovery, unchanged ready startup, exact
cohort reuse, source rehash before new counts, and a pinned real dcm2niix conversion
whose geometry and every HU voxel are verified with no inference jobs created.
Tests use synthetic images in isolated temporary roots.
