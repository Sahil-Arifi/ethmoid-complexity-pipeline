# Protocol v2 Clinical Review Form

This form contains only the decisions still needed before `Ethmoid Air-Cell Annotation Protocol v2` can become `APPROVED_FOR_PILOT`.

| Field | Value |
| --- | --- |
| Protocol version | `ethmoid-air-cell-v2-draft.1` |
| Document status | `DRAFT_FOR_CLINICAL_REVIEW` |
| Training eligibility | `NOT_FOR_TRAINING` |

For every item, select one decision and provide exact replacement language when requested. `Unresolved` keeps the protocol blocked. A grouped blocker such as B04 or B09 is resolved only when every table cell and subquestion under that ID is complete. Please do not answer with “use best judgment”; the annotation rule must be reproducible by another reviewer.

## Reviewer and approval roles

### B13 — Named roles

| Role | Name and credentials | Responsibility |
| --- | --- | --- |
| Primary clinical annotator |  | Creates one independent annotation |
| Independent clinical reviewer |  | Creates a second independent annotation |
| Clinical adjudicator |  | Resolves disagreements without deleting either original |
| Clinical protocol approver |  | Approves the frozen protocol and pilot thresholds |

## Anatomy and individual-cell rules

### B01 — What counts as one cell?

Write the exact rule that separates one cell from another:

**Decision:**

### B02 — What tissue or space is outlined?

- [ ] Aerated lumen only
- [ ] Full anatomical cell cavity, including opacified contents
- [ ] Other:
- [ ] Unresolved

Clarify how partially and completely opacified cells are recognized:

**Decision:**

### B03 — Openings and incomplete boundaries

Define how to treat communicating spaces, ostia, incomplete septa, very thin septa, partial-volume boundaries, and uncertain separation:

**Decision:**

### B04 — Outer compartment boundaries

Provide a reproducible rule for each boundary:

| Direction | Approved anatomical boundary |
| --- | --- |
| Superior |  |
| Inferior |  |
| Medial |  |
| Lateral |  |
| Anterior |  |
| Posterior |  |

### B05 — Bone labeling

Proposed default: bony septations define cell boundaries but are not a separate output mask.

- [ ] Approve proposed default
- [ ] Require a separate bone mask
- [ ] Replace with:
- [ ] Unresolved

### B06 — Anterior/posterior classification

- [ ] Required for every cell
- [ ] Optional metadata only
- [ ] Not required
- [ ] Unresolved

If required or optional, state the exact dividing landmark and how to classify a cell crossing it:

**Decision:**

### B07 — Anatomical variants

Mark each structure `include`, `exclude`, or `separate category`, and state how it is recognized:

| Structure | Decision and rule |
| --- | --- |
| Agger nasi cell |  |
| Haller cell |  |
| Onodi cell |  |
| Frontal cell |  |
| Supraorbital ethmoid cell |  |
| Concha bullosa |  |
| Other relevant variants |  |

### B08 — Laterality near the midline

Define how a cell crossing or touching the midline is assigned to left or right:

**Decision:**

## Scan eligibility and difficult cases

### B09 — Include, exclude, or escalate

For each condition, state `include`, `exclude`, or `escalate`, followed by the annotation rule or exclusion threshold:

| Condition | Decision and rule |
| --- | --- |
| Partial opacification |  |
| Complete opacification |  |
| Prior sinus surgery |  |
| Severe inflammatory disease |  |
| Distorted anatomy |  |
| Pediatric scan |  |
| Motion artifact |  |
| Dental artifact |  |
| Incomplete ethmoid coverage |  |
| Inadequate slice thickness or reconstruction |  |

State any minimum or maximum acceptable CT acquisition/reconstruction requirements:

**Decision:**

Define the uncertainty-escalation process: who receives an uncertain cell or case, which final dispositions are allowed, who closes the decision, and when the entire case must be excluded.

**Decision:**

## Landmark counting rules

### B10 — Plane selection

For each station, define exactly how a second reviewer selects the same coronal plane. Name the anatomical landmark and state how to resolve an ambiguous or oblique scan.

| Station | Exact plane-selection rule |
| --- | --- |
| Middle turbinate |  |
| Anterior ethmoid |  |
| Posterior ethmoid |  |

Is “middle turbinate” itself a scored station or only a positioning landmark?

**Decision:**

### B11 — Meaning of the red points

Confirm what one red point represents and whether counts use only the selected slice, a slab, the maximum count across nearby slices, or another rule. Define how much of a partially visible cell must appear before it is counted.

**Decision:**

### B12 — Reference examples

- [ ] The nine completed examples can be mapped to neutral source-volume labels and exact physical plane locations.
- [ ] The examples are illustrative only and will not be used as machine-readable reference labels.
- [ ] Other:

The incomplete tenth page is:

- [ ] A template only
- [ ] A missing case that will be completed
- [ ] Excluded for another reason:
- [ ] Unresolved

## Measurements and pilot acceptance

### B14A — Required first-version outputs

Proposed first-version outputs:

- 3D unique-cell count per side;
- landmark count per station and side;
- total compartment volume per side; and
- per-cell volume.

- [ ] Approve all proposed outputs
- [ ] Remove or change:
- [ ] Add:
- [ ] Unresolved

Confirm that Protocol v2 must omit a composite complexity score:

- [ ] Yes
- [ ] No; attach the exact approved formula, inputs, units, weights, and normalization
- [ ] Unresolved

### B14B — Predeclared pilot thresholds

Enter the required statistic and pass threshold before the first 2–3 scans are annotated:

| Evaluation | Statistic | Pass threshold |
| --- | --- | --- |
| Landmark counts by station and side |  |  |
| Total unique-cell counts by side |  |  |
| Bilateral compartment agreement |  |  |
| Missed/false individual cells |  |  |
| Split/merge errors |  |  |
| Boundary disagreement |  |  |
| Annotation or correction time |  |  |

Who confirms that the pilot passed?

**Decision:**

## Workflow approvals

### B15 — Conversion and application

Does the submitted conversion review demonstrate correct anatomy coverage, orientation, slice continuity, and DICOM/NIfTI agreement for the pilot case?

- [ ] Approved
- [ ] Needs changes; specify:
- [ ] Not reviewed

Is the application opening in a browser at `127.0.0.1` acceptable when all processing and files remain on the same local computer?

- [ ] Yes
- [ ] No; a native packaged interface is required
- [ ] Needs demonstration before deciding

Nonblocking product preference: should direct NIfTI import remain available in addition to DICOM import?

- [ ] Yes
- [ ] No
- [ ] Unresolved

## Final clinical decision

- [ ] All blocking decisions above are complete and internally consistent; the answers may be incorporated into a new draft revision.
- [ ] The review remains blocked; unresolved item IDs:

After this form is completed, engineering must incorporate the exact answers into both protocol artifacts, validate them, freeze their bytes, compute their SHA-256 hashes, and return those exact files for sign-off. Signing this questionnaire alone does not approve Protocol v2.

| Field | Response |
| --- | --- |
| Frozen protocol version |  |
| Human protocol SHA-256 |  |
| Machine configuration SHA-256 |  |
| Additional conditions |  |

| Role | Name and credentials | Decision | Date | Signature or recorded approval |
| --- | --- | --- | --- | --- |
| Primary clinical annotator |  |  |  |  |
| Independent clinical reviewer |  |  |  |  |
| Clinical adjudicator |  |  |  |  |
| Clinical protocol approver |  |  |  |  |
