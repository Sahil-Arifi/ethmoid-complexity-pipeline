# Ethmoid Air-Cell Annotation Protocol v2

| Field | Value |
| --- | --- |
| Protocol version | `ethmoid-air-cell-v2-draft.1` |
| Document status | `DRAFT_FOR_CLINICAL_REVIEW` |
| Training eligibility | `NOT_FOR_TRAINING` |
| Intended use | Research annotation-protocol development and feasibility testing |
| Current runtime protocol | `ethmoid-provisional-v1` (unchanged) |
| Clinical approver | Pending |
| Effective date | Pending |

> This draft is not a clinical definition and does not authorize model training. Every item marked `BLOCKING` must be answered and approved by the clinical team. Protocol v1 remains the only protocol currently loaded by the application.

## 1. How to read this document

Requirements are marked as:

- `CONFIRMED`: directly supported by the study-team response.
- `PROVISIONAL`: a proposed implementation choice that requires clinical approval. The machine configuration calls this `PROPOSED`.
- `BLOCKING`: an unanswered clinical decision. The machine configuration calls this `UNRESOLVED`. Engineers and annotators must not guess.

The companion [clinical review form](PROTOCOL_V2_CLINICAL_REVIEW.md) is the authoritative place to record answers, named roles, thresholds, and approval.

## 2. Purpose

Protocol v2 is intended to produce reproducible research annotations for:

1. the left and right ethmoid compartments;
2. every qualifying individual ethmoid air cell throughout the 3D CT volume; and
3. right/left manual cell counts at three approved coronal landmark planes.

These annotations are intended to support a future feasibility study of compartment segmentation, individual-cell identification, cell counting, measurements, and editable model predictions.

The protocol does not define a diagnosis, treatment recommendation, surgical-navigation product, validated clinical device, or composite complexity score.

## 3. Source evidence

This draft translates two study-team artifacts into an explicit protocol structure:

- `ML Component Questions.pdf`: proposed users, input, anatomy, correction workflow, outputs, and pilot objectives.
- `DICOM Annotated Scans.pdf`: nine completed examples of bilateral point counts at three coronal locations, plus one incomplete/template page.

The annotated examples are reference screenshots and counts. They are not full-volume contours, instance masks, DICOM annotations, or machine-readable 3D ground truth.

## 4. Definitions

### 4.1 Ethmoid compartment

A side-specific 3D region containing the qualifying ethmoid air cells. Its exact anatomical boundaries are `BLOCKING` pending clinical approval.

### 4.2 Individual cell instance

One qualifying 3D ethmoid air cell assigned one stable positive identifier across the case’s revision lineage. Retired identifiers are never reused. A split retires the original identifier and creates new identifiers; a merge retires the contributing identifiers and creates one new identifier. The exact clinical rule for deciding when spaces are one cell versus separate cells is `BLOCKING`.

### 4.3 3D unique-cell count

The number of qualifying individual-cell instances across the complete 3D volume. Each approved instance contributes exactly one count, regardless of how many slices it intersects.

### 4.4 2D landmark count

The number of qualifying cells marked on one approved landmark plane for one anatomical side. It is derived from the saved points on that plane.

Landmark counts and 3D unique-cell counts are different measurements. Counts from the three landmark planes must not be added together or treated as the total number of unique 3D cells because one cell may intersect more than one plane.

### 4.5 Adjudicated ground truth

A locked annotation created only after independent annotations have been compared, disagreements resolved by the named adjudicator, required integrity checks passed, and the result has been bound to the source-volume and protocol hashes.

## 5. Confirmed product requirements

- `CONFIRMED` DICOM is the primary source format.
- `CONFIRMED` Processing begins with one scan at a time and remains local.
- `CONFIRMED` Intended users are ENTs and anatomy-trained research assistants.
- `CONFIRMED` The bilateral ethmoid labyrinth and individual ethmoid air cells are the target of interest.
- `CONFIRMED` Left and right results must remain separate.
- `CONFIRMED` Users must be able to review and correct both the overall compartment and individual-cell results.
- `CONFIRMED` Intended outputs include NIfTI masks, CSV measurements, and representative image/report material.
- `CONFIRMED` Automated counts are intended to be compared with manual counts at middle-turbinate, anterior-ethmoid, and posterior-ethmoid locations.
- `CONFIRMED` The first dataset is a 10–20-scan feasibility pilot, not broad clinical validation.

## 6. Proposed annotation products

All products below are `PROVISIONAL` until clinical approval.

### 6.1 Bilateral compartment mask

- Geometry: exactly the same shape and affine as the source NIfTI volume.
- Data type: `uint8`.
- Values: `0` background, `1` left compartment, `2` right compartment.
- Left and right compartment voxels may not overlap.

### 6.2 Individual-cell instance mask

- Geometry: exactly the same shape and affine as the source volume and compartment mask.
- Canonical data type: `uint16`.
- Values: `0` background and a globally unique positive integer for each cell instance within the case revision.
- Numeric identifiers carry no anatomical meaning and must not encode side or anterior/posterior location.

### 6.3 Instance registry

Every nonzero instance identifier must have exactly one registry record containing:

- instance identifier;
- laterality: `left` or `right`;
- anterior/posterior classification: `anterior`, `posterior`, or `unassigned` until a divider is approved;
- review status: `draft`, `reviewed`, or `adjudicated`;
- an uncertainty flag, uncertainty reason, and reviewer notes when applicable; and
- reviewer identity or neutral reviewer code and revision provenance.

### 6.4 Landmark observations

Each observation must contain:

- station: `middle_turbinate`, `anterior_ethmoid`, or `posterior_ethmoid`;
- a plane origin and unit normal in NIfTI RAS millimetres;
- one 3D RAS coordinate per marked cell;
- laterality for every point;
- right and left counts derived from the stored points;
- source-volume hash, protocol version/hash, reviewer, revision, and timestamp.

A display slice number alone is not sufficient because it is not stable across reorientation or resampling.

### 6.5 Display default

The initial display uses a CT window center of `300 HU` and width of `3000 HU`. Reviewers may adjust the display without changing stored CT values or annotations.

## 7. Scan eligibility

Engineering eligibility and clinical suitability are separate decisions.

### 7.1 Engineering gate

A scan may enter the annotation workflow only when:

- the selected CT series passes the existing single-series preflight;
- DICOM-to-NIfTI geometry, orientation, and HU validation pass, or an external NIfTI has a valid matching QC report;
- the volume and its validation report pass their hash checks; and
- the volume has complete, readable geometry.

A failed conversion cannot be clinically approved or used for protocol-pilot annotations.

### 7.2 Clinical gate

A qualified reviewer must confirm:

- complete bilateral ethmoid-region coverage;
- correct orientation and laterality;
- sufficient image quality and slice continuity; and
- compliance with the approved clinical inclusion/exclusion rules.

Allowed reconstruction parameters and rules for age, disease, prior surgery, variants, artifacts, and incomplete coverage remain `BLOCKING`.

## 8. Annotation workflow

1. Import the scan and verify the source hash and engineering status.
2. Complete the clinical coverage, orientation, image-quality, and continuity checks.
3. Record an exclusion reason and stop if either eligibility gate fails.
4. Without seeing another reviewer’s annotation or a future model prediction, select each approved landmark plane.
5. Place one point for every qualifying cell visible on each side of the selected plane. Allow the application to derive the side-specific counts.
6. Trace the complete left and right ethmoid compartments in the 3D volume.
7. Assign every qualifying cell a unique instance identifier across the full volume.
8. Review axial, coronal, sagittal, and 3D views for missing cells, merged cells, split cells, boundary errors, and laterality errors.
9. Correct errors and assign laterality. Assign anterior/posterior metadata only after its anatomical divider is approved.
10. Mark uncertain or excluded structures with a reason rather than guessing.
11. Save an immutable draft revision bound to the source-volume and protocol hashes.
12. Submit the revision for independent review and adjudication.

Ground-truth creation must not expose either reviewer to the other reviewer’s work or to model predictions before the independent annotation is submitted.

## 9. Blocking clinical decisions

| ID | Decision required | Status |
| --- | --- | --- |
| `B01` | Exact rule defining one individual cell | `BLOCKING` |
| `B02` | Aerated lumen versus full anatomical cavity, including opacified contents | `BLOCKING` |
| `B03` | Communicating cells, ostia, incomplete septa, and uncertain boundaries | `BLOCKING` |
| `B04` | Exact six-direction outer compartment boundaries | `BLOCKING` |
| `B05` | Whether bone is only a boundary or also a separately labeled target | `BLOCKING` |
| `B06` | Whether anterior/posterior assignment is required and its exact divider | `BLOCKING` |
| `B07` | Inclusion and classification of named anatomical variants | `BLOCKING` |
| `B08` | Midline-crossing cell laterality | `BLOCKING` |
| `B09` | Disease, prior surgery, pediatric anatomy, artifact, incomplete coverage, and uncertainty-escalation rules | `BLOCKING` |
| `B10` | Reproducible selection rule for all three landmark planes | `BLOCKING` |
| `B11` | Meaning of each red dot and rule for partially visible cells | `BLOCKING` |
| `B12` | Mapping of reference examples to neutral source volumes and the incomplete tenth page | `BLOCKING` |
| `B13` | Named annotator, independent reviewer, adjudicator, and clinical approver roles | `BLOCKING` |
| `B14` | Required measurements, complexity-score decision, and predeclared pilot thresholds | `BLOCKING` |
| `B15` | Conversion approval and acceptance of the local browser interface as the desktop workflow | `BLOCKING` |

Answers must be recorded in the companion review form. A grouped blocker is resolved only when every field and subdecision under that ID has an exact answer. An unanswered, partially answered, or ambiguous decision prevents promotion to `APPROVED_FOR_PILOT`.

## 10. Automated integrity checks

Before a revision can be submitted for review:

- both masks must match the source CT shape and affine;
- the compartment mask may contain only `0`, `1`, and `2`;
- the instance mask may contain only integral values from `0` through the supported `uint16` range;
- every nonzero mask identifier must match exactly one registry record, and every included registry record must appear in the mask;
- every included instance must belong to exactly one side and lie completely inside that side’s compartment;
- landmark planes and points must be finite, inside the source volume, and bound to the source hash;
- derived side counts must equal the number of stored points for each plane and side; and
- protocol version/hash, source hash, reviewer, revision, and timestamp must be present.

A disconnected instance is flagged for clinical review rather than automatically rejected until `B01`–`B03` define the correct rule.

## 11. Clinical QC checklist

The reviewer must confirm:

- bilateral coverage and orientation are correct;
- the complete compartment boundary was reviewed in all three planes;
- every cell was checked for missed, merged, split, and incorrectly sided instances;
- landmark planes and point counts follow the approved rules;
- uncertainty and exclusion decisions are documented; and
- the revision contains no unresolved blocking issue.

## 12. Independent review and adjudication

- Two reviewers independently annotate the same initial 2–3 scans.
- Neither reviewer sees the other annotation or model output before submission.
- Both original revisions are preserved.
- The named adjudicator classifies disagreements as boundary, missed cell, split/merge, laterality, landmark-plane selection, landmark count, or eligibility.
- The adjudicator records the resolution and creates a separate locked adjudicated revision.
- If disagreement exposes an unclear rule, the protocol is revised and affected pilot scans are repeated. Existing revisions are never silently overwritten.

## 13. Measurements

The first approved measurements are proposed to be:

- 3D unique-cell count per side;
- landmark point count per station and side;
- total compartment volume per side; and
- per-cell volume.

Cell-volume distribution summaries may be added only after the clinical team selects the required statistics. Surface-area-to-volume ratios and a composite complexity score are excluded from Protocol v2 until their definitions, units, normalization, and validation criteria are approved.

Protocol-pilot evaluation must record, with thresholds supplied before annotation begins:

- landmark-count agreement;
- total unique-cell count agreement;
- compartment overlap or boundary agreement;
- per-instance missed-cell, false-cell, split, and merge errors;
- annotation and correction time; and
- uncertain or excluded case frequency.

## 14. Approval states and versioning

Approval follows this fixed round trip:

1. Complete the clinical review form.
2. Incorporate the exact approved language into a new draft revision of both the human protocol and machine configuration.
3. Validate both files, freeze their bytes, and compute their SHA-256 hashes externally.
4. Return the frozen version and hashes to all named reviewers.
5. Record acknowledgement from the annotator, independent reviewer, and adjudicator, followed by final approval from the clinical protocol approver.

Signing the questionnaire alone does not approve the protocol.

### `DRAFT_FOR_CLINICAL_REVIEW`

- All annotations are `NOT_FOR_TRAINING`.
- Protocol v1 remains active in the application.
- Clinical questions may be edited, but no answer may be inferred by engineering.

### `APPROVED_FOR_PILOT`

Requires:

- every `BLOCKING` item answered;
- named annotation, review, adjudication, and approval roles;
- predeclared pilot thresholds;
- explicit clinical signature and effective date; and
- a frozen protocol file and content hash.

Annotations remain `NOT_FOR_TRAINING` while the consistency pilot is evaluated.

### `APPROVED_FOR_TRAINING`

Requires:

- the 2–3-scan independent pilot meets every predeclared threshold;
- all disagreements are adjudicated;
- no high-impact clinical rule remains unresolved;
- technical integrity checks pass; and
- the clinical approver signs the frozen revision.

Any subsequent clinical rule change creates a new protocol version. Prior protocol files, annotations, hashes, and approval records remain immutable.

This draft file is never activated by changing its status in place. Approval produces a separately named, signed, content-hashed protocol file, followed by a separate application implementation and validation milestone.

## 15. Approval record

| Role | Name | Decision | Date | Signature or recorded approval |
| --- | --- | --- | --- | --- |
| Primary clinical annotator | Pending | Pending | Pending | Pending |
| Independent clinical reviewer | Pending | Pending | Pending | Pending |
| Clinical adjudicator | Pending | Pending | Pending | Pending |
| Clinical protocol approver | Pending | Pending | Pending | Pending |

| Frozen artifact | SHA-256 |
| --- | --- |
| Human protocol | Pending |
| Machine configuration | Pending |
