# Public data and model source review

Initially checked **2026-09-18**, with data-quality and source follow-up on
**2026-09-19**. Scope: author publications, author repositories and data-repository
records for conventional CT with separate ethmoid and sphenoid region labels,
preferably individual ethmoid-cell instances or independent 3D counts. The
follow-up downloaded NasalSeg v2 locally for the quality checks below. No external
weights were adopted, authors contacted or patient data uploaded. Separate
procedural training experiments do not measure clinical accuracy.

**No immediately usable, publicly downloadable conventional-CT dataset with both required sinus labels was verified in this search. No public individual ethmoid-cell instance/count reference set was verified.** This is a bounded search result, not proof that no such resource exists. Regional sinus segmentation also does not establish correct individual-cell separation or counting.

## Most relevant access-request leads

These are ranked by anatomical and modality fit, not by published accuracy. Access, permitted use and trained weights remain unconfirmed unless stated otherwise.

| Priority and source | Modality, cohort and actual target | Access and licensing | Weights and relevance |
| --- | --- | --- | --- |
| 1. [Whangbo et al., Sensors 2024](https://www.mdpi.com/1424-8220/24/6/1933) and [Song et al., Scientific Reports 2025](https://www.nature.com/articles/s41598-025-17266-w) | Conventional CT; the 2024 study reports 198 patients and separate frontal, ethmoid, sphenoid and maxillary region classes. These are sinus regions, not individual ethmoid cells. | The 2024 paper's data statement says raw data are available from the authors on request. The 2025 study explicitly withholds public data for institutional privacy restrictions and allows reasonable requests. Article open-access terms do not provide a license to their unreleased images or models. | No downloadable study-specific weights were verified. Strongest lead for obtaining the two required CT region labels. Ask for original calibrated volumes, masks, label definitions, patient grouping and written data/model usage terms. |
| 2. [Ionescu et al., 2026 CT sphenoid study](https://www.mdpi.com/2075-1729/16/8/1284), [author preprint and availability statement](https://www.preprints.org/manuscript/202607.0262) | Conventional bone CT; 117 sphenoid development examinations and 17 reserved examinations. One sphenoid compartment class, including internal septa and sinus contents; no ethmoid label or cell instances. | Data are private; availability requires a reasonable request and data-provider permission. The [author repository](https://github.com/ionescucristian1/skullbase-pneumatization-ai-code_SANITIZED_CODE_ONLY_20260628) is explicitly code-only. No explicit code license was found in its nine-file tree. | No `.pt`, `.pth`, `.ckpt` or `.onnx` weights were present in that repository. Potential targeted sphenoid-localization collaboration, but currently neither training data nor a deployable pretrained model. |
| 3. [Kaygisiz Yigit et al., 2025/2026](https://academic.oup.com/dmfr/article/55/1/61/8213614) | 97 **CBCT** examinations, 0.25 mm acquisition; distinct frontal, sphenoid, maxillary and ethmoid classes. The authors explicitly did not separate left/right. Nine scans were held out for testing. | No public dataset download or data reuse license was verified on the reviewed author publication. | Describes nnU-Net v2, but no downloadable study-specific weights were verified. Possible author enquiry; CBCT-to-conventional-CT transfer and laterality would still need validation. This cannot supply cell-count ground truth. |

The 2024 request-only statement is also present in the publisher's [collected full-text volume](https://mdpi-res.com/bookfiles/book/9901/Deep_Learning_for_Pathology_Detection_and_Diagnosis_in_Medical_Imaging.pdf?v=1746752861). Some publisher pages intermittently returned maintenance or access errors; they were not treated as dataset downloads.

## Public resources with a different target

| Resource | Verified labels, modality and size | Current access | Code, data and model terms | Appropriate use and limitation |
| --- | --- | --- | --- | --- |
| [CT-SCOPE / Craneal_CT](https://github.com/BYO-UPM/Craneal_CT) | Conventional CT from 40 subjects across six hospitals/four scanners; 13 annotated subjects, 696 manually annotated slice/mask pairs. Target is osseous structures surrounding the sinuses, not separate ethmoid/sphenoid cavities or cell instances. | The [Zenodo record](https://zenodo.org/records/15085103) currently marks files **Restricted**; its [API](https://zenodo.org/api/records/15085103) returned `access_right: restricted` with no public files. Separately, the author's GitHub still exposes DICOM directories for 12 internal and 28 external subjects, plus PNG labels and pseudo-labels; file listings in both DICOM groups were verified. Full corpus byte size/download completeness was not verified. | GitHub [LICENSE.MD](https://github.com/BYO-UPM/Craneal_CT/blob/main/LICENSE.MD) says **CC BY-NC-ND 4.0**. The linked [Hugging Face model card](https://huggingface.co/ai4c/skull_segmentation) instead declares **MIT**. These conflicting code/model declarations require clarification before incorporation or redistribution. The restricted Zenodo API did not expose a license. | Possible bone/septum auxiliary study after terms and source alignment are settled. It does not provide required sinus-region or cell-count ground truth. |
| [NasalSeg](https://zenodo.org/records/13893419) | 130 conventional head CTs. Exactly five targets: left/right nasal cavity, nasopharynx, left/right maxillary sinus. **No ethmoid or sphenoid labels.** | Public `NasalSeg.zip`, **224,005,800 bytes** in the [official API](https://zenodo.org/api/records/13893419). The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11621807/) reports 1.5 mm median slice spacing. | Dataset metadata explicitly says **CC BY 4.0**. This dataset license is separate from any training implementation or weights. No target-relevant weights are supplied by this dataset record. | Nasal/maxillary localization research or additional CT examples for new annotation. It cannot supervise ethmoid/sphenoid directly, and its spacing does not establish visibility of individual thin septa. |
| [PARASIDE](https://github.com/Hendrik-code/paraside) | **T1-weighted MRI**, with left/right maxillary, frontal, sphenoid and ethmoid regions, each separated into air and soft tissue: 16 labels. The [published 2026 paper](https://doi.org/10.1016/j.compbiomed.2026.111511) reports 292 annotated participants; its older preprint used a smaller cohort. These are regional labels, not cell instances. | [Release v1.0.0](https://github.com/Hendrik-code/paraside/releases/tag/v1.0.0) exposes `Paraside_model_weights_v1.zip`, **761,262,790 bytes**, verified in [release metadata](https://api.github.com/repos/Hendrik-code/paraside/releases). No open paired MRI-image/mask corpus was verified; the [author preprint](https://arxiv.org/html/2501.14514v1) describes SHIP data-committee access. | Repository code is **Apache-2.0**. Release metadata does not separately state a weights or patient-data license; code licensing does not settle data access. | Useful label-schema and workflow reference. MRI weights must not be presented as a validated CT localizer or used to manufacture CT reference labels. |
| [SinusSegment](https://github.com/rheadkaul/SinusSegment) | Conventional CT workflow; its [training implementation](https://github.com/rheadkaul/SinusSegment/blob/main/training.py) sets `num_classes=1`. It produces one combined sinonasal foreground, despite the README naming multiple anatomical sinuses. | `CT_train` and `CT_test` currently contain placeholder README files, not a released cohort. `default_parameters.pth` is a public Git LFS pointer declaring **188,724,906 bytes**; its weight payload was not downloaded or executed. | No license file was present in the reviewed repository root. No dataset license or explicit weight license was verified. | Not an ethmoid/sphenoid multiclass model. A combined mask cannot be promoted to ethmoid-cell labels by renaming it. |

CT-SCOPE's [model documentation](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/available_models.md) links public weights. The linked final bone model is approximately **95 MB**. That document also states the final model was trained using all its dataset slices, so that model must not be evaluated on that same collection as an independent test. The [CT-SCOPE dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/) distinguishes nine annotated internal scans with consecutive slices from four annotated external scans with nonconsecutive slices; these are not 13 complete expert-labelled 3D sinus volumes.

## What these sources can support

1. **Obtain actual CT region supervision.** The first access-request lead is the closest modality/target fit for ethmoid and sphenoid localization, subject to confirming its exact label schema, laterality and usage permission. The sphenoid-only CT lead could address one localization component. No outreach has been sent.
2. **Use auxiliary public data only for its annotated task.** NasalSeg can support nasal/maxillary work; CT-SCOPE can support a bone-boundary investigation. Neither should be counted as ethmoid-cell validation, and model-generated masks must remain explicitly identified as predictions.
3. **Keep local engineering experiments separate from accuracy evidence.** Physical-geometry corrections, memory-bounded processing and sensitivity experiments can be tested without new clinical labels. A higher automatic completion rate does not establish more accurate cell counts.
4. **Require independent expert references for cell counting.** A future reference set needs source-aligned ethmoid/sphenoid regions, individual-cell instances or adjudicated per-side/total 3D counts, opacity/variant definitions and patient-level grouping. Multiple reconstructions of one patient belong in one evaluation partition.

No listed resource supports a claim that all 98 local volumes will yield anatomically correct automatic counts. No local clinical-accuracy claim follows from this review.

## Pretrained model availability and label identity

**No deployable pretrained checkpoint for the required conventional-CT ethmoid and sphenoid regions was verified in this bounded review.** A relevant paper, available weights, or an anatomical name in metadata is insufficient without matching label definitions, preprocessing, geometry and permitted use.

- **Legacy sphenoid project:** the [sphenoid paper](https://ceur-ws.org/Vol-4164/short11.pdf) links to the [author repository](https://github.com/Tomg79/Deep-Learning-Methods-for-Sinus-Segmentaiton). Its [legacy attention-U-Net loader](https://github.com/Tomg79/Deep-Learning-Methods-for-Sinus-Segmentaiton/blob/main/Legacy_Attempts/3D_Attention_UNet/src/dataset.py) resizes images to 512 × 512, pads or truncates depth to 32 slices and divides intensity by 255. Its [legacy YOLO dataset builder](https://github.com/Tomg79/Deep-Learning-Methods-for-Sinus-Segmentaiton/blob/main/Legacy_Attempts/YOLOv8_Segment/src/build_yolo_dataset.py) collapses labels into one nasal/sinus foreground. The newer nnU-Net README section supplies commands and reported results, but no matching checkpoint, plans or dataset-label JSON was verified in the reviewed tree; no explicit license was found. These legacy paths do not establish calibrated, source-aligned ethmoid/sphenoid inference.
- **NVIDIA NV-Segment-CTMR:** its [label dictionary](https://raw.githubusercontent.com/NVIDIA-Medtech/NV-Segment-CTMR/main/NV-Segment-CTMR/configs/label_dict.json) names labels 196/197 as ethmoid bone. However, its [training-label mapping](https://raw.githubusercontent.com/NVIDIA-Medtech/NV-Segment-CTMR/main/NV-Segment-CTMR/configs/label_mappings.json) links them exclusively to SegRap2 labels 15/16. The [official SegRap dataset definitions](https://segrap2023.grand-challenge.org/dataset/) and [official output-conversion code](https://raw.githubusercontent.com/HiLab-git/SegRap2023/main/Docker_tutorial/stacked_results_to_4d_mha.py) identify the corresponding `ETbone_L/R` targets as Eustachian tubes, not ethmoid sinus regions. This is evidence of a likely naming mismatch; the unpublished upstream subset conversion and unexamined model outputs prevent proving the actual pixel semantics. The official label map therefore blocks adopting these channels as ethmoid anatomy. Even correctly labelled ethmoid bone would not equal ethmoid cavity or individual-cell labels.
- **SAT:** the [published 497-class vocabulary, section 8.5](https://arxiv.org/html/2312.17183v4), contains no ethmoid, sphenoid or sinus target. Public [checkpoints](https://huggingface.co/zzh99/SAT) and text prompting do not establish trained support for an absent anatomical class. The [official inference documentation](https://github.com/zhaoziheng/SAT#inference-guidance-command-line) reports approximately 24 GB GPU memory for SAT-Nano and 34 GB for SAT-Pro at patch batch size one, both above this workstation's 16 GB. These are the authors' documented configurations, not a locally measured impossibility of all smaller-memory variants; no suitable target-specific checkpoint or clear weight license was verified.

No dataset or weight payload was downloaded or executed for this availability review. Model-generated regions would remain predictions requiring source-geometry checks and independent expert evaluation, not cell-count ground truth.

## Anatomical localization methods worth investigating

Additional targeted checks did not supply the missing references. The [2026 ethmoid CBCT biometric study](https://pubmed.ncbi.nlm.nih.gov/41864172/) describes 565 examinations and gender classification; its classification F1 is not segmentation or counting accuracy. No public 3D mask/checkpoint release or reuse terms were verified, and inaccessible full methods prevent assuming a particular mask format. The [2026 hybrid CNN-transformer paper](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2026.1778260/full) links only to CT-SCOPE and does not expose its claimed four-compartment masks or weights. The original dataset's bone targets do not establish those additional region labels.

The [University of Washington skull-base atlas](https://digital.lib.washington.edu/researchworks/items/2b0f0e1e-ae81-4bf8-9b79-8dc6872b5ecc/full) is publicly downloadable under CC BY-NC-SA 3.0 US. Reading only the first 32 KB of the official segmentation response verified 14 segment names, including eyes, optic nerves, skull-base bone and cavernous sinus, but **no ethmoid or sphenoid paranasal sinus region**. No CT payload was downloaded. Cavernous sinus is a different anatomical structure and cannot substitute for sphenoid sinus. Thus this atlas does not directly provide the missing target masks.

The current sphenoid heuristic has two separate constraints: it accepts only the first anterior bone hit with a predominantly anterior-facing gradient, then requires one global quadratic to explain at least 70% of supported rays. A more flexible surface cannot repair absent or incorrectly selected bone evidence.

An anatomical study of 129 patients found that a 30-degree anteriorly tilted coronal CT view improved anterior sphenoid-wall visualization ([Hemmi et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39146618/)). Different wall configurations associated with Onodi anatomy are also documented ([Nomura et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26097234/)). These support investigating orientation-aware wall proposals; neither paper validates an automatic cell counter or supplies target masks for this collection.

Graph surface optimization can combine observed image evidence with explicit smoothness constraints ([Li et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122/)). Local image-guided surface refinement has also been studied for sinonasal anatomy ([atlas/snake study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5725758/)). Neither method has been implemented or evaluated here. Any experiment must independently require observed supporting bone, bound unsupported interpolation, reject competing anatomical surfaces and preserve native final cell decisions. Optimization always finding a mathematical solution is not anatomical evidence.

Useful additional tests would cover steep and asymmetric walls, internal septa before the true boundary, missing walls, competing compartments, opacity and thick-slice averaging. Additional diagnostics should separate missing bone, gradient-direction rejection and insufficient spatial support. These are proposed engineering experiments, not a reason to reduce the existing rejection thresholds or declare unresolved anatomy correct.

## Additional geometry and training-method review

The subsequent bounded search on 2026-09-18 found two additional geometry resources:

- [MIDA](https://itis.swiss/virtual-population/regional-human-models/mida-model) is
  a 0.5 mm MRI-derived digital head atlas. Its [official tissue list](https://itis.swiss/assets/Downloads/VirtualPopulation/MIDA1-0_tissue-list.txt)
  explicitly separates ethmoidal and sphenoidal internal air. The original MRI
  images are not released. Its [access form](https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-0)
  requires name, email and affiliation; no model payload was downloaded or size
  verified. The [current license](https://itis.swiss/assets/Downloads/VirtualPopulation/License_Agreements/LicenseAgreementMIDA_2024.pdf)
  restricts redistribution of model data and derivatives. Access and permission
  for the intended training/weight-distribution use remain unconfirmed. This is a
  possible labeled-geometry lead, not an acquired conventional-CT reference set.
- [NIH3D upper respiratory airways cast](https://3d.nih.gov/entries/3DPX-002305)
  is MDCT-derived and explicitly CC BY 4.0. The [download page](https://3d.nih.gov/entries/download/2305/2)
  exposes a 4,758,152-byte GLB and 9,920,484-byte STL. A bounded 1,344-byte GLB
  header/JSON read verified one combined mesh without ethmoid/sphenoid part labels.
  It may support auxiliary synthetic geometry; it cannot supply those region or
  individual-cell labels. The full mesh was not downloaded.

Additional [sphenoid code](https://github.com/Sallycube/BIKO-UA-Sphenoid-Sinus-Segmentation)
explicitly withholds the CT data and exposes no verified checkpoint/license.
[Another repository](https://github.com/astronomer2912/sphenoid_sinus_project)
contains code without a verified dataset/model release or license.
[SinusSegmentation](https://github.com/niyaziacer/SinusSegmentation) is an
MIT-licensed manual-seed region-growing implementation, not a pretrained model.
[BodyParts3D's v4 vocabulary](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_parts_list_e.txt)
contains ethmoid/sphenoid bones, not the required sinus compartments.
[AutoSeg4SinonasalCT](https://github.com/YuliangXiaoYLX/AutoSeg4SinonasalCT) concerns
septum, inferior turbinate, maxillary sinus and optic nerve; it does not provide
the required ethmoid/sphenoid targets.

[SynthSeg](https://github.com/BBillot/SynthSeg) demonstrates segmentation training
using randomized synthetic images generated from anatomical segmentations. It
still needs meaningful target segmentations. Its released brain model, fixed
1 mm outputs and CT brain-intensity preprocessing are not an ethmoid-cell model.
The [CVPR 2022 self-supervised Swin pretraining study](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf)
supports learning general image features before downstream supervised tasks;
pretraining does not establish missing anatomical label semantics or count truth.
Neither model's weights were adopted here.

Following the surface-method review above, an isolated
[procedurally supervised wall experiment](SYNTHETIC_WALL_TRAINING.md) is now
implemented for testing. It does not change the existing anatomical methods or
treat a graph solution as truth. No external data request, registration or author
message was submitted. The lack of an acquired target-reference dataset remains
a separate constraint on clinical accuracy evaluation.

## NasalSeg v2: downloaded quality review, 2026-09-19

The [official record](https://zenodo.org/records/13893419) provides an open
CC BY 4.0 archive. Downloaded size **224,005,800 bytes** and MD5
`ba83df47974d907798542101fa43ac7d` matched the official metadata. SHA-256 is
`60c6facf843685802c39e4adff4a05c081c1c4b6175c9cb573745c55abb0fa6a`.
The original ZIP and all extracted files stay outside Git.

The [dataset paper](https://www.nature.com/articles/s41597-024-04176-1) describes
human annotation, model-assisted refinement and expert review. Its exact targets
are right/left maxillary sinus, right/left nasal cavity and nasopharynx, numbered
1–5; zero is background. There are **no ethmoid, sphenoid, wall or individual-cell
reference labels**. The author repository contains no verified trained checkpoint.

Full local inspection decoded 130 image/label NRRD pairs. Files contain cropped
volumes rather than verified original full-head DICOMs. All masks contain labels
0–5 and CT values range from -1024 to 3071, but this range does not prove preserved
HU calibration or correspondence to original acquisition coordinates.

Measured findings:

- **12 pairs have substantive physical geometry disagreement.** Nine label
  headers reverse the Z direction and differ by 220.5 mm in origin; three use
  1 mm label spacing against 1.5 mm image spacing, producing 25.5–26.5 mm corner
  displacement. Matching array dimensions do not resolve these mismatches.
- **107 distinct decoded image/label contents exist among 130 records.** Nineteen
  repeated-content groups account for 42 records, or 23 redundant copies. Distinct
  contents are not proof of independent patients; differently reconstructed
  images from one patient could still have different arrays.
- The 118 geometry-matched records contain 97 distinct contents. Three additional
  nasal-centroid ordering flags are copies of one content, separate from the
  geometry failures. Such centroid ordering requires review; it is not proof that
  labels should be swapped. Removing that flagged content leaves 96 distinct
  geometry-matched contents without this particular flag.

No original was modified, reoriented or relabeled. These checks do not grant
clinical annotation validity. A future auxiliary experiment must freeze source
hashes, exclude unresolved geometry, group repeated content before splitting,
establish patient grouping, and verify crop/calibration and label semantics.
No training or inference on this dataset has occurred.

An independent standard-library NRRD header/gzip parser plus NumPy reproduced
all 130 pair checks without relying on the initial imaging library. It verified
archive hashes, ZIP CRCs, all extracted bytes, voxel-content groups and physical
extent disagreements. Audit SHA-256:
`d7e36295a00267b162806b9a53b36f17ab0f108482fc5c2d748ad7bd6ad77299`.
The immutable initial file-byte uniqueness report is superseded, for duplication
conclusions, by the decoded-voxel identity report and independent audit.

A bounded [v1 release](https://zenodo.org/records/12177181) comparison fetched
only one same-named full-head image/label pair and archive metadata (about
32.7 MB), not the full 4.4 GB release. Its opposite Z storage directions describe
the same physical extent through a lossless index reversal. The current v2
arrays retain those respective full-head headers after changing array size,
so their physical extents no longer match. The v2 image was not an exact direct
subvolume of that same-named v1 image. Filename correspondence therefore cannot
establish source identity, explain preprocessing or justify a geometry repair.
Original files from both releases remain unchanged.

Its relevance was checked against the actual v5 failures. Of the 66 sphenoid
outcomes, eight lack an eligible posterior compartment, four have competing
compartments, 28 lack enough oriented wall rays, 18 have incoherent wall fits,
six have spatially narrow support and two have degenerate fits. Thus **54/66
already reach wall analysis with a parent and interior seed**. Nasal anchors may
help investigate the first 12, but no current reference comparison establishes
nasal-mask error. The dataset's maxillary labels serve a later stage, and
nasopharynx is not currently a localizer input. Training these auxiliary labels
would not directly supervise the demonstrated sphenoid-wall problem.

Two additional primary-source leads do not resolve the target-data gap. The
[postmortem fluid study](https://www.nature.com/articles/s41598-026-44094-3)
segments sphenoid fluid rather than the complete compartment and explicitly
withholds public CT images. The [orbital-apex study](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2025.1609028/full)
uses 200 selected enhanced-CT images for sphenoid/optic-canal/carotid segmentation;
no downloadable complete CT/mask corpus or checkpoint was verified. Neither
provides individual ethmoid-cell ground truth.
