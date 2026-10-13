# Public annotated sinus CT resources

Checked October 12, 2026. This was public-source research only: no scans or annotation archives were downloaded, no authors were contacted, and no application or counting behavior changed.

## Question and result

V2 needs individual ethmoid pockets distinguished through a 3D volume, with left/right identity and the settled inclusion rules. A whole ethmoid-region mask cannot establish individual-cell count accuracy. A colored figure or selected slice does not establish availability of a complete scan and reusable annotation files.

No candidate met the user's conditions: free download, useful for this task without extra annotation or integration, and no author contact or access request. One new study describes separate 3D ethmoid-cell segmentation, but its data are request-only and whether the individual labels were retained in shareable files is unknown. Public CT-SCOPE and NasalSeg files exist, but their labels do not identify individual ethmoid cells.

## Resources and actual labels

| Resource | What the original source reports | Access and project fit |
| --- | --- | --- |
| [Developmental Morphometry of the Paranasal Sinuses, 2026](https://onlinelibrary.wiley.com/doi/10.1002/ca.70201) | Methods 2.6 describes separate segmentation of each air-filled ethmoid cell in 3D Slicer, followed by summing cell volumes/surface areas per side. The study includes 250 subjects aged 1-25 without sinonasal pathology. | Skip: request-only because of privacy/ethical restrictions. No download or reusable per-cell label files were verified. The protocol excludes inflammatory disease and measures aerated cavities, so it does not establish handling of filled cells. |
| [Infraorbital posterior ethmoid-cell study, 2025](https://link.springer.com/article/10.1186/s12880-025-01831-w) | Selected side-view CT images were exported as DICOM and outlined in Slicer by a radiologist, then checked by two senior radiologists. The target is one particular ethmoid-cell variant, not all ethmoid cells. | Skip: available on reasonable request and uses selected 2D images, not a complete 3D cell-count benchmark. [Figure 3](https://link.springer.com/article/10.1186/s12880-025-01831-w/figures/3) remains a public annotation example. |
| [CT-SCOPE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/), [author repository](https://github.com/BYO-UPM/Craneal_CT) and [Zenodo record](https://zenodo.org/records/15085103) | Original DICOM and PNG scans from 40 subjects; 13 subjects have 696 expert-reviewed slice masks of surrounding bone. Some other masks are automated pseudo-labels. | Skip: labels do not identify each pocket, so individual-cell references would need extra annotation. Zenodo restricts access, but the public author repository contains DICOM files and PNG masks, as documented in its [storage guide](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/data_storage.md). Nine annotated subjects have consecutive slices; four have selected nonconsecutive slices. |
| [NasalSeg original repository](https://github.com/YichiZhang98/NasalSeg), [paper](https://www.nature.com/articles/s41597-024-04176-1) and [download record](https://zenodo.org/records/13893419) | 130 scans with five manual classes: left/right nasal cavity, nasopharynx and left/right maxillary sinus. | Skip: free public download, but no ethmoid-cell labels. The original class list overrides broader descriptions in later papers. Creating the needed labels would require extra annotation. Dataset bytes and file geometry were not inspected. |
| [SinusSegment paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12645370/) and [repository](https://github.com/rheadkaul/SinusSegment) | 100 manually segmented CT scans; the target combines the sinonasal region, including air, soft tissue and bone. | Skip: original CT data are request-only and the combined mask does not provide individual-cell references. The public repository's training/testing folders contain README placeholders rather than scans or masks. |
| [HeadNeckBrainSpine sinus anatomy module](https://headneckbrainspine.com/Sinus-CT.html) | Educational labels for anterior/posterior ethmoid cells and named structures such as agger nasi, Haller cell and ethmoid bulla. | Skip as a dataset: reusable DICOM volumes and cell masks were not verified. The webpage can serve as an anatomy reference without a download or integration. |

## Screening outcome

No candidate was adopted. No author contact, access request, account registration, purchase, scan download or annotation work is part of this setup. The public papers and teaching page remain background references; they add no software dependency or prerequisite for engineering.

The access check corrected an earlier statement about CT-SCOPE: its Zenodo files are restricted, but the authors also publish files on GitHub. The complete public GitHub tree listed 8,416 DICOM files and 11,470 PNG files; the published storage guide distinguishes original CT, manual masks and pseudo-labels. These metadata checks do not verify geometry, patient de-identification or annotation correctness. NasalSeg's Zenodo API lists an open-access 224,005,800-byte archive under CC BY 4.0; its class list still lacks ethmoid labels.

Air-only, surrounding-bone and whole-region labels cannot establish accuracy for the project's individual-cell protocol, especially filled cells. Correctly matched expert evidence remains necessary to measure anatomical accuracy. This finding does not require exhaustive clinician annotation before bounded synthetic engineering checks.

## Appraisal and limits

The installed scientific-critical-thinking skill 1.5 was used to distinguish reported annotation methods, released files and applicability. Its citation is included here: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). The current arXiv record was checked and lists version 2.

This is a targeted search, not proof that no matching dataset exists. Checks covered primary-source descriptions, public repository inventories and access metadata; no scan or mask bytes were inspected. Restricted files, geometry, annotation correctness and suitability for the full protocol remain unverified. No private project material was supplied to literature services.
