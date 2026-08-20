# Manual 3D Slicer QC

Use pinned 3D Slicer 5.12.3 revision 34627. Perform this review locally; do not save a scene bundle, screenshot, DICOM database, or logs in Git.

1. Record reviewer/date, Windows architecture, the verified installer SHA-512, and Slicer's Help/About version and revision; they must report 5.12.3 revision 34627.
2. Create a dedicated temporary Slicer DICOM database under the PHI-sensitive local work root. Keep **Copy imported files** disabled.
3. Import only the selected neutral series and load its generated NIfTI.
4. Record dimensions, spacing, scalar type, origin/directions, IJK-to-RAS matrix, and physical bounds for both loads.
5. Link the volumes, overlay them, and fade between them in axial, coronal, and sagittal views.
6. Confirm left/right, anterior/posterior, and superior/inferior orientation; inspect the first, middle, and last slices.
7. Scroll the complete 224-slice stack while linked/overlaid. Confirm no translation, reflection, rotation error, truncation, duplicate/missing/reordered slice, blank slice, staircase artifact, or contaminating localizer/derived image.
8. Compare representative air, soft-tissue/water, and bone voxel values for display-level HU sanity.
9. The engineer records geometry/intensity/provenance pass or fail. A study clinician separately confirms the correct series and complete intended ethmoid-region coverage.
10. Close Slicer without saving an MRML/MRB scene. Remove the temporary DICOM database/cache under the approved local retention policy.

Any failure stops the project before segmentation work.
