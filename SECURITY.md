# PHI handling rules

- Treat every raw or derived medical image as PHI, regardless of metadata flags.
- Keep the raw DICOM root read-only and outside Git.
- Keep `%LOCALAPPDATA%\EthmoidComplexity` local, access-controlled, and outside synchronized folders.
- Never upload DICOM, NIfTI, Slicer databases/scenes, screenshots, converter logs, UID maps, metadata exports, or model outputs.
- Neutral filenames do not deidentify image pixels or embedded headers.
- Do not use Git LFS for medical data; LFS is still a commit and upload path.
- The staged-blob guard is defense in depth, not a substitute for the physical separation above.
- Missing `BurnedInAnnotation` metadata is not proof that pixels contain no identifiers. Visual review remains mandatory.

The pipeline fails closed on mixed acquisitions, irregular geometry, unsupported transfer syntax, localizer/derived images, burned-in-annotation flags, output collisions, converter warnings, multiple outputs, source hash changes, and failed affine/HU checks.

If any privacy or geometry check fails, do not continue to segmentation or external sharing.
