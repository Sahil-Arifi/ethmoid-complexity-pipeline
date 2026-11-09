# API-assisted native counter investigation

The API-assisted seed change did not improve the eight-study count result. It
passed the existing synthetic controls but lost both previously matched studies
and a provisionally located region in the native replay. It was rejected. The
application counter is unchanged.

## Comparison

An exact study match requires both supplied side counts to match. A matching
total with swapped or incorrect sides fails. Absolute side error is the sum of
the 16 absolute differences; smaller is better.

| Frozen method | Exact studies | Exact sides | Absolute side error |
| --- | ---: | ---: | ---: |
| Original baseline | 1/8 | 3/16 | 37 |
| Third ridge/enclosure variant | 1/8 | 3/16 | 35 |
| Revised seed rule, corrected implementation | 0/8 | 1/16 | 39 |

The two earlier rows match different studies. The revised rule preserved neither
match. All eight predictions repeated exactly in two fresh processes: arrays,
cell registries, metadata and compressed-array bytes matched. All 16 scheduled
calls completed without technical errors or withheld results. Repeatability
therefore passed; count improvement failed.

These are reused development studies with supplied counts, not an independent
accuracy estimate. The source studies and selected reconstructions are bound by
hashes, but exact correspondence with the narrated video reconstruction remains
unconfirmed. Partial interior annotations do not specify complete cell masks.

## Hypothesis and tests

With the user's authorization, an OpenAI API review examined four cropped CT
panels and allowlisted numerical functions. The panels omitted source headers,
names and acquisition identifiers. The review proposed separating the ridge's
seed role from its merge and enclosure roles. This was an engineering
hypothesis; the model's anatomical interpretation was not used as ground truth.

Three bounded variants were tested in order:

1. Removing ridge obstacles from the distance transform passed 9/12 fixed ridge
   controls. It failed the weak-sheet and two anisotropic-wall controls, so it
   stopped before native inference.
2. Using the original free-space distance for maxima, then clipping ridge voxels
   from those maxima, passed 11/12. Clipping split one connected plateau into
   extra markers and oversplit the weak-sheet control. It also stopped before
   native inference.
3. Removing the entire six-connected free-maxima component when it intersects a
   ridge passed 12/12. Air maxima were retained. Ridge-aware watershed, resolved
   interior distance, merge evidence, enclosure support and all cutoffs stayed
   fixed. This was the only variant replayed on the eight studies.

All three preserved the older 27 controls. The third variant also preserved the
96 spatial scenes exactly at markers, watershed, merged labels and accepted
labels: both methods passed 88/96, leaving the same eight failures. The shared
merger passed all 25 fragmentation controls with identical output. Passing this
synthetic gate did not predict native improvement.

An initial implementation of the third variant indexed with an unsigned-byte
mask returned by `h_maxima`. NumPy treated it as integer indexing. That attempt
is invalid evidence and remains preserved separately. The corrected variant
uses an explicit Boolean mask. The public helper rejects non-Boolean inputs,
and the source builder converts both free and air maxima to Boolean arrays.

## Native failure

The replay first reproduced the frozen third variant exactly from the original
HU crops, anatomical membership, exclusions, search regions, affine and midline.
The numerical closure was engine `0.2.0` at commit
`1e041f3823470f91eb45a07dd499d4187bd69f6c`; substituting current engine `0.2.3`
would test a different composition. Predictions were sealed before a separate
scorer loaded the supplied counts.

Relative to the third variant, none of the seven positive interior cores gained
missing voxels or fragmentation, and no additional explicit negative voxels were
accepted. Six of seven cores were complete and unfragmented; one remained
entirely outside the search region. Only one of three declared pairs remained
separated. The revised rule also lost all 63 voxels of a provisional video
locator. That failure vetoes promotion even apart from the worse counts.

Saved stages explain the locator loss: its former accepted basin expanded into
a larger merged region. Unsupported search-boundary contact rose from about
0.230 to 0.281 and failed the unchanged acceptance guard. Interior radius was
unchanged. The region did not lose its only seed: it belongs to a ridge-separated
free component with many remaining markers. Adding a fallback for unseeded
ridge-separated components would not repair this target, so that hypothesis was
discarded without another native run.

This exposes a proposal-generation problem: changing markers changes ownership
far from the ridge and can push an otherwise accepted region across a boundary
guard. Relaxing that guard would hide the failure. Supplied totals alone cannot
identify which of the other changed objects is an anatomical cell.

## Reproduction boundaries

The public research helper and source builder are separate from every
application counting method. The builder accepts only the normalized frozen
third-variant source SHA256
`2411b5d9b5220852e9ae26e537e28873a883cff799dd3ba7a7fbe72c32549b5d`.
It emits research source; it does not run a scan or select the old numerical
closure. The caller must retain that closure and the frozen crop provenance.

Original and failed attempts, source bindings, prediction seals, spatial
annotations and API accounting remain in protected external directories. No
patient input, crop, reference count, credential or generated graph is included
in this change. Further candidates need evidence that distinguishes a genuine
separator from a fold or spur and checks complete neighboring objects, rather
than selecting a rule because its totals fit these eight studies.
