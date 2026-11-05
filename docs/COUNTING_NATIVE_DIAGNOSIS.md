# Native proposal and rejection diagnosis

The native traces expose a limitation that the 132 procedural controls did not
test: two existing conditional different-cell pairs already share initial
watershed regions. A merge-only correction cannot separate those pairs. The
synthetic result remains **126/132**; the latest supplied-count comparison remains
**0/8 exact bilateral agreement**. This phase establishes no count improvement.

## Exact baseline reproduction

Four serial diagnostic replays of the unchanged
`native-bone-watershed-0.2.3` engine reproduced all four result JSONs and twelve
native NIfTI artifacts byte for byte. Arrays, registries, counts, coverage,
affines and qform/sform matched. All 1,430 frozen source, runtime, cache and input
pins remained unchanged; no new localizer ran. Independent stored-artifact
verification confirmed the results without repeating inference.

The existing eight-study evidence still contains fourteen available attempts,
two withheld attempts and no technical failures. Historical stage traces were
kept separate until the current replays reproduced their computational leads.
Exact per-study evidence and images remain outside this repository.

## What disappears where

| Observation | First observed stage | Consequence |
| --- | --- | --- |
| Two declared conditional pairs share raw basins | Initial partition | Merging alone cannot express their distinction. |
| A third conditional pair starts in separate basins and joins | Merging | Proposal granularity alone does not resolve this pair. |
| One existing core is entirely outside membership and search | Localization/search | Neither extra seeds nor a later acceptance change can recover it on the same free-space mask. |
| An in-region location joins an outside-heavy group, which fails membership | Merging followed by acceptance | The location is lost through the union, rather than missing from the initial search. |
| Saved versions subdivide previously accepted objects | Final objects; current interfaces traced | Count changes are reproducible, but their anatomical correctness is not established. |

The partial references are conditional development evidence. Their planning
reconstruction and whole-cell identities are not clinician-verified. Only three
different-core relations are declared; other pair relationships remain unknown.
These observations identify computational obstacles, not a list of confirmed
clinical misses or extra cells.

The exported `region` artifact includes accepted cavities and final side labels.
Its equality or difference cannot independently establish whether the original
localizer membership changed. The preserved comparison helper has an explicit
interpretation correction; the original evidence was not overwritten.

## Target-free proposal diagnostic

A separate crop-only experiment added deterministic markers using physical
1.5-mm bins and observed free-space distance. It retained every original marker
voxel and ID. Reference cores and target counts were used only after generation
for scoring. No merge, acceptance or full patient-count stage ran.

The original 8,192-marker operational gate stopped the experiment before the
fine watershed: it required 13,412 markers. A separately frozen budget-only
revision allowed 16,384 markers, with the generator and every selected winner
unchanged. It retained the 120-second / 2-GiB caps. The original gate remains
unmet; no density search or truncation converted that stop into a pass.

The diagnostic completed with 13,412 raw basins instead of 286. Foreground was
identical. All three declared pairs had disjoint positive raw-ID sets, with no
missing voxels in their six cores. Each core nevertheless occupied between two
and nine basins. The core outside free space remained unassigned. Also, 481 fine
basins overlapped multiple old basins: this is not a strict partition refinement.

This demonstrates that denser proposals can remove the two conditional raw
collisions on the same image/free-space data. It does not demonstrate valid
boundaries, accepted cells or improved counts. The result is not promoted.

A subsequent frozen merge-only replay used the unchanged separator policy on
those saved proposals. It reduced 13,412 raw groups to 6,531 graph groups while
preserving all three declared distinctions and identical foreground. One core
occupied a single group; the other five still occupied two to four groups each.
All 22,693 preserved constraints remained intact. These graph groups are not
accepted cells: acceptance and final counts were not evaluated.

An initial reporting loop also included a negative unknown annotation sentinel
as a pseudo-core. A separate saved-array scoring view now selects strictly
positive core IDs. Independent verification confirmed that all positive records
and declared-pair results were unchanged. The original output is retained; no
proposal or watershed reran for this reporting correction.

## Validation and remaining engineering work

The four baseline replays took 226.032 seconds in total, with recorded peak
committed process memory below the assigned 8-GiB limit. The fine-proposal worker
took 0.391 seconds; its subprocess took 1.609 seconds and peaked at
1,602,412,544 bytes. The merge-only replay took 65.094 seconds, with peak committed
memory 1,575,968,768 bytes under the same caps. Independent review checked frozen
pins, identical winner arrays, unchanged foreground, positive-only scoring and
all declared relations from saved arrays. It also reconstructed the complete
merge mapping from recorded unions and verified every preserved constraint.

The next implementation must preserve proposal distinctions while removing
unsupported fragments. The saved constraints blocking unions within individual
cores are the next diagnostic target. Separator identity still fails on six preserved
neck/fold controls. Localization/search loss and outside-heavy unions are
separate dependencies; their thresholds have not been relaxed. A raw proposal
success must pass merging, acceptance, existing controls and resource checks
before it can support another eight-study comparison.

Four private native review pages preserve the existing footprints and deferred
questions for the clinician. No new reference masks were drawn, and no outreach
occurred. Clinical uncertainty and unresolved engineering work remain separate.
The independent 48-geometry set remains unexposed to inference.

External evidence is grouped under `native-discrepancy-audit-01`: the baseline
traces and peer closure, object deltas and interpretation correction, reference
review packet, original proposal stop, budget-only revision, merge-only replay,
and their independent verification. Patient data and detailed artifacts stay local.
