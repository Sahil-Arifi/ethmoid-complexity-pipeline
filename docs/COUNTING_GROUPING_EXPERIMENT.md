# Current-group wall ownership experiment

This experiment changes numerical grouping. A bone witness supports a separator
only when its two sampled free-space endpoints belong to the two current
groups. Evidence is recomputed after each union. Previously supported
constraints remain in force. The support settings, proposal seeds, anatomical
membership and acceptance rules stay fixed.

The existing dynamic prototype repaired fragmented-wall and open-fold controls
that defeated static ownership checks. It still failed four no-septum neck
controls. The new implementation must preserve that counterevidence. It uses
all in-radius witnesses, rather than a nearest-witness approximation, and
physical interface area and component marker anchors for deterministic ties.
An index may accelerate evidence calculation; it must match the direct
pair-specific distance-transform calculation on the same partitions.

## Evaluation decision

The candidate first runs the preserved 96+36 full compartment-stage spatial
controls, the fragmented true-wall controls and representation checks. Original
oracles remain unchanged. Require no regressions among formerly passing
controls and actual changed accepted partitions before a native comparison.
Retain every failure; synthetic success is not a clinical result.

This explicitly amends the earlier repair plan's research sequence: a bounded
eight-study development comparison may evaluate a partial candidate that passes
the preceding non-regression and resource checks, even while known controls
still fail. Its purpose is to measure whether the numerical change improves
supplied-count agreement. It does not pass the promotion gate, which still
requires resolving known failures and independent geometry validation.

Before that comparison, freeze the complete candidate, source/runtime, original
localizers, all eight inputs and evaluation criteria. Targets stay outside
inference. Run twice in fresh processes; report every study, withheld outcome,
bilateral and side agreement, signed errors, coverage and native-artifact
repeatability against both preserved baselines. Do not tune the candidate from
those outcomes or omit failures. These studies are an exposed development set.

Exact bilateral agreement is the primary outcome. Also report exact sides and
total absolute side-count error on the same available sides. Lower error with
unchanged bilateral agreement is a secondary count improvement, not additional
correct studies. Report newly available or newly withheld studies separately so
a changed denominator cannot create an apparent improvement. Repeat agreement
and passing software tests do not count as accuracy improvement.

The 48 independent geometries remain unexposed until a candidate is ready for
that validation gate. Original results and conditional native-reference limits
remain binding. No new numerical method is installed in the application by
this experiment.

## Verified implementation

The experimental `experimental_dynamic_owned` module implements this policy;
application defaults still use the existing merger. Across the original 132
full-stage controls it passes **128/132**, versus **124/132** for baseline, with
zero regressions. Four accepted spatial partitions are repaired; the other 128
accepted arrays, all markers and all raw watershed arrays are byte-identical.
The four no-septum neck failures remain failures.

All **864 representation checks** and **25 fragmented-wall/permutation controls**
pass. A separate reviewer recomputed the saved spatial oracles, inverse-reindexed
the representation outputs, and verified 435 input/source pins and 2,070 output
hashes. All eight workers stayed within their declared 120-second and 2-GiB
limits. These results satisfy the bounded research prerequisite above; they do
not satisfy the promotion gate.

The repository's numerical CI suite passes **456 tests**, including 34 portable
contracts for this experimental module; 16 optional Torch checks are skipped
in the locked CPU test environment.

## Eight-study outcome: failed

The experimental grouping change did not improve the exposed development
comparison: it achieved **0/8 bilateral matches** and **0/16 exact sides**.
Across the same fourteen available sides, absolute count error increased from
**28 to 57** versus the 0.2.3 baseline. The older 0.2.2 baseline had 1/8 bilateral
matches, 3/16 exact sides and error 27. Every available study's total error
worsened against both baselines. Availability was unchanged: seven studies
provided counts and one withheld them.

All sixteen runs closed without a technical failure. Independent native checks
confirmed exact repeat arrays, geometry, registries, counts and coverage. On the
four studies with saved proposal traces, original raw watershed, marker, free
and bone arrays matched in both repeats. All 2,562 protected files and 131 closed
output files remained unchanged. A separate count evaluator independently
recomputed the metrics and agreed with the coordinator for both repeats.

This rejects the current-group ownership rule as an accuracy improvement for
these studies. Its synthetic repairs do not justify changing application
counting. The candidate remains opt-in research code and was not promoted.

A subsequent source-bound diagnostic found that adjacency labeling can fuse
two already-selected, non-overlapping peak-family components into one seed in
one archived raw basin. A second collision had only one overlapping identity,
so that mechanism was rejected there. Original fields, markers and peak counts
were reproduced exactly. No modified seed set, watershed or count was run by
that diagnostic; it does not establish a repaired native cell distinction.

## Seed identity follow-up: rejected

A separate experiment retained distinct free-space and air peak components
unless they shared voxels. It preserved the exact selected peak foreground,
prominence, distance fields, fallback positions, original 0.2.3 merger and
acceptance. Eleven analytic checks passed, including a displaced-peak
single chamber: two raw regions correctly merged into one accepted cell.
All original 132 full-stage accepted arrays remained byte-identical to baseline
(124/132 passing, zero repairs or regressions), and all 864 representation
contracts passed. Independent saved-output review confirmed those results.

The subsequent fixed native-crop RAW-only test failed its declared repair gate:
the target conditional pair still shared one initial watershed region despite
the additional marker identities. The other previously colliding pair also
remained joined. Coverage and within-core fragmentation were unchanged. A
separate reviewer decoded the saved arrays and reference footprints and
reproduced those findings; all 1,692 protected files stayed unchanged.

The test stopped before merging, acceptance or counting. This rejects seed
identity preservation as a repair for that declared native collision under the
unchanged fields and settings. No further eight-study replay was run for it.
Missing initial separations, false separator evidence and anatomical membership
omissions remain unresolved; no count-improving algorithm has been established
by either experiment.
