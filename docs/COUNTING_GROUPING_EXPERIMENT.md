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

Status: synthetic verification complete; the eight-study comparison is next.
No new clinical count result is established.
