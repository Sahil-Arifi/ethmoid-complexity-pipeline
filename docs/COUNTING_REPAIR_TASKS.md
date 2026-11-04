# Cell partition repair tasks

Status: causal audit completed for tasks 1-4 and 6; automatic numerical repair remains open. See [the audit](COUNTING_CAUSAL_AUDIT.md) for results and [the plan](COUNTING_REPAIR_PLAN.md) for sources, decisions and constraints.

## Task 1: Freeze diagnostic inputs

- [x] Seal source/runtime, the 96+36 procedural controls, original spatial oracles and 864 representation checks in an external manifest. The 864 results are reused, not rerun.
- [x] Record previously failed experiments and retain all original outputs; targets remain outside inference.
- [x] Preflight only the existing numerical stack and bounded exhaustive reference. Runtime, platform and package license metadata are saved; no dependency was added. Coverage and feature workers use explicit process caps.

Verification: hash audit and dry-run manifest validation. Dependencies: none. Likely files: external audit harness, existing scorer adapters. Scope: small.

## Task 2: Produce a stage error ledger

- [x] Verify partition comparison with hand-computed misses, splits, merges and background examples.
- [x] Map every known failed procedural group through markers, raw basins, merged groups and acceptance.
- [x] Preserve original spatial-oracle definitions and report missing foreground separately.

Verification: focused scorer checks and immutable stage-array comparison. Dependencies: 1. Likely files: external harness, tests/test_counting_diagnostic_trace.py only if an actual missing contract is found. Scope: small.

## Task 3: Test initial partition representability

- [x] Identify mixed-core basins and omitted foreground under existing procedural oracles. None occur in 132 cases; connected union witnesses also exist.
- [x] Assess deterministic synthetic group-contained seed interventions. Not run: no case meets the required mixed-core trigger.
- [x] Record whether proposal generation, merging, or both need repair; no change is selected from counts alone.

Verification: fixed-input intervention pairs and source hash checks. Dependencies: 2. Likely files: external harness, engine seed/watershed adapters. Scope: medium.

## Task 4: Isolate separator evidence and optimization

- [x] Compare current edges and diagnostic group-derived edges on representable fixed basins.
- [x] Verify graph construction/sign conventions/constraints with exhaustive reference partitions on graphs of at most eight nodes. The new objective is a comparator, not the greedy policy's contract.
- [x] Assess the conditional ELF adapter. Not needed: the false hard constraints make the reference infeasible regardless of optimizer.

Verification: exact-reference graph checks plus original spatial oracles. Dependencies: 3. Likely files: isolated graph adapter and its tests. Scope: medium.

## Checkpoint: Choose the repair from evidence

- [x] A ledger identifies the first causal failure and whether later failures coexist.
- [x] Independent review accepts the intervention conclusion; oracle results are not presented as automatic accuracy.
- [ ] A written observable rule, counterexamples and falsification criteria exist before numerical source edits.

## Task 5: Implement one demonstrated component repair

- [ ] Modify only the causal component; retain original evidence, thresholds and constraints unless the reviewed specification explicitly changes them.
- [ ] Resolve known failure families while preserving passing aperture, cantilever, foreign-wall and opaque controls.
- [ ] Retain uncertainty where image evidence is insufficient; do not use generator truth in inference.

Verification: targeted before/after witnesses and 96+36 preserved spatial controls. Dependencies: 4 and checkpoint. Likely files: engine.py or a quarantined research module, focused tests, version binding if numerical behavior changes. Scope: medium per component; split further if multiple components fail.

## Task 6: Diagnose coverage losses

- [x] On procedural inputs, vary search/membership/exclusions independently while holding partitions fixed where the intervention permits.
- [x] Attribute all losses to explicit gates; no recovery candidate or gate relaxation is justified yet.
- [x] Relate findings to preserved native traces as a hypothesis only; no clinical inference or rerun occurred.

Verification: procedural assignment and exclusion accounting; unchanged gate values. Dependencies: 2. May proceed independently of 3-5. Likely files: external harness and localization trace adapters; product changes only after a demonstrated cause. Scope: medium.

## Task 7: Freeze and pass independent geometry validation

- [ ] Freeze at least 48 geometries across at least eight independent structural families before candidate execution, with declared sampling/ambiguity outcomes.
- [ ] Pass the original 96+36 spatial controls, all known failure expectations, the 864 representation checks, and the independent family holdout without post-test tuning.
- [ ] Pass resource, opacity, source-immutability and applicable software checks. Retain every failed attempt.

Verification: independently authored manifest and evaluator report. Dependencies: 5; include 6 for a coverage candidate. Likely files: external renderer manifests, evaluator, focused regression tests. Scope: medium.

## Task 8: Compare a frozen automatic candidate

- [ ] Freeze one version, source/runtime and fixed localizers; run all eight development studies twice with targets excluded from inference.
- [ ] Independently verify artifacts, geometry, registries, coverage and counts against both preserved baselines.
- [ ] Report achieved versus unachieved: 8/8 development count agreement is the requested target; individual native references are still required to validate cell identities.

Verification: source-bound reproduction runner and independent closure. Dependencies: 7. Likely files: scripts/reproduce_clinician_counts.py configuration/binding, new external attempt manifest, README result update. Scope: medium.

## Final checkpoint

- [ ] No known preserved algorithm failure remains unresolved or hidden by a weakened gate.
- [ ] New numerical behavior has passed independent geometry tests and applicable CI.
- [ ] Any remaining clinical-reference uncertainty is stated separately; no claim of universal error-free counting or guaranteed clinical accuracy.
