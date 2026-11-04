# Cell partition repair tasks

Status: researched and planned; numerical implementation not started in this phase. See [the plan](COUNTING_REPAIR_PLAN.md) for sources, decisions and constraints.

## Task 1: Freeze diagnostic inputs

- [ ] Seal source/runtime, the 96+36 procedural controls, original spatial oracles and 864 representation checks in an external manifest.
- [ ] Record previously failed experiments and retain all original outputs; targets remain outside inference.
- [ ] Preflight only tools selected for the next task in an isolated environment. Record exact version/license/platform and enforce process caps; do not install all candidates.

Verification: hash audit and dry-run manifest validation. Dependencies: none. Likely files: external audit harness, existing scorer adapters. Scope: small.

## Task 2: Produce a stage error ledger

- [ ] Verify partition comparison with hand-computed misses, splits, merges and background examples.
- [ ] Map every known failed procedural group through markers, raw basins, merged groups and acceptance.
- [ ] Preserve original spatial-oracle definitions and report missing foreground separately.

Verification: focused scorer checks and immutable stage-array comparison. Dependencies: 1. Likely files: external harness, tests/test_counting_diagnostic_trace.py only if an actual missing contract is found. Scope: small.

## Task 3: Test initial partition representability

- [ ] Identify mixed-core basins and omitted foreground under existing procedural oracles.
- [ ] Run deterministic synthetic group-contained seed interventions, separately marked as diagnostic.
- [ ] Record whether proposal generation, merging, or both need repair; no change is selected from counts alone.

Verification: fixed-input intervention pairs and source hash checks. Dependencies: 2. Likely files: external harness, engine seed/watershed adapters. Scope: medium.

## Task 4: Isolate separator evidence and optimization

- [ ] Compare current edges and diagnostic group-derived edges on representable fixed basins.
- [ ] Verify graph construction/sign conventions/constraints with exhaustive reference partitions on graphs of at most eight nodes.
- [ ] Evaluate an ELF adapter only if this distinguishes an actionable optimizer issue; record backend and external resource caps.

Verification: exact-reference graph checks plus original spatial oracles. Dependencies: 3. Likely files: isolated graph adapter and its tests. Scope: medium.

## Checkpoint: Choose the repair from evidence

- [ ] A ledger identifies the first causal failure and whether later failures coexist.
- [ ] Independent review accepts the intervention conclusion; oracle results are not presented as automatic accuracy.
- [ ] A written observable rule, counterexamples and falsification criteria exist before numerical source edits.

## Task 5: Implement one demonstrated component repair

- [ ] Modify only the causal component; retain original evidence, thresholds and constraints unless the reviewed specification explicitly changes them.
- [ ] Resolve known failure families while preserving passing aperture, cantilever, foreign-wall and opaque controls.
- [ ] Retain uncertainty where image evidence is insufficient; do not use generator truth in inference.

Verification: targeted before/after witnesses and 96+36 preserved spatial controls. Dependencies: 4 and checkpoint. Likely files: engine.py or a quarantined research module, focused tests, version binding if numerical behavior changes. Scope: medium per component; split further if multiple components fail.

## Task 6: Diagnose coverage losses

- [ ] On procedural inputs, vary search/membership/exclusions independently while holding partitions fixed.
- [ ] Attribute all losses to explicit gates; recovery must not admit known excluded tissue.
- [ ] Relate findings to preserved native traces without claiming those traces establish anatomical truth.

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
