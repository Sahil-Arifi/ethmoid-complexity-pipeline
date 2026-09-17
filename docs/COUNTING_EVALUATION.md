# Frozen cohort evaluation

This runner measures whether the experimental pipeline executes and preserves
its source geometry, cell identities, and recorded configuration. It does not
measure clinical counting accuracy. Reference counts and expert review are
still required for that endpoint.

## Inputs and isolation

- Run `scripts/evaluate_counting_cohort.py` with the application's Python 3.13
  environment (`.venv\Scripts\python.exe`). The runner needs only the standard
  library, NumPy, and NiBabel.
- Each selected volume runs serially in a separate process using the isolated
  Python 3.12 inference runtime at
  `%LOCALAPPDATA%\EthmoidComplexity\tools\counting-runtime\Scripts\python.exe`.
- The frozen manifest is
  `%LOCALAPPDATA%\EthmoidComplexity\work\counting\cohort.v1.local.json`.
  It contains five development and five held-out studies. A run selects exactly
  one split. Development evaluation does not open held-out image content.
- Source volumes are read-only inputs. Sources, caches, worker logs, result
  files, and evaluations remain in the protected work root, outside Git and
  synchronized folders. Neutral labels do not deidentify image pixels.
- `--output` must name a **fresh** directory inside the protected work root.
  An existing directory is rejected, including an empty one. A repeat requires
  a different output directory; prior evidence is retained.

The default runtime must have the pinned dependencies, prefetched models, an
active outbound-network guard, and telemetry disabled. The runner verifies
model file sizes and hashes, probes the offline guard, records installed package
versions, and checks runtime provenance again after each worker. No model is
downloaded by this evaluation workflow.

## Freeze and run

First complete development and commit the engine, evaluator, and complete
`src/ethmoid_pipeline/counting/protocol.json` algorithm object. Every field of
`CountingConfig` must be explicit in `protocol.algorithm`; unspecified defaults
are rejected. A clean detached checkout lets documentation or UI work continue
on another checkout without changing the evaluated engine.

The following PowerShell example starts in the application repository. Choose
unused snapshot and output directory names. These commands are examples, not a
record of executed evaluations.

```powershell
$projectRoot = (Get-Location).Path
$appPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$protectedWork = Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\work'
$snapshot = Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\engineering-worktrees\counting-freeze-v1'

if (git status --porcelain=v1 --untracked-files=all) { throw 'Commit and clean the engine checkout before freezing.' }
$freezeCommit = git rev-parse HEAD
git worktree add --detach $snapshot $freezeCommit
if ($LASTEXITCODE -ne 0) { throw 'The clean snapshot was not created.' }

$protocol = Join-Path $snapshot 'src\ethmoid_pipeline\counting\protocol.json'
$cohort = Join-Path $protectedWork 'counting\cohort.v1.local.json'
$cohortSha = (Get-FileHash -LiteralPath $cohort -Algorithm SHA256).Hash.ToLowerInvariant()
$algorithmSha = & $appPython -c 'import hashlib,json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); print(hashlib.sha256(json.dumps(p["algorithm"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest())' $protocol
if ($LASTEXITCODE -ne 0) { throw 'The complete algorithm hash could not be calculated.' }

$runner = Join-Path $snapshot 'scripts\evaluate_counting_cohort.py'
$developmentOutput = Join-Path $protectedWork 'counting\evaluations\development-freeze-v1'
& $appPython $runner --pipeline-root $snapshot --split development --output $developmentOutput --expected-code-sha $freezeCommit --expected-algorithm-sha256 $algorithmSha --expected-cohort-sha256 $cohortSha
```

Inspect the protected development evidence before starting the held-out run.
Retain the same code commit, algorithm hash, cohort hash, and runtime/model
installation. Do not recompute these values from changed files simply to make a
failed freeze check pass. If development requires an algorithm change, create
and evaluate a new development freeze before using held-out results.

```powershell
$heldOutOutput = Join-Path $protectedWork 'counting\evaluations\held-out-freeze-v1'
& $appPython $runner --pipeline-root $snapshot --split held_out --output $heldOutOutput --expected-code-sha $freezeCommit --expected-algorithm-sha256 $algorithmSha --expected-cohort-sha256 $cohortSha
```

Held-out evaluation requires a clean committed checkout and all three expected
hash arguments. The runner checks the exact cohort bytes, selected input hashes,
engine module hashes, protocol bytes, and effective algorithm. It rejects
duplicate source paths or hashes across the frozen split. Source, code, cohort,
protocol, cache, or runtime changes detected during an attempt invalidate its
count. The expected hash arguments are optional for development, but using them
for the formal development run makes its intended freeze explicit.

`--runtime-python PATH` selects an already prepared isolated inference runtime;
it does not install one. `--timeout-seconds 1800` is the default per-worker limit.
A timed-out worker and its process tree are terminated, and the attempt is
recorded as failed with null counts. The console prints only an aggregate
summary. Exit code `0` means every selected attempt completed and passed the
engineering checks; `2` means preparation or at least one attempt failed.

## Cached localizers and producer provenance

The runner looks for `work\counting\localizers\Case###`. A reusable cache must
have `localizers.local.json` bound to the exact input SHA-256. Every listed mask
is checked against its recorded hash, source shape and affine, and binary values.
An absent cache lets the worker generate localizers within that attempt. An
invalid existing cache causes failure rather than silent replacement.

Early development caches recorded source and mask integrity but did not record
the paired localizers' producer weight-manifest hash at generation. They may be
used for development with an explicit producer-unverified warning. Today's
installed weights cannot retroactively establish which weights produced an
older cache. The evaluator records that limitation per attempt and counts those
cases in the aggregate summary.

Held-out cached or newly generated localizers require producer provenance
matching the evaluated weight-manifest SHA-256 and TotalSegmentator version.
New engine manifests record those fields and per-task settings. Generated
full-field retry caches are also recorded and checked when present. These
models provide anatomical landmarks; their masks are not ethmoid cell labels.

## Evidence and interpretation

Each fresh output directory contains:

- `provenance.local.json`: selected split, exact cohort/protocol/algorithm/code
  hashes, evaluator hash, model manifest, runtime packages, and start time.
- `protocol.local.json`: the exact protocol snapshot supplied to every worker.
- `Case###/attempt.local.json`: source hash, completion or failure, counts or
  null counts, elapsed time, worker return code, cache provenance, and validation
  evidence. Existing clinical/Slicer review states remain separate.
- `Case###/worker.local.log` and `Case###/worker/`: protected worker diagnostics,
  result JSON, and any generated native artifacts. Failed workers can retain
  diagnostic proposals; their presence does not make a count available.
- `summary.local.json`: aggregate execution results and diagnostic summaries.

A completed count must agree with a nonempty, unique automatic cell registry,
native cavity labels, per-cell voxel and air counts, physical centroids, and
localized laterality. Air labels must belong to their corresponding cavities.
All artifacts must match the original source shape and affine. No-cell,
inconsistent, invalid, failed, or unfinished results do not receive a numeric
count. If no attempt succeeds, aggregate count fields remain null.

Opaque/partly opaque and uncertainty summaries describe accepted candidates.
Structured air-coverage diagnostics, when available, describe assigned air
within the engine's inferred region. Coverage is reported for attempts that
expose that diagnostic, including failures; it is **not clinical sensitivity,
specificity, or counting accuracy**. Aggregate counts include completed cases
only and must not be presented as totals for failed cases or the whole cohort.

Execution success, mask consistency, and deterministic provenance cannot prove
that the proposed compartments are the correct anatomical cells. This workflow
does not replace eligibility review, specialist inspection, manual 3D reference
counts, or future comparison against those references. Do not tune thresholds
using held-out outputs and then describe the same run as an independent test.

## Formal results

Formal evaluation results are pending. Append only aggregates from executed,
preserved evaluation outputs, including failed cases and provenance limitations.
Do not infer count accuracy without reference counts.
