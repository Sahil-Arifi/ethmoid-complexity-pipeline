# Reproducing supplied counts

`scripts/reproduce_clinician_counts.py` compares automatic predictions with supplied left/right counts on fixed source volumes. These are development regression cases. References, source scans, acquisition metadata, localizer caches and results stay outside the checkout. The supplied targets are used only after inference; the worker receives the source, source-bound localizers and acquisition snapshot, method and one global configuration.

Create a protected local JSON manifest with an `entries` array. Each entry contains a unique `case_id`, absolute `input` volume path, `source_sha256`, absolute `landmarks` cache directory, source-bound `acquisition` object, and `reference` object containing nonnegative integer `left` and `right` counts. Confirm the chosen image series rather than selecting a reconstruction by its resulting count. Record any unresolved source-series correspondence alongside the private manifest.

Use the existing offline inference Python and set `PYTHONPATH=src`:

```text
python scripts/reproduce_clinician_counts.py --manifest <external-manifest.json> --output <fresh-external-directory> --method research-v5 --repeat 3
```

`--config` optionally supplies one global counting configuration. It applies identically to every case and repetition. The original source and prior attempts are never overwritten. Each repetition invokes a separate worker process, retains its logs and result, and records failures and withheld counts in the scheduled denominator.

Success requires every supplied left/right count to match in every repetition, identical cell registries and native artifact hashes between repetitions, valid native measurements, and unchanged inputs and code. A matching total with incorrect sides fails. A count match with changing masks also fails. With `--repeat 1`, the runner records the baseline but returns exit code 1 because a single execution cannot prove repeatability. Exit 0 means the complete reproduction requirement passed; exit 1 means an executed batch did not meet it; exit 2 means preflight or invocation failed.

The output includes per-attempt `comparison.local.json`, protected worker output, `provenance.local.json` and `summary.local.json`. Keep these outside source control. Code hashes bind the counting modules, protocols, runner and reused native-artifact validator. The result establishes reproduction on the chosen development inputs; it does not establish accuracy on additional scans.

Synthetic fault-injection tests cover incorrect sides, changing masks, missing output, invalid registry or geometry, wrong native HU measurements, method/configuration mismatches, modified inputs, duplicate cases and immutable output directories. They run in the dedicated CPU counting job. No reference counts or patient examples are embedded in the algorithm or its tests.
