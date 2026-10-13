"""Freeze and measure native-path evidence; never changes detector acceptance."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from native_passage_evidence import passage_evidence
from ridge_counting_candidate import count_with_ridges
from ridge_counting_experiment import ARRAY_NAMES, _external_path, _json, _sha
from synthetic_counting_experiment import REPO, array_hash

THRESHOLDS_HU = (-900, -800, -700, -500, 0, 199)


class EvidenceFailure(RuntimeError):
    """Retain completed detector artifacts when subsequent evidence extraction fails."""

    def __init__(self, error, partitions, diagnostics):
        super().__init__(type(error).__name__ + ': ' + str(error))
        self.witnesses = {'partitions': partitions}
        self.diagnostics = diagnostics


def _items(known_run):
    from native_passage_cases import cases, metadata
    rows = json.loads((known_run / 'results.json').read_text(encoding='utf-8'))['results']
    if len(rows) != 68 or len({row['name'] for row in rows}) != 68:
        raise ValueError('Exactly 68 unique historical records are required.')
    items = []
    for row in rows:
        name = row['name']
        if not isinstance(name, str) or not name or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid historical case name.')
        folder = (known_run / name).resolve()
        if not folder.is_relative_to(known_run):
            raise ValueError('Historical case leaves input directory.')
        if json.loads((folder / 'result.json').read_text(encoding='utf-8')) != row:
            raise ValueError('Historical aggregate and case records disagree.')
        with np.load(folder / 'input.npz', allow_pickle=False) as saved:
            if set(saved.files) != set(ARRAY_NAMES):
                raise ValueError('Historical input array manifest changed.')
            arrays = {key: saved[key] for key in ARRAY_NAMES}
        if {key: array_hash(value) for key, value in arrays.items()} != row['input_hashes']:
            raise ValueError('Historical input arrays changed.')
        with np.load(folder / 'proposal.npz', allow_pickle=False) as saved:
            output = {key: saved[key] for key in saved.files}
        if {key: array_hash(value) for key, value in output.items()} != row['candidate']['detected_hashes']:
            raise ValueError('Historical proposal arrays changed.')
        info = {'name': name, 'group': 'known_development',
                'physical_count': row['construction_counts']['total'] if 'construction_counts' in row else row['physical_count'],
                'equivalence_group': row.get('equivalence_group'),
                'transform': row.get('transform', 'base'), 'historical': row['candidate']['diagnostics']}
        items.append((arrays, info, output['partitions']))
    for case in cases():
        arrays = {key: getattr(case.fixture if key in ARRAY_NAMES[:4] else case, key) for key in ARRAY_NAMES}
        items.append((arrays, metadata(case) | {'group': 'new_perturbations'}, None))
    if len(items) != 100 or len({info['name'] for _, info, _ in items}) != 100:
        raise ValueError('Exactly 100 unique cases are required.')
    return items


def _snapshot(known_run):
    items = _items(known_run)
    sources = sorted((REPO / 'src/ethmoid_pipeline/counting').glob('*.py'))
    sources += [REPO / 'scripts' / name for name in (
        'native_passage_evidence.py', 'native_passage_experiment.py', 'native_passage_cases.py',
        'ridge_counting_candidate.py', 'ridge_counting_experiment.py',
        'synthetic_counting_experiment.py', 'synthetic_blur_experiment.py')]
    sources += [REPO / 'docs/NATIVE_PASSAGE_EXPERIMENT.md']
    return {'experiment': 'native-passage-evidence-v1', 'clinical_accuracy': 'NOT_MEASURED',
            'counts_changed': False, 'known_run': str(known_run), 'case_count': len(items),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'python': sys.version, 'runtime': {name: version(name) for name in ('numpy', 'scipy', 'scikit-image')},
            'thresholds_hu': list(THRESHOLDS_HU),
            'sources': {p.relative_to(REPO).as_posix(): _sha(p) for p in sources},
            'protected_files': {p.relative_to(known_run).as_posix(): _sha(p) for p in sorted(known_run.rglob('*')) if p.is_file()},
            'inputs': [{'metadata': info, 'hashes': {k: array_hash(v) for k, v in arrays.items()}}
                       for arrays, info, _ in items]}


def freeze(path, known_run):
    path = _external_path(path)
    known_run = Path(known_run).resolve()
    if path.is_relative_to(known_run):
        raise ValueError('Plan must stay outside protected historical inputs.')
    snapshot = _snapshot(known_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(_json(snapshot) + '\n')


def measure(arrays, partitions=None):
    """Only image-derived inputs reach either detector or evidence extractor."""
    inputs = {key: arrays[key].copy() for key in ('hu', 'roi', 'affine')}
    before = {key: array_hash(value) for key, value in inputs.items()}
    for value in inputs.values(): value.setflags(write=False)
    diagnostics = None
    if partitions is None:
        output = count_with_ridges(**inputs)
        partitions, diagnostics = output['partitions'], output['diagnostics']
    saved_partitions = partitions.copy()
    saved_partitions.setflags(write=False)
    partitions = saved_partitions.copy()
    partition_hash = array_hash(partitions)
    partitions.setflags(write=False)
    try:
        report, witnesses = passage_evidence(**inputs, partitions=partitions)
        if before != {key: array_hash(value) for key, value in inputs.items()} or partition_hash != array_hash(partitions):
            raise ValueError('Measurement mutated its inputs.')
    except Exception as error:
        raise EvidenceFailure(error, saved_partitions, diagnostics) from error
    return report, {'partitions': saved_partitions, **witnesses}, diagnostics


def execute(plan_path, destination):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    known_run = Path(plan['known_run']).resolve()
    if plan != _snapshot(known_run):
        raise ValueError('Frozen source, inputs, rules, runtime or historical files changed.')
    destination = _external_path(destination)
    if destination.is_relative_to(known_run):
        raise ValueError('Output must stay outside protected historical inputs.')
    destination.mkdir(parents=True)
    (destination / 'protocol.json').write_text(_json(plan) + '\n', encoding='utf-8')
    rows = []
    for arrays, info, partitions in _items(known_run):
        folder = destination / info['name']
        folder.mkdir()
        np.savez_compressed(folder / 'input.npz', **arrays)
        failure, failure_stage, report, witnesses, diagnostics = None, None, None, {}, None
        try:
            report, witnesses, diagnostics = measure(arrays, partitions)
        except EvidenceFailure as error:
            failure, failure_stage = str(error), 'evidence'
            witnesses, diagnostics = error.witnesses, error.diagnostics
        except Exception as error:
            failure = type(error).__name__ + ': ' + str(error)
            failure_stage = 'measurement'
        row = {'metadata': info, 'evidence': report, 'failure': failure,
               'failure_stage': failure_stage,
               'detector_diagnostics': diagnostics or info.get('historical'),
               'input_hashes': {key: array_hash(value) for key, value in arrays.items()},
               'output_hashes': {key: array_hash(value) for key, value in witnesses.items()}}
        np.savez_compressed(folder / 'evidence.npz', **witnesses)
        (folder / 'result.json').write_text(_json(row) + '\n', encoding='utf-8')
        rows.append(row)
        print(info['name'], 'ERROR' if failure else 'measured', flush=True)
    # These fixed thresholds are counterfactual diagnostics, never chosen settings.
    table = []
    for group in ('known_development', 'new_perturbations'):
        eligible = [r for r in rows if r['metadata']['group'] == group and r['evidence']
                    and len(r['evidence']['pairs']) == 1 and r['evidence']['pairs'][0]['status'] == 'measured']
        for threshold in THRESHOLDS_HU:
            flagged = [r for r in eligible if r['evidence']['pairs'][0]['native_bottleneck_hu'] <= threshold]
            table.append({'group': group, 'threshold_hu': threshold, 'eligible_cases': len(eligible),
                          'one_space_flagged': sum(r['metadata']['physical_count'] == 1 for r in flagged),
                          'sealed_pairs_falsely_flagged': sum(r['metadata']['physical_count'] == 2 for r in flagged),
                          'flagged_names': [r['metadata']['name'] for r in flagged]})
    preserved = plan['protected_files'] == {p.relative_to(known_run).as_posix(): _sha(p)
                                          for p in sorted(known_run.rglob('*')) if p.is_file()}
    summary = {'case_count': len(rows), 'execution_errors': sum(r['failure'] is not None for r in rows),
               'clinical_accuracy': 'NOT_MEASURED', 'counts_changed': False,
               'historical_files_preserved': preserved, 'threshold_diagnostics': table, 'results': rows}
    (destination / 'results.json').write_text(_json(summary) + '\n', encoding='utf-8')
    lines = ['# Native passage evidence', '', 'Diagnostic measurements only; counts and acceptance rules are unchanged.',
             '', '| Group | HU threshold | Eligible two-proposal cases | True one-space flagged | Sealed pair falsely flagged |',
             '|---|---:|---:|---:|---:|']
    lines += [f"| {r['group']} | {r['threshold_hu']} | {r['eligible_cases']} | {r['one_space_flagged']} | {r['sealed_pairs_falsely_flagged']} |" for r in table]
    lines += ['', f"Execution errors: {summary['execution_errors']}. Historical files preserved: {preserved}.",
              'Transformed cases are repeated observations, not independent examples. No threshold is selected.',
              'No adjacent proposal pair means no pair-level measurement; it is not proof of a closed wall.',
              'Voxel-center clearance is not a measured anatomical aperture width. Clinical accuracy is unmeasured.']
    (destination / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if summary['execution_errors'] or not preserved:
        raise RuntimeError('Experiment retained failures; inspect saved results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--known-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.known_run is None) == (args.output is None):
        parser.error('Choose --known-run to freeze or --output to execute.')
    if args.known_run is not None:
        freeze(args.plan, args.known_run.resolve())
    else:
        execute(args.plan, args.output)
