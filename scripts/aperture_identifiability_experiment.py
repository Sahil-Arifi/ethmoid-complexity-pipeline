"""Frozen synthetic opening-identifiability experiment, with unchanged counts."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from aperture_observation_model import aperture_delta_bound, compatibility, render
from ridge_counting_experiment import ARRAY_NAMES, _external_path, _json, _sha
from synthetic_counting_experiment import REPO, array_hash

ERRORS = (0., 1., 5., 25.)
WIDTHS = (.03125, .125, .25, .5, 1.)
COARSE_WIDTHS = (.25, .5, 1.)
HU_KEYS = ('sealed', 'open', 'midpoint', 'coarse_0', 'coarse_1', 'coarse_2')


def scenarios():
    result = []
    for ai, aperture in enumerate(WIDTHS):
        for ci, center in enumerate(((0., 0.), (.17, -.23))):
            for si, sigma in enumerate((0., .3)):
                name = f'aperture_{ai}_center_{ci}_sigma_{si}'
                result.append(dict(name=name, base_name=name, aperture_mm=aperture,
                    aperture_center_mm=list(center), sigma_mm=sigma, transform='base',
                    shape=[16, 14, 12], spacing_mm=[.5, .6, .8], origin_mm=[-3.75, -3.9, -4.4],
                    half_extent_mm=[2.5, 2.4, 2.4], wall_mm=1., air_hu=-900., bone_hu=800.,
                    physical_counts={'sealed': 2, 'open': 1}))
    return result + [info | {'name': info['name'] + '_swap_xz_flip_y',
                            'transform': 'swap_xz_flip_y'} for info in result]


def construct(info):
    parameters = dict(shape=info['shape'], spacing=info['spacing_mm'], origin=info['origin_mm'],
                      half_extent=info['half_extent_mm'], wall_mm=info['wall_mm'],
                      aperture_center=info['aperture_center_mm'], sigma=info['sigma_mm'],
                      air=info['air_hu'], bone=info['bone_hu'])
    sealed = render(**parameters, aperture_mm=0.)
    opened = render(**parameters, aperture_mm=info['aperture_mm'])
    arrays = {'sealed': sealed, 'open': opened, 'midpoint': (sealed + opened) / 2,
              **{f'coarse_{i}': render(**parameters, aperture_mm=width)
                 for i, width in enumerate(COARSE_WIDTHS)},
              'changed': (sealed != opened).astype(np.uint8)}
    affine = np.diag([*info['spacing_mm'], 1.])
    affine[:3, 3] = info['origin_mm']
    if info['transform'] == 'swap_xz_flip_y':
        arrays = {key: value.transpose(2, 1, 0)[:, ::-1, :].copy() for key, value in arrays.items()}
        mapping = np.array([[0, 0, 1, 0], [0, -1, 0, info['shape'][1]-1],
                            [1, 0, 0, 0], [0, 0, 0, 1]])
        affine = affine @ mapping
    elif info['transform'] != 'base':
        raise ValueError('Unknown exact index transformation.')
    arrays['affine'] = affine
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def measure(arrays):
    """Supply only isolated HU observations to the compatibility calculation."""
    values = {key: np.array(arrays[key], copy=True) for key in HU_KEYS}
    before = {key: array_hash(value) for key, value in values.items()}
    for value in values.values():
        value.setflags(write=False)
    pair = [values['sealed'], values['open']]
    restricted = [values['sealed'], *(values[f'coarse_{i}'] for i in range(3))]
    rows = []
    try:
        # The compatibility routine validates volumes and residual arithmetic.
        delta = compatibility(values['sealed'], [values['open']], 0.)['residual_linf'][0]
        for bound in ERRORS:
            midpoint = compatibility(values['midpoint'], pair, bound)
            opened = compatibility(values['open'], pair, bound)
            limited = compatibility(values['open'], restricted, bound)
            augmented = compatibility(values['open'], [*restricted, values['open']], bound)
            rows.append(dict(error_bound_hu=bound, pair_midpoint=midpoint, pair_open=opened,
                restricted_open=limited, augmented_open=augmented, overlap_expected=delta <= 2*bound,
                restricted_sealed_only=limited['compatible_indices'] == [0],
                augmented_has_opening=any(index > 0 for index in augmented['compatible_indices'])))
    finally:
        if before != {key: array_hash(value) for key, value in values.items()}:
            raise ValueError('Compatibility calculation mutated its isolated inputs.')
    return {'delta_hu': delta, 'changed_voxels': int(np.count_nonzero(pair[0] != pair[1])),
            'by_error': rows}


def _protected_roots(known_run):
    known_run = _external_path(known_run, fresh=False)
    protocol = json.loads((known_run / 'protocol.json').read_text(encoding='utf-8'))
    roots = (known_run.parent, *(_external_path(path, fresh=False) for path in protocol['protected_files']))
    if len(roots) != 3 or len(set(roots)) != 3 or not all(root.is_dir() for root in roots):
        raise ValueError('Three distinct historical directories are required.')
    return roots


def _protected_files(roots):
    return {str(root): {path.relative_to(root).as_posix(): _sha(path)
                       for path in sorted(root.rglob('*')) if path.is_file()} for root in roots}


def _destination(path, roots):
    path = _external_path(path)
    if any(path.is_relative_to(root) for root in roots):
        raise ValueError('New artifacts must remain outside protected history.')
    return path


def _history(known_run):
    """Validate prior evidence without executing a detector."""
    manifest = json.loads((known_run / 'sha256.json').read_text(encoding='utf-8'))
    actual = {path.relative_to(known_run).as_posix(): _sha(path)
              for path in sorted(known_run.rglob('*')) if path.is_file() and path != known_run / 'sha256.json'}
    if actual != manifest:
        raise ValueError('Historical file manifest changed.')
    protocol = json.loads((known_run / 'protocol.json').read_text(encoding='utf-8'))
    if protocol != json.loads((known_run.parent / 'plan.json').read_text(encoding='utf-8')):
        raise ValueError('Historical protocol differs from its original plan.')
    rows = json.loads((known_run / 'results.json').read_text(encoding='utf-8'))['results']
    names = [row['metadata']['name'] for row in rows]
    if len(rows) != 100 or len(set(names)) != 100:
        raise ValueError('Exactly 100 unique canonical records are required.')
    counts = dict(correct_accepted=0, wrong_accepted=0, withheld=0, errors=0)
    records = []
    for row, name in zip(rows, names):
        if (not isinstance(name, str) or not name or name in ('.', '..')
                or '/' in name or '\\' in name or Path(name).name != name):
            raise ValueError('Invalid historical record name.')
        folder = (known_run / name).resolve()
        if not folder.is_relative_to(known_run):
            raise ValueError('Historical record escapes its directory.')
        if json.loads((folder / 'result.json').read_text(encoding='utf-8')) != row:
            raise ValueError('Historical aggregate and per-case records differ.')
        for filename, field, expected in (
                ('input.npz', 'input_hashes', set(ARRAY_NAMES)),
                ('candidate.npz', 'output_hashes', {'partitions', 'cavities', 'air', 'component_bridge'})):
            with np.load(folder / filename, allow_pickle=False) as saved:
                if len(saved.files) != len(expected) or set(saved.files) != expected:
                    raise ValueError('Historical array manifest changed.')
                if {key: array_hash(saved[key]) for key in saved.files} != row[field]:
                    raise ValueError('Historical array hash changed.')
        details = row['candidate']['diagnostics']
        if row['failure']:
            counts['errors'] += 1
        elif details['status'] == 'withheld':
            counts['withheld'] += 1
        elif details['status'] == 'accepted':
            correct = details['counts']['total'] == row['metadata']['physical_count']
            counts['correct_accepted' if correct else 'wrong_accepted'] += 1
        else:
            raise ValueError('Invalid historical decision.')
        records.append({'name': name, 'input_hashes': row['input_hashes'],
                        'output_hashes': row['output_hashes'], 'saved_counts': details['counts'],
                        'saved_status': details['status']})
    return {'records': records, 'counts': counts, 'case_count': len(rows), 'detector_rerun': False}


def _snapshot(known_run):
    known_run = _external_path(known_run, fresh=False)
    history = _history(known_run)
    sources = sorted((REPO / 'src/ethmoid_pipeline/counting').glob('*.py'))
    sources += [REPO / 'scripts' / name for name in ('aperture_observation_model.py',
        'aperture_identifiability_experiment.py', 'ridge_counting_experiment.py',
        'synthetic_counting_experiment.py', 'synthetic_blur_experiment.py')]
    sources += [REPO / 'docs/APERTURE_IDENTIFIABILITY_EXPERIMENT.md']
    return {'experiment': 'aperture-identifiability-v1', 'clinical_accuracy': 'NOT_MEASURED',
            'known_run': str(known_run), 'scenario_count': 40, 'scenarios': scenarios(),
            'error_bounds_hu': list(ERRORS), 'coarse_apertures_mm': list(COARSE_WIDTHS),
            'historical_baseline': history,
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'python': sys.version, 'runtime': {name: version(name) for name in
                ('numpy', 'scipy', 'scikit-image', 'nibabel', 'Pillow')},
            'sources': {path.relative_to(REPO).as_posix(): _sha(path) for path in sources},
            'protected_files': _protected_files(_protected_roots(known_run))}


def freeze(path, known_run):
    known_run = _external_path(known_run, fresh=False)
    path = _destination(path, _protected_roots(known_run))
    snapshot = _snapshot(known_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(_json(snapshot) + '\n')
    return path


def summarize(rows):
    base = [row for row in rows if row['metadata']['transform'] == 'base']
    successful = [row for row in rows if row['failure'] is None]
    table = []
    for bound in ERRORS:
        chosen = [next(item for item in row['measurement']['by_error'] if item['error_bound_hu'] == bound)
                  for row in base if row['failure'] is None]
        table.append({'error_bound_hu': bound, 'base_scenarios': len(base),
                      'measured_base_scenarios': len(chosen),
                      'overlapping_pairs': sum(item['pair_midpoint']['compatible_indices'] == [0, 1]
                                               for item in chosen),
                      'restricted_sealed_only': sum(item['restricted_sealed_only'] for item in chosen),
                      'oracle_opening_retained': sum(item['augmented_has_opening'] for item in chosen)})
    bounds_ok = all(row['measurement']['delta_hu'] <= aperture_delta_bound(
        row['metadata']['bone_hu']-row['metadata']['air_hu'], row['metadata']['wall_mm'],
        row['metadata']['aperture_mm'], row['metadata']['spacing_mm']) + 1e-9 for row in successful)
    overlap_ok = all((item['pair_midpoint']['compatible_indices'] == [0, 1]) == item['overlap_expected']
                     for row in successful for item in row['measurement']['by_error'])
    gates = {'complete_census': len(rows) == len({row['metadata']['name'] for row in rows}) == 40
             and len(base) == 20, 'no_execution_errors': len(successful) == len(rows),
             'phase_independent_bound': bounds_ok, 'midpoint_overlap_identity': overlap_ok,
             'oracle_completeness': all(item['augmented_has_opening'] for row in successful
                                        for item in row['measurement']['by_error'])}
    return {'record_count': len(rows), 'base_scenarios': len(base),
            'execution_errors': len(rows)-len(successful), 'table': table, 'gates': gates,
            'clinical_accuracy': 'NOT_MEASURED', 'counts_changed': False,
            'scientific_outcome': 'Competing synthetic explanations; no count-recovery claim.',
            'independent_spatial_audit': 'NOT_PERFORMED'}


def execute(plan_path, destination):
    plan_path = _external_path(plan_path, fresh=False)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    known_run = Path(plan['known_run']).resolve()
    roots = _protected_roots(known_run)
    destination = _destination(destination, roots)
    if plan != _snapshot(known_run):
        raise ValueError('Frozen source, runtime, rules or history changed.')
    destination.mkdir(parents=True)
    (destination / 'protocol.json').write_text(_json(plan) + '\n', encoding='utf-8')
    rows = []
    for info in scenarios():
        folder = destination / info['name']
        folder.mkdir()
        arrays, measured, failure = {}, None, None
        try:
            arrays = construct(info)
            measured = measure(arrays)
        except Exception as error:
            failure = type(error).__name__ + ': ' + str(error)
        np.savez_compressed(folder / 'arrays.npz', **arrays)
        row = {'metadata': info, 'measurement': measured, 'failure': failure,
               'array_hashes': {key: array_hash(value) for key, value in arrays.items()}}
        (folder / 'result.json').write_text(_json(row) + '\n', encoding='utf-8')
        rows.append(row)
    summary = summarize(rows)
    preserved = plan['protected_files'] == _protected_files(roots)
    summary.update(historical_files_preserved=preserved, historical_baseline=plan.get('historical_baseline'),
                   results=rows)
    summary['gates']['protected_files_preserved'] = preserved
    (destination / 'results.json').write_text(_json(summary) + '\n', encoding='utf-8')
    lines = ['# Synthetic opening detectability', '',
             'Error bounds are illustrative maximum voxel errors, not calibrated scanner noise.',
             'The denominator is 20 paired imaging scenarios; 20 reindexings are checked separately.',
             'The augmented library contains the generating opening: oracle completeness is not recovered accuracy.', '',
             '| Error bound (HU) | Base scenarios | Measured | Overlapping pairs | Restricted sealed-only | Oracle opening retained |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in summary['table']:
        lines.append('| ' + ' | '.join(str(row[key]) for key in ('error_bound_hu', 'base_scenarios',
            'measured_base_scenarios', 'overlapping_pairs', 'restricted_sealed_only', 'oracle_opening_retained')) + ' |')
    lines += ['', f"Software gates: {summary['gates']}.",
              f"Execution errors: {summary['execution_errors']}. Historical files preserved: {preserved}.",
              'Accepted counter outputs remain unchanged; clinical accuracy is unmeasured.']
    (destination / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {path.relative_to(destination).as_posix(): _sha(path)
                for path in sorted(destination.rglob('*')) if path.is_file()}
    (destination / 'sha256.json').write_text(_json(manifest) + '\n', encoding='utf-8')
    if summary['execution_errors'] or not preserved:
        raise RuntimeError('Experiment retained failures; inspect the saved census.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--known-run', type=Path)
    choice.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.known_run is not None:
        print(freeze(args.plan, args.known_run))
    else:
        result = execute(args.plan, args.output)
        print(_json({key: value for key, value in result.items() if key not in ('results', 'historical_baseline')}))
        sys.exit(0 if all(result['gates'].values()) else 2)
