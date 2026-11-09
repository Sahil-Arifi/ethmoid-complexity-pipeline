"""Compare one isolated counter attempt with the canonical synthetic baseline."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np

import canonical_counting_experiment as canonical
from canonical_ridge_candidate import count_with_canonical_ridges
from ridge_counting_experiment import _json, _sha
from saturated_barrier_candidate import count_with_saturated_barrier
from synthetic_counting_experiment import REPO, array_hash, exact_correspondence

SOURCES = ('scripts/saturated_barrier_candidate.py', 'scripts/counter_failure_experiment.py',
           'tests/test_saturated_barrier_candidate.py', 'tests/test_counter_failure_experiment.py')
SUFFIX = '_swap_xz_flip_y'


def snapshot(known_run):
    result = canonical._snapshot(known_run)
    result.update(experiment='actual-counter-endpoint-falsification-v1', candidate_budget=1,
                  promotion_rule='No wrong acceptances on saved observations OR numerical adversaries; '
                                 'all baseline correct counts and masks preserved. Conditional results '
                                 'do not establish acquisition assumptions or clinical validity.')
    result['sources'].update({name: _sha(REPO / name) for name in SOURCES})
    return result


def measure(detector, arrays):
    # Reuse the historical output/input validator without changing its frozen source.
    previous = canonical.count_with_canonical_ridges
    try:
        canonical.count_with_canonical_ridges = detector
        return canonical.measure(arrays)
    finally:
        canonical.count_with_canonical_ridges = previous


def rounding_challenge():
    """A positive square through-opening is lost by the stored float32 average."""
    hu = np.full((30, 24, 20), 800, dtype=np.float32)
    hu[4:26, 4:20, 4:16] = -900
    hu[14:16, 4:20, 4:16] = 800
    opened = hu.copy()
    with localcontext() as context:
        context.prec = 80
        aperture, voxel = Decimal('0.000001'), Decimal('0.5')
        average = Decimal(800) - Decimal(1700) * (aperture / voxel) ** 2
        assert average < 800
        opened[14:16, 12, 10] = float(average)
    assert np.array_equal(hu, opened)
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = [-7.25, -5.75, -4.75]
    roi = np.ones(hu.shape, bool)
    baseline = count_with_canonical_ridges(opened, roi, affine)['diagnostics']
    candidate = count_with_saturated_barrier(opened, roi, affine)['diagnostics']
    return {'name': 'positive_through_opening_rounds_to_sealed_float32',
            'physical_count_open': 1, 'physical_count_sealed': 2,
            'aperture_mm': str(aperture), 'exact_wall_voxel_average_hu': str(average),
            'stored_wall_voxel_average_hu': float(opened[14, 12, 10]),
            'identical_full_observation': np.array_equal(hu, opened),
            'input_hashes': {k: array_hash(v) for k, v in dict(hu=opened, roi=roi, affine=affine).items()},
            'baseline': baseline, 'candidate': candidate,
            'wrong_accepted': candidate['status'] == 'accepted' and candidate['counts']['total'] != 1}


def summarize(rows, challenge):
    counts = {method: {key: 0 for key in ('correct_accepted', 'wrong_accepted', 'withheld', 'errors')}
              for method in ('baseline', 'candidate')}
    recovered, lost, masks_changed, spatial = [], [], [], []
    twins = {}
    for row in rows:
        classes = {}
        for method in counts:
            details = row[method]['diagnostics']
            category = canonical._category(details, row['metadata']['physical_count'], row['failure'])
            counts[method][category] += 1
            classes[method] = category
        name = row['metadata']['name']
        if classes['baseline'] == 'correct_accepted' and classes['candidate'] != 'correct_accepted':
            lost.append(name)
        if classes['baseline'] != 'correct_accepted' and classes['candidate'] == 'correct_accepted':
            recovered.append(name)
            if not row['exact_spatial_correspondence']:
                spatial.append(name)
        if row['output_hashes']['baseline'] != row['output_hashes']['candidate']:
            masks_changed.append(name)
        group = row['metadata'].get('equivalence_group')
        if group:
            twins.setdefault(group, []).append(row['candidate']['diagnostics']['status'])
    gates = {'all_100_retained': len(rows) == len({r['metadata']['name'] for r in rows}) == 100,
             'zero_saved_wrong_acceptances': counts['candidate']['wrong_accepted'] == 0,
             'baseline_correct_preserved': not lost, 'all_masks_preserved': not masks_changed,
             'recovered_exact_spatial_correspondence': not spatial,
             'all_material_twins_withheld': len(twins) == 4 and all(v == ['withheld', 'withheld'] for v in twins.values()),
             'no_execution_errors': not counts['candidate']['errors'],
             'numerical_adversary_safe': not challenge['wrong_accepted']}
    return {'case_count': len(rows), 'comparison': counts, 'recovered': recovered,
            'lost': lost, 'masks_changed': masks_changed, 'recovered_spatial_failures': spatial,
            'gates': gates, 'numerical_adversary': challenge,
            'clinical_accuracy': 'NOT_MEASURED', 'promotion': 'REJECTED' if not all(gates.values()) else 'REQUIRES_EXTERNAL_MODEL_VALIDATION'}


def execute(plan_path, output):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    known = Path(plan['known_run'])
    if plan != snapshot(known):
        raise ValueError('Frozen source, runtime or historical inputs changed.')
    roots = canonical._protected_roots(known)
    output = canonical._destination(output, roots)
    output.mkdir(parents=True)
    (output / 'protocol.json').write_text(_json(plan)+'\n', encoding='utf-8')
    rows, predictions = [], {}
    for arrays, info in canonical._items(known):
        name = info['name']
        folder = output / name
        folder.mkdir()
        records, masks, failure = {}, {}, None
        for method, detector in (('baseline', count_with_canonical_ridges), ('candidate', count_with_saturated_barrier)):
            try:
                records[method], masks[method] = measure(detector, arrays)
            except Exception as error:
                failure = {'method': method, 'type': type(error).__name__, 'message': str(error)}
                records[method], masks[method] = canonical._failed_prediction(arrays['hu'].shape)
            np.savez_compressed(folder / (method+'.npz'), **masks[method])
        predictions[name] = masks['candidate'], records['candidate']['cells']
        row = {'metadata': info, 'baseline': records['baseline'], 'candidate': records['candidate'],
               'failure': failure, 'input_hashes': {k: array_hash(v) for k,v in arrays.items()},
               'output_hashes': {method: {k: array_hash(v) for k,v in detected.items()} for method,detected in masks.items()},
               'exact_spatial_correspondence': exact_correspondence(arrays['labels'], masks['candidate']['cavities']) is not None}
        (folder / 'result.json').write_text(_json(row)+'\n', encoding='utf-8')
        rows.append(row)
    reindexed = []
    reindex_failures = []
    for name, (masks, cells) in predictions.items():
        if not name.endswith(SUFFIX):
            continue
        base_masks, base_cells = predictions[name[:-len(SUFFIX)]]
        exact = all(np.array_equal(masks[key], base_masks[key].transpose(2,1,0)[:,::-1,:]) for key in masks)
        registry = len(cells) == len(base_cells) and all(
            {k:v for k,v in a.items() if k != 'centroid_ras_mm'} == {k:v for k,v in b.items() if k != 'centroid_ras_mm'}
            and np.allclose(a['centroid_ras_mm'], b['centroid_ras_mm'], atol=1e-8, rtol=0)
            for a,b in zip(cells,base_cells))
        reindexed.append(name)
        if not (exact and registry):
            reindex_failures.append(name)
    result = summarize(rows, rounding_challenge())
    result.update(reindex_pairs=len(reindexed), reindex_failures=reindex_failures)
    result['gates'].update(exact_reindexing=len(reindexed) == 22 and not reindex_failures,
                           historical_files_preserved=plan['protected_files'] == canonical._protected_files(roots))
    if not all(result['gates'].values()):
        result['promotion'] = 'REJECTED'
    result['results'] = rows
    (output / 'results.json').write_text(_json(result)+'\n', encoding='utf-8')
    manifest = {p.relative_to(output).as_posix(): _sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'sha256.json').write_text(_json(manifest)+'\n', encoding='utf-8')
    print(_json({k:v for k,v in result.items() if k != 'results'}))
    if not result['gates']['no_execution_errors'] or not result['gates']['historical_files_preserved']:
        raise RuntimeError('Experiment retained execution failures or historical changes.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--known-run', type=Path)
    choice.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.known_run:
        path = canonical._destination(args.plan, canonical._protected_roots(args.known_run))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(_json(snapshot(args.known_run))+'\n')
        print(path)
    else:
        execute(args.plan, args.output)
