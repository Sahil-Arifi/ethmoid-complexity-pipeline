"""Synthetic CPU gradients and checkpoint gates for the isolated research fork."""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from ethmoid_pipeline.counting import connectivity_experiment as experiment
from ethmoid_pipeline.counting import instance_experiment as original
from ethmoid_pipeline.counting import instance_learning as learning


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv3d(2, 7, 1)
        with torch.no_grad():
            self.layer.weight.copy_(torch.linspace(-.3, .4, 14).reshape(7, 2, 1, 1, 1))
            self.layer.bias.copy_(torch.linspace(-.2, .2, 7))
        self.register_buffer('ramp', torch.linspace(-.3, .3, 64).reshape(1, 1, 64, 1, 1))

    def forward(self, values):
        return self.layer(values) + self.ramp


def explicit_original_average(hu, model):
    """Differentiable original forward order, retaining full tile graphs."""
    sums = torch.zeros((7, *hu.shape))
    weights = torch.zeros_like(sums)
    for x in original.starts(hu.shape[0]):
        for y in original.starts(hu.shape[1]):
            for z in original.starts(hu.shape[2]):
                low = np.array([x, y, z])
                high = np.minimum(low + 64, hu.shape)
                h = np.full((64, 64, 64), -1000, np.float32)
                box = tuple(slice(a, b) for a, b in zip(low, high))
                sub = tuple(slice(0, b - a) for a, b in zip(low, high))
                h[sub] = hu[box]
                probabilities = torch.sigmoid(model(torch.from_numpy(learning.normalize_hu(h)[None])))[0]
                for channel, offset in enumerate(((0, 0, 0), *learning.OFFSETS)):
                    edge_high = high - offset
                    if np.any(edge_high <= low):
                        continue
                    target = tuple(slice(a, b) for a, b in zip(low, edge_high))
                    source = tuple(slice(0, b - a) for a, b in zip(low, edge_high))
                    sums[(channel, *target)] += probabilities[(channel, *source)]
                    weights[(channel, *target)] += 1
    return sums / torch.where(weights > 0, weights, torch.ones_like(weights)), weights


def test_stream_backward_matches_explicit_overlapping_original_tile_autograd():
    hu = np.linspace(-1000, 500, 66 * 8 * 8, dtype=np.float32).reshape(66, 8, 8)
    valid = np.ones(hu.shape, bool)
    valid[30, 4, 4] = False
    gradient = np.zeros((7, *hu.shape), np.float32)
    derivatives = {(0, 0, 1, 1): .2, (0, 33, 2, 3): -.7, (0, 65, 2, 3): .3,
                   (1, 61, 3, 2): .5, (4, 61, 3, 2): -.4, (2, 33, 6, 2): .35,
                   (5, 33, 3, 2): -.6, (3, 33, 2, 6): .8, (6, 33, 2, 3): -.25,
                   (0, 30, 4, 4): 1000}
    for index, value in derivatives.items():
        gradient[index] = value
    streamed, explicit = ToyModel().eval(), ToyModel().eval()
    before = {key: value.clone() for key, value in streamed.state_dict().items()}
    hu_before, valid_before, gradient_before = hu.copy(), valid.copy(), gradient.copy()
    tiles, weights = experiment.tile_plan(hu.shape)
    average, explicit_weights = explicit_original_average(hu, explicit)
    np.testing.assert_array_equal(weights, explicit_weights.numpy())
    assert len(tiles) == 2 and weights[1, 61, 3, 2] == 2 and weights[4, 61, 3, 2] == 1
    fixed_foreground = torch.where(torch.from_numpy(valid), average[0], 0)
    fixed_average = torch.cat((fixed_foreground[None], average[1:]))
    (fixed_average * torch.from_numpy(gradient)).sum().backward()
    assert experiment.stream_backward(hu, valid, streamed, torch, gradient, tiles, weights) == 2
    for actual, expected in zip(streamed.parameters(), explicit.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-7)
        assert torch.isfinite(actual.grad).all()
    assert all(torch.equal(before[key], value) for key, value in streamed.state_dict().items())
    np.testing.assert_array_equal(hu, hu_before)
    np.testing.assert_array_equal(valid, valid_before)
    np.testing.assert_array_equal(gradient, gradient_before)


def test_invalid_foreground_gradient_does_not_reach_network():
    hu = np.zeros((66, 8, 8), np.float32)
    valid = np.ones_like(hu, bool)
    valid[31, 2, 3] = False
    gradient = np.zeros((7, *hu.shape), np.float32)
    gradient[0, 31, 2, 3] = 1e6
    gradient[1, 33, 2, 3] = .4
    tiles, weights = experiment.tile_plan(hu.shape)
    actual, expected = ToyModel(), ToyModel()
    experiment.stream_backward(hu, valid, actual, torch, gradient, tiles, weights)
    gradient[0, 31, 2, 3] = 0
    experiment.stream_backward(hu, valid, expected, torch, gradient, tiles, weights)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def loss_fixture():
    values = np.full((7, 1, 1, 8), .5, np.float32)
    values[0, 0, 0, :4] = [.2, .7, .8, .25]
    values[1, 0, 0, 4], values[4, 0, 0, 5] = .4, .8
    values[3, 0, 0, 6], values[6, 0, 0, 7] = .6, .7
    targets = np.zeros_like(values)
    known = np.zeros_like(values, bool)
    known[0, 0, 0, :4] = True
    targets[0, 0, 0, :2] = 1
    known[3, 0, 0, 6], known[6, 0, 0, 7] = True, True
    targets[3, 0, 0, 6] = 1
    constraints = [
        {'kind': 'same', 'ids': [1], 'capacity': float(values[1, 0, 0, 4]),
         'cache_flat_index': int(np.ravel_multi_index((1, 0, 0, 4), values.shape))},
        {'kind': 'different', 'ids': [1, 2], 'capacity': float(values[4, 0, 0, 5]),
         'cache_flat_index': int(np.ravel_multi_index((4, 0, 0, 5), values.shape))},
    ]
    return values, targets, known, constraints


def isolate_loss(monkeypatch, name):
    for component in ('local', 'connectivity', 'coverage'):
        monkeypatch.setitem(experiment.CONFIG, f'{component}_loss_weight', 1. if component == name else 0.)


def test_unknown_local_targets_have_exactly_zero_gradient(monkeypatch):
    isolate_loss(monkeypatch, 'local')
    values, targets, known, constraints = loss_fixture()
    before = values.copy()
    gradient, result = experiment.probability_loss(values, targets, known, constraints, torch)
    assert not gradient[~known].any()
    assert gradient[0, 0, 0, 0] < 0 and gradient[0, 0, 0, 2] > 0
    assert gradient[3, 0, 0, 6] < 0 and gradient[6, 0, 0, 7] > 0
    assert all(np.isfinite(value) for value in result.values())
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('selected_head', [0, 1])
def test_declared_pair_gradients_have_same_and_different_signs(monkeypatch, selected_head):
    isolate_loss(monkeypatch, 'connectivity')
    values, targets, known, constraints = loss_fixture()
    values[selected_head, 0, 0, 4] = .4
    constraints[0]['cache_flat_index'] = int(np.ravel_multi_index((selected_head, 0, 0, 4), values.shape))
    gradient, _ = experiment.probability_loss(values, targets, known, constraints, torch)
    expected = np.zeros_like(values)
    expected[selected_head, 0, 0, 4] = -.3
    expected[4, 0, 0, 5] = .5
    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-7)
    assert not known[selected_head, 0, 0, 4]  # Pair gradient is declared, not a voxel target.


def test_coverage_protects_worst_known_foreground_and_background(monkeypatch):
    isolate_loss(monkeypatch, 'coverage')
    values, targets, known, constraints = loss_fixture()
    gradient, _ = experiment.probability_loss(values, targets, known, constraints, torch)
    expected = np.zeros_like(values)
    expected[0, 0, 0, 0], expected[0, 0, 0, 2] = -1.4, 1.4
    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-7)
    assert not gradient[~known].any()


def saved_fixture():
    return {'version': experiment.VERSION, 'model_version': learning.MODEL_VERSION,
            'config': deepcopy(experiment.CONFIG), 'steps_completed': experiment.CONFIG['steps'],
            'complete': True, 'state_dict': ToyModel().state_dict(),
            'annotation_manifest_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64,
            'initial_checkpoint_sha256': 'c' * 64}


def test_complete_checkpoint_loads_without_changing_weights_or_file(tmp_path, monkeypatch):
    saved = saved_fixture()
    path = tmp_path / 'synthetic-checkpoint.pt'
    torch.save(saved, path)
    digest = original.sha256(path)
    monkeypatch.setattr(learning, 'build_model', ToyModel)
    model, restored = experiment.load_checkpoint(path, torch)
    assert restored['complete'] is True and original.sha256(path) == digest
    assert all(torch.equal(value, saved['state_dict'][key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize('mutation', ['incomplete', 'wrong_steps', 'wrong_config', 'wrong_version',
                                    'wrong_model', 'missing_hash', 'nonhex_hash'])
def test_incomplete_or_wrong_protocol_checkpoint_rejected_before_model(tmp_path, monkeypatch, mutation):
    saved = saved_fixture()
    if mutation == 'incomplete':
        saved['complete'] = False
    elif mutation == 'wrong_steps':
        saved['steps_completed'] -= 1
    elif mutation == 'wrong_config':
        saved['config']['different_margin'] += .1
    elif mutation == 'wrong_version':
        saved['version'] = 'different-experiment'
    elif mutation == 'wrong_model':
        saved['model_version'] = 'different-model'
    elif mutation == 'nonhex_hash':
        saved['protocol_sha256'] = 'z' * 64
    else:
        del saved['protocol_sha256']
    path = tmp_path / 'synthetic-invalid.pt'
    torch.save(saved, path)
    monkeypatch.setattr(learning, 'build_model', lambda: pytest.fail('Model initialized before rejection'))
    with pytest.raises(original.ExperimentError, match='CHECKPOINT_PROTOCOL_MISMATCH'):
        experiment.load_checkpoint(path, torch)


def test_checkpoint_requires_external_absolute_path():
    with pytest.raises(original.ExperimentError, match='EXTERNAL_ARTIFACT_REQUIRED'):
        experiment.load_checkpoint('relative.pt', torch)


def test_preservation_checks_committed_source_without_treating_it_as_patient_artifact(tmp_path):
    repo = tmp_path / 'synthetic-repository'
    repo.mkdir()
    (repo / '.git').mkdir()
    code = repo / 'synthetic.py'
    code.write_text('value = 1\n', encoding='utf-8')
    protocol = {'input_hashes': {str(code): original.sha256(code)}}
    with pytest.raises(original.ExperimentError, match='EXTERNAL_ARTIFACT_REQUIRED'):
        original.external(code)
    experiment._preserved(protocol)
    code.write_text('value = 2\n', encoding='utf-8')
    with pytest.raises(original.ExperimentError, match='FROZEN_INPUT_CHANGED'):
        experiment._preserved(protocol)


def test_preservation_rejects_relative_integrity_inputs():
    with pytest.raises(original.ExperimentError, match='FROZEN_INPUT_CHANGED'):
        experiment._preserved({'input_hashes': {'relative.py': '0' * 64}})
