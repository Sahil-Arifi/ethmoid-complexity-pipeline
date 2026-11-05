"""Annotation-component sampling contracts use generated labels only."""
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_learning as learning
from ethmoid_pipeline.counting import boundary_pipeline as pipeline
from ethmoid_pipeline.counting import instance_experiment as safe


def heads(shape):
    return {key: np.full(shape, -1, np.int8)
            for key in ("foreground_labels", "separator_labels")}


def test_pools_use_observed_six_connected_components_in_stable_native_order():
    arrays = heads((9, 9, 9))
    arrays["foreground_labels"][1, 1, 1:3] = 0
    arrays["foreground_labels"][2, 2, 2] = 0  # Diagonal contact stays separate.
    arrays["foreground_labels"][7, 7, 7] = 1
    arrays["separator_labels"][4, 4, 4] = 0
    arrays["separator_labels"][5, 5, 5] = 1
    before = {key: value.copy() for key, value in arrays.items()}
    first = pipeline._pools([(None, None, arrays)])
    second = pipeline._pools([(None, None, arrays)])
    regions = first[0][0][1]
    assert [r.tolist() for r in regions] == [
        [np.ravel_multi_index(p, (9, 9, 9)) for p in ((1, 1, 1), (1, 1, 2))],
        [np.ravel_multi_index((2, 2, 2), (9, 9, 9))]]
    for pool, repeated in zip(first, second):
        assert len(pool) == len(repeated) == 1
        for region, again in zip(pool[0][1], repeated[0][1]):
            assert np.array_equal(region, again)
    assert all(np.array_equal(value, before[key]) for key, value in arrays.items())


def test_large_pool_keeps_every_voxel_and_the_rare_component_after_100k():
    shape = (103, 103, 21)
    arrays = heads(shape)
    arrays["foreground_labels"][:101] = 0
    arrays["foreground_labels"][102, 102, 20] = 0
    arrays["foreground_labels"][101, 0, 0] = 1
    arrays["separator_labels"][101, 0, 1] = 0
    arrays["separator_labels"][101, 0, 2] = 1
    regions = pipeline._pools([(None, None, arrays)])[0][0][1]
    assert [len(r) for r in regions] == [101*103*21, 1]
    assert regions[1].tolist() == [np.ravel_multi_index((102, 102, 20), shape)]
    assert np.array_equal(np.sort(np.concatenate(regions)), np.flatnonzero(arrays["foreground_labels"] == 0))


def test_anchor_selection_balances_studies_then_components_then_voxels():
    pool = [(0, [np.array([10, 11, 12]), np.array([20])]),
            (1, [np.array([30]), np.array([40]), np.array([50]), np.array([60])])]
    rng = np.random.default_rng(83)
    component_rng = np.random.default_rng(89)
    draws = [pipeline._sample_anchor(pool, rng, component_rng) for _ in range(16000)]
    counts = Counter(draws)
    study_zero = sum(n for (study, _), n in counts.items() if study == 0)
    assert abs(study_zero / len(draws) - .5) < .025
    assert abs(counts[0, 20] / study_zero - .5) < .025
    for position in (10, 11, 12):
        assert abs(counts[0, position] / (study_zero-counts[0, 20]) - 1/3) < .035
    for position in (30, 40, 50, 60):
        assert abs(counts[1, position] / (len(draws)-study_zero) - .25) < .025


def test_component_draw_uses_independent_rng_and_repeats_deterministically():
    class RecordingRng:
        def __init__(self):
            self.calls = []

        def integers(self, high):
            self.calls.append(high)
            return 0

    pool = [(0, [np.array([3, 5]), np.array([8])]), (1, [np.array([13])])]
    main, component = RecordingRng(), RecordingRng()
    assert pipeline._sample_anchor(pool, main, component) == (0, 3)
    assert main.calls == [2, 2] and component.calls == [2]
    patch_rng = np.random.default_rng(learning.CONFIG["seed"])
    patch_state = patch_rng.bit_generator.state
    rngs = [np.random.default_rng(learning.CONFIG["seed"]) for _ in range(4)]
    first = [pipeline._sample_anchor(pool, rngs[0], rngs[1]) for _ in range(128)]
    repeat = [pipeline._sample_anchor(pool, rngs[2], rngs[3]) for _ in range(128)]
    assert first == repeat and patch_rng.bit_generator.state == patch_state


@pytest.mark.parametrize("old_model,old_policy", [(True, False), (False, True)])
def test_checkpoint_cannot_silently_reuse_old_model_or_anchor_policy(tmp_path, monkeypatch, old_model, old_policy):
    torch = pytest.importorskip("torch")
    config = dict(learning.CONFIG)
    if old_policy:
        config.pop("anchor_sampling")
        config.pop("anchor_component_rng")
    checkpoint = tmp_path/"prior.pt"
    torch.save({"version": learning.VERSION,
                "model_version": "native-membership-separator-unet-2" if old_model else learning.MODEL_VERSION,
                "config": config, "model_sources": pipeline._sources(), "model_runtime": {},
                "selected_step": 2000, "state_dict": {}}, checkpoint)
    monkeypatch.setattr(pipeline, "_runtime", lambda _: pytest.fail("incompatible checkpoint reached runtime"))
    monkeypatch.setattr(learning, "build_model", lambda: pytest.fail("incompatible checkpoint reached model"))
    with pytest.raises(ValueError, match="CHECKPOINT_PROTOCOL_MISMATCH"):
        pipeline._load_model(checkpoint, safe.sha256(checkpoint), torch)
