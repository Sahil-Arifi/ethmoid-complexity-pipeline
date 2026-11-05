import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_learning as learning
from ethmoid_pipeline.counting import instance_experiment as source


def test_targets_do_not_invent_separator_at_core_edges():
    foreground = np.full((5, 6, 7), -1, np.int8)
    foreground[2, 2, 2] = 1
    separator = np.full_like(foreground, -1)
    target, known = learning.targets({"foreground_labels": foreground, "separator_labels": separator})
    assert known[0].sum() == 1 and known[1].sum() == 0
    assert np.count_nonzero(target[1]) == 0


def test_unknown_loss_has_exact_zero_gradient_and_balances_heads():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
    target = torch.zeros_like(logits)
    known = torch.zeros_like(logits, dtype=torch.bool)
    known[0, 0, 0, 0, 0] = True
    known[0, 1, 0, 0, 0] = True
    target[0, 1, 0, 0, 0] = 1
    loss = learning.masked_loss(logits, target, known)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~known]) == 0
    assert logits.grad[0, 0, 0, 0, 0] > 0 > logits.grad[0, 1, 0, 0, 0]
    logits.grad.zero_()
    learning.masked_loss(logits, target, torch.zeros_like(known)).backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_model_reuses_backbone_and_has_only_two_heads():
    torch = pytest.importorskip("torch")
    model = learning.build_model()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 2, 8, 16, 24)))
    assert output.shape == (1, 2, 8, 16, 24)


def test_native_tiling_overlaps_padding_and_constant_probabilities():
    torch = pytest.importorskip("torch")
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 2, *x.shape[2:]), device=x.device)
    hu = np.zeros((11, 18, 21), np.float32)
    result = learning.probabilities(hu, Constant(), torch, device="cpu", patch_size=16, stride=8)
    assert result.shape == (2, *hu.shape)
    assert np.array_equal(result, np.full_like(result, .5))


def test_invalid_targets_fail_before_training():
    with pytest.raises(ValueError, match="BOUNDARY_TARGETS_INVALID"):
        learning.targets({"foreground_labels": np.zeros((2, 2, 2)),
                          "separator_labels": np.zeros((2, 2, 2))})


def test_valid_native_patch_bounds_retain_edge_and_interior_anchors():
    rng = np.random.default_rng(107)
    for length in (64, 65, 129):
        shape = np.asarray((length, length + 1, length + 2))
        for anchor in (np.zeros(3, int), shape - 1, shape // 2):
            for _ in range(64):
                center = learning.sample_patch_center(shape, anchor, rng)
                low = center - 32
                assert np.all(low >= 0) and np.all(low + 64 <= shape)
                assert np.all(anchor >= low) and np.all(anchor < low + 64)
    for shape, anchor, size, generator in (
        ((0, 64, 64), (0, 0, 0), 64, rng), ((64, 64), (0, 0, 0), 64, rng),
        ((64, 64, 64), (-1, 0, 0), 64, rng), ((64, 64, 64), (64, 0, 0), 64, rng),
        ((64, 64, 64), (0., 0., 0.), 64, rng), ((64, 64, 64), (0, 0, 0), 63, rng),
        ((64, 64, 64), (0, 0, 0), 64, None)):
        with pytest.raises(ValueError, match="BOUNDARY_PATCH_SAMPLER_INPUT_INVALID"):
            learning.sample_patch_center(shape, anchor, generator, size)


def test_short_axes_use_high_only_hu_padding_and_unknown_labels():
    shape, anchor = (17, 64, 31), (16, 63, 0)
    hu = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    foreground = np.full(shape, -1, np.int8); foreground[anchor] = 1
    separator = np.full(shape, -1, np.int8); separator[anchor] = 0
    center = learning.sample_patch_center(shape, anchor, np.random.default_rng(11))
    assert np.array_equal(center, [32, 32, 32])
    h = source.patch(hu, center, 64, -1000)
    f = source.patch(foreground, center, 64, -1)
    s = source.patch(separator, center, 64, -1)
    valid = np.zeros((64,) * 3, bool); valid[:17, :64, :31] = True
    assert np.array_equal(h[:17, :64, :31], hu)
    assert np.array_equal(f[:17, :64, :31], foreground)
    assert np.array_equal(s[:17, :64, :31], separator)
    assert np.all(h[~valid] == -1000) and np.all(f[~valid] == -1) and np.all(s[~valid] == -1)
    assert f[anchor] == 1 and s[anchor] == 0


def test_sampled_hu_and_labels_keep_native_alignment_through_flips():
    shape, anchor = (92, 83, 78), np.asarray((45, 41, 39))
    grid = np.indices(shape)
    hu = (grid[0] * 10000 + grid[1] * 100 + grid[2]).astype(np.float32)
    foreground = np.full(shape, -1, np.int8); foreground[tuple(anchor)] = 1
    separator = np.full(shape, -1, np.int8); separator[tuple(anchor)] = 0
    center = learning.sample_patch_center(shape, anchor, np.random.default_rng(37))
    low = center - 32
    expected = hu[tuple(slice(a, a + 64) for a in low)]
    patches = [source.patch(array, center, 64, fill) for array, fill in
               ((hu, -1000), (foreground, -1), (separator, -1))]
    for combination in range(8):
        axes = tuple(axis for axis in range(3) if combination & (1 << axis))
        h, f, s = [np.flip(patch, axes) for patch in patches]
        local = anchor - low
        local = tuple(63 - index if axis in axes else index for axis, index in enumerate(local))
        assert np.array_equal(h, np.flip(expected, axes))
        assert h[local] == hu[tuple(anchor)] and f[local] == 1 and s[local] == 0
        assert np.count_nonzero(f >= 0) == np.count_nonzero(s >= 0) == 1


def test_patch_positions_cover_pooling_phases_with_independent_deterministic_rng():
    main_rng = np.random.default_rng(learning.CONFIG["seed"])
    expected_state = main_rng.bit_generator.state
    global_state = np.random.get_state()
    first = np.random.default_rng(learning.CONFIG["seed"])
    second = np.random.default_rng(learning.CONFIG["seed"])
    shape, anchor = (192, 192, 192), np.asarray((96, 96, 96))
    centers = np.stack([learning.sample_patch_center(shape, anchor, first) for _ in range(4096)])
    repeated = np.stack([learning.sample_patch_center(shape, anchor, second) for _ in range(4096)])
    assert np.array_equal(centers, repeated)
    local_positions = anchor - (centers - 32)
    for axis in range(3):
        assert np.array_equal(np.unique(local_positions[:, axis]), np.arange(64))
        assert np.array_equal(np.unique((centers[:, axis] - 32) % 8), np.arange(8))
    assert main_rng.bit_generator.state == expected_state
    after = np.random.get_state()
    assert after[0] == global_state[0] and np.array_equal(after[1], global_state[1])
    assert after[2:] == global_state[2:]
