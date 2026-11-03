"""Synthetic contracts for the isolated partial-instance prototype."""
from __future__ import annotations

import numpy as np
import pytest

# The application environment excludes the numeric research stack; the pinned
# counting runtime executes this module, including the decoder, without skips.
pytest.importorskip("scipy")

from ethmoid_pipeline.counting import instance_learning as learning


def test_normalization_has_two_declared_windows_and_preserves_input():
    hu = np.array([-2000, -1000, -400, 0, 200, 1000, 3000], np.float32).reshape(1, 1, 7)
    original = hu.copy()
    values = learning.normalize_hu(hu)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values[0].ravel(), [-1, -1, -.4, 0, .2, 1, 1])
    np.testing.assert_allclose(values[1].ravel(), [-1, -1, 0, 2/3, 1, 1, 1])
    np.testing.assert_array_equal(hu, original)
    assert not np.shares_memory(hu, values)


@pytest.mark.parametrize("hu", [np.zeros((3, 3)), np.empty((0, 2, 2)), np.full((2, 2, 2), np.nan)])
def test_normalization_rejects_invalid_source(hu):
    with pytest.raises(ValueError):
        learning.normalize_hu(hu)


def test_affinity_supervision_requires_explicit_same_or_different_identity():
    ids = np.array([1, 1, -1, 0, 2, 3, 3], np.int32).reshape(7, 1, 1)
    targets, known = learning.build_targets(ids, distinct_pairs=[(2, 3)])
    np.testing.assert_array_equal(known[0, :, 0, 0], [1, 1, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(targets[0, :, 0, 0], [1, 1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(known[1, :, 0, 0], [1, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(targets[1, :, 0, 0], [1, 0, 0, 0, 0, 1, 0])
    assert not known[4, 0, 0, 0]  # IDs 1/2 are different numbers, not supported different-cell labels.
    assert not known[2].any() and not known[3].any()
    assert not known[5].any() and not known[6].any()


def test_foreground_only_observation_does_not_invent_an_instance():
    ids = np.full((5, 5, 5), -1, np.int32)
    fg = ids.copy()
    fg[2, 2, 2], fg[3, 2, 2] = 1, 0
    targets, known = learning.build_targets(ids, foreground_labels=fg)
    assert known[0].sum() == 2 and targets[0].sum() == 1
    assert not known[1:].any()
    with pytest.raises(ValueError):
        learning.build_targets(np.ones((5, 5, 5), np.int32), foreground_labels=fg)


@pytest.mark.parametrize("ids", [np.ones((3, 3)), np.ones((2, 2, 2), bool),
                                   np.full((2, 2, 2), -2, np.int32),
                                   np.ones((2, 2, 2), np.float32)])
def test_invalid_instance_annotation_rejected(ids):
    with pytest.raises(ValueError):
        learning.build_targets(ids)


@pytest.mark.parametrize("pair", [(1, 1), (0, 1), (True, 2), (1., 2), (1, 2, 3)])
def test_invalid_distinction_rejected(pair):
    with pytest.raises(ValueError):
        learning.build_targets(np.ones((2, 2, 2), np.int32), distinct_pairs=[pair])


def test_each_offset_uses_its_positive_axis_and_has_no_wraparound():
    shape = (7, 8, 9)
    ids = np.ones(shape, np.int32)
    targets, known = learning.build_targets(ids)
    for channel, offset in enumerate(learning.OFFSETS, 1):
        expected = np.zeros(shape, bool)
        for index in np.ndindex(shape):
            target = tuple(a+b for a, b in zip(index, offset))
            expected[index] = all(v < size for v, size in zip(target, shape))
        np.testing.assert_array_equal(known[channel], expected)
        np.testing.assert_array_equal(targets[channel], expected)


@pytest.mark.parametrize("transform", [lambda a: a[::-1], lambda a: a.transpose(1, 2, 0)])
def test_transform_and_rebuild_preserves_instance_partition(transform):
    ids = np.zeros((9, 9, 9), np.int32)
    ids[1:4, 2:7, 2:7] = 1
    ids[5:8, 2:7, 2:7] = 2
    ids = transform(ids).copy()
    targets, _ = learning.build_targets(ids, distinct_pairs=[(1, 2)])
    prediction = learning.decode_probabilities(targets)
    assert np.unique(prediction[prediction > 0]).size == 2
    for value in (1, 2):
        assert np.unique(prediction[ids == value]).size == 1
        assert prediction[ids == value].min() > 0
    assert not prediction[ids == 0].any()
    assert prediction[ids == 1][0] != prediction[ids == 2][0]


def test_continuous_u_is_one_instance_under_exact_supported_affinities():
    ids = np.zeros((9, 9, 9), np.int32)
    ids[1:3, 1:8, 3:6] = 1
    ids[6:8, 1:8, 3:6] = 1
    ids[1:8, 1:3, 3:6] = 1
    targets, _ = learning.build_targets(ids)
    result = learning.decode_probabilities(targets)
    assert set(np.unique(result)) == {0, 1}
    np.testing.assert_array_equal(result > 0, ids > 0)


def test_no_edges_is_all_split_and_all_foreground_edges_is_all_merged():
    probabilities = np.zeros((7, 5, 5, 5), np.float32)
    probabilities[0] = 1
    result = learning.decode_probabilities(probabilities)
    assert len(np.unique(result)) == 125
    probabilities[:] = 1
    result = learning.decode_probabilities(probabilities)
    np.testing.assert_array_equal(result, np.ones((5, 5, 5), np.int32))


def test_inclusive_threshold_and_long_edge_endpoint_rule():
    probabilities = np.zeros((7, 5, 1, 1), np.float32)
    probabilities[0, [0, 4], 0, 0] = .5
    probabilities[4, 0, 0, 0] = .5
    result = learning.decode_probabilities(probabilities)
    assert result[0, 0, 0] == result[4, 0, 0] > 0
    assert not result[1:4].any()
    probabilities[4, 0, 0, 0] = np.nextafter(np.float32(.5), np.float32(0))
    result = learning.decode_probabilities(probabilities)
    assert result[0, 0, 0] != result[4, 0, 0]


def test_empty_foreground_has_zero_labels():
    result = learning.decode_probabilities(np.zeros((7, 3, 3, 3), np.float32))
    assert result.dtype == np.int32 and not result.any()


@pytest.mark.parametrize("values", [np.zeros((6, 3, 3, 3)), np.full((7, 3, 3, 3), np.nan),
                                     np.full((7, 3, 3, 3), 1.1), np.empty((7, 0, 3, 3))])
def test_invalid_probabilities_rejected(values):
    with pytest.raises(ValueError):
        learning.decode_probabilities(values)


def test_masked_loss_balances_available_classes_and_ignores_unknowns():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 7, 1, 1, 4), requires_grad=True)
    targets = torch.full_like(logits, float("nan"))
    known = torch.zeros_like(logits, dtype=torch.bool)
    targets[0, 0, 0, 0] = torch.tensor([1., 0., 0., float("nan")])
    known[0, 0, 0, 0, :3] = True
    loss = learning.masked_balanced_bce(logits, targets, known)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    np.testing.assert_allclose(logits.grad[0, 0, 0, 0].numpy(), [-.25, .125, .125, 0])
    assert not logits.grad[:, 1:].any()


def test_all_unknown_loss_is_differentiable_zero():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 7, 2, 2, 2), requires_grad=True)
    loss = learning.masked_balanced_bce(logits, torch.full_like(logits, float("nan")),
                                        torch.zeros_like(logits, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert not logits.grad.any()


def test_single_class_loss_is_not_given_an_invented_opposite_class():
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 7, 1, 1, 1), 2., requires_grad=True)
    targets = torch.ones_like(logits)
    known = torch.ones_like(logits, dtype=torch.bool)
    loss = learning.masked_balanced_bce(logits, targets, known)
    assert loss.item() == pytest.approx(np.log1p(np.exp(-2)), rel=1e-6)


def test_known_nonbinary_target_is_rejected():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 7, 1, 1, 1))
    with pytest.raises(ValueError):
        learning.masked_balanced_bce(logits, torch.full_like(logits, .5),
                                     torch.ones_like(logits, dtype=torch.bool))


def test_unet_architecture_and_cpu_forward_are_finite():
    torch = pytest.importorskip("torch")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = learning.build_model().eval()
        assert [encoder[0].out_channels for encoder in model.encoders] == [8, 16, 32, 64]
        with torch.no_grad():
            values = model(torch.zeros(1, 2, 16, 16, 16))
        assert values.shape == (1, 7, 16, 16, 16) and torch.isfinite(values).all()
        with pytest.raises(ValueError):
            model(torch.zeros(1, 2, 15, 16, 16))
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("pattern",["random","ties","negative","noncontiguous","offset_ties"])
def test_nonoverlapping_pool_matches_native_cpu_forward_and_first_tie_backward(pattern):
    torch=pytest.importorskip("torch")
    generator=torch.Generator().manual_seed(72)
    values=torch.randn((2,3,4,6,8),generator=generator,dtype=torch.float64)
    if pattern == "ties":
        values.zero_()
    elif pattern == "negative":
        values=-values.abs()-1
    elif pattern == "noncontiguous":
        values=values.transpose(2,3)
        assert not values.is_contiguous()
    elif pattern == "offset_ties":
        values.fill_(-5)
        values[:,:,::2,1::2,::2]=-1
        values[:,:,1::2,::2,1::2]=-1
    actual=values.clone().requires_grad_()
    expected=values.clone().requires_grad_()
    a=learning.deterministic_max_pool3d(actual)
    b=torch.nn.MaxPool3d(2)(expected)
    assert torch.equal(a,b)
    upstream=torch.randn(a.shape,generator=generator,dtype=torch.float64)
    a.backward(upstream); b.backward(upstream)
    assert torch.equal(actual.grad,expected.grad)
    if pattern == "ties":
        mask=torch.zeros_like(actual,dtype=torch.bool); mask[:,:,::2,::2,::2]=True
        assert not actual.grad[~mask].any()


def test_nonoverlapping_pool_rejects_partial_windows():
    torch=pytest.importorskip("torch")
    with pytest.raises(ValueError,match="even spatial sizes"):
        learning.deterministic_max_pool3d(torch.zeros(1,1,3,4,4))
