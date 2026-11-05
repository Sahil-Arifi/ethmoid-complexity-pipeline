import numpy as np
import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_learning as learning


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
