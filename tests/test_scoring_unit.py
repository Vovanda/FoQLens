"""Mask-source building blocks on synthetic tensors: no model, CPU."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foqlens.quant import Level
from foqlens.scoring import (
    QuantGapRecorder,
    answer_losses,
    attention_received,
    batch_token_weights,
    block_norms,
    block_scores,
    center_positions,
    gini,
    normalized_entropy,
    sequence_losses,
    taylor_block_magnitudes,
    taylor_block_scores,
    token_weights,
    weighted_block_scores,
)


def test_centers_are_the_highest_scores_and_never_the_first_token():
    scores = torch.tensor([100.0, 1.0, 5.0, 1.0, 3.0, 1.0])  # <bos> with a huge score must still be skipped
    assert center_positions(scores, k=2).tolist() == [2, 4]


def test_centers_are_capped_by_the_sequence_length():
    assert center_positions(torch.randn(3), k=10).numel() == 2


def test_attention_received_is_divided_by_the_queries_allowed_to_look():
    seq, heads = 4, 2
    attn = torch.zeros(heads, seq, seq)
    attn[:, :, 0] = 1.0  # every query looks only at the first token
    assert attention_received(attn).tolist() == [1.0, 0.0, 0.0, 0.0]
    attn = torch.eye(seq).expand(heads, seq, seq)  # every query looks only at itself
    assert attention_received(attn).tolist() == pytest.approx([1 / 4, 1 / 3, 1 / 2, 1.0])


def test_token_weights_per_mode():
    hidden = torch.ones(5, 8)
    hidden[3] *= 2
    pooled = token_weights("pooled", hidden, None)
    assert pooled[0] == 0 and pooled.sum().item() == pytest.approx(1.0)
    assert token_weights("norm", hidden, None, k=1).tolist() == [0, 0, 0, 1.0, 0]
    attn = torch.zeros(2, 5, 5)
    attn[:, 4, 2] = 1.0
    assert token_weights("attention", hidden, attn, k=1).tolist() == [0, 0, 1.0, 0, 0]
    with pytest.raises(ValueError):
        token_weights("attention", hidden, None)


def test_batch_token_weights_ignore_padding():
    hidden = torch.ones(2, 5, 8)
    weights = batch_token_weights("pooled", hidden, None, lengths=[5, 3])
    assert weights[0].tolist() == pytest.approx([0, 0.25, 0.25, 0.25, 0.25])
    assert weights[1].tolist() == pytest.approx([0, 0.5, 0.5, 0, 0])


def test_block_scores_follow_the_weighted_tokens():
    output = torch.zeros(5, 130)  # 3 blocks of 64, the last one has 2 rows
    output[1, 64:128] = 1.0  # block 1 responds at a weighted token
    output[3, 0:64] = 9.0  # block 0 responds at a token with zero weight
    weights = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0])
    scores = block_scores(output, weights, block_rows=64)
    assert scores.tolist() == pytest.approx([0.0, 4.0, 0.0])  # norm 8 at one of two equally weighted tokens


def test_batched_block_scores_equal_per_sequence():
    torch.manual_seed(0)
    output = torch.randn(3, 6, 130)
    weights = torch.rand(3, 6)
    batched = weighted_block_scores(block_norms(output, 64), weights)
    for b in range(3):
        torch.testing.assert_close(batched[b], block_scores(output[b], weights[b], 64))


def test_taylor_block_scores_sum_grad_times_output_per_block_without_bos():
    output = torch.zeros(3, 130)
    grad = torch.zeros(3, 130)
    output[0, :], grad[0, :] = 100.0, 100.0  # <bos> is never counted
    output[1, 0:64], grad[1, 0:64] = 2.0, 0.5  # block 0: 64 * 1.0 = 64
    output[2, 0:64], grad[2, 0:64] = 1.0, -0.25  # block 0 again: -16, the sum is signed before abs
    output[1, 128:130], grad[1, 128:130] = -3.0, 1.0  # block 2: -6 -> 6
    assert taylor_block_scores(output, grad, block_rows=64).tolist() == pytest.approx([48.0, 0.0, 6.0])


def test_taylor_magnitudes_do_not_cancel_where_the_signed_sum_does():
    output = torch.zeros(3, 128)
    grad = torch.zeros(3, 128)
    output[1, 0:64], grad[1, 0:64] = 1.0, 1.0  # block 0: +64 at one token
    output[2, 0:64], grad[2, 0:64] = 1.0, -1.0  # and -64 at another: |sum| = 0, sum |.| = 128
    output[1:, 64:128], grad[1:, 64:128] = 2.0, 0.5  # block 1: every term +1, the two forms agree
    assert taylor_block_scores(output, grad, 64).tolist() == pytest.approx([0.0, 128.0])
    assert taylor_block_magnitudes(output, grad, 64).tolist() == pytest.approx([128.0, 128.0])


def test_taylor_magnitudes_are_never_below_the_signed_sum():
    torch.manual_seed(0)
    output, grad = torch.randn(2, 7, 130), torch.randn(2, 7, 130)
    valid = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0, 0]])
    signed = taylor_block_scores(output, grad, 64, valid)
    magnitudes = taylor_block_magnitudes(output, grad, 64, valid)
    assert torch.all(magnitudes >= signed - 1e-5)
    same_sign = taylor_block_magnitudes(output.abs(), grad.abs(), 64, valid)
    torch.testing.assert_close(same_sign, taylor_block_scores(output.abs(), grad.abs(), 64, valid))


def test_taylor_block_scores_skip_padding_in_a_batch():
    output = torch.ones(2, 4, 64)
    grad = torch.ones(2, 4, 64)
    valid = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]])
    assert taylor_block_scores(output, grad, 64, valid).flatten().tolist() == [192.0, 64.0]


def test_sequence_losses_are_per_sequence_means_over_real_targets():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    ids = torch.tensor([[1, 2, 3, 4], [1, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    losses = sequence_losses(logits, ids, mask)
    torch.testing.assert_close(losses[0], F.cross_entropy(logits[0, :3], ids[0, 1:]))
    torch.testing.assert_close(losses[1], F.cross_entropy(logits[1, :1], ids[1, 1:2]))


class _Head(torch.nn.Module):
    """A model with only a head and no softcapping: what answer_losses reads of a model."""

    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        text = type("Text", (), {"final_logit_softcapping": None})()
        self.config = type("Config", (), {"get_text_config": lambda self: text})()


def test_answer_losses_read_the_answer_tokens_only_and_never_padding():
    torch.manual_seed(0)
    model = _Head(8, 11)
    hidden = torch.randn(2, 5, 8)
    enc = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]),
           "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])}
    starts = [3, 2]  # prompts of 3 and 2 tokens: answers [4, 5] and [8]
    with torch.no_grad():
        logits = model.lm_head(hidden).float()
        losses = answer_losses(model, hidden, enc, starts)
    torch.testing.assert_close(losses[0], F.cross_entropy(logits[0, 2:4], torch.tensor([4, 5])))
    torch.testing.assert_close(losses[1], F.cross_entropy(logits[1, 1:2], torch.tensor([8])))


class _TwoLevels(torch.nn.Linear):
    """A linear layer with two read levels: the weight and the weight plus a known error."""

    block_rows = 2

    def __init__(self, error: torch.Tensor):
        super().__init__(error.shape[1], error.shape[0], bias=False)
        self.error = error

    def read_weight(self, level):
        return self.weight.detach() + (self.error if level is Level.D2 else 0)


def test_the_quant_gap_is_the_gradient_times_the_rung_gaps_change_of_the_output():
    torch.manual_seed(0)
    error = torch.zeros(4, 3)
    error[2:] = torch.randn(2, 3)  # the first block of rows reads the same weights at both levels
    module = _TwoLevels(error)
    x = torch.randn(1, 5, 3, requires_grad=True)
    valid = torch.tensor([[0, 1, 1, 1, 1]])
    with QuantGapRecorder({"m": module}, valid, Level.D2, Level.D8) as rec:
        out = module(x)
        out.pow(2).sum().backward()
    grad = 2 * module(x).detach()
    moved = x.detach() @ error.T
    expected = (grad * moved * valid[..., None]).unflatten(-1, (2, 2)).sum(dim=(1, 3))
    torch.testing.assert_close(rec.change["m"], expected)
    assert rec.change["m"][0, 0] == 0


def test_gini_and_entropy_bounds():
    flat = np.ones(100)
    peak = np.zeros(100)
    peak[7] = 1.0
    assert gini(flat) == pytest.approx(0.0)
    assert gini(peak) == pytest.approx(0.99)
    assert normalized_entropy(flat) == pytest.approx(1.0)
    assert normalized_entropy(peak) == pytest.approx(0.0)
