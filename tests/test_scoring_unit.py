"""Mask-source building blocks on synthetic tensors: no model, CPU."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foqlens.scoring import (
    attention_received,
    batch_token_weights,
    block_norms,
    block_scores,
    center_positions,
    gini,
    normalized_entropy,
    sequence_losses,
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


def test_gini_and_entropy_bounds():
    flat = np.ones(100)
    peak = np.zeros(100)
    peak[7] = 1.0
    assert gini(flat) == pytest.approx(0.0)
    assert gini(peak) == pytest.approx(0.99)
    assert normalized_entropy(flat) == pytest.approx(1.0)
    assert normalized_entropy(peak) == pytest.approx(0.0)
