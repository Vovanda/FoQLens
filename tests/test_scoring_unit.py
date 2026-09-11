"""The naive per-block score on synthetic tensors: no model, CPU."""

import numpy as np
import pytest
import torch

from foqlens.scoring import attention_received, block_scores, center_positions, gini, normalized_entropy, token_weights


def test_centers_are_the_highest_scores_and_never_the_first_token():
    scores = torch.tensor([100.0, 1.0, 5.0, 1.0, 3.0, 1.0])  # <bos> with a huge score must still be skipped
    assert center_positions(scores, k=2).tolist() == [2, 4]


def test_centers_are_capped_by_the_sequence_length():
    assert center_positions(torch.randn(3), k=10).numel() == 2


def test_attention_received_is_divided_by_the_queries_allowed_to_look():
    seq, heads = 4, 2
    attn = torch.zeros(heads, seq, seq)
    attn[:, :, 0] = 1.0  # every query looks only at the first token
    received = attention_received(attn)
    # token 0 is visible to all 4 queries in 2 heads: 8 / (2 * 4) = 1; nobody looks at the others
    assert received.tolist() == [1.0, 0.0, 0.0, 0.0]
    attn = torch.eye(seq).expand(heads, seq, seq)  # every query looks only at itself
    # token j gets 1 per head from one query, and seq - j queries could have looked at it
    assert attention_received(attn).tolist() == pytest.approx([1 / 4, 1 / 3, 1 / 2, 1.0])


def test_token_weights_per_mode():
    hidden = torch.ones(5, 8)
    hidden[3] *= 2
    pooled = token_weights("pooled", hidden, None)
    assert pooled[0] == 0 and pooled.sum().item() == pytest.approx(1.0)
    norm = token_weights("norm", hidden, None, k=1)
    assert norm.tolist() == [0, 0, 0, 1.0, 0]
    attn = torch.zeros(2, 5, 5)
    attn[:, 4, 2] = 1.0
    assert token_weights("attention", hidden, attn, k=1).tolist() == [0, 0, 1.0, 0, 0]
    with pytest.raises(ValueError):
        token_weights("attention", hidden, None)


def test_block_scores_follow_the_weighted_tokens():
    output = torch.zeros(5, 130)  # 3 blocks of 64, the last one has 2 rows
    output[1, 64:128] = 1.0  # block 1 responds at a weighted token
    output[3, 0:64] = 9.0  # block 0 responds at a token with zero weight
    weights = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0])
    scores = block_scores(output, weights, block_rows=64)
    assert scores.shape == (3,)
    assert scores[1] == pytest.approx(8.0 / 2)  # norm 8 at one of two equally weighted tokens
    assert scores[0] == 0 and scores[2] == 0


def test_gini_and_entropy_bounds():
    flat = np.ones(100)
    peak = np.zeros(100)
    peak[7] = 1.0
    assert gini(flat) == pytest.approx(0.0)
    assert gini(peak) == pytest.approx(0.99)
    assert normalized_entropy(flat) == pytest.approx(1.0)
    assert normalized_entropy(peak) == pytest.approx(0.0)
