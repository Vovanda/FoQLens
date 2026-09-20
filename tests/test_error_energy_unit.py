"""The oracle of the quantization error's energy: per block, over valid tokens, never negative."""

import torch

from foqlens.error_energy import block_energy


def test_the_energy_is_the_squared_norm_of_a_blocks_rows_over_valid_tokens_only():
    change = torch.tensor([[[1.0, 2.0, 0.0, 3.0], [5.0, 5.0, 5.0, 5.0]]])  # [batch 1, seq 2, out 4]
    valid = torch.tensor([[1.0, 0.0]])  # the second token is padding
    energy = block_energy(change, valid, block_rows=2)
    assert energy.tolist() == [[5.0, 9.0]]  # rows {0, 1}: 1 + 4; rows {2, 3}: 0 + 9


def test_no_error_is_no_energy_and_a_ragged_last_block_is_padded_with_zeros():
    valid = torch.ones(1, 3)
    assert torch.all(block_energy(torch.zeros(1, 3, 5), valid, 2) == 0)
    assert block_energy(torch.ones(1, 3, 5), valid, 2).tolist() == [[2.0, 2.0, 1.0]]
