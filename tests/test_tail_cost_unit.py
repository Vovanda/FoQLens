"""The costs of the exact tail's codings on hand-made distances: no model, milliseconds."""

import pytest
import torch

from foqlens.tail_cost import WIDTH_BYTE_BITS, conditional_entropy_bits, entropy_bits, exponent, fixed_width_bits


def test_the_entropy_of_equal_symbols_is_zero_and_of_equally_frequent_ones_their_bits():
    assert entropy_bits(torch.full((100,), 7)) == 0
    assert entropy_bits(torch.arange(16).repeat(10)) == pytest.approx(4)


def test_the_entropy_given_a_context_that_names_the_symbol_is_zero():
    symbols = torch.arange(8).repeat(4)
    assert conditional_entropy_bits(symbols, symbols) == 0
    assert conditional_entropy_bits(symbols, torch.zeros_like(symbols)) == pytest.approx(3)


def test_a_group_costs_its_width_and_the_byte_of_its_width():
    z = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]])  # widths 2 and 0 in groups of 4
    assert fixed_width_bits(z, 4) == pytest.approx(((2 + 0) / 2 * 4 + WIDTH_BYTE_BITS) / 4)


def test_the_exponent_is_the_field_of_the_float():
    assert exponent(torch.tensor([1.0, 2.0, 0.5], dtype=torch.bfloat16)).tolist() == [127, 128, 126]
    assert exponent(torch.tensor([1.0], dtype=torch.float32)).tolist() == [127]
