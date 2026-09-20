"""The best allocation of section 9: levels by sensitivity per weight against one price of memory."""

import numpy as np

from foqlens.layouts import RUNG_GAIN, KnapsackLevels, knapsack_levels, knapsack_price
from foqlens.quant import Level

LADDER = (Level.D2, Level.D4, Level.D6, Level.D8)


def test_a_block_rises_a_rung_for_every_sixteen_times_its_sensitivity_over_the_price():
    s = np.array([[0.5, 1.0, RUNG_GAIN, RUNG_GAIN ** 2, RUNG_GAIN ** 5]])
    codes = knapsack_levels(s, 1.0, Level.D2, LADDER)
    assert codes.tolist() == [[int(Level.D2), int(Level.D4), int(Level.D6), int(Level.D8), int(Level.D8)]]


def test_the_ends_of_the_price_read_everything_at_the_floor_or_at_the_top():
    s = np.random.default_rng(0).random((3, 50))
    assert np.all(knapsack_levels(s, np.inf, Level.D2, LADDER) == int(Level.D2))
    assert np.all(knapsack_levels(s, 0.0, Level.ZERO, LADDER) == int(Level.D8))


def test_the_price_spends_the_budget_and_more_sensitive_blocks_never_read_coarser():
    rng = np.random.default_rng(1)
    s, weights = rng.lognormal(0, 2, (20, 400)), rng.integers(1, 8, 400).astype(float)
    price = knapsack_price(s, weights, Level.D2, LADDER, budget_bits=4.0)
    codes = KnapsackLevels("k", s, price, Level.D2, LADDER).levels(np.arange(20))
    bits = np.array([lv.bits for lv in Level], dtype=float)[codes] @ weights / weights.sum()
    assert abs(bits.mean() - 4.0) < 0.1 and bits.mean() <= 4.0
    order = np.argsort(s[0])
    assert np.all(np.diff(codes[0][order].astype(int)) >= 0)
