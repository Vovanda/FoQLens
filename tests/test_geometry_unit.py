"""Step 2+ geometry on synthetic masks: no model, CPU."""

import numpy as np
import pytest

from foqlens import geometry as g

N = 1000


def bundle(lo: int, hi: int, n: int = N) -> np.ndarray:
    x = np.zeros(n)
    x[lo:hi] = 1.0
    return x


def test_fit_two_recovers_an_additive_mix():
    rng = np.random.default_rng(0)
    a, b = rng.random(N), rng.random(N)
    fit = g.fit_two(2 * a + 3 * b, a, b)
    assert fit["coef_a"] == pytest.approx(2) and fit["coef_b"] == pytest.approx(3)
    assert fit["r2"] == pytest.approx(1.0)


def test_top_set_size():
    assert g.top_set(np.arange(N, dtype=float), 0.05).sum() == 50
    assert g.top_set(np.arange(10, dtype=float), 0.01).sum() == 1


def test_junction_is_empty_for_a_union_and_full_for_a_new_zone():
    a, b = bundle(0, 50), bundle(100, 150)
    union = g.junction(a + b, a, b, frac=0.05)
    assert union["observed"] == 0.0
    own = g.junction(bundle(500, 550), a, b, frac=0.05)
    assert own["observed"] == 1.0
    assert own["chance"] == pytest.approx(0.95**2)


def test_isthmus_counts_blocks_raised_on_all_three_outside_the_tops():
    a, b = bundle(0, 50), bundle(100, 150)
    a[200:260] = 0.1  # weakly raised on both components and on the mix: a bridge
    b[200:260] = 0.1
    mixed = a + b
    res = g.isthmus(mixed, a, b, frac=0.05)
    assert res["outside_blocks"] == N - 100
    assert res["observed"] == pytest.approx(60 / (N - 100))
    assert res["observed"] > res["chance"]


def test_concentration_bounds():
    flat = np.ones(N)
    peak = np.zeros(N)
    peak[3] = 1.0
    assert g.concentration(flat)["participation"] == pytest.approx(1.0)
    assert g.concentration(peak)["participation"] == pytest.approx(1 / N)
    assert g.concentration(-flat)["participation"] == 0.0  # the positive part of a negative mask is empty


def test_hierarchy_counts_and_expectation_cover_every_block():
    rng = np.random.default_rng(1)
    means = {d: rng.random(N) for d in "abcd"}
    res = g.hierarchy(means, frac=0.1)
    assert sum(res["observed"]) == N
    assert sum(res["expected"]) == pytest.approx(N)
    shared = bundle(0, 100)
    stacked = g.hierarchy({d: shared + 0.01 * rng.random(N) for d in "abcd"}, frac=0.1)
    assert stacked["observed"][4] == 100  # the same top blocks in all four domains


def test_linearity_reports_both_sides():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=N), rng.normal(size=N)
    rep = {"m": a + b, "a": a, "b": b}
    mask = {"m": rng.normal(size=N), "a": a, "b": b}
    res = g.linearity(rep, mask, "m", "a", "b")
    assert res["representation"]["fit"]["r2"] == pytest.approx(1.0)
    assert res["mask"]["fit"]["r2"] < 0.05
    assert set(res["mask"]["cos"]) == {"m|a", "m|b", "a|b"}
