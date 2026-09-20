"""The precision field: the rungs' ratios, the levels a threshold reads, the search of the threshold, the measured
reference and its raising."""

import numpy as np
import pytest

from foqlens.precision_field import (
    FIELD_VALUE,
    alone_levels,
    field_levels,
    find_threshold,
    measure_field,
    priced_levels,
    raised,
    reference_layouts,
    rung_costs,
    rung_ratios,
    search_threshold,
    thresholds,
    values,
)
from foqlens.quant import Level

from test_group_oracle_unit import _Layouts, _Pacer

D2, D4, D6, D8 = (int(x) for x in (Level.D2, Level.D4, Level.D6, Level.D8))


def test_the_ratios_are_the_median_share_of_the_coarsest_rungs_error_and_one_where_it_makes_none():
    d2 = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, np.nan]])
    d4 = np.array([[0.1, 0.0, 0.4], [0.3, 0.0, 1.0]])
    ratios = rung_ratios([d2, d4])
    assert ratios[0].tolist() == [1.0, 1.0, 1.0]
    assert ratios[1] == pytest.approx([0.125, 1.0, 0.2])  # median of 0.1 and 0.15; the NaN question left out


def test_a_group_reads_the_coarsest_rung_within_the_threshold_and_the_top_where_none_is():
    costs = rung_costs(np.array([1.0, 10.0, -3.0]), np.array([[1.0] * 3, [0.1] * 3, [0.01] * 3]))
    assert field_levels(costs, np.array([0.5]))[0].tolist() == [D4, D6, D2]  # 1*0.1 fits, 10*0.01 fits; <0 costs 0
    assert field_levels(costs, np.array([2.0]))[0].tolist() == [D2, D4, D2]
    assert field_levels(costs, np.array([0.001]))[0].tolist() == [D8, D8, D2]


def test_a_priced_rung_is_taken_only_when_what_it_saves_is_worth_its_bits():
    costs = rung_costs(np.array([1.0, 10.0]), np.array([[1.0, 1.0], [0.1, 0.1], [0.01, 0.01]]))
    at_zero = priced_levels(costs, np.array([0.0]))[0]
    assert at_zero.tolist() == [D8, D8]  # memory free: read everything at the top
    at_huge = priced_levels(costs, np.array([1e9]))[0]
    assert at_huge.tolist() == [D2, D2]  # memory dear: read everything at the coarsest rung
    ladder = priced_levels(costs, np.array([0.0, 0.05, 0.5, 1e9])).astype(int)
    assert (np.diff(ladder[:, 0]) <= 0).all() and (np.diff(ladder[:, 1]) <= 0).all()  # never finer at a larger price
    # the group that stands to lose ten times as much holds its rung where the cheaper one has already let go
    assert ladder[2, 1] > ladder[2, 0]


def test_the_price_of_a_rung_counts_what_a_group_holds():
    costs = rung_costs(np.array([1.0, 1.0]), np.array([[1.0, 1.0], [0.1, 0.1], [0.01, 0.01]]))
    heavy = priced_levels(costs, np.array([0.05]), weights=np.array([10.0, 1.0]))[0]
    assert heavy[0] <= heavy[1]  # the same field, but the heavier group pays more for the same rung


def test_at_the_noise_models_ratios_the_field_is_the_knapsack_of_section_9():
    """The precision field generalizes layouts.knapsack_levels: with the rungs' ratios of the noise model (16 times
    per rung) the two give the same levels, and the field's threshold is the knapsack's price."""
    from foqlens.layouts import knapsack_levels
    from foqlens.quant import LADDER

    rng = np.random.default_rng(3)
    sensitivity = rng.random(50) * 100
    ratios = np.stack([np.full(50, 16.0 ** -k) for k in range(3)])
    price = 0.7
    ladder = tuple(lv for lv in LADDER if Level.D2 <= lv <= Level.D8)
    theirs = knapsack_levels(sensitivity[None], price, Level.D2, ladder)[0]
    ours = field_levels(rung_costs(sensitivity, ratios), np.array([price]))[0]
    assert np.array_equal(ours, theirs)


def test_the_levels_coarsen_with_the_threshold_from_the_top_to_the_coarsest_rung():
    rng = np.random.default_rng(0)
    costs = rung_costs(rng.random(20), np.array([np.ones(20), rng.random(20), rng.random(20) * 0.1]))
    candidates = thresholds(costs)
    levels = field_levels(costs, candidates)
    assert (np.diff(levels.astype(int), axis=0) <= 0).all()
    assert (levels[0] == D8).all() and (levels[-1] == D2).all()


def test_the_search_finds_the_largest_holding_index_in_few_rounds_and_only_returns_what_it_read():
    for last in (0, 1, 7, 99, 100):
        read, rounds = set(), []

        def holds(probes, last=last):
            read.update(probes.tolist())
            rounds.append(len(probes))
            return probes <= last

        index = search_threshold(holds, 101, width=9)
        assert index == last and (index == 0 or index in read)
        assert len(rounds) <= 3 and max(rounds) <= 9
    # holding that is not monotone still ends on an index that held
    assert search_threshold(lambda p: (p % 3) == 0, 50, width=4) % 3 == 0


def test_the_reference_reads_every_group_alone_at_every_rung_and_raises_the_layout_by_whole_rungs():
    groups = np.array([0, 0, 1])
    rows = reference_layouts(groups, 2)
    assert rows.shape == (1 + 3 * 2, 3) and (rows[0] == D8).all()
    assert rows[1].tolist() == [D2, D2, D8] and rows[2 + 2 * 2].tolist() == [D8, D8, D6]
    nll = np.array([[0.5, 0.0], [0.05, 0.0], [0.0, 0.0]])  # group 0 holds from D4, group 1 at D2
    assert alone_levels(nll, target=0.0, tolerance=0.1).tolist() == [D4, D2]
    assert raised(np.array([D2, D4, D8], dtype=np.uint8), 1).tolist() == [D4, D6, D8]
    assert (raised(np.array([D2, D6], dtype=np.uint8), 3) == D8).all()
    assert values(np.array([D8, D6, D4, D2])).tolist() == [FIELD_VALUE[Level(x)] for x in (D8, D6, D4, D2)]


def _answer_nll(ctl, cost):
    def nll(_model, _tok, prompts, _answers):
        import torch

        return torch.tensor([cost(lay) for lay in ctl.layout], dtype=torch.float64)

    return nll


def test_the_threshold_of_a_field_holds_the_answer_beside_every_block_at_the_top(monkeypatch):
    import foqlens.group_oracle as go

    groups = np.arange(4)
    loss = {D2: 1.0, D4: 0.1, D6: 0.01, D8: 0.0}
    need = np.array([0.0, 0.04, 0.5, 1.0])  # how much the answer listens to each group
    ctl = _Layouts()
    monkeypatch.setattr(go, "answer_nll", _answer_nll(ctl, lambda lay: sum(n * loss[int(v)] for n, v in zip(need, lay))))
    ratios = np.array([[1.0] * 4, [0.1] * 4, [0.01] * 4])
    found = find_threshold(None, None, ctl, _Pacer(), "p", "a", groups, rung_costs(need, ratios), tolerance=0.05,
                           width=3)
    assert found.nll <= found.target + 0.05 and found.target == 0.0
    # at eps 0.01: 0.004 + 0.005 + 0.01 fits; the next threshold, 0.04, puts group 1 at D2 and 0.055 does not
    assert found.levels.tolist() == [D2, D4, D6, D6]
    assert found.batches >= 2


def test_the_measured_field_raises_the_groups_alone_levels_until_they_hold_together(monkeypatch):
    import foqlens.group_oracle as go

    groups = np.array([0, 0, 1, 2])
    loss = {D2: 1.0, D4: 0.1, D6: 0.01, D8: 0.0}
    need = {0: 0.05, 1: 0.05, 2: 0.05}  # each group alone at D2 holds 0.1; the three together at D2 do not
    ctl = _Layouts()
    monkeypatch.setattr(go, "answer_nll", _answer_nll(ctl, lambda lay: sum(
        need[g] * loss[int(lay[np.flatnonzero(groups == g)[0]])] for g in need)))
    found = measure_field(None, None, ctl, _Pacer(), "p", "a", groups, 3, tolerance=0.1, size=4)
    assert found.alone_levels.tolist() == [D2, D2, D2]
    assert found.steps == 1 and found.levels.tolist() == [D4, D4, D4]
    assert found.joint[0] > 0.1 >= found.joint[1]
