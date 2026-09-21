"""The block of maps: every oracle's map of every question at every tolerance."""

import numpy as np
import pytest

from foqlens import maps
from foqlens.quant import Level


def test_a_looser_tolerance_lifts_the_bounds():
    ratios = np.array([[1.0], [0.53], [0.055]])
    tight, loose = maps.bounds(ratios, 1.0), maps.bounds(ratios, 2.0)
    assert (loose > tight).all()
    assert np.allclose(tight.ravel(), [0.5, 1 / 1.53, 1 / 1.055])


def test_a_group_takes_the_coarsest_rung_its_demand_allows():
    ratios = np.array([[1.0], [0.53], [0.055]])
    codes = [int(Level.D2), int(Level.D4), int(Level.D6)]
    demand = np.array([[0.4], [0.6], [0.9], [0.99]])
    got = maps.read_map(demand, ratios, 1.0, codes, int(Level.D8))
    assert got.ravel().tolist() == [int(Level.D2), int(Level.D4), int(Level.D6), int(Level.D8)]


def test_a_label_names_the_oracle_and_the_tolerance():
    assert maps.label("pooled", 1.0) == "pooled@k1"
    assert maps.label("drop", 2.5) == "drop@k2.5"


def block(tmp_path):
    ratios = np.array([[1.0, 1.0], [0.53, 0.53], [0.055, 0.055]])
    demand = np.array([[[0.4, 0.99]], [[0.6, 0.9]]])  # two oracles, one question, two groups
    codes = [int(Level.D2), int(Level.D4), int(Level.D6)]
    ks = np.array([1.0, 2.0])
    levels = np.stack([np.stack([maps.read_map(d, ratios, k, codes, int(Level.D8)) for k in ks]) for d in demand])
    return maps.Maps(corpus=np.array(["c"]), ids=np.array(["1"]), groups=np.array(["0.mlp", "0.attention"]),
                     oracles=np.array(["a", "b"]), ks=ks, demand=demand, levels=levels, ratios=ratios,
                     eps=np.array([[0.1], [0.2]]))


def test_the_flat_layouts_are_named_one_for_one(tmp_path):
    flat = block(tmp_path).flat()
    assert sorted(flat) == ["a@k1", "a@k2", "b@k1", "b@k2"]
    assert all(v.shape == (1, 2) for v in flat.values())


def test_the_block_survives_a_round_trip(tmp_path):
    kept = block(tmp_path)
    got = maps.Maps.read(kept.save(tmp_path / "maps.npz"))
    assert (got.levels == kept.levels).all() and got.flat().keys() == kept.flat().keys()


def test_every_map_of_the_block_is_what_its_bounds_give(tmp_path):
    kept = block(tmp_path)
    codes = [int(Level.D2), int(Level.D4), int(Level.D6)]
    for i in range(len(kept.oracles)):
        for j, k in enumerate(kept.ks):
            want = maps.read_map(kept.demand[i], kept.ratios, float(k), codes, int(Level.D8))
            assert (kept.levels[i, j] == want).all()


def test_the_three_likenesses_answer_different_questions():
    from foqlens.quant import Level

    bits = {int(Level.D2): 2.0, int(Level.D4): 4.0, int(Level.D6): 6.0, int(Level.D8): 8.0}
    weights = np.array([10.0, 1.0])
    base = int(Level.D2)
    same = np.array([[int(Level.D8), int(Level.D2)]])
    one_step = np.array([[int(Level.D6), int(Level.D2)]])       # the same group lifted, one rung lower
    elsewhere = np.array([[int(Level.D2), int(Level.D8)]])      # the lift moved onto the light group

    # a rung apart: the strictest measure says nothing is the same, the shape says everything is
    assert maps.rung_agreement(same, one_step) == 0.5
    assert maps.raised_jaccard(same, one_step, base) == 1.0
    assert maps.bits_apart(same, one_step, bits, weights) == pytest.approx(2 * 10 / 11)

    # the lift carried elsewhere: the shape now disagrees, and the memory apart is the largest
    assert maps.raised_jaccard(same, elsewhere, base) == 0.0
    assert maps.bits_apart(same, elsewhere, bits, weights) > maps.bits_apart(same, one_step, bits, weights)


def test_two_maps_that_lift_nothing_are_alike():
    from foqlens.quant import Level

    flat = np.full((2, 3), int(Level.D2))
    assert (maps.raised_jaccard(flat, flat, int(Level.D2)) == 1.0).all()


def point(oracle, bounds, cost, answered, hard):
    return maps.ScalePoint(oracle=oracle, bounds=bounds, cost=cost, answered=answered, hard=hard)


def test_an_oracles_own_scale_is_the_cheapest_of_those_that_hold_the_answer():
    points = [point("drop", (0.5, 0.6, 0.9), 0.60, 10, 5),   # holds, dear
              point("drop", (0.5, 0.8, 0.9), 0.40, 9, 5),    # holds, cheap - this one
              point("drop", (0.7, 0.8, 0.9), 0.30, 8, 4)]    # cheapest, but does not hold
    assert maps.own_scale(points, answered=9, hard=5).bounds == (0.5, 0.8, 0.9)
    assert maps.own_scale(points, answered=11, hard=5) is None


def test_a_point_that_holds_more_or_costs_less_is_not_charged_for_it():
    own = point("drop", (0.5, 0.8, 0.9), 0.40, 9, 5)
    assert maps.scale_loss(point("drop", (0.5, 0.6, 0.9), 0.50, 7, 4), own) == (2, 1, 0.1)
    assert maps.scale_loss(point("drop", (0.5, 0.6, 0.9), 0.30, 10, 5), own) == (0, 0, 0.0)


def test_the_least_worst_scale_is_the_smallest_worst_loss_not_the_smallest_mean():
    # at X two oracles lose nothing and the third loses three answers; at Y all three lose one. The mean loss is
    # one either way, and only the worst loss tells them apart
    x, y, own_c = (0.4, 0.6, 0.9), (0.5, 0.7, 0.9), (0.6, 0.8, 0.9)
    points = [point("a", x, 0.40, 10, 5), point("a", y, 0.40, 9, 5),
              point("b", x, 0.40, 10, 5), point("b", y, 0.40, 9, 5),
              point("c", x, 0.40, 7, 5), point("c", y, 0.40, 9, 5), point("c", own_c, 0.30, 10, 5)]
    bounds, losses = maps.least_worst_scale(points, answered=7, hard=5)
    assert bounds == y
    assert losses == {"a": (1, 0, 0.0), "b": (1, 0, 0.0), "c": (1, 0, 0.1)}


def test_a_field_no_point_of_which_holds_the_answer_is_left_out_of_the_charge():
    points = [point("a", (0.5, 0.6, 0.9), 0.40, 10, 5), point("b", (0.5, 0.6, 0.9), 0.40, 2, 0)]
    bounds, losses = maps.least_worst_scale(points, answered=9, hard=5)
    assert bounds == (0.5, 0.6, 0.9) and list(losses) == ["a"]
