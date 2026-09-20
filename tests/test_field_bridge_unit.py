"""The bridge from the address to a field: the mapper of the filter over the lifts, the price of memory as one shift,
and the bridges themselves - static, topic, nearest and ridge."""

import numpy as np
import pytest

from foqlens.field_bridge import (
    BRIDGES,
    EVEN_STOPS,
    bits_of,
    cosine,
    excess,
    lift_of,
    shift_for_bits,
    to_levels,
)
from foqlens.quant import Level

D2, D4, D6, D8 = (int(x) for x in (Level.D2, Level.D4, Level.D6, Level.D8))


def test_the_mapper_cuts_the_lift_at_what_every_rung_is_worth():
    # the bench's scale (precision_field.FIELD_VALUE): D4 from 0.8, D6 from 0.9, D8 at 1.0, the base under all of them
    lift = np.array([[0.0, 0.5, 0.79, 0.8, 0.9, 1.0]])
    assert to_levels(lift)[0].tolist() == [D2, D2, D2, D4, D6, D8]
    assert EVEN_STOPS == pytest.approx((0.0, 0.1, 0.2))
    # a profile of two stops: D8 and D6 kept, D4 skipped, so a lift that would have read D4 falls to the base
    assert to_levels(lift, stops=(0.0, 0.1))[0].tolist() == [D2, D2, D2, D2, D6, D8]


def test_a_level_and_its_lift_go_back_and_forth():
    levels = np.array([[D2, D4, D6, D8]], dtype=np.uint8)
    assert to_levels(lift_of(levels))[0].tolist() == levels[0].tolist()


def test_the_shift_buys_bits_and_the_price_found_spends_what_was_asked():
    rng = np.random.default_rng(0)
    field = rng.random((20, 8)) * 0.6 + 0.4  # 0.4 ... 1.0: the rungs above the base start at 0.8
    weights = np.full(8, 1000.0)
    spent = [float(bits_of(to_levels(field, s), weights).mean()) for s in (-0.2, 0.0, 0.2)]
    assert spent[0] < spent[1] < spent[2]
    shift = shift_for_bits(field, weights, bits=5.0)
    assert float(bits_of(to_levels(field, shift), weights).mean()) == pytest.approx(5.0, abs=0.3)


def test_the_static_bridge_gives_the_mean_map_and_the_nearest_one_finds_its_own_question():
    maps = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    address = np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    topics = np.array(["a", "b", "a"])
    static = BRIDGES["static"]().fit(address, maps, topics)
    assert static.read(address, topics)[0].tolist() == maps.mean(axis=0).tolist()
    nearest = BRIDGES["nearest"](k=1).fit(address, maps, topics)
    assert nearest.read(address, topics).tolist() == maps.tolist()
    topic = BRIDGES["topic"]().fit(address, maps, topics)
    assert topic.read(address, np.array(["b"]))[0].tolist() == [0.0, 1.0]


def test_a_zone_is_the_hill_around_a_peak_of_any_shape_and_never_a_ball():
    layers = np.repeat(np.arange(10), 2)
    address = np.zeros((2, 20))
    address[0, [6, 8, 10]] = [0.6, 1.0, 0.2]  # a hill leaning to one side: 0.2 is under half the peak
    address[1, 8], address[1, 18] = 1.0, 0.5  # two peaks, the second half as high
    zones = BRIDGES["zones"](layers=layers, quantile=0.9, slope=0.5).fit(np.zeros((1, 20)), None, None)
    lift = zones.read(address, np.array(["a", "b"]))
    assert lift[0][8] == pytest.approx(1.0) and lift[0][6] == pytest.approx(0.6)  # the hill holds on one side
    assert lift[0][10] == 0.0 and lift[0][0] == 0.0  # it breaks off where the address falls, and never spreads
    assert lift[1][18] == pytest.approx(0.5) and lift[1][8] == pytest.approx(1.0)  # every zone keeps its own height
    assert (lift <= 1.0).all()
    kept = zones.hill(address[0], 8)
    assert kept.sum() == 2 and kept[8] and kept[6]


def test_the_blocks_next_to_one_are_its_own_module_and_its_place_in_the_layers_next_to_it():
    from foqlens.field_bridge import neighbours_of_blocks

    layer = np.array([0, 0, 0, 1, 1, 1])
    kind = np.array(["mlp"] * 3 + ["mlp"] * 3)
    near = neighbours_of_blocks(layer, kind)
    assert sorted(x for x in near[1] if x >= 0) == [0, 2, 4]  # its own module either side, and its place in layer 1
    assert sorted(x for x in near[3] if x >= 0) == [0, 4]
    # a zone over the blocks grows by that table: it has a width inside a layer, which groups do not have
    address = np.zeros((1, 6))
    address[0, [1, 2]] = [1.0, 0.8]
    zones = BRIDGES["zones"](layers=layer, quantile=0.9, slope=0.5, near=near).fit(np.zeros((1, 6)), None, None)
    kept = zones.hill(address[0], 1)
    assert kept.tolist() == [False, True, True, False, False, False]


def test_the_ridge_recovers_a_linear_relation_and_the_excess_removes_what_every_question_shares():
    rng = np.random.default_rng(1)
    address = rng.random((60, 5))
    matrix = rng.random((5, 3))
    maps = address @ matrix
    ridge = BRIDGES["ridge"](relative_ridge=1e-8).fit(address, maps, np.zeros(60))
    assert np.allclose(ridge.read(address, np.zeros(60)), maps, atol=1e-6)
    assert np.allclose(excess(address).mean(axis=0), 0.0)
    assert cosine(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 5.0]]))[0].tolist() == [1.0, 0.0]
