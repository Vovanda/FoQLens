"""One ladder of precision in the code and in docs/precision-regulator.md: no model, CPU."""

import pytest

from foqlens import zones
from foqlens.quant import LADDER, Level


def test_the_code_of_a_rung_is_its_place_on_the_ladder():
    assert [int(lv) for lv in LADDER] == list(range(len(LADDER)))
    assert [lv.rung for lv in LADDER] == list(range(len(LADDER)))
    assert [lv.name for lv in LADDER] == ["ZERO", "D2", "D4", "D6", "D8", "BF16"]


def test_bits_grow_strictly_along_the_ladder():
    bits = [lv.bits for lv in LADDER]
    assert bits == [0, 2, 4, 6, 8, 16]


def test_levels_are_declared_in_code_order():
    # precision.BITS_BY_CODE and SLICES_BY_CODE index by code an array built by iterating Level
    assert [int(lv) for lv in Level] == list(range(len(Level)))


def test_only_the_read_depths_have_slices():
    assert {lv: lv.depth for lv in Level if lv.depth} == {Level.D2: 1, Level.D4: 2, Level.D6: 3, Level.D8: 4}


def test_a_quantizer_off_the_ladder_has_no_rung():
    for lv in (Level.INT8, Level.NF4):
        with pytest.raises(ValueError, match="not a rung"):
            _ = lv.rung


def test_the_deepest_read_of_a_set_of_rungs_is_its_largest_code():
    assert max(Level.D4, Level.BF16, Level.D8) is Level.BF16
    assert min(Level.D2, Level.ZERO, Level.D6) is Level.ZERO


@pytest.mark.parametrize(
    "floor, strength, halo, profile",
    [  # the worked examples of docs/precision-regulator.md, row by row
        (Level.D4, 1.0, False, "D8:0.5 D6:1"),
        (Level.D4, 0.5, False, "D6:1"),
        (Level.D2, 1.0, False, "D8:0.33 D6:0.67 D4:1"),
        (Level.ZERO, 1.0, False, "D8:0.25 D6:0.5 D4:0.75 D2:1"),
        (Level.ZERO, 0.5, False, "D4:0.5 D2:1"),
        (Level.ZERO, 0.5, True, "D4:1 D2:1.5"),  # the halo past the edge (docs, the profile of a zone)
    ],
)
def test_the_ceiling_and_the_profile_are_the_worked_examples_of_the_rules(floor, strength, halo, profile):
    ceiling = zones.ceiling_of(strength, floor)
    stops = zones.even_stops(floor, ceiling, halo=halo)
    assert " ".join(f"{lv.name}:{s:.2g}" for lv, s in stops) == profile


def test_strength_zero_leaves_the_zone_at_the_floor_on_every_floor():
    for floor in LADDER[:-1]:
        assert zones.ceiling_of(0.0, floor) is floor
        assert zones.even_stops(floor, floor) == ()
