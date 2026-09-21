"""The rules that turn a lift into levels (foqlens.zones, docs/precision-regulator.md rules 2-6), on any metric: CPU."""

import numpy as np
import pytest

from foqlens import zones
from foqlens.quant import Level


def test_the_ceiling_is_the_share_of_the_rungs_above_the_floor():
    assert [zones.ceiling_of(g, Level.D4).name for g in (0, 0.25, 0.5, 0.75, 1)] == ["D4", "D4", "D6", "D6", "D8"]
    assert [zones.ceiling_of(g, Level.ZERO).name for g in (0, 0.25, 0.5, 0.75, 1)] == ["ZERO", "D2", "D4", "D6", "D8"]
    with pytest.raises(ValueError, match="focus strength"):
        zones.ceiling_of(1.5, Level.D4)


def test_the_default_profile_is_even_and_a_halo_goes_past_the_edge():
    assert zones.even_stops(Level.D4, Level.D8) == ((Level.D8, 0.5), (Level.D6, 1.0))
    halo = zones.even_stops(Level.ZERO, Level.D8, halo=True)
    assert [lv for lv, _ in halo] == [Level.D8, Level.D6, Level.D4, Level.D2]
    assert [s for _, s in halo] == pytest.approx([1 / 3, 2 / 3, 1.0, zones.HALO_STOP])


def test_a_profile_that_is_not_a_falloff_is_refused_and_a_ring_past_the_edge_is_not():
    lift = np.array([1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="at least one stop"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ())
    with pytest.raises(ValueError, match="does not fall"):
        zones.levels_from_lift(lift, Level.D4, Level.D6, ((Level.D6, 0.5), (Level.D8, 1.0)))
    with pytest.raises(ValueError, match="do not grow"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, 0.8), (Level.D6, 0.5)))
    with pytest.raises(ValueError, match="below the floor"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, 0.5), (Level.D2, 1.0)))
    with pytest.raises(ValueError, match="not above 0"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, -0.5), (Level.D6, 1.0)))
    with pytest.raises(ValueError, match="not at the ceiling"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D6, 1.0),))
    # docs: a stop above 1 is a ring past the zone's edge, as many as there are such stops - no bound
    zones.levels_from_lift(lift, Level.ZERO, Level.D8, ((Level.D8, 0.5), (Level.D6, 1.0), (Level.D4, 1.2), (Level.D2, 2.5)))


def test_a_lift_reads_the_floor_at_0_the_ceiling_at_1_never_below_and_steps_one_level_a_stop():
    lift = np.linspace(1.0, 0.0, 201)  # from a zone's center out past its reach
    for floor in (Level.ZERO, Level.D2, Level.D4):
        ceiling = zones.ceiling_of(1.0, floor)
        codes = zones.levels_from_lift(lift, floor, ceiling, zones.even_stops(floor, ceiling))
        assert codes[0] == int(ceiling) and codes[-1] == int(floor), floor
        assert codes.min() >= int(floor), floor
        assert np.all(np.diff(codes.astype(int)) <= 0) and np.all(np.diff(codes.astype(int)) >= -1), floor
