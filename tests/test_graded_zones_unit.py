"""The graded zone layout: the invariants of zones.py and the layout it gives (docs/quantization-filter.md)."""

import numpy as np
import pytest

from foqlens import zones
from foqlens.layouts import GradedLevels, LiftField, ZoneLayout, graded_zone_layout
from foqlens.quant import Level
from foqlens.zones import Zones

# A line of blocks from 0 to 1, with one zone in the middle: the profile is read along it.
LINE = np.linspace(0, 1, 101)[:, None]
MIDDLE = Zones(centers=np.array([[0.5]]), radii=np.array([0.2]))
AS_FOUND = 0.5  # focus area at which a radius is the base radius


def line_layout(**kwargs) -> np.ndarray:
    layout = graded_zone_layout("t", _FixedSource(MIDDLE), coords=LINE, **kwargs)
    return layout.levels(np.array([0]))[0]


class _FixedSource:
    def __init__(self, zones_: Zones):
        self._zones = zones_

    def zones(self, index: int) -> Zones:
        return self._zones


def test_a_zone_lifts_to_one_at_its_center_and_to_nothing_at_its_reach():
    lift = zones.precision_lift(LINE, MIDDLE, AS_FOUND)
    assert lift[50] == pytest.approx(1.0)
    assert lift[30] == pytest.approx(0.0) and lift[70] == pytest.approx(0.0)
    assert np.all(lift[31:50] > 0) and np.all(np.diff(lift[:51]) >= 0)


def test_focus_area_0_lifts_nothing_and_focus_area_1_lifts_everything():
    assert np.all(zones.precision_lift(LINE, MIDDLE, 0.0) == 0)
    assert np.all(zones.precision_lift(LINE, MIDDLE, 1.0) == 1)


def test_a_reach_past_one_lifts_blocks_outside_the_radius():
    inside = zones.precision_lift(LINE, MIDDLE, AS_FOUND, reach=1.0)
    halo = zones.precision_lift(LINE, MIDDLE, AS_FOUND, reach=1.5)
    assert inside[25] == 0 and halo[25] > 0
    assert np.all(halo >= inside)


def test_lifts_add_up_but_never_below_the_strongest_one():
    pair = Zones(centers=np.array([[0.4], [0.6]]), radii=np.array([0.2, 0.2]))
    summed = zones.precision_lift(LINE, pair, AS_FOUND, combine="sum")
    strongest = zones.precision_lift(LINE, pair, AS_FOUND, combine="max")
    assert np.all(summed >= strongest)
    assert summed[50] > strongest[50]  # the isthmus between the two rises
    assert summed.max() <= 1.0


def test_an_unknown_combine_is_refused():
    with pytest.raises(ValueError, match="combine"):
        zones.precision_lift(LINE, MIDDLE, AS_FOUND, combine="mean")


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


def test_a_single_zone_reads_its_stepped_profile_exactly():
    codes = line_layout(focus_area=AS_FOUND, focus_strength=1.0, floor=Level.D4)
    # stops 0.5 and 1 of a radius of 0.2: D8 within 0.1 of the center, D6 to 0.2, D4 outside
    assert set(codes[:30]) == {int(Level.D4)}
    assert set(codes[46:55]) == {int(Level.D8)}
    assert int(Level.D6) in set(codes[31:45])
    assert np.all(np.abs(np.diff(codes.astype(int))) <= 1)


def test_focus_strength_0_leaves_the_whole_map_at_the_floor():
    assert set(line_layout(focus_area=AS_FOUND, focus_strength=0.0, floor=Level.D4)) == {int(Level.D4)}


def test_focus_area_0_leaves_the_whole_map_at_the_floor():
    assert set(line_layout(focus_area=0.0, focus_strength=1.0, floor=Level.D2)) == {int(Level.D2)}


def test_no_block_is_read_below_the_floor():
    for floor in (Level.ZERO, Level.D2, Level.D4):
        codes = line_layout(focus_area=AS_FOUND, focus_strength=1.0, floor=floor, halo=floor is Level.ZERO)
        assert codes.min() >= int(floor)


def test_an_overlap_lowers_no_block():
    pair = Zones(centers=np.array([[0.4], [0.6]]), radii=np.array([0.2, 0.2]))
    one = graded_zone_layout("one", _FixedSource(MIDDLE), 0.5, 1.0, LINE).levels(np.array([0]))[0]
    two = graded_zone_layout("two", _FixedSource(pair), 0.5, 1.0, LINE).levels(np.array([0]))[0]
    assert np.all(two >= np.minimum(one, int(Level.D4)))
    assert two[50] >= one[50]


def test_memory_grows_with_focus_area_and_with_focus_strength():
    def bits(**kwargs) -> float:
        codes = line_layout(**kwargs)
        return float(np.mean([Level(int(c)).bits for c in codes]))

    areas = [bits(focus_area=f, focus_strength=1.0, floor=Level.D2) for f in (0.1, 0.3, 0.5, 0.7)]
    strengths = [bits(focus_area=AS_FOUND, focus_strength=g, floor=Level.D2) for g in (0, 0.5, 0.75, 1.0)]
    assert areas == sorted(areas) and areas[0] < areas[-1]
    assert strengths == sorted(strengths) and strengths[0] < strengths[-1]


def test_a_profile_that_is_not_a_falloff_is_refused():
    lift = np.array([1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="at least one stop"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ())
    with pytest.raises(ValueError, match="does not fall"):
        zones.levels_from_lift(lift, Level.D4, Level.D6, ((Level.D6, 0.5), (Level.D8, 1.0)))
    with pytest.raises(ValueError, match="do not grow"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, 0.8), (Level.D6, 0.5)))
    with pytest.raises(ValueError, match="below the floor"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, 0.5), (Level.D2, 1.0)))
    with pytest.raises(ValueError, match="outside"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D8, 0.5), (Level.D6, 2.0)))
    with pytest.raises(ValueError, match="not at the ceiling"):
        zones.levels_from_lift(lift, Level.D4, Level.D8, ((Level.D6, 1.0),))


def test_the_layout_is_still_its_three_parts():
    layout = graded_zone_layout("t", _FixedSource(MIDDLE), AS_FOUND, 1.0, LINE, floor=Level.D2)
    assert isinstance(layout, ZoneLayout)
    assert isinstance(layout.field, LiftField) and isinstance(layout.rule, GradedLevels)
    by_parts = ZoneLayout("t", layout.source, LiftField(AS_FOUND, 1.0), GradedLevels(Level.D2, layout.rule.stops), LINE)
    assert np.array_equal(layout.levels(np.array([0])), by_parts.levels(np.array([0])))


def test_moved_zones_keep_the_figure_and_change_only_its_place():
    """The honest control: same count, same radii, same distances between the zones, another place."""
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(400, 2))
    figure = Zones(centers=np.array([[0.0, 0.0], [0.4, 0.1], [-0.2, 0.5]]), radii=np.array([0.2, 0.3, 0.1]))
    moved = zones.moved_zones(figure, coords, rng)
    pairwise = lambda c: np.linalg.norm(c[:, None] - c[None], axis=-1)
    assert len(moved.radii) == len(figure.radii)
    assert np.allclose(moved.radii, figure.radii)
    assert np.allclose(pairwise(moved.centers), pairwise(figure.centers))
    assert not np.allclose(moved.centers, figure.centers)


def test_a_moved_figure_can_be_landed_where_it_costs_the_same():
    """Blocks are not spread evenly, so the landing is chosen by how much weight the figure covers."""
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(300, 2)) * 0.1                      # a crowded patch
    sparse = rng.normal(size=(100, 2)) * 1.5 + np.array([4, 4])  # and a thin one
    coords = np.concatenate([dense, sparse])
    weights = np.ones(len(coords))
    figure = Zones(centers=np.array([[4.0, 4.0]]), radii=np.array([0.6]))   # sitting in the thin patch
    covered = lambda z: zones._covered(z, coords, weights, 1.0)
    plain = [covered(zones.moved_zones(figure, coords, np.random.default_rng(s))) for s in range(8)]
    matched = [covered(zones.moved_zones(figure, coords, np.random.default_rng(s), weights)) for s in range(8)]
    want = covered(figure)
    assert np.mean([abs(c - want) for c in matched]) < np.mean([abs(c - want) for c in plain])


def test_moved_zones_of_no_zones_stay_empty():
    empty = Zones(centers=np.zeros((0, 2)), radii=np.zeros(0))
    assert len(zones.moved_zones(empty, LINE, np.random.default_rng(0)).radii) == 0
