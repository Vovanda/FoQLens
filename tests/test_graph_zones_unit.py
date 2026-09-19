"""Zones on the block graph, their reach and their levels in rungs (#4, #19): no model, CPU."""

import numpy as np
import pytest
import torch

from foqlens import graph_zones as bz
from foqlens import zones
from foqlens.layouts import GraphZoneLayout, EqualStrength, QuantileLevels, QueryGraphZones
from foqlens.metric import StillSurface, geodesic, neighbour_table
from foqlens.quant import LADDER, Level


class PlaneMetric:
    """Blocks at points of a plane, the Euclidean distance between them: a metric with a picture, for tests."""

    def __init__(self, coords: np.ndarray):
        self.coords = torch.as_tensor(coords, dtype=torch.float32)
        self.n_blocks = len(coords)
        self.device = self.coords.device

    def distances(self, sources: torch.Tensor) -> torch.Tensor:
        return torch.cdist(self.coords[torch.as_tensor(sources)], self.coords)

    def diameter(self) -> float:
        return float(self.distances(torch.arange(self.n_blocks)).max())


SIDE = 30
GRID = np.stack(np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij"), axis=-1).reshape(-1, 2).astype(float)
PLANE = PlaneMetric(GRID)
TABLE = neighbour_table(PLANE, k=8)[0].numpy()
WEIGHTS = np.ones(len(GRID))


def bump(center, width: float) -> np.ndarray:
    return np.exp(-np.sum((GRID - center) ** 2, axis=1) / (2 * width**2))


TWO_BUMPS = bump([8, 8], 4.0) + bump([22, 20], 1.5)


def nearest(point) -> int:
    return int(np.linalg.norm(GRID - point, axis=1).argmin())


def test_both_peaks_are_found_and_the_wider_one_has_the_wider_radius():
    found = bz.find_graph_zones(TWO_BUMPS, PLANE, TABLE, WEIGHTS)
    assert len(found.centers) == 2
    wide, narrow = (found.centers == nearest([8, 8])).argmax(), (found.centers == nearest([22, 20])).argmax()
    assert found.centers[wide] == nearest([8, 8]) and found.centers[narrow] == nearest([22, 20])
    assert found.radii[wide] > found.radii[narrow]


def test_a_weaker_top_inside_a_stronger_hill_is_not_a_zone():
    ripple = bump([15, 15], 5.0) + 0.02 * bump([17, 15], 0.6)  # a small top on the flank of one hill
    assert len(bz.find_graph_zones(ripple, PLANE, TABLE, WEIGHTS).centers) == 1


def test_the_radius_is_the_smallest_that_holds_the_mass():
    d = PLANE.distances(torch.tensor([nearest([15, 15])]))[0].numpy()
    for mass in (1.0, 13.0, 50.0, 200.0):
        r = bz.mass_radius(d, WEIGHTS, mass)
        assert WEIGHTS[d <= r].sum() >= mass
        assert WEIGHTS[d < r].sum() < mass


def one_zone(radius: float = 5.0) -> bz.GraphZones:
    return bz.GraphZones(centers=np.array([nearest([15, 15])]), radii=np.array([radius]))


def test_a_lift_is_one_at_the_center_and_nothing_at_the_reach():
    zone = one_zone()
    lifts = bz.zone_lifts(zone, np.array([5.0]), PLANE)[0]
    d = PLANE.distances(torch.as_tensor(zone.centers))[0].numpy()
    assert lifts[zone.centers[0]] == 1.0
    assert np.all(lifts[d >= 5.0] == 0) and np.all(lifts[d < 5.0] > 0)


def test_the_reach_is_the_center_alone_at_zero_and_the_whole_network_at_one():
    zone = one_zone()
    assert np.all(bz.zone_lifts(zone, bz.EqualReach(1.0, PLANE.diameter()).radii(zone), PLANE) == 1)
    center = bz.zone_lifts(zone, bz.EqualReach(0.0, PLANE.diameter()).radii(zone), PLANE)[0]
    assert center[zone.centers[0]] == 1 and np.count_nonzero(center) == 1
    half = bz.zone_lifts(zone, bz.EqualReach(0.5, PLANE.diameter()).radii(zone), PLANE)[0]
    d = PLANE.distances(torch.as_tensor(zone.centers))[0].numpy()
    assert np.all(half[d >= 0.5 * PLANE.diameter()] == 0) and np.all(half[d < 0.5 * PLANE.diameter()] > 0)


def test_along_the_graph_a_zone_is_an_arbitrary_figure_not_a_ball():
    # a U of blocks: the tops of its two arms lie 6 apart on the plane and 24 apart along the U
    left = [(0.0, y) for y in range(9, 0, -1)]
    bottom = [(float(x), 0.0) for x in range(0, 7)]
    right = [(6.0, float(y)) for y in range(1, 10)]
    u = PlaneMetric(np.array(left + bottom + right))
    table, lengths = neighbour_table(u, k=2)
    along = geodesic(table, lengths)
    zone = bz.GraphZones(centers=np.array([0]), radii=np.array([8.0]))
    top_of_right = len(left) + len(bottom) + len(right) - 1
    straight = bz.zone_lifts(zone, np.array([8.0]), u)[0]
    graph = bz.zone_lifts(zone, np.array([8.0]), along)[0]
    assert straight[top_of_right] > 0 and graph[top_of_right] == 0
    # down its own arm the lift falls by 1/8 a block and ends at the block exactly 8 away; nothing further along the U
    assert graph[: len(left) - 1].min() > 0 and np.all(graph[len(left) - 1:] == 0)


def test_lifts_do_not_fall_as_the_focus_area_grows():
    zone = one_zone()
    lifts = [bz.zone_lifts(zone, bz.EqualReach(f, PLANE.diameter()).radii(zone), PLANE)[0] for f in (0.05, 0.2, 0.4, 0.7, 0.9)]
    assert all(np.all(b >= a) for a, b in zip(lifts, lifts[1:]))


FLOOR_CEILINGS = [(Level.D4, Level.D8), (Level.D4, Level.D6), (Level.D2, Level.D8), (Level.ZERO, Level.D8)]


@pytest.mark.parametrize("floor, ceiling", FLOOR_CEILINGS)
def test_rungs_with_one_ceiling_are_rule_five_on_the_even_profile(floor, ceiling):
    lifts = np.random.default_rng(0).uniform(-0.6, 0.9, size=(3, 500)).clip(0, None)
    stops = zones.even_stops(floor, ceiling)
    summed = zones.levels_from_lift(np.minimum(lifts.sum(axis=0), 1.0), floor, ceiling, stops)
    strongest = zones.levels_from_lift(lifts.max(axis=0), floor, ceiling, stops)
    assert np.array_equal(zones.levels_from_rungs(lifts, [ceiling] * 3, floor, "sum"), summed)
    assert np.array_equal(zones.levels_from_rungs(lifts, [ceiling] * 3, floor, "max"), strongest)


def test_zones_of_their_own_strength_stay_between_the_floor_and_the_highest_ceiling():
    lifts = np.random.default_rng(1).uniform(0, 1, size=(3, 400))
    ceilings = zones.zone_ceilings(np.array([1.0, 0.5, 0.0]), 1.0, Level.D2)
    assert [c.name for c in ceilings] == ["D8", "D4", "D2"]
    codes = zones.levels_from_rungs(lifts, ceilings, Level.D2)
    assert codes.min() >= int(Level.D2) and codes.max() <= int(Level.D8)
    alone = zones.levels_from_rungs(lifts[2:], ceilings[2:], Level.D2)
    assert np.all(alone == int(Level.D2))  # a zone of strength 0 lifts nothing


def test_full_strength_everywhere_is_one_ceiling_for_all():
    assert zones.zone_ceilings(np.ones(4), 0.5, Level.D4) == [zones.ceiling_of(0.5, Level.D4)] * 4
    with pytest.raises(ValueError, match="strength"):
        zones.zone_ceilings(np.array([1.2]), 1.0, Level.D4)


def bits(codes: np.ndarray) -> float:
    return float(np.mean([Level(int(c)).bits for c in codes.ravel()]))


def layout(focus_area: float, focus_strength: float, ladder=zones.READ_LEVELS) -> GraphZoneLayout:
    source = QueryGraphZones(TWO_BUMPS[None], StillSurface(TABLE, PLANE), WEIGHTS)
    reach = bz.EqualReach(focus_area, PLANE.diameter())
    return GraphZoneLayout("graph", source, reach, StillSurface(TABLE, PLANE), Level.D4, focus_strength, ladder=ladder)


def test_a_graph_layout_lifts_its_centers_to_the_ceiling_and_grows_with_f_and_g():
    codes = layout(0.3, 1.0).levels(np.array([0]))[0]
    assert codes[nearest([8, 8])] == int(Level.D8) and codes.min() == int(Level.D4)
    with_source = layout(0.3, 1.0, LADDER[1:]).levels(np.array([0]))[0]  # a run that reads its source weights
    assert with_source[nearest([8, 8])] == int(Level.BF16)
    areas = [bits(layout(f, 1.0).levels(np.array([0]))) for f in (0.1, 0.3, 0.5, 0.7)]
    assert areas == sorted(areas) and areas[0] < areas[-1]
    strengths = [bits(layout(0.3, g).levels(np.array([0]))) for g in (0.0, 0.5, 1.0)]
    assert strengths == sorted(strengths) and strengths[0] == Level.D4.bits


def test_a_weak_zone_never_rises_past_its_own_ceiling_beside_a_strong_one():
    # a weak zone (ceiling D4 on a D2 floor) at full lift, and a strong one (D8) barely reaching the same block
    lifts = np.array([[1.0, 1.0, 0.0], [0.0, 0.05, 1.0]])
    codes = zones.levels_from_rungs(lifts, [Level.D4, Level.D8], Level.D2)
    assert codes[0] == int(Level.D4)  # the weak zone alone: its own ceiling
    assert codes[2] == int(Level.D8)  # the strong zone alone: its own
    assert codes[1] == int(Level.D6)  # both reach it: 1 rung + 0.05 x 3 rungs, under the strong zone's ceiling
    weak_only = zones.levels_from_rungs(lifts[:1], [Level.D4], Level.D2)
    assert weak_only.max() == int(Level.D4)


def test_the_rungs_are_counted_on_the_run_s_ladder():
    short = (Level.D2, Level.D4)  # a copy cut to D4
    codes = zones.levels_from_rungs(np.ones((1, 4)), [Level.D4], Level.D2, ladder=short)
    assert codes.tolist() == [int(Level.D4)] * 4
    skipping = (Level.D2, Level.D6)  # a ladder without D4: a zone steps from the floor to D6
    half = zones.levels_from_rungs(np.array([[1.0, 0.4]]), [Level.D6], Level.ZERO, ladder=skipping)
    assert half.tolist() == [int(Level.D6), int(Level.D2)]


def test_a_layout_on_the_ladder_of_depths_never_rises_past_its_top():
    depths = tuple(lv for lv in zones.READ_LEVELS if lv.depth)
    codes = layout(0.3, 1.0, depths).levels(np.array([0]))[0]
    assert codes[nearest([8, 8])] == int(Level.D8) and codes.max() == int(Level.D8)
    control = QuantileLevels("control", TWO_BUMPS[None], 0.3, 1.0, Level.D4, ladder=depths).levels(np.array([0]))
    assert control.max() == int(Level.D8)


def test_the_per_block_control_spends_the_same_share_of_blocks_on_any_question():
    scores = np.random.default_rng(2).standard_normal((3, 1000)) * np.array([[1.0], [10.0], [0.1]])
    for f in (0.1, 0.3, 1.0):
        codes = QuantileLevels("control", scores, f, 1.0, Level.D4).levels(np.arange(3))
        above = (codes > int(Level.D4)).mean(axis=1)
        assert np.allclose(above, f, atol=2e-3)  # a preset budget: the query does not decide it
    top = QuantileLevels("top", scores, 0.0, 1.0, Level.D4).levels(np.arange(3))
    assert (top > int(Level.D4)).sum(axis=1).tolist() == [1, 1, 1]  # f = 0: the top block alone, as a zone's center
    assert top[np.arange(3), scores.argmax(axis=1)].tolist() == [int(Level.D8)] * 3


def test_the_ladder_the_rungs_map_onto_is_the_one_of_quant():
    codes = zones.levels_from_rungs(np.ones((1, 3)), [Level.BF16], Level.ZERO, ladder=LADDER[1:])
    assert codes.tolist() == [int(LADDER[-1])] * 3


def test_the_proportional_reach_keeps_the_mean_at_f_d_and_each_zone_s_own_width():
    two = bz.GraphZones(centers=np.array([0, 1]), radii=np.array([1.0, 3.0]))
    radii = bz.ProportionalReach(0.2, 10.0).radii(two)
    assert radii.mean() == pytest.approx(0.2 * 10.0) and radii[1] / radii[0] == pytest.approx(3.0)
    one = bz.GraphZones(centers=np.array([0]), radii=np.array([4.0]))
    assert bz.ProportionalReach(0.2, 10.0).radii(one).tolist() == bz.EqualReach(0.2, 10.0).radii(one).tolist()
    flat = bz.GraphZones(centers=np.array([0, 1]), radii=np.zeros(2))
    assert bz.ProportionalReach(0.2, 10.0).radii(flat).tolist() == [2.0, 2.0]
    assert np.all(np.isinf(bz.ProportionalReach(1.0, 10.0).radii(two)))
    assert np.all(bz.ProportionalReach(0.0, 10.0).radii(two) == 0)
