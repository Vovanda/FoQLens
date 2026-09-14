"""The weight map and the expert zones on it, on made-up masks: no model."""

import math

import numpy as np
import pytest

from foqlens import zones as zn
from foqlens.quant import Level
from foqlens.weight_map import coactivation_map

OUTSIDE = (-0.1, 1.5, math.nan)


def two_groups(n_questions: int = 60, per_group: int = 50, seed: int = 0) -> np.ndarray:
    """Masks [questions, 2 * per_group]: the first group lights up on half the questions, the second on the other half."""
    rng = np.random.default_rng(seed)
    on = np.arange(n_questions) < n_questions // 2
    a = np.where(on[:, None], 1.0, 0.0) + 0.1 * rng.standard_normal((n_questions, per_group))
    b = np.where(on[:, None], 0.0, 1.0) + 0.1 * rng.standard_normal((n_questions, per_group))
    return np.concatenate([a, b], axis=1)


def test_blocks_that_light_up_together_lie_together_on_the_map():
    xy = coactivation_map(two_groups())
    a, b = xy[:50], xy[50:]
    between = np.linalg.norm(a.mean(0) - b.mean(0))
    assert between > 5 * max(a.std(0).max(), b.std(0).max())


def test_the_map_ignores_the_scale_of_a_block_and_is_deterministic():
    masks = two_groups()
    scaled = masks.copy()
    scaled[:, 3] = 100 * scaled[:, 3] + 7
    assert np.allclose(coactivation_map(masks), coactivation_map(scaled))
    assert np.array_equal(coactivation_map(masks), coactivation_map(masks))


def grid_map(side: int = 40) -> np.ndarray:
    g = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1).reshape(-1, 2)
    return g.astype(float)


def test_find_zones_finds_both_peaks_and_a_wider_peak_has_a_wider_radius():
    xy = grid_map()
    wide = np.exp(-np.sum((xy - [10, 10]) ** 2, axis=1) / (2 * 5.0**2))
    narrow = np.exp(-np.sum((xy - [30, 28]) ** 2, axis=1) / (2 * 2.0**2))
    found = zn.find_zones(wide + narrow, xy)
    assert len(found.radii) == 2
    near = [np.linalg.norm(found.centers - c, axis=1).argmin() for c in ([10, 10], [30, 28])]
    assert np.linalg.norm(found.centers[near[0]] - [10, 10]) < 2 and np.linalg.norm(found.centers[near[1]] - [30, 28]) < 2
    assert found.radii[near[0]] > found.radii[near[1]]


def test_a_focus_area_outside_zero_to_one_is_refused_where_it_enters():
    from foqlens.layouts import FixedZones, graded_zone_layout

    xy = grid_map(4)
    one = zn.Zones(centers=np.array([[1.0, 1.0]]), radii=np.array([1.0]))
    for bad in OUTSIDE:
        with pytest.raises(ValueError):
            zn.check_focus_area(bad)
        with pytest.raises(ValueError):
            graded_zone_layout("z", FixedZones(one), bad, 1.0, xy)
    assert zn.check_focus_area(0.0) == 0.0 and zn.check_focus_area(1.0) == 1.0


def test_own_zones_lift_their_topic_and_the_paired_topics_zones_do_not():
    from foqlens.layouts import OtherZones, OwnZones, TopicMeans, TopicZones, graded_zone_layout

    xy = grid_map(12)
    field_a = np.exp(-np.sum((xy - [3, 3]) ** 2, axis=1) / 4.0)
    field_b = np.exp(-np.sum((xy - [9, 9]) ** 2, axis=1) / 4.0)
    means = TopicMeans(np.stack([field_a, field_a, field_b, field_b]), ("a", "a", "b", "b"))
    topics = TopicZones(means, {"a": "b", "b": "a"}, xy)
    idx = np.array([0, 2])
    top = int(zn.ceiling_of(1.0, Level.D4))
    own = graded_zone_layout("own", OwnZones(topics), 0.5, 1.0, xy).levels(idx)
    other = graded_zone_layout("other", OtherZones(topics), 0.5, 1.0, xy).levels(idx)
    center_a = np.linalg.norm(xy - [3, 3], axis=1).argmin()
    assert own[0, center_a] == top and other[0, center_a] != top


def test_a_zone_layout_is_its_parts_and_a_new_part_is_a_new_class():
    from foqlens.layouts import GradedLevels, LiftField, ZoneLayout

    xy = grid_map(10)

    class Everywhere:  # a new zone source: one zone at every question's own place, no change to ZoneLayout
        def zones(self, index: int) -> zn.Zones:
            return zn.Zones(centers=np.array([[float(index), float(index)]]), radii=np.array([1.0]))

    top = zn.ceiling_of(1.0, Level.D4)
    rule = GradedLevels(Level.D4, zn.even_stops(Level.D4, top))
    codes = ZoneLayout("new", Everywhere(), LiftField(0.5), rule, xy).levels(np.array([0, 9]))
    corner = {i: np.linalg.norm(xy - [i, i], axis=1).argmin() for i in (0, 9)}
    assert codes[0, corner[0]] == top and codes[1, corner[9]] == top


def test_random_zones_keep_count_and_radii():
    xy = grid_map(10)
    real = zn.Zones(centers=np.array([[1.0, 1.0], [8.0, 8.0]]), radii=np.array([2.0, 1.0]))
    fake = zn.random_zones(real, xy, np.random.default_rng(0))
    assert fake.centers.shape == real.centers.shape and np.array_equal(fake.radii, real.radii)
    assert not np.array_equal(fake.centers, real.centers)
