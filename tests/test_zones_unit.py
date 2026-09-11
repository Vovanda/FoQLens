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


def test_a_focus_area_or_a_precision_share_outside_zero_to_one_is_refused_where_it_enters():
    from foqlens.layouts import ZoneLayout

    xy = grid_map(4)
    one = zn.Zones(centers=np.array([[1.0, 1.0]]), radii=np.array([1.0]))
    weights = np.ones(len(xy))
    for bad in OUTSIDE:
        with pytest.raises(ValueError):
            zn.check_focus_area(bad)
        with pytest.raises(ValueError):
            zn.log_sharpness(xy, one, bad)
        with pytest.raises(ValueError):
            zn.budget_bits(bad)
        with pytest.raises(ValueError):
            ZoneLayout("fixed", "backbone", bad, 0.25, xy, weights, fixed=one)
        with pytest.raises(ValueError):
            ZoneLayout("fixed", "backbone", 0.5, bad, xy, weights, fixed=one)
    assert zn.check_focus_area(0.0) == 0.0 and zn.check_focus_area(1.0) == 1.0


def test_precision_share_spends_from_the_background_to_the_centers():
    assert (zn.budget_bits(0.0), zn.budget_bits(0.25), zn.budget_bits(1.0)) == (4.0, 5.0, 8.0)


def test_the_focus_area_sets_the_radius_from_zero_to_infinity():
    xy = grid_map(10)
    one = zn.Zones(centers=np.array([[2.0, 2.0]]), radii=np.array([1.0]))
    d = np.linalg.norm(xy - [2, 2], axis=1)
    assert np.all(zn.log_sharpness(xy, one, 1.0) == 0)  # focus_area 1: infinite radius, no mask
    assert np.allclose(zn.log_sharpness(xy, one, 0.5), -d)  # focus_area 0.5: the base radius
    assert np.allclose(zn.log_sharpness(xy, one, 0.25), -3 * d)  # R = r f / (1 - f) = r / 3
    assert np.allclose(zn.log_sharpness(xy, one, 0.0), -d)  # focus_area 0: only the order is left
    empty = zn.Zones(centers=np.zeros((0, 2)), radii=np.zeros(0))
    assert np.all(zn.log_sharpness(xy, empty, 0.5) == 0)  # no zone: no mask


def test_layout_holds_the_precision_share_and_puts_the_top_level_at_the_center():
    xy = grid_map(20)
    weights = np.full(len(xy), 64 * 256)
    one = zn.Zones(centers=np.array([[5.0, 5.0]]), radii=np.array([3.0]))
    tiebreak = np.random.default_rng(0).permutation(len(xy))
    bits = zn.budget_bits(0.25)
    for focus_area in (1.0, 0.7, 0.5, 0.2, 0.0):
        codes = zn.layout_at_budget(zn.log_sharpness(xy, one, focus_area), weights, bits, tiebreak, hard_edge=focus_area == 0.0)
        mean = np.mean([Level(int(c)).bits for c in codes])
        block_step = (4 if focus_area == 0.0 else 2) / len(xy)  # a hard edge moves a block D4 -> D8 in one step
        assert bits - block_step <= mean <= bits + block_step, focus_area
        if focus_area < 1.0:
            assert codes[np.linalg.norm(xy - [5, 5], axis=1).argmin()] == Level.D8
    hard = zn.layout_at_budget(zn.log_sharpness(xy, one, 0.0), weights, bits, tiebreak, hard_edge=True)
    assert set(np.unique(hard).tolist()) == {int(Level.D4), int(Level.D8)}


def test_zone_layouts_share_the_precision_share_and_own_puts_d8_on_its_topic():
    from foqlens.layouts import TopicMeans, TopicZones, ZoneLayout

    xy = grid_map(12)
    weights = np.full(len(xy), 64)
    field_a = np.exp(-np.sum((xy - [3, 3]) ** 2, axis=1) / 4.0)
    field_b = np.exp(-np.sum((xy - [9, 9]) ** 2, axis=1) / 4.0)
    means = TopicMeans(np.stack([field_a, field_a, field_b, field_b]), ("a", "a", "b", "b"))
    topics = TopicZones(means, {"a": "b", "b": "a"}, xy)
    idx = np.array([0, 2])
    own = ZoneLayout("own", "pooled", 0.5, 0.25, xy, weights, topics)
    layouts = {
        "own": own.levels(idx),
        "other": ZoneLayout("other", "pooled", 0.5, 0.25, xy, weights, topics).levels(idx),
        "random": ZoneLayout("random", "pooled", 0.5, 0.25, xy, weights, topics).levels(idx),
        "uniform": ZoneLayout("uniform", "-", 1.0, 0.25, xy, weights).levels(idx),
    }
    for name, codes in layouts.items():
        mean = np.array([[Level(int(c)).bits for c in row] for row in codes]).mean(axis=1)
        assert np.all(np.abs(mean - 5.0) <= 2 / len(xy) + 1e-9), name  # one block's step, up to rounding
    center_a = np.linalg.norm(xy - [3, 3], axis=1).argmin()
    assert layouts["own"][0, center_a] == Level.D8 and layouts["other"][0, center_a] != Level.D8
    assert own.name == "zone_own_pooled_fa0.50_ps0.250"
    assert ZoneLayout("uniform", "-", 1.0, 0.25, xy, weights).name == "zone_uniform_ps0.250"


def test_random_zones_keep_count_and_radii():
    xy = grid_map(10)
    real = zn.Zones(centers=np.array([[1.0, 1.0], [8.0, 8.0]]), radii=np.array([2.0, 1.0]))
    fake = zn.random_zones(real, xy, np.random.default_rng(0))
    assert fake.centers.shape == real.centers.shape and np.array_equal(fake.radii, real.radii)
    assert not np.array_equal(fake.centers, real.centers)
