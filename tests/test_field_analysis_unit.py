"""The statistics of the precision fields: the shares of the spread, the agreement of two fields, the patterns, the zone
centres and their profile."""

import numpy as np
import pytest

from foqlens.field_analysis import (
    against_background,
    agreement,
    clusters_against_corpora,
    centres,
    consensus,
    holm,
    jaccard,
    patterns,
    permutation_p,
    profile,
    radius,
    shapes,
    variance_shares,
    zone_stops,
)


def test_the_shares_sum_to_one_and_find_where_the_spread_lies():
    rng = np.random.default_rng(0)
    common = rng.random(10)
    corpus = np.array([0] * 20 + [1] * 20)
    only_common = np.tile(common, (40, 1))
    shares = variance_shares(only_common, corpus)
    assert sum(shares.values()) == pytest.approx(1.0) and shares["common"] == pytest.approx(1.0)
    noisy = only_common + rng.normal(0, 1, (40, 10))
    shares = variance_shares(noisy, corpus)
    assert sum(shares.values()) == pytest.approx(1.0) and shares["pattern"] > 0.5


def test_a_field_agrees_with_itself_in_every_measure():
    rng = np.random.default_rng(1)
    levels = rng.integers(1, 5, (5, 12))
    got = agreement(levels.astype(float), levels.astype(float), levels, levels)
    assert np.allclose(got["spearman"], 1) and np.allclose(got["kappa"], 1) and np.allclose(got["equal"], 1)
    median, band = consensus(np.stack([levels, levels, levels]).astype(float))
    assert np.array_equal(median, levels) and (band == 0).all()


def test_fields_of_two_patterns_need_two():
    rng = np.random.default_rng(2)
    h = np.array([[1.0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]])
    which = np.arange(30) % 2  # every question holds one of the two patterns, at its own height
    heights = (np.eye(2)[which] * (0.5 + rng.random((30, 1)))) @ h
    got = patterns(heights, [1, 2])
    assert got[0]["explained"] < 0.9 and got[1]["explained"] > 0.99


def test_a_single_zone_is_found_at_its_centre_and_its_profile_falls():
    layers = np.repeat(np.arange(10), 2)
    kinds = np.tile(np.array(["attention", "mlp"]), 10)
    value = np.where(kinds == "mlp", 1.0, 0.75) - 0.1 * np.abs(layers - 4)  # the mlp of layer 4 is the peak
    values = np.clip(value, 0.2, 1.0)[None]
    found = centres(values[0], layers, base=0.2)
    assert np.flatnonzero(found).tolist() == [9]  # 4.mlp
    zone = zone_stops(values[0], layers, found, base=0.2, rungs=[1.0, 0.75, 0.5])[0]
    assert zone["peak"] == 1.0 and zone["radius"] == 6  # past 5 layers of these 10 the zone has no group left
    assert zone["stops"][1.0] < zone["stops"][0.75] < zone["stops"][0.5] == 1.0
    curves = profile(values, layers, kinds, found[None], reach=3)
    assert curves["same"][0] == 1.0 and (np.diff(curves["same"]) < 0).all()
    assert radius(curves["same"], 0.75) == 2 and radius(curves["same"], 1.1) == -1


def test_a_permutation_of_labels_that_matter_is_rarely_as_strong():
    x = np.concatenate([np.zeros(20), np.ones(20)])
    labels = np.array([0] * 20 + [1] * 20)

    def gap(lab):
        return float(x[lab == 1].mean() - x[lab == 0].mean())

    assert permutation_p(gap, labels, 200, 0) < 0.01


def test_a_prediction_is_read_against_what_the_mean_of_the_questions_already_gives():
    true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    background = true.mean(axis=0)
    assert against_background(true, true, background)["error"] == 0.0
    plain = against_background(np.tile(background, (3, 1)), true, background)
    assert plain["ratio"] == 1.0  # answering with the background is the background
    half = against_background((true + background) / 2, true, background)
    assert half["ratio"] == 0.5  # halfway to the truth is half the background's error


def test_the_centres_two_fields_agree_on():
    a = np.array([[True, False, True], [False, False, False]])
    b = np.array([[True, True, False], [False, False, False]])
    share = jaccard(a, b)
    assert share[0] == 1 / 3  # one centre of three the two hold together
    assert np.isnan(share[1])  # neither field has a centre: nothing to agree about
    assert jaccard(a, a)[0] == 1.0


def test_the_questions_cluster_along_their_corpora_when_the_fields_hold_the_corpus():
    fields = np.concatenate([np.tile([1.0, 0.0], (6, 1)), np.tile([0.0, 1.0], (6, 1))])  # two groups of six
    along = np.array(["a"] * 6 + ["b"] * 6)  # the corpora are the groups
    assert clusters_against_corpora(fields, along)["rand"] == 1.0
    across = np.array(["a", "b"] * 6)  # the same fields, corpora that cut across them
    assert clusters_against_corpora(fields, across)["rand"] < 0.1


def test_a_flat_layout_is_named_by_the_rung_it_sits_on_and_never_called_a_map():
    levels = np.array([[1, 1, 1], [4, 4, 4], [1, 4, 2], [2, 2, 2]])  # base, top, a map, a flat middle rung
    got = shapes(levels, base=1, top=4)
    assert got.tolist() == ["base", "top", "map", "map"]
    assert set(got) <= {"base", "map", "top"}  # every question falls into exactly one shape


def test_holm_multiplies_the_smallest_by_the_family_and_keeps_the_order():
    got = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert got["a"] == pytest.approx(0.03) and got["b"] == pytest.approx(0.08) and got["c"] == pytest.approx(0.5)
    assert got["a"] <= got["b"] <= got["c"]
    assert holm({"only": 0.2})["only"] == pytest.approx(0.2)
    assert holm({"a": 0.9, "b": 0.95})["a"] == 1.0  # a correction never goes past 1
