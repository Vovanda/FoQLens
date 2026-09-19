"""The overlay of the oracles: groups, ranks, the static share, the topics' Jaccards and the bootstrap."""

import numpy as np
import pytest

from foqlens.oracle_overlay import bootstrap, contrast, group_ranks, jaccard_within_between, static_share, to_groups


def test_blocks_sum_into_their_groups_and_a_missing_score_counts_nothing():
    scores = np.array([[1.0, 2.0, np.nan, 4.0]])
    assert to_groups(scores, np.array([0, 0, 1, 1]), 2).tolist() == [[3.0, 4.0]]


def test_the_static_share_is_one_when_every_question_ranks_alike_and_small_when_at_random():
    alike = group_ranks(np.tile(np.arange(10.0), (50, 1)))
    assert static_share(alike) == pytest.approx(1.0)
    random = group_ranks(np.random.default_rng(0).random((500, 10)))
    assert static_share(random) < 0.05


def test_a_topic_with_its_own_zone_shows_a_higher_jaccard_within_than_between():
    masks = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    within, between = jaccard_within_between(masks, np.array(["a", "a", "b", "b"]))
    assert within == 1.0 and between == 0.0


def test_the_contrast_has_mean_one_and_ignores_a_scale_per_group():
    scores = np.random.default_rng(0).random((30, 5)) + 0.1
    scores[:, 4] = 0.0
    got = contrast(scores)
    assert np.allclose(got[:, :4].mean(axis=0), 1.0) and not got[:, 4].any()
    assert np.allclose(contrast(scores * np.array([1.0, 10.0, 100.0, 0.5, 3.0])), got)


def test_the_bootstrap_interval_holds_the_statistic_of_a_constant_sample():
    assert bootstrap(np.full(20, 3.0), np.mean, draws=50, seed=0, level=0.95) == (3.0, 3.0)


def test_chains_antinodes_and_bands_read_the_sets_as_they_are():
    from foqlens.oracle_overlay import antinodes, band_spread, chain_sets, frequency, pair_jaccard

    orders = np.array([[2, 0, 1, 3], [2, 1, 0, 3], [0, 1, 2, 3]])
    chain, off = chain_sets(orders, np.array([1, 2, -1]), np.array([1, 0, 2]), 4)
    assert chain.tolist() == [[False, False, True, False], [False, True, True, False], [False] * 4]
    assert off.tolist() == [[False, False, False, True], [False] * 4, [False] * 4]
    assert pair_jaccard(chain, chain)[:2].tolist() == [1.0, 1.0] and np.isnan(pair_jaccard(chain, chain)[2])
    assert frequency(chain).tolist() == [0.0, 1 / 3, 2 / 3, 0.0]
    assert antinodes(frequency(chain), 0.5).tolist() == [2]
    bands = band_spread(chain, np.array([0, 0, 1, 1]), [(0, 1), (1, 2)], np.array(["t", "t", "s"]))
    assert bands[0]["mean"] == 1 / 6 and bands[1]["mean"] == 1 / 3


def test_lenses_are_runs_of_groups_above_the_base_connected_along_the_depth():
    from foqlens.oracle_overlay import lenses

    layers = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # attention and MLP of five layers
    levels = np.array([2, 8, 6, 2, 2, 2, 2, 4, 8, 2])  # above D2 (code 2): groups 1, 2 (layers 0-1) and 7, 8 (3-4)
    found = lenses(levels, layers, base=2)
    assert [f.tolist() for f in found] == [[1, 2], [7, 8]]
    assert lenses(np.full(10, 2), layers, base=2) == []
