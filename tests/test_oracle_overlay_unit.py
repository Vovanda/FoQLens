"""The overlay of the oracles: groups, ranks, the static share, the topics' Jaccards and the bootstrap."""

import numpy as np
import pytest

from foqlens.oracle_overlay import bootstrap, group_ranks, jaccard_within_between, static_share, to_groups


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


def test_the_bootstrap_interval_holds_the_statistic_of_a_constant_sample():
    assert bootstrap(np.full(20, 3.0), np.mean, draws=50, seed=0, level=0.95) == (3.0, 3.0)
