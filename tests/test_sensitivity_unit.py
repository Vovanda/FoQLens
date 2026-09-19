"""Does the address stand for the sensitivity: top overlap and rank correlation per question."""

import numpy as np
import pytest

from foqlens.sensitivity import rank_correlation, top_overlap


def test_a_score_against_itself_overlaps_fully_and_correlates_to_one():
    s = np.random.default_rng(0).random((3, 100))
    assert np.allclose(top_overlap(s, s, 0.05), 1.0) and np.allclose(rank_correlation(s, s), 1.0)


def test_the_overlap_counts_the_estimates_top_found_in_the_oracles_top():
    estimate = np.array([[10.0, 9.0, 1.0, 0.0]])
    oracle = np.array([[10.0, 0.0, 9.0, 1.0]])
    assert top_overlap(estimate, oracle, 0.5).tolist() == [0.5]  # top 2: {0, 1} against {0, 2}


def test_the_tail_correlation_leaves_the_estimates_top_out():
    estimate = np.array([[100.0, 4.0, 3.0, 2.0, 1.0]])
    oracle = np.array([[-5.0, 4.0, 3.0, 2.0, 1.0]])  # the top disagrees, the tail agrees
    assert rank_correlation(estimate, oracle, outside_share=0.2)[0] == pytest.approx(1.0)
    assert rank_correlation(estimate, oracle)[0] < 1.0
