"""Carrying a map onto another question at the same cost."""

import numpy as np

from foqlens import carry


def test_the_nearest_costs_make_a_pair():
    assert carry.pairs_by_cost(np.array([10.0, 1.0, 11.0, 2.0])) == [(1, 3), (0, 2)]


def test_an_odd_question_is_left_out_rather_than_paired_far_away():
    pairs = carry.pairs_by_cost(np.array([1.0, 2.0, 50.0]))
    assert pairs == [(0, 1)]


def test_every_question_appears_once():
    costs = np.array([5.0, 3.0, 9.0, 1.0, 7.0, 2.0])
    seen = [i for pair in carry.pairs_by_cost(costs) for i in pair]
    assert sorted(seen) == list(range(6))


def test_a_pair_exchanges_its_rows_and_the_rest_keep_theirs():
    rows = np.array([[1, 1], [2, 2], [3, 3]])
    got = carry.swap_rows(rows, [(0, 1)])
    assert got.tolist() == [[2, 2], [1, 1], [3, 3]]


def test_swapping_leaves_the_original_alone():
    rows = np.array([[1, 1], [2, 2]])
    carry.swap_rows(rows, [(0, 1)])
    assert rows.tolist() == [[1, 1], [2, 2]]


def test_a_figure_moves_only_onto_groups_of_the_same_weight():
    levels = np.array([[3, 0, 2, 1]])
    weights = np.array([10.0, 10.0, 7.0, 7.0])
    got = carry.moved_elsewhere(levels, weights, shift=1)
    # within each class of equal weight the rungs roll: [3,0] -> [0,3] and [2,1] -> [1,2]
    assert got.tolist() == [[0, 3, 1, 2]]


def test_a_class_of_one_group_keeps_its_rung():
    levels = np.array([[3, 1, 1]])
    got = carry.moved_elsewhere(levels, np.array([5.0, 9.0, 9.0]), shift=1)
    assert got[0, 0] == 3


def test_moving_a_figure_leaves_the_cost_where_it_was():
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 4, size=(8, 12))
    weights = np.repeat([10.0, 7.0, 3.0], 4).astype(float)
    got = carry.moved_elsewhere(levels, weights, shift=2)
    assert np.allclose((levels * weights).sum(axis=1), (got * weights).sum(axis=1))


def test_dealing_the_rungs_at_random_leaves_the_cost_where_it_was():
    rng = np.random.default_rng(1)
    levels = rng.integers(0, 4, size=(6, 9))
    weights = np.repeat([10.0, 7.0, 3.0], 3).astype(float)
    got = carry.shuffled_elsewhere(levels, weights, seed=5)
    assert np.allclose((levels * weights).sum(axis=1), (got * weights).sum(axis=1))
    assert not np.array_equal(got, levels)
