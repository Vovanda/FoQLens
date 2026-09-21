"""Carrying one question's map onto another, at the same cost.

A map that holds an answer has to be shown to be this question's map and not any map of that price. So every question
is given someone else's layout while the price stays where it was: the questions are paired by what their maps cost,
nearest first, and each of a pair answers on the other's map. What the pair loses against its own maps is what being
the question's own map is worth.

- pairing: greedy over the costs - the two nearest costs make a pair, then the two nearest of the rest.

Invariant: pairing returns every question once, and a question is never paired with itself.
Invariant: moving or dealing the rungs leaves every question's cost where it was - a rung lands only on a
group of the same weight.
Invariant: the questions of a pair stand next to each other in the order of the costs, so no pair is charged for a
price gap another pairing would have avoided.
"""

from __future__ import annotations

import numpy as np


def pairs_by_cost(costs: np.ndarray) -> list[tuple[int, int]]:
    """Questions paired by the cost of their maps, nearest first: [(i, j), ...], every index once.

    An odd count leaves the last question out - it has no partner of its own price, and giving it one from further
    away would charge the swap for a price gap.
    """
    order = np.argsort(np.asarray(costs, dtype=float), kind="stable")
    return [(int(order[k]), int(order[k + 1])) for k in range(0, len(order) - 1, 2)]


def shuffled_elsewhere(levels: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    """Every question's rungs dealt at random over the groups, at exactly the same cost: [questions, groups].

    The control of H3.3 - what a layout of this price is worth when nothing chose where its sharpness goes. As in
    moved_elsewhere a rung may only land on a group of the same weight, and every question is dealt its own order.
    """
    rng = np.random.default_rng(seed)
    out = np.array(levels, copy=True)
    for weight in np.unique(weights):
        where = np.flatnonzero(weights == weight)
        if len(where) > 1:
            for q in range(len(levels)):
                out[q, where] = levels[q, rng.permutation(where)]
    return out


def moved_elsewhere(levels: np.ndarray, weights: np.ndarray, shift: int = 1) -> np.ndarray:
    """Every question's layout carried elsewhere in the network at exactly the same cost: [questions, groups].

    The control of H3.4 - is it this figure, or this figure in this place. A rung may only move onto a group that
    holds as many weights, or the memory would move with it, so the groups are gathered into classes of equal weight
    and every class is rolled by `shift` within itself. A class of one group cannot move and keeps its rung; its
    share of the network is what this control cannot reach.
    """
    out = np.array(levels, copy=True)
    for weight in np.unique(weights):
        where = np.flatnonzero(weights == weight)
        if len(where) > 1:
            out[:, where] = levels[:, np.roll(where, shift)]
    return out


def swap_rows(rows: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """`rows` with the two members of every pair exchanged; a question outside the pairs keeps its own row."""
    out = np.array(rows, copy=True)
    for i, j in pairs:
        out[i], out[j] = rows[j], rows[i]
    return out
