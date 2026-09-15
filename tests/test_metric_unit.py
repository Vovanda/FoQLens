"""Distances between blocks on made-up masks and graphs: no model, CPU."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from foqlens.metric import (
    PAD,
    CoactivationMetric,
    MediumSurface,
    ResistiveMetric,
    geodesic,
    harmonic_conductance,
    neighbour_table,
    sweep_width,
)


def random_masks(questions: int = 40, blocks: int = 30, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((questions, blocks))


def full(metric) -> np.ndarray:
    return metric.distances(torch.arange(metric.n_blocks)).numpy()


def test_the_coactivation_distance_is_sqrt_two_one_minus_the_correlation():
    masks = random_masks()
    d = full(CoactivationMetric(masks))
    expected = np.sqrt(np.clip(2 * (1 - np.corrcoef(masks.T)), 0, None))
    np.testing.assert_allclose(d, expected, atol=1e-5)


def test_the_coactivation_distance_is_a_metric():
    d = full(CoactivationMetric(random_masks(seed=1)))
    assert np.allclose(d, d.T, atol=1e-6) and np.all(np.diag(d) == 0)
    # [a, b, c]: d(a, c) <= d(a, b) + d(b, c) for every a, b, c
    assert (d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-5).all()


def test_scaling_or_shifting_a_block_does_not_move_it():
    masks = random_masks(seed=2)
    moved = masks.copy()
    moved[:, 5] = 100 * moved[:, 5] + 7
    np.testing.assert_allclose(full(CoactivationMetric(masks)), full(CoactivationMetric(moved)), atol=1e-5)


def test_a_constant_block_is_one_from_every_varying_block_and_zero_from_another_constant_one():
    masks = random_masks(seed=3)
    masks[:, 0] = 4.0
    masks[:, 1] = -2.0
    d = full(CoactivationMetric(masks))
    assert d[0, 1] == pytest.approx(0) and np.allclose(d[0, 2:], 1, atol=1e-6)


def test_the_diameter_is_the_largest_distance():
    metric = CoactivationMetric(random_masks(seed=4))
    assert metric.diameter() == pytest.approx(full(metric).max(), rel=1e-6)


def test_a_neighbour_table_is_symmetric_without_self_and_carries_the_distances():
    metric = CoactivationMetric(random_masks(seed=5))
    table, lengths = neighbour_table(metric, k=4)
    d = full(metric)
    pairs = {(a, int(b)) for a in range(len(table)) for b in table[a] if b != PAD}
    assert all((b, a) in pairs for a, b in pairs) and all(a != b for a, b in pairs)
    for a in range(len(table)):
        real = table[a] != PAD
        np.testing.assert_allclose(lengths[a][real].numpy(), d[a, table[a][real].numpy()], atol=1e-5)
        assert real.sum() >= 4  # its own four nearest at least


def as_graph(table: torch.Tensor, edges: torch.Tensor) -> csr_matrix:
    heads = np.repeat(np.arange(len(table)), table.shape[1])
    tails, w = table.numpy().ravel(), edges.numpy().ravel()
    keep = tails != PAD
    return csr_matrix((w[keep], (heads[keep], tails[keep])), shape=(len(table),) * 2)


def test_with_activity_one_everywhere_the_medium_is_the_shortest_path_of_its_table():
    metric = CoactivationMetric(random_masks(blocks=50, seed=6))
    table, lengths = neighbour_table(metric, k=3)
    medium = ResistiveMetric(table, lengths, torch.ones(50))
    sources = torch.tensor([0, 7, 31])
    expected = dijkstra(as_graph(table, lengths), indices=sources.numpy())
    np.testing.assert_allclose(medium.distances(sources).numpy(), expected, rtol=1e-5)


def test_the_geodesic_is_the_medium_at_rest():
    metric = CoactivationMetric(random_masks(blocks=40, seed=8))
    table, lengths = neighbour_table(metric, k=3)
    sources = torch.tensor([1, 5])
    torch.testing.assert_close(geodesic(table, lengths).distances(sources),
                               ResistiveMetric(table, lengths, torch.ones(40)).distances(sources))


def line(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocks 0 - 1 - ... - n-1, one unit apart."""
    table = torch.full((n, 2), PAD)
    lengths = torch.full((n, 2), torch.inf)
    for i in range(n):
        for slot, j in enumerate((i - 1, i + 1)):
            if 0 <= j < n:
                table[i, slot], lengths[i, slot] = j, 1.0
    return table, lengths


def test_the_sweep_width_of_a_line_is_its_length_from_any_start():
    along = geodesic(*line(12))
    assert [sweep_width(along, start=s) for s in (0, 5, 11)] == [11.0, 11.0, 11.0]


def test_an_edge_conducts_between_the_smaller_activity_of_its_ends_and_twice_that():
    rng = np.random.default_rng(7)
    activity = torch.tensor(rng.lognormal(sigma=2.0, size=40), dtype=torch.float32)
    table = torch.tensor(rng.integers(0, 40, size=(40, 5)))
    c = harmonic_conductance(activity, table)
    low = torch.minimum(activity[:, None].expand_as(table), activity[table])
    assert torch.all(c >= low * (1 - 1e-5)) and torch.all(c <= 2 * low * (1 + 1e-5))


def test_a_quiet_block_holds_the_wave_back():
    n = 10  # a line of blocks; block 5 goes quiet
    table, lengths = line(n)
    loud = ResistiveMetric(table, lengths, torch.ones(n)).distances(torch.tensor([0]))[0]
    activity = torch.ones(n)
    activity[5] = 0.01
    quiet = ResistiveMetric(table, lengths, activity).distances(torch.tensor([0]))[0]
    assert torch.allclose(loud, torch.arange(n, dtype=torch.float32))
    assert torch.allclose(quiet[:5], loud[:5]) and torch.all(quiet[5:] > loud[5:] + 40)


def test_the_medium_is_each_questions_own_and_at_rest_it_is_the_geodesic():
    table, lengths = line(10)
    activity = np.ones((2, 10))
    activity[1, 5] = 0.01  # question 1 is quiet at block 5, question 0 is not
    surface = MediumSurface(table, lengths, activity, background=np.ones(10))
    rest = geodesic(table, lengths).distances(torch.tensor([0]))
    torch.testing.assert_close(surface.metric(0).distances(torch.tensor([0])), rest)
    assert torch.all(surface.metric(1).distances(torch.tensor([0]))[0, 5:] > rest[0, 5:] + 40)
    assert np.array_equal(surface.table, table.numpy())


def test_a_block_loud_on_every_question_conducts_no_better_than_its_background():
    table, lengths = line(6)
    activity = np.full((1, 6), 50.0)  # every block as loud as always
    surface = MediumSurface(table, lengths, activity, background=np.full(6, 50.0))
    torch.testing.assert_close(surface.metric(0).distances(torch.tensor([0])), geodesic(table, lengths).distances(torch.tensor([0])))
