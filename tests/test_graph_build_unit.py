"""How the block graph is built, and how it measures up for hubness (#4): no model, CPU."""

import numpy as np
import torch

from foqlens.metric import PAD, CoactivationMetric, graph_report, mutual_nicdm_table, nearest, neighbour_table


def hub_masks(questions: int = 60, blocks: int = 200, seed: int = 0) -> np.ndarray:
    """Masks sharing one factor, and a last block that is the factor alone - the mean profile, where hubs sit.

    Every block is 0.6 of the factor and 0.8 of its own noise: it correlates 0.6 with the hub and 0.36
    with any other block, so the hub is everyone's nearest.
    """
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((questions, 1))
    masks = 0.6 * factor + 0.8 * rng.standard_normal((questions, blocks))
    masks[:, -1] = factor[:, 0]
    return masks


def pairs(table: torch.Tensor) -> set[tuple[int, int]]:
    return {(a, int(b)) for a in range(len(table)) for b in table[a] if b != PAD}


def test_the_mutual_nicdm_graph_is_symmetric_and_never_links_a_block_to_itself():
    table, lengths = mutual_nicdm_table(CoactivationMetric(hub_masks()), k=8)
    edges = pairs(table)
    assert all((b, a) in edges for a, b in edges) and all(a != b for a, b in edges)
    assert torch.all(lengths[table != PAD] >= 0) and torch.all(torch.isinf(lengths[table == PAD]))


def test_it_keeps_the_union_graph_connected_where_the_union_is():
    metric = CoactivationMetric(hub_masks(seed=1))
    union = graph_report(neighbour_table(metric, k=8)[0])
    mutual = graph_report(mutual_nicdm_table(metric, k=8)[0])
    assert mutual["components"] == union["components"] and mutual["isolated_share"] == 0


def test_the_block_at_the_mean_profile_is_a_hub_of_the_union_and_not_of_the_mutual_graph():
    metric = CoactivationMetric(hub_masks(seed=2))
    union, _ = neighbour_table(metric, k=8)
    mutual, _ = mutual_nicdm_table(metric, k=8)
    hub = metric.n_blocks - 1
    assert int((union[hub] != PAD).sum()) > metric.n_blocks // 2  # among the 8 nearest of most blocks
    assert int((mutual[hub] != PAD).sum()) <= 8 + 2  # its own mutual pairs and at most a join or two


def test_the_report_counts_degrees_components_and_the_k_occurrence():
    table = torch.tensor([[1, PAD], [0, 2], [1, PAD], [PAD, PAD]])  # 0 - 1 - 2, and 3 alone
    report = graph_report(table, indices=np.array([[1], [0], [1], [1]]))
    assert report["degree_max"] == 2 and report["components"] == 2 and report["isolated_share"] == 0.25
    assert report["never_neighbour_share"] == 0.5  # blocks 2 and 3 are in nobody's list
    assert report["symmetric_share"] == 0.5  # (0, 1) and (1, 0) are mutual, (2, 1) and (3, 1) are not


def test_nearest_rescaled_by_ones_is_nearest():
    metric = CoactivationMetric(hub_masks(seed=3))
    plain, rescaled = nearest(metric, 5), nearest(metric, 5, scale=torch.ones(metric.n_blocks))
    assert torch.equal(plain[0], rescaled[0]) and torch.allclose(plain[1], rescaled[1])
