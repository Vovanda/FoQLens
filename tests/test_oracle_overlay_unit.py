"""The oracles' block scores in groups and the bootstrap."""

import numpy as np

from foqlens.oracle_overlay import block_group_ids, bootstrap, to_groups


def test_blocks_sum_into_their_groups_and_a_missing_score_counts_nothing():
    scores = np.array([[1.0, 2.0, np.nan, 4.0]])
    assert to_groups(scores, np.array([0, 0, 1, 1]), 2).tolist() == [[3.0, 4.0]]


def test_a_block_joins_its_layers_attention_or_the_rest_of_its_layer():
    ids = block_group_ids(np.array([0, 0, 1]), np.array(["self_attn.q_proj", "mlp.up_proj", "per_layer_projection"]),
                          ["0.attention", "0.mlp", "1.mlp"])
    assert ids.tolist() == [0, 1, 2]


def test_the_bootstrap_interval_holds_the_statistic_of_a_constant_sample():
    assert bootstrap(np.full(20, 3.0), np.mean, draws=50, seed=0, level=0.95) == (3.0, 3.0)
