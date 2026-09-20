"""The checks of a new oracle against what is known of the same questions: matched by key, read only where both hold."""

import numpy as np

from foqlens.oracle_checks import end_gaps, matched, nll_agreement, rank_agreement, rung_ratio
from foqlens.oracle_overlay import block_group_ids


def test_questions_match_by_corpus_and_id_in_the_first_files_order():
    a = [("t", "1"), ("n", "1"), ("t", "2"), ("s", "9")]
    b = [("t", "2"), ("t", "1"), ("n", "1")]
    left, right = matched(a, b)
    assert left.tolist() == [0, 1, 2] and right.tolist() == [1, 2, 0]
    assert [len(x) for x in matched(a, [("x", "0")])] == [0, 0]


def test_nll_agreement_reads_only_questions_both_sides_measured():
    found = nll_agreement(np.array([1.0, np.nan, 2.0]), np.array([1.5, 3.0, 2.0]))
    assert found["questions"] == 2 and found["max_abs"] == 0.5 and found["median_abs"] == 0.25


def test_rank_agreement_is_one_for_the_same_order_and_minus_one_reversed():
    scores = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    found = rank_agreement(scores, np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]))
    assert found["rho_median"] == 0.0 and found["rho_positive_share"] == 0.5


def test_the_upper_rung_ratio_skips_blocks_the_lower_rung_never_moves():
    lower = np.array([[4.0, 0.0, 16.0]])
    upper = np.array([[1.0, 0.0, 1.0]])
    found = rung_ratio(upper, lower)
    assert found["ratio_median"] == (0.25 + 1 / 16) / 2 and found["upper_not_above_share"] == 1.0 and found["finite"]


def test_end_gaps_count_where_the_low_base_is_within_the_tolerance_or_better():
    ends = np.array([[0.5, 0.49], [0.3, 0.4], [2.0, 0.1], [1.0, 1.0]])
    found = end_gaps(ends, np.array(["t", "t", "t", "s"]), tolerance=0.02)
    assert found["t"]["questions"] == 3 and found["t"]["within_tolerance"] == 2 / 3
    assert found["t"]["low_better"] == 1 / 3 and found["s"]["within_tolerance"] == 1.0


def test_blocks_go_to_their_layers_attention_or_the_rest_of_the_layer():
    names = ["0.attention", "0.mlp", "1.attention", "1.mlp"]
    got = block_group_ids(np.array([0, 0, 1, 1]), np.array(["self_attn.q_proj", "mlp.up_proj", "mlp.down_proj",
                                                             "self_attn.o_proj"]), names)
    assert got.tolist() == [0, 1, 3, 2]
