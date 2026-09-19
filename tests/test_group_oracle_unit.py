"""The oracles by trying: the groups of blocks, the layouts of lifted groups, and the minimal mask's prefix."""

import numpy as np

from foqlens.group_oracle import block_groups, joined_answer, lift_layouts, minimal_layouts, minimal_prefix
from foqlens.precision import Controller
from foqlens.quant import Level

from test_regulator_unit import module


def test_a_layers_attention_is_one_group_and_the_rest_of_the_layer_another():
    names = ("layers.0.self_attn.q_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj", "layers.1.self_attn.o_proj")
    groups, labels = block_groups(Controller({n: module(n) for n in names}))
    assert labels == ["0.attention", "0.mlp", "1.attention"]
    assert groups.tolist() == [0, 0, 1, 1, 1, 1, 2, 2]


def test_no_group_lifted_is_the_base_and_every_group_is_the_top():
    groups = np.array([0, 0, 1, 2])
    rows = lift_layouts(groups, [np.array([], dtype=int), np.arange(3), np.array([1])], Level.D2, Level.D8)
    assert rows[0].tolist() == [int(Level.D2)] * 4 and rows[1].tolist() == [int(Level.D8)] * 4
    assert rows[2].tolist() == [int(Level.D2), int(Level.D2), int(Level.D8), int(Level.D2)]


def test_the_answer_follows_a_turn_ending_on_a_newline_without_a_space_and_a_label_with_one():
    assert joined_answer("<|turn>model\n", " John Ford") == "John Ford"
    assert joined_answer("Answer:", "John Ford") == " John Ford"


def test_a_minimal_layout_lifts_the_questions_first_groups_by_lift():
    groups = np.array([0, 0, 1, 2])
    lift = np.array([[0.1, 0.5, 0.3], [0.9, 0.0, 0.2]])
    rows = minimal_layouts(groups, lift, np.array([2, 0]), Level.D2, Level.D8)
    d2, d8 = int(Level.D2), int(Level.D8)
    assert rows.tolist() == [[d2, d2, d8, d8], [d2, d2, d2, d2]]


def test_the_minimal_mask_is_the_shortest_prefix_within_the_tolerance():
    nll = np.array([3.0, 2.0, 1.2, 1.05, 1.0])  # k groups lifted -> the answer's NLL; every group lifted gives 1.0
    assert minimal_prefix(nll, target=1.0, tolerance=0.1) == 3
    assert minimal_prefix(nll, target=1.0, tolerance=0.0) == 4
    assert minimal_prefix(np.array([2.0, 1.5]), target=1.0, tolerance=0.1) == 1  # none within: the whole sweep
