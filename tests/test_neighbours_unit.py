"""Neighbour tables and dilation on a small made-up decoder: no model."""

import numpy as np
import pytest

from foqlens import budget as bg
from foqlens import neighbours as nb
from foqlens.quant import Level

# block index ranges: q 0-1, o L0 2-6, gate 7-10, up 11-14, down L0 15-19, o L1 20-24, down L1 25-29
BLOCKS = {
    "layers.0.self_attn.q_proj": 2,
    "layers.0.self_attn.o_proj": 5,
    "layers.0.mlp.gate_proj": 4,
    "layers.0.mlp.up_proj": 4,
    "layers.0.mlp.down_proj": 5,
    "layers.1.self_attn.o_proj": 5,
    "layers.1.mlp.down_proj": 5,
}


def of(table: np.ndarray, block: int) -> set[int]:
    return set(table[block][table[block] != nb.PAD].tolist())


def test_structural_links_the_same_neurons_and_the_same_stream_coordinates():
    t = nb.structural(BLOCKS)
    assert of(t, 7) == {11}  # gate block 0 <-> up block 0: the same MLP neurons
    assert of(t, 2) == {15, 20, 25}  # o_proj L0 block 0: down_proj L0, o_proj L1, down_proj L1
    assert of(t, 21) == {3, 16, 26}  # o_proj L1 block 1: o_proj L0, down_proj L0, down_proj L1
    assert of(t, 0) == set()  # q_proj rows share no stream coordinate


def test_structural_is_symmetric_and_never_links_a_block_to_itself():
    t = nb.structural(BLOCKS)
    for block in range(len(t)):
        for other in of(t, block):
            assert other != block and block in of(t, other)


def test_index_control_has_as_many_neighbours_from_the_same_matrix():
    t = nb.structural(BLOCKS)
    idx = nb.index_matched(BLOCKS, t)
    assert ((t != nb.PAD).sum(axis=1) == (idx != nb.PAD).sum(axis=1)).all()
    assert of(idx, 7) == {8}  # gate block 0: the next gate block
    assert of(idx, 2) == {3, 4, 5}  # o_proj L0 block 0: the three nearest o_proj L0 blocks
    assert of(idx, 4) == {3, 5, 6}


def test_dilate_follows_each_block_with_its_neighbours_once():
    table = np.array([[1, -1], [0, 2], [1, -1], [-1, -1]])
    assert bg.dilate(np.array([3, 2, 0, 1]), table).tolist() == [3, 2, 1, 0]


def test_dilated_fill_spends_the_same_budget_on_fewer_seeds():
    weights = np.full(10, 10)
    backbone = np.arange(10, 0, -1, dtype=float)  # blocks 0, 1 first
    fill = np.arange(10, dtype=float)  # blocks 9, 8, ... first
    table = np.full((10, 1), -1)
    table[9, 0], table[5, 0] = 5, 9
    plain = bg.layered(backbone, fill, weights, 0.4, 0.5)
    wide = bg.layered(backbone, fill, weights, 0.4, 0.5, table)
    assert np.flatnonzero(plain).tolist() == [0, 1, 8, 9]
    assert np.flatnonzero(wide).tolist() == [0, 1, 5, 9]  # seed 9 brought its neighbour 5 instead of 8
    assert weights[wide].sum() == weights[plain].sum()


def test_backbone_fill_names_its_dilation_and_needs_a_table_for_it():
    from foqlens.layouts import BackboneFill, OwnTopic, TopicMeans

    weights = np.full(4, 10)
    backbone = np.array([9.0, 0.0, 0.0, 0.0])
    scores = np.array([[0, 0, 5.0, 0], [0, 0, 5.0, 0], [0, 5.0, 0, 0], [0, 5.0, 0, 0]])
    means = TopicMeans(scores, ("a", "a", "b", "b"))
    table = np.array([[-1], [-1], [3], [2]])
    wide = BackboneFill(OwnTopic(means), "pooled", 0.75, 1 / 3, backbone, weights, Level.ZERO, "struct", table)
    plain = BackboneFill(OwnTopic(means), "pooled", 0.75, 1 / 3, backbone, weights, Level.ZERO)
    sharp = lambda p: np.flatnonzero(p.levels(np.array([0]))[0] == Level.BF16).tolist()  # noqa: E731
    assert sharp(wide) == [0, 2, 3] and sharp(plain) == [0, 1, 2]
    assert wide.name == "bb0.33_own_pooled_struct_0.750" and plain.name == "bb0.33_own_pooled_0.750"
    with pytest.raises(ValueError):
        BackboneFill(OwnTopic(means), "pooled", 0.75, 1 / 3, backbone, weights, Level.ZERO, "struct")
