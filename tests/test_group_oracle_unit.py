"""The oracles by trying: the groups of blocks, the layouts of lifted groups, and the batches of a question's variants."""

import numpy as np

from foqlens.group_oracle import block_groups, joined_answer, lift_layouts
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


def test_every_batch_of_a_questions_variants_has_one_shape_and_every_variant_is_read_once(monkeypatch):
    import foqlens.group_oracle as go

    ctl, shapes = _Layouts(), []

    def nll(_model, _tok, prompts, _answers):
        import torch

        shapes.append(len(prompts))
        return torch.tensor(ctl.layout[:, 0].astype(float))

    monkeypatch.setattr(go, "answer_nll", nll)
    layouts = np.arange(5, dtype=np.uint8)[:, None]
    got = go.variants_nll(None, None, ctl, _Pacer(), "p", "a", layouts, size=2)
    assert got.tolist() == [0, 1, 2, 3, 4] and shapes == [2, 2, 2]


class _Layouts:
    """A controller that keeps the layout it was given, for an answer NLL read off the layout."""

    def set_layout(self, layout):
        self.layout = layout


class _Pacer:
    def batch(self):
        from contextlib import nullcontext

        return nullcontext()
