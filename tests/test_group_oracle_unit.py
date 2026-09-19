"""The oracles by trying: the groups of blocks, the layouts of lifted groups, and the minimal mask's prefix."""

import numpy as np

from foqlens.group_oracle import (
    block_groups,
    group_order,
    joined_answer,
    lift_layouts,
    minimal_layouts,
    minimal_prefix,
)
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


def test_a_minimal_layout_lifts_the_questions_first_groups_of_its_order():
    groups = np.array([0, 0, 1, 2])
    lift = np.array([[0.1, 0.5, 0.3], [0.9, 0.0, 0.2]])
    rows = minimal_layouts(groups, group_order(lift), np.array([2, 0]), Level.D2, Level.D8)
    d2, d8 = int(Level.D2), int(Level.D8)
    assert rows.tolist() == [[d2, d2, d8, d8], [d2, d2, d2, d2]]


def test_the_minimal_mask_is_the_shortest_prefix_within_the_tolerance():
    nll = np.array([3.0, 2.0, 1.2, 1.05, 1.0])  # k groups lifted -> the answer's NLL; every group lifted gives 1.0
    assert minimal_prefix(nll, target=1.0, tolerance=0.1) == 3
    assert minimal_prefix(nll, target=1.0, tolerance=0.0) == 4
    assert minimal_prefix(np.array([2.0, 1.5]), target=1.0, tolerance=0.1) == 1  # none within: the whole sweep


def test_zeroing_switches_off_the_least_needed_groups_outside_the_minimal_mask():
    from foqlens.group_oracle import chain_layout, zero_order, zeroed_count, zeroed_layouts

    groups = np.array([0, 0, 1, 2, 3])
    order = group_order(np.array([0.9, 0.1, 0.5, 0.0]))  # the mask of 1 is group 0; outside, least needed: 3, 1, 2
    assert zero_order(order, 1).tolist() == [3, 1, 2]
    rows = zeroed_layouts(groups, order, 1, Level.D2, Level.D8)
    d2, d8, z = int(Level.D2), int(Level.D8), int(Level.ZERO)
    assert rows.tolist() == [[d8, d8, d2, d2, z], [d8, d8, z, d2, z], [d8, d8, z, z, z]]
    assert chain_layout(groups, order, 1, 2, Level.D2, Level.D8).tolist() == rows[1].tolist()
    assert chain_layout(groups, order, 1, 0, Level.D2, Level.D8).tolist() == [d8, d8, d2, d2, d2]
    assert zeroed_count(np.array([0.10, 0.11, 0.40]), target=0.1, tolerance=0.02) == 2
    assert zeroed_count(np.array([0.5]), target=0.1, tolerance=0.02) == 0


class _Layouts:
    """A controller that keeps the layout it was given, for an answer NLL read off the layout."""

    def set_layout(self, layout):
        self.layout = layout


class _Pacer:
    def batch(self):
        from contextlib import nullcontext

        return nullcontext()


def test_a_chain_stops_its_sweeps_at_the_first_prefix_within_and_the_first_step_out(monkeypatch):
    import foqlens.group_oracle as go

    groups = np.arange(4)  # a block a group
    need = np.array([0.0, 2.0, 0.0, 1.0])  # what reading group g coarse costs the answer; ZERO costs 10x
    ctl, calls = _Layouts(), []

    def nll(_model, _tok, prompts, _answers):
        import torch

        calls.append(len(prompts))
        lay = ctl.layout
        cost = ((lay == int(Level.D2)) * need + (lay == int(Level.ZERO)) * 10 * need).sum(axis=1)
        return torch.tensor(cost, dtype=torch.float64)

    monkeypatch.setattr(go, "answer_nll", nll)
    order = go.group_order(need)  # 1, 3, then 0 and 2 that cost nothing
    chain = go.build_chain(None, None, ctl, _Pacer(), "p", "a", groups, order, np.array([3.0, 0.0]), Level.D2,
                           Level.D8, 0.1, 0.1, size=1)
    assert chain.minimal == 2 and np.isnan(chain.prefix[2:]).all()  # stopped at the first prefix within
    assert chain.zeroed == 2  # groups 2 and 0 cost nothing off; the sweep ran out of groups, none left the tolerance
    assert len(calls) == 2 + 2


def test_grading_takes_each_chain_groups_coarsest_rung_that_holds_least_needed_first(monkeypatch):
    import foqlens.group_oracle as go

    groups = np.arange(3)
    # what a group costs below D8: group 0 nothing at D4; group 1 too much at D4, nothing at D6; group 2 too much at both
    cost = {(0, Level.D4): 0.0, (0, Level.D6): 0.0, (1, Level.D4): 1.0, (1, Level.D6): 0.0,
            (2, Level.D4): 1.0, (2, Level.D6): 1.0}
    ctl = _Layouts()

    def nll(_model, _tok, prompts, _answers):
        import torch

        rows = [sum(cost.get((g, Level(int(v))), 0.0) for g, v in enumerate(lay)) for lay in ctl.layout]
        return torch.tensor(rows, dtype=torch.float64)

    monkeypatch.setattr(go, "answer_nll", nll)
    layout = np.full(3, int(Level.D8), dtype=np.uint8)
    graded, levels = go.grade_chain(None, None, ctl, _Pacer(), "p", "a", groups, layout, np.array([2, 1, 0]),
                                    (Level.D4, Level.D6), target=0.0, tolerance=0.1, size=2)
    assert levels.tolist() == [int(Level.D8), int(Level.D6), int(Level.D4)]  # chain order: 2, 1, 0
    assert graded.tolist() == [int(Level.D4), int(Level.D6), int(Level.D8)]
