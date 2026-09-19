"""The regulator (foqlens.regulator): layouts checked against the kernel's ladder, their bytes, their spread over layers."""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from foqlens.kquant import QK_K
from foqlens.precision import Controller, MixedPrecisionLinear
from foqlens.quant import MAX_DEPTH, Level
from foqlens.refinements import KQuantLadder
from foqlens.regulator import Regulator, ReadCost, block_layers, check, kernel_ladder

BLOCK_ROWS = 64
NAMES = ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj", "layers.1.mlp.up_proj")  # Q2_K, Q4_K, Q2_K bases
DEPTHS = (Level.D2, Level.D4, Level.D6, Level.D8)


def module(name: str, out: int = 2 * BLOCK_ROWS, inp: int = QK_K) -> MixedPrecisionLinear:
    torch.manual_seed(0)
    linear = nn.Linear(inp, out, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.02)
    made = MixedPrecisionLinear(linear, BLOCK_ROWS, partial(KQuantLadder().quantize, name))
    made.set_levels(Level.D8)  # the copy is built on the first read of a depth
    return made


def controller() -> Controller:
    return Controller({name: module(name) for name in NAMES})


@dataclass(frozen=True)
class Fixed:
    """A policy that answers every question with the same codes."""

    codes: np.ndarray
    name: str = "fixed"

    def levels(self, indices):
        return np.stack([self.codes] * len(indices))


def test_the_kernel_ladder_is_the_depths_every_module_holds():
    ctl = controller()
    assert kernel_ladder(ctl) == DEPTHS
    short = module(NAMES[0])
    short.readable = frozenset({Level.ZERO, Level.D2, Level.D4})  # a copy cut to D4
    assert kernel_ladder(Controller({NAMES[0]: short, NAMES[1]: module(NAMES[1])})) == (Level.D2, Level.D4)


def test_a_level_the_kernel_does_not_read_is_refused():
    check(np.array([[int(Level.ZERO), int(Level.D8)]], dtype=np.uint8), DEPTHS)
    for level in (Level.BF16, Level.NF4):
        with pytest.raises(ValueError, match=level.name):
            check(np.array([int(Level.D4), int(level)], dtype=np.uint8), DEPTHS)
    with pytest.raises(ValueError, match="D8"):
        check(np.array([int(Level.D8)], dtype=np.uint8), DEPTHS[:2])


def test_bytes_are_the_base_blocks_and_a_plane_per_refinement_and_nothing_at_zero():
    ctl = controller()
    cost = ReadCost(ctl)
    q = ctl.modules[NAMES[0]].refined  # Q2_K: 84 bytes a super-block, base depth 1
    down = ctl.modules[NAMES[1]].refined  # Q4_K: 144 bytes, base depth 2
    assert (q.blocks.shape[-1], down.blocks.shape[-1]) == (84, 144)
    plane = QK_K // 4
    assert cost.table[0].tolist() == [0] + [BLOCK_ROWS * (84 + e * plane) for e in range(MAX_DEPTH)]
    # a Q4_K base reads the same at D2 and D4: never shallower than its base
    assert cost.table[2].tolist() == [0, BLOCK_ROWS * 144, BLOCK_ROWS * 144, BLOCK_ROWS * (144 + plane),
                                      BLOCK_ROWS * (144 + 2 * plane)]
    zero = np.full(ctl.n_blocks, int(Level.ZERO), dtype=np.uint8)
    assert cost.read_bytes(zero).tolist() == [0]
    one = zero.copy()
    one[0] = int(Level.D6)
    assert cost.read_bytes(one).tolist() == [BLOCK_ROWS * (84 + 2 * plane)]


def test_a_batch_step_reads_every_block_at_its_deepest_depth_over_the_samples():
    ctl = controller()
    cost = ReadCost(ctl)
    rng = np.random.default_rng(0)
    codes = rng.choice([int(Level.ZERO), *(int(lv) for lv in DEPTHS)], size=(5, ctl.n_blocks)).astype(np.uint8)
    deepest = codes.max(axis=0)  # the codes of ZERO and the depths follow the ladder
    assert cost.step_bytes(codes) == cost.read_bytes(deepest)[0]
    assert cost.step_bytes(codes) >= cost.read_bytes(codes).max()


def test_the_regulator_sets_each_question_its_own_layout_and_refuses_what_the_kernel_cannot_read():
    ctl = controller()
    codes = np.full(ctl.n_blocks, int(Level.D2), dtype=np.uint8)
    codes[:2] = int(Level.D8)
    regulator = Regulator(Fixed(codes), ctl)
    applied = regulator.apply(np.arange(3))
    assert applied.shape == (3, ctl.n_blocks)
    assert ctl.modules[NAMES[0]].levels.tolist() == [[int(Level.D8)] * 2] * 3
    bf16 = codes.copy()
    bf16[3] = int(Level.BF16)
    with pytest.raises(ValueError, match="BF16"):
        Regulator(Fixed(bf16), ctl).apply(np.arange(2))


def test_q_and_k_blocks_left_at_zero_read_the_attention_level_and_nothing_else_moves():
    ctl = controller()
    codes = np.full(ctl.n_blocks, int(Level.ZERO), dtype=np.uint8)
    codes[1] = int(Level.D6)  # a q_proj block a zone lifted keeps its level
    regulator = Regulator(Fixed(codes), ctl)
    laid = regulator.layout(np.arange(2))
    # q_proj: blocks 0, 1; down_proj and up_proj: 2 ... 5 stay at ZERO
    assert laid[:, 0].tolist() == [int(Level.D2)] * 2 and laid[:, 1].tolist() == [int(Level.D6)] * 2
    assert np.all(laid[:, 2:] == int(Level.ZERO))
    raised = Regulator(Fixed(codes), ctl, attention_level=Level.D4).layout(np.arange(1))
    assert raised[0, 0] == int(Level.D4)


def test_the_working_layers_read_the_default_level_and_the_filter_acts_after_them():
    from foqlens.layouts import WorkingLayers

    ctl = controller()
    codes = np.full(ctl.n_blocks, int(Level.ZERO), dtype=np.uint8)
    codes[1], codes[5] = int(Level.D8), int(Level.D6)  # a zone in layer 0 and one in layer 1
    working = block_layers(ctl) < 1  # layer 0 is read before the filter acts
    laid = Regulator(WorkingLayers(Fixed(codes), working, Level.D2), ctl).layout(np.arange(1))[0]
    assert np.all(laid[working] == int(Level.D2))  # the zone in layer 0 does not lift it, ZERO does not empty it
    assert laid[5] == int(Level.D6) and laid[4] == int(Level.ZERO)


def test_by_layer_counts_every_block_once_in_its_layer_and_kind():
    ctl = controller()
    codes = np.full(ctl.n_blocks, int(Level.D2), dtype=np.uint8)
    codes[0] = int(Level.D8)  # one block of layer 0's q_proj
    rows = Regulator(Fixed(codes), ctl).by_layer(codes)
    assert [(r["layer"], r["kind"], r["blocks"]) for r in rows] == [
        (0, "self_attn.q_proj", 2), (0, "mlp.down_proj", 2), (1, "mlp.up_proj", 2)]
    assert sum(r["blocks"] for r in rows) == ctl.n_blocks
    assert rows[0]["above_min"] == 0.5 and rows[1]["above_min"] == 0.0


def test_every_block_is_named_by_its_layer_in_the_controllers_order():
    assert block_layers(controller()).tolist() == [0, 0, 0, 0, 1, 1]


def test_a_regulated_reading_lays_every_row_out_by_its_question_and_keeps_the_codes():
    from types import SimpleNamespace

    ctl = controller()
    base = np.full(ctl.n_blocks, int(Level.D2), dtype=np.uint8)

    @dataclass(frozen=True)
    class ByQuestion:
        """Question i lifts block i to D8."""

        name: str = "by-question"

        def levels(self, indices):
            out = np.stack([base] * len(indices))
            out[np.arange(len(indices)), np.asarray(indices)] = int(Level.D8)
            return out

    reading = Regulator(ByQuestion(), ctl).reading({("c", "a"): 2, ("c", "b"): 4}, "zones")
    rows = [SimpleNamespace(id="b"), SimpleNamespace(id="a")]
    reading.apply(ctl, "c", rows)
    assert reading.label == "zones" and len(reading.laid) == 1
    assert reading.laid[0][0, 4] == int(Level.D8) and reading.laid[0][1, 2] == int(Level.D8)
    assert ctl.modules[NAMES[1]].levels.tolist() == [[int(Level.D2), int(Level.D2)], [int(Level.D8), int(Level.D2)]]
    with pytest.raises(ValueError, match="bench"):
        reading.apply(controller(), "c", rows)


def test_the_regulator_builds_on_a_module_whose_copy_is_not_read_yet():
    # a bench loaded from its file holds the copy unbuilt until a depth is first read (the smoke of 19.09 found it)
    torch.manual_seed(0)
    linear = nn.Linear(QK_K, 2 * BLOCK_ROWS, bias=False, dtype=torch.bfloat16)
    fresh = MixedPrecisionLinear(linear, BLOCK_ROWS, partial(KQuantLadder().quantize, NAMES[0]))
    assert fresh.levels.tolist() == [int(Level.BF16)] * 2  # nothing read below bf16 yet
    ctl = Controller({NAMES[0]: fresh})
    assert ReadCost(ctl).table.shape == (2, MAX_DEPTH + 1)
