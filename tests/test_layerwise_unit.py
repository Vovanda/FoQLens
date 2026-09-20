"""The layer-wise regulator: the blocks' static norms, the levels a price gives, the rim, and what a pass counts."""

import numpy as np
import pytest
import torch

from foqlens.layerwise import Activity, LayerwiseRegulator, block_weights, row_norms
from foqlens.layouts import knapsack_levels
from foqlens.precision import Controller
from foqlens.quant import Level

from test_regulator_unit import module

NAME = "layers.0.mlp.gate_proj"
D2, D4, D8 = (int(x) for x in (Level.D2, Level.D4, Level.D8))


def a_regulator(price: float, rim: float = 0.0) -> LayerwiseRegulator:
    ctl = Controller({NAME: module(NAME)})
    return LayerwiseRegulator(ctl, Activity(), price=price, rim=rim)


def a_state(regulator: LayerwiseRegulator, batch: int = 2) -> torch.Tensor:
    return torch.ones(batch, 3, regulator.ctl.modules[NAME].weight.shape[1], dtype=torch.bfloat16)


def test_a_blocks_norms_and_weights_are_read_from_the_module():
    made = module("layers.0.mlp.gate_proj")
    norms, weights = row_norms(made), block_weights(made)
    assert norms.shape == (made.n_blocks,) and (norms > 0).all()
    assert weights.sum().item() == made.weight.numel()


def test_the_price_runs_from_every_block_at_the_ceiling_to_every_block_at_the_base():
    x = torch.ones(1, 3, module("layers.0.mlp.gate_proj").weight.shape[1], dtype=torch.bfloat16)
    high = a_regulator(price=0.0).levels_for("layers.0.mlp.gate_proj", x)
    low = a_regulator(price=1e30).levels_for("layers.0.mlp.gate_proj", x)
    assert (high == D8).all() and (low == D2).all()


def test_every_question_of_a_batch_gets_a_layout_of_its_own():
    inp = module("layers.0.mlp.gate_proj").weight.shape[1]
    x = torch.stack([torch.full((3, inp), 1.0), torch.full((3, inp), 0.01)]).to(torch.bfloat16)
    codes = a_regulator(price=1e-3).levels_for("layers.0.mlp.gate_proj", x)
    assert codes.shape[0] == 2 and (codes[0] > codes[1]).any()  # the loud sample reads deeper than the quiet one


def test_the_rim_holds_a_band_one_rung_over_the_base():
    x = torch.ones(1, 3, module("layers.0.mlp.gate_proj").weight.shape[1], dtype=torch.bfloat16)
    plain = a_regulator(price=1e30).levels_for("layers.0.mlp.gate_proj", x)
    with_rim = a_regulator(price=1e30, rim=0.5).levels_for("layers.0.mlp.gate_proj", x)
    assert (plain == D2).all()
    blocks = with_rim.shape[1]
    assert (with_rim == D4).sum() == round(0.5 * blocks) and (with_rim == D2).sum() == blocks - (with_rim == D4).sum()


def test_the_levels_on_the_card_are_the_ones_the_knapsack_gives():
    regulator = a_regulator(price=1e-3)
    read = regulator.ctl.modules[NAME]
    x = torch.stack([torch.full((3, read.weight.shape[1]), 1.0),
                     torch.full((3, read.weight.shape[1]), 0.05)]).to(torch.bfloat16)
    codes = regulator.levels_for(NAME, x)
    reduced = (Activity().scores(read, x, row_norms(read)) / block_weights(read)).numpy()
    wanted = knapsack_levels(reduced, 1e-3, Level.D2, (Level.D2, Level.D4, Level.D6, Level.D8))
    assert np.array_equal(codes.numpy(), wanted)


def test_a_score_that_lands_on_a_threshold_rises_where_the_host_rule_raises_it():
    """The edge of a rung: a price equal to a block's own score per weight. The thresholds are float64 on the card as
    they are on the host, so the block lands on the same rung and not one below."""
    read = module(NAME)
    x = torch.ones(1, 3, read.weight.shape[1], dtype=torch.bfloat16)
    reduced = (Activity().scores(read, x, row_norms(read)) / block_weights(read)).numpy()
    for price in (float(reduced.max()), float(reduced.max()) / 16.0):
        regulator = LayerwiseRegulator(Controller({NAME: read}), Activity(), price=price)
        wanted = knapsack_levels(reduced, price, Level.D2, (Level.D2, Level.D4, Level.D6, Level.D8))
        assert np.array_equal(regulator.levels_for(NAME, x).numpy(), wanted), price


def test_the_rim_takes_the_loudest_of_the_blocks_left_at_the_base():
    regulator = a_regulator(price=1e30, rim=0.5)  # every block rests at the base, so the rim chooses among them all
    read = regulator.ctl.modules[NAME]
    x = torch.ones(1, 3, read.weight.shape[1], dtype=torch.bfloat16)
    reduced = (Activity().scores(read, x, row_norms(read)) / block_weights(read)).numpy()[0]
    width = round(0.5 * read.n_blocks)
    risen = np.flatnonzero(regulator.levels_for(NAME, x).numpy()[0] == D4)
    assert sorted(risen.tolist()) == sorted(np.argsort(-reduced)[:width].tolist())


def test_a_layout_set_on_the_card_builds_the_tables_one_from_the_host_builds():
    made = module(NAME)
    codes = torch.tensor([D2, D8] * (made.n_blocks // 2) + [D2] * (made.n_blocks % 2), dtype=torch.uint8)
    made.set_levels_on_device(codes, (Level.D2, Level.D4, Level.D6, Level.D8))
    rows, depths = made._row_layout(), made._depths
    assert np.array_equal(made.levels, codes.numpy())  # read back on demand, outside the pass
    made.set_levels(codes.numpy())  # the pass over the two layouts is read on the card (test_layerwise_gpu)
    assert torch.equal(rows, made._row_layout()) and torch.equal(depths, made._depths)
    made.set_levels(np.full(made.n_blocks, D4, dtype=np.uint8))
    assert np.array_equal(made.levels, np.full(made.n_blocks, D4))  # the codes left on the card are not the layout


def test_the_layout_read_back_holds_what_the_hooks_wrote():
    regulator = a_regulator(price=0.0)
    x = torch.ones(1, 2, regulator.ctl.modules[NAME].weight.shape[1], dtype=torch.bfloat16)
    codes = regulator.levels_for(NAME, x)
    regulator.levels_set[NAME] = codes
    assert np.array_equal(regulator.layout(), codes)


def test_a_pass_at_a_rung_costs_that_rungs_bits_a_weight():
    for price, rung in ((0.0, Level.D8), (1e30, Level.D2)):
        regulator = a_regulator(price=price)
        regulator.start_counting(2)
        regulator.decide(NAME, a_state(regulator))
        assert np.allclose(regulator.bits_a_weight(), float(rung.bits))


def test_the_count_ends_at_the_prompts_pass_and_the_generation_adds_nothing():
    regulator = a_regulator(price=0.0)
    state = a_state(regulator)
    regulator.start_counting(2)
    regulator.decide(NAME, state)
    for _ in range(5):  # the tokens of the generation decide the module again; the count is the prompt's pass alone
        regulator.decide(NAME, state)
    assert np.allclose(regulator.bits_a_weight(), float(Level.D8.bits))


def test_a_pass_that_did_not_reach_every_module_is_refused():
    ctl = Controller({NAME: module(NAME), "layers.0.mlp.up_proj": module("layers.0.mlp.up_proj")})
    regulator = LayerwiseRegulator(ctl, Activity(), price=0.0)
    regulator.start_counting(2)
    regulator.decide(NAME, a_state(regulator))
    with pytest.raises(RuntimeError, match="1 of 2 modules"):
        regulator.bits_a_weight()


def test_the_loud_question_of_a_batch_reads_more_bits_than_the_quiet_one():
    regulator = a_regulator(price=1e-3)
    inp = regulator.ctl.modules[NAME].weight.shape[1]
    state = torch.stack([torch.full((3, inp), 1.0), torch.full((3, inp), 0.01)]).to(torch.bfloat16)
    regulator.start_counting(2)
    regulator.decide(NAME, state)
    loud, quiet = regulator.bits_a_weight()
    assert loud > quiet and float(Level.D2.bits) <= quiet and loud <= float(Level.D8.bits)
