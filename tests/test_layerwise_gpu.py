"""The layer-wise regulator on the card: a layout decided inside the pass reads as one set from the host."""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from foqlens.kernels.kquant import TILE_ROWS
from foqlens.layerwise import Activity, LayerwiseRegulator
from foqlens.precision import Controller, MixedPrecisionLinear
from foqlens.quant import Level
from foqlens.refinements import KQuantLadder

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
NAME = "layers.0.mlp.gate_proj"
OUT, IN = 4 * TILE_ROWS, 1536  # four blocks, one per rung of the ladder
LADDER = (Level.D2, Level.D4, Level.D6, Level.D8)
# The kernel reads a short input straight from the copy and unpacks a long one for a GEMM (precision.KERNEL_MAX_TOKENS):
# both paths take the layout, so both are read here.
SHORT, LONG = 3, 64


def module(name: str = NAME, out: int = OUT, inp: int = IN) -> MixedPrecisionLinear:
    torch.manual_seed(0)
    linear = nn.Linear(inp, out, bias=False, device=DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.02)
    made = MixedPrecisionLinear(linear, TILE_ROWS, partial(KQuantLadder().quantize, name))
    made.set_levels(Level.D8)  # the copy is built on the first read of a depth
    return made


def a_ladder_layout(n_blocks: int) -> torch.Tensor:
    """One block per rung, as many times as the module has blocks."""
    rungs = torch.tensor([int(lv) for lv in LADDER], dtype=torch.uint8, device=DEVICE)
    return rungs.repeat(-(-n_blocks // len(LADDER)))[:n_blocks]


@pytest.mark.parametrize("tokens", [SHORT, LONG])
@pytest.mark.parametrize("per_sample", [False, True])
def test_a_layout_set_on_the_card_reads_as_one_set_from_the_host(per_sample, tokens):
    made = module()
    codes = a_ladder_layout(made.n_blocks)
    if per_sample:
        codes = torch.stack([codes, codes.flip(0)])
    x = torch.randn(codes.shape[0] if per_sample else 1, tokens, IN, device=DEVICE, dtype=torch.bfloat16)
    made.set_levels(codes.cpu().numpy())
    want = made(x)
    made.set_levels_on_device(codes, LADDER)
    assert torch.equal(made(x), want)


def test_a_sample_of_a_layout_on_the_card_reads_its_own_rows():
    made = module()
    codes = torch.stack([a_ladder_layout(made.n_blocks), a_ladder_layout(made.n_blocks).flip(0)])
    x = torch.randn(codes.shape[0], SHORT, IN, device=DEVICE, dtype=torch.bfloat16)
    made.set_levels_on_device(codes, LADDER)
    whole = made(x)
    try:
        for sample in range(codes.shape[0]):
            made.read_samples(slice(sample, sample + 1))
            assert torch.equal(made(x[sample:sample + 1]), whole[sample:sample + 1]), sample
    finally:
        made.read_samples(slice(None))


def test_the_pass_never_waits_for_the_card():
    """The debug mode turns every device-to-host synchronization into an exception: the regulator decides a layout,
    writes it and counts what it reads without one, so the pipeline is never stopped in front of a layer."""
    made = module()
    regulator = LayerwiseRegulator(Controller({NAME: made}), Activity(), price=1e-3, rim=0.25)
    state = torch.randn(2, SHORT, IN, device=DEVICE, dtype=torch.bfloat16)
    regulator.start_counting(state.shape[0])
    regulator.decide(NAME, state)  # the first pass loads the kernel and builds the copies
    made(state)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        regulator.decide(NAME, state)
        made(state)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_a_captured_step_decides_the_layout_again_at_every_replay():
    """The point of keeping the layout on the card: a step is captured with the decision inside it, and a replay runs
    the decision again instead of freezing the layout it was captured on. Nothing of Python runs at a replay, so the
    proof is that two states replayed through the same graph leave two layouts, each the one the eager path gives."""
    made = module()
    regulator = LayerwiseRegulator(Controller({NAME: made}), Activity(), price=1e-3)
    state = torch.empty(1, SHORT, IN, device=DEVICE, dtype=torch.bfloat16)  # the only buffer the captured step reads

    def step() -> None:
        regulator.decide(NAME, state)
        made(state)

    state.fill_(1.0)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):  # the first run allocates and loads the kernel; a capture may do neither
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    decided = {}
    for name, value in (("loud", 1.0), ("quiet", 0.02)):
        state.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        decided[name] = made.levels
        assert np.array_equal(decided[name], regulator.levels_for(NAME, state).cpu().numpy()), name
    assert not np.array_equal(decided["loud"], decided["quiet"])  # a frozen layout would give one answer twice


def test_the_pass_reads_the_layout_the_regulator_decided_on_the_card():
    made = module()
    regulator = LayerwiseRegulator(Controller({NAME: made}), Activity(), price=1e-3)
    state = torch.randn(2, SHORT, IN, device=DEVICE, dtype=torch.bfloat16)
    regulator.start_counting(state.shape[0])
    codes = regulator.decide(NAME, state)
    assert codes.device.type == DEVICE and codes.shape == (state.shape[0], made.n_blocks)
    got = made(state)
    made.set_levels(codes.cpu().numpy())
    assert torch.equal(made(state), got)
    spent = regulator.bits_a_weight()
    assert (float(Level.D2.bits) <= spent).all() and (spent <= float(Level.D8.bits)).all()
