"""Refinements over the k-quant base on synthetic weights: no model, seconds."""

import pytest
import torch
from test_kquant_unit import DEVICE, ROWS, SUPER_BLOCKS, make_weight

from foqlens.kquant import Q2_K, Q4_K, QK_K
from foqlens.refinements import SENSITIVE_CLASSES, KQuantLadder, KRefinedWeight
from foqlens.quant import MAX_DEPTH, Level, _unpack

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_refinements_shrink_the_error_by_their_step(fmt):
    weight = make_weight(seed=2)
    copy = KRefinedWeight.quantize(weight, fmt)
    x = weight.float().view(ROWS, SUPER_BLOCKS, QK_K // fmt.block, fmt.block)
    step = copy.base.steps()
    within = (copy.base.dequantize() - x).abs() <= step / 2 * (1 + 1e-4)
    assert within.float().mean() > 0.9  # the rest are the weights the search clipped
    for depth in range(fmt.base_depth, MAX_DEPTH + 1):
        err = (copy.dequantize(torch.float32, depth).view_as(x) - x).abs()
        bound = step / 2 / 4 ** (depth - fmt.base_depth)
        assert torch.all(err[within.expand_as(err)] <= bound.expand_as(err)[within.expand_as(err)] * (1 + 1e-3) + 1e-7), depth


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_a_deeper_refinement_never_changes_a_shallower_read(fmt):
    weight = make_weight(seed=3)
    full = KRefinedWeight.quantize(weight, fmt)
    for depth in range(fmt.base_depth, MAX_DEPTH + 1):
        shallow = KRefinedWeight(base=full.base, refinements=full.refinements[: depth - fmt.base_depth].clone(), shape=full.shape)
        assert torch.equal(shallow.dequantize(torch.float32, depth), full.dequantize(torch.float32, depth))


def test_a_copy_is_never_read_shallower_than_its_base_and_costs_its_bits():
    copy = KRefinedWeight.quantize(make_weight(), Q4_K)
    assert torch.equal(copy.dequantize(torch.float32, 1), copy.dequantize(torch.float32, 2))
    assert Q2_K.bits_per_weight == 2.625 and Q4_K.bits_per_weight == 4.5
    assert copy.bits_per_weight(1) == 4.5 and copy.bits_per_weight(MAX_DEPTH) == 8.5


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_the_copy_reads_as_the_controller_reads_it_at_every_depth(fmt):
    """The base plus the refinements added one at a time, as the copy read before it became a _DepthReader."""
    weight = make_weight(seed=5)
    copy = KRefinedWeight.quantize(weight, fmt)
    x = torch.randn(3, weight.shape[1], device=DEVICE, dtype=torch.bfloat16)
    depths = list(range(1, MAX_DEPTH + 1))
    outputs = copy.linear_at_depths(x, None, depths)
    for depth, out in zip(depths, outputs, strict=True):
        w = copy.base.dequantize()
        step = copy.base.steps() / 4
        for e in range(copy.read_depth(depth) - fmt.base_depth):
            w = w + step * (_unpack(copy.refinements[e]).view(w.shape).float() - 1.5)
            step = step / 4
        expected = w.reshape(copy.shape).to(torch.bfloat16)
        assert torch.equal(copy.dequantize(torch.bfloat16, depth), expected), depth
        assert torch.equal(out, torch.nn.functional.linear(x, expected)), depth


def test_the_ladder_reads_each_class_from_its_own_base():
    ladder = KQuantLadder()
    weight = make_weight(seed=4)
    for name, fmt in (("layers.3.self_attn.q_proj", Q2_K), (f"layers.3.{SENSITIVE_CLASSES[0]}", Q4_K)):
        assert ladder.format_for(name) is fmt
        for level in (Level.D2, Level.D4, Level.D8):
            depth = max(level.depth, fmt.base_depth)
            expected = KRefinedWeight.quantize(weight, fmt, depth=depth).dequantize(weight.dtype, depth)
            assert torch.equal(ladder.read(name, weight, level), expected), (name, level)
