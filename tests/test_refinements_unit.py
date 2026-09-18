"""Refinements over the k-quant base on synthetic weights: no model, seconds."""

import pytest
import torch
from test_kquant_unit import DEVICE, ROWS, SUPER_BLOCKS, make_weight

from foqlens.kquant import Q2_K, Q4_K, QK_K
from foqlens.refinements import ORDER_BITS, SENSITIVE_CLASSES, ExactTail, KQuantLadder, KRefinedWeight, from_ulp_order, ulp_order
from foqlens.quant import MAX_DEPTH, Level, _unpack

pytestmark = pytest.mark.gpu

SOURCE_TYPES = list(ORDER_BITS)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal bit for bit - a plain == would let -0.0 pass for +0.0 and fail every NaN."""
    return a.dtype == b.dtype and torch.equal(ulp_order(a), ulp_order(b))


@pytest.mark.parametrize("dtype", SOURCE_TYPES)
def test_the_ulp_order_keeps_the_order_of_floats_and_inverts_exactly(dtype):
    specials = torch.tensor([-float("inf"), -1.0, -1e-30, -0.0, 0.0, 1e-30, 1.0, float("inf")])
    values = torch.cat([specials.to(DEVICE), make_weight(seed=7).float().flatten()]).to(dtype)
    order = ulp_order(values)
    assert same_bits(from_ulp_order(order, dtype), values)
    ranked = values[torch.argsort(order)].float()
    assert torch.all(ranked[1:] >= ranked[:-1])
    assert ulp_order(torch.tensor([-0.0, 0.0], dtype=dtype)).tolist() == [-1, 0]


@pytest.mark.parametrize("dtype", SOURCE_TYPES)
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_the_exact_tail_restores_the_source_bit_for_bit(dtype, fmt):
    source = make_weight(seed=8).to(dtype)
    source[0, :QK_K] = 0.0  # a group the prediction may already hit, and signed zeros
    source[1, :3] = torch.tensor([-0.0, 0.0, -0.0], dtype=dtype)
    copy = KRefinedWeight.quantize(source, fmt)
    tail = ExactTail.encode(source, copy.prediction())
    assert same_bits(tail.decode(copy.prediction()), source)
    assert tail.bits_per_weight < ORDER_BITS[dtype]


def test_the_exact_tail_reads_back_from_its_tensors():
    source = make_weight(seed=9)
    copy = KRefinedWeight.quantize(source, Q2_K)
    tail = ExactTail.from_tensors(ExactTail.encode(source, copy.prediction()).tensors(), source.dtype)
    assert same_bits(tail.decode(copy.prediction()), source)


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
@pytest.mark.parametrize("depth", [1, 2, 3, MAX_DEPTH])
def test_a_copy_reads_the_same_after_its_tensors_are_taken_apart(fmt, depth):
    copy = KRefinedWeight.quantize(make_weight(seed=10), fmt, depth=max(depth, fmt.base_depth))
    back = KRefinedWeight.from_tensors(fmt, copy.tensors(), copy.shape)
    assert back.depth == copy.depth
    for read in range(1, MAX_DEPTH + 1):
        assert torch.equal(back.dequantize(torch.float32, read), copy.dequantize(torch.float32, read)), read


def test_the_prediction_is_the_same_on_the_cpu_and_the_gpu():
    """The tail is written on one device and may be read on another: its prediction must not depend on it."""
    copy = KRefinedWeight.quantize(make_weight(seed=11), Q2_K)
    on_cpu = KRefinedWeight.from_tensors(Q2_K, {k: t.cpu() for k, t in copy.tensors().items()}, copy.shape)
    assert same_bits(on_cpu.prediction(), copy.prediction().cpu())


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
        shallow = KRefinedWeight(fmt=full.fmt, blocks=full.blocks, refinements=full.refinements[: depth - fmt.base_depth].clone(), shape=full.shape)
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


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_a_copy_holds_its_bits_per_weight_and_no_byte_more(fmt):
    copy = KRefinedWeight.quantize(make_weight(seed=12), fmt)
    assert copy.nbytes * 8 == copy.bits_per_weight(MAX_DEPTH) * copy.shape[0] * copy.shape[1]
