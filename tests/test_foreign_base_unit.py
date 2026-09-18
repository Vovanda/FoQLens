"""A base read from a published GGUF file's bytes, and the stack built over it: synthetic blocks on the CPU, seconds."""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize

from foqlens.gguf_weights import GgufWeights
from foqlens.kquant import Q2_K, Q3_K, Q4_K, Q6_K, QK_K, KBase, from_gguf_blocks, gguf_blocks
from foqlens.refinements import ExactTail, ForeignLadder, KQuantLadder, KRefinedWeight, ulp_order
from foqlens.quant import MAX_DEPTH

ROWS, SUPER_BLOCKS = 4, 3
BLOCK_BYTES = {Q3_K.name: 110, Q6_K.name: 210}  # sizeof(block_q3_K), sizeof(block_q6_K)
QTYPE = {Q3_K.name: GGMLQuantizationType.Q3_K, Q6_K.name: GGMLQuantizationType.Q6_K}
D_SCALE = 1e-3  # the super-block multiplier of a trained layer's weights, roughly
SYMMETRIC = [Q3_K, Q6_K]


def foreign_blocks(fmt, seed: int = 0) -> torch.Tensor:
    """uint8 [rows, super-blocks, bytes]: random codes and scales - any bytes are a valid block - with a finite d last."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (ROWS * SUPER_BLOCKS, BLOCK_BYTES[fmt.name]), dtype=np.uint8)
    d = (rng.random(ROWS * SUPER_BLOCKS) * D_SCALE).astype(np.float16)
    blocks[:, -2:] = d.view(np.uint8).reshape(-1, 2)
    return torch.from_numpy(blocks).view(ROWS, SUPER_BLOCKS, -1)


@pytest.mark.parametrize("fmt", SYMMETRIC, ids=lambda f: f.name)
def test_a_symmetric_base_reads_exactly_as_gguf_dequantizes_its_bytes(fmt):
    blocks = foreign_blocks(fmt)
    ours = from_gguf_blocks(blocks, fmt).dequantize().numpy().reshape(-1, QK_K)
    theirs = dequantize(blocks.numpy().reshape(-1), QTYPE[fmt.name]).reshape(-1, QK_K)
    np.testing.assert_array_equal(ours, theirs)


def test_a_symmetric_format_has_no_quantizer_and_its_size_is_ggmls():
    with pytest.raises(ValueError, match="no quantizer"):
        KBase.quantize(torch.zeros(ROWS, QK_K), Q3_K)
    for fmt in SYMMETRIC:
        assert fmt.bits_per_weight * QK_K / 8 == BLOCK_BYTES[fmt.name]


def source_over(base: KBase, seed: int) -> torch.Tensor:
    """fp32 weights around the base, each within half its block's step - as a quantizer leaves them. fp32, so that
    rounding the source does not carry a weight past the half step the first refinement covers."""
    torch.manual_seed(seed)
    read = base.dequantize()
    noise = (torch.rand_like(read) - 0.5) * base.steps().abs()
    return (read + noise).reshape(ROWS, SUPER_BLOCKS * QK_K)


@pytest.mark.parametrize("fmt", SYMMETRIC, ids=lambda f: f.name)
def test_the_stack_over_a_foreign_base_reads_the_base_as_it_lies_and_refines_toward_the_source(fmt):
    blocks = foreign_blocks(fmt, seed=1)
    base = from_gguf_blocks(blocks, fmt)
    source = source_over(base, seed=2)
    copy = KRefinedWeight.over(blocks, fmt, source)
    assert copy.depth == MAX_DEPTH and torch.equal(copy.blocks, blocks)
    at_base = copy.dequantize(torch.float32, fmt.base_depth)
    assert torch.equal(at_base, base.dequantize().reshape(source.shape))
    step = base.steps().abs().expand(-1, -1, -1, fmt.block).reshape(source.shape)
    for k in range(1, MAX_DEPTH - fmt.base_depth + 1):
        error = (copy.dequantize(torch.float32, fmt.base_depth + k) - source.float()).abs()
        assert torch.all(error <= step / 2 / 4**k * (1 + 1e-5)), k


@pytest.mark.parametrize("fmt", SYMMETRIC, ids=lambda f: f.name)
def test_the_exact_tail_over_a_foreign_base_restores_the_source_bit_for_bit(fmt):
    blocks = foreign_blocks(fmt, seed=3)
    source = source_over(from_gguf_blocks(blocks, fmt), seed=4)
    copy = KRefinedWeight.over(blocks, fmt, source)
    tail = ExactTail.encode(source, copy.prediction())
    assert torch.equal(ulp_order(tail.decode(copy.prediction())), ulp_order(source))


def gguf_file(path, blocks: torch.Tensor) -> None:
    """A GGUF file with one Q3_K and one float module of layer 0, as llama.cpp names them."""
    writer = GGUFWriter(path, "gemma4")
    rows = blocks.numpy().reshape(ROWS, -1)
    writer.add_tensor("blk.0.attn_q.weight", rows, raw_dtype=QTYPE[Q3_K.name])  # the shape follows from the bytes
    writer.add_tensor("blk.0.attn_k.weight", np.zeros((ROWS, SUPER_BLOCKS * QK_K), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_a_published_files_k_quant_tensor_is_the_base_and_a_float_one_keeps_ours(tmp_path):
    blocks = foreign_blocks(Q3_K, seed=6)
    gguf_file(tmp_path / "model.gguf", blocks)
    ladder = ForeignLadder(GgufWeights(tmp_path / "model.gguf").base_blocks)
    shape = (ROWS, SUPER_BLOCKS * QK_K)
    q = ladder.quantize("layers.0.self_attn.q_proj", source_over(from_gguf_blocks(blocks, Q3_K), seed=7))
    assert q.fmt is Q3_K and torch.equal(q.blocks, blocks) and q.depth == MAX_DEPTH
    k = ladder.quantize("layers.0.self_attn.k_proj", (torch.randn(shape) * 0.02).to(torch.bfloat16))
    assert k.fmt is KQuantLadder().format_for("layers.0.self_attn.k_proj")


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K], ids=lambda f: f.name)
def test_our_own_copy_is_the_stack_over_its_own_blocks(fmt):
    torch.manual_seed(5)
    weight = (torch.randn(ROWS, SUPER_BLOCKS * QK_K) * 0.02).to(torch.bfloat16)
    ours = KRefinedWeight.quantize(weight, fmt)
    over = KRefinedWeight.over(gguf_blocks(KBase.quantize(weight, fmt)), fmt, weight)
    assert torch.equal(ours.blocks, over.blocks) and torch.equal(ours.refinements, over.refinements)
