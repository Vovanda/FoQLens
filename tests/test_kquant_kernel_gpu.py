"""The k-quant kernels (kernels/kquant_matmul.cu, kquant_mma.cu) against the torch path: every depth, mixed layouts, a
layout per token."""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from foqlens.kernels.kquant import CUDA_CORES, TENSOR_CORES, TILE_ROWS, kquant_matmul, kquant_matmul_fp32
from foqlens.kquant import Q2_K, Q3_K, Q4_K, Q6_K, QK_K, from_gguf_blocks
from foqlens.quant import MAX_DEPTH, Level
from foqlens import precision
from foqlens.precision import MixedPrecisionLinear, samples
from foqlens.refinements import KQuantLadder, KRefinedWeight

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
OUT, IN, TOKENS = 3 * TILE_ROWS + 5, 1536, 11  # a partial block of rows and a partial tile of tokens
# Both kernels build the torch path's bf16 weights bit for bit and sum them times the inputs in fp32. Against the same
# products summed in fp64 the CUDA cores are off by at most 1.6e-7 of the largest output; the tensor cores by 2.9e-6,
# since an mma adds its products with truncation (Fasi et al. 2021, "Numerical behavior of NVIDIA tensor cores") -
# still a thousandth of the bf16 step the module rounds its output to (measured 2026-09-18).
TOLERANCE = {"cuda-cores": 1e-6, "tensor-cores": 1e-5}
# A module rounds its output to bf16: the kernel and unpacking may land a sum on neighbouring bf16 values.
BF16_ULP = 2**-7
LARGE_OUT, LARGE_IN = 12288, 1536  # E2B's widest module, for the timing


KERNELS = {"cuda-cores": CUDA_CORES, "tensor-cores": TENSOR_CORES}
by_kernel = pytest.mark.parametrize("kernel", list(KERNELS))  # a kernel's name: TOLERANCE and KERNELS are keyed by it


def make_copy(fmt, out=OUT, inp=IN, seed=0) -> KRefinedWeight:
    torch.manual_seed(seed)
    weight = (torch.distributions.StudentT(3.0).sample((out, inp)) * 0.02).to(DEVICE, torch.bfloat16)
    return KRefinedWeight.quantize(weight, fmt)


def make_tokens(n=TOKENS, inp=IN) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, inp, device=DEVICE, dtype=torch.bfloat16)


def reference(copy: KRefinedWeight, x: torch.Tensor, depth: int) -> torch.Tensor:
    """float32: the torch path's weight at `depth`, rounded to bf16, times the input, summed in fp64 then fp32."""
    weight = copy.dequantize(torch.bfloat16, depth).double()
    return (x.double() @ weight.T).float()


def blocks(depths: list[int], out=OUT) -> torch.Tensor:
    return torch.tensor(depths, dtype=torch.uint8, device=DEVICE)[: -(-out // TILE_ROWS)]


def close(got: torch.Tensor, expected: torch.Tensor, kernel: str = "cuda-cores") -> bool:
    scale = expected.abs().max().clamp_min(1e-12)
    return bool(((got - expected).abs().max() / scale) <= TOLERANCE[kernel])


def read(copy: KRefinedWeight, x: torch.Tensor, depth: torch.Tensor, kernel: str) -> torch.Tensor:
    return kquant_matmul_fp32(copy, x, depth, KERNELS[kernel])


@by_kernel
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
@pytest.mark.parametrize("depth", range(1, MAX_DEPTH + 1))
def test_a_uniform_depth_reads_as_the_torch_path(fmt, depth, kernel):
    copy, x = make_copy(fmt), make_tokens()
    got = read(copy, x, blocks([depth] * 4), kernel)
    assert close(got, reference(copy, x, depth), kernel)
    bf16 = kquant_matmul(copy, x, blocks([depth] * 4))
    assert bf16.dtype == torch.bfloat16 and bf16.shape == (TOKENS, OUT)


# Input widths of E2B's modules and one super-block: the tensor cores split the input over 1, 6 and 8 warps.
WIDTHS = [256, 1536, 2048, 12288]


@by_kernel
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
@pytest.mark.parametrize("width", WIDTHS)
def test_every_input_width_reads_as_the_torch_path(fmt, width, kernel):
    copy, x = make_copy(fmt, inp=width), make_tokens(inp=width)
    assert close(read(copy, x, blocks([MAX_DEPTH, 1, 3, 2]), kernel)[:, :TILE_ROWS],
                 reference(copy, x, MAX_DEPTH)[:, :TILE_ROWS], kernel)


@by_kernel
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_every_block_is_read_at_its_own_depth_and_zero_reads_nothing(fmt, kernel):
    copy, x = make_copy(fmt), make_tokens()
    layout = [4, 0, 1, 3]
    got = read(copy, x, blocks(layout), kernel)
    for block, depth in enumerate(layout):
        rows = slice(block * TILE_ROWS, min((block + 1) * TILE_ROWS, OUT))
        if depth == 0:
            assert torch.equal(got[:, rows], torch.zeros_like(got[:, rows]))
        else:
            assert close(got[:, rows], reference(copy, x, depth)[:, rows], kernel), block


@by_kernel
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_every_token_reads_its_own_layout(fmt, kernel):
    copy, x = make_copy(fmt), make_tokens()
    layouts = torch.randint(0, MAX_DEPTH + 1, (TOKENS, -(-OUT // TILE_ROWS)), device=DEVICE, dtype=torch.uint8)
    got = read(copy, x, layouts, kernel)
    for token in range(TOKENS):
        alone = read(copy, x[token:token + 1], layouts[token], kernel)
        assert torch.equal(got[token], alone[0]), token


MANY_TOKENS = 70  # past two thread blocks of tokens on tensor cores, a partial one at the end


@by_kernel
@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_many_tokens_each_at_its_own_layout_read_as_the_torch_path(fmt, kernel):
    copy, x = make_copy(fmt), make_tokens(MANY_TOKENS)
    n_blocks = -(-OUT // TILE_ROWS)
    layouts = torch.randint(0, MAX_DEPTH + 1, (MANY_TOKENS, n_blocks), device=DEVICE, dtype=torch.uint8)
    got = read(copy, x, layouts, kernel)
    at = {depth: reference(copy, x, depth) for depth in range(1, MAX_DEPTH + 1)}
    expected = torch.zeros_like(got)
    for token in range(MANY_TOKENS):
        for block in range(n_blocks):
            depth = int(layouts[token, block])
            if depth:
                rows = slice(block * TILE_ROWS, min((block + 1) * TILE_ROWS, OUT))
                expected[token, rows] = at[depth][token, rows]
    assert close(got, expected, kernel)
    unread = layouts.repeat_interleave(TILE_ROWS, dim=1)[:, :OUT] == 0
    assert torch.equal(got[unread], torch.zeros_like(got[unread]))


@by_kernel
def test_a_copy_cut_short_is_never_read_deeper_than_it_holds(kernel):
    full = make_copy(Q2_K)
    short = KRefinedWeight(fmt=full.fmt, blocks=full.blocks, refinements=full.refinements[:1].clone(), shape=full.shape)
    x = make_tokens()
    assert torch.equal(read(short, x, blocks([4] * 4), kernel), read(full, x, blocks([2] * 4), kernel))


def seconds(call, repeats: int = 20) -> float:
    """Seconds a call on the device, `repeats` calls replayed as one CUDA graph: a decoding step replays a graph, and
    timed from the host a small kernel measures its launch instead (0.12 ms for either kernel at any depth)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            call()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph.reset()
    return start.elapsed_time(end) / 1e3 / repeats


# A shallower read decodes fewer refinements: on tensor cores D2 took 0.060 ms against 0.103 at D8 for 8 tokens
# (2026-09-18), so a shallower read must not be slower, within the noise of a shared card.
TIME_NOISE = 1.15
# Unpacking a 12288x1536 module to D8 for a GEMM took 3.8 ms, the kernel 0.10: it must stay well below.
KERNEL_SHARE_OF_UNPACKING = 0.25


def test_the_kernel_reads_far_faster_than_unpacking_and_a_shallower_read_is_not_slower():
    copy, x = make_copy(Q2_K, LARGE_OUT, LARGE_IN), make_tokens(8, LARGE_IN)
    layout = {depth: torch.full((LARGE_OUT // TILE_ROWS,), depth, dtype=torch.uint8, device=DEVICE) for depth in (1, MAX_DEPTH)}
    kernel = {depth: seconds(lambda d=depth: kquant_matmul(copy, x, layout[d])) for depth in layout}
    unpacking = seconds(lambda: copy.matmul(x, None, MAX_DEPTH), repeats=5)
    print(f"12288x1536, 8 tokens: kernel D2 {kernel[1] * 1e3:.3f} ms, D8 {kernel[MAX_DEPTH] * 1e3:.3f} ms, "
          f"unpacking D8 {unpacking * 1e3:.3f} ms")
    assert kernel[MAX_DEPTH] < KERNEL_SHARE_OF_UNPACKING * unpacking
    assert kernel[1] < TIME_NOISE * kernel[MAX_DEPTH]


def kquant_module(out=OUT, inp=IN) -> MixedPrecisionLinear:
    torch.manual_seed(2)
    linear = nn.Linear(inp, out, bias=False, device=DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.02)
    return MixedPrecisionLinear(linear, TILE_ROWS, partial(KQuantLadder().quantize, "layers.0.self_attn.q_proj"))


def unpacked(module: MixedPrecisionLinear, x: torch.Tensor, monkeypatch) -> torch.Tensor:
    """The module's output with the kernel off: every read unpacks the weight."""
    with monkeypatch.context() as patch:
        patch.setattr(precision, "KERNEL", False)
        module.set_levels(module.levels)
        out = module(x)
    module.set_levels(module.levels)
    return out


def test_a_module_reads_a_mixed_layout_through_the_kernel_as_through_unpacking(monkeypatch):
    module = kquant_module()
    layout = np.array([Level.D8, Level.ZERO, Level.D2, Level.D6], dtype=np.uint8)
    module.set_levels(layout)
    assert module._depths is not None
    x = make_tokens(3, IN).view(3, 1, IN)
    got = module(x).float()
    expected = unpacked(module, x, monkeypatch).float()
    assert got.shape == expected.shape
    assert (got - expected).abs().max() <= BF16_ULP * expected.abs().max()
    assert torch.equal(got[..., TILE_ROWS:2 * TILE_ROWS], torch.zeros_like(got[..., TILE_ROWS:2 * TILE_ROWS]))


def test_per_sample_layouts_and_a_part_of_the_batch_read_through_the_kernel(monkeypatch):
    module = kquant_module()
    layouts = np.array([[Level.D8, Level.D2, Level.ZERO, Level.D4],
                        [Level.D2, Level.D2, Level.D6, Level.ZERO],
                        [Level.ZERO, Level.D8, Level.D8, Level.D2]], dtype=np.uint8)
    module.set_levels(layouts)
    x = make_tokens(6, IN).view(3, 2, IN)  # three samples of two tokens
    got = module(x)
    for sample in range(3):
        alone = kquant_module()
        alone.set_levels(layouts[sample])
        assert torch.equal(got[sample], alone(x[sample:sample + 1])[0]), sample
    with samples(module, slice(1, 3)):
        assert torch.equal(module(x[1:3]), got[1:3])


def test_a_long_input_unpacks_instead(monkeypatch):
    module = kquant_module()
    module.set_levels(Level.D4)
    x = make_tokens(precision.KERNEL_MAX_TOKENS + 1, IN)
    assert torch.equal(module(x), unpacked(module, x, monkeypatch))


def test_a_copy_of_another_kind_and_a_baked_level_are_never_read_by_the_kernel():
    symmetric = MixedPrecisionLinear(nn.Linear(IN, OUT, bias=False, device=DEVICE, dtype=torch.bfloat16), TILE_ROWS)
    symmetric.set_levels(Level.D4)
    assert symmetric._depths is None
    baked = kquant_module()
    baked.bake(Level.D4)
    assert baked._depths is None
    mixed_bf16 = kquant_module()
    mixed_bf16.set_levels(np.array([Level.BF16, Level.D4, Level.D4, Level.D4], dtype=np.uint8))
    assert mixed_bf16._depths is None


# ==== A published file's bases: Q3_K and Q6_K (#29) ====
# A copy over bartowski's Q2_K holds 70 modules on Q3_K and 17 on Q6_K. Random bytes are a valid block of either but for
# d, so the blocks are random codes and scales with a finite d of a trained layer's size.
FOREIGN = [Q3_K, Q6_K]
BLOCK_BYTES = {Q3_K.name: 110, Q6_K.name: 210}  # sizeof(block_q3_K), sizeof(block_q6_K)
D_SCALE = 1e-3


def foreign_copy(fmt, seed=0) -> KRefinedWeight:
    """The stack over random blocks of `fmt`, refining a source within half a block step of the base."""
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randint(0, 256, (OUT, IN // QK_K, BLOCK_BYTES[fmt.name]), dtype=torch.uint8, generator=generator)
    d = (torch.rand(OUT, IN // QK_K, generator=generator) * D_SCALE).half()
    blocks[..., -2:] = d.view(torch.uint8).view(OUT, IN // QK_K, 2)
    base = from_gguf_blocks(blocks, fmt)
    noise = (torch.rand(base.codes.shape, generator=generator) - 0.5) * base.steps().abs()
    source = (base.dequantize() + noise).reshape(OUT, IN)
    return KRefinedWeight.over(blocks.to(DEVICE), fmt, source.to(DEVICE))


@pytest.mark.parametrize("fmt", FOREIGN, ids=lambda f: f.name)
def test_a_published_base_reads_as_the_torch_path_at_every_depth(fmt):
    copy, x = foreign_copy(fmt), make_tokens()
    for depth in range(fmt.base_depth, MAX_DEPTH + 1):
        got = read(copy, x, blocks([depth] * 4), "tensor-cores")
        assert close(got, reference(copy, x, depth), "tensor-cores"), depth


@pytest.mark.parametrize("fmt", FOREIGN, ids=lambda f: f.name)
def test_a_published_base_reads_a_mixed_layout_and_zero_reads_nothing(fmt):
    copy, x = foreign_copy(fmt, seed=1), make_tokens()
    layout = [MAX_DEPTH, 0, fmt.base_depth, MAX_DEPTH]
    got = read(copy, x, blocks(layout), "tensor-cores")
    for block, depth in enumerate(layout):
        rows = slice(block * TILE_ROWS, min((block + 1) * TILE_ROWS, OUT))
        if depth == 0:
            assert torch.equal(got[:, rows], torch.zeros_like(got[:, rows]))
        else:
            assert close(got[:, rows], reference(copy, x, depth)[:, rows], "tensor-cores"), block


@pytest.mark.parametrize("fmt", FOREIGN, ids=lambda f: f.name)
def test_every_token_reads_its_own_layout_over_a_published_base(fmt):
    copy, x = foreign_copy(fmt, seed=2), make_tokens()
    layouts = torch.randint(0, MAX_DEPTH + 1, (TOKENS, -(-OUT // TILE_ROWS)), device=DEVICE, dtype=torch.uint8)
    got = read(copy, x, layouts, "tensor-cores")
    for token in range(TOKENS):
        assert torch.equal(got[token], read(copy, x[token:token + 1], layouts[token], "tensor-cores")[0]), token
