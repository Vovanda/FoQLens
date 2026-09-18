"""The k-quant kernel (kernels/kquant_matmul.cu) against the torch path: every depth, mixed layouts, a layout per token."""

import time
from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from foqlens.kernels.kquant import TILE_ROWS, kquant_matmul, kquant_matmul_fp32
from foqlens.kquant import Q2_K, Q4_K
from foqlens.quant import MAX_DEPTH, Level
from foqlens import precision
from foqlens.precision import MixedPrecisionLinear, samples
from foqlens.refinements import KQuantLadder, KRefinedWeight

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
OUT, IN, TOKENS = 3 * TILE_ROWS + 5, 1536, 11  # a partial block of rows and a partial tile of tokens
# The kernel builds the torch path's bf16 weights bit for bit and sums them times the inputs in fp32: against the same
# products summed in fp64 it is off by at most 1.6e-7 of the largest output (measured 2026-09-18).
TOLERANCE = 1e-6
# A module rounds its output to bf16: the kernel and unpacking may land a sum on neighbouring bf16 values.
BF16_ULP = 2**-7
LARGE_OUT, LARGE_IN = 12288, 1536  # E2B's widest module, for the timing


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


def close(got: torch.Tensor, expected: torch.Tensor) -> bool:
    scale = expected.abs().max().clamp_min(1e-12)
    return bool(((got - expected).abs().max() / scale) <= TOLERANCE)


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
@pytest.mark.parametrize("depth", range(1, MAX_DEPTH + 1))
def test_a_uniform_depth_reads_as_the_torch_path(fmt, depth):
    copy, x = make_copy(fmt), make_tokens()
    got = kquant_matmul_fp32(copy, x, blocks([depth] * 4))
    assert close(got, reference(copy, x, depth))
    bf16 = kquant_matmul(copy, x, blocks([depth] * 4))
    assert bf16.dtype == torch.bfloat16 and bf16.shape == (TOKENS, OUT)


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_every_block_is_read_at_its_own_depth_and_zero_reads_nothing(fmt):
    copy, x = make_copy(fmt), make_tokens()
    layout = [4, 0, 1, 3]
    got = kquant_matmul_fp32(copy, x, blocks(layout))
    for block, depth in enumerate(layout):
        rows = slice(block * TILE_ROWS, min((block + 1) * TILE_ROWS, OUT))
        if depth == 0:
            assert torch.equal(got[:, rows], torch.zeros_like(got[:, rows]))
        else:
            assert close(got[:, rows], reference(copy, x, depth)[:, rows]), block


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_every_token_reads_its_own_layout(fmt):
    copy, x = make_copy(fmt), make_tokens()
    layouts = torch.randint(0, MAX_DEPTH + 1, (TOKENS, -(-OUT // TILE_ROWS)), device=DEVICE, dtype=torch.uint8)
    got = kquant_matmul_fp32(copy, x, layouts)
    for token in range(TOKENS):
        alone = kquant_matmul_fp32(copy, x[token:token + 1], layouts[token])
        assert torch.equal(got[token], alone[0]), token


def test_a_copy_cut_short_is_never_read_deeper_than_it_holds():
    full = make_copy(Q2_K)
    short = KRefinedWeight(fmt=full.fmt, blocks=full.blocks, refinements=full.refinements[:1].clone(), shape=full.shape)
    x = make_tokens()
    assert torch.equal(kquant_matmul_fp32(short, x, blocks([4] * 4)), kquant_matmul_fp32(full, x, blocks([2] * 4)))


def seconds(call, repeats: int = 20) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


# The kernel is bound by its arithmetic, not by the bytes it reads: D2 and D8 take about the same time (0.20 and 0.21
# ms for 8 tokens, 2026-09-18), so a shallower read must merely not be slower, within the noise of a shared card.
TIME_NOISE = 1.15
# Unpacking a 12288x1536 module to D8 for a GEMM took 3.9 ms, the kernel 0.21: it must stay well below.
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
