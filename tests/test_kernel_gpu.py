"""The sliced matmul kernel: it reads a row only to the depth its block is set to (foqlens/kernels)."""

import pytest
import torch

from foqlens.kernels import launch, load
from foqlens.quant import SLICE_GROUP, SlicedWeight

pytestmark = pytest.mark.gpu

TILE_ROWS, BATCH_PER_BLOCK = 64, 16  # must match sliced_matmul.cu
OUT, IN, BATCH = 256, 512, 5
TOLERANCE = 1e-4  # against the same read in torch: the kernel sums in another order, in fp32


@pytest.fixture(scope="module")
def sliced():
    torch.manual_seed(0)
    weight = torch.randn(OUT, IN, device="cuda") * 0.02
    copy = SlicedWeight.quantize(weight)
    x = torch.randn(BATCH, IN, device="cuda")
    return copy, x


def kernel_matmul(copy: SlicedWeight, x: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    out_features = copy.packed.shape[1]
    y = torch.empty(x.shape[0], out_features, device=x.device, dtype=torch.float32)
    grid = (out_features // TILE_ROWS, (x.shape[0] + BATCH_PER_BLOCK - 1) // BATCH_PER_BLOCK, 1)
    launch(load("sliced_matmul"), grid, (BATCH_PER_BLOCK, TILE_ROWS, 1),
           copy.packed.contiguous(), copy.scale.view(out_features, -1).contiguous(), x, depths, y,
           x.shape[0], x.shape[1], out_features)
    torch.cuda.synchronize()
    return y


def uniform(depth: int) -> torch.Tensor:
    return torch.full((OUT // TILE_ROWS,), depth, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_the_kernel_reads_a_uniform_depth_as_torch_does(sliced, depth):
    copy, x = sliced
    want = torch.nn.functional.linear(x, copy.dequantize(torch.float32, depth))
    got = kernel_matmul(copy, x, uniform(depth))
    assert (got - want).abs().max() / want.abs().max() < TOLERANCE


def test_a_block_at_zero_reads_nothing(sliced):
    copy, x = sliced
    assert kernel_matmul(copy, x, uniform(0)).abs().max() == 0


def test_every_block_is_read_at_its_own_depth(sliced):
    copy, x = sliced
    depths = uniform(2)
    depths[0], depths[-1] = 4, 0
    rows = torch.cat([copy.dequantize(torch.float32, int(d))[i * TILE_ROWS:(i + 1) * TILE_ROWS]
                      for i, d in enumerate(depths.tolist())])
    want = torch.nn.functional.linear(x, rows)
    got = kernel_matmul(copy, x, depths)
    assert (got - want).abs().max() / want.abs().max() < TOLERANCE
    assert got[:, -TILE_ROWS:].abs().max() == 0  # the block at ZERO stays zero inside a mixed layout


def test_the_scale_is_read_per_group_of_a_row(sliced):
    """A group's scale must reach only its own weights: changing one group moves only that output."""
    copy, x = sliced
    baseline = kernel_matmul(copy, x, uniform(4))
    touched = SlicedWeight(packed=copy.packed, scale=copy.scale.clone())
    touched.scale[0, 0] *= 2  # the first group of row 0
    moved = kernel_matmul(touched, x, uniform(4))
    difference = (moved - baseline).abs().max(dim=0).values
    assert difference[0] > 0 and difference[1:].max() == 0


def test_reading_shallower_costs_less_time(sliced):
    """The point of the kernel: the work follows the depth, so D2 is faster than D8."""
    import time

    copy, x = sliced
    big = torch.randn(BATCH, IN, device="cuda")

    def timed(depth: int) -> float:
        kernel_matmul(copy, big, uniform(depth))
        t0 = time.perf_counter()
        for _ in range(20):
            kernel_matmul(copy, big, uniform(depth))
        return time.perf_counter() - t0

    assert timed(1) < timed(4)
