"""The k-quant copy multiplied straight from its bytes (kquant_matmul.cu): every block of rows read to its own depth.

Invariant: the output is the torch path's - the copy read to each block's depth, rounded to bf16, times the input -
within the rounding of a sum over the input in fp32 (tests/test_kquant_kernel_gpu.py); a block at depth 0 outputs zero.
Invariant: the kernel reads a row's refinements only to its depth, so a shallower layout reads fewer bytes.
"""

from __future__ import annotations

import torch

from foqlens.kernels import HERE, launch, load
from foqlens.kquant import Q2_K, Q4_K
from foqlens.refinements import KRefinedWeight

TILE_ROWS = 64  # must match kquant_matmul.cu: the block of rows a depth is set on
BATCH_TILE = 8  # tokens a warp carries
WARPS_PER_CTA = 8  # warps in a thread block, one row each: 8 divides TILE_ROWS, so a thread block reads one depth
WARP = 32
KERNEL_SOURCE = HERE / "kquant_matmul.cu"  # both kernels live in one source
_KERNELS = {Q2_K.name: "kquant_matmul_q2_k", Q4_K.name: "kquant_matmul_q4_k"}  # one kernel per format


def kquant_matmul(copy: KRefinedWeight, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """bf16 [..., out]: `x` bf16 [..., in] times the copy, each block of TILE_ROWS rows read to `depth`.

    `depth` is uint8 [out / TILE_ROWS] for one layout, or [tokens, out / TILE_ROWS] for a layout per token; 0 reads
    nothing. No host synchronization: the launch reads only shapes.
    """
    out_features = copy.shape[0]
    return kquant_matmul_fp32(copy, x.reshape(-1, x.shape[-1]), depth).to(x.dtype).reshape(*x.shape[:-1], out_features)


def kquant_matmul_fp32(copy: KRefinedWeight, tokens: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """float32 [tokens, out]: the sums before the output is rounded to the input's type."""
    out_features, in_features = copy.shape
    tokens = tokens.contiguous()
    n_tokens = tokens.shape[0]
    stride = 0 if depth.ndim == 1 else depth.shape[-1]
    y = torch.empty(n_tokens, out_features, device=tokens.device, dtype=torch.float32)
    refinements = copy.refinements if copy.refinements.numel() else copy.blocks  # any valid pointer when there are none
    grid = (-(-out_features // WARPS_PER_CTA), -(-n_tokens // BATCH_TILE), 1)
    launch(load(_KERNELS[copy.fmt.name], KERNEL_SOURCE), grid, (WARP, WARPS_PER_CTA, 1),
           copy.blocks.contiguous(), refinements.contiguous(), tokens.view(torch.int16), depth.contiguous(), y,
           copy.refinements.shape[0], n_tokens, in_features, out_features, stride)
    return y
