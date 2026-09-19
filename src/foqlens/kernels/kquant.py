"""The k-quant copy multiplied straight from its bytes: every block of rows read to its own depth.

Two kernels read the copy the same way and differ in where they multiply: kquant_matmul.cu on CUDA cores, a warp per
output row, and kquant_mma.cu on tensor cores, a warp per 16 rows decoding the weight straight into the mma's A
fragment.

Invariant: the output is the torch path's - the copy read to each block's depth, rounded to bf16, times the input -
within the rounding of a sum over the input in fp32: against an fp64 sum at most 1.6e-7 of the largest output on CUDA
cores, 2.9e-6 on tensor cores, whose mma adds with truncation (tests/test_kquant_kernel_gpu.py); a block at depth 0
outputs zero.
Invariant: the kernel reads a row's refinements only to its depth, so a shallower layout reads fewer bytes.
Invariant: a token's output does not depend on the other tokens of the batch.
Invariant: kquant_unpack writes the torch path's bf16 weight bit for bit, every block of rows at its depth and zero at
depth 0, on every base (tests/test_kquant_kernel_gpu.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from foqlens.kernels import HERE, launch, load
from foqlens.kquant import Q2_K, Q3_K, Q4_K, Q6_K, QK_K, KFormat
from foqlens.quant import MAX_DEPTH
from foqlens.refinements import KRefinedWeight

TILE_ROWS = 64  # must match both sources: the block of rows a depth is set on
WARP = 32
PLANE_ROW_BYTES = QK_K // 4  # a refinement over a super-block row: a 2-bit code a weight


@dataclass(frozen=True)
class KQuantKernel:
    """One source's kernels, one per format, and the shape of its launch."""

    source: Path
    names: dict[str, str]  # format name -> kernel name
    rows_per_cta: int  # output rows a thread block covers
    tokens_per_cta: int  # tokens a thread block covers
    warps: int  # warps in a thread block; with split_k, the most it takes
    split_k: bool = False  # the warps split the input: as many as the largest divisor of its super-blocks
    # format name -> refinements a warp stages beside the base block, in dynamic shared memory (Stage in the source)
    stage_refinements: dict[str, int] | None = None

    def shared_bytes(self, copy: KRefinedWeight, warps: int) -> int:
        """Dynamic shared memory: every warp stages its rows' base blocks and refinement planes of a super-block."""
        if not self.stage_refinements:
            return 0
        row = copy.blocks.shape[-1] + self.stage_refinements[copy.fmt.name] * PLANE_ROW_BYTES
        return warps * self.rows_per_cta * row

    def warps_for(self, in_features: int) -> int:
        if not self.split_k:
            return self.warps
        super_blocks = in_features // QK_K
        return max(w for w in range(1, self.warps + 1) if super_blocks % w == 0)

    def __call__(self, copy: KRefinedWeight, tokens: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        out_features, in_features = copy.shape
        n_tokens = tokens.shape[0]
        stride = 0 if depth.ndim == 1 else depth.shape[-1]
        y = torch.empty(n_tokens, out_features, device=tokens.device, dtype=torch.float32)
        refinements = copy.refinements if copy.refinements.numel() else copy.blocks  # any valid pointer when none
        grid = (-(-out_features // self.rows_per_cta), -(-n_tokens // self.tokens_per_cta), 1)
        warps = self.warps_for(in_features)
        shared = self.shared_bytes(copy, warps)
        launch(load(self.names[copy.fmt.name], self.source), grid, (WARP, warps, 1),
               copy.blocks.contiguous(), refinements.contiguous(), tokens.view(torch.int16), depth.contiguous(), y,
               copy.refinements.shape[0], n_tokens, in_features, out_features, stride, shared_bytes=shared)
        return y


# 8 warps of a row each: 8 divides TILE_ROWS, so a thread block reads one depth; a warp carries 8 tokens
CUDA_CORES = KQuantKernel(HERE / "kquant_matmul.cu", {Q2_K.name: "kquant_matmul_q2_k", Q4_K.name: "kquant_matmul_q4_k"},
                          rows_per_cta=8, tokens_per_cta=8, warps=8)
# 16 rows, the mma's M, and 32 tokens, four tiles of its N = 8; up to 8 warps split the input (MAX_SPLIT). A warp
# stages 16 rows of a super-block: the base block and the refinements to D8 (MAX_REFINEMENTS in the source). It reads
# every base the model file holds: our Q2_K / Q4_K and a published file's Q3_K / Q6_K.
TENSOR_CORES = KQuantKernel(
    HERE / "kquant_mma.cu",
    {fmt.name: f"kquant_mma_{fmt.name.lower()}" for fmt in (Q2_K, Q3_K, Q4_K, Q6_K)},
    rows_per_cta=16, tokens_per_cta=32, warps=8, split_k=True,
    stage_refinements={fmt.name: MAX_DEPTH - fmt.base_depth for fmt in (Q2_K, Q3_K, Q4_K, Q6_K)})
# The kernel the bench multiplies with. 12288x1536, D2 / D8, in a CUDA graph (scripts/kernel_speed.py, 2026-09-18):
# 1 token 0.056 / 0.100 ms on tensor cores against 0.078 / 0.093 on CUDA cores, 32 tokens 0.101 / 0.137 against
# 0.333 / 0.402 - the bench decodes batches of up to 32 rows.
KERNEL = TENSOR_CORES


def kernel_reads_format(fmt: KFormat) -> bool:
    """Whether the bench's kernel has a variant for a base of `fmt`; a copy on another base is unpacked."""
    return fmt.name in KERNEL.names


def kquant_matmul(copy: KRefinedWeight, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """bf16 [..., out]: `x` bf16 [..., in] times the copy, each block of TILE_ROWS rows read to `depth`.

    `depth` is uint8 [out / TILE_ROWS] for one layout, or [tokens, out / TILE_ROWS] for a layout per token; 0 reads
    nothing. No host synchronization: the launch reads only shapes.
    """
    out_features = copy.shape[0]
    return kquant_matmul_fp32(copy, x.reshape(-1, x.shape[-1]), depth).to(x.dtype).reshape(*x.shape[:-1], out_features)


def kquant_matmul_fp32(copy: KRefinedWeight, tokens: torch.Tensor, depth: torch.Tensor,
                       kernel: KQuantKernel | None = None) -> torch.Tensor:
    """float32 [tokens, out]: the sums before the output is rounded to the input's type."""
    return (kernel or KERNEL)(copy, tokens.contiguous(), depth)


UNPACK_SOURCE = HERE / "kquant_mma.cu"  # the unpacking kernels build a weight as the mma kernels do, from one source
UNPACK_ROWS = 16  # rows of a thread block, as UNPACK_ROWS in the source
UNPACK_STEPS = QK_K // 16  # threads along a row of a super-block: one step of 16 weights each


def kquant_unpack(copy: KRefinedWeight, depth: torch.Tensor) -> torch.Tensor:
    """bf16 [out, in]: the copy read to `depth`, uint8 [out / TILE_ROWS], a block at depth 0 zero.

    Equals copy.dequantize(torch.bfloat16, d) on every block of depth d, bit for bit; one pass over the copy's bytes
    instead of a pass of torch kernels per depth. No host synchronization.
    """
    out_features, in_features = copy.shape
    weight = torch.empty(out_features, in_features, device=copy.blocks.device, dtype=torch.bfloat16)
    refinements = copy.refinements if copy.refinements.numel() else copy.blocks  # any valid pointer when none
    launch(load(f"kquant_unpack_{copy.fmt.name.lower()}", UNPACK_SOURCE),
           (-(-out_features // UNPACK_ROWS), in_features // QK_K, 1), (UNPACK_STEPS, UNPACK_ROWS, 1),
           copy.blocks.contiguous(), refinements.contiguous(), depth.contiguous(), weight.view(torch.int16),
           copy.refinements.shape[0], in_features, out_features)
    return weight
