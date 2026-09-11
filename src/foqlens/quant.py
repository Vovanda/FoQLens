"""Weight precision levels, quantization into them, and multiplication at each level.

Weights are stored packed and multiplied without keeping a dequantized copy: nf4 through the
fused bitsandbytes kernel, int8 through a per-call dequantized temporary of one module. This is
simulated quantization for quality measurements: the bench keeps the bf16 weight next to the
packed copies, so it saves no memory by itself (see docs/plan.md, step 5, for the real saving).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import bitsandbytes as bnb
import bitsandbytes.functional as bnbf
import torch
import torch.nn.functional as F


class Level(IntEnum):
    """Precision level of a block. The value is its code in the module's level array.

    ZERO is the limit of precision: the block is not read at all and its output is zero. It gives
    the step 3 sweep polar regions - kept blocks against removed ones - instead of bf16 against nf4.
    """

    BF16 = 0
    INT8 = 1
    NF4 = 2
    ZERO = 3

    @property
    def bits(self) -> int:
        return {Level.BF16: 16, Level.INT8: 8, Level.NF4: 4, Level.ZERO: 0}[self]


NF4_BLOCKSIZE = 64


@dataclass
class Int8Weight:
    """Row-wise absmax int8: the scheme of bitsandbytes int8_vectorwise_quant, without outliers."""

    q: torch.Tensor  # int8 [out, in]
    scale: torch.Tensor  # float32 [out, 1]

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> Int8Weight:
        w = weight.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
        return cls(q=q, scale=scale)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.q.float() * self.scale).to(dtype)

    def matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype), bias)


@dataclass
class Nf4Weight:
    """bitsandbytes NF4, absmax per group of NF4_BLOCKSIZE consecutive weights of a row."""

    packed: torch.Tensor
    state: bnbf.QuantState

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> Nf4Weight:
        # A group must not cross a row boundary, otherwise a block of rows
        # would not be the same weights as in the matrix quantized as a whole.
        assert weight.shape[1] % NF4_BLOCKSIZE == 0, weight.shape
        packed, state = bnbf.quantize_4bit(weight.contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4")
        return cls(packed=packed, state=state)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return bnbf.dequantize_4bit(self.packed, self.state).to(dtype)

    def matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Fused 4-bit GEMM straight from the packed weights: no dequantized copy is kept."""
        return bnb.matmul_4bit(x, self.packed, self.state, bias=bias)
