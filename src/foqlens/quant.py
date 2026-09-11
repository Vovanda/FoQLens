"""Weight precision levels and quantization into them.

Weights are stored packed and computed through dequantization to bf16. This is simulated
quantization: the outputs are the same numbers a weight read at that precision would give,
but memory and speed are not saved - enough for quality measurements (see docs/plan.md,
"What to honestly expect").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import bitsandbytes.functional as bnbf
import torch


class Level(IntEnum):
    """Precision level of a block. The value is its code in the module's level buffer."""

    BF16 = 0
    INT8 = 1
    NF4 = 2

    @property
    def bits(self) -> int:
        return {Level.BF16: 16, Level.INT8: 8, Level.NF4: 4}[self]


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


@dataclass
class Nf4Weight:
    """bitsandbytes NF4, absmax per group of NF4_BLOCKSIZE consecutive weights of a row."""

    packed: torch.Tensor
    state: bnbf.QuantState

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> Nf4Weight:
        # A group must not cross a row boundary, otherwise a block of rows
        # cannot be cut out of a matrix quantized as a whole.
        assert weight.shape[1] % NF4_BLOCKSIZE == 0, weight.shape
        packed, state = bnbf.quantize_4bit(weight.contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4")
        return cls(packed=packed, state=state)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return bnbf.dequantize_4bit(self.packed, self.state).to(dtype)
