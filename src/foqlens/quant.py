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
    D2 ... D8 are read depths of one SlicedWeight: the first 1 ... 4 slices of 2 bits.
    """

    BF16 = 0
    INT8 = 1
    NF4 = 2
    ZERO = 3
    D2 = 4
    D4 = 5
    D6 = 6
    D8 = 7

    @property
    def bits(self) -> int:
        return {Level.BF16: 16, Level.INT8: 8, Level.NF4: 4, Level.ZERO: 0}.get(self, self.slices * SLICE_BITS)

    @property
    def slices(self) -> int:
        """Slices of the SlicedWeight this level reads; 0 for a level that is not a read depth."""
        return self - Level.ZERO if self > Level.ZERO else 0


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


SLICE_BITS = 2
N_SLICES = 4
# A group never crosses a row boundary (as for nf4), so a block of rows owns its scales.
SLICE_GROUP = 64
_SLICE_LEVELS = 2**SLICE_BITS
_SLICE_ZERO = _SLICE_LEVELS // 2  # symmetric: codes 0..3 read as -1.5, -0.5, 0.5, 1.5 steps
_CODES_PER_BYTE = 8 // SLICE_BITS


@dataclass
class SlicedWeight:
    """Recursive residual quantization, after MoBiQuant (arXiv 2602.20191).

    Slice 1 quantizes the weight to SLICE_BITS bits; slice e quantizes what slices 1..e-1 left,
    with a step 2**SLICE_BITS times finer. Reading the first k slices gives a k * SLICE_BITS-bit
    weight, so one stored copy serves every depth: 2 / 4 / 6 / 8 bits for 4 slices.

    Invariant: after k slices the error is at most half the step of slice k, s1 / 2 / 4**(k-1).
    Invariant: slice k+1 never changes what the first k slices read.
    """

    packed: torch.Tensor  # uint8 [N_SLICES, out, in // _CODES_PER_BYTE]
    scale: torch.Tensor  # float32 [out, in // SLICE_GROUP, 1] - the step of slice 1

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> SlicedWeight:
        out, inp = weight.shape
        assert inp % SLICE_GROUP == 0, weight.shape
        residual = weight.float().view(out, inp // SLICE_GROUP, SLICE_GROUP)
        scale = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / _SLICE_ZERO
        step, codes = scale, []
        for _ in range(N_SLICES):
            code = torch.floor(residual / step + _SLICE_ZERO).clamp_(0, _SLICE_LEVELS - 1)
            residual = residual - step * (code - _SLICE_ZERO + 0.5)
            codes.append(code.to(torch.uint8).view(out, inp))
            step = step / _SLICE_LEVELS
        return cls(packed=_pack(torch.stack(codes)), scale=scale)

    def dequantize(self, dtype: torch.dtype, depth: int = N_SLICES) -> torch.Tensor:
        """The weight read to the first `depth` slices, accumulated one slice at a time (one fp32 temporary)."""
        out, inp = self.packed.shape[1], self.packed.shape[2] * _CODES_PER_BYTE
        groups = (out, inp // SLICE_GROUP, SLICE_GROUP)
        w = torch.zeros(groups, device=self.packed.device, dtype=torch.float32)
        for e in range(depth):
            code = _unpack(self.packed[e]).view(groups)
            w += (code.float() - (_SLICE_ZERO - 0.5)) * float(_SLICE_LEVELS) ** -e
        return (w * self.scale).view(out, inp).to(dtype)

    def matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None, depth: int = N_SLICES) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype, depth), bias)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., n] codes of SLICE_BITS bits -> [..., n / _CODES_PER_BYTE] bytes, first code in the low bits."""
    c = codes.view(*codes.shape[:-1], -1, _CODES_PER_BYTE)
    shifts = torch.arange(_CODES_PER_BYTE, device=codes.device, dtype=torch.uint8) * SLICE_BITS
    return (c << shifts).sum(dim=-1, dtype=torch.uint8)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(_CODES_PER_BYTE, device=packed.device, dtype=torch.uint8) * SLICE_BITS
    c = (packed.unsqueeze(-1) >> shifts) & (_SLICE_LEVELS - 1)
    return c.view(*packed.shape[:-1], -1)
