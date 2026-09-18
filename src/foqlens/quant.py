"""Weight precision levels, quantization into them, and multiplication at each level.

Weights are stored packed and multiplied without keeping a dequantized copy: nf4 through the
fused bitsandbytes kernel, int8 through a per-call dequantized temporary of one module. This is
simulated quantization for quality measurements: the bench keeps the bf16 weight next to the
packed copies, so it saves no memory by itself (see docs/plan.md, step 5, for the real saving).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
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
    D2 ... D8 are read depths of one RefinedWeight: depth 1 ... 4, the base and 0 ... 3 refinements of 2 bits.
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
        return {Level.BF16: 16, Level.INT8: 8, Level.NF4: 4, Level.ZERO: 0}.get(self, self.depth * DEPTH_BITS)

    @property
    def depth(self) -> int:
        """The depth of the RefinedWeight this level reads; 0 for a level that is not a read depth."""
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


DEPTH_BITS = 2
MAX_DEPTH = 4
# A group never crosses a row boundary (as for nf4), so a block of rows owns its scales.
SCALE_GROUP = 64
_DEPTH_LEVELS = 2**DEPTH_BITS
_DEPTH_CENTER = _DEPTH_LEVELS // 2  # symmetric: codes 0..3 read as -1.5, -0.5, 0.5, 1.5 steps
_CODES_PER_BYTE = 8 // DEPTH_BITS


class _DepthReader:
    """Reading a refined copy: the base and the refinements add up one depth at a time in fp32; a depth reads the
    sum so far.

    Subclasses give the empty accumulator, how depth e adds into it, and the scale.

    Invariant: linear_at_depths gives at every depth exactly what matmul gives at that depth - the
    same additions in the same order, shared instead of repeated.
    """

    scale: torch.Tensor

    def _accumulator(self) -> torch.Tensor:
        raise NotImplementedError

    def _add_depth(self, w: torch.Tensor, e: int) -> None:
        raise NotImplementedError

    def _weight(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (w * self.scale).view(w.shape[0], -1).to(dtype)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """The fp32 accumulator after each of the ascending `depths` - one tensor, added to in place, depth by depth."""
        w, done = self._accumulator(), 0
        for depth in depths:
            for e in range(done, depth):
                self._add_depth(w, e)
            done = depth
            yield w

    def dequantize(self, dtype: torch.dtype, depth: int = MAX_DEPTH) -> torch.Tensor:
        """The weight read to `depth` (one fp32 temporary)."""
        return self._weight(next(self._sums([depth])), dtype)

    def matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None, depth: int = MAX_DEPTH) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype, depth), bias)

    def linear_at_depths(self, x: torch.Tensor, bias: torch.Tensor | None, depths: list[int]) -> list[torch.Tensor]:
        """F.linear at every depth of the ascending `depths`, with every depth unpacked and added once."""
        return [F.linear(x, self._weight(w, x.dtype), bias) for w in self._sums(depths)]


def _depth_delta(codes: torch.Tensor, e: int) -> torch.Tensor:
    """What depth e adds, in steps of the base: (code - 1.5) / 4**e."""
    return (codes.float() - (_DEPTH_CENTER - 0.5)) * float(_DEPTH_LEVELS) ** -e


@dataclass
class RefinedWeight(_DepthReader):
    """Recursive residual quantization, after MoBiQuant (arXiv 2602.20191).

    The base quantizes the weight to DEPTH_BITS bits; every refinement quantizes what the depths before it left,
    with a step 2**DEPTH_BITS times finer. Reading to depth k gives a k * DEPTH_BITS-bit weight, so one stored
    copy serves every depth: 2 / 4 / 6 / 8 bits at depth 1 ... 4.

    Invariant: at depth k the error is at most half the step of depth k, s1 / 2 / 4**(k-1).
    Invariant: depth k+1 never changes what depth k reads.
    """

    packed: torch.Tensor  # uint8 [MAX_DEPTH, out, in // _CODES_PER_BYTE]: the base, then the refinements
    scale: torch.Tensor  # float32 [out, in // SCALE_GROUP, 1] - the step of the base

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> RefinedWeight:
        out, inp = weight.shape
        assert inp % SCALE_GROUP == 0, weight.shape
        residual = weight.float().view(out, inp // SCALE_GROUP, SCALE_GROUP)
        scale = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / _DEPTH_CENTER
        step, codes = scale, []
        for _ in range(MAX_DEPTH):
            code = torch.floor(residual / step + _DEPTH_CENTER).clamp_(0, _DEPTH_LEVELS - 1)
            residual = residual - step * (code - _DEPTH_CENTER + 0.5)
            codes.append(code.to(torch.uint8).view(out, inp))
            step = step / _DEPTH_LEVELS
        return cls(packed=_pack(torch.stack(codes)), scale=scale)

    def _accumulator(self) -> torch.Tensor:
        out, inp = self.packed.shape[1], self.packed.shape[2] * _CODES_PER_BYTE
        return torch.zeros((out, inp // SCALE_GROUP, SCALE_GROUP), device=self.packed.device, dtype=torch.float32)

    def _add_depth(self, w: torch.Tensor, e: int) -> None:
        w += _depth_delta(_unpack(self.packed[e]).view(w.shape), e)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """As _DepthReader._sums, bit for bit, but every depth needed is unpacked and scaled in one go.

        One pass of a few large kernels over all depths replaces a handful of small ones per depth; the
        additions are the same, in the same order, so the sums are identical.
        """
        top = depths[-1]
        if top == 0:
            yield from super()._sums(depths)
            return
        shape = self._accumulator_shape()
        deltas = (_unpack(self.packed[:top]).view(top, *shape).float() - (_DEPTH_CENTER - 0.5)) * _depth_steps(top, self.packed.device)
        w, done = deltas[0].clone(), 1  # 0 + d0 is d0 exactly (a delta is never -0.0)
        for depth in depths:
            for e in range(done, depth):
                w += deltas[e]
            done = max(done, depth)
            yield w

    def _accumulator_shape(self) -> tuple[int, int, int]:
        out, inp = self.packed.shape[1], self.packed.shape[2] * _CODES_PER_BYTE
        return out, inp // SCALE_GROUP, SCALE_GROUP

    @property
    def nbytes(self) -> int:
        """Bytes of the stored depths, without the scales."""
        return self.packed.numel() * self.packed.element_size()


@dataclass
class CappedRefinedWeight(_DepthReader):
    """A RefinedWeight whose every row keeps only its depths up to `cap`: depth e is stored for the rows with cap > e.

    The memory then follows the layout of depths instead of holding every depth for every row. A row
    read deeper than its cap gets only the depths it keeps.

    Invariant: a row read no deeper than its cap reads exactly what the full RefinedWeight reads.
    """

    stored: tuple[torch.Tensor, ...]  # depth e: uint8 [rows kept, in // _CODES_PER_BYTE]
    blocks: tuple[torch.Tensor, ...]  # depth e: int32 indices of the blocks of rows it is kept for
    block_rows: int
    scale: torch.Tensor
    out_features: int

    @classmethod
    def from_full(cls, full: RefinedWeight, block_caps: torch.Tensor, block_rows: int) -> CappedRefinedWeight:
        """block_caps: [n_blocks] the depth every block of block_rows rows keeps, 0 ... MAX_DEPTH."""
        out = full.packed.shape[1]
        blocks = tuple(torch.nonzero(block_caps > e).squeeze(1).to(torch.int32) for e in range(full.packed.shape[0]))
        stored = tuple(full.packed[e, _rows_of(kept, block_rows, out)].contiguous() for e, kept in enumerate(blocks))
        return cls(stored=stored, blocks=blocks, block_rows=block_rows, scale=full.scale, out_features=out)

    @property
    def nbytes(self) -> int:
        """Bytes of the stored depths, without the scales."""
        return sum(s.numel() * s.element_size() for s in self.stored)

    def _accumulator(self) -> torch.Tensor:
        inp = self.stored[0].shape[1] * _CODES_PER_BYTE
        return torch.zeros((self.out_features, inp // SCALE_GROUP, SCALE_GROUP), device=self.scale.device, dtype=torch.float32)

    def _add_depth(self, w: torch.Tensor, e: int) -> None:
        codes = _unpack(self.stored[e]).view(-1, *w.shape[1:])
        w.index_add_(0, _rows_of(self.blocks[e], self.block_rows, self.out_features), _depth_delta(codes, e))


def _rows_of(blocks: torch.Tensor, block_rows: int, out_features: int) -> torch.Tensor:
    """Row indices of the given blocks, in order; the last block may be partial."""
    rows = (blocks.long()[:, None] * block_rows + torch.arange(block_rows, device=blocks.device)).reshape(-1)
    return rows[rows < out_features]


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., n] codes of DEPTH_BITS bits -> [..., n / _CODES_PER_BYTE] bytes, first code in the low bits."""
    c = codes.view(*codes.shape[:-1], -1, _CODES_PER_BYTE)
    shifts = torch.arange(_CODES_PER_BYTE, device=codes.device, dtype=torch.uint8) * DEPTH_BITS
    return (c << shifts).sum(dim=-1, dtype=torch.uint8)


@functools.cache
def _shifts(device: torch.device) -> torch.Tensor:
    """The bit shift of every code in a byte, made once per device instead of on every unpack."""
    return torch.arange(_CODES_PER_BYTE, device=device, dtype=torch.uint8) * DEPTH_BITS


@functools.cache
def _depth_steps(n: int, device: torch.device) -> torch.Tensor:
    """[n, 1, 1, 1] fp32: the step of depth e in steps of the base, 4**-e - the factor _depth_delta uses."""
    return torch.tensor([float(_DEPTH_LEVELS) ** -e for e in range(n)], device=device, dtype=torch.float32).view(n, 1, 1, 1)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    c = (packed.unsqueeze(-1) >> _shifts(packed.device)) & (_DEPTH_LEVELS - 1)
    return c.view(*packed.shape[:-1], -1)
