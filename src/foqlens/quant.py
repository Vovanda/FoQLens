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


class _SliceReader:
    """Reading a sliced copy: the slices add up one at a time in fp32; a depth reads the sum so far.

    Subclasses give the empty accumulator, how slice e adds into it, and the scale.

    Invariant: linear_at_depths gives at every depth exactly what matmul gives at that depth - the
    same additions in the same order, shared instead of repeated.
    """

    scale: torch.Tensor

    def _accumulator(self) -> torch.Tensor:
        raise NotImplementedError

    def _add_slice(self, w: torch.Tensor, e: int) -> None:
        raise NotImplementedError

    def _weight(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (w * self.scale).view(w.shape[0], -1).to(dtype)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """The fp32 accumulator after each of the ascending `depths` - one tensor, added to in place, slice by slice."""
        w, done = self._accumulator(), 0
        for depth in depths:
            for e in range(done, depth):
                self._add_slice(w, e)
            done = depth
            yield w

    def dequantize(self, dtype: torch.dtype, depth: int = N_SLICES) -> torch.Tensor:
        """The weight read to the first `depth` slices (one fp32 temporary)."""
        return self._weight(next(self._sums([depth])), dtype)

    def matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None, depth: int = N_SLICES) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype, depth), bias)

    def linear_at_depths(self, x: torch.Tensor, bias: torch.Tensor | None, depths: list[int]) -> list[torch.Tensor]:
        """F.linear at every depth of the ascending `depths`, with the slices unpacked and added once."""
        return [F.linear(x, self._weight(w, x.dtype), bias) for w in self._sums(depths)]


def _slice_delta(codes: torch.Tensor, e: int) -> torch.Tensor:
    """What slice e adds, in steps of slice 1: (code - 1.5) / 4**e."""
    return (codes.float() - (_SLICE_ZERO - 0.5)) * float(_SLICE_LEVELS) ** -e


@dataclass
class SlicedWeight(_SliceReader):
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

    def _accumulator(self) -> torch.Tensor:
        out, inp = self.packed.shape[1], self.packed.shape[2] * _CODES_PER_BYTE
        return torch.zeros((out, inp // SLICE_GROUP, SLICE_GROUP), device=self.packed.device, dtype=torch.float32)

    def _add_slice(self, w: torch.Tensor, e: int) -> None:
        w += _slice_delta(_unpack(self.packed[e]).view(w.shape), e)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """As _SliceReader._sums, bit for bit, but every slice needed is unpacked and scaled in one go.

        One pass of a few large kernels over all slices replaces a handful of small ones per slice; the
        additions are the same, in the same order, so the sums are identical.
        """
        top = depths[-1]
        if top == 0:
            yield from super()._sums(depths)
            return
        shape = self._accumulator_shape()
        deltas = (_unpack(self.packed[:top]).view(top, *shape).float() - (_SLICE_ZERO - 0.5)) * _slice_steps(top, self.packed.device)
        w, done = deltas[0].clone(), 1  # 0 + d0 is d0 exactly (a delta is never -0.0)
        for depth in depths:
            for e in range(done, depth):
                w += deltas[e]
            done = max(done, depth)
            yield w

    def _accumulator_shape(self) -> tuple[int, int, int]:
        out, inp = self.packed.shape[1], self.packed.shape[2] * _CODES_PER_BYTE
        return out, inp // SLICE_GROUP, SLICE_GROUP

    @property
    def nbytes(self) -> int:
        """Bytes of the stored slices, without the scales."""
        return self.packed.numel() * self.packed.element_size()


@dataclass
class CappedSlicedWeight(_SliceReader):
    """A SlicedWeight whose every row keeps only its first `cap` slices: slice e is stored for the rows with cap > e.

    The memory then follows the layout of depths instead of holding every slice for every row. A row
    read deeper than its cap gets only the slices it keeps.

    Invariant: a row read no deeper than its cap reads exactly what the full SlicedWeight reads.
    """

    slices: tuple[torch.Tensor, ...]  # slice e: uint8 [rows kept, in // _CODES_PER_BYTE]
    blocks: tuple[torch.Tensor, ...]  # slice e: int32 indices of the blocks of rows it is kept for
    block_rows: int
    scale: torch.Tensor
    out_features: int

    @classmethod
    def from_sliced(cls, sliced: SlicedWeight, block_caps: torch.Tensor, block_rows: int) -> CappedSlicedWeight:
        """block_caps: [n_blocks] number of slices every block of block_rows rows keeps, 0 ... N_SLICES."""
        out = sliced.packed.shape[1]
        blocks = tuple(torch.nonzero(block_caps > e).squeeze(1).to(torch.int32) for e in range(sliced.packed.shape[0]))
        slices = tuple(sliced.packed[e, _rows_of(kept, block_rows, out)].contiguous() for e, kept in enumerate(blocks))
        return cls(slices=slices, blocks=blocks, block_rows=block_rows, scale=sliced.scale, out_features=out)

    @property
    def nbytes(self) -> int:
        """Bytes of the stored slices, without the scales."""
        return sum(s.numel() * s.element_size() for s in self.slices)

    def _accumulator(self) -> torch.Tensor:
        inp = self.slices[0].shape[1] * _CODES_PER_BYTE
        return torch.zeros((self.out_features, inp // SLICE_GROUP, SLICE_GROUP), device=self.scale.device, dtype=torch.float32)

    def _add_slice(self, w: torch.Tensor, e: int) -> None:
        codes = _unpack(self.slices[e]).view(-1, *w.shape[1:])
        w.index_add_(0, _rows_of(self.blocks[e], self.block_rows, self.out_features), _slice_delta(codes, e))


def _rows_of(blocks: torch.Tensor, block_rows: int, out_features: int) -> torch.Tensor:
    """Row indices of the given blocks, in order; the last block may be partial."""
    rows = (blocks.long()[:, None] * block_rows + torch.arange(block_rows, device=blocks.device)).reshape(-1)
    return rows[rows < out_features]


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., n] codes of SLICE_BITS bits -> [..., n / _CODES_PER_BYTE] bytes, first code in the low bits."""
    c = codes.view(*codes.shape[:-1], -1, _CODES_PER_BYTE)
    shifts = torch.arange(_CODES_PER_BYTE, device=codes.device, dtype=torch.uint8) * SLICE_BITS
    return (c << shifts).sum(dim=-1, dtype=torch.uint8)


@functools.cache
def _shifts(device: torch.device) -> torch.Tensor:
    """The bit shift of every code in a byte, made once per device instead of on every unpack."""
    return torch.arange(_CODES_PER_BYTE, device=device, dtype=torch.uint8) * SLICE_BITS


@functools.cache
def _slice_steps(n: int, device: torch.device) -> torch.Tensor:
    """[n, 1, 1, 1] fp32: the step of slice e in steps of slice 1, 4**-e - the factor _slice_delta uses."""
    return torch.tensor([float(_SLICE_LEVELS) ** -e for e in range(n)], device=device, dtype=torch.float32).view(n, 1, 1, 1)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    c = (packed.unsqueeze(-1) >> _shifts(packed.device)) & (_SLICE_LEVELS - 1)
    return c.view(*packed.shape[:-1], -1)
