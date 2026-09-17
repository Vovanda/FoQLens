"""A k-quant base after llama.cpp with residual slices over it: one copy read at its base and deeper.

The base is Q2_K or Q4_K as llama.cpp quantizes them without an importance matrix
(`quantize_row_q2_K_ref`, `quantize_row_q4_K_ref` and `make_qkx2_quants` in ggml/src/ggml-quants.c at
llama.cpp c6824a9): a super-block of QK_K weights split into blocks, every block an asymmetric grid - a
scale and a min searched by weighted least squares - and the scales and mins of a super-block quantized
themselves against one fp16 pair. Ported as vectorized tensor ops, block for block.

Over the base lie residual slices as in quant.SlicedWeight: slice k quantizes what the base and the slices
before it left, 2-bit symmetric codes with step block_step / 4**k, where block_step is the base's own
quantized step d * scale of that block - a slice stores codes only, no scale.

A base of Q2_K is one 2-bit plane, a base of Q4_K two; a read depth counts planes, and a copy cannot be
read shallower than its base.

Invariant: the base read back equals gguf-py's dequantizer on the same blocks laid out as GGUF bytes, exactly.
Invariant: the block search matches a loop port of llama.cpp's make_qkx2_quants: scales and mins within 1e-5
relative (float32 sums in another order), codes identical but for a near tie - measured 0 of 18,432 for Q2_K,
5 of 18,432 for Q4_K on heavy-tailed weights.
Invariant: a weight whose base error is within half its block step stays within block_step / 2 / 4**k after
k residual slices.
Invariant: a deeper plane never changes what the shallower planes read.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from foqlens.quant import N_SLICES, SLICE_BITS, _pack, _SliceReader, _unpack

QK_K = 256  # weights in a super-block, as in ggml
_FP16_BITS = 16
_RESIDUAL_LEVELS = 2**SLICE_BITS
_RESIDUAL_CENTER = _RESIDUAL_LEVELS // 2  # codes 0..3 read as -1.5 .. 1.5 steps, as in quant.SlicedWeight


@dataclass(frozen=True)
class KFormat:
    """One k-quant type: its code width, block size, scale width and the search llama.cpp runs for it."""

    name: str
    bits: int
    block: int
    scale_bits: int
    rmin: float
    rdelta: float
    nstep: int
    use_mad: bool
    rms_weighted: bool  # Q4_K weighs a weight by rms(block) + |x|, Q2_K by |x|

    @property
    def planes(self) -> int:
        return self.bits // SLICE_BITS

    @property
    def bits_per_weight(self) -> float:
        """Codes, the quantized scale and min of a block, the fp16 d and dmin of a super-block."""
        return self.bits + 2 * self.scale_bits / self.block + 2 * _FP16_BITS / QK_K


# The search parameters are the ones quantize_row_q2_K_ref / quantize_row_q4_K_ref pass to make_qkx2_quants.
Q2_K = KFormat("Q2_K", bits=2, block=16, scale_bits=4, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True, rms_weighted=False)
Q4_K = KFormat("Q4_K", bits=4, block=32, scale_bits=6, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, rms_weighted=True)


def make_qkx2_quants(x: torch.Tensor, weights: torch.Tensor, nmax: int, fmt: KFormat) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """llama.cpp's make_qkx2_quants over the last axis of `x`, every block at once.

    Returns (scale, min, codes): the block reads as scale * code - min; scale and min keep the last axis as 1.
    """
    lo = x.amin(-1, keepdim=True).clamp_max(0)
    hi = x.amax(-1, keepdim=True)
    span = hi - lo
    flat = span == 0
    safe_span = torch.where(flat, torch.ones_like(span), span)

    def error(scale: torch.Tensor, mn: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        diff = scale * codes + mn - x
        return (weights * (diff.abs() if fmt.use_mad else diff * diff)).sum(-1, keepdim=True)

    iscale = _divide(nmax, safe_span)
    codes = torch.round(iscale * (x - lo)).clamp_(0, nmax)
    scale, mn = 1 / iscale, lo
    best = error(scale, mn, codes)
    sum_w = weights.sum(-1, keepdim=True)
    sum_x = (weights * x).sum(-1, keepdim=True)
    for step in range(fmt.nstep + 1):
        # llama.cpp grids every trial from the best min found so far, not from the block's own min
        trial_span = hi - mn
        trial_iscale = _divide(_trial_numerator(fmt, step, nmax), torch.where(trial_span > 0, trial_span, torch.ones_like(trial_span)))
        trial = torch.round(trial_iscale * (x - mn)).clamp_(0, nmax)
        sum_l = (weights * trial).sum(-1, keepdim=True)
        sum_l2 = (weights * trial * trial).sum(-1, keepdim=True)
        sum_xl = (weights * trial * x).sum(-1, keepdim=True)
        det = sum_w * sum_l2 - sum_l * sum_l
        safe_det = torch.where(det > 0, det, torch.ones_like(det))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_det
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_det
        positive = this_min > 0
        this_min = torch.where(positive, torch.zeros_like(this_min), this_min)
        this_scale = torch.where(positive, sum_xl / torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2)), this_scale)
        trial_error = error(this_scale, this_min, trial)
        take = (det > 0) & (trial_error < best)
        best = torch.where(take, trial_error, best)
        codes = torch.where(take, trial, codes)
        scale = torch.where(take, this_scale, scale)
        mn = torch.where(take, this_min, mn)
    codes = torch.where(flat, torch.zeros_like(codes), codes)
    scale = torch.where(flat, torch.zeros_like(scale), scale)
    return scale, -mn, codes


def _divide(numerator: float, denominator: torch.Tensor) -> torch.Tensor:
    """a / b as C divides floats. torch turns `number / tensor` into number * reciprocal, an ulp away - and the
    largest weight of a block lands exactly on a half code in a trial, so an ulp picks its code."""
    return torch.full_like(denominator, numerator) / denominator


@functools.cache
def _trial_numerator(fmt: KFormat, step: int, nmax: int) -> float:
    """rmin + rdelta*is + nmax summed in float32, as the C expression is."""
    f = np.float32
    return float(f(fmt.rmin) + f(fmt.rdelta) * f(step) + f(nmax))


@dataclass
class KBase:
    """A quantized base: codes, per-block quantized scales and mins, per-super-block fp16 d and dmin."""

    fmt: KFormat
    codes: torch.Tensor  # uint8 [out, super-blocks, blocks, block]
    scales: torch.Tensor  # uint8 [out, super-blocks, blocks]
    mins: torch.Tensor  # uint8 [out, super-blocks, blocks]
    d: torch.Tensor  # fp16 [out, super-blocks]
    dmin: torch.Tensor  # fp16 [out, super-blocks]

    @classmethod
    def quantize(cls, weight: torch.Tensor, fmt: KFormat) -> KBase:
        x = _blocks(weight, fmt)
        nmax, top = 2**fmt.bits - 1, 2**fmt.scale_bits - 1
        if fmt.rms_weighted:
            weights = x.pow(2).mean(-1, keepdim=True).sqrt() + x.abs()
        else:
            weights = x.abs()
        scale, mn, codes = make_qkx2_quants(x, weights, nmax, fmt)
        scale, mn = scale.squeeze(-1), mn.squeeze(-1)
        # max_scale and max_min start at 0 in llama.cpp: a super-block of non-positive scales reads as zero
        max_scale = scale.amax(-1, keepdim=True).clamp_min(0)
        max_min = mn.amax(-1, keepdim=True).clamp_min(0)
        q_scale = _quantize_scale(scale, max_scale, top)
        q_min = _quantize_scale(mn, max_min, top)
        d = torch.where(max_scale > 0, max_scale / top, torch.zeros_like(max_scale)).squeeze(-1)
        dmin = torch.where(max_min > 0, max_min / top, torch.zeros_like(max_min)).squeeze(-1)
        d16, dmin16 = d.half(), dmin.half()
        # requantize against the scales as they will be read
        step = (d16.float()[..., None] * q_scale.float())[..., None]
        offset = (dmin16.float()[..., None] * q_min.float())[..., None]
        requantized = torch.round((x + offset) / torch.where(step != 0, step, torch.ones_like(step))).clamp_(0, nmax)
        codes = torch.where(step != 0, requantized, codes)
        return cls(fmt=fmt, codes=codes.to(torch.uint8), scales=q_scale, mins=q_min, d=d16, dmin=dmin16)

    def steps(self) -> torch.Tensor:
        """float32 [out, super-blocks, blocks, 1]: every block's read step d * scale."""
        return (self.d.float()[..., None] * self.scales.float())[..., None]

    def offsets(self) -> torch.Tensor:
        return (self.dmin.float()[..., None] * self.mins.float())[..., None]

    def dequantize(self) -> torch.Tensor:
        """float32 [out, super-blocks, blocks, block]."""
        return self.steps() * self.codes.float() - self.offsets()


def _quantize_scale(values: torch.Tensor, max_value: torch.Tensor, top: int) -> torch.Tensor:
    inv = torch.where(max_value > 0, _divide(top, torch.where(max_value > 0, max_value, torch.ones_like(max_value))), torch.zeros_like(max_value))
    return torch.round(inv * values).clamp_(0, top).to(torch.uint8)


def _blocks(weight: torch.Tensor, fmt: KFormat) -> torch.Tensor:
    out, inp = weight.shape
    assert inp % QK_K == 0, weight.shape
    return weight.float().view(out, inp // QK_K, QK_K // fmt.block, fmt.block)


@dataclass
class KSlicedWeight(_SliceReader):
    """A k-quant base and residual slices over it, up to N_SLICES planes in all.

    A _SliceReader, so the precision controller reads it by blocks at several depths the way it reads quant.SlicedWeight.
    """

    base: KBase
    residual: torch.Tensor  # uint8 [residual slices, out, in // 4] packed 2-bit codes
    shape: tuple[int, int]

    @classmethod
    def quantize(cls, weight: torch.Tensor, fmt: KFormat, planes: int = N_SLICES) -> KSlicedWeight:
        assert fmt.planes <= planes <= N_SLICES, (fmt, planes)
        base = KBase.quantize(weight, fmt)
        rest = _blocks(weight, fmt) - base.dequantize()
        step = base.steps() / _RESIDUAL_LEVELS
        live = step != 0
        safe_step = torch.where(live, step, torch.ones_like(step))
        codes = []
        for _ in range(planes - fmt.planes):
            code = torch.where(live, torch.floor(rest / safe_step + _RESIDUAL_CENTER).clamp_(0, _RESIDUAL_LEVELS - 1),
                               torch.full_like(rest, _RESIDUAL_CENTER))
            rest = rest - step * (code - _RESIDUAL_CENTER + 0.5)
            codes.append(_pack(code.to(torch.uint8).view(weight.shape)))
            step = step / _RESIDUAL_LEVELS
            safe_step = safe_step / _RESIDUAL_LEVELS
        residual = torch.stack(codes) if codes else torch.empty((0, *weight.shape[:-1], weight.shape[1] // 4), dtype=torch.uint8, device=weight.device)
        return cls(base=base, residual=residual, shape=tuple(weight.shape))

    @property
    def planes(self) -> int:
        return self.base.fmt.planes + self.residual.shape[0]

    def read_planes(self, depth: int) -> int:
        """The planes a read asking for `depth` planes gets: never fewer than the base, never more than stored."""
        return min(max(depth, self.base.fmt.planes), self.planes)

    def bits_per_weight(self, depth: int) -> float:
        return self.base.fmt.bits_per_weight + SLICE_BITS * (self.read_planes(depth) - self.base.fmt.planes)

    @property
    def nbytes(self) -> int:
        """Bytes of the stored codes, block scales and super-block pairs."""
        parts = (self.base.codes, self.base.scales, self.base.mins, self.base.d, self.base.dmin, self.residual)
        return sum(t.numel() * t.element_size() for t in parts)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """The float32 blocks read to each of the ascending `depths`: the base, then one residual slice at a time."""
        w = self.base.dequantize()
        step, done = self.base.steps() / _RESIDUAL_LEVELS, 0
        for depth in depths:
            for e in range(done, self.read_planes(depth) - self.base.fmt.planes):
                w += step * (_unpack(self.residual[e]).view(w.shape).float() - _RESIDUAL_CENTER + 0.5)
                step = step / _RESIDUAL_LEVELS
                done = e + 1
            yield w

    def _weight(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return w.reshape(self.shape).to(dtype)


# The classes whose weights the floor keeps on a Q4_K base: raised one at a time over a 2-bit floor none brings the
# knowledge back, together they do (E017, exploration-module-classes: EM 0.048 -> 0.413, 2% of the weights in k and
# the per-layer modules); unsloth's UD-Q2_K_XL raises the same classes.
SENSITIVE_CLASSES = ("self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj",
                     "per_layer_input_gate", "per_layer_projection")


@dataclass(frozen=True)
class KQuantLadder:
    """The read depths of one k-quant copy per module: the sensitive classes on a Q4_K base, the rest on Q2_K.

    A WeightSource for Controller.bake_from. A level reads its planes, never fewer than the base holds: D2 of a
    sensitive module reads its Q4_K base.

    Invariant: read(name, weight, level) equals KSlicedWeight.quantize(weight, format_for(name)) read to the level's
    planes.
    """

    sensitive: tuple[str, ...] = SENSITIVE_CLASSES

    def format_for(self, name: str) -> KFormat:
        return Q4_K if name.endswith(self.sensitive) else Q2_K

    def quantize(self, name: str, weight: torch.Tensor) -> KSlicedWeight:
        """The module's whole copy, every plane: a precision.SliceCopy, so the controller reads it by blocks."""
        return KSlicedWeight.quantize(weight, self.format_for(name))

    def read(self, name: str, weight: torch.Tensor, level) -> torch.Tensor:
        fmt = self.format_for(name)
        planes = max(level.slices, fmt.planes)
        return KSlicedWeight.quantize(weight, fmt, planes=planes).dequantize(weight.dtype, planes)
