"""k-quants after llama.cpp: Q2_K and Q4_K, the base of the bench's copy (refinements.py).

Q2_K and Q4_K as llama.cpp quantizes them without an importance matrix (`quantize_row_q2_K_ref`,
`quantize_row_q4_K_ref` and `make_qkx2_quants` in ggml/src/ggml-quants.c at llama.cpp c6824a9): a super-block
of QK_K weights split into blocks, every block an asymmetric grid - a scale and a min searched by weighted least
squares - and the scales and mins of a super-block quantized themselves against one fp16 pair. Ported as
vectorized tensor ops, block for block.

Invariant: the base read back equals gguf-py's dequantizer on the same blocks laid out as GGUF bytes, exactly.
Invariant: the block search matches a loop port of llama.cpp's make_qkx2_quants: scales and mins within 1e-5
relative (float32 sums in another order), codes identical but for a near tie - measured 0 of 18,432 for Q2_K,
5 of 18,432 for Q4_K on heavy-tailed weights.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from foqlens.quant import DEPTH_BITS

QK_K = 256  # weights in a super-block, as in ggml
_FP16_BITS = 16


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
    def base_depth(self) -> int:
        return self.bits // DEPTH_BITS

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
