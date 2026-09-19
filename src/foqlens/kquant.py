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
Invariant: a base laid out as GGUF blocks (gguf_blocks) is ggml's block_q2_K / block_q4_K byte for byte, and
from_gguf_blocks reads it back to the same base exactly.

Q3_K and Q6_K are only read: a published GGUF file's blocks of them become a base (refinements.PublishedBaseLadder); no
quantizer of them is ported.
Invariant: from_gguf_blocks reads block_q3_K / block_q6_K bytes exactly as gguf-py dequantizes them - every
controlled tensor of bartowski's E2B-it Q2_K file, 2026-09-18.
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
    """One k-quant type: its code width, block size, scale width and, where it is ported, the search llama.cpp runs
    for it.

    An asymmetric type (Q2_K, Q4_K) reads a block as d*scale*code - dmin*min; a symmetric one (Q3_K, Q6_K) as
    d*scale*(code - zero_point), with a signed scale and no min.
    """

    name: str
    bits: int
    block: int
    scale_bits: int
    rmin: float | None = None  # None: no quantizer ported, the base is only read from a GGUF file
    rdelta: float | None = None
    nstep: int | None = None
    use_mad: bool = False
    rms_weighted: bool = False  # Q4_K weighs a weight by rms(block) + |x|, Q2_K by |x|
    zero_point: int = 0  # symmetric types: the code that reads as zero; 0 for an asymmetric type

    @property
    def base_depth(self) -> int:
        return self.bits // DEPTH_BITS

    @property
    def symmetric(self) -> bool:
        return self.zero_point > 0

    @property
    def bits_per_weight(self) -> float:
        """Codes, the quantized scale (and min) of a block, the fp16 d (and dmin) of a super-block."""
        pairs = 1 if self.symmetric else 2
        return self.bits + pairs * (self.scale_bits / self.block + _FP16_BITS / QK_K)


# The search parameters are the ones quantize_row_q2_K_ref / quantize_row_q4_K_ref pass to make_qkx2_quants.
Q2_K = KFormat("Q2_K", bits=2, block=16, scale_bits=4, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True, rms_weighted=False)
Q4_K = KFormat("Q4_K", bits=4, block=32, scale_bits=6, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, rms_weighted=True)
# Read from published GGUF files as a base (dequantize_row_q3_K / _q6_K in ggml-quants.c); no quantizer ported.
Q3_K = KFormat("Q3_K", bits=3, block=16, scale_bits=6, zero_point=4)
Q6_K = KFormat("Q6_K", bits=6, block=16, scale_bits=8, zero_point=32)
FORMATS = {fmt.name: fmt for fmt in (Q2_K, Q3_K, Q4_K, Q6_K)}


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
    scales: torch.Tensor  # uint8 [out, super-blocks, blocks]; int8 for a symmetric type, its scale signed
    mins: torch.Tensor  # uint8 [out, super-blocks, blocks]; zeros for a symmetric type
    d: torch.Tensor  # fp16 [out, super-blocks]
    dmin: torch.Tensor  # fp16 [out, super-blocks]; zeros for a symmetric type

    @classmethod
    def quantize(cls, weight: torch.Tensor, fmt: KFormat) -> KBase:
        if fmt.rmin is None:
            raise ValueError(f"no quantizer ported for {fmt.name}: a base of it is read from a GGUF file")
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
        if self.fmt.symmetric:
            return self.steps() * self.fmt.zero_point
        return (self.dmin.float()[..., None] * self.mins.float())[..., None]

    def dequantize(self) -> torch.Tensor:
        """float32 [out, super-blocks, blocks, block]."""
        return self.steps() * self.codes.float() - self.offsets()


# ==== GGUF block layout ====
# The base is stored as ggml stores it: the codes, block scales and super-block pair with no bit to spare
# (KFormat.bits_per_weight), checked byte for byte against gguf-py's dequantizer.

_FP16_BYTES = 2
_Q4_K_SPLIT = 4  # block_q4_K keeps the low 6 bits of scales and mins 0..3 whole and splits those of 4..7
_LOW6, _LOW4 = 0x3F, 0x0F
_Q3_K_SCALE_BIAS = 32  # block_q3_K stores a signed 6-bit scale as scale + 32


def gguf_blocks(base: KBase) -> torch.Tensor:
    """uint8 [out, super-blocks, bytes]: every super-block of the base as ggml's block_q2_K or block_q4_K."""
    out, n_super = base.d.shape
    codes = base.codes.reshape(out * n_super, QK_K)
    scales = base.scales.reshape(out * n_super, -1)
    mins = base.mins.reshape(out * n_super, -1)
    d = base.d.reshape(-1, 1).view(torch.uint8)
    dmin = base.dmin.reshape(-1, 1).view(torch.uint8)
    if base.fmt is Q2_K:
        # qs[32 * half + i] holds the codes 128 * half + 32 * g + i of the four groups g, group g in bits 2g
        groups = codes.view(-1, 2, 4, 32)
        qs = (groups[:, :, 0] | groups[:, :, 1] << 2 | groups[:, :, 2] << 4 | groups[:, :, 3] << 6).reshape(-1, QK_K // 4)
        blocks = torch.cat([scales | mins << 4, qs, d, dmin], dim=1)
    elif base.fmt is Q4_K:
        # qs[32 * quarter + i] holds codes 64 * quarter + i (low nibble) and 64 * quarter + 32 + i (high)
        halves = codes.view(-1, 4, 2, 32)
        qs = (halves[:, :, 0] | halves[:, :, 1] << 4).reshape(-1, QK_K // 2)
        low, high = slice(0, _Q4_K_SPLIT), slice(_Q4_K_SPLIT, None)
        packed = torch.cat([scales[:, low] | (scales[:, high] >> 4) << 6,
                            mins[:, low] | (mins[:, high] >> 4) << 6,
                            (scales[:, high] & _LOW4) | (mins[:, high] & _LOW4) << 4], dim=1)
        blocks = torch.cat([d, dmin, packed, qs], dim=1)
    else:
        raise ValueError(f"no GGUF layout for {base.fmt.name}")
    return blocks.view(out, n_super, -1)


def from_gguf_blocks(blocks: torch.Tensor, fmt: KFormat) -> KBase:
    """The base gguf_blocks laid out: uint8 [out, super-blocks, bytes] in, the codes and scales as KBase holds them out."""
    out, n_super, _ = blocks.shape
    flat = blocks.reshape(out * n_super, -1)
    n_blocks = QK_K // fmt.block
    if fmt is Q2_K:
        packed, qs, tail = flat[:, :n_blocks], flat[:, n_blocks:n_blocks + QK_K // 4], flat[:, n_blocks + QK_K // 4:]
        scales, mins = packed & _LOW4, packed >> 4
        shifts = torch.arange(0, 8, 2, device=flat.device, dtype=torch.uint8)
        codes = ((qs.view(-1, 2, 1, 32) >> shifts.view(1, 1, 4, 1)) & 3).reshape(-1, QK_K)
        d, dmin = tail[:, :_FP16_BYTES], tail[:, _FP16_BYTES:]
    elif fmt is Q4_K:
        d, dmin = flat[:, :_FP16_BYTES], flat[:, _FP16_BYTES:2 * _FP16_BYTES]
        packed, qs = flat[:, 2 * _FP16_BYTES:2 * _FP16_BYTES + 12], flat[:, 2 * _FP16_BYTES + 12:]
        first, second, third = packed[:, :4], packed[:, 4:8], packed[:, 8:]
        scales = torch.cat([first & _LOW6, (third & _LOW4) | (first >> 6) << 4], dim=1)
        mins = torch.cat([second & _LOW6, (third >> 4) | (second >> 6) << 4], dim=1)
        quarters = qs.view(-1, 4, 1, 32)
        codes = torch.cat([quarters & _LOW4, quarters >> 4], dim=2).reshape(-1, QK_K)
    elif fmt is Q3_K:
        # hmask[32], qs[64], scales[12], d: weight 128h + 32g + j reads bits 2g of qs[32h + j] and, as its high bit,
        # bit (4h + g) of hmask[j]; scale k is the low nibble of byte k mod 8 (shifted by 4 for k >= 8) with bits
        # 2 * (k div 4) of byte 8 + k mod 4 above it, less 32
        hmask, qs = flat[:, :QK_K // 8], flat[:, QK_K // 8:QK_K // 8 + QK_K // 4]
        packed, d = flat[:, 3 * QK_K // 8:3 * QK_K // 8 + 12], flat[:, 3 * QK_K // 8 + 12:]
        low = (qs.view(-1, 2, 1, 32) >> torch.arange(0, 8, 2, device=flat.device, dtype=torch.uint8).view(1, 1, 4, 1)) & 3
        high = (hmask.view(-1, 1, 32) >> torch.arange(8, device=flat.device, dtype=torch.uint8).view(1, 8, 1)) & 1
        codes = (low.reshape(-1, QK_K) | high.reshape(-1, QK_K) << 2)
        k = torch.arange(n_blocks, device=flat.device)
        nibbles = (packed[:, k % 8] >> (4 * (k // 8)).to(torch.uint8)) & _LOW4
        tops = (packed[:, 8 + k % 4] >> (2 * (k // 4)).to(torch.uint8)) & 3
        scales = (nibbles | tops << 4).to(torch.int8) - _Q3_K_SCALE_BIAS
        mins = torch.zeros_like(nibbles)
        dmin = torch.zeros_like(d)
    elif fmt is Q6_K:
        # ql[128], qh[64], scales[16] int8, d: weight 128h + 32(2p + c) + j reads nibble p of ql[64h + 32c + j] and,
        # as its high two bits, bits 2(2p + c) of qh[32h + j]
        ql, qh = flat[:, :QK_K // 2], flat[:, QK_K // 2:QK_K // 2 + QK_K // 4]
        packed, d = flat[:, 3 * QK_K // 4:3 * QK_K // 4 + n_blocks], flat[:, 3 * QK_K // 4 + n_blocks:]
        # the shifts 0, 4 are made on the device: a tensor from a host list is a copy a CUDA graph cannot capture
        low = (ql.view(-1, 2, 1, 64) >> torch.arange(0, 8, 4, device=flat.device, dtype=torch.uint8).view(1, 1, 2, 1)) & _LOW4
        high = (qh.view(-1, 2, 1, 32) >> torch.arange(0, 8, 2, device=flat.device, dtype=torch.uint8).view(1, 1, 4, 1)) & 3
        codes = low.reshape(-1, QK_K) | high.reshape(-1, QK_K) << 4
        scales = packed.contiguous().view(torch.int8)
        mins = torch.zeros(scales.shape, dtype=torch.uint8, device=flat.device)
        dmin = torch.zeros_like(d)
    else:
        raise ValueError(f"no GGUF layout for {fmt.name}")
    return KBase(fmt=fmt,
                 codes=codes.reshape(out, n_super, n_blocks, fmt.block).contiguous(),
                 scales=scales.reshape(out, n_super, n_blocks).contiguous(),
                 mins=mins.reshape(out, n_super, n_blocks).contiguous(),
                 d=d.contiguous().view(torch.float16).reshape(out, n_super),
                 dmin=dmin.contiguous().view(torch.float16).reshape(out, n_super))


def _quantize_scale(values: torch.Tensor, max_value: torch.Tensor, top: int) -> torch.Tensor:
    inv = torch.where(max_value > 0, _divide(top, torch.where(max_value > 0, max_value, torch.ones_like(max_value))), torch.zeros_like(max_value))
    return torch.round(inv * values).clamp_(0, top).to(torch.uint8)


def _blocks(weight: torch.Tensor, fmt: KFormat) -> torch.Tensor:
    out, inp = weight.shape
    assert inp % QK_K == 0, weight.shape
    return weight.float().view(out, inp // QK_K, QK_K // fmt.block, fmt.block)
