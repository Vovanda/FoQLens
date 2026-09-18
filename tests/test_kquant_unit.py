"""The k-quant base on synthetic weights: no model, seconds."""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from foqlens.kquant import Q2_K, Q4_K, QK_K, KBase, KFormat, make_qkx2_quants

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# a heavy tail, as trained weights have, so that the search clips and the scales differ between blocks
ROWS, SUPER_BLOCKS = 6, 3


def make_weight(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.distributions.StudentT(3.0).sample((ROWS, SUPER_BLOCKS * QK_K)) * 0.02).to(DEVICE, torch.bfloat16)


def gguf_bytes_q2_k(base: KBase) -> np.ndarray:
    """block_q2_K as ggml lays it out: scales[16] (scale | min << 4), qs[64], d, dmin."""
    codes = base.codes.cpu().numpy().reshape(-1, QK_K)
    scales = (base.scales | (base.mins << 4)).cpu().numpy().reshape(-1, QK_K // 16)
    qs = np.zeros((codes.shape[0], QK_K // 4), dtype=np.uint8)
    for j in range(0, QK_K, 128):
        for i in range(32):
            qs[:, j // 4 + i] = codes[:, j + i] | codes[:, j + i + 32] << 2 | codes[:, j + i + 64] << 4 | codes[:, j + i + 96] << 6
    d = base.d.cpu().numpy().reshape(-1, 1).view(np.uint8)
    dmin = base.dmin.cpu().numpy().reshape(-1, 1).view(np.uint8)
    return np.concatenate([scales, qs, d, dmin], axis=1)


def gguf_bytes_q4_k(base: KBase) -> np.ndarray:
    """block_q4_K as ggml lays it out: d, dmin, scales[12] (6-bit scales and mins), qs[128]."""
    codes = base.codes.cpu().numpy().reshape(-1, QK_K)
    ls = base.scales.cpu().numpy().reshape(-1, 8)
    lm = base.mins.cpu().numpy().reshape(-1, 8)
    packed = np.zeros((codes.shape[0], 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            packed[:, j] = ls[:, j]
            packed[:, j + 4] = lm[:, j]
        else:
            packed[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
            packed[:, j - 4] |= (ls[:, j] >> 4) << 6
            packed[:, j] |= (lm[:, j] >> 4) << 6
    qs = np.zeros((codes.shape[0], QK_K // 2), dtype=np.uint8)
    for j in range(0, QK_K, 64):
        qs[:, j // 2: j // 2 + 32] = codes[:, j: j + 32] | codes[:, j + 32: j + 64] << 4
    d = base.d.cpu().numpy().reshape(-1, 1).view(np.uint8)
    dmin = base.dmin.cpu().numpy().reshape(-1, 1).view(np.uint8)
    return np.concatenate([d, dmin, packed, qs], axis=1)


@pytest.mark.parametrize(("fmt", "layout", "qtype"), [(Q2_K, gguf_bytes_q2_k, GGMLQuantizationType.Q2_K),
                                                      (Q4_K, gguf_bytes_q4_k, GGMLQuantizationType.Q4_K)])
def test_the_base_reads_exactly_as_gguf_dequantizes_its_bytes(fmt, layout, qtype):
    base = KBase.quantize(make_weight(), fmt)
    ours = base.dequantize().cpu().numpy().reshape(-1, QK_K)
    theirs = dequantize(layout(base).reshape(-1), qtype).reshape(-1, QK_K)
    np.testing.assert_array_equal(ours, theirs)


def reference_make_qkx2_quants(x: np.ndarray, weights: np.ndarray, nmax: int, fmt: KFormat):
    """ggml-quants.c make_qkx2_quants, line by line in float32."""
    f = np.float32
    lo, hi = f(x.min()), f(x.max())
    lo = min(lo, f(0))
    if hi == lo:
        return f(0), -lo, np.zeros(len(x), dtype=np.uint8)
    iscale = f(nmax) / (hi - lo)
    scale = f(1) / iscale
    codes = np.clip(np.rint(iscale * (x - lo)), 0, nmax)
    best = sum(weights[i] * (abs(scale * codes[i] + lo - x[i]) if fmt.use_mad else (scale * codes[i] + lo - x[i]) ** 2) for i in range(len(x)))
    mn, sum_w, sum_x = lo, f(weights.sum()), f((weights * x).sum())
    for step in range(fmt.nstep + 1):
        trial_iscale = (f(fmt.rmin) + f(fmt.rdelta) * f(step) + f(nmax)) / (hi - mn)
        trial = np.clip(np.rint(trial_iscale * (x - mn)), 0, nmax).astype(np.float32)
        sum_l, sum_l2, sum_xl = f((weights * trial).sum()), f((weights * trial * trial).sum()), f((weights * trial * x).sum())
        det = sum_w * sum_l2 - sum_l * sum_l
        if det > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / det
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det
            if this_min > 0:
                this_min, this_scale = f(0), sum_xl / sum_l2
            diff = this_scale * trial + this_min - x
            err = f((weights * (np.abs(diff) if fmt.use_mad else diff * diff)).sum())
            if err < best:
                codes, best, scale, mn = trial, err, this_scale, this_min
    return scale, -mn, codes


@pytest.mark.parametrize("fmt", [Q2_K, Q4_K])
def test_the_block_search_matches_a_loop_port_of_llama_cpp(fmt):
    weight = make_weight(seed=1)
    nmax = 2**fmt.bits - 1
    xt = weight.float().view(ROWS, SUPER_BLOCKS, QK_K // fmt.block, fmt.block)
    wt = xt.pow(2).mean(-1, keepdim=True).sqrt() + xt.abs() if fmt.rms_weighted else xt.abs()
    scale, mn, codes = make_qkx2_quants(xt, wt, nmax, fmt)
    scale, mn, codes = (t.cpu().numpy().reshape(-1, t.shape[-1]) for t in (scale, mn, codes))
    x, w = xt.cpu().numpy().reshape(-1, fmt.block), wt.cpu().numpy().reshape(-1, fmt.block)
    flipped = 0
    for b in range(x.shape[0]):
        ref_scale, ref_min, ref_codes = reference_make_qkx2_quants(x[b], w[b], nmax, fmt)
        np.testing.assert_allclose([scale[b, 0], mn[b, 0]], [ref_scale, ref_min], rtol=1e-5, atol=1e-9)
        flipped += int((codes[b] != ref_codes).sum())
    assert flipped <= codes.size * 2e-3  # near ties only: measured at most 5 of 4,608
