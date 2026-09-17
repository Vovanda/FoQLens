"""The resident bench on Gemma 4 E2B: without the bf16 weights the depths read the same and the memory is freed."""

import numpy as np
import pytest
import torch

from foqlens import model as fm
from foqlens.precision import ResidualSlices, install
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

TEXT = "The mitochondria produce most of the chemical energy a cell needs, stored as ATP."
BF16_BYTES = 2
# Requested bytes are the tensors' own sizes; the only other change is the capped copies' block
# indices (0.11 MiB on E2B). Allocated bytes also carry the caching allocator's unsplit block
# remainders - 11 MiB on E2B for the caps - which say nothing about what the bench holds.
TOLERANCE = 0.001
D4_SLICES = 2


def allocated() -> int:
    """Bytes requested by live tensors, without the allocator's rounding."""
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def test_dropping_bf16_frees_its_bytes_and_keeps_the_logits():
    model, tokenizer = fm.load(fm.E2B)
    try:
        ctl = install(model)
        ctl.set_all(Level.D8)
        ref = fm.logits(model, tokenizer, TEXT)
        bf16_bytes = sum(m.in_features * m.out_features * BF16_BYTES for m in ctl.modules.values())
        before = allocated()
        ctl.drop_bf16()
        after = allocated()
        assert abs((before - after) - bf16_bytes) <= TOLERANCE * bf16_bytes
        assert torch.equal(fm.logits(model, tokenizer, TEXT), ref)
        with pytest.raises(ValueError):
            ctl.set_all(Level.BF16)
    finally:
        del model
        torch.cuda.empty_cache()


def test_caps_at_d4_free_half_of_the_sliced_copy_and_keep_the_d4_logits():
    model, tokenizer = fm.load(fm.E2B)
    try:
        ctl = install(model, copy=ResidualSlices())  # caps are cut from quant.SlicedWeight only
        ctl.set_all(Level.D4)
        ref = fm.logits(model, tokenizer, TEXT)
        ctl.drop_bf16()
        full = sum(m.stored_bytes() for m in ctl.modules.values())
        before = allocated()
        ctl.set_caps(np.full(ctl.n_blocks, D4_SLICES))
        after = allocated()
        assert sum(m.stored_bytes() for m in ctl.modules.values()) == full // 2
        assert abs((before - after) - full / 2) <= TOLERANCE * full / 2
        assert ctl.stored_bits() == 4.0
        assert torch.equal(fm.logits(model, tokenizer, TEXT), ref)
        with pytest.raises(ValueError):
            ctl.set_all(Level.D6)
    finally:
        del model
        torch.cuda.empty_cache()
