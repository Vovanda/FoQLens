"""The resident bench on Gemma 4 E2B: without the bf16 weights the depths read the same and the memory is freed."""

import pytest
import torch

from foqlens import model as fm
from foqlens.precision import install
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

TEXT = "The mitochondria produce most of the chemical energy a cell needs, stored as ATP."
BF16_BYTES = 2
TOLERANCE = 0.01


def test_dropping_bf16_frees_its_bytes_and_keeps_the_logits():
    model, tokenizer = fm.load(fm.E2B)
    try:
        ctl = install(model)
        ctl.set_all(Level.D8)
        ref = fm.logits(model, tokenizer, TEXT)
        bf16_bytes = sum(m.in_features * m.out_features * BF16_BYTES for m in ctl.modules.values())
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        ctl.drop_bf16()
        after = torch.cuda.memory_allocated()
        assert abs((before - after) - bf16_bytes) <= TOLERANCE * bf16_bytes
        assert torch.equal(fm.logits(model, tokenizer, TEXT), ref)
        with pytest.raises(ValueError):
            ctl.set_all(Level.BF16)
    finally:
        del model
        torch.cuda.empty_cache()
