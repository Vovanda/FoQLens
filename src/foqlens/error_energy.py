"""An oracle of the sensitivity: how much a block's output moves on this question when its weights are read coarse.

For every block b and question q, over the question's valid tokens t:

    e_b(q) = mean_t || (W_low - W_high)_b x_t ||^2

- the energy of the change the quantization error of the rung gap puts into the block's output, on the question's own
inputs. It holds both factors the working address holds only one of: the signal through the block (x) and the error a
coarser level really makes in it (W_low - W_high, read from the refined copy, not the noise model). It does not hold
how much the loss listens to that output - the gradient's half. It needs the whole network read at full precision,
so it is a reference of the bench, never a signal at inference.

Invariants:
- Invariant: padding and the first token never contribute.
- Invariant: a block whose two levels read the same weights scores 0; the energy is never negative.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from foqlens import model as fm
from foqlens.activity import block_offsets, token_mean, valid_tokens
from foqlens.precision import MixedPrecisionLinear
from foqlens.quant import Level


def block_energy(change: torch.Tensor, valid: torch.Tensor, block_rows: int) -> torch.Tensor:
    """change [batch, seq, out] -> per block of block_rows output rows, the squared norm averaged over valid tokens."""
    n = change.shape[-1]
    groups = -(-n // block_rows)
    padded = F.pad(change.float(), (0, groups * block_rows - n))
    return token_mean(padded.unflatten(-1, (groups, block_rows)).square().sum(dim=-1), valid)


class ErrorEnergyScorer:
    """The energy of the rung gap's quantization error in every block's output, per question: [batch, n_blocks]."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear], low: Level = Level.D2, high: Level = Level.D8):
        self.modules, self.low, self.high = modules, low, high
        self.offsets = block_offsets(modules)
        self.n_blocks = sum(m.n_blocks for m in modules.values())

    @torch.no_grad()
    def score_batch(self, model: nn.Module, tokenizer, texts: list[str]) -> np.ndarray:
        enc = fm.encode(tokenizer, texts, model.device)
        valid = valid_tokens(enc["attention_mask"])
        out = torch.zeros(len(texts), self.n_blocks, device=model.device)

        def hook(name: str, module: MixedPrecisionLinear):
            def record(_m: nn.Module, args) -> None:
                # the error is read from the copy per batch: kept for every module it would hold a second model
                gap = module.read_weight(self.low) - module.read_weight(self.high)
                energy = block_energy(F.linear(args[0].detach(), gap), valid, module.block_rows)
                start = self.offsets[name]
                out[:, start : start + energy.shape[1]] = energy

            return record

        handles = [m.register_forward_pre_hook(hook(n, m)) for n, m in self.modules.items()]
        try:
            model(**enc)
        finally:
            for h in handles:
                h.remove()
        return out.cpu().numpy()
