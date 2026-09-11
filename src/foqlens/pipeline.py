"""The bench as a unit: a loaded model with the precision controller and its two mask sources.

Shared by the step 3 scripts so that loading and mask computation exist once.

Invariant: masks are always computed with every block at bf16.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from foqlens import model as fm
from foqlens.precision import Controller, install
from foqlens.quality import compute_masks
from foqlens.quant import Level
from foqlens.scoring import BlockScorer, GradientScorer

MASK_SOURCES = ("pooled", "gradient")


@dataclass
class Bench:
    model: object
    tokenizer: object
    ctl: Controller

    @classmethod
    def load(cls, model_id: str, attn_implementation: str = "sdpa") -> Bench:
        model, tokenizer = fm.load(model_id, attn_implementation=attn_implementation)
        return cls(model, tokenizer, install(model))

    @cached_property
    def pooled(self) -> BlockScorer:
        return BlockScorer(self.ctl.modules)

    @cached_property
    def gradient(self) -> GradientScorer:
        return GradientScorer(self.model, self.ctl.modules)

    def masks(self, prompts: list[str], pooled_batch: int, gradient_batch: int) -> dict[str, np.ndarray]:
        """Raw masks of every prompt from every source: {source: [prompts, n_blocks]}."""
        self.ctl.set_all(Level.BF16)
        m, t = self.model, self.tokenizer
        sources = {
            "pooled": (lambda texts: [r["pooled"][0] for r in self.pooled.score_batch(m, t, texts, modes=("pooled",))], pooled_batch),
            "gradient": (lambda texts: [r["gradient"][0] for r in self.gradient.score_batch(m, t, texts)], gradient_batch),
        }
        masks = {name: compute_masks(score, prompts, batch) for name, (score, batch) in sources.items()}
        # the backward pass leaves a fragmented cache behind; evaluation starts from a clean one
        torch.cuda.empty_cache()
        return masks


def subtract_background(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each source's masks minus their mean over all prompts of the run."""
    return {name: m - m.mean(axis=0) for name, m in masks.items()}
