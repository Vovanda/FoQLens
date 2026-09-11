"""The bench as a unit: a loaded model with the precision controller and the sources of its masks.

Shared by the step 3 scripts so that loading and mask computation exist once. A mask source is
anything with a name, a batch size and a score_batch (MaskSource): Bench.masks does not know which
scorer is behind it, so a new score is a new source class. The bench runs at a share of the GPU
(gpu_share.py): that share of the VRAM, and rest after every batch.

Invariant: masks are always computed with every block at bf16.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
import torch

from foqlens import model as fm
from foqlens.gpu_share import FULL, Throttle
from foqlens.precision import Controller, install
from foqlens.quality import compute_masks
from foqlens.quant import Level
from foqlens.scoring import BlockScorer, GradientScorer

MASK_SOURCES = ("pooled", "gradient")  # the names of Bench.sources, in order


class MaskSource(Protocol):
    name: str
    batch_size: int

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        """One mask vector [n_blocks] per text, in order."""
        ...


@dataclass(frozen=True)
class PooledMask:
    """The naive score (ADDENDUM-01, mode B): block output norms averaged over the query's tokens."""

    scorer: BlockScorer
    batch_size: int
    name: str = "pooled"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["pooled"][0] for r in self.scorer.score_batch(model, tokenizer, texts, modes=("pooled",))]


@dataclass(frozen=True)
class GradientMask:
    """Gradient x activation of the query's own language-model loss (ADDENDUM-02)."""

    scorer: GradientScorer
    batch_size: int
    name: str = "gradient"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["gradient"][0] for r in self.scorer.score_batch(model, tokenizer, texts)]


@dataclass
class Bench:
    model: object
    tokenizer: object
    ctl: Controller
    throttle: Throttle = FULL

    @classmethod
    def load(cls, model_id: str, attn_implementation: str = "sdpa", gpu_share: float = 1.0) -> Bench:
        model, tokenizer = fm.load(model_id, attn_implementation=attn_implementation, gpu_share=gpu_share)
        return cls(model, tokenizer, install(model), Throttle(gpu_share))

    @cached_property
    def pooled(self) -> BlockScorer:
        return BlockScorer(self.ctl.modules)

    @cached_property
    def gradient(self) -> GradientScorer:
        return GradientScorer(self.model, self.ctl.modules)

    def sources(self, pooled_batch: int, gradient_batch: int) -> list[MaskSource]:
        """The standard mask sources, named as MASK_SOURCES."""
        return [PooledMask(self.pooled, pooled_batch), GradientMask(self.gradient, gradient_batch)]

    def masks(self, prompts: list[str], sources: list[MaskSource]) -> dict[str, np.ndarray]:
        """Raw masks of every prompt from every source: {source name: [prompts, n_blocks]}."""
        self.ctl.set_all(Level.BF16)
        masks = {
            src.name: compute_masks(lambda texts, src=src: src.score_batch(self.model, self.tokenizer, texts),
                                    prompts, src.batch_size, self.throttle)
            for src in sources
        }
        # the backward pass leaves a fragmented cache behind; evaluation starts from a clean one
        torch.cuda.empty_cache()
        return masks


def subtract_background(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each source's masks minus their mean over all prompts of the run."""
    return {name: m - m.mean(axis=0) for name, m in masks.items()}
