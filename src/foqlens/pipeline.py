"""The bench as a unit: a loaded model with the precision controller and the sources of its masks.

Shared by the step 3 scripts so that loading and mask computation exist once. A mask source is
anything with a name, a batch size and a score_batch (MaskSource): Bench.masks does not know which
scorer is behind it, so a new score is a new source class. The bench runs at a share of the GPU
(gpu_share.py): that share of the VRAM, and rest after every batch.

Invariant: masks are always computed with every block at bf16.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Protocol

import numpy as np
import torch

from foqlens import model as fm
from foqlens.gpu_monitor import gpu_temperature
from foqlens.gpu_share import FULL, Cooldown, Pacer, ThermalGuard, Throttle
from foqlens.precision import Controller, install
from foqlens.quality import compute_masks, token_batches
from foqlens.quant import Level
from foqlens.scoring import BlockScorer, GradientScorer

MASK_SOURCES = ("pooled", "gradient")  # the names of Bench.sources, in order
# Mask passes are batched by tokens (quality.token_batches): a batch holds at most as many padded tokens as
# *_BATCH of the run's longest prompts, and at most MAX_BATCH_FACTOR times as many prompts. Measured on E2B
# (RTX 3090 Ti, 2026-09-12): 8 of the longest prompts are the gradient batch known to fit the 0.8 GPU share
# (E009, 16.6 GiB reserved), 16 of them reserve 22 GiB. The gradient pass is bound by kernel launches - 16
# short prompts take the time of 8 - so short prompts gain from larger batches at the same memory.
# Masks are not batch-invariant in bf16: runs that compare masks keep one batching.
GRADIENT_BATCH = 8
POOLED_BATCH = 32
MAX_BATCH_FACTOR = 4


class MaskSource(Protocol):
    name: str
    batch_size: int

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        """One mask vector [n_blocks] per text, in order."""
        ...


@dataclass(frozen=True)
class PooledMask:
    """The naive score (E001 PREREG, mode B): block output norms averaged over the query's tokens."""

    scorer: BlockScorer
    batch_size: int
    name: str = "pooled"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["pooled"][0] for r in self.scorer.score_batch(model, tokenizer, texts, modes=("pooled",))]


@dataclass(frozen=True)
class GradientMask:
    """Gradient x activation of the query's own language-model loss (E002)."""

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
    throttle: Pacer = FULL

    @classmethod
    def load(cls, model_id: str, attn_implementation: str = "sdpa", gpu_share: float = 1.0) -> Bench:
        model, tokenizer = fm.load(model_id, attn_implementation=attn_implementation, gpu_share=gpu_share)
        # the share paces the batches, the hourly break rests the card, the guard keeps it under its
        # ceiling whatever the share
        log = partial(print, flush=True)
        pacer = ThermalGuard(Cooldown(Throttle(gpu_share), log=log), gpu_temperature, log=log)
        return cls(model, tokenizer, install(model), pacer)

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
        lengths = [len(ids) for ids in self.tokenizer(prompts)["input_ids"]]  # the tokens encode() pads to
        longest = max(lengths, default=1)
        masks = {
            src.name: compute_masks(
                lambda texts, src=src: src.score_batch(self.model, self.tokenizer, texts), prompts, src.batch_size,
                self.throttle, groups=token_batches(lengths, src.batch_size * longest, src.batch_size * MAX_BATCH_FACTOR),
            )
            for src in sources
        }
        # the backward pass leaves a fragmented cache behind; evaluation starts from a clean one
        torch.cuda.empty_cache()
        return masks


def subtract_background(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each source's masks minus their mean over all prompts of the run."""
    return {name: m - m.mean(axis=0) for name, m in masks.items()}
