"""The bench as a unit: a loaded model with the precision controller and the sources of its masks.

Shared by the run scripts so that loading and mask computation exist once. A mask source is
anything with a name, a batch size and a score_batch (MaskSource): Bench.masks does not know which
scorer is behind it, so a new score is a new source class. The bench runs at a share of the GPU
(gpu_share.py): that share of the VRAM, and rest after every batch.

The address sources of #18 are registered by name in ADDRESS_SOURCES with their batch and whether they can read the
first layers only; a new source is one more entry, and every script that lists sources takes it from there.

Invariant: masks are computed with every block at one level - bf16 unless a working address asks for its base.
Invariant: every name of ADDRESS_SOURCES builds a source of that name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch

from foqlens import model as fm
from foqlens import refocustensors
from foqlens.activity import HeadEnergyScorer, HybridScorer, NeuronActivityScorer
from foqlens.error_energy import ErrorEnergyScorer
from foqlens.gpu_monitor import gpu_temperature
from foqlens.gpu_share import FULL, Cooldown, Pacer, ThermalGuard, Throttle
from foqlens.precision import Controller, install
from foqlens.quality import compute_masks, token_batches
from foqlens.quant import Level
from foqlens.scoring import BlockScorer, GradientScorer
from foqlens.runlog import stage

LOG = logging.getLogger(__name__)

MASK_SOURCES = ("pooled", "gradient")  # the names of Bench.sources, in order
# Mask passes are batched by tokens (quality.token_batches): a batch holds at most as many padded tokens as
# *_BATCH of the run's longest prompts, and at most MAX_BATCH_FACTOR times as many prompts. Measured on E2B
# (RTX 3090 Ti, 2026-09-12): 8 of the longest prompts are the gradient batch known to fit the 0.8 GPU share
# (16.6 GiB reserved), 16 of them reserve 22 GiB. The gradient pass is bound by kernel launches - 16
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
    """The naive score: block output norms averaged over the query's tokens."""

    scorer: BlockScorer
    batch_size: int
    name: str = "pooled"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["pooled"][0] for r in self.scorer.score_batch(model, tokenizer, texts, modes=("pooled",))]


@dataclass(frozen=True)
class GradientMask:
    """Gradient x activation of the query's own language-model loss."""

    scorer: GradientScorer
    batch_size: int
    name: str = "gradient"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["gradient"][0] for r in self.scorer.score_batch(model, tokenizer, texts)]


@dataclass(frozen=True)
class GradientMagnitudeMask:
    """The same backward pass summed as |gradient x activation|: terms of opposite sign do not cancel (issue #17)."""

    scorer: GradientScorer
    batch_size: int
    name: str = "gradient_magnitude"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return [r["gradient_magnitude"][0] for r in self.scorer.score_batch(model, tokenizer, texts)]


@dataclass(frozen=True)
class NeuronActivityMask:
    """Source 2 of #18: phi(gate) * up per group of neurons, on their gate and up blocks - forward only."""

    scorer: NeuronActivityScorer
    batch_size: int
    name: str = "neuron_activity"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return list(self.scorer.score_batch(model, tokenizer, texts))


@dataclass(frozen=True)
class HeadEnergyMask:
    """Source 3 of #18: every head's output energy at the input of o_proj, on its q_proj blocks - forward only."""

    scorer: HeadEnergyScorer
    batch_size: int
    name: str = "head_energy"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return list(self.scorer.score_batch(model, tokenizer, texts))


@dataclass(frozen=True)
class HybridMask:
    """Neuron activity and head energy in one forward pass, each at length 1 per question, summed (activity.HybridScorer):
    the MLP's gate and up blocks and the attention's q blocks in one address."""

    scorer: HybridScorer
    batch_size: int
    name: str = "hybrid"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return list(self.scorer.score_batch(model, tokenizer, texts))


@dataclass(frozen=True)
class ErrorEnergyMask:
    """An oracle of the sensitivity (error_energy.py): the energy of the D2-D8 quantization error in every block's output,
    on the question's inputs at full precision - a reference of the bench, never a signal at inference."""

    scorer: ErrorEnergyScorer
    batch_size: int
    name: str = "error_energy"

    def score_batch(self, model, tokenizer, texts: list[str]) -> list[np.ndarray]:
        return list(self.scorer.score_batch(model, tokenizer, texts))


class ModelSource(Enum):
    """Where the bench takes its model from: the cut .refocustensors folder, read to the source (FILE) or only to its
    depths with no source weight of a controlled module loaded (RESIDENT - D2 ... D8, no bf16, so no masks), or the
    Hugging Face checkpoint."""

    FILE = "file"
    RESIDENT = "resident"
    CHECKPOINT = "checkpoint"


@dataclass
class Bench:
    model: object
    tokenizer: object
    ctl: Controller
    throttle: Pacer = FULL

    @classmethod
    def load(cls, model_id: str, attn_implementation: str = "sdpa", gpu_share: float = 1.0,
             source: ModelSource = ModelSource.FILE, directory: Path | None = None) -> Bench:
        """The bench on the model's cut folder (scripts/cut_model.py) - its copy read from the file, never quantized
        again - or on the Hugging Face checkpoint, the copy then quantized from bf16 as it is first read. `directory`
        names another cut folder of the same model, one cut over a published file's base (cut_model.py --base-gguf)."""
        directory = directory or refocustensors.model_directory(model_id)
        if source is not ModelSource.CHECKPOINT and not (directory / refocustensors.FILE).exists():
            raise FileNotFoundError(f"no cut model in {directory}: run scripts/cut_model.py, or load the checkpoint")
        with stage(LOG, f"load {model_id} from {source.name.lower()} {directory.name}"):
            if source is ModelSource.FILE:
                model, tokenizer, copy = refocustensors.load(directory, attn_implementation=attn_implementation,
                                                             gpu_share=gpu_share)
                ctl = install(model, copy=copy)
            elif source is ModelSource.RESIDENT:
                model, tokenizer, ctl = refocustensors.load_resident(directory, attn_implementation=attn_implementation,
                                                                     gpu_share=gpu_share)
            else:
                model, tokenizer = fm.load(model_id, attn_implementation=attn_implementation, gpu_share=gpu_share)
                ctl = install(model)
        # the share paces the batches, the hourly break rests the card, the guard keeps it under its
        # ceiling whatever the share
        log = LOG.info
        pacer = ThermalGuard(Cooldown(Throttle(gpu_share), log=log), gpu_temperature, log=log)
        return cls(model, tokenizer, ctl, pacer)

    @cached_property
    def pooled(self) -> BlockScorer:
        return BlockScorer(self.ctl.modules)

    @cached_property
    def gradient(self) -> GradientScorer:
        return GradientScorer(self.model, self.ctl.modules)

    def sources(self, pooled_batch: int, gradient_batch: int) -> list[MaskSource]:
        """The standard mask sources, named as MASK_SOURCES."""
        return [PooledMask(self.pooled, pooled_batch), GradientMask(self.gradient, gradient_batch)]

    def source(self, name: str, batch_size: int | None = None, layers: set[int] | None = None) -> MaskSource:
        """A mask source of #18 by its name (ADDRESS_SOURCES), at its own batch unless `batch_size` is given; `layers`
        limits a source that can read the first layers only - the first N of a working address."""
        if name not in ADDRESS_SOURCES:
            raise ValueError(f"unknown mask source {name!r}, expected one of {sorted(ADDRESS_SOURCES)}")
        spec = ADDRESS_SOURCES[name]
        if layers is not None and not spec.reads_layers:
            raise ValueError(f"{name} reads every layer; only {sorted(n for n, s in ADDRESS_SOURCES.items() if s.reads_layers)} read the first ones")
        return spec.make(self, batch_size or spec.batch, layers)

    @property
    def n_heads(self) -> int:
        return self.model.config.get_text_config(decoder=True).num_attention_heads

    def masks(self, prompts: list[str], sources: list[MaskSource], level: Level = Level.BF16) -> dict[str, np.ndarray]:
        """Raw masks of every prompt from every source, the model read at `level`: {source name: [prompts, n_blocks]}.
        bf16 is the reference; a working address reads at the base precision."""
        self.ctl.set_all(level)
        lengths = [len(ids) for ids in self.tokenizer(prompts)["input_ids"]]  # the tokens encode() pads to
        longest = max(lengths, default=1)
        masks = {}
        for src in sources:
            with stage(LOG, f"masks {src.name} of {len(prompts)} prompts at {level.name}"):
                masks[src.name] = compute_masks(
                    lambda texts, src=src: src.score_batch(self.model, self.tokenizer, texts), prompts, src.batch_size,
                    self.throttle,
                    groups=token_batches(lengths, src.batch_size * longest, src.batch_size * MAX_BATCH_FACTOR),
                    lengths=lengths,
                )
        # the backward pass leaves a fragmented cache behind; evaluation starts from a clean one
        torch.cuda.empty_cache()
        return masks


@dataclass(frozen=True)
class AddressSource:
    """How a mask source of #18 is built on a bench: make(bench, batch size, layers or None), its own batch, and whether
    it can read the first layers only (a forward source) or needs the whole pass (a backward one)."""

    make: Callable[[Bench, int, set[int] | None], MaskSource]
    batch: int
    reads_layers: bool


ADDRESS_SOURCES: dict[str, AddressSource] = {
    "pooled": AddressSource(lambda b, n, _: PooledMask(b.pooled, n), POOLED_BATCH, False),
    "neuron_activity": AddressSource(
        lambda b, n, layers: NeuronActivityMask(NeuronActivityScorer(b.ctl.modules, layers), n), POOLED_BATCH, True),
    "head_energy": AddressSource(
        lambda b, n, layers: HeadEnergyMask(HeadEnergyScorer(b.ctl.modules, b.n_heads, layers), n), POOLED_BATCH, True),
    "hybrid": AddressSource(
        lambda b, n, layers: HybridMask(HybridScorer((NeuronActivityScorer(b.ctl.modules, layers),
                                                     HeadEnergyScorer(b.ctl.modules, b.n_heads, layers))), n),
        POOLED_BATCH, True),
    "gradient": AddressSource(lambda b, n, _: GradientMask(b.gradient, n), GRADIENT_BATCH, False),
    "error_energy": AddressSource(lambda b, n, _: ErrorEnergyMask(ErrorEnergyScorer(b.ctl.modules), n), POOLED_BATCH, False),
    "gradient_magnitude": AddressSource(lambda b, n, _: GradientMagnitudeMask(b.gradient, n), GRADIENT_BATCH, False),
}


def subtract_background(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each source's masks minus their mean over all prompts of the run."""
    return {name: m - m.mean(axis=0) for name, m in masks.items()}
