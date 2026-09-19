"""The regulator: a layout policy's levels for a batch of questions, checked against what the kernel reads, set on the
bench and measured in the bytes a decoding step reads.

A policy (layouts.LayoutPolicy) answers where precision goes; the regulator is where its answer meets the model. The
tensor-core kernel (kernels/kquant.py) reads a block of 64 rows at a depth - the base and the refinements up to it,
nothing at ZERO - so a layout the kernel reads is made of ZERO and the read depths every module holds
(kernel_ladder). A layout with a level off that ladder would leave the kernel for unpacking, or be refused by a model
cut short of it; the regulator refuses it before it reaches the controller.

Memory is what a layout reads (docs/quantization-filter.md, rule 7, counted in the kernel's bytes): a block at depth d
reads its rows' base blocks and d - base refinement planes of in / 4 bytes a row. A batch decodes in one step, and
the kernel reads a block of a thread block's tokens to the deepest depth any of them asks: a step of a batch reads,
per block, the deepest depth over its samples - the union of their zones, not their sum.

Invariant: a layout the kernel cannot read - a level other than ZERO and the ladder's depths - is refused by check.
Invariant: read_bytes of a sample is the sum over blocks of the bytes the kernel reads for it: 0 at ZERO, the base
blocks at the base depth, one plane of in / 4 bytes a row per refinement above it, never deeper than stored.
Invariant: step_bytes of a batch equals read_bytes of the layout that reads every block at its deepest depth over the
batch, and is at least the largest read_bytes of one sample.
Invariant: by_layer counts every block once, in the layer and the module kind its name gives.
Invariant: no q_proj or k_proj block of a layout the regulator gives reads ZERO (rule 6); no other block changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from foqlens.layouts import AttentionLevel, LayoutPolicy
from foqlens.precision import DEPTH_BY_CODE, Controller
from foqlens.quant import LADDER, MAX_DEPTH, Level
from foqlens.refinements import KRefinedWeight

PLANE_BYTES_PER_WEIGHT = 0.25  # a refinement is a 2-bit code a weight: in / 4 bytes a row


def kernel_ladder(ctl: Controller) -> tuple[Level, ...]:
    """The rungs above ZERO a zone may lift a block to: the read depths every controlled module holds."""
    return tuple(lv for lv in LADDER if lv.depth and all(lv in m.readable for m in ctl.modules.values()))


class ReadCost:
    """The bytes the kernel reads for every block at every depth, [n_blocks, MAX_DEPTH + 1], built once per bench."""

    def __init__(self, ctl: Controller):
        rows = []
        for name, module in ctl.modules.items():
            copy = module.refined
            if not isinstance(copy, KRefinedWeight):
                raise ValueError(f"{name} holds no k-quant copy: the kernel reads nothing of it")
            base_row = copy.blocks.shape[1] * copy.blocks.shape[2]  # super-blocks x block bytes
            plane_row = copy.shape[1] * PLANE_BYTES_PER_WEIGHT
            per_row = [0.0] + [base_row + (copy.read_depth(d) - copy.fmt.base_depth) * plane_row
                               for d in range(1, MAX_DEPTH + 1)]
            rows.append(np.outer(module.block_sizes(), per_row))
        self.table = np.concatenate(rows).astype(np.int64)  # [n_blocks, MAX_DEPTH + 1]

    def read_bytes(self, codes: np.ndarray) -> np.ndarray:
        """Bytes every sample's layout reads in one step: codes [n, n_blocks] (or [n_blocks]) -> [n]."""
        depths = np.atleast_2d(DEPTH_BY_CODE[np.asarray(codes)])
        return self.table[np.arange(self.table.shape[0]), depths].sum(axis=1)

    def step_bytes(self, codes: np.ndarray) -> int:
        """Bytes one decoding step of the whole batch reads: every block at its deepest depth over the samples."""
        deepest = np.atleast_2d(DEPTH_BY_CODE[np.asarray(codes)]).max(axis=0)
        return int(self.table[np.arange(self.table.shape[0]), deepest].sum())


def check(codes: np.ndarray, ladder: tuple[Level, ...]) -> None:
    """Refuse a layout the kernel does not read: every level ZERO or a depth of `ladder`."""
    allowed = np.array(sorted({int(Level.ZERO), *(int(lv) for lv in ladder)}), dtype=np.uint8)
    off = np.setdiff1d(np.unique(codes), allowed)
    if len(off):
        raise ValueError(f"levels {[Level(int(c)).name for c in off]} are off the ladder the kernel reads "
                         f"({[lv.name for lv in ladder]} and ZERO)")


ATTENTION_KINDS = ("self_attn.q_proj", "self_attn.k_proj")  # a zero row flattens attention (docs, ZERO base)


def attention_blocks(ctl: Controller) -> np.ndarray:
    """bool [n_blocks]: the blocks of the modules that must not read ZERO - q_proj and k_proj."""
    return np.concatenate([np.full(m.n_blocks, name.endswith(ATTENTION_KINDS)) for name, m in ctl.modules.items()])


def _layer_and_kind(name: str) -> tuple[int, str]:
    """layers.12.mlp.up_proj -> (12, 'mlp.up_proj')."""
    _, layer, kind = name.split(".", 2)
    return int(layer), kind


def block_kinds(ctl: Controller) -> np.ndarray:
    """The module kind of every block in the controller's block order: [n_blocks] str, as 'mlp.up_proj'."""
    return np.concatenate([np.full(m.n_blocks, _layer_and_kind(name)[1]) for name, m in ctl.modules.items()])


def block_layers(ctl: Controller) -> np.ndarray:
    """The layer of every block in the controller's block order: [n_blocks]."""
    return np.concatenate([np.full(m.n_blocks, _layer_and_kind(name)[0]) for name, m in ctl.modules.items()])


@dataclass
class Regulator:
    """A policy's layouts on one bench: levels of a batch of questions, checked, set on the controller, measured.

    Rule 6 holds whatever the policy: q_proj and k_proj blocks it leaves at ZERO read `attention_level`, the
    lowest rung of the kernel's ladder unless given (layouts.AttentionLevel).
    """

    policy: LayoutPolicy
    ctl: Controller
    attention_level: Level | None = None
    ladder: tuple[Level, ...] = field(init=False)
    cost: ReadCost = field(init=False)
    _policy: LayoutPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ladder = kernel_ladder(self.ctl)
        self.cost = ReadCost(self.ctl)
        level = self.ladder[0] if self.attention_level is None else self.attention_level
        self._policy = AttentionLevel(self.policy, attention_blocks(self.ctl), level)

    def layout(self, indices: np.ndarray) -> np.ndarray:
        """Level codes [len(indices), n_blocks] of the questions, refused if the kernel cannot read them."""
        codes = np.asarray(self._policy.levels(np.asarray(indices)), dtype=np.uint8)
        check(codes, self.ladder)
        return codes

    def apply(self, indices: np.ndarray) -> np.ndarray:
        """Set the questions' layouts on the bench, one per sample of the batch in the order of `indices`."""
        codes = self.layout(indices)
        self.ctl.set_layout(codes)
        return codes

    def reading(self, index: Mapping[tuple[str, str], int], label: str) -> RegulatedReading:
        """The regulator as an answering.Reading: `index` maps (corpus, question id) to the question's row of the
        policy's scores."""
        return RegulatedReading(self, index, label)

    def by_layer(self, codes: np.ndarray) -> list[dict]:
        """Per layer and module kind, over the samples: its blocks, their shallowest and mean depth, and the share read
        deeper than the shallowest - where the zones went and where they did not."""
        depths = np.atleast_2d(DEPTH_BY_CODE[np.asarray(codes)])
        rows, bounds = [], np.cumsum([0] + [m.n_blocks for m in self.ctl.modules.values()])
        for (start, stop), name in zip(zip(bounds[:-1], bounds[1:]), self.ctl.modules):
            layer, kind = _layer_and_kind(name)
            part = depths[:, start:stop]
            rows.append({"layer": layer, "kind": kind, "blocks": int(stop - start),
                         "min_depth": int(part.min()), "mean_depth": float(part.mean()),
                         "above_min": float((part > part.min()).mean())})
        return rows


@dataclass
class RegulatedReading:
    """A batch read at every question's own layout (answering.Reading); every batch's codes are kept for its bytes."""

    regulator: Regulator
    index: Mapping[tuple[str, str], int]
    label: str
    laid: list[np.ndarray] = field(default_factory=list)

    def apply(self, ctl: Controller, corpus: str, rows: list) -> None:
        if ctl is not self.regulator.ctl:
            raise ValueError("a regulated reading sets the bench its regulator was built on")
        self.laid.append(self.regulator.apply(np.array([self.index[(corpus, r.id)] for r in rows])))
