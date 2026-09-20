"""The layer-wise regulator (docs/layerwise-regulator.md, variant B): the level of a block is decided while the pass
runs, from what its layer is given, and never before the pass.

A hook sits in front of every layer. It is given the state entering the layer, scores that layer's blocks, turns the
scores into levels by the knapsack's rule (foqlens.layouts.knapsack_levels - the same rule the filter uses) and writes
them into the controller before the layer multiplies anything. Nothing is fitted: the price of memory and the ceiling
are the controls, and the score is read in the pass.

The sources of a score, in the order of what they can tell (docs/layerwise-regulator.md):

- `activity`: the norm of what enters a block times the norms of its own rows. Inside a layer gate, up, q, k and v are
  given one vector, so between their blocks this is the rows alone - a property of the model. It is the floor.
- `votes`: a block's score is the activity of the blocks that feed it, weighted by the coupling of the signal's path.
  It is the only source that looks ahead per question; it needs the coupling table (foqlens.coupling).

Invariants:
- Invariant: a layer's levels are written before that layer runs and never after it.
- Invariant: with the price at zero every block reads the ceiling, and with the price above every score the base.
- Invariant: the rim never puts a block above one rung over the base.
- Invariant: what a run counts is one pass over the prompts - every module of the controller once, whatever the
  generation that follows decides; with every block at a rung the count is that rung's bits a weight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from foqlens.layouts import knapsack_levels
from foqlens.precision import Controller, MixedPrecisionLinear
from foqlens.quant import Level


@dataclass(frozen=True)
class HookedReading:
    """An answering.Reading that leaves the levels to the hooks: the base is set before the batch, and every layer is
    decided while the pass runs."""

    regulator: LayerwiseRegulator
    label: str

    def apply(self, ctl: Controller, corpus: str, rows: list) -> None:
        ctl.set_all(self.regulator.base)


def row_norms(module: MixedPrecisionLinear) -> torch.Tensor:
    """The norm of every block's rows of a module, read once from the weights: [blocks]."""
    weight = module.weight.data.float()
    rows = module.block_rows
    blocks = -(-weight.shape[0] // rows)
    padded = torch.zeros(blocks * rows, weight.shape[1], dtype=weight.dtype, device=weight.device)
    padded[:weight.shape[0]] = weight
    return padded.view(blocks, rows, -1).flatten(1).norm(dim=1)


def block_weights(module: MixedPrecisionLinear) -> torch.Tensor:
    """The weights every block of a module holds: [blocks]."""
    rows = module.block_rows
    blocks = -(-module.weight.shape[0] // rows)
    held = torch.full((blocks,), float(rows * module.weight.shape[1]), device=module.weight.device)
    tail = module.weight.shape[0] - rows * (blocks - 1)
    held[-1] = tail * module.weight.shape[1]
    return held


@dataclass
class Activity:
    """The floor source: what enters a block, through the norms of the block's own rows."""

    name: str = "activity"

    def scores(self, module: MixedPrecisionLinear, x: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """[batch, blocks]: one score per sample, so every question of a batch gets a layout of its own."""
        # the same vector reaches every block of a module: between them this is the rows alone
        through = x.float().square().sum(dim=-1).mean(dim=-1) if x.dim() == 3 else x.float().square().sum(dim=-1)
        return through[:, None] * norms.square()[None, :]


SOURCES: dict[str, Callable[[], object]] = {"activity": Activity}


@dataclass
class LayerwiseRegulator:
    """Variant B over a loaded bench: hooks in front of the layers, a price of memory and a ceiling.

    `price` is the knapsack's lambda over the reduced score (score per weight): a block rises a rung for every 16 times
    it stands above it. `rim` is the share of a layer's blocks, taken after the risen ones, held one rung over the base.
    """

    ctl: Controller
    source: object
    price: float
    base: Level = Level.D2
    ceiling: Level = Level.D8
    rim: float = 0.0
    ladder: tuple[Level, ...] = (Level.D2, Level.D4, Level.D6, Level.D8)
    _norms: dict[str, torch.Tensor] = field(default_factory=dict)
    _weights: dict[str, torch.Tensor] = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    levels_set: dict[str, np.ndarray] = field(default_factory=dict)
    # the layout is decided again at every token; what is counted is the first pass over a batch - the prompt's.
    # The tables the count reads sit on the host from the start: the hook must not wait for the card to answer
    _held: dict[str, np.ndarray] = field(default_factory=dict)
    _bits: np.ndarray | None = None
    _total: float = 0.0
    _spent: np.ndarray | None = None
    _counted: set[str] = field(default_factory=set)
    _counting: bool = False

    def __post_init__(self) -> None:
        for name, module in self.ctl.modules.items():
            self._norms[name] = row_norms(module)
            self._weights[name] = block_weights(module)
            self._held[name] = self._weights[name].cpu().numpy()
        self._bits = np.zeros(int(Level.BF16) + 1)
        for level in self.ladder:
            self._bits[int(level)] = level.bits
        self._total = float(sum(held.sum() for held in self._held.values()))

    def layer_of(self, name: str) -> int:
        return int(name.split(".")[1])

    def levels_for(self, name: str, x: torch.Tensor) -> np.ndarray:
        """The levels of one module's blocks from the state `x` entering its layer: [batch, blocks], a layout per
        sample - the questions of a batch are read each by its own."""
        module = self.ctl.modules[name]
        reduced = (self.source.scores(module, x, self._norms[name]) / self._weights[name]).cpu().numpy()
        above = tuple(lv for lv in self.ladder if self.base <= lv <= self.ceiling)
        codes = knapsack_levels(reduced, self.price, self.base, above)
        if self.rim > 0:  # the band under the risen part, held one rung over the base
            width = max(int(round(self.rim * codes.shape[1])), 0)
            rung = int(above[1]) if len(above) > 1 else int(self.base)
            for row, score in zip(codes, reduced):
                resting = np.flatnonzero(row <= int(self.base))
                row[resting[np.argsort(-score[resting])][:width]] = rung
        return codes

    def attach(self, model: nn.Module) -> LayerwiseRegulator:
        """Put a hook in front of every layer of the model; the hooks stay until `detach`."""
        layers = {self.layer_of(name) for name in self.ctl.modules}
        decoder = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for number in sorted(layers):
            layer = decoder.layers[number]
            self._handles.append(layer.register_forward_pre_hook(self._before(number)))
        return self

    def decide(self, name: str, state: torch.Tensor) -> np.ndarray:
        """Read one module's levels from the state entering its layer and write them, before the layer runs."""
        codes = self.levels_for(name, state)
        self.levels_set[name] = codes
        self.ctl.modules[name].set_levels(codes)
        if self._counting:
            self._spend(name, codes)
        return codes

    def _before(self, number: int):
        def hook(_module: nn.Module, args, kwargs=None) -> None:
            state = args[0] if args else kwargs["hidden_states"]
            for name in self.ctl.names(number):
                self.decide(name, state)

        return hook

    def start_counting(self, batch: int) -> None:
        """Count what the next pass over a batch of `batch` questions reads - the pass over their prompts. The layout
        is decided again at every token of the generation, so the count ends by itself the moment a module is decided
        a second time: by then the prompt's pass has been through every module once."""
        self._spent, self._counted, self._counting = np.zeros(batch), set(), True

    def _spend(self, name: str, codes: np.ndarray) -> None:
        if name in self._counted:  # a module decided twice: the prompt's pass is over and the generation has begun
            self._counting = False
            return
        self._counted.add(name)
        self._spent = self._spent + np.atleast_1d((self._bits[codes] * self._held[name]).sum(axis=-1))

    def bits_a_weight(self) -> np.ndarray:
        """What the counted pass read, in bits a weight, per question of the batch: [batch]."""
        self._counting = False
        if self._spent is None:
            return np.zeros(0)
        if len(self._counted) < len(self.ctl.modules):  # a pass that did not reach every module cannot be divided
            raise RuntimeError(f"the counted pass read {len(self._counted)} of {len(self.ctl.modules)} modules")
        return self._spent / self._total

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reading(self, label: str) -> HookedReading:
        """The regulator as an answering.Reading: it sets the base before a batch and the hooks do the rest."""
        return HookedReading(self, label)

    def layout(self) -> np.ndarray:
        """The levels the hooks have written, in the controller's block order: [batch, n_blocks]."""
        rows = []
        for name, module in self.ctl.modules.items():
            written = self.levels_set.get(name)
            rows.append(np.atleast_2d(written) if written is not None
                        else np.full((1, module.n_blocks), int(self.base), dtype=np.uint8))
        width = max(row.shape[0] for row in rows)
        return np.concatenate([np.repeat(row, width // row.shape[0], axis=0) for row in rows], axis=1)
