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

When a layout is decided again and when the one in place is kept is not the regulator's own business: it asks the
inertia it is given (foqlens.inertia), whose condition is outside it and independent of what it decides. The default
is a decision in front of every layer at every token; the address of the query is not read before `address_layers`
layers have run, and until then nothing is raised at all.

Invariants:
- Invariant: the pass never waits for the card - the scores, the knapsack, the levels and the count of what was read
  all stay on the device, and the only read back to the host is the batch's bits a weight at the end.
- Invariant: under the densest inertia every module is decided at every visit; a module whose layout is held carries
  the one it was given, and the state it was not read on changes nothing.
- Invariant: below `address_layers` every block reads the base, whatever the price.
- Invariant: the levels the hooks write are the ones layouts.knapsack_levels gives for the same scores and price.
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

from foqlens.inertia import EVERY_LAYER, Inertia
from foqlens.layouts import RUNG_GAIN
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
    """The floor source: what enters a block, through the norms of the block's own rows.

    It tells the blocks of one module apart by their weights alone - the state gives one number to all of them - so a
    question moves the threshold of a module and never the order of its blocks.
    """

    name: str = "activity"

    def prepare(self, module: MixedPrecisionLinear, norms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """The static half of the score, already per weight of the block: read once for the run, not at every layer."""
        return norms.square() / weights

    def scores(self, module: MixedPrecisionLinear, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """[batch, blocks] per weight: one score per sample, so every question of a batch gets a layout of its own."""
        through = (x * x).sum(dim=-1, dtype=torch.float32)
        return (through.mean(dim=-1) if through.dim() == 2 else through)[:, None] * static[None, :]


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
    # The layers the address of the query is read from (E004: the hybrid at 6 identifies a paraphrase 0.917, and the
    # address of the rest of the network follows from it). The regulator hangs in front of every layer from the first,
    # and until the address is there it raises nothing: those layers are read at the base.
    address_layers: int = 0
    inertia: Inertia = EVERY_LAYER  # when a layout is decided again and when the one in place is kept
    ladder: tuple[Level, ...] = (Level.D2, Level.D4, Level.D6, Level.D8)
    _norms: dict[str, torch.Tensor] = field(default_factory=dict)
    _weights: dict[str, torch.Tensor] = field(default_factory=dict)
    _static: dict[str, torch.Tensor] = field(default_factory=dict)
    _base_codes: dict[str, torch.Tensor] = field(default_factory=dict)
    _visits: dict[str, int] = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    levels_set: dict[str, torch.Tensor] = field(default_factory=dict)
    # the layout is decided again at every token; what is counted is the first pass over a batch - the prompt's.
    # Both the levels and the count stay on the card: the hook must never wait for it to answer
    _held: dict[str, torch.Tensor] = field(default_factory=dict)
    _bits: torch.Tensor | None = None
    _used: tuple[Level, ...] = ()
    _thresholds: torch.Tensor | None = None
    _codes_table: torch.Tensor | None = None
    _total: float = 0.0
    _spent: torch.Tensor | None = None
    _counted: set[str] = field(default_factory=set)
    _counting: bool = False

    def __post_init__(self) -> None:
        for name, module in self.ctl.modules.items():
            self._norms[name] = row_norms(module)
            self._weights[name] = block_weights(module)
            self._held[name] = self._weights[name]
            self._static[name] = self.source.prepare(module, self._norms[name], self._weights[name])
            self._base_codes[name] = torch.full((module.n_blocks,), int(self.base), dtype=torch.uint8,
                                                device=self._weights[name].device)
            self._visits[name] = 0
        device = next(iter(self._weights.values())).device if self._weights else torch.device("cpu")
        # the rule of the knapsack as two tables on the card: a block rises a rung for every RUNG_GAIN it stands over
        # the price, and the rung it lands on is read out of the codes (layouts.knapsack_levels, the same rule)
        self._used = tuple(lv for lv in self.ladder if self.base <= lv <= self.ceiling)
        above = [lv for lv in self._used if lv > self.base]
        # float64, as the thresholds of layouts.knapsack_levels are: a score that lands on a threshold must rise on the
        # card exactly where it rises on the host, and rounding the threshold to float32 would move that edge
        self._thresholds = torch.tensor([self.price * RUNG_GAIN ** k for k in range(len(above))],
                                        dtype=torch.float64, device=device)
        self._codes_table = torch.tensor([int(self.base)] + [int(lv) for lv in above],
                                         dtype=torch.uint8, device=device)
        self._bits = torch.zeros(int(Level.BF16) + 1, device=device)
        for level in self.ladder:
            self._bits[int(level)] = level.bits
        self._total = float(sum(float(held.sum()) for held in self._held.values()))

    def layer_of(self, name: str) -> int:
        return int(name.split(".")[1])

    def levels_for(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """The levels of one module's blocks from the state `x` entering its layer: [batch, blocks] of level codes on
        the card, a layout per sample - the questions of a batch are read each by its own.

        Nothing here leaves the device: the hook runs in front of every layer of every pass, and a read back to the
        host would stop the pipeline at each of them. It is not what holds the card down, though - over the 5% the
        pass kept it at 17.6% with the layout on the card against 16.2% with the knapsack on the host, where a uniform
        rung keeps it at 81% (runs/regulator-card, runs/regulator/e2b-it, 20.09). The difference left between them is
        that a rung's step is replayed from a captured graph while a decided layout leaves every launch to Python;
        what each of the two costs is measured by scripts/decode_step_speed.py --layerwise-price.
        """
        module = self.ctl.modules[name]
        if self.layer_of(name) < self.address_layers:  # the address is not read yet: nothing is raised here
            return self._base_codes[name]  # one layout for the batch: at the base every question reads the same
        reduced = self.source.scores(module, x, self._static[name])
        rungs = (reduced[..., None] >= self._thresholds).sum(dim=-1)
        codes = self._codes_table[rungs]
        if self.rim > 0:  # the band under the risen part, held one rung over the base
            width = max(int(round(self.rim * codes.shape[-1])), 0)
            rung = self._codes_table[1] if len(self._codes_table) > 1 else self._codes_table[0]
            resting = codes <= int(self.base)
            # stable, so that blocks of an equal score enter the band in their own order and a pass is reproducible
            order = torch.where(resting, reduced, torch.full_like(reduced, -float("inf"))).argsort(
                dim=-1, descending=True, stable=True)
            chosen = torch.zeros_like(codes, dtype=torch.bool).scatter_(-1, order[..., :width], True)
            codes = torch.where(chosen & resting, rung, codes)
        return codes

    def attach(self, model: nn.Module) -> LayerwiseRegulator:
        """Put a hook in front of every layer of the model; the hooks stay until `detach`."""
        layers = {self.layer_of(name) for name in self.ctl.modules}
        decoder = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for number in sorted(layers):
            layer = decoder.layers[number]
            self._handles.append(layer.register_forward_pre_hook(self._before(number)))
        return self

    def decide(self, name: str, state: torch.Tensor) -> torch.Tensor:
        """Read one module's levels from the state entering its layer and write them, before the layer runs.

        Whether it is read again here or the layout in place is kept, the inertia says (foqlens.inertia) - the
        condition sits outside the regulator and is asked, never inferred. What the module reads is counted either
        way: a layout that was kept is read as much as one just decided.
        """
        visit = self._visits[name]
        self._visits[name] = visit + 1
        if name in self.levels_set and self.inertia.holds(self.layer_of(name), visit):
            codes = self.levels_set[name]
        else:
            codes = self.levels_for(name, state)
            self.levels_set[name] = codes
            self.ctl.modules[name].set_levels_on_device(codes, self._used)
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
        self._spent = torch.zeros(batch, device=self._bits.device)
        self._counted, self._counting = set(), True
        self._visits = dict.fromkeys(self._visits, 0)  # a batch is a pass of its own: the inertia starts over with it

    def _spend(self, name: str, codes: torch.Tensor) -> None:
        """Add what one module read to the running count, on the card: the sum comes back to the host once, at the end
        of the batch (`bits_a_weight`), so that counting costs the pass nothing."""
        if name in self._counted:  # a module decided twice: the prompt's pass is over and the generation has begun
            self._counting = False
            return
        self._counted.add(name)
        self._spent = self._spent + (self._bits[codes.long()] * self._held[name]).sum(dim=-1).reshape(-1)

    def bits_a_weight(self) -> np.ndarray:
        """What the counted pass read, in bits a weight, per question of the batch: [batch]."""
        self._counting = False
        if self._spent is None:
            return np.zeros(0)
        if len(self._counted) < len(self.ctl.modules):  # a pass that did not reach every module cannot be divided
            raise RuntimeError(f"the counted pass read {len(self._counted)} of {len(self.ctl.modules)} modules")
        return (self._spent / self._total).cpu().numpy()

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
            rows.append(np.atleast_2d(written.to("cpu", torch.uint8).numpy()) if written is not None
                        else np.full((1, module.n_blocks), int(self.base), dtype=np.uint8))
        width = max(row.shape[0] for row in rows)
        return np.concatenate([np.repeat(row, width // row.shape[0], axis=0) for row in rows], axis=1)
