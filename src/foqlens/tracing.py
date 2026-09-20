"""A pass of one question, written down layer by layer, so that it is walked afterwards instead of run again.

A live pass cannot be stepped back through: it is gone once it has run. A trace holds what a layer was given and what
its blocks answered, so the walk goes forward and back at no cost, on the host, while the card is free - and a
candidate formula for a block's importance is tried over the arrays, not over the model.

What a trace holds for one question:

- `state[layer]`: the state entering that layer, [tokens, model] - every formula over the input of a layer is
  computable from it, not only the ones thought of when the trace was written;
- `response[name]`: the norm of what a block put out, per token, [tokens, blocks] - the network's own answer about
  which of its blocks this question uses, which the input alone does not give;
- `static[name]`: what does not depend on the question - the norm of a block's rows, the weights it holds and the
  norm of the error the rungs themselves make (`gap`, ||W_base - W_ceiling|| a block);
- `tokens`: the prompt's ids, so that a step of the walk can be read as text.

The oracle's field for the same question is kept beside a trace, never inside it: it is measured elsewhere
(foqlens.precision_field) and a trace must not pretend to know it.

Invariant: a trace holds every controlled module of every layer the pass ran, and one row of `response` per token of
the prompt.
Invariant: `static` does not depend on the question - two traces of the same bench hold the same static part.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from foqlens.kernels.kquant import kquant_unpack
from foqlens.precision import DEPTH_BY_CODE, Controller, MixedPrecisionLinear
from foqlens.quant import Level


@dataclass(frozen=True)
class BlockStatic:
    """What a block is, apart from any question: rows, weights and the error a coarse reading makes in it."""

    norms: np.ndarray  # [blocks] the norm of the block's rows
    weights: np.ndarray  # [blocks] the weights the block holds
    gap: np.ndarray  # [blocks] the norm of W(base) - W(ceiling) over the block's rows


@dataclass
class Trace:
    """One question's pass, as arrays on the host."""

    tokens: list[int] = field(default_factory=list)
    state: dict[int, np.ndarray] = field(default_factory=dict)
    response: dict[str, np.ndarray] = field(default_factory=dict)
    static: dict[str, BlockStatic] = field(default_factory=dict)

    def layers(self) -> list[int]:
        return sorted(self.state)

    def modules_of(self, layer: int) -> list[str]:
        return sorted(name for name in self.response if int(name.split(".")[1]) == layer)


def block_response(module: MixedPrecisionLinear, out: torch.Tensor) -> torch.Tensor:
    """[tokens, blocks]: the norm of what every block of rows put out, for each token."""
    flat = out.reshape(-1, out.shape[-1])
    blocks = -(-module.out_features // module.block_rows)
    padded = torch.zeros(flat.shape[0], blocks * module.block_rows, device=flat.device, dtype=torch.float32)
    padded[:, : module.out_features] = flat.float()
    return padded.view(flat.shape[0], blocks, module.block_rows).norm(dim=-1)


def rung_gap(module: MixedPrecisionLinear, base: Level, ceiling: Level) -> torch.Tensor:
    """[blocks]: the norm, over a block's rows, of the difference between reading it at `ceiling` and at `base` - the
    error the rung itself makes, as against the norm of the weights, which says only how large the block is."""
    copy = module.refined
    device = copy.blocks.device
    at = {}
    for level in (base, ceiling):
        depths = torch.full((module.n_blocks,), int(DEPTH_BY_CODE[int(level)]), dtype=torch.uint8, device=device)
        at[level] = kquant_unpack(copy, depths)
    difference = (at[ceiling] - at[base]).float()
    rows = module.block_rows
    blocks = module.n_blocks
    padded = torch.zeros(blocks * rows, difference.shape[1], device=difference.device)
    padded[: module.out_features] = difference
    return padded.view(blocks, rows, -1).flatten(1).norm(dim=1)


class Tracer:
    """Hooks over a controller's modules that write one pass into a Trace; `record` holds them for that pass."""

    def __init__(self, ctl: Controller, base: Level = Level.D2, ceiling: Level = Level.D8):
        self.ctl, self.base, self.ceiling = ctl, base, ceiling
        self.trace = Trace()
        self._handles: list = []

    def statics(self) -> None:
        """The part of a trace that does not depend on the question; read once, before any pass."""
        from foqlens.layerwise import block_weights, row_norms

        for name, module in self.ctl.modules.items():
            self.trace.static[name] = BlockStatic(
                norms=row_norms(module).cpu().numpy(),
                weights=block_weights(module).cpu().numpy(),
                gap=rung_gap(module, self.base, self.ceiling).cpu().numpy())

    def attach(self, model: nn.Module) -> Tracer:
        decoder = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for number in sorted({int(name.split(".")[1]) for name in self.ctl.modules}):
            self._handles.append(decoder.layers[number].register_forward_pre_hook(self._state(number)))
        for name, module in self.ctl.modules.items():
            self._handles.append(module.register_forward_hook(self._response(name)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _state(self, number: int):
        def hook(_module: nn.Module, args, kwargs=None) -> None:
            state = args[0] if args else kwargs["hidden_states"]
            self.trace.state[number] = state[0].float().cpu().numpy()  # one question a trace

        return hook

    def _response(self, name: str):
        def hook(module: MixedPrecisionLinear, _args, out: torch.Tensor) -> None:
            self.trace.response[name] = block_response(module, out).cpu().numpy()

        return hook
