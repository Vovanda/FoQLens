"""Forward signals of the address (issue #18): what a query lights up, read in one forward pass.

- NeuronActivityScorer (source 2): per group of 64 neurons the norm of phi(gate) * up, the input of
  down_proj - the forward half of the Taylor score of up_proj. It scores the gate and up blocks of those
  neurons; GRIFFIN selects neurons per prompt by the same statistic.
- HeadEnergyScorer (source 3): per head the norm of its output at the input of o_proj - the forward
  half of a head's Taylor score (Michel et al. 2019). It scores the q_proj blocks of that head; a head
  looking at the attention sink returns almost nothing, so its energy is small by itself.

A signal is the mean over the query's valid tokens - padding and the first token left out, as in
scoring.py. Blocks a signal does not see score 0: carrying it onto them is the projection of #18
(projection.py), not this module. Which layers are read is a parameter: the working role of a source
reads the first N layers only, at the floor.

Invariants:
- Invariant: padding and the first token never contribute to a signal.
- Invariant: the gate and up blocks of one group of neurons get the same score.
- Invariant: every q_proj block of one head gets that head's energy; blocks of unread layers and modules score 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from foqlens import model as fm
from foqlens.precision import MixedPrecisionLinear


def valid_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """The tokens a signal counts: real ones, without the first (<bos>) - [batch, seq] of 0/1."""
    valid = attention_mask.clone().float()
    valid[:, 0] = 0
    return valid


def token_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """values [batch, seq, ...] averaged over valid tokens: [batch, ...]."""
    weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.einsum("bs...,bs->b...", values, weights)


def block_activity(inputs: torch.Tensor, valid: torch.Tensor, block_rows: int) -> torch.Tensor:
    """Per group of block_rows input columns, the norm of the input averaged over valid tokens: [batch, groups]."""
    n = inputs.shape[-1]
    groups = -(-n // block_rows)
    padded = torch.nn.functional.pad(inputs.float(), (0, groups * block_rows - n))
    return token_mean(padded.unflatten(-1, (groups, block_rows)).norm(dim=-1), valid)


def head_energy(inputs: torch.Tensor, valid: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Per head, the norm of its slice of the o_proj input (heads side by side) averaged over valid tokens: [batch, n_heads]."""
    return token_mean(inputs.float().unflatten(-1, (n_heads, -1)).norm(dim=-1), valid)


def block_offsets(modules: dict[str, MixedPrecisionLinear]) -> dict[str, int]:
    """Where every module's blocks start in the controller's block order."""
    starts = np.cumsum([0, *(m.n_blocks for m in modules.values())])[:-1]
    return dict(zip(modules, starts.tolist()))


def _layer(name: str) -> int:
    return int(name.split(".")[1])


class _InputRecorder:
    """Forward pre-hooks that reduce a module's input where it arrives and keep only the reduction."""

    def __init__(self, modules: dict[str, nn.Module], reduce):
        self.modules, self.reduce = modules, reduce
        self.values: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        self._handles = [m.register_forward_pre_hook(self._hook(n)) for n, m in self.modules.items()]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, name: str):
        def hook(_m: nn.Module, args) -> None:
            self.values[name] = self.reduce(args[0].detach())

        return hook


class NeuronActivityScorer:
    """Source 2 of #18: the activity of every group of MLP neurons, on their gate and up blocks."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear], layers: set[int] | None = None):
        self.modules = modules
        self.offsets = block_offsets(modules)
        self.n_blocks = sum(m.n_blocks for m in modules.values())
        self.down = {n: m for n, m in modules.items() if n.endswith("mlp.down_proj") and (layers is None or _layer(n) in layers)}

    @torch.no_grad()
    def score_batch(self, model: nn.Module, tokenizer, texts: list[str]) -> np.ndarray:
        enc = fm.encode(tokenizer, texts, model.device)
        valid = valid_tokens(enc["attention_mask"])
        block_rows = next(iter(self.modules.values())).block_rows
        with _InputRecorder(self.down, lambda x: block_activity(x, valid, block_rows)) as rec:
            model(**enc)
        return self.assemble(rec.values, len(texts)).cpu().numpy()

    def assemble(self, activity: dict[str, torch.Tensor], batch: int) -> torch.Tensor:
        """The recorded activity of every read down_proj onto the gate and up blocks of its layer: [batch, n_blocks]."""
        out = torch.zeros(batch, self.n_blocks, device=next(iter(activity.values())).device if activity else "cpu")
        for name, values in activity.items():
            prefix = name.removesuffix("mlp.down_proj")
            for target in ("mlp.gate_proj", "mlp.up_proj"):
                start = self.offsets[prefix + target]
                out[:, start : start + values.shape[1]] = values
        return out


class HeadEnergyScorer:
    """Source 3 of #18: the output energy of every attention head, on the q_proj blocks of that head."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear], n_heads: int, layers: set[int] | None = None):
        self.modules, self.n_heads = modules, n_heads
        self.offsets = block_offsets(modules)
        self.n_blocks = sum(m.n_blocks for m in modules.values())
        self.o = {n: m for n, m in modules.items() if n.endswith("self_attn.o_proj") and (layers is None or _layer(n) in layers)}

    @torch.no_grad()
    def energies(self, model: nn.Module, tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
        """Every read layer's head energies: {o_proj name: [batch, n_heads]}."""
        enc = fm.encode(tokenizer, texts, model.device)
        valid = valid_tokens(enc["attention_mask"])
        with _InputRecorder(self.o, lambda x: head_energy(x, valid, self.n_heads)) as rec:
            model(**enc)
        return rec.values

    def score_batch(self, model: nn.Module, tokenizer, texts: list[str]) -> np.ndarray:
        return self.assemble(self.energies(model, tokenizer, texts), len(texts)).cpu().numpy()

    def assemble(self, energy: dict[str, torch.Tensor], batch: int) -> torch.Tensor:
        """Every head's energy onto the q_proj blocks of its head: [batch, n_blocks]."""
        out = torch.zeros(batch, self.n_blocks, device=next(iter(energy.values())).device if energy else "cpu")
        for name, values in energy.items():
            q = name.removesuffix("self_attn.o_proj") + "self_attn.q_proj"
            per_head = self.modules[q].n_blocks // self.n_heads
            start = self.offsets[q]
            out[:, start : start + per_head * self.n_heads] = values.repeat_interleave(per_head, dim=1)
        return out
