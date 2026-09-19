"""Oracles of the sensitivity by trying: how much the answer gains when one group of blocks is read higher, and loses
when one is read lower - measured on the answer itself, not estimated.

A group is a layer's attention or its MLP (and the layer's per-layer modules with its MLP): about 70 on E2B. For a
question, every variant is a layout of its own read by one sample of a batch (Controller.set_layout per sample), and
the measure is the mean negative log-likelihood of the reference answer's tokens after the prompt:

- lift: from every block at `low`, one group at `high` - the gain of the group over the base (the knapsack's step);
- drop: from every block at `high`, one group at `low` - the loss of the group from the top;
- minimal mask: groups added at `high` in the order of their lift, until the answer's NLL is within `tolerance` of every
  block at `high` - the least of the network the question needs sharp, the ideal a filter approximates.

It reads the whole network many times per question, so it is a reference of the bench, never a signal at inference.

Invariants:
- Invariant: the variant of no group lifted is the base, and of every group lifted is every block at `high`.
- Invariant: a minimal mask is a prefix of the lift order, and no shorter prefix is within the tolerance.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from foqlens import model as fm
from foqlens.precision import Controller
from foqlens.quant import Level
from foqlens.scoring import answer_losses

ATTENTION = "self_attn."
RUN_FIELDS = frozenset({"groups"})  # of a run's .npz, what describes the run, not its questions (io.read_npz_parts)


def block_groups(ctl: Controller) -> tuple[np.ndarray, list[str]]:
    """Every block's group - its layer's attention or the rest of the layer (the MLP and the per-layer modules) - and
    the groups' names: ([n_blocks] int, [groups] str)."""
    names, ids = [], []
    for name, module in ctl.modules.items():
        _, layer, kind = name.split(".", 2)
        group = f"{layer}.{'attention' if kind.startswith(ATTENTION) else 'mlp'}"
        if group not in names:
            names.append(group)
        ids.append(np.full(module.n_blocks, names.index(group)))
    return np.concatenate(ids), names


def lift_layouts(groups: np.ndarray, chosen: list[np.ndarray], low: Level, high: Level) -> np.ndarray:
    """A layout per set of groups: every block at `low`, the blocks of the chosen groups at `high`: [variants, n_blocks]."""
    out = np.full((len(chosen), len(groups)), int(low), dtype=np.uint8)
    for row, picked in enumerate(chosen):
        out[row, np.isin(groups, picked)] = int(high)
    return out


def joined_answer(prompt: str, answer: str) -> str:
    """The answer as the model would write it after `prompt`: right after a prompt that ends on whitespace (the -it
    turn ends with "model\\n", where a leading space makes another first token and costs it ~20 nats), else after one."""
    answer = answer.strip()
    return answer if not prompt or prompt[-1].isspace() else " " + answer


def minimal_prefix(nll: np.ndarray, target: float, tolerance: float) -> int:
    """The fewest groups of a prefix sweep (nll[k] = the answer's NLL with the first k groups lifted) whose NLL is
    within `tolerance` of `target`; the whole sweep when none is."""
    within = np.flatnonzero(nll <= target + tolerance)
    return int(within[0]) if len(within) else len(nll) - 1


def minimal_layouts(groups: np.ndarray, lift: np.ndarray, minimal: np.ndarray, low: Level, high: Level) -> np.ndarray:
    """Every question's minimal mask as a layout: its first `minimal` groups by lift at `high`, the rest at `low`, the
    order the prefix sweep lifted them in: [questions, n_blocks]."""
    order = np.argsort(-lift, axis=1, kind="stable")
    return np.concatenate([lift_layouts(groups, [o[:k]], low, high) for o, k in zip(order, minimal)])


@torch.no_grad()
def answer_nll(model: nn.Module, tokenizer, prompts: list[str], answers: list[str]) -> torch.Tensor:
    """The mean negative log-likelihood of every answer's tokens after its prompt, one sample each: [batch].

    Logits are taken at the answer's positions only, so a long prompt costs no [seq, vocab] logits."""
    starts = [len(tokenizer(p)["input_ids"]) for p in prompts]
    enc = fm.encode(tokenizer, [p + a for p, a in zip(prompts, answers)], model.device)
    return answer_losses(model, model.model(**enc).last_hidden_state, enc, starts)
