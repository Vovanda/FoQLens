"""Oracles of the sensitivity by trying: how much the answer gains when one group of blocks is read higher, and loses
when one is read lower - measured on the answer itself, not estimated.

A group is a layer's attention or its MLP (and the layer's per-layer modules with its MLP): about 70 on E2B. For a
question, every variant is a layout of its own read by one sample of a batch (Controller.set_layout per sample), and
the measure is the mean negative log-likelihood of the reference answer's tokens after the prompt:

- lift: from every block at `low`, one group at `high` - the gain of the group over the base (the knapsack's step);
- drop: from every block at `high`, one group at `low` - the loss of the group from the top;
- minimal mask: groups added at `high` in the order of their lift, until the answer's NLL is within `tolerance` of every
  block at `high` - the least of the network the question needs sharp, the ideal a filter approximates;
- zeroing: with the minimal mask at `high` and the rest at `low`, the groups outside the mask switched off (ZERO) one by
  one, the least needed first, while the answer holds - what the question does not need at all.

It reads the whole network many times per question, so it is a reference of the bench, never a signal at inference.

Invariants:
- Invariant: the variant of no group lifted is the base, and of every group lifted is every block at `high`.
- Invariant: a minimal mask is a prefix of the lift order, and no shorter prefix is within the tolerance.
- Invariant: a zeroing never switches off a group of the minimal mask, and its j-th variant has exactly j groups off.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def group_order(scores: np.ndarray) -> np.ndarray:
    """The groups by an oracle's score, the most needed first: [..., groups] -> the same shape of group indices. Every
    oracle - the lift by trying, the gradient, the error energy, the address - gives a chain in its own order."""
    return np.argsort(-scores, axis=-1, kind="stable")


def minimal_layouts(groups: np.ndarray, orders: np.ndarray, minimal: np.ndarray, low: Level, high: Level) -> np.ndarray:
    """Every question's minimal mask as a layout: its first `minimal` groups of its order at `high`, the rest at `low`:
    [questions, n_blocks]."""
    return np.concatenate([lift_layouts(groups, [o[:k]], low, high) for o, k in zip(orders, minimal)])


def zero_order(order: np.ndarray, minimal: int) -> np.ndarray:
    """The groups outside a question's minimal mask, the least needed first: the order they are switched off in, once
    the chain that gives the answer is at `high` and the rest at `low` (Volodya 20.09 01:55)."""
    return order[minimal:][::-1]


def zeroed_layouts(groups: np.ndarray, order: np.ndarray, minimal: int, low: Level, high: Level,
                   off: Level = Level.ZERO) -> np.ndarray:
    """From the minimal mask at `high` and the rest at `low`, the first j groups of zero_order at `off`, j = 1..the
    groups outside the mask: [variants, n_blocks]."""
    base = lift_layouts(groups, [order[:minimal]], low, high)[0]
    offs = zero_order(order, minimal)
    out = np.repeat(base[None], len(offs), axis=0)
    for j in range(len(offs)):
        out[j, np.isin(groups, offs[:j + 1])] = int(off)
    return out


def chain_layout(groups: np.ndarray, order: np.ndarray, minimal: int, zeroed: int, low: Level, high: Level,
                 off: Level = Level.ZERO) -> np.ndarray:
    """A question's chain in three levels: its minimal mask at `high`, the `zeroed` least needed groups at `off`, the
    rest at `low`: [n_blocks]."""
    if zeroed <= 0:
        return lift_layouts(groups, [order[:minimal]], low, high)[0]
    return zeroed_layouts(groups, order, minimal, low, high, off)[zeroed - 1]


def zeroed_count(nll: np.ndarray, target: float, tolerance: float) -> int:
    """How many groups of a zeroing sweep (nll[j-1] = the answer's NLL with the first j switched off) go off before the
    answer's NLL first leaves `tolerance` of `target`: the longest prefix that holds."""
    out = np.flatnonzero(nll > target + tolerance)
    return int(out[0]) if len(out) else len(nll)


def variants_nll(model: nn.Module, tokenizer, ctl: Controller, pacer, prompt: str, answer: str, layouts: np.ndarray,
                 size: int) -> np.ndarray:
    """The answer's NLL under every layout, `size` variants a batch: [variants].

    The last batch is filled up to `size` by repeating its last layout, so every batch of a question is one GEMM shape:
    bf16 rounds by the shape, and the variants of a question are compared across batches."""
    return sweep_until(model, tokenizer, ctl, pacer, prompt, answer, layouts, size, lambda _: False)


def sweep_until(model: nn.Module, tokenizer, ctl: Controller, pacer, prompt: str, answer: str, layouts: np.ndarray,
                size: int, done) -> np.ndarray:
    """As variants_nll, stopping after the first batch where `done(nll so far)` holds: the rest stays NaN. A chain's
    sweep ends at its first prefix within the tolerance, a zeroing at its first step out of it - most questions need
    one or two batches, not the whole order."""
    out = np.full(len(layouts), np.nan)
    for start in range(0, len(layouts), size):
        part = layouts[start:start + size]
        full = np.concatenate([part, np.repeat(part[-1:], size - len(part), axis=0)])
        with pacer.batch():
            ctl.set_layout(full)
            out[start:start + len(part)] = answer_nll(model, tokenizer, [prompt] * size,
                                                      [answer] * size)[:len(part)].cpu().numpy()
        if done(out[:start + len(part)]):
            break
    return out


@dataclass
class Chain:
    """A question's chain in one oracle's order: the prefix sweep (NLL with the first k+1 groups at `high`, NaN after
    the stop), the minimal mask, the zeroing sweep (NLL with the first j+1 least needed off) and how many went off."""

    prefix: np.ndarray
    minimal: int
    zero_sweep: np.ndarray
    zeroed: int


def build_chain(model: nn.Module, tokenizer, ctl: Controller, pacer, prompt: str, answer: str, groups: np.ndarray,
                order: np.ndarray, ends: np.ndarray, low: Level, high: Level, tolerance: float,
                zero_tolerance: float | None, size: int) -> Chain:
    """The chain that gives the answer in `order` - the fewest groups at `high` over `low` within `tolerance` of every
    block at `high` (ends = the NLL with every block at low, at high) - then, with zero_tolerance, the least needed of
    the rest switched off while the answer stays within it."""
    g = len(order)
    target = ends[1]
    prefix = np.full(g, np.nan)
    if ends[0] > target + tolerance:  # the base alone is not enough: lift in the order until it is
        prefix = sweep_until(model, tokenizer, ctl, pacer, prompt, answer,
                             lift_layouts(groups, [order[:k + 1] for k in range(g)], low, high), size,
                             lambda nll: bool((nll <= target + tolerance).any()))
    minimal = minimal_prefix(np.concatenate([[ends[0]], prefix]), target, tolerance)
    zero_sweep, zeroed = np.full(g, np.nan), -1
    if zero_tolerance is not None and minimal < g:
        off = zeroed_layouts(groups, order, minimal, low, high)
        zero_sweep[:len(off)] = sweep_until(model, tokenizer, ctl, pacer, prompt, answer, off, size,
                                            lambda nll: bool((nll > target + zero_tolerance).any()))
        zeroed = zeroed_count(zero_sweep[:len(off)], target, zero_tolerance)
    return Chain(prefix, minimal, zero_sweep, zeroed)


def grade_chain(model: nn.Module, tokenizer, ctl: Controller, pacer, prompt: str, answer: str, groups: np.ndarray,
                layout: np.ndarray, chain: np.ndarray, rungs: tuple[Level, ...], target: float, tolerance: float,
                size: int) -> tuple[np.ndarray, np.ndarray]:
    """The chain graded (Volodya 20.09 02:10: not the whole chain at the top): its groups from the least needed (the
    last of `chain`) to the most, each read at the coarsest of `rungs` (coarsest first, all below the chain's level)
    that keeps the answer within `tolerance` of `target`, on top of the grades already taken; a group no rung holds
    stays. The rungs of one group are one batch. Returns the graded layout [n_blocks] and every chain group's level."""
    layout = layout.copy()
    levels = np.array([layout[groups == c][0] for c in chain], dtype=np.uint8)
    for k in range(len(chain) - 1, -1, -1):
        tries = np.repeat(layout[None], len(rungs), axis=0)
        for r, rung in enumerate(rungs):
            tries[r, groups == chain[k]] = int(rung)
        nll = sweep_until(model, tokenizer, ctl, pacer, prompt, answer, tries, max(1, min(size, len(rungs))),
                          lambda _: False)
        held = np.flatnonzero(nll <= target + tolerance)
        if len(held):
            layout = tries[held[0]]
            levels[k] = int(rungs[held[0]])
    return layout, levels


@torch.no_grad()
def answer_nll(model: nn.Module, tokenizer, prompts: list[str], answers: list[str]) -> torch.Tensor:
    """The mean negative log-likelihood of every answer's tokens after its prompt, one sample each: [batch].

    Logits are taken at the answer's positions only, so a long prompt costs no [seq, vocab] logits."""
    starts = [len(tokenizer(p)["input_ids"]) for p in prompts]
    enc = fm.encode(tokenizer, [p + a for p, a in zip(prompts, answers)], model.device)
    return answer_losses(model, model.model(**enc).last_hidden_state, enc, starts)
