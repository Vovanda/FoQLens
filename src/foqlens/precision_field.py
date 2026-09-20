"""The precision field of a question (Volodya 20.09 03:10): a field over the groups of blocks whose value is itself the
level a group is read at - 1.0 is D8, 0.75 D6, 0.5 D4, 0.2 D2 - its peak at D8 and falling from it. Reading the field is
the layout; the question is whether the answer on it is the answer of every block at D8.

An oracle gives a field of importance s_g - what reading group g coarse costs the answer. The field of precision is
read from it with one threshold a question:

- the cost of a group at a rung r is c_g(r) = s_g * e_g(r), where e_g(r) is how much the rung cuts the group's error
  against D2 (e_g(D2) = 1), measured from the error energies of the rungs on the same questions (rung_ratios);
- a group's level at a threshold eps is the coarsest rung whose cost is within eps, the top where none is
  (field_levels);
- eps is the largest threshold whose layout keeps the model's own answer within the tolerance of every block at the
  top, read in batches beside every block at the top (search_threshold over the sorted costs, as the layouts coarsen
  with eps).

The reference is measured, not scored: every block at the top and one group read at each rung below it; a group's
level is the coarsest rung that holds the answer alone (alone_levels), then the whole layout is raised by whole rungs
until it holds jointly (raised).

It reads the whole network many times a question, so it is a reference of the bench, never a signal at inference.

Invariants:
- Invariant: field_levels is monotone in eps: a larger threshold never reads a group finer.
- Invariant: at thresholds()[0] every group is at the top; at the last threshold every group of a positive field is at
  its coarsest rung.
- Invariant: search_threshold returns an index that was read and held, or 0 - every group at the top - when none did.
- Invariant: raised by as many rungs as lie between the coarsest and the top, every group is at the top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from torch import nn

from foqlens import group_oracle
from foqlens.quant import Level

RUNGS = (Level.D2, Level.D4, Level.D6)  # below the top, coarsest first
TOP = Level.D8
# How much of the base rung's error every rung leaves, measured on the bench over both shards of 2% (rung_ratios over
# the error energies, median over the groups; the shards agree to the third digit). The noise model would give 1/16 a
# rung - 0.063 and 0.004 - and the measurement does not: D4 cuts the error by half, D6 by eighteen times.
RUNG_ERROR = {Level.D2: 1.0, Level.D4: 0.53, Level.D6: 0.055, Level.D8: 0.0}
# The field's value of every level: what a rung is worth, 0 at the base and 1 at the top. Volodya 20.09 sets the
# middle rungs close to the top - D6 0.9, D4 0.8 - so that a map asking for precision is served by the cheapest rung
# that nearly does the job: on these values D4 removes 0.20 of the base's error a bit against D6's 0.15 and D8's 0.125.
FIELD_VALUE = {Level.D2: 0.0, Level.D4: 0.8, Level.D6: 0.9, Level.D8: 1.0}


def rung_ratios(energies: Sequence[np.ndarray]) -> np.ndarray:
    """How much each rung cuts a group's error against the coarsest: energies [rungs][questions, groups], coarsest
    first, of the same questions -> [rungs, groups], the median over the questions of E_r / E_coarsest; the first row
    is 1. A group whose coarsest rung makes no error on any question (it reads the same weights) keeps 1 at every rung."""
    base = energies[0]
    out = np.ones((len(energies), base.shape[1]))
    for r, e in enumerate(energies[1:], start=1):
        ratio = np.where(base > 0, e / np.where(base > 0, base, 1.0), np.nan)
        ratio[np.isnan(e) | np.isnan(base)] = np.nan
        seen = ~np.isnan(ratio).all(axis=0)
        out[r, seen] = np.nanmedian(ratio[:, seen], axis=0)
    return out


def rung_costs(field: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """A group's cost at every rung: [groups] field, [rungs, groups] ratios -> [rungs, groups]. A negative importance
    (the group read coarse helps the answer) costs nothing."""
    return np.maximum(field, 0.0)[None, :] * ratios


def field_levels(costs: np.ndarray, eps: np.ndarray, rungs: Sequence[Level] = RUNGS, top: Level = TOP) -> np.ndarray:
    """Every group's level at every threshold: the coarsest rung whose cost is within it, else the top: [rungs, groups]
    costs, [k] thresholds -> [k, groups] codes."""
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    within = costs[None] <= eps[:, None, None]  # [k, rungs, groups]
    codes = np.array([int(r) for r in rungs], dtype=np.uint8)
    return np.where(within.any(axis=1), codes[within.argmax(axis=1)], int(top)).astype(np.uint8)


def priced_levels(costs: np.ndarray, price: np.ndarray, weights: np.ndarray | None = None,
                  rungs: Sequence[Level] = RUNGS, top: Level = TOP,
                  bits: Sequence[float] | None = None) -> np.ndarray:
    """Every group's level at every price of memory: the rung where the cost of reading it and the cost of storing it
    are together the least - [rungs, groups] costs, [k] prices -> [k, groups] codes.

    Reading a group at a rung leaves it the cost `costs` (the top leaves none); storing it takes the rung's bits a
    weight, which the old reading of a field ignored, so a map took a middle rung whenever it helped at all and paid
    for it with memory (measured 20.09: the reference spends 66% of the network on D4 and D6). Here a rung is taken
    only when what it saves is worth its bits, one price of memory a question.

    `weights` scales the bits by what a group holds; without it every group counts alike. `bits` replaces what a rung
    is charged for, rungs first and the top last: a ladder that charges the top more than its eight bits asks whether
    the map holds when its peaks are made dear (Volodya 20.09).

    Invariant: at price 0 every group reads the top, and at a price above every saving every group reads the coarsest
    rung; in between a larger price never reads a group finer.
    """
    price = np.atleast_1d(np.asarray(price, dtype=float))
    ladder = [*rungs, top]
    charged = np.array([lv.bits for lv in ladder] if bits is None else list(bits), dtype=float)
    if len(charged) != len(ladder):
        raise ValueError(f"{len(charged)} prices for {len(ladder)} rungs")
    charged = charged[:, None]
    if weights is not None:
        charged = charged * (weights / weights.mean())[None, :]
    whole = np.concatenate([costs, np.zeros((1, costs.shape[1]))])  # the top leaves no cost behind
    total = whole[None] + price[:, None, None] * charged[None]  # [k, ladder, groups]
    codes = np.array([int(lv) for lv in ladder], dtype=np.uint8)
    return codes[total.argmin(axis=1)].astype(np.uint8)


def thresholds(costs: np.ndarray) -> np.ndarray:
    """The thresholds at which the layout changes, sorted, after -inf (every group at the top): [1 + distinct costs]."""
    return np.concatenate([[-np.inf], np.unique(costs[np.isfinite(costs)])])


def search_threshold(holds: Callable[[np.ndarray], np.ndarray], n: int, width: int) -> int:
    """The largest index of n sorted candidates whose reading holds, index 0 holding by definition: every round reads up
    to `width` indices spread over the open interval (holds(indices) -> bool per index), keeps the largest that held as
    the lower end and the next one read above it as the upper end, until the two meet. Exact when holding is monotone;
    otherwise a holding index all the same."""
    lo, hi = 0, n
    while hi - lo > 1:
        inside = hi - lo - 1
        if inside <= width:
            probes = np.arange(lo + 1, hi)
        else:
            probes = np.unique(np.linspace(lo, hi, width + 2)[1:-1].round().astype(int))
        held = np.asarray(holds(probes), dtype=bool)
        good = np.flatnonzero(held)
        j = int(good[-1]) if len(good) else -1
        lo = int(probes[j]) if j >= 0 else lo
        hi = int(probes[j + 1]) if j + 1 < len(probes) else hi
    return lo


def reference_layouts(groups: np.ndarray, n_groups: int, rungs: Sequence[Level] = RUNGS,
                      top: Level = TOP) -> np.ndarray:
    """Every block at the top, then one group at each rung: [1 + rungs * groups, n_blocks], rung-major."""
    out = np.full((1 + len(rungs) * n_groups, len(groups)), int(top), dtype=np.uint8)
    for r, rung in enumerate(rungs):
        for g in range(n_groups):
            out[1 + r * n_groups + g, groups == g] = int(rung)
    return out


def alone_levels(nll: np.ndarray, target: float, tolerance: float, rungs: Sequence[Level] = RUNGS,
                 top: Level = TOP) -> np.ndarray:
    """Every group's coarsest rung that holds the answer alone: [rungs, groups] NLL of one group at a rung, the rest at
    the top -> [groups] codes, the top where no rung holds."""
    return field_levels(nll - target, np.array([tolerance]), rungs, top)[0]


def raised(levels: np.ndarray, steps: int, rungs: Sequence[Level] = RUNGS, top: Level = TOP) -> np.ndarray:
    """Every group raised by `steps` rungs of the ladder rungs + top, the top staying the top: [groups] codes."""
    ladder = np.array([int(r) for r in rungs] + [int(top)])
    place = np.searchsorted(ladder, levels)
    return ladder[np.minimum(place + steps, len(ladder) - 1)].astype(np.uint8)


def values(levels: np.ndarray) -> np.ndarray:
    """The field's value of every level code (FIELD_VALUE): the same shape."""
    table = np.zeros(max(int(k) for k in FIELD_VALUE) + 1)
    for level, v in FIELD_VALUE.items():
        table[int(level)] = v
    return table[levels]


@dataclass
class Found:
    """A field read at its threshold: the threshold's index among the candidates and its value, the levels [groups],
    the answer's NLL on the layout and on every block at the top in the same batch, and the batches read."""

    index: int
    eps: float
    levels: np.ndarray
    nll: float
    target: float
    batches: int


def find_threshold(model: nn.Module, tokenizer, ctl, pacer, prompt: str, answer: str, groups: np.ndarray,
                   costs: np.ndarray, tolerance: float, width: int, rungs: Sequence[Level] = RUNGS,
                   top: Level = TOP, share: float = 0.0) -> Found:
    """A question's threshold for one field (search_threshold over thresholds(costs)): every batch holds every block at
    the top first and `width` layouts after it, padded to one shape, and a layout holds when its NLL is within the
    tolerance of that row's - `tolerance` nats, or `share` of what the top rung itself spends, whichever is larger.

    The share is what makes the tolerance reachable on a question the top rung is unsure of: a flat 0.02 nats beside a
    top rung that answers at 0.15 leaves no layout but the whole network at the top.
    """
    candidates = thresholds(costs)
    everything = np.full(len(groups), int(top), dtype=np.uint8)
    seen: dict[int, tuple[float, float]] = {}
    batches = 0

    def holds(probes: np.ndarray) -> np.ndarray:
        nonlocal batches
        levels = field_levels(costs, candidates[probes], rungs, top)
        layouts = np.concatenate([everything[None], levels[:, groups]])
        nll = group_oracle.variants_nll(model, tokenizer, ctl, pacer, prompt, answer, layouts, width + 1)
        batches += 1
        for p, v in zip(probes.tolist(), nll[1:].tolist()):
            seen[p] = (v, float(nll[0]))
        return nll[1:] <= nll[0] + max(tolerance, share * abs(float(nll[0])))

    index = search_threshold(holds, len(candidates), width)
    levels = field_levels(costs, candidates[index:index + 1], rungs, top)[0]
    if index == 0:  # nothing coarser held: every group at the top, read as the reference of the first batch
        first = next(iter(seen.values()), (np.nan, np.nan))
        nll, target = first[1], first[1]
    else:
        nll, target = seen[index]
    return Found(index, float(candidates[index]), levels, nll, target, batches)


@dataclass
class Measured:
    """The reference field of a question: the NLL of every group alone at every rung [rungs, groups], of every block at
    the top, each group's coarsest rung alone, the joint NLL of the layout raised by 0.. rungs, the rungs it took and
    the levels it holds at [groups]."""

    alone: np.ndarray
    target: float
    alone_levels: np.ndarray
    joint: np.ndarray
    steps: int
    levels: np.ndarray


def measure_field(model: nn.Module, tokenizer, ctl, pacer, prompt: str, answer: str, groups: np.ndarray,
                  n_groups: int, tolerance: float, size: int, rungs: Sequence[Level] = RUNGS,
                  top: Level = TOP, share: float = 0.0) -> Measured:
    """A question's reference field: every group alone at every rung beside every block at the top (reference_layouts,
    `size` a batch), each group at its coarsest rung that holds alone, then the layout raised by 0, 1, ... rungs in one
    batch beside every block at the top, the fewest that hold kept.

    The tolerance is `tolerance` nats or `share` of what the top rung spends on this question, whichever is larger.
    """
    nll = group_oracle.variants_nll(model, tokenizer, ctl, pacer, prompt, answer,
                                    reference_layouts(groups, n_groups, rungs, top), size)
    target = float(nll[0])
    tolerance = max(tolerance, share * abs(target))
    alone = nll[1:].reshape(len(rungs), n_groups)
    single = alone_levels(alone, target, tolerance, rungs, top)
    steps = np.arange(len(rungs) + 1)
    trials = np.stack([raised(single, int(k), rungs, top) for k in steps])
    everything = np.full(len(groups), int(top), dtype=np.uint8)
    joint_all = group_oracle.variants_nll(model, tokenizer, ctl, pacer, prompt, answer,
                                          np.concatenate([everything[None], trials[:, groups]]), len(steps) + 1)
    joint = joint_all[1:]
    held = np.flatnonzero(joint <= joint_all[0] + tolerance)
    k = int(held[0]) if len(held) else len(rungs)  # raised by every rung is every block at the top
    return Measured(alone, target, single, joint, k, trials[k])
