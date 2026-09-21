"""The oracles' scores at the granularity of the groups, brought to one scale and laid over one another - read
from their kept files, never from the model.

The scale is the whole point. An oracle measures in its own units and an experiment has to compare them, add them
and read them all at one band, so every one of them is carried onto the same quantity: what a group demands.

- what each oracle measures, and why they may all be divided by the same threshold:

  | oracle | its own units | what a group's value is |
  | --- | --- | --- |
  | lift | nats | how much the answer's likelihood improves if this group alone is read at the top |
  | drop | nats | how much it worsens if this group alone falls to the base |
  | answer_gradient | nats a weight | the gradient of the answer's likelihood by this group's weights |
  | answer_quant_gap | nats | that gradient against the D2-D8 gap: the first order of dropping the group |
  | error_energy | squared output | the energy a rung's error puts into this group's output |
  | pooled | none | the others, each scaled to unit mass over the groups, averaged |

  Every one of them is a loss the answer takes when this group is read coarsely - in nats, in energy, in the first
  order of nats. The question's threshold is a loss of the same kind: what the answer may lose in all. So the ratio
  of the two is a pure number in every case, and it is the same pure number regardless of which oracle measured it:
  the share of the allowance this group would spend.

- demand: that ratio, folded into 0 ... 1. Nothing else in this module invents a scale.
- block_group_ids, to_groups: a block oracle's scores summed into the groups the oracles by trying read.
- overlay: several fields as one - `product` is agreement, `mean` the soft one, `least` the severest.
- bootstrap: an interval of a statistic of the questions by resampling them.

Invariants:
- Invariant: to_groups keeps the sum of the scores; a NaN score adds nothing.
- Invariant: levels_of over a demand field gives back the map the threshold search itself wrote, group for group -
  the check that the scale carries what the map was built from (tests/test_oracle_overlay_unit.py).
- Invariant: an overlay of one field is that field; an overlay of fields in 0 ... 1 stays in 0 ... 1.
- Invariant: least <= product <= mean everywhere - the usual order of those means.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def block_group_ids(block_layer: np.ndarray, block_kind: np.ndarray, names: list[str]) -> np.ndarray:
    """Every block's group in the oracle by trying's order (foqlens.group_oracle names), from its layer and module
    kind: [n_blocks]."""
    label = [f"{layer}.{'attention' if kind.startswith('self_attn.') else 'mlp'}"
             for layer, kind in zip(block_layer.tolist(), block_kind.tolist())]
    return np.array([names.index(g) for g in label])


def to_groups(scores: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """[questions, n_blocks] -> [questions, n_groups]: every group's blocks summed."""
    out = np.zeros((scores.shape[0], n_groups))
    np.add.at(out.T, groups, np.nan_to_num(scores, nan=0.0).T)
    return out


class Oracle(ABC):
    """An oracle of the bench, from outside.

    The contract is the field and it is one for all of them: a matrix [questions, groups] in 0 ... 1, the questions
    and the groups in an order given from outside and shared by every oracle, a larger number meaning a group this
    question needs more. Whatever was measured and in whatever units - an answer swept group by group, an energy of
    the error, a gradient - lives inside the oracle and never leaves it. Only under one contract do the rungs lay
    out over the fields at all, and only then is one analysis written once and run over all of them.

    What the number becomes in rungs belongs to a reading strategy, not to the oracle.
    """

    name: str

    @abstractmethod
    def field(self) -> np.ndarray:
        """[questions, groups] in 0 ... 1."""
def demand(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """An oracle's values as what a group demands, in 0 ... 1 and on one scale for every oracle and every question.

    A group's value over its question's threshold says how much of the answer's whole allowance reading it coarsely
    would spend: below 1 the group fits at the base, above it asks for a sharper rung, and the further above the
    sharper. That ratio is already free of the oracle's units and of how unsure the model was of the question, so it
    is the thing every oracle can be compared on - but it runs to infinity, and a field is 0 ... 1.

    So it is folded: demand = d / (1 + d) for d the ratio. Nothing is lost, the fold keeps the order, and the bounds
    of a reading fold with it - a rung whose error is a share r of the base's is taken while demand stays below
    1 / (1 + r). They are the same numbers for every oracle and every question.
    """
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    thresholds = np.asarray(thresholds, dtype=float)[:, None]
    ratio = np.divide(values, thresholds, out=np.full(values.shape, np.inf), where=thresholds > 0)
    return np.nan_to_num(ratio / (1.0 + ratio), nan=1.0, posinf=1.0)
@dataclass(frozen=True)
class Kept(Oracle):
    """An oracle read from what it has already written down: its values and the thresholds of the questions they
    were measured against, carried onto the contract by `demand`."""

    name: str
    values: np.ndarray  # [questions, groups] in the oracle's own units
    thresholds: np.ndarray  # [questions] what the answer may lose in all, in those same units

    def field(self) -> np.ndarray:
        return demand(self.values, self.thresholds)


def overlay(fields: list[np.ndarray], how: str = "product") -> np.ndarray:
    """Several unit fields as one, on the same scale as a single field.

    `product` - the n-th root of the product, agreement: one oracle near zero puts a group out. `mean` - the soft
    one, a high oracle carries a group against the others. `least` - the lowest oracle alone.
    """
    stack = np.stack([np.asarray(f, dtype=float) for f in fields])
    if how == "product":
        return np.exp(np.log(np.clip(stack, 1e-12, None)).mean(axis=0))
    if how == "mean":
        return stack.mean(axis=0)
    if how == "least":
        return stack.min(axis=0)
    raise ValueError(f"no overlay {how!r}: product, mean or least")


def bootstrap(values: np.ndarray, statistic: Callable[[np.ndarray], float], draws: int, seed: int,
              level: float) -> tuple[float, float]:
    """A `level` interval of `statistic` over `values`' first axis, by `draws` resamples."""
    rng = np.random.default_rng(seed)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))
