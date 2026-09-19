"""How much of the network a layout sharpens, per question and over a run (Volodya 19.09): the zones of a question, the
share of the network's weights under each and under all of them, each zone's own ceiling, and what that costs against
the uniform ladder - so that a run shows at once whether two zones swallowed the network and what the filter saves.

A zone's share is the weight of the blocks its lift reaches (lift > 0), over all the controlled weights. The share
lifted is the weight of the blocks the layout reads above the base precision: every zone together, overlaps counted
once, plus what rule 6 lifts from a ZERO base. A layout with no zones (the per-block control) has only the second.

Invariant: every share is in [0, 1]; the share lifted is at least the largest zone's share and at most their sum,
whenever the base is not ZERO (rule 6 adds attention blocks outside the zones there).
"""

from __future__ import annotations

import numpy as np

from foqlens.layouts import LayoutPolicy, Zoned
from foqlens.quant import Level

OVER = 0.5  # a question whose layout lifts more than half the network: the zones no longer select anything
TAIL = 0.9  # the quantile beside the median and the maximum


def zone_shares(lifts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Every zone's share of the network: lifts [zones, n_blocks], weights [n_blocks] -> [zones]."""
    return (np.asarray(lifts) > 0).astype(float) @ weights / weights.sum()


def lifted_shares(codes: np.ndarray, floor: Level, weights: np.ndarray) -> np.ndarray:
    """The share of the network every question reads above the base: codes [questions, n_blocks] -> [questions]."""
    return (np.asarray(codes) > int(floor)).astype(float) @ weights / weights.sum()


def level_shares(codes: np.ndarray, weights: np.ndarray) -> list[dict[str, float]]:
    """The share of the network every question reads at every level it uses, by weight: codes [questions, n_blocks]
    -> per question {level name: share}, the shares summing to 1."""
    codes = np.asarray(codes)
    total = weights.sum()
    return [{Level(int(c)).name: float(weights[row == c].sum() / total) for c in np.unique(row)} for row in codes]


def question_coverage(policy: LayoutPolicy, codes: np.ndarray, floor: Level, weights: np.ndarray,
                      read: np.ndarray) -> list[dict]:
    """Every question's share lifted, its share at every level and the bytes read, and - where the policy has zones -
    every zone's share and its own ceiling: codes [questions, n_blocks] are the policy's layouts of questions 0..n-1,
    read [questions] their bytes."""
    lifted = lifted_shares(codes, floor, weights)
    levels = level_shares(codes, weights)
    rows = []
    for i in range(len(codes)):
        row = {"lifted_share": float(lifted[i]), "levels": levels[i], "bytes": int(read[i])}
        if isinstance(policy, Zoned):
            lifts, ceilings = policy.zone_cover(i)
            row |= {"zone_shares": zone_shares(lifts, weights).tolist(), "zone_ceilings": [lv.name for lv in ceilings]}
        rows.append(row)
    return rows


def spread(values: np.ndarray) -> dict:
    """The median, the TAIL quantile and the maximum of a run's values; None for an empty run."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"median": None, "p90": None, "max": None}
    return {"median": float(np.median(values)), "p90": float(np.quantile(values, TAIL)), "max": float(values.max())}


def run_coverage(rows: list[dict], uniform: dict[str, int]) -> dict:
    """Over a run's questions (question_coverage): how many zones, how much of the network lifted and how many lift
    more than OVER of it, and the bytes read against every rung of the uniform ladder (bytes / uniform: below 1 is
    cheaper)."""
    lifted = np.array([r["lifted_share"] for r in rows])
    read = np.array([r["bytes"] for r in rows], dtype=float)
    found = {"lifted_share": spread(lifted), "over_half": int((lifted > OVER).sum()), "questions": len(rows),
             "levels": {lv.name: spread(np.array([r.get("levels", {}).get(lv.name, 0.0) for r in rows]))
                        for lv in Level if any(lv.name in r.get("levels", {}) for r in rows)},
             "bytes": spread(read), "bytes_to_uniform": {rung: spread(read / b) for rung, b in uniform.items()}}
    if rows and "zone_shares" in rows[0]:
        found["zones"] = spread(np.array([len(r["zone_shares"]) for r in rows]))
    return found
