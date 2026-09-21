"""A file of precision fields with every question's map carried onto another question of the same cost
(foqlens.carry): the control of H3.1 and H3.4 - is a map this question's map, or would any map of that price do.

The questions are paired by what their maps cost, nearest first, and the two of a pair exchange their layouts. The
questions, the corpora and the ids stay as they were, so the answers are still asked of the same questions and only
the precision they are given moves.

    uv run python scripts/carry_fields.py --fields runs/E006-.../precision-fields-...npz --source lift_per_weight
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from foqlens import carry, runlog
from foqlens.io import save_npz_atomic
from foqlens.oracle_overlay import block_group_ids
from foqlens.small_corpus import StoredMasks

LOG = logging.getLogger("foqlens.carry_fields")
NOT_READ = 255  # a question whose field was never read: its layout is left where it is


def cost_of(levels: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """What every question's map costs: the mean rung over its groups, by the weights of the groups where they are
    given. The scale does not matter - only the order, which decides who is paired with whom."""
    kept = np.where(levels == NOT_READ, np.nan, levels).astype(float)
    return np.nansum(kept * (weights if weights is not None else 1.0), axis=1) / (
        np.nansum(np.where(np.isnan(kept), np.nan, weights if weights is not None else 1.0), axis=1))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", type=Path, required=True, help="the precision fields to carry")
    parser.add_argument("--out", type=Path, required=True, help="the file written with the maps carried")
    parser.add_argument("--how", choices=("other-question", "elsewhere", "random"), default="other-question",
                        help="other-question: the map of the neighbour by cost (H3.1); "
                             "elsewhere: this map carried elsewhere in the network at the same cost (H3.4); "
                             "random: its rungs dealt over the groups at that same cost (H3.3)")
    parser.add_argument("--masks", type=Path, default=None,
                        help="a kept masks file the weights of the groups are read from; --how elsewhere needs it")
    parser.add_argument("--shift", type=int, default=1, help="groups a class of equal weight is rolled by")
    parser.add_argument("--seed", type=int, default=0, help="the deal of --how random")
    args = parser.parse_args(argv)
    run = Path(args.out).stem
    runlog.setup(args.out.parent / "logs" / f"carry_fields-{run}.jsonl", {"run": run, "script": "carry_fields"})
    kept = dict(np.load(args.fields, allow_pickle=True))
    sources = [str(s) for s in kept["sources"]] + (["reference"] if "levels_reference" in kept else [])
    weights = None
    if args.how in ("elsewhere", "random"):
        if args.masks is None:
            raise SystemExit("--how elsewhere and --how random need --masks: the weights of the groups decide what may move where")
        names = [str(g) for g in kept["groups"]]
        masks = StoredMasks.read(str(args.masks))
        weights = np.bincount(block_group_ids(masks.block_layer, masks.block_kind, names),
                              weights=masks.block_weights, minlength=len(names))
    for source in sources:
        levels = kept[f"levels_{source}"]
        if args.how in ("elsewhere", "random"):
            kept[f"levels_{source}"] = (carry.moved_elsewhere(levels, weights, args.shift) if args.how == "elsewhere"
                                        else carry.shuffled_elsewhere(levels, weights, args.seed))
            LOG.info("carried", extra={"source": source, "how": args.how, "questions": len(levels)})
        else:
            pairs = carry.pairs_by_cost(cost_of(levels))
            kept[f"levels_{source}"] = carry.swap_rows(levels, pairs)
            LOG.info("carried", extra={"source": source, "pairs": len(pairs), "questions": len(levels)})
    save_npz_atomic(args.out, **kept)
    LOG.info("written %s", args.out)
    return args.out


if __name__ == "__main__":
    main()
