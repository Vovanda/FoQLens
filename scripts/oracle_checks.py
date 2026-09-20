"""The recorded checks of the new oracles (plan ideal-models, foqlens.oracle_checks), read from their files - no model:

    gradient: the answer-gradient oracle's NLL at D8 against the oracle by trying's everything-D8 end, its groups'
              ranks against the lift, and the quant gap's signed group sums against the measured drop
    energy:   the D4-D8 error energy against the D2-D8 on the same questions
    ends:     per topic, where every block low already answers within the tolerance of every block high

    uv run python scripts/oracle_checks.py --group-oracle runs/oracles/e2b-it/group-oracle-d2d8-....npz \
        gradient --masks runs/masks/e2b-it/answer_gradient-d8-....npz --nll runs/masks/e2b-it/answer_nll-d8-....npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts
from foqlens.oracle_checks import end_gaps, matched, nll_agreement, rank_agreement, rung_ratio
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.small_corpus import StoredMasks


def keys(corpus: np.ndarray, ids: np.ndarray) -> list[tuple[str, str]]:
    return list(zip(corpus.tolist(), ids.tolist()))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group-oracle", default=None,
                        help="the oracle by trying over the D2 base (a glob joins parts); every check but energy")
    parser.add_argument("--out", type=Path, default=None, help="where the JSON goes; printed either way")
    sub = parser.add_subparsers(dest="check", required=True)
    gradient = sub.add_parser("gradient")
    gradient.add_argument("--masks", required=True)
    gradient.add_argument("--nll", required=True)
    gradient.add_argument("--gap", default=None, help="the quant_gap masks of the same pass")
    energy = sub.add_parser("energy")
    energy.add_argument("--upper", required=True)
    energy.add_argument("--lower", required=True)
    ends = sub.add_parser("ends")
    ends.add_argument("--tolerance", type=float, required=True, help="nats of the answer's NLL above every block high")
    args = parser.parse_args(argv)
    if args.check != "energy":
        if args.group_oracle is None:
            parser.error(f"{args.check} reads --group-oracle")
        # the fields this check reads, named so that shards of two versions of the oracle still join
        trying = read_npz_parts(args.group_oracle, RUN_FIELDS,
                                only={"corpus", "ids", "ends", "lift", "drop"})
        trying_keys = keys(trying["corpus"], trying["ids"])

    if args.check == "gradient":
        kept, nll = StoredMasks.read(args.masks), read_npz_parts(args.nll)
        a, b = matched(keys(nll["corpus"], nll["ids"]), trying_keys)
        found = {"nll_vs_everything_high": nll_agreement(nll["nll"][a], trying["ends"][b, 1]),
                 "without_mask": int(np.isnan(kept.masks).any(axis=1).sum()), "questions": int(len(kept.ids))}
        m, t = matched(keys(kept.corpus, kept.ids), trying_keys)
        has = ~np.isnan(kept.masks[m]).any(axis=1) & ~np.isnan(trying["lift"][t]).any(axis=1)
        groups = to_groups(np.abs(kept.masks[m][has]),
                           block_group_ids(kept.block_layer, kept.block_kind, trying["groups"].tolist()),
                           len(trying["groups"]))
        found["groups_vs_lift"] = rank_agreement(groups, trying["lift"][t][has])
        if args.gap:
            # the first order of the drop: the signed sum over a group's blocks against the measured drop
            gap = StoredMasks.read(args.gap)
            g, t = matched(keys(gap.corpus, gap.ids), trying_keys)
            has = ~np.isnan(gap.masks[g]).any(axis=1) & ~np.isnan(trying["drop"][t]).any(axis=1)
            summed = to_groups(gap.masks[g][has], block_group_ids(gap.block_layer, gap.block_kind,
                                                                   trying["groups"].tolist()), len(trying["groups"]))
            measured = trying["drop"][t][has]
            found["quant_gap_vs_drop"] = rank_agreement(summed, measured) | {
                "sum_ratio_median": float(np.median(summed.sum(axis=1) / measured.sum(axis=1)))}
    elif args.check == "energy":
        upper, lower = StoredMasks.read(args.upper), StoredMasks.read(args.lower)
        a, b = matched(keys(upper.corpus, upper.ids), keys(lower.corpus, lower.ids))
        has = ~np.isnan(upper.masks[a]).any(axis=1) & ~np.isnan(lower.masks[b]).any(axis=1)
        found = {"questions": int(has.sum()), **rung_ratio(upper.masks[a][has], lower.masks[b][has])}
    else:
        found = end_gaps(trying["ends"], trying["corpus"], args.tolerance)
    text = json.dumps(found, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return found


if __name__ == "__main__":
    main()
