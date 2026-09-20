"""How deep a pass must read before the rest of the map is predictable (plan ideal-models, the layer-wise regulator):
for every cut k, a ridge from the address over the groups of the layers before k to the precision map of the groups
from k on, fitted on one half of the questions and read on the other. No model is loaded.

Two designs are judged by the same numbers: deciding everything up front needs one k where the rest is already
predictable; deciding layer by layer needs every k to predict the layer just after it.

    uv run python scripts/read_depth_check.py --fields "runs/oracles/e2b-it/precision-fields-*shard1of5.npz" \
        --out runs/oracles/e2b-it/read-depth-shard1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from foqlens.field_analysis import against_background
from foqlens.field_bridge import lift_of
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, write_json
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.projection import Projection
from foqlens.small_corpus import StoredMasks

RELATIVE_RIDGE = 0.1
RIDGE_FLOOR = 1e-6  # of the signals' mean square: what keeps the solve from failing on a handful of questions
NOT_READ = 255


def agreement(pred: np.ndarray, true: np.ndarray, background: np.ndarray) -> dict:
    """How well a prediction ranks the groups of every question, whole and on the excess over the background."""
    whole = [spearmanr(p, t).statistic for p, t in zip(pred, true) if np.ptp(p) > 0 and np.ptp(t) > 0]
    excess = [spearmanr(p - background, t - background).statistic for p, t in zip(pred, true)
              if np.ptp(p) > 0 and np.ptp(t) > 0]
    return {"questions": len(whole), "rank": float(np.median(whole)) if whole else None,
            "rank_on_excess": float(np.nanmedian(excess)) if excess else None}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--target", default="common", help="the map read as the truth (a field's name)")
    parser.add_argument("--address", default="runs/masks/e2b-it/hybrid-file-bartowski-Q2_K-small-corpus.npz")
    parser.add_argument("--cuts", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7],
                        help="the shares of the depth a pass has read when it decides")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources", "ratios", "rungs"})
    names = z["groups"].tolist()
    layers = np.array([int(n.split(".")[0]) for n in names])
    levels = z[f"levels_{args.target}"]
    has = (levels != NOT_READ).all(axis=1)
    maps = lift_of(levels[has])
    keys = [k for k, ok in zip(zip(z["corpus"].tolist(), z["ids"].tolist()), has) if ok]
    masks = StoredMasks.read(args.address)
    address = to_groups(np.abs(masks.rows_of(keys)),
                        block_group_ids(masks.block_layer, masks.block_kind, names), len(names))
    half = len(maps) // 2
    found = {"questions": len(maps), "target": args.target, "cuts": {}}
    for cut in args.cuts:
        k = max(1, round(cut * (layers.max() + 1)))
        read, rest = layers < k, layers >= k
        if not rest.any():
            continue
        x, y = address[:half][:, read], maps[:half][:, rest]
        # with few questions the addresses are nearly collinear: the ridge keeps the solve from failing
        alpha = max(RELATIVE_RIDGE * float(x.var(axis=0).mean()) * len(x), RIDGE_FLOOR * float((x ** 2).mean()))
        fitted = Projection.fit(x, y, alpha)
        pred = np.asarray(fitted.apply(address[half:][:, read]))
        true = maps[half:][:, rest]
        background = maps[:half][:, rest].mean(axis=0)
        # the next layer alone: what a layer-wise rule has to get right at this cut. A layer of the bench holds two
        # groups, attention and mlp, and a rank over two of anything is +1 or -1: this one is read against the
        # background - the error of the prediction over the error of answering with the mean of the questions
        just_after = layers == k
        after = against_background(pred[:, just_after[rest]], maps[half:][:, just_after],
                                   maps[:half][:, just_after].mean(axis=0))
        found["cuts"][f"{cut:g}"] = {"layers_read": int(k), "groups_read": int(read.sum()),
                                     "groups_after": int(just_after.sum()),
                                     "rest": agreement(pred, true, background), "next_layer": after}
    write_json(args.out, found)
    for cut, row in found["cuts"].items():
        print(f"read {row['layers_read']:>2} layers ({cut}): the rest rank {row['rest']['rank']:+.3f} "
              f"on the excess {row['rest']['rank_on_excess']:+.3f}; the next layer over the background "
              f"{row['next_layer']['ratio']:.3f} ({row['groups_after']} groups)")
    return args.out


if __name__ == "__main__":
    main()
