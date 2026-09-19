"""The overlay of the oracles on the small corpus's laid-out questions the model knows (foqlens.oracle_overlay;
config.OverlayCheck): every oracle at the granularity of groups, their agreement per question, the static share of each,
and the minimal masks - their size by topic with intervals, and whether a topic shares them.

    uv run python scripts/oracle_overlay.py --config configs/oracle-overlay.toml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, runlog
from foqlens.io import write_json
from foqlens.oracle_overlay import bootstrap, contrast, group_ranks, jaccard_within_between, static_share, to_groups
from foqlens.sensitivity import rank_correlation, top_overlap
from foqlens.small_corpus import StoredMasks

LOG = logging.getLogger("foqlens.oracle_overlay")


def block_group_ids(masks: StoredMasks, names: list[str]) -> np.ndarray:
    """Every block's group in the group oracle's order, from its layer and module kind."""
    label = [f"{layer}.{'attention' if kind.startswith('self_attn.') else 'mlp'}"
             for layer, kind in zip(masks.block_layer.tolist(), masks.block_kind.tolist())]
    return np.array([names.index(g) for g in label])


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"oracle_overlay-{run}.jsonl", {"run": run, "script": "oracle_overlay"})
    check = config.read(args.config, config.OverlayCheck)
    trying = np.load(check.group_oracle)
    names = trying["groups"].tolist()
    keys = list(zip(trying["corpus"].tolist(), trying["ids"].tolist()))  # ids repeat across corpora
    known = ~trying["unknown"] & ~np.isnan(trying["lift"]).any(axis=1)
    scores = {"lift": trying["lift"][known], "drop": trying["drop"][known]}
    for name, path in check.block_oracles.items():
        kept = StoredMasks.load(Path(path))
        rows = {q: i for i, q in enumerate(zip(kept.corpus.tolist(), kept.ids.tolist())) if kept.laid[i]}
        picked = kept.masks[[rows[q] for q, k in zip(keys, known) if k]]
        scores[name] = to_groups(np.abs(picked), block_group_ids(kept, names), len(names))
    ranks = {name: group_ranks(s) for name, s in scores.items()}
    found = {"questions": int(known.sum()), "groups": len(names), "static_share": {}, "agreement": {}}
    for name, r in ranks.items():
        found["static_share"][name] = {"value": static_share(r), "interval": bootstrap(
            r, static_share, check.bootstrap_draws, check.seed, check.level)}
        LOG.info("%s: static share %.3f", name, found["static_share"][name]["value"], extra={"oracle": name})
    # a block oracle's contrast has mean 1 in every group by construction: compared, never given a static share
    compared = scores | {f"{name}/contrast": contrast(scores[name]) for name in check.block_oracles}
    order = list(compared)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            rho = rank_correlation(compared[a], compared[b])
            overlap = top_overlap(compared[a], compared[b], check.top_share)
            found["agreement"][f"{a} ~ {b}"] = {"rho_median": float(np.median(rho)),
                                                "top_overlap_median": float(np.median(overlap))}
            LOG.info("%s ~ %s: rho median %.3f, top %g overlap %.3f", a, b, float(np.median(rho)), check.top_share,
                     float(np.median(overlap)), extra={"pair": f"{a} ~ {b}"})
    minimal = trying["minimal"][known]
    topics = trying["corpus"][known]
    order_lift = np.argsort(-scores["lift"], axis=1, kind="stable")
    masks = np.zeros_like(scores["lift"], dtype=bool)
    for q, k in enumerate(minimal):
        masks[q, order_lift[q, :k]] = True
    found["minimal"] = {str(t): {"median": float(np.median(minimal[topics == t])), "interval": bootstrap(
        minimal[topics == t].astype(float), np.median, check.bootstrap_draws, check.seed, check.level)}
        for t in np.unique(topics)}
    found["minimal_all"] = {"median": float(np.median(minimal)), "share_of_groups": float(np.median(minimal) / len(names))}
    within, between = jaccard_within_between(masks, topics)
    found["minimal_jaccard"] = {"within_topic": within, "between_topics": between}
    LOG.info("minimal mask median %d of %d groups; Jaccard within a topic %.3f, between %.3f",
             int(np.median(minimal)), len(names), within, between)
    target = args.out / f"{args.config.stem}.json"
    write_json(target, found)
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
