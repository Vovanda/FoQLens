"""The overlay of the chains every oracle gives (config.ChainOverlay; foqlens.oracle_overlay): per oracle the size of
its chain and of its switched-off set with intervals, the share of MLP in the chain, the antinodes and nodes, the
spread of each band of layers within and between topics; between the oracles the Jaccard of their chains and of their
switched-off sets on the same questions - what the oracles share and where they differ (Volodya 20.09 02:08). From the
files, never the model.

    uv run python scripts/chain_overlay.py --config configs/chain-overlay.toml --out runs/oracles/e2b-it/round1
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, runlog
from foqlens.group_oracle import RUN_FIELDS, group_order
from foqlens.io import read_npz_parts, write_json
from foqlens.oracle_overlay import antinodes, band_spread, bootstrap, chain_sets, frequency, lenses, pair_jaccard
from foqlens.quant import Level

LOG = logging.getLogger("foqlens.chain_overlay")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"chain_overlay-{run}.jsonl", {"run": run, "script": "chain_overlay"})
    check = config.read(args.config, config.ChainOverlay)
    chains = read_npz_parts(check.chains, RUN_FIELDS | {"sources"})
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS)
    names = chains["groups"].tolist()
    g = len(names)
    layers = np.array([int(n.split(".")[0]) for n in names])
    mlp = np.array([n.endswith(".mlp") for n in names])
    at = {k: i for i, k in enumerate(zip(trying["corpus"].tolist(), trying["ids"].tolist()))}
    rows = np.array([at[k] for k in zip(chains["corpus"].tolist(), chains["ids"].tolist())])
    topics = chains["corpus"]
    sets = {"lift": chain_sets(group_order(trying["lift"][rows]), trying["minimal"][rows], trying["zeroed"][rows], g)}
    for source in chains["sources"].tolist():
        sets[source] = chain_sets(chains[f"order_{source}"], chains[f"minimal_{source}"], chains[f"zeroed_{source}"], g)
    draws = (check.bootstrap_draws, check.seed, check.level)
    found = {"questions": int(len(rows)), "groups": g, "oracles": {}, "between": {}}
    for name, (chain, off) in sets.items():
        size, gone = chain.sum(axis=1).astype(float), off.sum(axis=1).astype(float)
        freq, freq_off = frequency(chain), frequency(off)
        found["oracles"][name] = {
            "chain_median": float(np.median(size)), "chain_interval": bootstrap(size, np.median, *draws),
            "chain_p90": float(np.percentile(size, 90)), "off_median": float(np.median(gone)),
            "off_interval": bootstrap(gone, np.median, *draws),
            "mlp_share_of_chain": float(chain[:, mlp].sum() / max(chain.sum(), 1)),
            "antinodes": [f"{names[i]} {freq[i]:.2f}" for i in antinodes(freq, check.antinode_share)],
            "nodes": [f"{names[i]} {freq_off[i]:.2f}" for i in antinodes(freq_off, check.antinode_share)],
            "bands_chain": band_spread(chain, layers, [tuple(b) for b in check.bands], topics),
            "bands_off": band_spread(off, layers, [tuple(b) for b in check.bands], topics),
        }
        LOG.info("%s: chain median %g (p90 %g), off median %g, MLP %.2f of the chain, %d antinodes, %d nodes", name,
                 np.median(size), np.percentile(size, 90), np.median(gone),
                 found["oracles"][name]["mlp_share_of_chain"], len(found["oracles"][name]["antinodes"]),
                 len(found["oracles"][name]["nodes"]), extra={"oracle": name})
    if check.grades:  # every chain's map in every level: how many groups each level holds, how many lenses it has
        graded = read_npz_parts(check.grades, RUN_FIELDS | {"sources", "rungs"})
        base = int(Level[str(graded["low"]).upper()])
        shown = [Level[str(graded["high"]).upper()], *(Level[r.upper()] for r in graded["rungs"].tolist()[::-1]),
                 Level[str(graded["low"]).upper()], Level.ZERO]
        for name in graded["sources"].tolist():
            maps = graded[f"grades_{name}"]
            maps = maps[(maps != 255).all(axis=1)]
            counts = {lv.name: float(np.median((maps == int(lv)).sum(axis=1))) for lv in shown}
            count_lenses = np.array([len(lenses(m, layers, base)) for m in maps], dtype=float)
            found["oracles"].setdefault(name, {})["map"] = {
                "questions": int(len(maps)), "median_groups": counts,
                "lenses_median": float(np.median(count_lenses)) if len(maps) else None,
                "lenses_histogram": {int(k): int((count_lenses == k).sum()) for k in np.unique(count_lenses)}}
            LOG.info("%s map: median groups %s, lenses median %s", name, counts,
                     found["oracles"][name]["map"]["lenses_median"], extra={"oracle": name})
    order = list(sets)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            chain_j, off_j = pair_jaccard(sets[a][0], sets[b][0]), pair_jaccard(sets[a][1], sets[b][1])
            found["between"][f"{a} ~ {b}"] = {"chain_jaccard_mean": float(np.nanmean(chain_j)),
                                              "off_jaccard_mean": float(np.nanmean(off_j))}
            LOG.info("%s ~ %s: chains Jaccard %.2f, switched-off Jaccard %.2f", a, b, np.nanmean(chain_j),
                     np.nanmean(off_j), extra={"pair": f"{a} ~ {b}"})
    target = args.out / f"{args.config.stem}.json"
    write_json(target, found)
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
