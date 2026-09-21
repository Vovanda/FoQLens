"""The block of maps of a run, built from the oracles' fields (foqlens.maps, foqlens.oracle_overlay).

Every oracle's values become one comparable field - its value over the question's threshold, folded into 0 ... 1 -
and every tolerance reads that field into its own map. Beside the oracles go their overlays, and beside every field
the same field with the network's own part taken out: what every question of the corpus shares is the average field,
and what is left of a question after it is the question's own demand.

The block is one file: the questions and the groups listed once, every map at its place, named `<oracle>@k<k>`.
Rebuilding it is this one command, so nothing of it lives in a shell line or in anybody's memory.

    uv run python scripts/build_maps.py --fields runs/E006-.../precision-fields-...npz --out runs/E006-.../maps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from foqlens import maps, runlog
from foqlens.oracle_overlay import demand, overlay
from foqlens.quant import Level

LOG = logging.getLogger("foqlens.build_maps")
OVERLAYS = ("product", "mean", "least")
NET_OUT = "-net_out"  # the suffix of a field the network's own part was taken out of


def network_part(field: np.ndarray) -> np.ndarray:
    """What every question shares of one oracle's field: [questions, groups] -> [groups]."""
    return field.mean(axis=0)


def question_part(field: np.ndarray) -> np.ndarray:
    """A field with the network's own part taken out, kept in 0 ... 1."""
    return np.clip(field - network_part(field), 0.0, 1.0)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", type=Path, required=True, help="the precision fields of the run")
    parser.add_argument("--out", type=Path, required=True, help="the folder the block is written to")
    parser.add_argument("--sources", nargs="+", default=None, help="the oracles taken; every one by default")
    parser.add_argument("--ks", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0],
                        help="the tolerances a field is read at")
    parser.add_argument("--overlays", action="store_true", default=True, help="lay the oracles over one another")
    parser.add_argument("--no-overlays", dest="overlays", action="store_false")
    args = parser.parse_args(argv)
    runlog.setup(args.out / "logs" / f"build_maps-{args.fields.stem}.jsonl",
                 {"run": args.fields.stem, "script": "build_maps"})
    kept = np.load(args.fields, allow_pickle=True)
    sources = args.sources or [str(s) for s in kept["sources"] if f"field_{s}" in kept and f"eps_{s}" in kept]
    fields = [demand(kept[f"field_{s}"], kept[f"eps_{s}"]) for s in sources]
    eps = [kept[f"eps_{s}"] for s in sources]
    if args.overlays:
        for how in OVERLAYS:
            sources.append(how)
            fields.append(overlay(fields[:len(eps)], how))
            eps.append(np.full(len(fields[0]), np.nan))
    names = list(sources) + [s + NET_OUT for s in sources]
    every = fields + [question_part(f) for f in fields]
    codes = [int(Level[str(r)]) for r in kept["rungs"]]
    levels = np.stack([np.stack([maps.read_map(f, kept["ratios"], k, codes, int(Level.D8)) for k in args.ks])
                       for f in every])
    block = maps.Maps(corpus=kept["corpus"], ids=kept["ids"], groups=kept["groups"], oracles=np.array(names),
                      ks=np.array(args.ks), demand=np.stack(every), levels=levels, ratios=kept["ratios"],
                      eps=np.stack(eps + eps))
    written = block.save(args.out / f"maps-{args.fields.stem.replace('precision-fields-', '')}.npz")
    np.savez_compressed(args.out / f"network-part-{args.fields.stem.replace('precision-fields-', '')}.npz",
                        oracles=np.array(sources), groups=kept["groups"],
                        network=np.stack([network_part(f) for f in fields]))
    LOG.info("written", extra={"path": str(written), "oracles": len(names), "layouts": len(block.flat())})
    return written


if __name__ == "__main__":
    main()
