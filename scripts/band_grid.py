"""A grid of field scales over the oracles' fields, as layouts a run can answer (foqlens.maps).

A field scale is the three bounds a demand field is read into rungs at (docs/glossary.md). The bounds derived from the rungs' error ratios are one point
of a space, and whether that point is the cheapest one holding the answer is a question for the answers, not for the
derivation. This writes every point of a grid as its own layout, so scripts/oracle_answers.py reads them all in one
run and the cost of each is known before the model is touched.

A layout's name becomes a folder under `answers/`, so it holds no colon and no slash: `drop_b050070095` is the
oracle and its scale in hundredths.

    uv run python scripts/band_grid.py --maps runs/E006-.../maps/maps-...npz --fields runs/E006-.../precision-...npz \
        --only runs/E006-.../sample/search.json --out runs/E006-.../grid
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from foqlens import maps, runlog
from foqlens.oracle_overlay import block_group_ids
from foqlens.quant import Level
from foqlens.small_corpus import StoredMasks

LOG = logging.getLogger("foqlens.band_grid")
BITS = {int(Level.D2): 2.0, int(Level.D4): 4.0, int(Level.D6): 6.0, int(Level.D8): 8.0}
# a scale whose middle rungs hold less than this is no scale: the map is binary and the point says nothing
LEAST_MIDDLE = 0.04


def band_name(oracle: str, bounds: tuple[float, float, float]) -> str:
    """The name a scale is answered under: the oracle and its bounds in hundredths, safe as a path."""
    return "%s_b%03d%03d%03d" % (oracle, *[round(b * 100) for b in bounds])


def named_bands(scales: list[str]) -> list[tuple[float, float, float]]:
    """Scales written out, `0.50,0.95,1.00` each, instead of a lattice - what a search already settled.

    The top bound may equal 1: the top rung is then never taken, and the map lives on the base, D4 and D6.
    """
    bands = [tuple(float(x) for x in scale.split(",")) for scale in scales]
    if any(len(band) != 3 or not band[0] < band[1] <= band[2] for band in bands):
        raise ValueError("a scale is three rising bounds, as 0.50,0.95,1.00")
    return bands


def read_band(field: np.ndarray, bounds: tuple[float, float, float]) -> np.ndarray:
    """A field read into rungs at a field scale: below the first bound the base, then D4, D6 and the top."""
    b4, b6, b8 = bounds
    out = np.full(field.shape, int(Level.D2), dtype=np.uint8)
    out[field >= b4] = int(Level.D4)
    out[field >= b6] = int(Level.D6)
    out[field >= b8] = int(Level.D8)
    return out


def group_weights(masks: Path, names: list[str]) -> np.ndarray:
    kept = StoredMasks.read(str(masks))
    return np.bincount(block_group_ids(kept.block_layer, kept.block_kind, names),
                       weights=kept.block_weights, minlength=len(names))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=Path, required=True, help="the block of maps the fields are read from")
    parser.add_argument("--fields", type=Path, required=True, help="the precision fields, for the run's own keys")
    parser.add_argument("--masks", type=Path, required=True, help="a kept masks file, for the weights of the groups")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", type=Path, default=None, help="a json list of [corpus, id]: these questions alone")
    parser.add_argument("--sources", nargs="+", default=["drop", "lift_per_weight", "pooled", "error_energy",
                                                         "product"])
    parser.add_argument("--base-d4", nargs="+", type=float, default=[0.35, 0.50, 0.65])
    parser.add_argument("--d4-d6", nargs="+", type=float, default=[0.60, 0.70, 0.85])
    parser.add_argument("--d6-d8", nargs="+", type=float, default=[0.88, 0.95, 0.99])
    # named scales instead of the whole lattice: what a search on the micro corpus already settled goes to the
    # sample as its own few points, and the run costs what those points cost
    parser.add_argument("--bands", nargs="+", default=None, metavar="B4,B6,B8",
                        help="explicit scales, each three bounds - the lattice arguments are then ignored")
    args = parser.parse_args(argv)
    runlog.setup(args.out / "logs" / "band_grid.jsonl", {"run": "band_grid", "script": "band_grid"})

    block = maps.Maps.read(args.maps)
    names = [str(o) for o in block.oracles]
    groups = [str(x) for x in block.groups]
    weights = group_weights(args.masks, groups)
    wanted = ({tuple(p) for p in json.loads(args.only.read_text(encoding="utf-8"))} if args.only else None)
    keep = np.array([wanted is None or (str(c), str(i)) in wanted for c, i in zip(block.corpus, block.ids)])

    bands = named_bands(args.bands) if args.bands else \
        [(b4, b6, b8) for b4 in args.base_d4 for b6 in args.d4_d6 for b8 in args.d6_d8 if b4 < b6 < b8]
    layouts, cost = {}, {}
    for source in args.sources:
        field = block.demand[names.index(source)][keep]
        for bounds in bands:
            levels = read_band(field, bounds)
            middle = ((levels == int(Level.D4)) | (levels == int(Level.D6))).mean()
            if middle < LEAST_MIDDLE:
                continue
            name = band_name(source, bounds)
            layouts[name] = levels
            bits = np.vectorize(BITS.get)(levels)
            cost[name] = round(float((bits * weights).sum(axis=1).mean() / (8 * weights.sum())), 4)

    kept = dict(np.load(args.fields, allow_pickle=True))
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(keep) else v)
           for k, v in kept.items()
           if not k.startswith(("field_", "levels_", "eps_", "nll_", "target_", "batches_"))}
    out["sources"] = np.array(list(layouts))
    for name, levels in layouts.items():
        out["levels_" + name] = levels
    args.out.mkdir(parents=True, exist_ok=True)
    written = args.out / args.fields.name
    np.savez_compressed(written, **out)
    (args.out / "cost.json").write_text(json.dumps(cost, indent=1), encoding="utf-8")
    LOG.info("written", extra={"path": str(written), "bands": len(layouts), "questions": int(keep.sum())})
    return written


if __name__ == "__main__":
    main()
