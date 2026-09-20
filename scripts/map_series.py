"""A series of maps from the fields already read (foqlens.field_bridge.to_levels): the same oracle field cut into
rungs by different stops, so that the rungs above the base are made dear and the map keeps only what it must.

The stops are the outer edge of every rung's ring, the top rung first: even thirds are what the bench read until now,
and a narrower first stop gives the top rung to the peak of the field alone. Nothing here reads the model - the fields
are on disk, and what comes out is one .npz of levels per stop, in the shape `oracle_answers` reads.

    uv run python scripts/map_series.py --fields "runs/oracles/e2b-it/precision-fields-*-shard*of5.npz" \
        --out runs/oracles/e2b-it/map-series.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens.field_bridge import EVEN_STOPS, bits_of, to_levels
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, save_npz_atomic
from foqlens.oracle_overlay import block_group_ids
from foqlens.quant import Level
from foqlens.small_corpus import StoredMasks

# the series (Volodya 20.09): every rung above the base made dearer by giving it a narrower ring of the field

ORACLES = ("lift", "drop", "answer_gradient", "error_energy", "pooled")


def by_peak(field: np.ndarray) -> np.ndarray:
    """A question's field as a lift in 0 ... 1 against its own peak: linear, so a group an order below the peak is at
    0.1 and every rung above the base falls to the few groups at the very top."""
    positive = np.maximum(np.asarray(field, dtype=float), 0.0)
    peak = positive.max(axis=-1, keepdims=True)
    return np.divide(positive, peak, out=np.zeros_like(positive), where=peak > 0)


def by_rank(field: np.ndarray) -> np.ndarray:
    """A question's field as a lift in 0 ... 1 by rank: a group's lift is the share of the question's groups at or
    below it, so the scale's intervals name shares of the network - the top tenth reads D6, the next tenth D4."""
    order = np.argsort(np.argsort(np.asarray(field, dtype=float), axis=-1), axis=-1)
    return (order + 1) / field.shape[-1]


def by_decade(field: np.ndarray) -> np.ndarray:
    """A question's field as a lift in 0 ... 1 in the logarithm, over `DECADES` orders under the peak: importances lie
    orders apart, and a linear scale reads all of them as the same nothing."""
    positive = np.maximum(np.asarray(field, dtype=float), 0.0)
    peak = positive.max(axis=-1, keepdims=True)
    under = np.log10(np.divide(positive, peak, out=np.full_like(positive, 10.0 ** -DECADES), where=positive > 0))
    return np.clip(1.0 + under / DECADES, 0.0, 1.0)


DECADES = 3.0  # orders of magnitude under a question's peak that the logarithmic scale spreads over 1 ... 0
SCALES = {"peak": by_peak, "rank": by_rank, "decade": by_decade}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True, help="the precision fields' files (a glob joins shards)")
    parser.add_argument("--weights", default="runs/masks/e2b-it/error_energy-file-bartowski-Q2_K-small-corpus.npz")
    parser.add_argument("--oracles", nargs="*", default=list(ORACLES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources"},
                       only={"ids", "corpus", "sources", "tolerance"} | {f"field_{s}" for s in args.oracles})
    names = z["groups"].tolist()
    kept = StoredMasks.read(args.weights)
    weights = np.bincount(block_group_ids(kept.block_layer, kept.block_kind, names), weights=kept.block_weights,
                          minlength=len(names))
    out = {"groups": z["groups"], "ids": z["ids"], "corpus": z["corpus"], "tolerance": z["tolerance"]}
    sources = []
    print(f"{'map':34s} {'bits/weight':>11s} {'D2':>6s} {'D4':>6s} {'D6':>6s} {'D8':>6s}")
    for oracle in args.oracles:
        for name, scale in SCALES.items():
            levels = to_levels(scale(z[f"field_{oracle}"]), 0.0, EVEN_STOPS)
            source = f"{oracle}-{name}"
            out[f"levels_{source}"] = levels
            sources.append(source)
            shares = [float((levels == int(lv)).mean()) for lv in (Level.D2, Level.D4, Level.D6, Level.D8)]
            print(f"{source:34s} {float(bits_of(levels, weights).mean()):11.2f} "
                  + " ".join(f"{s:6.2f}" for s in shares))
    out["sources"] = np.array(sources)
    save_npz_atomic(args.out, **out)
    print(f"\n{args.out}")
    return args.out


if __name__ == "__main__":
    main()
