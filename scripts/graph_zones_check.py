"""Zones on the block graph, on masks already on disk - a check of the mechanics at full size, not a result.

Reads raw masks [questions, n_blocks], builds the co-activation metric (M1) and its neighbour graph,
walks the graph (its components, its width), finds the zones of a few questions along it and prints, per
question, how many zones there are, their radii and how many blocks their figures cover at R = f D,
with the time of every step. Block sizes are not known without the model, so every block weighs
one here. Masks of the dead experiments E001-E014 only show that the mechanics runs; they are no evidence.

    uv run python scripts/graph_zones_check.py --masks runs/E001-run1-exploration/step1/e2b/raw/vectors_pooled.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from foqlens import graph_zones as gz
from foqlens.metric import (
    CoactivationMetric,
    geodesic,
    graph_report,
    mutual_nicdm_table,
    nearest,
    neighbour_table,
    nicdm_scale,
    sweep_width,
)

FRONT_AREA = 0.2  # rule 1, R = f D: a fifth of the width of the network
GRAPHS = {"union": neighbour_table, "mutual-nicdm": mutual_nicdm_table}  # how the block graph is built (#4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", type=Path, required=True, help=".npy of raw masks [questions, n_blocks]")
    parser.add_argument("--k", type=int, default=16, help="nearest blocks every block is linked to")
    parser.add_argument("--questions", type=int, default=8, help="questions whose zones are found")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph", choices=sorted(GRAPHS), default="union", help="how the block graph is built")
    args = parser.parse_args()

    clock = time.perf_counter()
    masks = np.load(args.masks)
    metric = CoactivationMetric(masks, args.device)
    table, lengths = GRAPHS[args.graph](metric, args.k)
    along = geodesic(table, lengths)
    table_np = table.cpu().numpy()
    print(f"{masks.shape[0]} questions x {masks.shape[1]} blocks; metric and {args.graph} graph of {args.k} neighbours "
          f"(up to {table_np.shape[1]} a block) in {time.perf_counter() - clock:.1f} s", flush=True)

    clock = time.perf_counter()
    lists = {"raw": nearest(metric, args.k)[0], "nicdm": nearest(metric, args.k, scale=nicdm_scale(metric, args.k))[0]}
    for name, indices in lists.items():
        print(f"k-nearest lists on the {name} distance: {graph_report(table, indices)}", flush=True)
    n_parts, parts = connected_components(gz.block_graph(table_np), directed=False)
    width = sweep_width(along)
    print(f"graph: {n_parts} component(s), the largest {np.bincount(parts).max()} blocks; width along it "
          f"{width:.3f} (M1 diameter {metric.diameter():.3f}) in {time.perf_counter() - clock:.1f} s", flush=True)

    fields = masks - masks.mean(axis=0)  # background subtracted, as Bench.masks' callers do
    weights = np.ones(masks.shape[1])
    for q in range(min(args.questions, len(fields))):
        clock = time.perf_counter()
        zones = gz.find_graph_zones(fields[q], along, table_np, weights)
        front = gz.zone_lifts(zones, gz.FrontReach(FRONT_AREA, width).radii(zones), along)
        covered = int((front.max(axis=0) > 0).sum()) if len(front) else 0
        print(f"q{q}: {len(zones.centers)} zones, radii {np.round(zones.radii, 3).tolist()}; "
              f"blocks covered {covered}; {time.perf_counter() - clock:.2f} s", flush=True)


if __name__ == "__main__":
    main()
