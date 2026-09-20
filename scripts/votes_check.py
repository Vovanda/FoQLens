"""The voting rule (plan ideal-models, Volodya 20.09): a block votes for the blocks it feeds, the weight of a vote is
the static coupling read from the model's weights (foqlens.coupling), its strength is the block's activity now. The
votes of a question are its score for what lies ahead; this reads whether that score picks the groups a precision map
raises - no generation, no training.

The coupling table is built once per model and kept beside the masks; `--layers` builds it over the first layers only,
which is what a pass has anyway and what fits in host memory beside a run.

    uv run python scripts/votes_check.py --fields "runs/oracles/e2b-it/precision-fields-*shard1of5.npz" \
        --table runs/masks/e2b-it/signal-path.npz --out runs/oracles/e2b-it/votes-shard1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from foqlens import model as fm
from foqlens import refocustensors
from foqlens.coupling import signal_path_table, votes
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, save_npz_atomic, write_json
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.pipeline import Bench
from foqlens.precision_field import values
from foqlens.small_corpus import StoredMasks

NOT_READ = 255


def table_of(path: Path, base: str, layers: int | None, gpu_share: float) -> tuple[np.ndarray, np.ndarray]:
    """The signal path's neighbour table and edge lengths, read from `path` or built and kept there."""
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["near"], z["length"]
    bench = Bench.load(fm.E2B_IT, gpu_share=gpu_share, directory=refocustensors.model_directory(fm.E2B_IT, base))
    weights = {name: module.weight.data for name, module in bench.ctl.modules.items()
               if layers is None or int(name.split(".")[1]) < layers}
    near, length = signal_path_table(weights, bench.n_heads)
    save_npz_atomic(path, near=near.cpu().numpy(), length=length.cpu().numpy())
    return near.cpu().numpy(), length.cpu().numpy()


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--target", default="common", help="the map the votes are read against")
    parser.add_argument("--address", default="runs/masks/e2b-it/hybrid-file-bartowski-Q2_K-small-corpus.npz")
    parser.add_argument("--table", type=Path, default=Path("runs/masks/e2b-it/signal-path.npz"))
    parser.add_argument("--base", default="bartowski-Q2_K")
    parser.add_argument("--layers", type=int, default=None, help="build the table over the first N layers only")
    parser.add_argument("--spread", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                        help="how wide a voter looks - the share of its edges its vote is given to")
    parser.add_argument("--gpu-share", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources", "ratios", "rungs"})
    names = z["groups"].tolist()
    levels = z[f"levels_{args.target}"]
    has = (levels != NOT_READ).all(axis=1)
    maps = values(levels[has])
    keys = [k for k, ok in zip(zip(z["corpus"].tolist(), z["ids"].tolist()), has) if ok]
    masks = StoredMasks.read(args.address)
    blocks = np.abs(masks.rows_of(keys))
    near, length = table_of(args.table, args.base, args.layers, args.gpu_share)
    ids = block_group_ids(masks.block_layer, masks.block_kind, names)
    plain = to_groups(blocks, ids, len(names))
    scored = {f"votes-{s:g}": to_groups(votes(blocks, near, length, s), ids, len(names)) for s in args.spread}
    background = {tag: x.mean(axis=0) for tag, x in ({"address": plain} | scored).items()}
    background["map"] = maps.mean(axis=0)
    found = {"questions": len(maps), "target": args.target, "blocks": int(blocks.shape[1])}
    for tag, x in ({"address": plain} | scored).items():
        whole = [spearmanr(a, m).statistic for a, m in zip(x, maps) if np.ptp(a) > 0 and np.ptp(m) > 0]
        excess = [spearmanr(a - background[tag], m - background["map"]).statistic for a, m in zip(x, maps)
                  if np.ptp(a) > 0 and np.ptp(m) > 0]
        found[tag] = {"rank": float(np.median(whole)), "rank_on_excess": float(np.nanmedian(excess)),
                      "questions": len(whole)}
    write_json(args.out, found)
    print(found)
    return args.out


if __name__ == "__main__":
    main()
