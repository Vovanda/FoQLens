"""Every oracle's kept field on the one contract (foqlens.oracle_overlay.Oracle), written once beside the originals.

In goes what the oracles wrote in their own units, out comes one file of fields in 0 ... 1 over the same questions and
the same groups - the layout every reader can count on. The originals are left where they are: the normalization is a
reading of them, not a replacement.

    uv run python scripts/unify_fields.py --fields "runs/oracles/e2b-it/precision-fields-*-shard1of5.npz" \\
        --out runs/oracles/e2b-it/unified-fields-shard1of5.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, save_npz_atomic
from foqlens.oracle_overlay import Kept, overlay

OVERLAYS = ("product", "mean", "least")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True, help="the oracles' kept fields (a glob joins shards)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--share", type=float, default=0.99, help="the quantile of its own values an oracle's scale is")
    parser.add_argument("--overlay-of", nargs="*", default=None,
                        help="the oracles an overlay is laid of; without it, every oracle of the file")
    args = parser.parse_args(argv)

    head = read_npz_parts(args.fields, RUN_FIELDS | {"sources"}, only={"ids", "sources"})
    # the file names among its sources the ones it only read into levels (the derived maps): they carry no field
    held = set(np.load(sorted(Path().glob(args.fields))[0]).files)
    sources = [s for s in head["sources"].tolist() if f"field_{s}" in held]
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources"},
                       only={"ids", "corpus", "sources", "groups"} | {f"field_{s}" for s in sources})
    made = {s: Kept(s, z[f"field_{s}"], args.share).field() for s in sources}
    laid = args.overlay_of or sources
    unknown = [s for s in laid if s not in made]
    if unknown:
        raise SystemExit(f"no such oracles in the file: {unknown}")
    overlays = {how: overlay([made[s] for s in laid], how) for how in OVERLAYS}

    save_npz_atomic(args.out, ids=z["ids"], corpus=z["corpus"], groups=z["groups"], sources=np.array(sources),
                    overlaid=np.array(laid), **{f"field_{s}": f for s, f in made.items()},
                    **{f"overlay_{how}": f for how, f in overlays.items()})
    print(f"{args.out}: {len(z['ids'])} questions, {len(z['groups'])} groups, {len(sources)} oracles")
    for name, f in list(made.items()) + [(f"overlay {how}", f) for how, f in overlays.items()]:
        print(f"  {name:28} mean {f.mean():.3f}  above 0.5 {float((f > 0.5).mean()):.3f}  at 1.0 "
              f"{float((f >= 1.0).mean()):.4f}")
    return args.out


if __name__ == "__main__":
    main()
