"""Does the address stand for the sensitivity: the address's kept masks against the oracle's, block by block, on the
laid-out questions the model knows (foqlens.sensitivity; config.OracleCheck).

    uv run python scripts/address_vs_oracle.py --config configs/address-vs-oracle.toml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, runlog
from foqlens.coverage import spread
from foqlens.io import write_json
from foqlens.sensitivity import rank_correlation, top_overlap
from foqlens.small_corpus import StoredMasks

LOG = logging.getLogger("foqlens.address_vs_oracle")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/address/e2b-it"))
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"address_vs_oracle-{run}.jsonl", {"run": run, "script": "address_vs_oracle"})
    check = config.read(args.config, config.OracleCheck)
    estimate, oracle = StoredMasks.load(Path(check.estimate)), StoredMasks.load(Path(check.oracle))
    if estimate.ids.tolist() != oracle.ids.tolist() or estimate.masks.shape != oracle.masks.shape:
        raise ValueError("the address and the oracle hold other questions or blocks")
    # questions whose prompt was too long for the oracle's backward pass have no oracle mask: counted apart
    has_oracle = ~np.isnan(oracle.masks).any(axis=1)
    known = estimate.laid & ~estimate.unknown & has_oracle
    e, o = estimate.masks[known], np.abs(oracle.masks[known])
    found = {"questions": int(known.sum()),
             "without_oracle": int((estimate.laid & ~estimate.unknown & ~has_oracle).sum()),
             "estimate": estimate.meta, "oracle": oracle.meta,
             "rank_correlation": spread(rank_correlation(e, o)), "shares": {}, "by_kind": {}}
    for share in check.shares:
        found["shares"][share] = {"top_overlap": spread(top_overlap(e, o, share)), "chance": share,
                                  "tail_rank_correlation": spread(rank_correlation(e, o, outside_share=share))}
        LOG.info("top %g: overlap median %.3f (chance %g), tail rho median %.3f", share,
                 found["shares"][share]["top_overlap"]["median"], share,
                 found["shares"][share]["tail_rank_correlation"]["median"], extra={"share": share})
    for kind in np.unique(estimate.block_kind):
        of = estimate.block_kind == kind
        found["by_kind"][str(kind)] = spread(rank_correlation(e[:, of], o[:, of]))
    LOG.info("rank correlation median %.3f over %d questions", found["rank_correlation"]["median"], found["questions"])
    target = args.out / f"{args.config.stem}.json"
    write_json(target, found)
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
