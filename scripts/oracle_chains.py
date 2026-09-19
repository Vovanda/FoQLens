"""The chain every block oracle gives on the small corpus (config.OracleChains; foqlens.group_oracle.build_chain).

For every question the oracle by trying tried, and every block oracle in the config (the answer gradient, the quant
gap, the error energies, the address): the oracle's scores summed into groups give an order; the groups are lifted to
the high level over the low one in that order until the answer - the model's own reply - is within the tolerance of
every block at the high level, and then the least needed of the rest go off (ZERO) while it stays within the zero
tolerance. The ends (every block at low, at high) are the oracle by trying's, read on the same target. So the oracles
are compared by what they are for: whose chain is shortest for the same answer (Volodya 20.09 01:58).

    uv run python scripts/oracle_chains.py --config configs/oracle-chains.toml --shard 1
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, refocustensors, runlog
from foqlens import model as fm
from foqlens.gpu_share import default_share
from foqlens.group_oracle import RUN_FIELDS, block_groups, build_chain, group_order, joined_answer
from foqlens.io import Checkpoint, plan_of, read_npz_parts, save_npz_atomic
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.small_corpus import StoredMasks, draw, pick_shard, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.oracle_chains")
SAVE_EVERY = 25  # questions between two saves of the partial pass, as the oracle by trying


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N questions only: a check")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"oracle_chains-{run}.jsonl", {"run": run, "script": "oracle_chains"})
    check = config.read(args.config, config.OracleChains)
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS)
    aim = check.replies or "reference"
    if str(trying["target"]) != aim:
        raise ValueError(f"the oracle by trying read {trying['target']}, the chains {aim}: one target for both")
    low, high = Level[str(trying["low"]).upper()], Level[str(trying["high"]).upper()]
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    groups, names = block_groups(bench.ctl)
    if names != trying["groups"].tolist():
        raise ValueError("the oracle by trying's groups are not this model's")
    g = len(names)
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"), small,
                               args.shard)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    ends_of = {k: trying["ends"][i] for i, k in enumerate(zip(trying["corpus"].tolist(), trying["ids"].tolist()))
               if not np.isnan(trying["ends"][i]).any()}
    laid = [(c, r) for c, r in found.laid if (c, r.id) in ends_of][:args.limit]
    prompts = found.prompts(fmt, laid)
    texts = targets(laid, check.replies)
    keys = [(c, r.id) for c, r in laid]
    sources = list(check.orders)
    orders = {}
    for name in sources:
        kept = StoredMasks.read(check.orders[name])
        rows = kept.rows_of(keys)
        orders[name] = group_order(to_groups(np.abs(rows), block_group_ids(kept.block_layer, kept.block_kind, names), g))
        orders[name][np.isnan(rows).any(axis=1)] = -1  # a question this oracle holds no mask for: no chain
    arrays = {f"{kind}_{name}": np.full((len(laid), g), np.nan) for name in sources for kind in ("prefix", "zero_sweep")}
    arrays |= {f"{kind}_{name}": np.full(len(laid), -1) for name in sources for kind in ("minimal", "zeroed")}
    subset = "" if args.corpora == list(SETUPS) else "-" + "+".join(args.corpora)
    target = args.out / f"chains-{low.name.lower()}{high.name.lower()}-{check.base}-{Path(check.corpus).stem}{subset}{suffix}.npz"
    checkpoint = Checkpoint(target.with_name(target.stem + ".partial.npz"), plan_of(check.__dict__, keys))
    first = 0
    if (kept := checkpoint.load()) is not None:  # a stopped or crashed pass goes on from its last saved question
        arrays, first = {k: kept[k] for k in arrays}, int(kept["done"])
        LOG.info("resumed after %d of %d questions from %s", first, len(laid), checkpoint.path)
    progress = Progress(len(laid) - first, "question")
    with stage(LOG, f"chains of {len(sources)} oracles on {len(laid)} questions"):
        for q in range(first, len(laid)):
            (corpus, row), prompt = laid[q], prompts[q]
            answer = joined_answer(prompt, texts[q])
            size = max(1, min(check.batch_tokens // len(bench.tokenizer(prompt + answer)["input_ids"]), g))
            found_here = []
            for name in sources:
                if orders[name][q, 0] < 0:
                    continue
                chain = build_chain(bench.model, bench.tokenizer, bench.ctl, bench.throttle, prompt, answer, groups,
                                    orders[name][q], ends_of[keys[q]], low, high, check.tolerance, check.zero_tolerance,
                                    size)
                arrays[f"prefix_{name}"][q], arrays[f"minimal_{name}"][q] = chain.prefix, chain.minimal
                arrays[f"zero_sweep_{name}"][q], arrays[f"zeroed_{name}"][q] = chain.zero_sweep, chain.zeroed
                found_here.append(f"{name} {chain.minimal}/{chain.zeroed}")
            LOG.info(progress.step(f"{corpus} {row.id}: chain/off {', '.join(found_here)}"),
                     extra={"corpus": corpus, "id": row.id})
            if (q + 1) % SAVE_EVERY == 0:
                checkpoint.save(done=q + 1, **arrays)
    save_npz_atomic(target, groups=np.array(names), corpus=np.array([c for c, _ in laid]),
                    ids=np.array([r.id for _, r in laid]), ends=np.array([ends_of[k] for k in keys]),
                    low=str(trying["low"]), high=str(trying["high"]), tolerance=check.tolerance,
                    zero_tolerance=np.nan if check.zero_tolerance is None else check.zero_tolerance, target=aim,
                    sources=np.array(sources), **{f"order_{n}": orders[n] for n in sources}, **arrays)
    checkpoint.clear()
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
