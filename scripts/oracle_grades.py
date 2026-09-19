"""Every chain graded (config.ChainGrades; foqlens.group_oracle.grade_chain): on top of its three levels - the chain at
the top, the rest at the base, the least needed off - each chain group from the least needed is read at the coarsest
rung that keeps the model's own reply within the tolerance of every block at the top. Every question then has its lens
in every level: how few groups at D8, D6, D4, D2 and how many off give the answer undistorted (Volodya 20.09 02:12).
The chain of the lift (the oracle by trying) and of every block oracle (oracle_chains.py).

    uv run python scripts/oracle_grades.py --config configs/oracle-grades.toml --shard 1
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
from foqlens.group_oracle import RUN_FIELDS, block_groups, chain_layout, grade_chain, group_order, joined_answer
from foqlens.io import Checkpoint, plan_of, read_npz_parts, save_npz_atomic
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.small_corpus import draw, pick_shard, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.oracle_grades")
SAVE_EVERY = 25  # questions between two saves of the partial pass


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
    runlog.setup(args.out / "logs" / f"oracle_grades-{run}.jsonl", {"run": run, "script": "oracle_grades"})
    check = config.read(args.config, config.ChainGrades)
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS)
    aim = check.replies or "reference"
    if str(trying["target"]) != aim:
        raise ValueError(f"the oracle by trying read {trying['target']}, the grades {aim}: one target for both")
    low, high = Level[str(trying["low"]).upper()], Level[str(trying["high"]).upper()]
    rungs = tuple(Level[r.upper()] for r in check.rungs)
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    groups, names = block_groups(bench.ctl)
    g = len(names)
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"), small,
                               args.shard)
    at = {k: i for i, k in enumerate(zip(trying["corpus"].tolist(), trying["ids"].tolist()))
          if trying["minimal"][i] >= 0}
    laid = [(c, r) for c, r in found.laid if (c, r.id) in at][:args.limit]
    keys = [(c, r.id) for c, r in laid]
    rows = [at[k] for k in keys]
    # every source's order, chain length and zeroing per question: the lift's, then every block oracle's
    sources = {"lift": (group_order(trying["lift"][rows]), trying["minimal"][rows], trying["zeroed"][rows])}
    if check.chains:
        chains = read_npz_parts(check.chains, RUN_FIELDS | {"sources"})
        held = {k: i for i, k in enumerate(zip(chains["corpus"].tolist(), chains["ids"].tolist()))}
        pick = [held.get(k, -1) for k in keys]
        for source in chains["sources"].tolist():
            sources[source] = tuple(np.array([chains[f"{kind}_{source}"][i] if i >= 0 else fill for i in pick])
                                    for kind, fill in (("order", np.full(g, -1)), ("minimal", -1), ("zeroed", -1)))
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    prompts = found.prompts(fmt, laid)
    texts = targets(laid, check.replies)
    grades = {name: np.full((len(laid), g), 255, dtype=np.uint8) for name in sources}  # 255: no lens
    target = args.out / (f"grades-{low.name.lower()}{high.name.lower()}-{check.base}-{Path(check.corpus).stem}"
                         f"{'' if args.corpora == list(SETUPS) else '-' + '+'.join(args.corpora)}{suffix}.npz")
    checkpoint = Checkpoint(target.with_name(target.stem + ".partial.npz"), plan_of(check.__dict__, keys))
    first = 0
    if (kept := checkpoint.load()) is not None:
        grades, first = {name: kept[f"grades_{name}"] for name in sources}, int(kept["done"])
        LOG.info("resumed after %d of %d questions from %s", first, len(laid), checkpoint.path)
    progress = Progress(len(laid) - first, "question")
    with stage(LOG, f"grades of {len(sources)} chains on {len(laid)} questions"):
        for q in range(first, len(laid)):
            (corpus, row), prompt = laid[q], prompts[q]
            answer = joined_answer(prompt, texts[q])
            size = max(1, min(check.batch_tokens // len(bench.tokenizer(prompt + answer)["input_ids"]), len(rungs)))
            summary = []
            for name, (orders, minimal, zeroed) in sources.items():
                m, z = int(minimal[q]), int(zeroed[q])
                if m < 0:
                    continue
                layout = chain_layout(groups, orders[q], m, max(z, 0), low, high)
                graded, _ = grade_chain(bench.model, bench.tokenizer, bench.ctl, bench.throttle, prompt, answer,
                                        groups, layout, orders[q][:m], rungs, float(trying["ends"][rows[q], 1]),
                                        check.tolerance, size)
                grades[name][q] = [graded[groups == k][0] for k in range(g)]
                counts = {lv.name: int((grades[name][q] == int(lv)).sum()) for lv in (high, *rungs[::-1], low)}
                summary.append(f"{name} " + "/".join(str(v) for v in counts.values()))
            LOG.info(progress.step(f"{corpus} {row.id}: {high.name}/{'/'.join(r.name for r in rungs[::-1])}/"
                                   f"{low.name} {', '.join(summary)}"), extra={"corpus": corpus, "id": row.id})
            if (q + 1) % SAVE_EVERY == 0:
                checkpoint.save(done=q + 1, **{f"grades_{n}": v for n, v in grades.items()})
    save_npz_atomic(target, groups=np.array(names), corpus=np.array([c for c, _ in laid]),
                    ids=np.array([r.id for _, r in laid]), sources=np.array(list(sources)), low=str(trying["low"]),
                    high=str(trying["high"]), rungs=np.array(list(check.rungs)), tolerance=check.tolerance,
                    target=aim, **{f"grades_{n}": v for n, v in grades.items()})
    checkpoint.clear()
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
