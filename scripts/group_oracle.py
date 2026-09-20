"""The oracles by trying on the small corpus's laid-out questions (foqlens.group_oracle; config.GroupOracleCheck).

Per question, every variant a layout of its own in one batch: every block at low, every block at high, every group
lifted, every group dropped. The NLL of every variant is kept: the lift and the drop are fields the precision fields
are read from (scripts/precision_fields.py).

    uv run python scripts/group_oracle.py --config configs/group-oracle.toml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, refocustensors, runlog
from foqlens import model as fm
from foqlens.gpu_share import default_share
from foqlens.group_oracle import block_groups, joined_answer, lift_layouts, variants_nll
from foqlens.io import save_npz_atomic
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.small_corpus import draw, named, pick_shard, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.group_oracle")
# the file is written every this many questions: a stopped run loses at most a few minutes at SQuAD's pace and resumes
# from the file; a HotpotQA question alone takes ~55 s
SAVE_EVERY = 25


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N laid-out questions only: a check")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    parser.add_argument("--only", type=Path, default=None,
                        help="a json list of [corpus, id]: read these questions of the draw and no others")
    parser.add_argument("--also", type=Path, default=None,
                        help="a json list of [corpus, id]: lay these questions out on top of the share")
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"group_oracle-{run}.jsonl", {"run": run, "script": "group_oracle"})
    check = config.read(args.config, config.GroupOracleCheck)
    low, high = Level[check.low.upper()], Level[check.high.upper()]
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name, also=named(args.also)), small, args.shard)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    laid = found.laid[:args.limit] if args.limit else found.laid
    if args.only:  # a batch named by hand: the questions a reading of the answers picked out, in the draw's order
        wanted = {tuple(pair) for pair in json.loads(args.only.read_text(encoding="utf-8"))}
        laid = [(c, r) for c, r in laid if (c, r.id) in wanted]
        LOG.info("%d of %d questions named by %s", len(laid), len(wanted), args.only)
    prompts = found.prompts(fmt, laid)
    groups, names = block_groups(bench.ctl)
    g = len(names)
    singles = [np.array([k]) for k in range(g)]
    fixed = lift_layouts(groups, [np.array([], dtype=int), np.arange(g)] + singles, low, high)
    dropped = lift_layouts(groups, [np.setdiff1d(np.arange(g), s) for s in singles], low, high)
    every = np.concatenate([fixed, dropped])

    lift, drop = np.full((len(laid), g), np.nan), np.full((len(laid), g), np.nan)
    ends = np.full((len(laid), 2), np.nan)
    subset = "" if args.corpora == list(SETUPS) else "-" + "+".join(args.corpora)
    target = (args.out / f"group-oracle-{check.low}{check.high}-{check.base}-{Path(check.corpus).stem}{subset}"
              f"{suffix}.npz")
    keys = [(c, r.id) for c, r in laid]

    aim = check.replies or "reference"  # what the likelihood is read on (small_corpus.targets)
    texts = targets(laid, check.replies)

    def save() -> None:
        save_npz_atomic(target, lift=lift, drop=drop, ends=ends, groups=np.array(names),
                        corpus=np.array([c for c, _ in laid]), ids=np.array([r.id for _, r in laid]),
                        unknown=np.array([(c, r.id) in found.unknown for c, r in laid]), low=check.low,
                        high=check.high, target=aim)

    done = set()
    if target.exists():  # a run stopped midway: the questions it finished are read back, not asked again
        with np.load(target, allow_pickle=False) as kept:
            ran = (str(kept["low"]), str(kept["high"]), str(kept["target"]))
            if ran != (check.low, check.high, aim):
                raise ValueError(f"{target} is another oracle's run {ran}; move it away to start over")
            where = {k: i for i, k in enumerate(zip(kept["corpus"].tolist(), kept["ids"].tolist()))}
            for q, k in enumerate(keys):
                i = where.get(k)
                if i is not None and not np.isnan(kept["ends"][i]).any():
                    lift[q], drop[q], ends[q] = kept["lift"][i], kept["drop"][i], kept["ends"][i]
                    done.add(q)
        LOG.info("resumed %d of %d questions from %s", len(done), len(laid), target)
    progress = Progress(len(laid) - len(done), "question")
    with stage(LOG, f"oracles by trying on {len(laid)} questions, {g} groups"):
        for q, ((corpus, row), prompt) in enumerate(zip(laid, prompts)):
            if q in done:
                continue
            answer = joined_answer(prompt, texts[q]) if texts[q].strip() else ""
            if not answer.strip():
                LOG.warning("%s %s has no %s text and is left out", corpus, row.id, aim)
                continue
            length = len(bench.tokenizer(prompt + answer)["input_ids"])
            size = max(1, min(check.batch_tokens // length, len(every)))
            nll = variants_nll(bench.model, bench.tokenizer, bench.ctl, bench.throttle, prompt, answer, every, size)
            ends[q] = nll[:2]  # every block at low, every block at high
            lift[q] = nll[0] - nll[2:2 + g]
            drop[q] = nll[2 + g:] - nll[1]
            LOG.info(progress.step(f"{corpus} {row.id}: ends {ends[q][0]:.3f} / {ends[q][1]:.3f}"),
                     extra={"corpus": corpus, "id": row.id})
            if (q + 1) % SAVE_EVERY == 0:
                save()
    save()
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
