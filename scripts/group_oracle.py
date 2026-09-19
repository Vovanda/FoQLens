"""The oracles by trying on the small corpus's laid-out questions (foqlens.group_oracle; config.GroupOracleCheck).

Per question, every variant a layout of its own in one batch: every block at low, every block at high, every group
lifted, every group dropped; then the prefixes of the lift order, whose first within the tolerance of every block at
high is the question's minimal mask. The NLL of every variant is kept, so the tolerance can be chosen again later.

    uv run python scripts/group_oracle.py --config configs/group-oracle.toml
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
from foqlens.group_oracle import block_groups, build_chain, group_order, joined_answer, lift_layouts, variants_nll
from foqlens.io import save_npz_atomic
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.small_corpus import draw, pick_shard, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.group_oracle")
# the file is written every this many questions: a stopped run loses at most ~5 minutes at SQuAD's pace (13 s a
# question) and resumes from the file; a HotpotQA question alone takes ~55 s
SAVE_EVERY = 25


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N laid-out questions only: a check")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
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
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name), small, args.shard)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    laid = found.laid[:args.limit] if args.limit else found.laid
    prompts = found.prompts(fmt, laid)
    groups, names = block_groups(bench.ctl)
    g = len(names)
    singles = [np.array([k]) for k in range(g)]
    fixed = lift_layouts(groups, [np.array([], dtype=int), np.arange(g)] + singles, low, high)
    dropped = lift_layouts(groups, [np.setdiff1d(np.arange(g), s) for s in singles], low, high)

    lift, drop, prefix, zero_sweep = (np.full((len(laid), g), np.nan) for _ in range(4))
    ends, minimal, zeroed = np.full((len(laid), 2), np.nan), np.full(len(laid), -1), np.full(len(laid), -1)
    subset = "" if args.corpora == list(SETUPS) else "-" + "+".join(args.corpora)
    target = (args.out / f"group-oracle-{check.low}{check.high}-{check.base}-{Path(check.corpus).stem}{subset}"
              f"{suffix}.npz")
    keys = [(c, r.id) for c, r in laid]

    aim = check.replies or "reference"  # what the likelihood is read on (small_corpus.targets)
    texts = targets(laid, check.replies)

    def save() -> None:
        save_npz_atomic(target, lift=lift, drop=drop, prefix=prefix, ends=ends, minimal=minimal,
                        groups=np.array(names), corpus=np.array([c for c, _ in laid]),
                        ids=np.array([r.id for _, r in laid]),
                        unknown=np.array([(c, r.id) in found.unknown for c, r in laid]), low=check.low,
                        high=check.high, tolerance=check.tolerance, target=aim, zero_sweep=zero_sweep,
                        zeroed=zeroed, zero_tolerance=np.nan if check.zero_tolerance is None else check.zero_tolerance)

    done = set()
    if target.exists():  # a run stopped midway: the questions it finished are read back, not asked again
        with np.load(target, allow_pickle=False) as kept:
            ran = (str(kept["low"]), str(kept["high"]), float(kept["tolerance"]),
                   str(kept["target"]) if "target" in kept.files else "reference",
                   str(kept["zero_tolerance"]) if "zero_tolerance" in kept.files else "nan")
            if ran != (check.low, check.high, check.tolerance, aim, str(float(check.zero_tolerance or np.nan))):
                raise ValueError(f"{target} is another oracle's run {ran}; move it away to start over")
            where = {k: i for i, k in enumerate(zip(kept["corpus"].tolist(), kept["ids"].tolist()))}
            for q, k in enumerate(keys):
                i = where.get(k)
                if i is not None and not np.isnan(kept["ends"][i]).any():
                    lift[q], drop[q], prefix[q] = kept["lift"][i], kept["drop"][i], kept["prefix"][i]
                    ends[q], minimal[q] = kept["ends"][i], kept["minimal"][i]
                    if "zeroed" in kept.files:
                        zero_sweep[q], zeroed[q] = kept["zero_sweep"][i], kept["zeroed"][i]
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
            first = np.concatenate([fixed, dropped])
            length = len(bench.tokenizer(prompt + answer)["input_ids"])
            size = max(1, min(check.batch_tokens // length, len(first)))
            reading = (bench.model, bench.tokenizer, bench.ctl, bench.throttle, prompt, answer)
            nll = variants_nll(*reading, first, size)
            ends[q] = nll[:2]  # every block at low, every block at high
            lift[q] = nll[0] - nll[2:2 + g]
            drop[q] = nll[2 + g:] - nll[1]
            # the chain that gives the answer in the order of the lift, then the least needed of the rest off
            chain = build_chain(*reading, groups, group_order(lift[q]), ends[q], low, high, check.tolerance,
                                check.zero_tolerance, size)
            prefix[q], minimal[q], zero_sweep[q], zeroed[q] = chain.prefix, chain.minimal, chain.zero_sweep, chain.zeroed
            LOG.info(progress.step(f"{corpus} {row.id}: minimal mask {minimal[q]} of {g} groups, "
                                   f"{zeroed[q]} of the rest off"),
                     extra={"corpus": corpus, "id": row.id, "minimal": int(minimal[q]), "zeroed": int(zeroed[q])})
            if (q + 1) % SAVE_EVERY == 0:
                save()
    save()
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
