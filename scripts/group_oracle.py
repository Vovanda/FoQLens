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
from foqlens.group_oracle import answer_nll, block_groups, lift_layouts, minimal_prefix
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.small_corpus import draw

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.group_oracle")


def variants_nll(bench: Bench, prompt: str, answer: str, layouts: np.ndarray, batch_tokens: int) -> np.ndarray:
    """The answer's NLL under every layout, as many variants a batch as `batch_tokens` holds: [variants]."""
    length = len(bench.tokenizer(prompt + answer)["input_ids"])
    size = max(1, batch_tokens // length)
    out = []
    for start in range(0, len(layouts), size):
        part = layouts[start:start + size]
        with bench.throttle.batch():
            bench.ctl.set_layout(part)
            out.append(answer_nll(bench.model, bench.tokenizer, [prompt] * len(part), [answer] * len(part)).cpu())
    return np.concatenate([o.numpy() for o in out])


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N laid-out questions only: a check")
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"group_oracle-{run}.jsonl", {"run": run, "script": "group_oracle"})
    check = config.read(args.config, config.GroupOracleCheck)
    low, high = Level[check.low.upper()], Level[check.high.upper()]
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    found = draw(args.corpora, args.frozen, config.read(Path(check.corpus), config.SmallCorpus), name)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    laid = found.laid[:args.limit] if args.limit else found.laid
    prompts = found.prompts(fmt, laid)
    groups, names = block_groups(bench.ctl)
    g = len(names)
    singles = [np.array([k]) for k in range(g)]
    fixed = lift_layouts(groups, [np.array([], dtype=int), np.arange(g)] + singles, low, high)
    dropped = lift_layouts(groups, [np.setdiff1d(np.arange(g), s) for s in singles], low, high)

    lift, drop, prefix = (np.full((len(laid), g), np.nan) for _ in range(3))
    ends, minimal = np.full((len(laid), 2), np.nan), np.full(len(laid), -1)
    progress = Progress(len(laid), "question")
    with stage(LOG, f"oracles by trying on {len(laid)} questions, {g} groups"):
        for q, ((corpus, row), prompt) in enumerate(zip(laid, prompts)):
            answer = " " + row.answers[0] if row.answers else ""
            if not answer.strip():
                LOG.warning("%s %s has no reference answer and is left out", corpus, row.id)
                continue
            nll = variants_nll(bench, prompt, answer, np.concatenate([fixed, dropped]), check.batch_tokens)
            ends[q] = nll[:2]  # every block at low, every block at high
            lift[q] = nll[0] - nll[2:2 + g]
            drop[q] = nll[2 + g:] - nll[1]
            order = np.argsort(-lift[q], kind="stable")
            prefix[q] = variants_nll(bench, prompt, answer,
                                     lift_layouts(groups, [order[:k + 1] for k in range(g)], low, high),
                                     check.batch_tokens)
            minimal[q] = minimal_prefix(np.concatenate([[ends[q, 0]], prefix[q]]), ends[q, 1], check.tolerance)
            LOG.info(progress.step(f"{corpus} {row.id}: minimal mask {minimal[q]} of {g} groups"),
                     extra={"corpus": corpus, "id": row.id, "minimal": int(minimal[q])})
    target = args.out / f"group-oracle-{check.base}-{Path(check.corpus).stem}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, lift=lift, drop=drop, prefix=prefix, ends=ends, minimal=minimal, groups=np.array(names),
             corpus=np.array([c for c, _ in laid]), ids=np.array([r.id for _, r in laid]),
             unknown=np.array([(c, r.id) in found.unknown for c, r in laid]), low=check.low, high=check.high,
             tolerance=check.tolerance)
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
