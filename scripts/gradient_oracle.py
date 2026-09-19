"""The gradient oracle on the small corpus (config.GradientOracle): gradient x activation of every block for the loss of
the question's reference answer - the target the oracles by trying measure (scripts/group_oracle.py) - with every block
at one level. Laid-out and calibration questions alike, kept as StoredMasks in both forms, so a knapsack reads them as
any mask (strategy_answers --masks).

    uv run python scripts/gradient_oracle.py --config configs/gradient-oracle.toml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from foqlens import config, refocustensors, runlog
from foqlens import model as fm
from foqlens.gpu_share import default_share
from foqlens.group_oracle import joined_answer
from foqlens.io import Checkpoint, plan_of, save_npz_atomic
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.regulator import block_kinds, block_layers
from foqlens.runlog import stage
from foqlens.scoring import GRADIENT_BACKENDS, GradientScorer
from foqlens.small_corpus import draw, pick_shard, store, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.gradient_oracle")
FORMS = ("gradient", "gradient_magnitude")
# the partial pass is saved every this many batches: a batch of short questions takes about a second, of HotpotQA's
# longest several - a stop loses at most a minute or two
SAVE_EVERY = 20


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N questions only: a check")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    parser.add_argument("--out", type=Path, default=Path("runs/masks/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"gradient_oracle-{run}.jsonl", {"run": run, "script": "gradient_oracle"})
    check = config.read(args.config, config.GradientOracle)
    level = Level[check.level.upper()]
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    bench.ctl.set_all(level)
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name), small, args.shard)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    pairs = found.laid + found.calibration
    prompts = found.prompts(fmt, pairs)
    answers = [joined_answer(p, t) if t.strip() else "" for p, t in zip(prompts, targets(pairs, check.replies))]
    lengths = np.array([len(ids) for ids in bench.tokenizer([p + a for p, a in zip(prompts, answers)])["input_ids"]])
    # the calibration is ordered by corpus: evenly spaced picks reach every corpus; a shard takes its share of them
    wanted = check.calibration_questions // (small.shards if args.shard else 1)
    picked = set(range(len(found.laid))) | {len(found.laid) + int(i) for i in np.unique(np.linspace(
        0, len(found.calibration) - 1, min(wanted, len(found.calibration))).round())}
    todo = [i for i in sorted(picked) if answers[i].strip() and lengths[i] <= check.max_tokens]
    todo = todo[:args.limit] if args.limit else todo
    LOG.info("%d of %d questions within %d tokens with an answer", len(todo), len(pairs), check.max_tokens)
    gap = (Level[check.gap_low.upper()], Level[check.gap_high.upper()]) if check.gap_low else None
    scorer = GradientScorer(bench.model, bench.ctl.modules, gap=gap,
                            attention=GRADIENT_BACKENDS[config.choose(check.attention, GRADIENT_BACKENDS, "attention")])
    # every form's source name: the rung gap is part of the quant_gap form's name
    forms_named = {form: form for form in FORMS} | (
        {"quant_gap": f"quant_gap_{gap[0].name.lower()}{gap[1].name.lower()}"} if gap else {})
    masks = {form: np.full((len(pairs), bench.ctl.n_blocks), np.nan, dtype=np.float32) for form in forms_named}
    nll = np.full(len(pairs), np.nan, dtype=np.float32)
    # longest first, so the first batch shows the peak; a batch holds up to batch_tokens padded tokens
    todo.sort(key=lambda i: -lengths[i])
    batches, start = [], 0
    while start < len(todo):
        size = max(1, check.batch_tokens // int(lengths[todo[start]]))
        batches.append(todo[start:start + size])
        start += size
    stem = f"{level.name.lower()}-file-{check.base}-{Path(check.corpus).stem}{suffix}"
    checkpoint = Checkpoint(args.out / f"answer_gradient-{stem}.partial.npz",
                            plan_of(check.__dict__, batches, [r.id for _, r in pairs]))
    first = 0
    if (kept := checkpoint.load()) is not None:  # a stopped or crashed pass goes on from its last saved batch
        nll, first = kept["nll"], int(kept["batches_done"])
        masks = {form: kept[f"mask_{form}"] for form in forms_named}
        LOG.info("resumed after %d of %d batches from %s", first, len(batches), checkpoint.path)

    def score(part: list[int]) -> bool:
        """The gradients of `part` into the masks; False where it ran out of memory. The retry happens outside the
        except, so the failed pass's tensors are freed first."""
        try:
            with bench.throttle.batch():
                forms, nll[part] = scorer.answer_batch(bench.model, bench.tokenizer, [prompts[i] for i in part],
                                                       [answers[i] for i in part])
        except torch.OutOfMemoryError:
            return False
        for form in forms_named:
            masks[form][part] = forms[form]
        return True

    progress = Progress(len(batches) - first, "batch")
    lost = int(kept["lost"]) if kept is not None else 0
    with stage(LOG, f"answer gradients of {len(todo)} questions at {level.name}"):
        for b, batch in enumerate(batches[first:], start=first):
            if not score(batch):
                torch.cuda.empty_cache()
                # one question at a time; one that does not fit alone keeps no mask (NaN), counted like an overlong one
                failed = list(batch) if len(batch) == 1 else []
                for i in batch if len(batch) > 1 else []:
                    if not score([i]):
                        torch.cuda.empty_cache()
                        failed.append(i)
                for i in failed:
                    LOG.warning("%s %s (%d tokens) runs out of memory alone and keeps no mask", pairs[i][0],
                                pairs[i][1].id, lengths[i], extra={"tokens": int(lengths[i])})
                lost += len(failed)
            LOG.info(progress.step(f"{len(batch)} questions, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"),
                     extra={"questions": len(batch), "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
            if (b + 1) % SAVE_EVERY == 0:
                checkpoint.save(nll=nll, batches_done=b + 1, lost=lost, **{f"mask_{f}": m for f, m in masks.items()})
    modules = bench.ctl.modules.values()
    targets = []
    for form, source in forms_named.items():
        kept = store(found, masks[form], block_layers(bench.ctl), block_kinds(bench.ctl),
                     np.concatenate([m.block_sizes() * m.in_features for m in modules]),
                     {"model": name, "source": f"answer_{source}", "model_source": "file", "base": check.base,
                      "level": level.name, "corpus": check.corpus, "corpora": args.corpora,
                      "max_tokens": check.max_tokens, "without_mask": int(len(pairs) - len(todo) + lost),
                      "out_of_memory": lost, "attention": check.attention})
        target = args.out / f"answer_{source}-{stem}.npz"
        kept.save(target)
        LOG.info("written %s", target)
        targets.append(target)
    # the NLL the gradient was taken of, to be checked against the oracle by trying's at the same level
    target = args.out / f"answer_nll-{stem}.npz"
    save_npz_atomic(target, nll=nll, corpus=np.array([c for c, _ in pairs]), ids=np.array([r.id for _, r in pairs]))
    LOG.info("written %s", target)
    checkpoint.clear()
    return targets + [target]


if __name__ == "__main__":
    main()
