"""The precision fields of the oracles on the small corpus (config.PrecisionFields; foqlens.precision_field): every
oracle's field of importance - the lift and the drop by trying, the block oracles' scores summed into groups - read at
the question's threshold into a level a group, and the reference measured: every group alone at D6, D4 and D2 under
every block at D8, then raised by whole rungs until the whole holds. The ratios of the rungs come from the error
energies of D2, D4 and D6 on the same questions. Deterministic: the same questions in the same batches.

    uv run python scripts/precision_fields.py --config configs/precision-fields.toml --shard 1
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
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.group_oracle import RUN_FIELDS, block_groups, joined_answer
from foqlens.io import Checkpoint, plan_of, read_npz_parts, save_npz_atomic, write_json
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.pipeline import Bench
from foqlens.precision_field import RUNGS, find_threshold, measure_field, rung_costs, rung_ratios
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.runlog import stage
from foqlens.small_corpus import StoredMasks, draw, named, pick_shard, targets

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.precision_fields")
SAVE_EVERY = 10  # questions between two saves of the partial pass: a question reads ~400 layouts


def group_fields(pattern: str, keys: list[tuple[str, str]], names: list[str]) -> np.ndarray:
    """A block oracle's kept scores as a field over the groups: |score| summed over a group's blocks, NaN rows where the
    file holds no mask for the question: [questions, groups]."""
    kept = StoredMasks.read(pattern)
    rows = kept.rows_of(keys)
    out = to_groups(np.abs(rows), block_group_ids(kept.block_layer, kept.block_kind, names), len(names))
    out[np.isnan(rows).any(axis=1)] = np.nan
    return out


def group_weights(pattern: str, names: list[str]) -> np.ndarray:
    """The weights every group holds, from a kept masks file's blocks: [groups]."""
    kept = StoredMasks.read(pattern)
    return np.bincount(block_group_ids(kept.block_layer, kept.block_kind, names), weights=kept.block_weights,
                       minlength=len(names))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--limit", type=int, default=None, help="the first N questions only: a check")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    parser.add_argument("--only", type=Path, default=None,
                        help="a json list of [corpus, id]: read these questions of the draw and no others")
    parser.add_argument("--also", type=Path, default=None,
                        help="a json list of [corpus, id]: lay these questions out on top of the share")
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"precision_fields-{run}.jsonl", {"run": run, "script": "precision_fields"})
    check = config.read(args.config, config.PrecisionFields)
    # name what is needed: shards of the oracle by trying written before and after the masks were dropped hold
    # different fields, and joining them whole fails on a field only the older shard has
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS,
                            only={"target", "corpus", "ids", "ends", "lift", "drop"})
    aim = check.replies or "reference"
    if str(trying["target"]) != aim:
        raise ValueError(f"the oracle by trying read {trying['target']}, the fields {aim}: one target for both")
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    groups, names = block_groups(bench.ctl)
    if names != trying["groups"].tolist():
        raise ValueError("the oracle by trying's groups are not this model's")
    g = len(names)
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, f"{MODEL}@{fm.REVISIONS[MODEL][:8]}",
                                    also=named(args.also)), small,
                               args.shard)
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    rows_of = {k: i for i, k in enumerate(zip(trying["corpus"].tolist(), trying["ids"].tolist()))
               if not np.isnan(trying["ends"][i]).any()}
    laid = [(c, r) for c, r in found.laid if (c, r.id) in rows_of][:args.limit]
    if args.only:  # a batch named by hand: the questions a reading of the answers picked out
        wanted = {tuple(pair) for pair in json.loads(args.only.read_text(encoding="utf-8"))}
        laid = [(c, r) for c, r in laid if (c, r.id) in wanted]
        LOG.info("%d of %d questions named by %s", len(laid), len(wanted), args.only)
    prompts = found.prompts(fmt, laid)
    texts = targets(laid, check.replies)
    keys = [(c, r.id) for c, r in laid]
    at = [rows_of[k] for k in keys]

    # every field over the groups [questions, groups]: the oracle by trying's own two, then the block oracles'
    fields = {"lift": trying["lift"][at], "drop": trying["drop"][at]}
    fields |= {name: group_fields(pattern, keys, names) for name, pattern in check.fields.items()}
    # the knapsack lifts by the gain a byte (docs/quantization-filter-math.md, section 9): every field also over the
    # weights of its groups - an MLP group holds several times an attention group's weights
    # the oracles measure one thing in their own units: each field scaled to unit mass a question, then summed, is
    # their joint estimate of it (Volodya 20.09) - a field of its own, read at its own threshold like any other
    scaled = np.stack([np.maximum(f, 0) / np.nansum(np.maximum(f, 0), axis=1, keepdims=True) for f in fields.values()])
    fields["pooled"] = np.nanmean(scaled, axis=0)  # an oracle that holds no field for a question does not vote on it
    weights = group_weights(check.energies[0], names)
    fields |= {f"{name}_per_weight": field / weights for name, field in list(fields.items())}
    energies = [group_fields(pattern, keys, names) for pattern in check.energies]
    ratios = rung_ratios(energies)
    LOG.info("rung ratios over the groups (median, min, max): %s", "; ".join(
        f"{r.name} {np.median(x):.4f} {x.min():.4f} {x.max():.4f}" for r, x in zip(RUNGS, ratios)))
    sources = list(fields)

    arrays = {f"levels_{s}": np.full((len(laid), g), 255, dtype=np.uint8) for s in sources}
    arrays |= {f"{kind}_{s}": np.full(len(laid), np.nan) for s in sources for kind in ("eps", "nll", "target")}
    arrays |= {f"batches_{s}": np.full(len(laid), -1) for s in sources}
    if check.reference:
        arrays |= {"alone": np.full((len(laid), len(RUNGS), g), np.nan), "target_reference": np.full(len(laid), np.nan),
                   "alone_levels": np.full((len(laid), g), 255, dtype=np.uint8),
                   "joint": np.full((len(laid), len(RUNGS) + 1), np.nan), "steps": np.full(len(laid), -1),
                   "levels_reference": np.full((len(laid), g), 255, dtype=np.uint8)}
    subset = "" if args.corpora == list(SETUPS) else "-" + "+".join(args.corpora)
    target = args.out / f"precision-fields-{check.base}-{Path(check.corpus).stem}{subset}{suffix}.npz"
    checkpoint = Checkpoint(target.with_name(target.stem + ".partial.npz"), plan_of(check.__dict__, keys))
    first = 0
    if (kept := checkpoint.load()) is not None:  # a stopped or crashed pass goes on from its last saved question
        arrays, first = {k: kept[k] for k in arrays}, int(kept["done"])
        LOG.info("resumed after %d of %d questions from %s", first, len(laid), checkpoint.path)
    progress = Progress(len(laid) - first, "question")
    with stage(LOG, f"precision fields of {len(sources)} oracles on {len(laid)} questions"), GpuMonitor() as gpu:
        for q in range(first, len(laid)):
            (corpus, row), prompt = laid[q], prompts[q]
            answer = joined_answer(prompt, texts[q])
            length = len(bench.tokenizer(prompt + answer)["input_ids"])
            size = max(2, min(check.batch_tokens // length, 1 + len(RUNGS) * g))
            width = max(1, min(check.probes, size - 1))
            reading = (bench.model, bench.tokenizer, bench.ctl, bench.throttle, prompt, answer)
            said = []
            for s in sources:
                if np.isnan(fields[s][q]).any():
                    continue  # the oracle holds no field for this question (a prompt too long for the gradient)
                got = find_threshold(*reading, groups, rung_costs(fields[s][q], ratios), check.tolerance, width,
                                    share=check.tolerance_share)
                arrays[f"levels_{s}"][q], arrays[f"eps_{s}"][q] = got.levels, got.eps
                arrays[f"nll_{s}"][q], arrays[f"target_{s}"][q] = got.nll, got.target
                arrays[f"batches_{s}"][q] = got.batches
                said.append(f"{s} {np.bincount(got.levels, minlength=5)[1:5].tolist()}")
            if check.reference:
                ref = measure_field(*reading, groups, g, check.tolerance, size, share=check.tolerance_share)
                arrays["alone"][q], arrays["target_reference"][q] = ref.alone, ref.target
                arrays["alone_levels"][q], arrays["joint"][q] = ref.alone_levels, ref.joint
                arrays["steps"][q], arrays["levels_reference"][q] = ref.steps, ref.levels
                said.append(f"reference +{ref.steps} {np.bincount(ref.levels, minlength=5)[1:5].tolist()}")
            LOG.info(progress.step(f"{corpus} {row.id}: groups at D2/D4/D6/D8 {'; '.join(said)}"),
                     extra={"corpus": corpus, "id": row.id})
            if (q + 1) % SAVE_EVERY == 0:
                checkpoint.save(done=q + 1, **arrays)
    # what the oracles agree on, at no cost on the card: the lowest level of every group over the oracles that hold a
    # field for the question (their common part), and the highest (all of them together). Several maps hold one
    # answer - the maps need not agree (Volodya 20.09) - so the common part is answered too
    got = [s for s in sources if not s.endswith("_per_weight")]
    stack = np.stack([arrays[f"levels_{s}"] for s in got])
    read = (stack != 255).all(axis=0)
    middle = np.median(stack, axis=0).round().astype(np.uint8)  # a map between them: is the region of maps that hold
    for label, take in (("common", stack.min(axis=0)), ("middle", middle), ("together", stack.max(axis=0))):
        arrays[f"levels_{label}"] = np.where(read, take, 255).astype(np.uint8)
        sources.append(label)
    save_npz_atomic(target, groups=np.array(names), corpus=np.array([c for c, _ in laid]),
                    ids=np.array([r.id for _, r in laid]), unknown=np.array([k in found.unknown for k in keys]),
                    ratios=ratios, rungs=np.array([r.name for r in RUNGS]), tolerance=check.tolerance, target=aim,
                    sources=np.array(sources), **{f"field_{s}": field for s, field in fields.items()}, **arrays)
    write_json(target.with_suffix(".gpu.json"), {"gpu": gpu.summary(), "pacer": bench.throttle.stats()})
    checkpoint.clear()
    LOG.info("written %s", target)
    return target


if __name__ == "__main__":
    main()
