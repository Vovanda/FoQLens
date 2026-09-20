"""The masks of the small corpus, computed once and kept (foqlens.small_corpus.StoredMasks).

A mask pass is minutes of GPU that no knob of a layout changes, so a run of layouts reads it from disk
(strategy_answers --masks). The oracle - the gradient's Taylor score - is read at full precision from the checkpoint
(--model-source checkpoint); a backward pass over the longest prompts (HotpotQA's passages, over 3000 tokens) runs out
of memory even alone, so prompts over --max-tokens get no mask (NaN) and every reader counts them apart.

    uv run python scripts/small_corpus_masks.py --source gradient --model-source checkpoint --max-tokens 2048
    uv run python scripts/small_corpus_masks.py --source hybrid --base bartowski-Q2_K
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, refocustensors, runlog
from foqlens import model as fm
from foqlens.gguf_weights import PUBLISHED
from foqlens.gpu_share import default_share
from foqlens.io import Checkpoint, plan_of
from foqlens.pipeline import ADDRESS_SOURCES, Bench, ModelSource
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.regulator import block_kinds, block_layers
from foqlens.runlog import stage
from foqlens.small_corpus import draw, named, pick_shard, store

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LOG = logging.getLogger("foqlens.small_corpus_masks")
# prompts a chunk of the pass holds between two saves: a shard's ~950 prompts are 4 chunks of about a minute and a half,
# the whole 5% ~19; a stop or a crash loses one chunk at most
CHUNK = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--model-source", choices=[s.value for s in ModelSource], default=ModelSource.FILE.value)
    parser.add_argument("--base", choices=sorted(PUBLISHED), default=None, help="the model cut over a published base")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/small-corpus.toml"))
    parser.add_argument("--source", choices=sorted(ADDRESS_SOURCES), required=True)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="prompts longer than this get no mask (NaN): a backward pass over them runs out of memory")
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    parser.add_argument("--only", type=Path, default=None,
                        help="a json list of [corpus, id]: read these questions of the draw and no others")
    parser.add_argument("--also", type=Path, default=None,
                        help="a json list of [corpus, id]: lay these questions out on top of the share")
    parser.add_argument("--out", type=Path, default=Path("runs/masks"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out / args.model
    runlog.setup(out / "logs" / f"small_corpus_masks-{run}.jsonl", {"run": run, "script": "small_corpus_masks"})
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, args.base) if args.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, source=ModelSource(args.model_source), directory=directory)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    small = config.read(args.corpus_config, config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name, also=named(args.also)), small, args.shard)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    source = bench.source(args.source)
    asked = found.laid + found.calibration
    if args.only:  # a batch named by hand: its masks are written beside the run that holds the rest, not instead of it
        wanted_ids = {tuple(pair) for pair in named(args.only)}
        asked = [(c, r) for c, r in asked if (c, r.id) in wanted_ids]
        LOG.info("%d of %d questions named by %s", len(asked), len(wanted_ids), args.only)
    prompts = found.prompts(fmt, asked)
    lengths = np.array([len(ids) for ids in bench.tokenizer(prompts)["input_ids"]])
    # a backward pass over a prompt longer than the cap runs out of memory alone: its mask stays NaN, and every reader
    # counts such questions apart
    fits = lengths <= args.max_tokens if args.max_tokens else np.ones(len(prompts), dtype=bool)
    LOG.info("%d of %d prompts within %s tokens", int(fits.sum()), len(prompts), args.max_tokens,
             extra={"within": int(fits.sum()), "prompts": len(prompts)})
    wanted = np.flatnonzero(fits)
    chunks = [wanted[i:i + CHUNK] for i in range(0, len(wanted), CHUNK)]
    batch = f"-{args.only.stem}" if args.only else ""  # a named batch writes beside the full run, not over it
    stem = f"{args.source}-{args.model_source}-{args.base or 'own'}-{args.corpus_config.stem}{suffix}{batch}"
    checkpoint = Checkpoint(out / f"{stem}.partial.npz", plan_of(vars(args), len(prompts), [c.tolist() for c in chunks]))
    masks = np.full((len(prompts), bench.ctl.n_blocks), np.nan, dtype=np.float32)
    first = 0
    if (kept := checkpoint.load()) is not None:  # a stopped or crashed pass goes on from its last saved chunk
        masks, first = kept["masks"], int(kept["chunks_done"])
        LOG.info("resumed after %d of %d chunks from %s", first, len(chunks), checkpoint.path)
    for k, chunk in enumerate(chunks[first:], start=first):
        with stage(LOG, f"chunk {k + 1} of {len(chunks)}: {len(chunk)} prompts"):
            masks[chunk] = bench.masks([prompts[i] for i in chunk], [source])[source.name]
        checkpoint.save(masks=masks, chunks_done=k + 1)
    modules = bench.ctl.modules.values()
    kept = store(found, masks, block_layers(bench.ctl), block_kinds(bench.ctl),
                 np.concatenate([m.block_sizes() * m.in_features for m in modules]),
                 {"model": name, "source": args.source, "model_source": args.model_source, "base": args.base,
                  "level": Level.BF16.name, "corpus": str(args.corpus_config), "corpora": args.corpora,
                  "max_tokens": args.max_tokens, "without_mask": int((~fits).sum())}, pairs=asked)
    target = out / f"{stem}.npz"
    with stage(LOG, f"save {target}"):
        kept.save(target)
    checkpoint.clear()
    return target


if __name__ == "__main__":
    main()
