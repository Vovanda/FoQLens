"""One question's pass, written down layer by layer (foqlens.tracing), so that it is walked afterwards on the host.

The card is touched once, for seconds; the walk over the trace costs nothing and goes forward and back as often as a
formula for a block's importance has to be tried.

    uv run python scripts/trace_question.py --question tc_1516 --base bartowski-Q2_K
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from foqlens import config, refocustensors
from foqlens import model as fm
from foqlens.gguf_weights import PUBLISHED
from foqlens.field_bridge import to_levels
from foqlens.gpu_share import default_share
from foqlens.io import save_npz_atomic
from foqlens.pipeline import Bench
from foqlens.oracle_overlay import block_group_ids
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.regulator import block_kinds, block_layers
from foqlens.small_corpus import draw
from foqlens.tracing import Tracer

MODEL = fm.E2B_IT


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True, help="the id of a question of the frozen corpus")
    parser.add_argument("--base", choices=sorted(PUBLISHED), default=None)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/small-corpus.toml"))
    parser.add_argument("--level", default="d2", help="the rung the pass is read at while it is traced")
    parser.add_argument("--fields", default=None, help="unified fields (scripts/unify_fields.py): trace the pass "
                        "under an oracle's own map of this question instead of one rung everywhere")
    parser.add_argument("--oracle", default=None, help="whose map of the question the pass is read under")
    parser.add_argument("--out", type=Path, default=Path("runs/traces/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    args = parser.parse_args(argv)

    directory = refocustensors.model_directory(MODEL, args.base) if args.base else None
    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=directory)
    small = config.read(args.corpus_config, config.SmallCorpus)
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    found = draw(args.corpora, args.frozen, small, name)
    asked = [(corpus, record) for corpus, record in found.laid if record.id == args.question]
    if not asked:
        raise SystemExit(f"no question {args.question} in the draw of {args.corpora}")
    fmt = fm.prompt_format(MODEL, bench.tokenizer)
    prompt = found.prompts(fmt, asked)[0]

    if args.fields:  # the pass under an oracle's own map: every block reads the rung its group was read at
        if not args.oracle:
            raise SystemExit("--fields needs --oracle: whose map the pass is read under")
        z = np.load(args.fields)
        ids, groups = z["ids"].tolist(), z["groups"].tolist()
        of_group = to_levels(z[f"field_{args.oracle}"][ids.index(args.question)][None])[0]
        at = block_group_ids(block_layers(bench.ctl), block_kinds(bench.ctl), groups)
        bench.ctl.set_layout(of_group[at])
        read = f"{args.oracle}-map"
    else:
        bench.ctl.set_all(Level[args.level.upper()])
        read = args.level
    tracer = Tracer(bench.ctl)
    tracer.statics()
    tracer.attach(bench.model)
    try:
        encoded = fm.encode_left(bench.tokenizer, [prompt], bench.model.device)
        with torch.no_grad():
            bench.model(**encoded)
    finally:
        tracer.detach()
    trace = tracer.trace
    trace.tokens = encoded["input_ids"][0].tolist()

    names = sorted(trace.response)
    target = args.out / f"{args.question}-{args.base or 'own'}-{read}.npz"
    save_npz_atomic(target, modules=np.array(names), tokens=np.array(trace.tokens),
                    layers=np.array(trace.layers()),
                    **{f"state_{layer}": trace.state[layer] for layer in trace.layers()},
                    **{f"response_{i}": trace.response[n] for i, n in enumerate(names)},
                    **{f"injected_{i}": trace.injected[n] for i, n in enumerate(names)},
                    **{f"norms_{i}": trace.static[n].norms for i, n in enumerate(names)},
                    **{f"weights_{i}": trace.static[n].weights for i, n in enumerate(names)},
                    **{f"gap_{i}": trace.static[n].gap for i, n in enumerate(names)})
    print(f"{target}: {len(trace.layers())} layers, {len(names)} modules, {len(trace.tokens)} tokens")
    return target


if __name__ == "__main__":
    main()
