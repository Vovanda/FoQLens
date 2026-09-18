"""Cut a downloaded model into a .refocustensors folder the bench runs from without the Hugging Face checkpoint.

The checkpoint must be downloaded first (scripts/download_models.py); the folder goes to
~/.cache/foqlens/models/<name>@<revision> (FOQLENS_HOME moves it) unless --out says otherwise.

    uv run python scripts/download_models.py e2b-it
    uv run python scripts/cut_model.py e2b-it
    uv run python scripts/cut_model.py e2b-it --depth D8   # the stack to D8 only, read resident
    uv run python scripts/cut_model.py e2b-it --base bartowski-Q2_K   # over a published file's base, in ...@rev-bartowski-Q2_K
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from foqlens.gguf_weights import PUBLISHED, GgufWeights, published_path
from foqlens.model import E2B, E2B_IT, E4B, E4B_IT, REVISIONS
from foqlens.quant import Level
from foqlens.refinements import ForeignLadder
from foqlens.refocustensors import BENCH_COPY, model_directory, source_id, write

MODELS = {"e2b": E2B, "e4b": E4B, "e2b-it": E2B_IT, "e4b-it": E4B_IT}
GIB = 2**30
DEPTHS = [level.name for level in Level if level.depth]  # D2 ... D8


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--out", type=Path, help="the model folder (default: ~/.cache/foqlens/models/...)")
    parser.add_argument("--depth", choices=DEPTHS, help="stop the stack at this level, with no exact tail: a smaller "
                        "file read resident only (default: every depth and the tail to the source)")
    parser.add_argument("--base", choices=sorted(PUBLISHED), help="take every module's base from this published GGUF "
                        "file where it is a k-quant (Q2_K, Q3_K, Q4_K, Q6_K) and build the refinements and the tail "
                        "over it; the folder is named after it")
    args = parser.parse_args(argv)
    model_id = MODELS[args.model]
    source = Path(snapshot_download(model_id, revision=REVISIONS[model_id], local_files_only=True))
    suffix = "-".join(part for part in (args.base, args.depth) if part) or None
    out = args.out or model_directory(model_id, suffix)
    copy = ForeignLadder(GgufWeights(published_path(args.base)).base_blocks) if args.base else BENCH_COPY
    start = time.perf_counter()
    path = write(source, out, source_id(model_id), copy=copy,
                 depth=None if args.depth is None else Level[args.depth].depth)
    source_bytes = sum(f.stat().st_size for f in source.glob("*.safetensors"))
    print(f"{path}: {path.stat().st_size / GIB:.2f} GiB, the source {source_bytes / GIB:.2f} GiB, "
          f"{time.perf_counter() - start:.0f} s")
    return path


if __name__ == "__main__":
    main()
