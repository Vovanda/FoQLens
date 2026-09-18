"""Cut a downloaded model into a .refocustensors folder the bench runs from without the Hugging Face checkpoint.

The checkpoint must be downloaded first (scripts/download_models.py); the folder goes to
~/.cache/foqlens/models/<name>@<revision> (FOQLENS_HOME moves it) unless --out says otherwise.

    uv run python scripts/download_models.py e2b-it
    uv run python scripts/cut_model.py e2b-it
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from foqlens.model import E2B, E2B_IT, E4B, E4B_IT, REVISIONS
from foqlens.refocustensors import model_directory, source_id, write

MODELS = {"e2b": E2B, "e4b": E4B, "e2b-it": E2B_IT, "e4b-it": E4B_IT}
GIB = 2**30


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--out", type=Path, help="the model folder (default: ~/.cache/foqlens/models/...)")
    args = parser.parse_args(argv)
    model_id = MODELS[args.model]
    source = Path(snapshot_download(model_id, revision=REVISIONS[model_id], local_files_only=True))
    out = args.out or model_directory(model_id)
    start = time.perf_counter()
    path = write(source, out, source_id(model_id))
    source_bytes = sum(f.stat().st_size for f in source.glob("*.safetensors"))
    print(f"{path}: {path.stat().st_size / GIB:.2f} GiB, the source {source_bytes / GIB:.2f} GiB, "
          f"{time.perf_counter() - start:.0f} s")
    return path


if __name__ == "__main__":
    main()
