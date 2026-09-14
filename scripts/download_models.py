"""Download the bench models at their pinned revisions.

Weights are not stored in the repository. This fetches them from Hugging Face into the
regular HF cache (or HF_HOME, if set), exactly at the commits in foqlens.model.REVISIONS,
so every researcher runs the bench on the same bytes.

    uv run python scripts/download_models.py            # E2B, ~10 GB - enough for the tests
    uv run python scripts/download_models.py e4b        # E4B, ~16 GB - the confirmation model
    uv run python scripts/download_models.py e2b-it     # E2B-it, ~10 GB - the model the corpus is selected on
    uv run python scripts/download_models.py e2b e4b
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from foqlens.model import E2B, E2B_IT, E4B, E4B_IT, REVISIONS

MODELS = {"e2b": E2B, "e4b": E4B, "e2b-it": E2B_IT, "e4b-it": E4B_IT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # no list default here: argparse checks a list default against choices as a whole and rejects it
    parser.add_argument("models", nargs="*", choices=sorted(MODELS), help="models to fetch (default: e2b)")
    args = parser.parse_args()

    for key in args.models or ["e2b"]:
        repo_id = MODELS[key]
        revision = REVISIONS[repo_id]
        print(f"{repo_id} @ {revision}")
        path = snapshot_download(repo_id, revision=revision)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
