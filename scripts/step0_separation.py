"""Step 0: do topics separate in the representations of the model.

Reads prompts/<domain>.jsonl (one {"text": ...} per line), mean-pools the hidden states of every
layer for every prompt, and writes per-layer separation metrics to
runs/E001-run1-exploration/step0/<model>/summary.json. Index i in "layers" is hidden_states[i]: 0 is the embeddings,
i + 1 is the output of decoder layer i.

This is the one step that is looked at right away (see prereg): it checks that the model is fit
for the bench, it is not a result.

    uv run python scripts/step0_separation.py                         # E2B, biology vs math
    uv run python scripts/step0_separation.py --model e4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foqlens import model as fm
from foqlens.separation import mean_pool, per_layer

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}


def read_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=["biology", "math"])
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--out", type=Path, default=Path("runs/E001-run1-exploration/step0"))
    args = parser.parse_args()

    texts, labels = [], []
    for domain in args.domains:
        domain_texts = read_prompts(args.prompts_dir / f"{domain}.jsonl")
        texts += domain_texts
        labels += [domain] * len(domain_texts)

    model_id = MODELS[args.model]
    model, tokenizer = fm.load(model_id)
    pooled = np.stack([mean_pool(fm.hidden_states(model, tokenizer, t)) for t in texts])

    layers = per_layer(pooled, labels)
    middle = len(fm.text_layers(model)) // 2
    summary = {
        "model": model_id,
        "revision": fm.REVISIONS[model_id],
        "domains": {d: labels.count(d) for d in args.domains},
        "middle_layer_index": middle + 1,
        "separates_at_middle_layer": {
            variant: layers[middle + 1][variant]["separates"] for variant in ("raw", "centered")
        },
        "layers": layers,
    }
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
