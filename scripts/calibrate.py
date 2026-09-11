"""Scale calibration: the working range of the cosine before thresholds are fixed.

Reads prompts/calibration/paraphrase.jsonl and unrelated.jsonl (pairs {"a": ..., "b": ...}) and,
for every pair, computes the cosine between the two texts for:

- the representation: hidden states of the middle decoder layer, mean-pooled over tokens;
- the mask vectors of all three center modes of prereg/ADDENDUM-01.md, raw and with the
  background subtracted (the mean mask over every calibration text).

Paraphrases give the top of the scale, unrelated pairs its floor. The summary holds the
distribution (mean, std, quantiles) per pair kind and vector kind in
runs/calibration/<model>/summary.json. Calibration is not a result: the summary is printed.

    uv run python scripts/calibrate.py
    uv run python scripts/calibrate.py --limit 3 --out /tmp/calibration     # smoke check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foqlens import model as fm
from foqlens.precision import install
from foqlens.scoring import MODES, BlockScorer
from foqlens.separation import mean_pool

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
KINDS = ("paraphrase", "unrelated")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def read_pairs(path: Path, limit: int | None) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        pairs = [(d["a"], d["b"]) for d in map(json.loads, f) if d]
    return pairs[:limit] if limit else pairs


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def distribution(values: list[float]) -> dict:
    v = np.asarray(values)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "quantiles": {str(q): float(np.quantile(v, q)) for q in QUANTILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts/calibration"))
    parser.add_argument("--out", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--limit", type=int, default=None, help="pairs per kind, for a smoke check")
    args = parser.parse_args()

    pairs = {kind: read_pairs(args.prompts_dir / f"{kind}.jsonl", args.limit) for kind in KINDS}
    texts = sorted({t for kind in KINDS for pair in pairs[kind] for t in pair})

    model_id = MODELS[args.model]
    model, tokenizer = fm.load(model_id, attn_implementation="eager")
    scorer = BlockScorer(install(model).modules)
    middle = len(fm.text_layers(model)) // 2 + 1  # hidden_states index of the middle layer's output

    rep, masks = {}, {mode: {} for mode in MODES}
    for text in texts:
        rep[text] = mean_pool(fm.hidden_states(model, tokenizer, text))[middle]
        for mode, (vec, _) in scorer.score(model, tokenizer, text).items():
            masks[mode][text] = vec
    background = {mode: np.mean(list(masks[mode].values()), axis=0) for mode in MODES}

    summary = {
        "model": model_id,
        "revision": fm.REVISIONS[model_id],
        "attn_implementation": "eager",
        "middle_layer_index": middle,
        "pairs": {kind: len(pairs[kind]) for kind in KINDS},
        "cosine": {},
    }
    for kind in KINDS:
        entry = {"representation": distribution([cosine(rep[a], rep[b]) for a, b in pairs[kind]])}
        for mode in MODES:
            m, bg = masks[mode], background[mode]
            entry[f"mask_{mode}_raw"] = distribution([cosine(m[a], m[b]) for a, b in pairs[kind]])
            entry[f"mask_{mode}_background_subtracted"] = distribution(
                [cosine(m[a] - bg, m[b] - bg) for a, b in pairs[kind]]
            )
        summary["cosine"][kind] = entry

    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key in summary["cosine"]["paraphrase"]:
        p, u = summary["cosine"]["paraphrase"][key], summary["cosine"]["unrelated"][key]
        print(f"{key:40} paraphrase {p['mean']:+.3f} ± {p['std']:.3f}   unrelated {u['mean']:+.3f} ± {u['std']:.3f}")
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
