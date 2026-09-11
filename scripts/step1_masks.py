"""Step 1: are per-block masks separable by topic and concentrated.

Reads prompts/<domain>.jsonl (one {"text": ...} per line), computes the naive mask vectors of
every query in all three center modes of experiments/E001-run1-exploration/ADDENDUM-01.md (norm, pooled, attention) in one
eager-attention pass, and writes:

- runs/E001-run1-exploration/step1/<model>/raw/vectors_<mode>.npy and raw/queries.json - mask vectors, labels, the
  token positions and tokens each mode was built on;
- runs/E001-run1-exploration/step1/<model>/summary.json - for every mode: separation for every pair of domains and over
  all domains, on raw vectors and with the background subtracted (the mean vector over all
  queries), plus the concentration (Gini, normalized entropy) of every query's mask.

Blind analysis (see prereg): this script prints no metrics. During the runs only check that it
did not crash; summaries are opened together once every run is done.

    uv run python scripts/step1_masks.py
    uv run python scripts/step1_masks.py --domains biology math chemistry physics biophysics
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from foqlens import model as fm
from foqlens.precision import install
from foqlens.scoring import DEFAULT_TOP_K, MODES, BlockScorer, GradientScorer, gini, normalized_entropy
from foqlens.separation import permutation_test, separation

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
DEBUG_DOMAINS = ["biology", "math", "chemistry", "physics", "biophysics"]


def read_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def separations(vectors: np.ndarray, labels: list[str], domains: list[str], n_perm: int = 0) -> dict:
    """Separation over all domains and for every pair; with n_perm > 0 also a label-permutation test per pair."""
    labels_arr = np.asarray(labels)
    out = {"all": separation(vectors, labels).as_dict(), "pairs": {}}
    for a, b in itertools.combinations(domains, 2):
        mask = np.isin(labels_arr, [a, b])
        pair = separation(vectors[mask], list(labels_arr[mask])).as_dict()
        if n_perm:
            pair["permutation"] = permutation_test(vectors[mask], list(labels_arr[mask]), n_perm=n_perm)
        out["pairs"][f"{a}|{b}"] = pair
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=DEBUG_DOMAINS)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--out", type=Path, default=Path("runs/E001-run1-exploration/step1"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="center modes to compute")
    parser.add_argument("--permutations", type=int, default=0, help="label permutations per domain pair (0 = none)")
    parser.add_argument(
        "--instrument", choices=["centers", "gradient"], default="centers",
        help="centers: the naive score of ADDENDUM-01 (--modes apply); gradient: gradient x activation of ADDENDUM-02",
    )
    args = parser.parse_args()
    modes = ("gradient",) if args.instrument == "gradient" else tuple(args.modes)

    # a domain may be given with a subdirectory (heldout/history); its label is the bare name
    queries = []
    for spec in args.domains:
        queries += [(Path(spec).name, t) for t in read_prompts(args.prompts_dir / f"{spec}.jsonl")]
    domain_names = [Path(spec).name for spec in args.domains]

    model_id = MODELS[args.model]
    model, tokenizer = fm.load(model_id, attn_implementation="eager")
    modules = install(model).modules
    if args.instrument == "gradient":
        scorer = GradientScorer(model, modules)
        score = lambda text: scorer.score(model, tokenizer, text)  # noqa: E731
    else:
        scorer = BlockScorer(modules, top_k=args.top_k)
        score = lambda text: scorer.score(model, tokenizer, text, modes=modes)  # noqa: E731

    vectors = {mode: [] for mode in modes}
    records = []
    for domain, text in queries:
        scored = score(text)
        ids = tokenizer(text).input_ids
        record = {"domain": domain, "text": text, "positions": {}, "tokens": {}}
        for mode, (vec, positions) in scored.items():
            vectors[mode].append(vec)
            record["positions"][mode] = positions
            record["tokens"][mode] = tokenizer.convert_ids_to_tokens([ids[p] for p in positions])
        records.append(record)
    labels = [r["domain"] for r in records]

    out_dir = args.out / args.model
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / "queries.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    by_mode = {}
    for mode in modes:
        stacked = np.stack(vectors[mode])
        np.save(out_dir / "raw" / f"vectors_{mode}.npy", stacked)
        background = stacked.mean(axis=0, keepdims=True)
        by_mode[mode] = {
            "separation": {
                "raw": separations(stacked, labels, domain_names, args.permutations),
                "background_subtracted": separations(stacked - background, labels, domain_names, args.permutations),
            },
            "concentration": [
                {"domain": d, "gini": gini(v), "normalized_entropy": normalized_entropy(v)}
                for d, v in zip(labels, stacked)
            ],
        }

    summary = {
        "model": model_id,
        "revision": fm.REVISIONS[model_id],
        "attn_implementation": "eager",
        "instrument": args.instrument,
        "top_k": args.top_k,
        "n_blocks": scorer.n_blocks,
        "permutations": args.permutations,
        "domains": {d: labels.count(d) for d in domain_names},
        "modes": by_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"scored {len(records)} queries x {len(modes)} modes over {scorer.n_blocks} blocks -> {out_dir}")


if __name__ == "__main__":
    main()
