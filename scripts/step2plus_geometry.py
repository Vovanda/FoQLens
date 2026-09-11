"""Step 2+: mask geometry on the sealed step 1 vectors (tests defined in foqlens.geometry).

Reads runs/step1/<model>/raw/vectors_<mode>.npy and raw/queries.json, checks their sha256
against the values recorded in the sealed step 1 commit, computes mean-pooled middle-layer
representations of the same queries (inputs, not results) for the linearity test, and writes
runs/step2plus/<model>/summary.json.

Every test is run per center mode and per vector variant: raw masks, and masks with the
background (the mean mask over all debugging queries) subtracted. Concentration needs
non-negative masks, so on the background-subtracted variant it uses the positive part.

Blind analysis: this script prints no metrics. Its summary is opened together with step 1.

    uv run python scripts/step2plus_geometry.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from foqlens import geometry as g
from foqlens import model as fm
from foqlens.scoring import MODES
from foqlens.separation import mean_pool

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
# recorded in the sealed step 1 commit c407c45
SEALED_SHA256 = {
    "e2b": {
        "vectors_norm.npy": "172c9910239448cfd46c6892da016eab62875ca890e27adb583d337a588fac9e",
        "vectors_pooled.npy": "41147e5d1ca8aed8c1279a307e853e81025c44b794e815d892fa039a778f2169",
        "vectors_attention.npy": "6c8bfb0fcc0fb00285a9440b5da014d0c96a3b057cc33f71f18ad158c175da15",
        "queries.json": "c6ed2b5abf0b431d1445e4abedc9ca91250dee11f301cc1bb4c1061d42d3475b",
    }
}
MIXED, A, B = "biophysics", "biology", "physics"
SCIENCE = ["biology", "chemistry", "physics", "biophysics"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_of(means: dict[str, np.ndarray], nonneg: bool) -> dict:
    out = {
        "additivity": g.fit_two(means[MIXED], means[A], means[B]),
        "concentration": {
            d: g.concentration(v if nonneg else np.clip(v, 0, None)) for d, v in means.items()
        },
        "by_fraction": {},
    }
    for frac in g.FRACTIONS:
        out["by_fraction"][str(frac)] = {
            "junction": g.junction(means[MIXED], means[A], means[B], frac),
            "isthmus": g.isthmus(means[MIXED], means[A], means[B], frac),
            "hierarchy": g.hierarchy({d: means[d] for d in SCIENCE}, frac),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--step1", type=Path, default=Path("runs/step1"))
    parser.add_argument("--out", type=Path, default=Path("runs/step2plus"))
    parser.add_argument("--skip-hash-check", action="store_true", help="for smoke checks on other vectors")
    args = parser.parse_args()

    raw = args.step1 / args.model / "raw"
    if not args.skip_hash_check:
        for name, expected in SEALED_SHA256[args.model].items():
            if sha256(raw / name) != expected:
                raise SystemExit(f"{name} does not match the sealed step 1 commit")

    records = json.loads((raw / "queries.json").read_text(encoding="utf-8"))
    labels = [r["domain"] for r in records]
    domains = list(dict.fromkeys(labels))

    model_id = MODELS[args.model]
    model, tokenizer = fm.load(model_id, attn_implementation="eager")
    middle = len(fm.text_layers(model)) // 2 + 1
    reps = np.stack([mean_pool(fm.hidden_states(model, tokenizer, r["text"]))[middle] for r in records])
    rep_means = g.domain_means(reps - reps.mean(axis=0, keepdims=True), labels, domains)

    by_mode = {}
    for mode in MODES:
        vectors = np.load(raw / f"vectors_{mode}.npy")
        variants = {"raw": vectors, "background_subtracted": vectors - vectors.mean(axis=0, keepdims=True)}
        by_mode[mode] = {}
        for name, v in variants.items():
            means = g.domain_means(v, labels, domains)
            entry = geometry_of(means, nonneg=(name == "raw"))
            entry["linearity"] = g.linearity(rep_means, means, MIXED, A, B)
            by_mode[mode][name] = entry

    summary = {
        "model": model_id,
        "revision": fm.REVISIONS[model_id],
        "step1_sha256": SEALED_SHA256.get(args.model) if not args.skip_hash_check else None,
        "mixed": MIXED,
        "components": [A, B],
        "fractions": list(g.FRACTIONS),
        "domains": {d: labels.count(d) for d in domains},
        "modes": by_mode,
    }
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"geometry over {len(records)} queries x {len(MODES)} modes -> {out_dir}")


if __name__ == "__main__":
    main()
