"""The precision fields read (foqlens.field_analysis), from the kept files, never from the model: for every field its
levels and bits, the shares of its spread, how it agrees with the other fields and the reference a question, the
patterns it is made of, and its zones - the centres, the profile of the value around them in layers and the radius of
every rung.

    uv run python scripts/precision_report.py --fields "runs/oracles/e2b-it/precision-fields-*-shard*of5.npz" \
        --out runs/oracles/e2b-it/precision-report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens.field_analysis import (
    agreement,
    centres,
    clusters_against_corpora,
    consensus,
    holm,
    jaccard,
    patterns,
    permutation_p,
    profile,
    radius,
    tubes,
    variance_shares,
    zone_stops,
)
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, write_json
from foqlens.oracle_overlay import block_group_ids, bootstrap
from foqlens.precision_field import FIELD_VALUE, values
from foqlens.quant import Level
from foqlens.small_corpus import StoredMasks

REACH = 8  # layers from a centre the profile is read over: a quarter of E2B's 35
PATTERNS = [1, 2, 3, 4, 6, 8, 12]
DRAWS, LEVEL, SEED = 1000, 0.95, 0
# shuffles of the corpus labels. The smallest p a permutation test can give is 1 / (draws + 1), and Holm multiplies it
# by the size of the family - with ~18 fields, 200 draws cannot reach 0.05 however strong the effect is; 500 can.
PERMUTATIONS = 500
BASE = FIELD_VALUE[Level.D2]


def rung_bits(levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The nominal bits a weight of every question's layout [questions, groups] of level codes: [questions]."""
    bits = np.full(256, np.nan)
    for lv in Level:
        bits[int(lv)] = lv.bits
    return (bits[levels] * weights).sum(axis=1) / weights.sum()


def summary(v: np.ndarray) -> dict:
    """The median, the mean and the interquartile range of a sample, NaNs left out."""
    v = v[~np.isnan(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def interval(v: np.ndarray) -> list[float]:
    v = v[~np.isnan(v)]
    return list(bootstrap(v, np.mean, DRAWS, SEED, LEVEL)) if len(v) > 1 else [np.nan, np.nan]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True, help="the precision fields' files (a glob joins shards)")
    parser.add_argument("--weights", default="runs/masks/e2b-it/error_energy-file-bartowski-Q2_K-small-corpus.npz",
                        help="a kept masks file: the weights of the blocks, summed into groups")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources", "ratios", "rungs"})
    names = z["groups"].tolist()
    layers = np.array([int(n.split(".")[0]) for n in names])
    kinds = np.array([n.split(".")[1] for n in names])
    kept = StoredMasks.read(args.weights)
    weights = np.bincount(block_group_ids(kept.block_layer, kept.block_kind, names), weights=kept.block_weights,
                          minlength=len(names))
    corpus = z["corpus"]
    sources = z["sources"].tolist() + (["reference"] if "levels_reference" in z else [])
    report = {"questions": int(len(corpus)), "corpora": {c: int((corpus == c).sum()) for c in np.unique(corpus)},
              "ratios": {r: {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}
                         for r, x in zip(z["rungs"].tolist(), z["ratios"])},
              "weights_attention_over_mlp": float(np.median(weights[kinds == "attention"])
                                                  / np.median(weights[kinds == "mlp"])),
              "fields": {}, "agreement": {}}
    levels = {s: z[f"levels_{s}"] for s in sources}
    has = {s: (levels[s] != 255).all(axis=1) for s in sources}
    for s in sources:
        lv, ok = levels[s][has[s]], has[s]
        v = values(lv)
        count = {Level(k).name: float((lv == k).mean()) for k in (int(Level.D2), int(Level.D4), int(Level.D6),
                                                                    int(Level.D8))}
        entry = {"questions": int(ok.sum()), "share_of_groups": count,
                 "bits_a_weight": summary(rung_bits(lv, weights)), "d8_groups": summary((lv == int(Level.D8)).sum(1)),
                 "variance_shares": variance_shares(v, corpus[ok]) if ok.sum() > 2 else None}
        if ok.sum() > 2:
            # is the corpus's share of the spread more than shuffled labels give, and do the questions cluster along
            # the corpora at all - the coarse half an address has to predict, before the question's own pattern
            entry["clusters"] = clusters_against_corpora(v, corpus[ok], SEED)
            entry["corpus_share_p"] = permutation_p(lambda lab: variance_shares(v, lab)["corpus"], corpus[ok],
                                                    PERMUTATIONS, SEED)
        if f"nll_{s}" in z:  # a field read at a threshold; the derived maps and the reference hold no threshold
            entry["held"] = float((z[f"nll_{s}"][ok] <= z[f"target_{s}"][ok] + float(z["tolerance"])).mean())
            entry["batches"] = summary(z[f"batches_{s}"][ok].astype(float))
        if s == "reference":
            entry["raised_by"] = {int(k): int((z["steps"][ok] == k).sum()) for k in np.unique(z["steps"][ok])}
            alone = z["alone_levels"][ok]
            entry["alone_share_of_groups"] = {Level(k).name: float((alone == k).mean())
                                             for k in (int(Level.D2), int(Level.D4), int(Level.D6), int(Level.D8))}
        # the zones of the field: the centres (a local maximum above the base, of any height), the profile of the
        # value around them, and every zone in the filter's notation - its peak, its radius and its rungs' stops
        found = np.stack([centres(row, layers, BASE) for row in v]) if len(v) else np.zeros((0, len(names)), bool)
        curves = profile(v, layers, kinds, found, REACH)
        rungs = [FIELD_VALUE[lv] for lv in (Level.D8, Level.D6, Level.D4)]
        zones = [zone for q in range(len(v)) for zone in zone_stops(v[q], layers, found[q], BASE, rungs)]
        entry["zones"] = {
            "centres_a_question": summary(found.sum(axis=1).astype(float)),
            "centre_layers": np.bincount(layers[np.nonzero(found)[1]], minlength=layers.max() + 1).tolist(),
            "centre_kinds": {k: int(found[:, kinds == k].sum()) for k in np.unique(kinds)},
            "peaks": {Level(k).name: int(sum(zone["peak"] == FIELD_VALUE[Level(k)] for zone in zones))
                      for k in (int(Level.D4), int(Level.D6), int(Level.D8))},
            "radius_in_layers": summary(np.array([float(zone["radius"]) for zone in zones])),
            # relative, so that a rule read here carries to a network of another depth
            "centre_depth_share": summary(layers[np.nonzero(found)[1]] / max(layers.max(), 1)),
            "radius_share_of_depth": summary(np.array([zone["radius"] / (layers.max() + 1) for zone in zones])),
            "stops": {Level(k).name: summary(np.array([zone["stops"][FIELD_VALUE[Level(k)]] for zone in zones
                                                       if FIELD_VALUE[Level(k)] in zone["stops"]]))
                      for k in (int(Level.D8), int(Level.D6), int(Level.D4))},
            "tubes_a_question": summary(np.array([float(len(tubes(row, layers, BASE))) for row in v])),
            "tube_length_in_layers": summary(np.array([t["layers"] for row in v for t in tubes(row, layers, BASE)],
                                                      dtype=float)),
            "tube_widest": summary(np.array([t["widest"] for row in v for t in tubes(row, layers, BASE)], dtype=float)),
            "tube_share_of_depth": summary(np.array([t["to"] - t["from"] for row in v
                                                     for t in tubes(row, layers, BASE)], dtype=float)),
            "profile_same_kind": [None if np.isnan(x) else round(float(x), 4) for x in curves["same"]],
            "profile_other_kind": [None if np.isnan(x) else round(float(x), 4) for x in curves["other"]],
            "radius_of_rung": {Level(k).name: radius(curves["same"], FIELD_VALUE[Level(k)]) for k in (3, 2)},
        }
        heights = np.maximum(v - BASE, 0.0)
        if len(heights) > max(PATTERNS):
            entry["patterns_explained"] = {p["k"]: round(p["explained"], 4) for p in patterns(heights, PATTERNS)}
        report["fields"][s] = entry
    # the agreement of every two fields on the questions both hold
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            ok = has[a] & has[b]
            if ok.sum() < 2:
                continue
            va, vb = values(levels[a][ok]), values(levels[b][ok])
            got = agreement(va, vb, levels[a][ok], levels[b][ok])
            row = {k: {"mean": float(np.nanmean(x)), "ci": interval(x)} for k, x in got.items()}
            # a rank says how two fields order the groups; this says whether they put their peaks in the same places
            shared = jaccard(np.stack([centres(q, layers, BASE) for q in va]),
                             np.stack([centres(q, layers, BASE) for q in vb]))
            row["centres"] = {"mean": float(np.nanmean(shared)), "ci": interval(shared[~np.isnan(shared)])}
            report["agreement"][f"{a}|{b}"] = row
    # the corpus's share of the spread is tested once per field, so the p-values are a family: Holm over all of them
    family = {s: f["corpus_share_p"] for s, f in report["fields"].items() if "corpus_share_p" in f}
    for s, corrected in holm(family).items():
        report["fields"][s]["corpus_share_p_holm"] = corrected
    # the map of intervals (Volodya 20.09): a group's level is not one number but the band the oracles leave it -
    # where the band is zero the group is fixed, where it is wide the bridge may choose and pay less
    oracle_maps = [s for s in sources if s not in ("reference", "common", "middle", "together")
                   and not s.endswith("_per_weight")]
    both = np.logical_and.reduce([has[s] for s in oracle_maps])
    if both.sum() > 1:
        stack = np.stack([levels[s][both] for s in oracle_maps]).astype(int)
        low, high = stack.min(axis=0), stack.max(axis=0)
        width = high - low
        report["intervals"] = {
            "oracles": oracle_maps, "questions": int(both.sum()),
            "width_in_rungs": summary(width.ravel().astype(float)),
            "fixed_share": float((width == 0).mean()),
            "fixed_share_by_layer": [round(float((width[:, layers == la] == 0).mean()), 4)
                                     for la in range(layers.max() + 1)],
            "lowest_bits_a_weight": summary(rung_bits(low.astype(np.uint8), weights)),
            "highest_bits_a_weight": summary(rung_bits(high.astype(np.uint8), weights)),
            "raised_by_every_oracle": float((low > int(Level.D2)).mean()),
            "raised_by_any_oracle": float((high > int(Level.D2)).mean()),
        }
    # the consensus of the oracles (not the reference): its band of uncertainty and its own patterns
    oracles = [s for s in sources if s != "reference" and not s.endswith("_per_weight")]
    ok = np.logical_and.reduce([has[s] for s in oracles])
    if ok.sum() > max(PATTERNS):
        median, band = consensus(np.stack([values(levels[s][ok]) for s in oracles]))
        report["consensus"] = {"oracles": oracles, "questions": int(ok.sum()),
                               "band": summary(band.ravel()), "band_by_layer": [
                                   round(float(band[:, layers == la].mean()), 4) for la in range(layers.max() + 1)],
                               "variance_shares": variance_shares(median, corpus[ok]),
                               "patterns_explained": {p["k"]: round(p["explained"], 4)
                                                      for p in patterns(np.maximum(median - BASE, 0), PATTERNS)}}
    write_json(args.out, report)
    print(args.out)
    return args.out


if __name__ == "__main__":
    main()
