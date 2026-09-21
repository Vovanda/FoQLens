"""The bridge from the address to a precision field (foqlens.field_bridge): fitted on the questions of one part of the
small corpus, read on another it has never seen, and written in the format of the precision fields, so the same script
answers its maps beside the oracles' (scripts/oracle_answers.py). No model is loaded - the addresses and the maps are
read from their files.

    uv run python scripts/field_bridge.py --fit "runs/oracles/e2b-it/precision-fields-*shard1of5.npz" \
        --read "runs/oracles/e2b-it/precision-fields-*shard2of5.npz" --target common \
        --out runs/oracles/e2b-it/bridge-shard2of5.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foqlens.field_bridge import BRIDGES, EVEN_STOPS, bits_of, lift_of, shift_for_bits, to_levels
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, save_npz_atomic, write_json
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.projection import Projection
from foqlens.small_corpus import StoredMasks

NEAREST_K = 8  # fitted questions a nearest bridge averages: the address names its question, its neighbours fill it in
RELATIVE_RIDGE = 0.1  # as the working address's projection (docs/bench-math.md, section 5)
NOT_READ = 255


def group_address(masks: StoredMasks, keys: list[tuple[str, str]], names: list[str]) -> np.ndarray:
    """The address of every question over the groups: |score| summed over a group's blocks, [questions, groups]."""
    rows = masks.rows_of(keys)
    return to_groups(np.abs(rows), block_group_ids(masks.block_layer, masks.block_kind, names), len(names))


def projected_address(masks: StoredMasks, names: list[str], layers: np.ndarray, depth_share: float,
                      relative_ridge: float) -> tuple[np.ndarray, dict]:
    """The address of every question of the file as one pass would have it: read over the groups of the first
    `depth_share` of the layers and carried onto every group by a ridge fitted on the calibration questions
    (foqlens.projection, docs/bench-math.md, section 5). Returns [questions, groups] and what was fitted."""
    ids = block_group_ids(masks.block_layer, masks.block_kind, names)
    every = to_groups(np.abs(masks.masks), ids, len(names))
    early = layers < max(1, round(depth_share * (layers.max() + 1)))
    calibration = ~masks.laid
    x = every[calibration][:, early]
    alpha = relative_ridge * float(x.var(axis=0).mean()) * len(x)
    fitted = Projection.fit(x, every[calibration], alpha)
    return fitted.apply(every[:, early]).numpy(), {
        "layers_read": int(early.sum() // 2), "groups_read": int(early.sum()),
        "calibration_questions": int(calibration.sum()), "relative_ridge": relative_ridge}


def maps_of(fields: dict, target: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """A file's target maps as lifts, the questions that hold one, and their keys."""
    levels = fields[f"levels_{target}"]
    has = (levels != NOT_READ).all(axis=1)
    keys = list(zip(fields["corpus"].tolist(), fields["ids"].tolist()))
    return lift_of(levels[has]), has, [k for k, ok in zip(keys, has) if ok]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", required=True, help="the precision fields the bridges are fitted on")
    parser.add_argument("--read", required=True, help="the precision fields of the questions they are read on")
    parser.add_argument("--target", default="common", help="the map every bridge is fitted to (a field's name)")
    parser.add_argument("--address", default="runs/masks/e2b-it/hybrid-file-bartowski-Q2_K-small-corpus.npz")
    parser.add_argument("--weights", default="runs/masks/e2b-it/error_energy-file-bartowski-Q2_K-small-corpus.npz")
    parser.add_argument("--bridges", nargs="+", default=sorted(BRIDGES), choices=sorted(BRIDGES))
    # a zone is the hill around a peak, of any shape - no radius and no ball (Volodya 20.09)
    parser.add_argument("--slope", type=float, default=0.5, help="a hill holds while the address stays above this share of its peak")
    parser.add_argument("--diameter", type=float, default=0.2, help="the widest a zone may span, as a share of the depth")
    parser.add_argument("--peak-quantile", type=float, default=0.9,
                        help="a centre must pass this quantile of the question's address")
    parser.add_argument("--read-depth", type=float, default=None,
                        help="the share of the layers one pass reads the address on, carried onto the rest by the "
                             "projection; without it the address of the whole network is used")
    parser.add_argument("--gradient", default="runs/masks/e2b-it/answer_gradient-d8-file-bartowski-Q2_K-small-corpus-shard1of5.npz",
                        help="a gradient oracle's masks: the backward half the product bridge is fitted to")
    parser.add_argument("--keep", type=int, default=3, help="maps of one family (the zones' knobs) that go on to the answers")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fit, read = (read_npz_parts(p, RUN_FIELDS | {"sources", "ratios", "rungs"}) for p in (args.fit, args.read))
    names = fit["groups"].tolist()
    if read["groups"].tolist() != names:
        raise ValueError("the two runs read different groups")
    layers = np.array([int(n.split(".")[0]) for n in names])
    fit_maps, _, fit_keys = maps_of(fit, args.target)
    read_maps, read_has, read_keys = maps_of(read, args.target)
    masks = StoredMasks.read(args.address)
    kept = StoredMasks.read(args.weights)
    weights = np.bincount(block_group_ids(kept.block_layer, kept.block_kind, names), weights=kept.block_weights,
                          minlength=len(names))
    how = {"read": "every layer at bf16"}
    if args.read_depth is None:  # the address of the whole network: the ceiling of what an address could name
        fit_address, read_address = (group_address(masks, k, names) for k in (fit_keys, read_keys))
    else:  # the address as one pass has it: the first share of the layers, carried onto the rest by the projection
        projected, how = projected_address(masks, names, layers, args.read_depth, RELATIVE_RIDGE)
        at = {k: i for i, k in enumerate(zip(masks.corpus.tolist(), masks.ids.tolist()))}
        fit_address, read_address = (projected[[at[k] for k in keys]] for keys in (fit_keys, read_keys))
    fit_topics = np.array([c for c, _ in fit_keys])
    read_topics = np.array([c for c, _ in read_keys])
    # the price of memory is one for every question and fixed where the bridge was fitted: the bits the target maps
    # spend there (docs/bench-math.md, section 9)
    bits = float(bits_of(to_levels(fit_maps), weights).mean())

    made = {"target": args.target, "bits_asked": bits, "questions_fitted": len(fit_keys),
            "questions_read": len(read_keys), "address": how, "bridges": {}}
    arrays, sources = {}, []
    asked = []
    for name in args.bridges:  # the zones are a family: one map per reach and quantile, the knobs are swept, not guessed
        if name == "zones":
            asked += [(f"zones-{'own' if own else 'top'}",
                       {"layers": layers, "slope": args.slope, "quantile": args.peak_quantile, "own_height": own,
                        "diameter": args.diameter})
                      for own in (True, False)]
        elif name == "product":  # the backward half is fitted where a gradient oracle has run, on its own questions
            gradient = StoredMasks.read(args.gradient)
            of = to_groups(np.abs(gradient.masks),
                           block_group_ids(gradient.block_layer, gradient.block_kind, names), len(names))
            keys_of = list(zip(gradient.corpus.tolist(), gradient.ids.tolist()))
            seen = ~np.isnan(gradient.masks).any(axis=1) & ~gradient.laid  # its calibration questions only
            early = layers < max(1, round((args.read_depth or 0.3) * (layers.max() + 1)))
            x = group_address(masks, [k for k, ok in zip(keys_of, seen) if ok], names)[:, early]
            alpha = RELATIVE_RIDGE * float(x.var(axis=0).mean()) * len(x)
            asked.append((name, {"backward": Projection.fit(x, of[seen], alpha), "early": early}))
            made["product_fitted_on"] = int(seen.sum())
        else:
            asked.append((name, {"nearest": {"k": NEAREST_K},
                                 "ridge": {"relative_ridge": RELATIVE_RIDGE}}.get(name, {})))
    fit_levels = to_levels(fit_maps)
    built = []
    for label, knobs in asked:
        bridge = BRIDGES[label.split("-")[0]](**knobs).fit(fit_address, fit_maps, fit_topics)
        shift = shift_for_bits(bridge.read(fit_address, fit_topics), weights, bits)
        # the knobs are chosen where the bridge was fitted, never on the part it is read on
        on_fit = float((to_levels(bridge.read(fit_address, fit_topics), shift, EVEN_STOPS) == fit_levels).mean())
        built.append((on_fit, label, bridge, shift))
        made["bridges"][label] = {"shift": shift, "groups_on_the_target_level_fitted": on_fit}
    families = {}
    for on_fit, label, _, _ in built:  # of a family of knobs, the best few on the fitted part go on to the answers
        families.setdefault(label.split("-")[0], []).append((on_fit, label))
    keep = {label for name, rows in families.items() for _, label in sorted(rows, reverse=True)[:args.keep]}
    made["kept"] = sorted(keep)
    for on_fit, label, bridge, shift in built:
        if label not in keep:
            continue
        levels = to_levels(bridge.read(read_address, read_topics), shift, EVEN_STOPS)
        rows = np.full((len(read_has), len(names)), NOT_READ, dtype=np.uint8)
        rows[read_has] = levels
        arrays[f"levels_{label}"] = rows
        sources.append(label)
        made["bridges"][label] |= {
            "bits_a_weight": float(bits_of(levels, weights).mean()),
            "groups_on_the_target_level": float((levels == to_levels(read_maps)).mean()),
            "rungs_off_median": float(np.median(np.abs(levels.astype(int) - to_levels(read_maps).astype(int)))),
        }
    save_npz_atomic(args.out, groups=np.array(names), corpus=read["corpus"], ids=read["ids"],
                    sources=np.array(sources), tolerance=float(read["tolerance"]), target=args.target,
                    ratios=read["ratios"], rungs=read["rungs"], **arrays)
    write_json(args.out.with_suffix(".json"), made)
    print(json.dumps(made, indent=2))
    return args.out


if __name__ == "__main__":
    main()
