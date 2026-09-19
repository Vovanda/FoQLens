"""How two cuts of one model hold its weights: the share of weights out of the refinements' reach and the error at every
depth, module class by module class (refinements.stack_fit).

A refinement codes the rest within half its step, so a weight whose base error is past half the block's step stays
where the first refinement clamped it. The cut over bartowski's imatrix Q2_K lost 0.7 and 1.3 points at D6 and D8
against our own base (E003); this reads whether its base leaves more weights out of reach. Both cuts hold the same
source weights (the exact tail), read from the file on the CPU.

    uv run python scripts/stack_fit.py
    uv run python scripts/stack_fit.py --layers 0 17 34   # a few layers
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from foqlens import model as fm
from foqlens.refinements import stack_fit
from foqlens.refocustensors import FILE, ModelFile, model_directory

BASE = "bartowski-Q2_K"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 17, 26, 34])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ours = ModelFile(model_directory(fm.E2B_IT) / FILE, device=args.device)
    theirs = ModelFile(model_directory(fm.E2B_IT, BASE) / FILE, device=args.device)
    rows = defaultdict(lambda: defaultdict(list))
    for module in ours.modules.values():
        name = module["name"]
        layer, kind = int(name.split(".")[1]), name.split(".", 2)[2]
        if layer not in args.layers:
            continue
        source = ours.source_weight(name)
        for label, cut in (("ours", ours), ("bartowski", theirs)):
            copy = cut.copy(name)
            fit = stack_fit(copy, source)
            rows[kind][label].append((fit.past_half_step, fit.past_bound_at_top, fit.error_by_depth.get(4, np.nan),
                                      copy.fmt.name))
        print(f"{name}: ours {rows[kind]['ours'][-1][:3]} / bartowski {rows[kind]['bartowski'][-1]}", flush=True)
    print("\nkind | base ours / bartowski | past half a step | past the D8 bound | D8 error / source RMS")
    for kind, by in sorted(rows.items()):
        o, t = np.array([r[:3] for r in by["ours"]]), np.array([r[:3] for r in by["bartowski"]])
        print(f"{kind} | {by['ours'][0][3]} / {by['bartowski'][0][3]} | {o[:, 0].mean():.4f} / {t[:, 0].mean():.4f} | "
              f"{o[:, 1].mean():.4f} / {t[:, 1].mean():.4f} | {o[:, 2].mean():.5f} / {t[:, 2].mean():.5f}")


if __name__ == "__main__":
    main()
