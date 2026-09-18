"""Bits per weight of the exact tail under different codings, over every controlled weight of a cut model.

For each depth the tail could start after (the base ... the deepest refinement), prints the cost of the refinements
under it plus the tail's cost at a fixed width per group of 256, 32 and 16 weights (the file writes 16), as an entropy coder
would write it, and as one that knows the exponent of the prediction; and the entropy of the source itself.

    uv run python scripts/exact_tail_cost.py e2b-it
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from foqlens.quant import DEPTH_BITS, MAX_DEPTH
from foqlens.refinements import _zigzag, ulp_order
from foqlens.refocustensors import DTYPES, FILE, ModelFile, model_directory
from foqlens.tail_cost import conditional_entropy_bits, entropy_bits, exponent, fixed_width_bits
from cut_model import MODELS

GROUPS = (256, 32, 16)  # a super-block, and finer


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args(argv)
    file = ModelFile(model_directory(MODELS[args.model]) / FILE)
    state = file.source_state()
    sums: dict[tuple, float] = defaultdict(float)
    total = 0
    for key, module in file.modules.items():
        copy = file.copy(module["name"])
        source = state[key]
        n = source.numel()
        total += n
        sums[("source",)] += n * entropy_bits(ulp_order(source))
        for depth in range(copy.base.fmt.base_depth, MAX_DEPTH + 1):
            prediction = copy.dequantize(torch.float32, depth).to(DTYPES[module["dtype"]])
            z = _zigzag(ulp_order(source) - ulp_order(prediction))
            under = copy.bits_per_weight(depth)
            for group in GROUPS:
                sums[(depth, f"fixed/{group}")] += n * (under + fixed_width_bits(z, group))
            sums[(depth, "entropy")] += n * (under + entropy_bits(z))
            sums[(depth, "entropy|exponent")] += n * (under + conditional_entropy_bits(z, exponent(prediction)))
            sums[(depth, "weights")] += n
    print(f"source entropy {sums[('source',)] / total:.2f} bits per weight over {total:,} controlled weights")
    print("tail after   " + "  ".join(f"{c:>18s}" for c in [f"fixed/{g}" for g in GROUPS] + ["entropy", "entropy|exponent"]))
    for depth in range(1, MAX_DEPTH + 1):
        n = sums[(depth, "weights")]
        if n:
            cells = [sums[(depth, c)] / n for c in [f"fixed/{g}" for g in GROUPS] + ["entropy", "entropy|exponent"]]
            print(f"D{depth * DEPTH_BITS:<11d}" + "  ".join(f"{c:18.2f}" for c in cells) + f"   ({n / total:.0%} of weights)")
    return dict(sums)


if __name__ == "__main__":
    main()
