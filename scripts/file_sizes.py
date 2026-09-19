"""The size table of a cut model (docs/refocustensors.md): for every top of the stack the bits per controlled weight,
the controlled weights and the whole file in GiB, read from the file's header alone - no tensor is loaded.

    uv run python scripts/file_sizes.py                          # the bench's own base
    uv run python scripts/file_sizes.py --base bartowski-Q2_K    # a base read from a published file
"""

from __future__ import annotations

import argparse
import math

from foqlens import model as fm
from foqlens.quant import MAX_DEPTH
from foqlens.refocustensors import FILE, ModelFile, model_directory

GIB = 2**30
BITS_PER_LEVEL = 2  # D2, D4, D6, D8: a depth is two bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=None, help="the folder suffix of a cut over a published base")
    args = parser.parse_args()

    path = model_directory(fm.E2B_IT, args.base) / FILE
    file = ModelFile(path, device="cpu")
    weights = sum(math.prod(module["shape"]) for module in file.modules.values())
    passed = file.passed_bytes()
    print(f"{path}: {weights / 1e9:.2f} billion controlled weights, {passed / GIB:.2f} GiB the regulator does not read")
    print("| Top of the stack | Controlled weights, bits per weight | Controlled weights, GiB | File, GiB |")
    for depth in range(1, MAX_DEPTH + 1):
        stack = file.stack_bytes(depth)
        print(f"| D{BITS_PER_LEVEL * depth} | {8 * stack / weights:.2f} | {stack / GIB:.2f} | {(stack + passed) / GIB:.2f} |")
    size = path.stat().st_size
    print(f"| the source (`exact`) | {8 * (size - passed) / weights:.2f} | {(size - passed) / GIB:.2f} | {size / GIB:.2f} |")


if __name__ == "__main__":
    main()
