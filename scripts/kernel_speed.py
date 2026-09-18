"""The time of one module's multiplication by each way of reading its k-quant copy, for a range of token counts.

A random 12288x1536 module (E2B's widest) on a Q2_K base: dense bf16 on cuBLAS, the copy unpacked to D8 for a GEMM -
in torch and by the unpacking kernel - and each kernel of foqlens.kernels.kquant at D2 and D8. Times are on the
device, the median of several rounds.

    uv run python scripts/kernel_speed.py
    uv run python scripts/kernel_speed.py --tokens 256 1024 4096 16384   # a prefill
    uv run python scripts/kernel_speed.py --tokens 1 8 --rounds 3   # smoke check
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from foqlens.kernels.kquant import CUDA_CORES, TENSOR_CORES, TILE_ROWS, kquant_unpack
from foqlens.kquant import Q2_K
from foqlens.quant import MAX_DEPTH
from foqlens.refinements import KRefinedWeight

OUT, IN = 12288, 1536
CALLS = 20  # calls a round times
WARMUP = 3
WEIGHT_SCALE = 0.02  # the spread of a trained module's weights, roughly


def ms(call, rounds: int) -> float:
    """Milliseconds a call, CALLS of them captured in one CUDA graph: a decoding step replays a graph, so the host's
    launch - slower than a small kernel - is not what is timed."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS):
            call()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / CALLS)
    graph.reset()
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(0)
    weight = (torch.randn(OUT, IN, device="cuda") * WEIGHT_SCALE).to(torch.bfloat16)
    copy = KRefinedWeight.quantize(weight, Q2_K)
    depth = {d: torch.full((OUT // TILE_ROWS,), d, dtype=torch.uint8, device="cuda") for d in (1, MAX_DEPTH)}
    kernels = {"CUDA cores": CUDA_CORES, "tensor cores": TENSOR_CORES}

    columns = ["bf16", "unpacked D8", "kernel-unpacked D8"] + [f"{name} D{2 * d}" for name in kernels for d in depth]
    print("tokens | " + " | ".join(columns), flush=True)
    for n in args.tokens:
        x = torch.randn(n, IN, device="cuda", dtype=torch.bfloat16)
        row = [ms(lambda: F.linear(x, weight), args.rounds), ms(lambda: copy.matmul(x, None, MAX_DEPTH), args.rounds),
               ms(lambda: F.linear(x, kquant_unpack(copy, depth[MAX_DEPTH])), args.rounds)]
        for kernel in kernels.values():
            for d in depth.values():
                row.append(ms(lambda k=kernel, d=d: k(copy, x, d), args.rounds))
        print(f"{n} | " + " | ".join(f"{t:.3f}" for t in row), flush=True)


if __name__ == "__main__":
    main()
