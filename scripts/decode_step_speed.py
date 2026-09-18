"""The time of one graphed decoding step of the model at each way of reading its weights, for a few batch sizes.

bf16; a uniform depth and a mixed layout of depths, each read by the k-quant kernel and with the kernel off (a uniform
depth then multiplies a weight unpacked once, a mixed one picks every block's rows out of the weights unpacked at its
levels). The prompts are the same short question for every row; the step is replayed after its capture and timed on
the device.

    uv run python scripts/decode_step_speed.py
    uv run python scripts/decode_step_speed.py --batches 1 8 --steps 20   # smoke check
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from foqlens import model as fm
from foqlens import precision
from foqlens.graph_decode import GraphedStep, StaticRun
from foqlens.pipeline import Bench
from foqlens.quant import Level

WARMUP = 3  # replays before the timing: the first call runs eagerly and captures
PROMPT = "What is the capital of France?"
DEPTH_LEVELS = np.array([Level.D8, Level.D6, Level.D4, Level.D2], dtype=np.uint8)
NEVER = -1  # no token id is negative: no row stops


def step_ms(bench: Bench, batch: int, steps: int) -> float:
    """Milliseconds of one replayed step at the controller's current layout."""
    tokenizer, model = bench.tokenizer, bench.model
    enc = fm.encode_left(tokenizer, [PROMPT] * batch, model.device)
    stop = torch.tensor([NEVER], device=model.device)
    run = StaticRun(model, enc["input_ids"], enc["attention_mask"], WARMUP + steps + 1, stop)
    step = GraphedStep(run)
    try:
        for _ in range(WARMUP):
            step()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / steps
    finally:
        step.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=fm.E2B_IT)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    bench = Bench.load(args.model)
    mixed = np.random.default_rng(0).choice(DEPTH_LEVELS, size=bench.ctl.n_blocks)
    readings = {
        "bf16": (Level.BF16, False),
        "D4 unpacked": (Level.D4, False),
        "D4 kernel": (Level.D4, True),
        "mixed unpacked": (mixed, False),
        "mixed kernel": (mixed, True),
    }
    print("ms a step | " + " | ".join(f"batch {b}" for b in args.batches), flush=True)
    for name, (layout, kernel) in readings.items():
        precision.KERNEL = kernel  # read when a layout is set
        if isinstance(layout, Level):
            bench.ctl.set_all(layout)
        else:
            bench.ctl.set_layout(layout)
        through_kernel = [module._depths is not None for module in bench.ctl.modules.values()]
        if name != "bf16" and not all(k == kernel for k in through_kernel):
            raise RuntimeError(f"{name}: {sum(through_kernel)} of {len(through_kernel)} modules read by the kernel")
        times = [step_ms(bench, batch, args.steps) for batch in args.batches]
        print(f"{name} | " + " | ".join(f"{t:.2f}" for t in times), flush=True)
    bench.ctl.set_all(Level.BF16)


if __name__ == "__main__":
    main()
