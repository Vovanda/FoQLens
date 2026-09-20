"""The time of one graphed decoding step of the model at each way of reading its weights, for a few batch sizes.

bf16; a uniform depth and a mixed layout of depths, each read by the k-quant kernel and with the kernel off (a uniform
depth then multiplies a weight unpacked once, a mixed one picks every block's rows out of the weights unpacked at its
levels). The prompts are the same short question for every row; the step is replayed after its capture and timed on
the device.

With --layerwise-price the same mixed reading is timed twice more, both launched from Python instead of replayed: as
it is, and with the layer-wise regulator deciding the layout inside the step. The three numbers say what the graph is
worth and what the regulator costs on top of giving it up.

    uv run python scripts/decode_step_speed.py
    uv run python scripts/decode_step_speed.py --batches 1 8 --steps 20   # smoke check
    uv run python scripts/decode_step_speed.py --batches 32 --steps 20 --layerwise-price 1
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from foqlens import model as fm
from foqlens import precision, refocustensors
from foqlens.graph_decode import GraphedStep, StaticRun
from foqlens.layerwise import Activity, LayerwiseRegulator
from foqlens.pipeline import Bench
from foqlens.quant import Level

WARMUP = 3  # replays before the timing: the first call runs eagerly and captures
PROMPT = "What is the capital of France?"
DEPTH_LEVELS = np.array([Level.D8, Level.D6, Level.D4, Level.D2], dtype=np.uint8)
NEVER = -1  # no token id is negative: no row stops


def step_ms(bench: Bench, batch: int, steps: int, graphed: bool = True,
            regulator: LayerwiseRegulator | None = None) -> float:
    """Milliseconds of one step at the controller's current layout: replayed from a graph, or launched from Python.

    A regulator decides the layout inside the step from the state entering every layer. Its decision is a chain of
    kernels over tensors on the card, so the step is captured with the decision inside it and the layout is decided
    again at every replay; the same step launched from Python is what the capture is measured against.
    """
    tokenizer, model = bench.tokenizer, bench.model
    enc = fm.encode_left(tokenizer, [PROMPT] * batch, model.device)
    stop = torch.tensor([NEVER], device=model.device)
    run = StaticRun(model, enc["input_ids"], enc["attention_mask"], WARMUP + steps + 1, stop)
    if regulator is not None:
        regulator.attach(model)  # after the prefill: what is timed is the decoding step
    step = GraphedStep(run) if graphed else run.advance
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
        if graphed:
            step.release()
        if regulator is not None:
            regulator.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=fm.E2B_IT)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--kernel-max-tokens", type=int, default=precision.KERNEL_MAX_TOKENS,
                        help="tokens of a step up to which a layout is read by the kernel; past it, one unpacking by the "
                        "kernel and a GEMM - to choose precision.KERNEL_MAX_TOKENS on the whole model")
    parser.add_argument("--base", default=None,
                        help="the cut folder of a copy on a published file's base, as the runs name it "
                        "(bartowski-Q2_K); without it, the model's own cut folder")
    parser.add_argument("--layerwise-price", type=float, default=None,
                        help="also time the layer-wise regulator at this price (foqlens.layerwise) against the same "
                        "mixed reading launched from Python: what the regulator costs, apart from the graph it gives up")
    args = parser.parse_args()
    precision.KERNEL_MAX_TOKENS = args.kernel_max_tokens  # read at every forward

    bench = Bench.load(args.model, directory=refocustensors.model_directory(args.model, args.base))
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
    if args.layerwise_price is not None:
        precision.KERNEL = True
        bench.ctl.set_layout(mixed)
        price = args.layerwise_price
        arms = [("mixed kernel from Python", False, None),
                (f"layer-wise p{price:g} from Python", False, LayerwiseRegulator(bench.ctl, Activity(), price=price)),
                (f"layer-wise p{price:g} graphed", True, LayerwiseRegulator(bench.ctl, Activity(), price=price))]
        for name, graphed, regulator in arms:
            bench.ctl.set_layout(mixed)  # a regulator leaves its own layout behind; every arm starts from this one
            times = [step_ms(bench, batch, args.steps, graphed=graphed, regulator=regulator) for batch in args.batches]
            print(f"{name} | " + " | ".join(f"{t:.2f}" for t in times), flush=True)
    bench.ctl.set_all(Level.BF16)


if __name__ == "__main__":
    main()
