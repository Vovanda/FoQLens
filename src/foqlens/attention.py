"""Which sdpa kernel each phase of the bench may use: a named plan, chosen per run.

The memory-efficient kernel collapses every padded row of some left-padded batches into one state at
the prefill (issue #14), so the process runs on math by default (model.load). Math is also slow where it
is not needed: a decode step asks one query per row and costs 2-6x on math (35.0 against 16.6 ms at 29
rows, 132.8 against 23.3 at 102, 2026-09-16), and the judge pads on the right and was not hit (46 verdicts
of 20,471 flip when its kernel changes, the noise of near ties). A plan says, phase by phase, which
kernels may run; the rest of the bench asks for a phase, not for a kernel.

Invariant: the prefill of generation runs on math in every plan.
Invariant: a plan changes which kernel runs, never what is asked (tests/test_padded_batch_gpu.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.nn.attention import SDPBackend

MATH = (SDPBackend.MATH,)
# The memory-efficient kernel where it applies, math where it does not (head_dim 512 is served by both).
FAST = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)


@dataclass(frozen=True)
class AttentionPlan:
    """The kernels each phase may use."""

    name: str
    prefill: tuple[SDPBackend, ...]  # a left-padded prompt read at once: the phase the collapse hit
    decode: tuple[SDPBackend, ...]   # one query per row, over the cache
    judge: tuple[SDPBackend, ...]    # a right-padded forward of the judge's prompts


MATH_ONLY = AttentionPlan("math", MATH, MATH, MATH)
SPLIT = AttentionPlan("split", MATH, FAST, FAST)
PLANS = {plan.name: plan for plan in (MATH_ONLY, SPLIT)}
