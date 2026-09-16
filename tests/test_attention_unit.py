"""Attention plans: which sdpa kernels each phase may use."""

from torch.nn.attention import SDPBackend

from foqlens.attention import MATH_ONLY, PLANS, SPLIT


def test_every_plan_reads_the_prefill_on_math():
    assert all(plan.prefill == (SDPBackend.MATH,) for plan in PLANS.values())


def test_the_split_plan_moves_the_steps_to_the_memory_efficient_kernel():
    assert SPLIT.decode[0] == SDPBackend.EFFICIENT_ATTENTION and SDPBackend.MATH in SPLIT.decode


def test_the_math_plan_runs_math_everywhere():
    assert MATH_ONLY.prefill == MATH_ONLY.decode == (SDPBackend.MATH,)


def test_a_plan_is_found_by_its_name():
    assert PLANS == {"math": MATH_ONLY, "split": SPLIT}
