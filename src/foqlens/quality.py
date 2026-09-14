"""Step 3 runs: masks over a question set, quality under a layout policy, and the summary.

Each function does one thing over batches; none of them knows where questions come from or where
results go, nor which quality metric is used (evaluate.QualityMetric). Every batch runs inside a
Throttle (gpu_share.py): below a share of 1 the GPU rests after it.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from functools import partial

import numpy as np

from foqlens.evaluate import QualityMetric, Question
from foqlens.gpu_share import FULL, Pacer
from foqlens.layouts import LayoutPolicy
from foqlens.precision import Controller
from foqlens.progress import Progress

BatchScorer = Callable[[list[str]], list[np.ndarray]]


def batches(n: int, size: int) -> Iterator[np.ndarray]:
    """Index arrays covering range(n) in order, size items each (the last one may be shorter)."""
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n))


def token_batches(lengths: Sequence[int], max_tokens: int, max_batch: int) -> list[np.ndarray]:
    """Batches of indices, longest first: each holds at most max_tokens padded tokens and max_batch items.

    A padded batch costs its size times its longest item, so sorting by length lets short items go in
    larger batches without a batch ever costing more than the budget.

    Invariant: every index is in exactly one batch; a batch holds at most max_batch items and at most
    max_tokens padded tokens, unless one item alone is longer than max_tokens.
    """
    order = np.argsort(-np.asarray(lengths), kind="stable")
    out, current = [], []
    for i in order.tolist():
        if current and ((len(current) + 1) * lengths[current[0]] > max_tokens or len(current) == max_batch):
            out.append(np.array(current))
            current = []
        current.append(i)
    if current:
        out.append(np.array(current))
    return out


def compute_masks(
    score: BatchScorer, prompts: list[str], batch_size: int, throttle: Pacer = FULL, groups: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Mask vectors of all prompts: [len(prompts), n_blocks], in the prompts' order.

    `groups` are the batches of prompt indices (token_batches); without them, batch_size prompts in order.
    """
    groups = list(batches(len(prompts), batch_size)) if groups is None else groups
    parts = []
    for idx in groups:
        with throttle.batch():
            parts.append(np.stack(score([prompts[i] for i in idx])))
    stacked = np.concatenate(parts)
    out = np.empty_like(stacked)
    out[np.concatenate(groups)] = stacked
    return out


def policy_layouts(policy: LayoutPolicy, groups: list[np.ndarray]) -> list[np.ndarray]:
    """The layouts of a policy for every batch of question indices."""
    return [policy.levels(idx) for idx in groups]


def warm_up() -> None:
    """Nothing: calling it makes a worker import this module - and torch with it - before the first layouts."""


def layout_pool() -> Executor:
    """One worker process for building layouts - spawned, so it never touches CUDA.

    The worker starts at once and imports while the caller goes on (a spawned process importing torch
    takes seconds): create the pool before loading the model, and the start is hidden behind it.
    """
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(warm_up)
    return pool


def evaluate_policy(
    model, tokenizer, ctl: Controller, questions: list[Question], policy: LayoutPolicy, metric: QualityMetric,
    batch_size: int, throttle: Pacer = FULL, layouts: list[np.ndarray] | None = None,
) -> list[dict]:
    """Per question: the metric's values and the mean bits spent.

    `layouts` are the policy's layouts per batch, built ahead (evaluate_all); without them they are built
    here first. Only this thread touches the controller.
    """
    groups = list(batches(len(questions), batch_size))
    layouts = policy_layouts(policy, groups) if layouts is None else layouts
    rows = []
    for idx, layout in zip(groups, layouts, strict=True):
        with throttle.batch():
            rows += _evaluate_batch(model, tokenizer, ctl, [questions[i] for i in idx], layout, metric)
    return rows


def _evaluate_batch(
    model, tokenizer, ctl: Controller, questions: list[Question], layout: np.ndarray, metric: QualityMetric
) -> list[dict]:
    """One batch of questions under its layout: the metric's row per question, with the mean bits spent."""
    ctl.set_layout(layout)
    values = metric.score(model, tokenizer, questions)
    bits = np.broadcast_to(ctl.mean_bits(), (len(questions),))
    return [row | {"mean_bits": float(b)} for row, b in zip(values, bits)]


def evaluate_all(
    model, tokenizer, ctl: Controller, questions: list[Question], policies: list[LayoutPolicy], metric: QualityMetric,
    batch_size: int, log: Callable[[str], None] = partial(print, flush=True), throttle: Pacer = FULL,
    pool: Executor | None = None, layouts: list[Future] | None = None,
) -> dict[str, list[dict]]:
    """evaluate_policy for every policy, by name, logging each finished one with the percent and the expected end.

    The layouts are built in a worker process (`pool`, one of layout_pool() when not given) while the GPU
    runs: all policies are queued at once (submit_layouts), or the caller passes its queued `layouts` so
    the worker can already be on the next call's policies. A process, not a thread: layout code is Python
    and numpy, and a thread takes the GIL from the evaluation thread, which launches thousands of kernels
    per batch (measured 1.48 s -> 4.21 s per batch, GPU utilization 86% -> 32%).
    """
    groups = list(batches(len(questions), batch_size))
    results, progress = {}, Progress(len(policies))
    own_pool = pool is None and layouts is None
    pool = layout_pool() if own_pool else pool
    try:
        queued = submit_layouts(pool, policies, groups) if layouts is None else layouts
        for policy, future in zip(policies, queued, strict=True):
            results[policy.name] = evaluate_policy(model, tokenizer, ctl, questions, policy, metric, batch_size, throttle,
                                                   layouts=future.result())
            log(progress.step(policy.name))
    finally:
        if own_pool:
            pool.shutdown()
    return results


def submit_layouts(pool: Executor, policies: list[LayoutPolicy], groups: list[np.ndarray]) -> list[Future]:
    """Queue the layouts of every policy in the worker, in order: it runs as far ahead of the GPU as it can."""
    return [pool.submit(policy_layouts, policy, groups) for policy in policies]


def summarize(results: dict[str, list[dict]], questions: list[Question]) -> dict:
    """The mean of every value per policy - whatever the metric names them - overall and per domain."""
    domains = list(dict.fromkeys(q.domain for q in questions))
    by_domain = {d: np.array([q.domain == d for q in questions]) for d in domains}
    out = {}
    for name, rows in results.items():
        values = {key: np.array([r[key] for r in rows], dtype=float) for key in rows[0]}
        out[name] = {key: float(v.mean()) for key, v in values.items()}
        out[name]["by_domain"] = {key: {d: float(v[m].mean()) for d, m in by_domain.items()} for key, v in values.items()
                                  if key != "mean_bits"}
    return out
