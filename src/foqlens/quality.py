"""Step 3 runs: masks over a question set, quality under a layout policy, and the summary.

Each function does one thing over batches; none of them knows where questions come from or where
results go, nor which quality metric is used (evaluate.QualityMetric). Every batch runs inside a
Throttle (gpu_share.py): below a share of 1 the GPU rests after it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from foqlens.evaluate import QualityMetric, Question
from foqlens.gpu_share import FULL, Throttle
from foqlens.layouts import LayoutPolicy
from foqlens.precision import Controller

BatchScorer = Callable[[list[str]], list[np.ndarray]]


def batches(n: int, size: int) -> Iterator[np.ndarray]:
    """Index arrays covering range(n) in order, size items each (the last one may be shorter)."""
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n))


def compute_masks(score: BatchScorer, prompts: list[str], batch_size: int, throttle: Throttle = FULL) -> np.ndarray:
    """Mask vectors of all prompts: [len(prompts), n_blocks]."""
    parts = []
    for idx in batches(len(prompts), batch_size):
        with throttle.batch():
            parts.append(np.stack(score([prompts[i] for i in idx])))
    return np.concatenate(parts)


def evaluate_policy(
    model, tokenizer, ctl: Controller, questions: list[Question], policy: LayoutPolicy, metric: QualityMetric,
    batch_size: int, throttle: Throttle = FULL,
) -> list[dict]:
    """Per question: the metric's values and the mean bits spent.

    The layouts of the next batch are built on a worker thread while the GPU runs the current batch:
    a layout depends only on the question indices, and only this thread touches the controller.
    """
    groups = list(batches(len(questions), batch_size))
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(policy.levels, groups[0]) if groups else None
        for k, idx in enumerate(groups):
            layout = upcoming.result()
            if k + 1 < len(groups):
                upcoming = pool.submit(policy.levels, groups[k + 1])
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
    batch_size: int, log: Callable[[str], None] = partial(print, flush=True), throttle: Throttle = FULL,
) -> dict[str, list[dict]]:
    """evaluate_policy for every policy, by name, logging each finished one so a long run can be followed."""
    results = {}
    for i, policy in enumerate(policies, 1):
        results[policy.name] = evaluate_policy(model, tokenizer, ctl, questions, policy, metric, batch_size, throttle)
        log(f"[{i}/{len(policies)}] {policy.name}")
    return results


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
