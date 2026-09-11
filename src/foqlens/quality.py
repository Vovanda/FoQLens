"""Step 3 runs: masks over a question set, quality under a layout policy, and the summary.

Each function does one thing over batches; none of them knows where questions come from or where
results go.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial

import numpy as np

from foqlens.evaluate import Question, letter_logprobs_batch
from foqlens.layouts import LayoutPolicy
from foqlens.precision import Controller

BatchScorer = Callable[[list[str]], list[np.ndarray]]


def batches(n: int, size: int) -> Iterator[np.ndarray]:
    """Index arrays covering range(n) in order, size items each (the last one may be shorter)."""
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n))


def compute_masks(score: BatchScorer, prompts: list[str], batch_size: int) -> np.ndarray:
    """Mask vectors of all prompts: [len(prompts), n_blocks]."""
    return np.concatenate([np.stack(score([prompts[i] for i in idx])) for idx in batches(len(prompts), batch_size)])


def evaluate_policy(
    model, tokenizer, ctl: Controller, questions: list[Question], policy: LayoutPolicy, ids: list[int], batch_size: int
) -> list[dict]:
    """Per question: whether the top letter is right, the log-probability of the right one, the mean bits spent."""
    rows = []
    for idx in batches(len(questions), batch_size):
        ctl.set_layout(policy.levels(idx))
        logprobs = letter_logprobs_batch(model, tokenizer, [questions[i].prompt for i in idx], ids)
        bits = np.broadcast_to(ctl.mean_bits(), (len(idx),))
        for i, lp, b in zip(idx, logprobs, bits):
            answer = questions[i].answer
            rows.append({"correct": bool(lp.argmax() == answer), "logprob": float(lp[answer]), "mean_bits": float(b)})
    return rows


def evaluate_all(
    model, tokenizer, ctl: Controller, questions: list[Question], policies: list[LayoutPolicy], ids: list[int],
    batch_size: int, log: Callable[[str], None] = partial(print, flush=True),
) -> dict[str, list[dict]]:
    """evaluate_policy for every policy, by name, logging each finished one so a long run can be followed."""
    results = {}
    for i, policy in enumerate(policies, 1):
        results[policy.name] = evaluate_policy(model, tokenizer, ctl, questions, policy, ids, batch_size)
        log(f"[{i}/{len(policies)}] {policy.name}")
    return results


def summarize(results: dict[str, list[dict]], questions: list[Question]) -> dict:
    """Accuracy, mean log-probability of the right answer and mean bits per policy, overall and per domain."""
    domains = list(dict.fromkeys(q.domain for q in questions))
    by_domain = {d: np.array([q.domain == d for q in questions]) for d in domains}
    out = {}
    for name, rows in results.items():
        correct = np.array([r["correct"] for r in rows])
        out[name] = {
            "accuracy": float(correct.mean()),
            "logprob": float(np.mean([r["logprob"] for r in rows])),
            "mean_bits": float(np.mean([r["mean_bits"] for r in rows])),
            "by_domain": {d: float(correct[m].mean()) for d, m in by_domain.items()},
        }
    return out
