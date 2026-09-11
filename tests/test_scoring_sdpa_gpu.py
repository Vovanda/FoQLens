"""Gradient masks on the attention of the runs (sdpa) and on texts as long as their questions: the scoring.py invariants."""

import numpy as np
import pytest

from foqlens.scoring import GradientScorer

pytestmark = pytest.mark.gpu

SENTENCES = [
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The derivative of a function measures how fast its value changes with its input.",
    "The Nile flows north through eleven countries before it reaches the Mediterranean Sea.",
    "Mitochondria produce most of the chemical energy a cell needs, stored as ATP.",
    "The Treaty of Westphalia ended the Thirty Years' War in the middle of the seventeenth century.",
    "A prime number has exactly two divisors: one and the number itself.",
    "Plate tectonics explains how mountains rise where two continental plates collide.",
    "Enzymes lower the activation energy of reactions without being consumed by them.",
    "The Silk Road linked China with the Mediterranean world for more than a thousand years.",
    "An integral adds up infinitely many infinitely small pieces of a quantity.",
]
# Every text holds all sentences in another order: another text of about the same length.
TEXTS = [" ".join(SENTENCES[i:] + SENTENCES[:i]) for i in range(8)]
# Memory-efficient attention's backward went non-deterministic on 131-token questions (scoring.GRADIENT_ATTENTION);
# 64-token ones were still deterministic.
LONG_TOKENS = 128


def test_a_long_batch_on_sdpa_gives_identical_gradient_masks(e2b_sdpa):
    model, tokenizer, ctl = e2b_sdpa
    assert min(len(tokenizer(t).input_ids) for t in TEXTS) > LONG_TOKENS
    scorer = GradientScorer(model, ctl.modules)
    first, second = scorer.score_batch(model, tokenizer, TEXTS), scorer.score_batch(model, tokenizer, TEXTS)
    for a, b in zip(first, second):
        assert np.array_equal(a["gradient"][0], b["gradient"][0])
