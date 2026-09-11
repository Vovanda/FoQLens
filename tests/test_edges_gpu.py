"""The precision share at its edges on Gemma 4 E2B: where the effect of a mask must be obvious.

- precision share 1 is the original model, bit for bit;
- precision share 0 is the uniform coarse model, bit for bit;
- with the ZERO coarse level (blocks outside the precision share removed) quality must collapse at
  precision share 0 and come back as the share rises - a mask that changes nothing would fail here.
"""

from pathlib import Path

import numpy as np
import pytest

from foqlens import budget as bg
from foqlens.evaluate import LetterChoice
from foqlens.io import read_questions
from foqlens.layouts import Directed, Uniform
from foqlens.quality import evaluate_policy

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
BATCH = 12


@pytest.fixture(scope="module")
def setup(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    questions = [q for s in ("biology", "math") for q in read_questions(ROOT / "prompts", s, 12)]
    rng = np.random.default_rng(0)
    scores = rng.random((len(questions), ctl.n_blocks))  # any mask will do: the edges must not depend on it
    return model, tokenizer, ctl, questions, scores, bg.block_weights(ctl), LetterChoice.for_tokenizer(tokenizer)


def run(setup, policy) -> list[dict]:
    model, tokenizer, ctl, questions, _, _, metric = setup
    return evaluate_policy(model, tokenizer, ctl, questions, policy, metric, BATCH)


def logprob(rows: list[dict]) -> float:
    return float(np.mean([r["logprob"] for r in rows]))


@pytest.mark.parametrize("coarse", [bg.Level.NF4, bg.Level.ZERO])
def test_precision_share_one_is_the_original_model_and_zero_the_coarse_one(setup, coarse):
    _, _, ctl, _, scores, weights, _ = setup
    all_sharp = run(setup, Directed("any", 1.0, scores, weights, coarse))
    all_coarse = run(setup, Directed("any", 0.0, scores, weights, coarse))
    assert all_sharp == run(setup, Uniform(bg.Level.BF16, ctl.n_blocks))
    assert all_coarse == run(setup, Uniform(coarse, ctl.n_blocks))
    assert all(r["mean_bits"] == 16.0 for r in all_sharp)
    assert all(r["mean_bits"] == coarse.bits for r in all_coarse)


def test_a_model_with_every_block_removed_answers_uniformly(setup):
    _, _, ctl, _, _, _, _ = setup
    rows = run(setup, Uniform(bg.Level.ZERO, ctl.n_blocks))
    assert all(abs(r["logprob"] - np.log(0.25)) < 1e-3 for r in rows)  # every letter log(1/4): nothing is left to answer with


def test_removing_a_random_twentieth_of_blocks_already_breaks_the_answers(setup):
    """Measured on these 24 questions: -1.04 at bf16, -2.75 with 5% of blocks removed at random.

    Random removal is not quality-monotone - a half-removed model is confidently wrong (-6.6),
    worse than the fully removed one that guesses (-1.39) - so only the robust fact is asserted.
    """
    _, _, ctl, _, scores, weights, _ = setup
    bf16 = logprob(run(setup, Uniform(bg.Level.BF16, ctl.n_blocks)))
    holed = logprob(run(setup, Directed("any", 0.95, scores, weights, bg.Level.ZERO)))
    assert bf16 - holed > 1.0
