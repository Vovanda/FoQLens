"""Mask sources and answer scoring on Gemma 4 E2B: the invariants of scoring.py and evaluate.py."""

import numpy as np
import pytest

from foqlens.evaluate import letter_ids, letter_logprobs_batch, mc_prompt
from foqlens.quant import Level
from foqlens.scoring import MODES, BlockScorer, GradientScorer

pytestmark = pytest.mark.gpu

TEXT = "Photosynthesis converts light energy into chemical energy stored in glucose."
TEXTS = [TEXT, "What is the derivative of x squared?", "Name the longest river in Africa and the countries it flows through."]
PROMPTS = [mc_prompt(t, ["one", "two", "three", "four"]) for t in TEXTS]
MASK_COSINE = 0.99  # batched vs alone, see the scoring.py invariants
LOGPROB_ATOL = 0.25  # batched vs alone, measured 0.09-0.19, see the evaluate.py invariants


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(autouse=True)
def bf16(e2b_eager):
    e2b_eager[2].set_all(Level.BF16)


def test_every_mode_covers_every_block_and_skips_bos(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    scorer = BlockScorer(ctl.modules)
    scored = scorer.score(model, tokenizer, TEXT)
    for mode in MODES:
        vec, positions = scored[mode]
        assert vec.shape == (scorer.n_blocks,) and np.all(vec >= 0) and vec.sum() > 0, mode
        assert 0 not in positions, mode
    assert scored["pooled"][1] == list(range(1, len(tokenizer(TEXT).input_ids)))


def test_same_batch_gives_identical_masks_and_scores(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    blocks, grads = BlockScorer(ctl.modules), GradientScorer(model, ctl.modules)
    a, b = blocks.score_batch(model, tokenizer, TEXTS), blocks.score_batch(model, tokenizer, TEXTS)
    g1, g2 = grads.score_batch(model, tokenizer, TEXTS), grads.score_batch(model, tokenizer, TEXTS)
    ids = letter_ids(tokenizer)
    for i in range(len(TEXTS)):
        for mode in MODES:
            assert np.array_equal(a[i][mode][0], b[i][mode][0]), mode
        assert np.array_equal(g1[i]["gradient"][0], g2[i]["gradient"][0])
    assert np.array_equal(letter_logprobs_batch(model, tokenizer, PROMPTS, ids), letter_logprobs_batch(model, tokenizer, PROMPTS, ids))


def test_the_magnitude_form_of_taylor_is_never_below_the_signed_one(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    for got in GradientScorer(model, ctl.modules).score_batch(model, tokenizer, TEXTS):
        signed, magnitude = got["gradient"][0], got["gradient_magnitude"][0]
        assert magnitude.shape == signed.shape and np.all(magnitude >= signed - 1e-4 * np.abs(signed).max())
        assert np.any(magnitude > signed * 1.01)  # somewhere the terms of a block do cancel (#17)


def test_padding_positions_never_enter_a_mask(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    batched = BlockScorer(ctl.modules).score_batch(model, tokenizer, TEXTS)
    for text, got in zip(TEXTS, batched):
        n = len(tokenizer(text).input_ids)
        for mode in MODES:
            assert max(got[mode][1]) < n, mode
    grads = GradientScorer(model, ctl.modules).score_batch(model, tokenizer, TEXTS)
    assert [g["gradient"][1] for g in grads] == [list(range(1, len(tokenizer(t).input_ids))) for t in TEXTS]


def test_batched_masks_point_the_same_way_as_alone(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    blocks, grads = BlockScorer(ctl.modules), GradientScorer(model, ctl.modules)
    batched, gbatched = blocks.score_batch(model, tokenizer, TEXTS), grads.score_batch(model, tokenizer, TEXTS)
    for text, got, ggot in zip(TEXTS, batched, gbatched):
        alone, galone = blocks.score(model, tokenizer, text), grads.score(model, tokenizer, text)
        assert got["pooled"][1] == alone["pooled"][1]
        assert cosine(got["pooled"][0], alone["pooled"][0]) >= MASK_COSINE
        assert cosine(ggot["gradient"][0], galone["gradient"][0]) >= MASK_COSINE
        for mode in ("norm", "attention"):  # discrete picks: one center of four may swap on a near-tie
            assert len(set(got[mode][1]) & set(alone[mode][1])) >= blocks.top_k - 1, mode
    assert all(p.grad is None for p in model.parameters())


SMALL_CHUNK = 16  # the test texts have ~60 positions in a batch: several chunks and a ragged last one
LOSS_RTOL = 1e-3  # the head's GEMM on a chunk of rows may round differently from the whole matrix
CHUNKED_MASK_COSINE = 0.999


def test_chunked_losses_and_masks_match_the_full_logits(e2b_eager):
    import torch

    from foqlens import model as fm
    from foqlens.scoring import TaylorRecorder, chunked_sequence_losses, sequence_losses

    model, tokenizer, ctl = e2b_eager
    scorer = GradientScorer(model, ctl.modules, loss_chunk=SMALL_CHUNK)  # freezes the parameters first
    enc = fm.encode(tokenizer, TEXTS, model.device)
    with torch.no_grad():
        full = sequence_losses(model(**enc).logits, enc["input_ids"], enc["attention_mask"])
        chunked = chunked_sequence_losses(model, model.model(**enc).last_hidden_state, enc["input_ids"],
                                          enc["attention_mask"], SMALL_CHUNK)
    torch.testing.assert_close(chunked, full, rtol=LOSS_RTOL, atol=0)

    valid = enc["attention_mask"].clone()
    valid[:, 0] = 0
    with torch.enable_grad(), TaylorRecorder(ctl.modules, valid) as rec:
        sequence_losses(model(**enc).logits, enc["input_ids"], enc["attention_mask"]).sum().backward()
    reference = torch.cat([rec.scores[n] for n in ctl.modules], dim=1).cpu().numpy()
    for b, got in enumerate(scorer.score_batch(model, tokenizer, TEXTS)):
        assert cosine(got["gradient"][0], reference[b]) >= CHUNKED_MASK_COSINE, b


def test_batched_letter_scores_are_close_to_alone(e2b_eager):
    model, tokenizer, _ = e2b_eager
    ids = letter_ids(tokenizer)
    batched = letter_logprobs_batch(model, tokenizer, PROMPTS, ids)
    for p, got in zip(PROMPTS, batched):
        np.testing.assert_allclose(got, letter_logprobs_batch(model, tokenizer, [p], ids)[0], atol=LOGPROB_ATOL)


def test_per_question_layouts_are_exact_against_the_same_batch(e2b_eager):
    """Invariant of precision.py: a per-sample layout changes nothing but the selection."""
    model, tokenizer, ctl = e2b_eager
    ids = letter_ids(tokenizer)
    rng = np.random.default_rng(0)
    layouts = np.where(rng.random((len(PROMPTS), ctl.n_blocks)) < 0.3, Level.BF16, Level.NF4).astype(np.uint8)
    ctl.set_layout(layouts)
    mixed = letter_logprobs_batch(model, tokenizer, PROMPTS, ids)
    for b in range(len(PROMPTS)):
        ctl.set_layout(np.repeat(layouts[b : b + 1], len(PROMPTS), axis=0))
        assert np.array_equal(mixed[b], letter_logprobs_batch(model, tokenizer, PROMPTS, ids)[b]), b
