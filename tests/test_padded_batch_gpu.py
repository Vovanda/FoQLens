"""A padded row reads what it reads alone: the batch that collapsed under the memory-efficient sdpa kernel.

ARC-Challenge, round 4 of E016's schedule (seed 0): 29 prompts in one left-padded batch. Under the
memory-efficient kernel every one of its 28 padded rows ended on the same logits and opened with the
same token, 令 at D8 and <h2> at bf16, whatever its question (issue #14). On the math kernel each
padded row writes the first token it writes alone, up to bf16's batch noise.

The attention plans are held to the same batch (foqlens.attention, issue #15): the split plan - prefill on
math, decode steps and the judge on the memory-efficient kernel - opens every row as math does, keeps the
padded rows apart, and the judge gives the verdicts it gives on math.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import JUDGE_BATCH, Asking
from foqlens.attention import MATH_ONLY, SPLIT
from foqlens.generation import generate_replies
from foqlens.graph_decode import STATIC, StaticDecoder
from foqlens.io import read_frozen
from foqlens.judging import ModelJudge
from foqlens.prompt_variants import SETUPS, examples_for, setup_named
from foqlens.quant import Level
from foqlens.schedule import Schedule
from foqlens.selection import JUDGE_YES

pytestmark = pytest.mark.gpu

FROZEN = Path("corpus/e2b-it")
CORPUS, SETUP = "arc_challenge_closed", "solve-0"
ROUND, SEED = 4, 0  # the round and seed of stage1_answers whose ARC-Challenge batch collapsed
# Padded rows allowed to open otherwise than alone: bf16 is not batch-invariant and a near tie can go
# the other way; the math kernel moved 0 of these 28 rows, the collapsed kernel moved all 28 (2026-09-15).
NOISE_ROWS = 2
DECODE_TOKENS = 16  # a collapsed batch shows by then: its padded rows go on from one state, word for word
# Verdicts allowed to flip between the judge's kernels: 46 of 20,471 flipped on E016's bf16 answers
# (0.22%), all near ties - none is expected among 29.
JUDGE_FLIPS = 1


@pytest.fixture(scope="module")
def collapsed_batch(e2b_it_sdpa):
    model, tokenizer, _ = e2b_it_sdpa
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    rows = {}
    for corpus in SETUPS:
        asked = set(read_frozen(FROZEN / f"{corpus}.json").asked())
        rows[corpus] = [r for r in corpora.read(corpus)[0] if r.id in asked]
    todo = dict(Schedule.build({c: [r.id for r in rs] for c, rs in rows.items()}, SEED).pending({c: set() for c in rows}))
    by_id = {r.id: r for r in rows[CORPUS]}
    setup = setup_named(CORPUS, SETUP)
    asking = Asking(CORPUS, "", "", Level.BF16, setup, examples_for(CORPUS, setup, [], SEED))
    [batch] = asking.batches(fmt, tokenizer, [by_id[i] for i in todo[ROUND][CORPUS]])
    return asking.prompts(fmt, batch), setup.stop, batch


def padded_rows(tokenizer, prompts: list[str]) -> list[int]:
    lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
    return [i for i, n in enumerate(lengths) if n < max(lengths)]


def test_the_memory_efficient_kernel_is_off_after_load(e2b_it_sdpa):
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_a_padded_row_opens_with_the_token_it_opens_with_alone(e2b_it_sdpa, collapsed_batch):
    model, tokenizer, ctl = e2b_it_sdpa
    prompts, stop, _ = collapsed_batch
    ctl.set_all(Level.BF16)
    batched = [w.text for w in generate_replies(model, tokenizer, prompts, 1, stop, STATIC)]
    alone = [generate_replies(model, tokenizer, [p], 1, stop, STATIC)[0].text for p in prompts]
    padded = padded_rows(tokenizer, prompts)
    moved = [i for i in padded if batched[i] != alone[i]]
    assert len(padded) == 28
    assert len({batched[i] for i in padded}) > 1, "every padded row opened with the same token"
    assert len(moved) <= NOISE_ROWS, [(batched[i], alone[i]) for i in moved]


def test_the_split_plan_opens_every_row_as_math_does(e2b_it_sdpa, collapsed_batch):
    model, tokenizer, ctl = e2b_it_sdpa
    prompts, stop, _ = collapsed_batch
    ctl.set_all(Level.BF16)
    split = [w.text for w in generate_replies(model, tokenizer, prompts, 1, stop, StaticDecoder(attention=SPLIT))]
    math = [w.text for w in generate_replies(model, tokenizer, prompts, 1, stop, StaticDecoder(attention=MATH_ONLY))]
    assert split == math


def test_the_split_plan_keeps_the_padded_rows_apart(e2b_it_sdpa, collapsed_batch):
    model, tokenizer, ctl = e2b_it_sdpa
    prompts, stop, _ = collapsed_batch
    ctl.set_all(Level.BF16)
    texts = [w.text for w in generate_replies(model, tokenizer, prompts, DECODE_TOKENS, stop,
                                              StaticDecoder(attention=SPLIT))]
    padded = padded_rows(tokenizer, prompts)
    assert len({texts[i] for i in padded}) > len(padded) // 2, "the padded rows went on from one state"


def test_the_judge_gives_on_the_split_plan_the_verdicts_it_gives_on_math(e2b_it_sdpa, collapsed_batch):
    model, tokenizer, ctl = e2b_it_sdpa
    _, _, batch = collapsed_batch
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    # every other answer is the next question's reference: the batch holds answers right and wrong
    answers = [r.answers[0] if i % 2 == 0 else batch[(i + 1) % len(batch)].answers[0] for i, r in enumerate(batch)]
    questions, references = [r.question for r in batch], [list(r.answers) for r in batch]
    verdicts = {}
    for plan in (MATH_ONLY, SPLIT):
        judge = ModelJudge.build(model, tokenizer, ctl, fmt, batch_size=JUDGE_BATCH, kernels=plan.judge)
        verdicts[plan.name] = judge.p_yes(questions, answers, references) > JUDGE_YES
    assert 0 < verdicts["math"].sum() < len(batch)
    assert np.sum(verdicts["math"] != verdicts["split"]) <= JUDGE_FLIPS
