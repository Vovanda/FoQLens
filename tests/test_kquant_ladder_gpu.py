"""The bench's copy answers as the committed k-quant ladder of E002 did, at every read depth.

E002's ladder was answered with each level baked into the weights. Here the model is read from its cut folder
(.refocustensors, no Hugging Face checkpoint) and the controller reads the k-quant copy from the file
by blocks (set_all), the path the filter's zones take, on 17 known questions of each of the six corpora, and the
replies are held against the committed answers of those questions.

bf16 is not batch-invariant: these questions sit in other batches than in the full run, so a near tie can turn a
greedy reply, and a long written solution almost always turns somewhere. The short answers are held to the share
bf16 keeps against itself; the written ones are left to the judge.
"""

import json
from pathlib import Path

import pytest

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import Asking
from foqlens.attention import SPLIT
from foqlens.graph_decode import StaticDecoder
from foqlens.io import read_answers, read_frozen
from foqlens.judging import NotJudged
from foqlens.prompt_variants import examples_for, setup_named
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

FROZEN = Path("corpus/e2b-it")
LADDER = Path("runs/E002-base-precision-d2/ladder/e2b-it")
PER_CORPUS = 17  # six corpora: 102 questions
# Only a short answer can be held word for word: bf16 against itself in other batches (E001, bf16-unknown-again)
# keeps 0.95 of TriviaQA, 0.98 of SQuAD, 0.89 of NQ-open, but 0.02 of ARC-Challenge's 488-token solutions.
SHORT_CORPORA = ("triviaqa", "squad_v2", "nq_open")
# bf16's own share on these three is 0.96; the copy read by blocks kept 0.94-1.00 at every level (2026-09-17)
SHORT_ANSWERS_SAME = 0.90
LEVELS = (Level.D8, Level.D6, Level.D4, Level.D2)
SEED = 0


@pytest.fixture(scope="module")
def questions():
    setups = json.loads((LADDER / "summary-d2.json").read_text(encoding="utf-8"))["setups"]
    picked = {}
    for corpus, setup_name in setups.items():
        kept = sorted(read_frozen(FROZEN / f"{corpus}.json").kept)[:PER_CORPUS]
        by_id = {r.id: r for r in corpora.read(corpus)[0]}
        picked[corpus] = (setup_named(corpus, setup_name), [by_id[i] for i in kept])
    return picked


@pytest.fixture(scope="module")
def replies(e2b_it_refocused, questions):
    """{level: {(corpus, id): (reply now, reply committed)}}"""
    model, tokenizer, ctl = e2b_it_refocused
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    decoder = StaticDecoder(attention=SPLIT)
    out = {}
    for level in LEVELS:
        pairs = {}
        for corpus, (setup, rows) in questions.items():
            committed = {a.id: a for a in read_answers(LADDER / "unjudged" / level.name.lower() / f"{corpus}.jsonl")}
            asking = Asking(corpus, "", "", level, setup, examples_for(corpus, setup, [], SEED))
            for batch in asking.batches(fmt, tokenizer, rows):
                for a in asking.answer(model, tokenizer, ctl, fmt, NotJudged(), batch, decoder=decoder):
                    pairs[(corpus, a.id)] = (a, committed[a.id])
        out[level] = pairs
    return out


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.name)
def test_the_copy_read_by_blocks_answers_as_the_committed_ladder(replies, level):
    pairs = replies[level]
    assert len(pairs) == PER_CORPUS * 6
    short = [(now, then) for (corpus, _), (now, then) in pairs.items() if corpus in SHORT_CORPORA]
    same = sum(now.answer == then.answer for now, then in short) / len(short)
    print(f"{level.name}: short answers {same:.2f} of {len(short)} the same")
    assert same >= SHORT_ANSWERS_SAME, level
