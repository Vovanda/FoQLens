"""The static cache and its CUDA graph against the eager static step and against the dynamic loop, on E2B-it.

Two claims, each over the axes stage 1 and stage 2 run on:
- the graph replays the static step exactly: the same tokens as the step run eagerly, at every precision
  layout, stop rule, answer length and batch shape;
- the static cache reads what the dynamic loop reads: fed the dynamic loop's own tokens, it picks the same
  next token at every step, except where the two best tokens nearly tie. Feeding the same tokens checks every
  step of every row, where comparing two free runs would stop checking a row at its first near tie.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import DynamicCache

from foqlens import model as fm
from foqlens.answering import BATCH_TOKENS
from foqlens.extractive import qa_prompt
from foqlens.generation import END_OF_TURN, FIRST_LINE, mask_positions
from foqlens.graph_decode import StaticDecoder, StaticRun
from foqlens.io import read_jsonl
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
DOMAINS = ("biology", "chemistry", "math", "physics")  # never the held-out topics
# Passages of 32 and 40 questions of a topic (~850 ... 1800 tokens), past the sliding window of 512, so
# the sliding layers stop seeing the start of the prompt both in the prefill and in every step. 16 prompts
# padded to the longest stay under the 32k tokens a batch of the runs holds (answering.BATCH_TOKENS).
PASSAGE_QUESTIONS = (32, 40)
TOKENS = 48
SHORT_TOKENS = 32   # a short answer of the runs (prompt_variants.SHORT_TOKENS)
# A written answer of the runs is up to 512 tokens (prompt_variants.WRITTEN_TOKENS): on a prompt of ~40
# tokens the query then moves past slot 512 and the sliding window slides over the answer itself.
WRITTEN_TOKENS = 560
# The static cache sums its keys in another order than the dynamic one, so where the two best tokens nearly
# tie the other one can win. Measured 2026-09-14 on free runs: the three rows of 16 that parted did so where
# the dynamic loop's top two logits were 0.06, 0.13 and 0.25 apart, while the median step has them 5.7
# apart and the 5th percentile 0.48. A pick that differs anywhere wider is a key one path sees and the other not.
NEAR_TIE = 0.5
NEVER = -1  # no token id is negative: a row with this stop runs every step
# Levels the precision layouts of the tests draw from; every layout is seeded.
MIXED_LEVELS = np.array([Level.BF16, Level.D8, Level.D4, Level.D2], dtype=np.uint8)
SERIES = 60                # batches in a row, as a night of stage 1 decodes hundreds
SERIES_GROWTH = 64 * 2**20  # allocated memory may move by fragments of a step, never by a batch's cache


def questions(domain: str, n: int) -> list[str]:
    return [q["text"] for q in read_jsonl(PROMPTS / f"{domain}.jsonl", n)]


@pytest.fixture(scope="module")
def batches(e2b_it_sdpa) -> dict[str, list[str]]:
    _, tokenizer, _ = e2b_it_sdpa
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    ask = lambda text, context=None: qa_prompt(context, text, (), fmt)  # noqa: E731
    short = [ask(t) for d in DOMAINS for t in questions(d, 32)]
    long = [ask("Which of the questions in the passage is the hardest, and why?", " ".join(questions(d, 40)[:n]))
            for d in DOMAINS for n in PASSAGE_QUESTIONS]
    return {
        "mixed": [short[i * 32 + j] for i in range(len(DOMAINS)) for j in range(2)] + long,  # 8 short, 8 past the window
        "short128": short,
        "short8": short[::16],
        "one": long[-1:],
        "unpadded": [short[0]] * 4,  # no padding: the dynamic loop's masks may then skip to is_causal
    }


@pytest.fixture(autouse=True)
def bf16(e2b_it_sdpa):
    e2b_it_sdpa[2].set_all(Level.BF16)
    yield
    e2b_it_sdpa[2].set_all(Level.BF16)


def set_layout(ctl, layout: str, batch: int) -> None:
    """A level for the whole model, or a seeded mixed layout - per block, or per block and sample."""
    rng = np.random.default_rng(0)
    if layout == "blocks":
        ctl.set_layout(rng.choice(MIXED_LEVELS, size=ctl.n_blocks))
    elif layout == "samples":
        ctl.set_layout(rng.choice(MIXED_LEVELS, size=(batch, ctl.n_blocks)))
    else:
        ctl.set_all(Level[layout])


def encode(model, tokenizer, prompts):
    enc = fm.encode_left(tokenizer, prompts, model.device)
    return enc["input_ids"], enc["attention_mask"]


def stop_ids(model, tokenizer, stop) -> torch.Tensor:
    ids = (NEVER,) if stop is None else stop.ids(model, tokenizer)
    return torch.tensor(ids, device=model.device)


def test_the_long_prompts_reach_past_the_sliding_window(e2b_it_sdpa, batches):
    model, tokenizer, _ = e2b_it_sdpa
    window = model.config.get_text_config(decoder=True).sliding_window
    lengths = [len(ids) for ids in tokenizer(batches["mixed"])["input_ids"]]
    print(f"prompt tokens: {lengths}")
    assert min(lengths[8:]) > window and max(lengths) * len(lengths) <= BATCH_TOKENS
    assert max(len(ids) for ids in tokenizer(batches["short8"])["input_ids"]) + WRITTEN_TOKENS > window + 64


# (batch, layout, stop, tokens): the layouts on the mixed batch; the stop rules on the batch of the runs;
# the lengths around the first step, the capture and the look every 8 steps; the odd shapes.
CAPTURE_CASES = (
    [("mixed", layout, END_OF_TURN, TOKENS) for layout in ("BF16", "D8", "D6", "D4", "D2", "blocks", "samples")]
    + [("short128", "BF16", stop, SHORT_TOKENS) for stop in (FIRST_LINE, END_OF_TURN)]
    + [("mixed", "BF16", None, n) for n in (1, 2, 8, 9)]
    + [("one", "BF16", END_OF_TURN, TOKENS), ("unpadded", "BF16", END_OF_TURN, TOKENS)]
)


@pytest.mark.parametrize("batch, layout, stop, tokens", CAPTURE_CASES,
                         ids=[f"{b}-{l}-{s.__class__.__name__ if s else 'never'}-{n}" for b, l, s, n in CAPTURE_CASES])
def test_the_graph_writes_what_the_same_step_writes_eagerly(e2b_it_sdpa, batches, batch, layout, stop, tokens):
    model, tokenizer, ctl = e2b_it_sdpa
    prompts = batches[batch]
    set_layout(ctl, layout, len(prompts))
    ids, mask = encode(model, tokenizer, prompts)
    stop = stop_ids(model, tokenizer, stop)
    eager = StaticDecoder(graph=False).tokens(model, ids, mask, tokens, stop)
    graphed = StaticDecoder(graph=True).tokens(model, ids, mask, tokens, stop)
    print(f"{batch} {layout}: {tuple(graphed.shape)}")
    assert torch.equal(eager, graphed)


def test_a_batch_of_short_answers_ends_at_the_first_look_after_its_last_line(e2b_it_sdpa, batches):
    """Every row of a short answer ends its line: the batch stops at a look, not at the limit."""
    model, tokenizer, _ = e2b_it_sdpa
    ids, mask = encode(model, tokenizer, batches["short8"])
    stop = stop_ids(model, tokenizer, FIRST_LINE)
    graphed = StaticDecoder().tokens(model, ids, mask, 4 * SHORT_TOKENS, stop)
    written = (graphed[..., None] == stop).any(-1).int().cummax(-1).values
    last_line = int((1 - written).sum(-1).max()) + 1  # steps the slowest row needs to write its stop token
    print(f"steps run {graphed.shape[1]}, the slowest row stops at step {last_line}")
    assert graphed.shape[1] < 4 * SHORT_TOKENS and graphed.shape[1] % 8 == 0
    assert last_line <= graphed.shape[1] < last_line + 8


@torch.no_grad()
def dynamic_margins(model, input_ids, attention_mask, steps) -> tuple[torch.Tensor, torch.Tensor]:
    """The dynamic loop's tokens for `steps` steps and the gap between its two best logits at every step."""
    cache = DynamicCache(config=model.config.get_text_config(decoder=True))
    positions = mask_positions(attention_mask)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    tokens, margins, position = [], [], positions[:, -1:]
    for _ in range(steps):
        logits = out.logits[:, -1]
        # the token as the loop picks it: bf16 logits tie exactly (their step is 1/16 ... 1/8 at the values a
        # softcap of 30 leaves), and argmax takes the first of a tie where topk takes any
        tokens.append(logits.argmax(-1))
        top = logits.float().topk(2, -1).values
        margins.append(top[:, 0] - top[:, 1])
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
        position = position + 1
        out = model(input_ids=tokens[-1][:, None], attention_mask=attention_mask, position_ids=position,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    return torch.stack(tokens, 1), torch.stack(margins, 1)


@torch.no_grad()
def static_picks_on(model, input_ids, attention_mask, fed: torch.Tensor) -> torch.Tensor:
    """What the static step picks at every step when it is fed `fed` - the tokens of another loop - instead of its own."""
    run = StaticRun(model, input_ids, attention_mask, fed.shape[1], torch.tensor([NEVER], device=model.device))
    for step in range(1, fed.shape[1]):
        run.token.copy_(fed[:, step - 1 : step])
        run.advance()
    return run.tokens


FED_CASES = [("mixed", TOKENS), ("short8", WRITTEN_TOKENS), ("short128", SHORT_TOKENS), ("one", TOKENS),
             ("unpadded", TOKENS)]


@pytest.mark.parametrize("batch, steps", FED_CASES, ids=[f"{b}-{n}" for b, n in FED_CASES])
def test_the_static_cache_picks_what_the_dynamic_loop_picks_but_at_near_ties(e2b_it_sdpa, batches, batch, steps):
    model, tokenizer, _ = e2b_it_sdpa
    ids, mask = encode(model, tokenizer, batches[batch])
    dynamic, margins = dynamic_margins(model, ids, mask, steps)
    picks = static_picks_on(model, ids, mask, dynamic)
    differ = picks != dynamic
    worst = float(margins[differ].max()) if differ.any() else 0.0
    print(f"{batch}: {int(differ.sum())} of {differ.numel()} picks differ, the widest at a gap of {worst:.3f}; "
          f"rows with no difference {int((~differ.any(-1)).sum())}/{len(differ)}")
    assert (margins[differ] < NEAR_TIE).all()


def test_a_night_of_batches_keeps_its_memory_and_its_answers(e2b_it_sdpa, batches):
    """Graphs are captured and released batch after batch: memory stays flat and a batch reads the same at the end."""
    model, tokenizer, _ = e2b_it_sdpa
    rng = np.random.default_rng(0)
    pool = batches["short128"] + batches["mixed"][8:]
    stop = stop_ids(model, tokenizer, END_OF_TURN)
    first_ids, first_mask = encode(model, tokenizer, batches["mixed"])
    first = StaticDecoder().tokens(model, first_ids, first_mask, 8, stop)
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    for _ in range(SERIES):
        size = int(rng.integers(1, 33))
        chosen = [pool[i] for i in rng.choice(len(pool), size=size, replace=False)]
        ids, mask = encode(model, tokenizer, chosen)
        StaticDecoder().tokens(model, ids, mask, int(rng.integers(2, 24)), stop)
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - start
    print(f"allocated after {SERIES} batches: {grown / 2**20:+.1f} MiB")
    assert grown < SERIES_GROWTH
    assert torch.equal(first, StaticDecoder().tokens(model, first_ids, first_mask, 8, stop))
