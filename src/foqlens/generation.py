"""Free-text answers of the model under a precision layout, greedy, a batch of prompts at a time.

Decoding is bound by reading the weights: one step costs the same for one row and for thirty-two
(250 ms at bf16 either way, 14.09 profile), so a row of the batch is nearly free and one prompt at a
time throws the card away.

The batch is padded on the left, so every prompt ends where its answer starts, and each row takes its
positions from its own attention mask - the way generate() builds them (transformers 5.17,
generation/utils.py: cumsum of the mask minus one, padding at 0, then one more per step). A plain
forward would take them from arange(seq) and shift every real token of a padded row (model.load).
The loop is our own rather than generate(): the static cache and the CUDA graph of the next steps
need it, and generate() with a static cache compiles the step with triton.

Where an answer ends is a StopRule. A short answer is its first line (FIRST_LINE): after it a base
checkpoint goes on to invent the next question, and in Gemma 4 a newline is always a token of its
own, so nothing of the line is lost. A written solution or justification is the whole reply, to the
end of the model's turn (END_OF_TURN). Every row stops on its own, and the batch stops when every row
has. Whether they all have is read on the host once every STOP_CHECK_EVERY steps, the only
synchronization of the loop.

`generate_answer` is the reference: generate() on one prompt, no padding at all.

Invariant: greedy decoding - the same layout and the same prompt in the same batch give the same text.
Invariant: an answer is the first line of what generate() writes for the same batch, exactly.
Invariant (approximate, bf16): a prompt answered in a batch writes the text it writes alone; bf16 is
not batch-invariant, so a near tie between two tokens can go the other way (measured: 30 of 32 prompts
write the same text in a batch of 32 as alone, tests/test_generation_gpu.py). Comparisons between
layouts therefore always use the same batches.
Invariant: the host reads the device once every STOP_CHECK_EVERY steps and once at the end, never more.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import torch
from transformers import DynamicCache

from foqlens import model as fm

# Steps between two looks at whether every row has stopped: a look waits for the device, so it costs
# the pipelining of one step; after the last row stops at most this many steps are run for nothing.
STOP_CHECK_EVERY = 8


def question_prompt(question: str) -> str:
    """A plain question for a free answer, without the options."""
    return f"Question: {question}\nAnswer:"


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """generate() on one prompt: the reference the batched loop is held to."""
    inputs = fm.encode(tokenizer, [prompt], model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def mask_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids of a left-padded batch: each row counts from 0 at its first real token, padding at 0."""
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


@torch.no_grad()
def greedy_tokens(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, stop: torch.Tensor,
    check_every: int = STOP_CHECK_EVERY,
) -> torch.Tensor:
    """The greedy continuation of a left-padded batch: [batch, steps run] on the device.

    A row is done once it writes a token of `stop`; what it writes after that is cut on the host
    (answer_texts). The loop ends at `max_new_tokens` or at the first look that finds every row done.
    Only the logits of the last position are computed (logits_to_keep=1).
    """
    batch = input_ids.shape[0]
    tokens = torch.empty(batch, max_new_tokens, dtype=torch.long, device=input_ids.device)
    done = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    cache = DynamicCache(config=model.config.get_text_config(decoder=True))
    positions = mask_positions(attention_mask)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                past_key_values=cache, use_cache=True, logits_to_keep=1)
    step_mask = attention_mask.new_ones(batch, 1)
    position = positions[:, -1:]
    for step in range(max_new_tokens):
        tokens[:, step] = out.logits[:, -1].argmax(-1)
        done |= (tokens[:, step, None] == stop).any(-1)  # torch.isin waits for the device three times a step
        if step + 1 == max_new_tokens or ((step + 1) % check_every == 0 and bool(done.all())):
            return tokens[:, : step + 1]
        attention_mask = torch.cat([attention_mask, step_mask], dim=-1)
        position = position + 1
        out = model(input_ids=tokens[:, step : step + 1], attention_mask=attention_mask, position_ids=position,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    return tokens


def end_ids(model) -> tuple[int, ...]:
    """The tokens that end a text, as generate() reads them from the generation config."""
    eos = model.generation_config.eos_token_id
    return tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)


@lru_cache(maxsize=None)
def newline_ids(tokenizer) -> tuple[int, ...]:
    """Every token that writes a newline; the vocabulary is decoded once per tokenizer (262k tokens, ~2 s)."""
    texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    return tuple(i for i, text in enumerate(texts) if "\n" in text)


@dataclass(frozen=True)
class Reply:
    text: str
    tokens: int     # the tokens of the text, the stop token not counted
    stopped: bool   # it ended on a stop token rather than at the limit


def replies(tokenizer, tokens: torch.Tensor, stop: tuple[int, ...]) -> list[Reply]:
    """Each row decoded up to its first stop token, without it."""
    out = []
    for row in tokens.tolist():
        end = next((i for i, token in enumerate(row) if token in stop), None)
        cut = row if end is None else row[:end]
        out.append(Reply(tokenizer.decode(cut, skip_special_tokens=True), len(cut), end is not None))
    return out


def answer_texts(tokenizer, tokens: torch.Tensor, stop: tuple[int, ...]) -> list[str]:
    return [r.text for r in replies(tokenizer, tokens, stop)]


class StopRule(Protocol):
    def ids(self, model, tokenizer) -> tuple[int, ...]:
        """The tokens that end an answer; the text is cut before the first of them."""
        ...


@dataclass(frozen=True)
class FirstLine:
    """A short answer: the first line of what the model writes."""

    def ids(self, model, tokenizer) -> tuple[int, ...]:
        return end_ids(model) + newline_ids(tokenizer)


@dataclass(frozen=True)
class EndOfTurn:
    """A written solution or justification: the whole reply, to the end of the model's turn."""

    def ids(self, model, tokenizer) -> tuple[int, ...]:
        return end_ids(model)


FIRST_LINE, END_OF_TURN = FirstLine(), EndOfTurn()


def generate_replies(model, tokenizer, prompts: list[str], max_new_tokens: int, stop: StopRule = FIRST_LINE) -> list[Reply]:
    """What the model writes for each prompt, under the layout currently set, as one batch, cut by `stop`."""
    stop = stop.ids(model, tokenizer)
    enc = fm.encode_left(tokenizer, prompts, model.device)
    tokens = greedy_tokens(model, enc["input_ids"], enc["attention_mask"], max_new_tokens,
                           torch.tensor(stop, device=model.device))
    return replies(tokenizer, tokens, stop)


def generate_answers(model, tokenizer, prompts: list[str], max_new_tokens: int, stop: StopRule = FIRST_LINE) -> list[str]:
    return [r.text for r in generate_replies(model, tokenizer, prompts, max_new_tokens, stop)]
