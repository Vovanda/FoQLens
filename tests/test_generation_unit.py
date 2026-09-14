"""Greedy generation a batch at a time: positions from the mask, a mask that grows, rows that stop on their own."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from foqlens import extractive, generation
from foqlens.evaluate import Question
from foqlens.generation import answer_texts, end_ids, greedy_tokens, mask_positions, newline_ids

PAD, VOCAB = 0, 17
NEVER = torch.tensor([VOCAB])  # a stop token the toy never writes
ROWS = [[5, 3], [2, 7, 1, 4, 9], [8]]


class CountingModel:
    """A causal toy: the next token is a function of the row's real tokens and of the last position.

    It keeps what it has been fed on the cache, and checks on every call that the mask covers all of it
    and that the positions are the ones the mask gives - so a batch agrees with its rows alone only when
    the loop feeds both right.
    """

    config = SimpleNamespace(get_text_config=lambda decoder=False: PretrainedConfig(num_hidden_layers=1))

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, logits_to_keep):
        self.calls += 1
        seen = getattr(past_key_values, "seen", input_ids.new_empty(input_ids.shape[0], 0))
        seen = past_key_values.seen = torch.cat([seen, input_ids], dim=-1)
        assert attention_mask.shape == seen.shape and logits_to_keep == 1
        assert torch.equal(position_ids, mask_positions(attention_mask)[:, -input_ids.shape[1]:])
        nxt = ((seen * attention_mask).sum(-1) + 3 * position_ids[:, -1]) % VOCAB
        return SimpleNamespace(logits=torch.nn.functional.one_hot(nxt, VOCAB).float()[:, None])


def left_padded(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows))
    ids = torch.tensor([[PAD] * (width - len(r)) + r for r in rows])
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    return ids, mask


def test_positions_count_from_the_first_real_token_of_each_row():
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert mask_positions(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_a_left_padded_batch_writes_what_each_row_writes_alone():
    batched = greedy_tokens(CountingModel(), *left_padded(ROWS), max_new_tokens=6, stop=NEVER)
    for i, row in enumerate(ROWS):
        alone = greedy_tokens(CountingModel(), *left_padded([row]), max_new_tokens=6, stop=NEVER)
        assert batched[i].tolist() == alone[0].tolist()


def test_without_a_stop_the_loop_writes_exactly_max_new_tokens():
    assert greedy_tokens(CountingModel(), *left_padded([[1, 2]]), 1, NEVER).shape == (1, 1)
    assert greedy_tokens(CountingModel(), *left_padded([[1, 2]]), 4, NEVER).shape == (1, 4)


def steps_to_run(full: list[list[int]], stop: set[int], check_every: int, tokens: int) -> int:
    """The rule: the first look (every check_every steps) after the last row wrote a stop token, or the limit."""
    firsts = [next((i for i, t in enumerate(row) if t in stop), None) for row in full]
    if None in firsts:
        return tokens
    return min(-(-(max(firsts) + 1) // check_every) * check_every, tokens)


@pytest.mark.parametrize("stop", [{3, 10}, {3}])  # the toy's rows 0 and 2 write 3, row 1 writes 10 and never 3
@pytest.mark.parametrize("check_every", [1, 3, 8])
def test_the_batch_ends_at_the_first_look_after_its_last_row_stopped(stop, check_every):
    """A look every check_every steps is the only wait for the device; rows stopped earlier run on until it."""
    tokens = 30
    full = greedy_tokens(CountingModel(), *left_padded(ROWS), tokens, NEVER).tolist()
    model = CountingModel()
    out = greedy_tokens(model, *left_padded(ROWS), tokens, torch.tensor(sorted(stop)), check_every)
    assert out.shape[1] == steps_to_run(full, stop, check_every, tokens)
    assert out.tolist() == [row[: out.shape[1]] for row in full]
    assert model.calls == out.shape[1]  # the prefill and one call per step after the first


class Letters:
    """A tokenizer that spells token i as the letter i, token 9 as a newline, and drops the special tokens 0 and 1."""

    def __len__(self):
        return 12

    def spell(self, i: int) -> str:
        return "\n" if i == 9 else chr(ord("a") + i) if i > 1 else ""

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return "".join(self.spell(i) for i in ids)

    def batch_decode(self, rows):
        return ["".join(self.spell(i) for i in row) for row in rows]


def test_a_row_ends_at_its_first_stop_token_without_it():
    tokens = torch.tensor([[2, 3, 1, 4, 5], [6, 7, 8, 10, 11], [1, 2, 3, 4, 5], [2, 9, 3, 1, 4]])
    assert answer_texts(Letters(), tokens, stop=(1, 9)) == ["cd", "ghikl", "", "c"]


def test_the_newline_tokens_are_found_by_decoding_the_vocabulary():
    assert newline_ids(Letters()) == (9,)


def test_a_short_answer_stops_at_a_newline_and_a_written_one_only_at_the_end_of_the_turn():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 106]))
    assert generation.FIRST_LINE.ids(model, Letters()) == (1, 106, 9)
    assert generation.END_OF_TURN.ids(model, Letters()) == (1, 106)


@pytest.mark.parametrize("eos, expected", [(1, (1,)), ([1, 106], (1, 106))])
def test_the_end_tokens_are_read_as_generate_reads_them(eos, expected):
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))
    assert end_ids(model) == expected


def test_context_qa_scores_the_first_line_of_what_the_batch_wrote(monkeypatch):
    written = {"p1": " Paris", "p2": "Lyon"}
    monkeypatch.setattr(extractive, "generate_answers",
                        lambda model, tok, prompts, tokens: [written[p] for p in prompts])
    metric = extractive.ContextQA(passages={"p1": ["Paris"], "p2": ["Paris"]})
    rows = metric.score(None, None, [Question("x", "p1", 0), Question("x", "p2", 0)])
    assert [r["exact_match"] for r in rows] == [1.0, 0.0]


def test_free_answers_go_in_batches_and_keep_the_order(monkeypatch):
    calls = []

    def fake(model, tok, prompts, tokens):
        calls.append(len(prompts))
        return [f" {p} " for p in prompts]

    monkeypatch.setattr(generation, "generate_answers", fake)
    from foqlens.matching import free_answers

    assert free_answers(None, None, list("abcde"), 8, batch_size=2) == list("abcde")
    assert calls == [2, 2, 1]
