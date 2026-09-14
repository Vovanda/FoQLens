"""One batch answered in one setup: written at its level, judged at bf16, one answer line per question in order."""

import pytest

from foqlens import answering
from foqlens.answering import Asking, level_label
from foqlens.corpora import Row
from foqlens.generation import Reply
from foqlens.prompt_variants import SETUPS, setup_named
from foqlens.prompting import PLAIN
from foqlens.quant import Level

ROWS = [Row("1", "Capital of France?", ("Paris",)), Row("2", "Who wrote Hamlet?", ("William Shakespeare", "Shakespeare"))]


class Controller:
    def __init__(self, log):
        self.log = log

    def set_all(self, level):
        self.log.append(("level", level))


class Judge:
    def __init__(self, log):
        self.log = log

    def p_yes(self, questions, answers, references):
        self.log.append(("judge", references is not None))
        return [0.9 if a == "Paris" else 0.2 for a in answers]


@pytest.fixture
def run(monkeypatch):
    log = []

    def replies(model, tokenizer, prompts, max_new_tokens, stop):
        log.append(("generate", len(prompts), max_new_tokens))
        return [Reply("It is the capital.\nAnswer: Paris", 8, True),
                Reply("It was Marlowe.\n**Answer:** Marlowe", 9, False)][: len(prompts)]

    monkeypatch.setattr(answering, "generate_replies", replies)
    return log


def test_a_batch_is_written_at_its_level_then_judged_with_the_reference_and_without(run):
    asking = Asking("arc_challenge_closed", "rev", "m@1", Level.D4, setup_named("arc_challenge_closed", "solve-0"))
    answers = asking.answer(None, None, Controller(run), PLAIN, Judge(run), ROWS)
    assert run == [("level", Level.D4), ("generate", 2, asking.setup.max_new_tokens), ("judge", True), ("judge", False)]
    assert [a.id for a in answers] == ["1", "2"]
    assert answers[1].reasoning == "It was Marlowe." and answers[1].answer == "Marlowe"
    assert (answers[1].exact_match, answers[1].judge_with_reference, answers[1].stopped) == (0.0, 0.2, False)
    assert (answers[0].exact_match, answers[0].f1, answers[0].level, answers[0].prompt) == (1.0, 1.0, "d4", "solve-0")


def test_the_prompts_carry_the_setup_and_its_examples():
    asking = Asking("triviaqa", "rev", "m", Level.BF16, setup_named("triviaqa", "short-2a"),
                    ((None, "Capital of Italy?", "Rome"),))
    (prompt,) = asking.prompts(PLAIN, ROWS[:1])
    assert "Capital of Italy?" in prompt and "Rome" in prompt and prompt.endswith("Question: Capital of France?\nAnswer:")


class Lengths:
    """A tokenizer whose token count is the prompt's character count."""

    def __call__(self, prompts):
        return {"input_ids": [[0] * len(p) for p in prompts]}


def test_batches_hold_every_row_once_within_the_token_budget(monkeypatch):
    monkeypatch.setattr(answering, "BATCH_TOKENS", 400)
    monkeypatch.setattr(answering, "BATCH", 3)
    rows = [Row(str(i), "q" * (10 + 40 * (i % 4)), ("a",)) for i in range(11)]
    asking = Asking("triviaqa", "rev", "m", Level.BF16, setup_named("triviaqa", "bare"))
    batches = asking.batches(PLAIN, Lengths(), rows)
    assert sorted(r.id for b in batches for r in b) == sorted(r.id for r in rows)
    for b in batches:
        longest = max(len(p) for p in asking.prompts(PLAIN, b))
        assert len(b) <= 3 and (len(b) * longest <= 400 or len(b) == 1)


def test_levels_are_named_as_the_answer_files_are():
    assert [level_label(level) for level in (Level.BF16, Level.D8, Level.D4, Level.D2)] == ["bf16", "d8", "d4", "d2"]


def test_an_unknown_setup_is_refused_with_the_ones_there_are():
    with pytest.raises(ValueError, match="short-0"):
        setup_named("triviaqa", "nonsense")
    assert {s.name for s in SETUPS["hotpotqa"]} == {"justify", "justify-terse"}
