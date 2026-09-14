"""One prompt, two forms: a document for a base checkpoint, the chat template for an instruction-tuned one."""

import pytest

from foqlens import model as fm
from foqlens.extractive import qa_messages, qa_prompt
from foqlens.prompting import ASSISTANT, PLAIN, USER, ChatFormat, PlainFormat

SHOTS = ((None, "Capital of France?", "Paris"),)


def test_the_plain_form_is_the_document_a_base_checkpoint_continues():
    assert qa_prompt(None, "Who wrote Hamlet?", SHOTS) == \
        "Question: Capital of France?\nAnswer: Paris\n\nQuestion: Who wrote Hamlet?\nAnswer:"


def test_examples_become_earlier_turns():
    roles = [m["role"] for m in qa_messages("ctx", "q", SHOTS * 2)]
    assert roles == [USER, ASSISTANT, USER, ASSISTANT, USER]


@pytest.mark.parametrize("roles", [[], [ASSISTANT], [USER, USER], [USER, ASSISTANT]])
def test_turns_that_do_not_alternate_or_end_with_the_user_are_refused(roles, fmt=PlainFormat()):
    with pytest.raises(ValueError):
        fmt.render([{"role": r, "content": "x"} for r in roles])


class Template:
    """A tokenizer that records how its chat template was asked for."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|model:"


def test_the_chat_form_is_the_tokenizer_template_with_the_model_turn_opened():
    tok = Template()
    text = qa_prompt(None, "Who wrote Hamlet?", SHOTS, ChatFormat(tok))
    assert text == "user:Question: Capital of France?|assistant:Paris|user:Question: Who wrote Hamlet?|model:"
    assert tok.calls[0][1:] == (False, True)


def test_the_cue_that_opens_a_reply_is_already_in_a_base_document_and_added_to_a_chat_turn():
    assert qa_prompt(None, "q").endswith("Answer:") and PLAIN.cue == ""
    assert ChatFormat(Template()).cue == "Answer:"


@pytest.mark.parametrize("model_id, chat", [(fm.E2B, False), (fm.E4B, False), (fm.E2B_IT, True), (fm.E4B_IT, True)])
def test_an_instruction_tuned_checkpoint_reads_its_chat_template(model_id, chat):
    assert isinstance(fm.prompt_format(model_id, Template()), ChatFormat) == chat


def test_every_checkpoint_the_bench_reads_is_pinned():
    assert {fm.E2B, fm.E4B, fm.E2B_IT, fm.E4B_IT} <= set(fm.REVISIONS)
