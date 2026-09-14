"""How a conversation becomes the text a checkpoint reads: a document for a base model, the chat
template for an instruction-tuned one.

A prompt is built once as messages - the examples as earlier turns, the question as the last user
turn - and a format renders them. The same messages give the base checkpoint the document it has
to continue ("Question: ...\\nAnswer: ...") and the -it checkpoint its turns, so a prompt is one
thing whichever model reads it.

`cue` opens the model's reply so that its next word is the answer, " Paris" after "Answer:": a base
checkpoint's document already ends with the cue, an -it model's turn is opened with it. A judge that
reads the log-probability of one word needs that word first - left to itself an -it model starts with
"**Answer: Yes**", and its first token is no verdict at all (2026-09-14, HotpotQA).

Invariant: the messages alternate user, assistant, ..., and end with the user's turn the model answers.
Invariant: PlainFormat renders exactly the prompts the base checkpoint was always given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

USER, ASSISTANT = "user", "assistant"  # the Gemma 4 template turns assistant into its model turn
ANSWER_CUE = "\nAnswer:"  # where a base checkpoint is shown that the answer comes next

Message = dict[str, str]


def check_turns(messages: list[Message]) -> None:
    roles = [m["role"] for m in messages]
    expected = [USER if i % 2 == 0 else ASSISTANT for i in range(len(roles))]
    if not roles or roles != expected or roles[-1] != USER:
        raise ValueError(f"turns must alternate user/assistant and end with the user: {roles}")


class PromptFormat(Protocol):
    cue: str  # appended to a rendered prompt, it opens the reply so the next word is the answer

    def render(self, messages: list[Message]) -> str:
        """The text the model reads; the reply starts right after it."""
        ...


@dataclass(frozen=True)
class PlainFormat:
    """A base checkpoint: every turn a block ending in the answer cue, the blocks apart by a blank line."""

    cue: str = ""  # the document already ends with it

    def render(self, messages: list[Message]) -> str:
        check_turns(messages)
        blocks = []
        for m in messages:
            if m["role"] == USER:
                blocks.append(m["content"] + ANSWER_CUE)
            else:
                blocks[-1] += " " + m["content"]  # the answer follows its cue after a space: "Answer: Paris"
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class ChatFormat:
    """An instruction-tuned checkpoint: its own chat template, with the model's turn opened."""

    tokenizer: object
    cue: str = ANSWER_CUE.strip()

    def render(self, messages: list[Message]) -> str:
        check_turns(messages)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


PLAIN = PlainFormat()
