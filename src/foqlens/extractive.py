"""Question answering over a given context: the model reads the passage and writes the answer.

This is the metric the bench should have started with. The letter-choice metric turned out to measure
partly a lean toward a letter, and the address read off a prompt carrying the options moved by 6.3
points of accuracy when those options were merely reordered (runs/reference/address-stability). Here
there are no options: nothing to lean on, nothing to reorder, and the answer is in the passage, so
the model is asked to read rather than to recall. It is also the shape a precision regulator is
actually for - retrieval-augmented answering, not a quiz.

Scored as SQuAD scores it: exact match after normalization, and the token F1 against the best of the
reference answers. F1 is a continuous scale, so it sees damage that four letters cannot.

An unanswerable question (SQuAD v2) has no reference answers; the right response is to say so, and
`NO_ANSWER` is what counts as saying it.

Without a passage the same format and the same score ask for recall instead of reading: the answer
can then only come from the weights (closed-book QA - TriviaQA, NQ-open), which is the regime a
precision regulator is meant to decide.

Invariant: greedy decoding - the same layout and the same prompt in the same batch give the same text,
hence the same score (foqlens.generation). Scores never depend on the order of anything.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from foqlens.generation import generate_answers
from foqlens.prompting import ASSISTANT, PLAIN, USER, Message, PromptFormat

NO_ANSWER = "unanswerable"
ANSWER_TOKENS = 16  # a SQuAD answer is a span of a few words
ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, no punctuation, no articles, single spaces."""
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCTUATION)).split())


def exact_match(prediction: str, references: list[str]) -> float:
    gold = [normalize(r) for r in references] or [NO_ANSWER]
    return float(normalize(prediction) in gold)


def token_f1(prediction: str, references: list[str]) -> float:
    """The best token F1 against any reference - what SQuAD reports beside exact match."""
    predicted = normalize(prediction).split()
    best = 0.0
    for reference in references or [NO_ANSWER]:
        gold = normalize(reference).split()
        if not predicted or not gold:
            best = max(best, float(predicted == gold))
            continue
        shared = sum((Counter(predicted) & Counter(gold)).values())
        if shared:
            precision, recall = shared / len(predicted), shared / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def qa_question(context: str | None, question: str) -> str:
    """One question as the user asks it; no `Context:` line when there is no passage, so a closed-book question is bare."""
    head = f"Context: {context}\n" if context is not None else ""
    return f"{head}Question: {question}"


def qa_messages(context: str | None, question: str, shots: tuple = ()) -> list[Message]:
    """The examples as earlier turns, answered, then the question."""
    turns = []
    for c, q, a in shots:
        turns += [{"role": USER, "content": qa_question(c, q)}, {"role": ASSISTANT, "content": a}]
    return turns + [{"role": USER, "content": qa_question(context, question)}]


def qa_prompt(context: str | None, question: str, shots: tuple = (), fmt: PromptFormat = PLAIN) -> str:
    """The question with its examples, in the form the model reads (foqlens.prompting)."""
    return fmt.render(qa_messages(context, question, shots))


def first_line(text: str) -> str:
    """The answer, without the next example the base model goes on to invent."""
    return text.split("\n", 1)[0].strip()


@dataclass(frozen=True)
class ContextQA:
    """SQuAD-style answering: exact match and token F1 of what the model writes.

    `passages` holds the context, question and reference answers of every question, by prompt, since
    a Question carries only the prompt the model sees.
    """

    passages: dict[str, list[str]] = field(default_factory=dict)
    shots: tuple = ()
    max_new_tokens: int = ANSWER_TOKENS
    name: str = "context_qa"
    primary: str = "f1"

    def score(self, model, tokenizer, questions: list) -> list[dict[str, float]]:
        """The questions are answered as one batch (foqlens.generation), so the caller sizes the batch."""
        written = generate_answers(model, tokenizer, [q.prompt for q in questions], self.max_new_tokens)
        rows = []
        for question, text in zip(questions, written, strict=True):
            answer = first_line(text)
            references = self.passages[question.prompt]
            rows.append({"exact_match": exact_match(answer, references), "f1": token_f1(answer, references)})
        return rows
