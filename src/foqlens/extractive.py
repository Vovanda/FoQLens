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

Invariant: greedy decoding - the same layout and the same prompt give the same text, hence the same
score. Scores never depend on the order of anything.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

import torch

from foqlens import model as fm

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


def qa_prompt(context: str, question: str, shots: tuple = ()) -> str:
    """A base checkpoint has no chat template, so the format is shown to it in a couple of examples."""
    blocks = [f"Context: {c}\nQuestion: {q}\nAnswer: {a}" for c, q, a in shots]
    blocks.append(f"Context: {context}\nQuestion: {question}\nAnswer:")
    return "\n\n".join(blocks)


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

    @torch.no_grad()
    def score(self, model, tokenizer, questions: list) -> list[dict[str, float]]:
        rows = []
        for question in questions:
            inputs = fm.encode(tokenizer, [question.prompt], model.device)
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            written = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answer = first_line(written)
            references = self.passages[question.prompt]
            rows.append({"exact_match": exact_match(answer, references), "f1": token_f1(answer, references)})
        return rows
