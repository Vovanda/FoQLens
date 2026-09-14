"""The prompt setups tried on the tuning share of each corpus, before a single question is selected.

What the model is asked, and how, moves what it answers (Volodya 14.09: the input and the setting of
the task matter too). So each corpus is asked in several setups on its tuning share, and the setup
that answers best and most stably is frozen before the selection; the tuning questions are never
measured on again.

A setup is an instruction, a number of worked examples from the corpus's train split, and where the
answer ends. Two kinds of answer (Volodya 14.09):

- a short answer - TriviaQA, NQ-open, SQuAD v2: the answer alone, its first line;
- a written one - ARC-Challenge writes its solution, HotpotQA justifies its answer from the two
  passages; both end with "Answer: ..." on the last line, and the answer is taken from there.

The same setup given a different draw of examples measures stability: a setup whose accuracy moves
with the examples it happened to be shown is not one to freeze.

Invariant: a setup's examples come from the train split or from the worked examples below - never
from the questions it is measured on.
Invariant: the same corpus, setup and seed give the same prompts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from foqlens.corpora import Row
from foqlens.extractive import NO_ANSWER, first_line, qa_question
from foqlens.generation import END_OF_TURN, FIRST_LINE, StopRule
from foqlens.prompting import ASSISTANT, USER, Message
from foqlens.selection import JUDGE_YES, Answer, is_refusal, split_reasoning, strip_markup

SHORT = "Answer with the answer alone - a few words, not a sentence."
SHORT_PASSAGE = ("Answer from the passage with the answer alone - a few words, not a sentence. "
                 "If the passage does not contain the answer, reply: " + NO_ANSWER)
SOLVE = "Solve the question step by step. Then give the answer on the last line as: Answer: <answer>"
SOLVE_BRIEF = ("Solve the question in a few short steps, plain text, no headings. "
               "Then give the answer on the last line as: Answer: <answer>")
JUSTIFY = ("Say briefly which facts in the passages lead to the answer. "
           "Then give the answer on the last line as: Answer: <answer>")
JUSTIFY_TERSE = "Justify the answer in one or two sentences, then end with a line: Answer: <answer>"

SHORT_TOKENS = 32     # a few words; beyond this the model is explaining itself
# An -it model writes its solution in markdown steps: at 256 tokens 13 of 16 ARC replies ran out before
# their answer line (smoke 2026-09-14).
WRITTEN_TOKENS = 512

# Worked ARC-Challenge solutions, written and checked against the dataset's answer (train split, allenai/ai2_arc).
WORKED = (
    ("Mercury_7041860",
     "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels "
     "northeast. In which direction is the wind most likely applying force to the sails of the boat?",
     "The current pushes the boat north. The boat moves northeast, so something also pushes it east. The only "
     "other force is the wind, so the wind pushes the sails toward the east.\nAnswer: east"),
    ("Mercury_SC_415702",
     "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
     "Rubbing makes heat through friction, and more friction makes more heat. Water, oil and lotion all make "
     "the skin slippery and lower the friction. Dry skin grips the most, so dry palms produce the most heat."
     "\nAnswer: dry palms"),
    ("MDSA_2008_5_30",
     "On Earth, water can be a solid, a liquid, or a gas. Which energy source has the greatest influence on "
     "the state of matter of water?",
     "Whether water is ice, liquid or vapour depends on its temperature. What warms the water on Earth's "
     "surface, melting ice and evaporating water, is mostly sunlight. So the sun has the greatest influence."
     "\nAnswer: the sun"),
)

@dataclass(frozen=True)
class Variant:
    name: str
    instruction: str | None  # put before every question, the examples' included; None asks the bare question
    shots: int = 0
    shot_set: int = 0        # which draw of examples: two sets of the same size measure stability
    written: bool = False    # the reply reasons first and ends with "Answer: ..."

    @property
    def stop(self) -> StopRule:
        return END_OF_TURN if self.written else FIRST_LINE

    @property
    def max_new_tokens(self) -> int:
        return WRITTEN_TOKENS if self.written else SHORT_TOKENS

    def ask(self, context: str | None, question: str) -> str:
        text = qa_question(context, question)
        return text if self.instruction is None else f"{self.instruction}\n\n{text}"

    def messages(self, context: str | None, question: str, examples: tuple = ()) -> list[Message]:
        """Worked examples as earlier turns - (context, question, reply) each - then the question."""
        turns = []
        for c, q, reply in examples:
            turns += [{"role": USER, "content": self.ask(c, q)}, {"role": ASSISTANT, "content": reply}]
        return turns + [{"role": USER, "content": self.ask(context, question)}]

    def extract(self, reply: str) -> tuple[str | None, str]:
        """The reasoning, where there is one, and the answer without its markup; "" where a written reply gave none."""
        if self.written:
            reasoning, answer = split_reasoning(reply)
            return reasoning, answer or ""
        return None, strip_markup(first_line(reply))


SHORT_SETUPS = (
    Variant("bare", None),
    Variant("short-0", SHORT),
    Variant("short-2a", SHORT, shots=2, shot_set=0),
    Variant("short-2b", SHORT, shots=2, shot_set=1),
    Variant("short-5", SHORT, shots=5),
)
PASSAGE_SETUPS = (
    Variant("bare", None),
    Variant("passage-0", SHORT_PASSAGE),
    Variant("passage-2a", SHORT_PASSAGE, shots=2, shot_set=0),
    Variant("passage-2b", SHORT_PASSAGE, shots=2, shot_set=1),
)
SETUPS = {
    "triviaqa": SHORT_SETUPS,
    "nq_open": SHORT_SETUPS,
    "squad_v2": PASSAGE_SETUPS,
    "arc_challenge_closed": (
        Variant("solve-0", SOLVE, written=True),
        Variant("solve-brief", SOLVE_BRIEF, written=True),
        Variant("solve-2a", SOLVE, shots=2, shot_set=0, written=True),
        Variant("solve-2b", SOLVE, shots=2, shot_set=1, written=True),
    ),
    # No worked justifications exist for HotpotQA, so its setups differ in wording rather than in examples.
    "hotpotqa": (
        Variant("justify", JUSTIFY, written=True),
        Variant("justify-terse", JUSTIFY_TERSE, written=True),
    ),
}


def setup_named(corpus: str, name: str) -> Variant:
    found = [s for s in SETUPS[corpus] if s.name == name]
    if not found:
        raise ValueError(f"{corpus} has no setup {name!r}: {[s.name for s in SETUPS[corpus]]}")
    return found[0]


def needs_train(variant: Variant) -> bool:
    """Examples drawn from the train split; written setups are shown the worked solutions instead."""
    return variant.shots > 0 and not variant.written


def examples_for(corpus: str, variant: Variant, train: list[Row], seed: int) -> tuple:
    """The worked examples a setup is shown: a seeded draw from the train split, or the written ARC solutions."""
    if variant.shots == 0:
        return ()
    if variant.written:
        # two of the three worked solutions, a different pair per set: 0 -> (0, 1), 1 -> (1, 2)
        picked = WORKED[variant.shot_set: variant.shot_set + variant.shots]
        return tuple((None, q, reply) for _, q, reply in picked)
    rng = np.random.default_rng([seed, variant.shot_set, variant.shots])
    picked = rng.choice(len(train), size=variant.shots, replace=False)
    return tuple((train[i].context, train[i].question, (train[i].answers or (NO_ANSWER,))[0])
                 for i in sorted(picked.tolist()))


def setup_summary(answers: list[Answer]) -> dict:
    """What a setup scored on the tuning share: the two automatic judges, refusals, length."""
    return {
        "questions": len(answers),
        "exact_match": float(np.mean([a.exact_match for a in answers])),
        "f1": float(np.mean([a.f1 for a in answers])),
        "judged_right": float(np.mean([a.judge_with_reference > JUDGE_YES for a in answers])),
        "judged_right_without_reference": float(np.mean([a.judge_without_reference > JUDGE_YES for a in answers])),
        "refusals": float(np.mean([is_refusal(a.reply) for a in answers])),
        "stopped": float(np.mean([a.stopped for a in answers])),
        "tokens": float(np.mean([a.tokens for a in answers])),
    }


def agreement(answers: list[Answer]) -> float:
    """The share of questions every setup judges the same way - how little the verdict hangs on the setup."""
    verdicts = defaultdict(set)
    for a in answers:
        verdicts[a.id].add(a.judge_with_reference > JUDGE_YES)
    return float(np.mean([len(v) == 1 for v in verdicts.values()]))
