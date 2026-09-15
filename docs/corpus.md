---
title: The corpus
---

# The corpus

How questions get into the bench; the source of truth for it.

## The first corpus

The first corpus chosen - four MMLU subjects with a letter to pick - was rejected. Reordering the options and asking the same
question without them showed that the model did not know most of the questions, and the letter hid it.
The requirements below are what that taught.

## What a corpus has to satisfy

1. **A large core.** The share of questions the model answers correctly under every ordering of the
   options. Below roughly half, most of what is measured is arrangement, not knowledge. Measured by
   `scripts/corpus_calibration.py`.
2. **Answers that are recalled, not computed** - unless computation is what is being tested, and then
   the format has to give room to compute. The crude tell, reported per subject, is the share of
   questions whose options are all numbers or formulas.
3. **Nothing for the model to lean on.** Either no options at all, or a score that is invariant to
   their order. The order of the options must not be able to move the result, and this is checked by
   running every question in several orders, not assumed.
4. **A scale finer than four letters.** With four options a single question moves accuracy by 0.25
   points on 395 questions, and everything below two points needs a bootstrap to see at all.
5. **Topics that can be paired**, since the bench compares a query's own zones against another
   topic's - which requires at least two topics far enough apart to have different zones.

## What is being built

**The model writes its own answer, in every regime.** It is given the question - with a passage or
without one - and writes the answer itself; it never sees options. What it writes is compared with the
reference answer of the dataset, and three judges read it: exact match and token F1 as SQuAD scores
them; the full model, grading the answer as an exam against the reference; and Claude, who reads every
answer the first two do not settle. Where they disagree, Claude's verdict decides. No options to lean
on, nothing to reorder, a continuous scale. Code: `src/foqlens/extractive.py`.

**The corpus is selected, then frozen.** The full model first answers every question of the full
datasets, and a question stays if the answer is right. The corpus is then a file of question numbers -
kept and excluded, each with its reason - with the dataset revisions pinned. Every later run reads that
file; how it was assembled is recorded, not re-derived. The set grows by new files: a new dataset is
selected the same way and frozen in a file of its own, and a frozen file is never changed, so a run stays
comparable with the runs made before it.

**Questions the model did not know go back in, marked.** The set the coarsened models answer is 90%
questions the full model knew and 10% questions it got wrong, drawn at random. The 90% is where the
bench is confident of the model's knowledge; the 10% is there to check for emergent properties - in
case a model answers some of them after all. That is unlikely, and they are counted apart.

**Three regimes, so that the claim can fail.** The bench's claim is that precision should follow the
query because the knowledge a query needs sits in particular weights. That claim predicts different
things in different regimes, and until now every corpus was of one kind:

| Regime | Corpus | Why this one | What the zones should do |
| --- | --- | --- | --- |
| the answer is in the passage | SQuAD v2 | the standard reading set; half its questions have no answer in the passage, and saying so is part of reading | little - the knowledge came in with the prompt |
| the answer is only in the weights | TriviaQA, NQ-open, ARC-Challenge and ARC-Easy without their options | facts and school science; for ARC the reference is the text of the right option, and the questions that point at their options ("which of the following") are left out - 206 of 1165 in ARC-Challenge, 360 of 2376 in ARC-Easy | decide - this is the regime the idea is about |
| two passages and a step between them | HotpotQA | the answer is assembled from two paragraphs among eight that do not hold it | matter, and matter less than in the second |

ARC-Challenge and ReClor with a letter to pick stay only in the calibration table below, as
measurements: ARC's core is 41.5% and ReClor's 14.8%, and ReClor looks for its answer in an attached
passage - reasoning over a text, not knowledge held in the weights.

If the zones help as much when the answer is sitting in the context as when it is only in the weights,
the mechanism is not doing what it is claimed to do. That is the test the old corpus could not run.

## Selection on E2B-it: what came up, what was done, where it stands

**The model.** The corpus is selected on gemma-4-E2B-it, not on the base checkpoint. The base does not
follow requests - it writes no ARC solution and no HotpotQA justification - and as a judge without the
reference it could not tell a right answer from a wrong one. The -it checkpoint is pinned by revision;
the base stays for what needs no instructions.

**Stage 1.** The model answered every question of the six datasets but the 1% spent on choosing the
prompt: 35,387 answers. Each decoding step is one CUDA graph, 5-6 times faster than the loop; the first
five datasets took 92 minutes, ARC-Easy, added on 2026-09-15 in a run of its own, three and a half.

**Three judges.** Exact match and F1 against the reference; the model itself, with a Yes/No verdict
against the reference; Claude, reading the answers. Claude's verdict decides. Claude read in four turns,
from the least settled to the most:

1. where exact match and the judge disagree - 5,961 answers;
2. where both say no, but not surely - 2,214;
3. where both surely say no - SQuAD, both ARCs and HotpotQA in full, TriviaQA and NQ-open on a sample,
   where the model knows under 2% and the rest stays unread;
4. where both say yes - 12,465.

23,196 answers read in all.

**What came up.**

- *The judge answers the question itself.* Asked whether an answer "means the same as the reference",
  the model heard "is it right, as far as I know" and said No to 4% of exact matches ("Captain Flint"
  against "Captain Flint"). The effect is deterministic - a hundred repeats give one number: it is the
  prompt, not noise.
- *Matching the reference is not being right.* 31 of the 37 wrong answers that both exact match and
  the judge accepted repeat the words of the question: asked who is better known as Barbara
  McCorquodale, the model answers "Barbara McCorquodale", a name among the reference's aliases.
- *The full model gives nothing to test a scale on.* Its answers are whole or wrong; the middle of a
  scale shows only on degraded or coarsened answers.
- *Shorter is not better.* The same ladder compressed into a CSV table made the judge a yes-sayer: it
  accepted 22 of 60 wrong answers.

**What was done.**

- One judge on every answer, with no exception for exact matches.
- The judge grades an exam: the correct answers are known, so it does not answer the question but
  checks the examinee's answer against them. It sees the question as context.
- It replies "Yes, grade" on the ladder Correct, Nearly, Partial, Related, Wrong, and an answer counts
  from Nearly. The ladder is a working choice: better ones likely exist, and there was no time to try
  them all. The Yes/No is the verdict; the middle grades are a reserve for when the judge uses them, on
  the incomplete answers of coarsened models. The words were picked for meaning: "less precise" let a
  wrong neighbour into the accepted grades, and "Hint" was never chosen at all.

**Where it stands.**

- *The corpus is frozen* (`corpus/e2b-it/`). The model knows 18,576 questions - TriviaQA 4,111,
  NQ-open 812, SQuAD v2 6,123 (552 of them the right answer that the passage holds none),
  ARC-Challenge 520, ARC-Easy 1,362, HotpotQA 5,648 - and 2,064 questions it did not know are marked,
  a tenth of the stage 2 set.
- *The new judge against Claude's verdicts*, on the 21,242 answers read before ARC-Easy: agreement
  0.888 → 0.930; false noes 1,672 → 501, false yeses 711 → 980. On ARC-Challenge it is worse than the
  old one, and on ARC-Easy it is not measured yet; both are open.
- *A synthetic check* (`runs/reference/judge-synthetic/e2b-it/`): 250 right answers and three
  degradations of each, graded by Claude beforehand. Accepted: right 250 of 250, incomplete 72%,
  partial 24%, wrong but plausible 4%, plainly wrong 0 of 250. The middle grades are named rarely and
  loosely - "Related" hardly ever, plausible wrong answers are called Wrong: the verdict is right, the
  grade is not, and the ladder works as a reserve.

## Candidates, measured

Core measured with `scripts/corpus_calibration.py` on Gemma 4 E2B at bf16, six orders of the options,
full subjects. Raw numbers in `runs/reference/corpus/`.

| Corpus | Core | Accuracy | All-numeric options |
| --- | --- | --- | --- |
| ARC-Easy | 66.9% | 82.4% | 0.9% |
| marketing | 64.6% | 81.6% | 0% |
| world_religions | 62.6% | 78.1% | 6.1% |
| high_school_psychology | 57.3% | 74.5% | 2.1% |
| logical_fallacies | 56.8% | 72.7% | 0% |
| miscellaneous | 56.7% | 73.5% | 5.6% |
| sociology | 44.9% | 69.7% | 0% |
| ARC-Challenge | 41.5% | 62.5% | 4.5% |
| *high_school_geography - in the old set* | *44.0%* | *65.3%* | *0%* |
| *high_school_biology - in the old set* | *40.0%* | *63.9%* | *2.1%* |
| *prehistory - in the old set* | *36.0%* | *58.3%* | *3.0%* |
| OpenBookQA | 34.0% | 59.5% | 0.8% |
| global_facts | 3.4% | 27.1% | 69.3% |
| *high_school_mathematics - in the old set* | *3.0%* | *31.8%* | *91.0%* |

The old set occupied the bottom half of this table, and one of its four subjects was unmeasurable.

## Open

Whether the mask follows the form of the text. On the first corpus reordering the options - same
question, same meaning - moved a zone layout's result, while bf16 and uniform quantization repeated
exactly. The new corpus has no options to reorder; the question is measured again on it.
