---
title: The corpus
---

# The corpus: what the bench measures on, and why the first one did not work

This page is the source of truth for how questions get into the bench. It exists because the first
corpus was chosen by eye and could not measure a layout, which invalidated every quality verdict the
bench had produced. Written 2026-09-13, after that was found and before the new corpus is fixed.

## What went wrong

The bench asked multiple-choice questions - four options, scored by which of " A" … " D" the model
ranks highest - drawn from four MMLU subjects: high-school biology, mathematics, prehistory and
high-school geography. Three checks, run for unrelated reasons, all landed on the corpus rather than
on the layouts.

**The model does not know most of the set.** Ask every question in six different orders of its
options and count the questions answered correctly in *every* order - the ones where the answer comes
from knowledge rather than from the arrangement. That is **121 of 395, or 31%**. On 69 questions the
model is never right. On the mathematics subject the core is **3 of 100**.

**Where the core is small, the reason is calculation.** 91 of those 100 mathematics questions ask for
a computed value - factor a quadratic, integrate, count committees of five - and the format allows
exactly one token to answer in, with no room to work. That is not a model failing to know
mathematics; it is a metric that cannot see mathematics. The same tell appears in `global_facts`
(69% of its questions have all-numeric options, core 3.4%), and in half of high-school chemistry and
physics.

**So the comparisons were measuring the guessing.** Split the 395 questions by the core and compare a
zone layout against uniform quantization of the same cost: inside the core the zones are behind by
5.1 points of accuracy, outside it they are ahead by 1.0, and over the whole set the two cancel into
a difference that looks positive and clears no interval. The headline result of E013 - zones ahead
of uniform D6 at the same memory - came from the outside, and did not survive reordering the options.

**And the letters themselves carry a lean.** Over six orders, the full model picks C 29% of the time
and a zone layout 35%. On a set where half the answers are guesses, a layout that shifts the lean
looks like a layout that answers better.

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
them; the full model, asked whether the answer agrees with the reference; and Claude, who reads every
answer the first two do not settle. Where they disagree, Claude's verdict decides. No options to lean
on, nothing to reorder, a continuous scale. Code: `src/foqlens/extractive.py`.

**The corpus is selected, then frozen.** The full model first answers every question of the full
datasets, and a question stays if the answer is right. The corpus is then a file of question numbers -
kept and excluded, each with its reason - with the dataset revisions pinned. Every later run reads that
file; how it was assembled is recorded, not re-derived.

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
| the answer is only in the weights | TriviaQA, NQ-open, ARC-Challenge without its options | facts and school science; for ARC the reference is the text of the right option, and the questions that point at their options ("which of the following") are left out - 206 of 1165 | decide - this is the regime the idea is about |
| two passages and a step between them | HotpotQA | the answer is assembled from two paragraphs among eight that do not hold it | matter, and matter less than in the second |

ARC-Challenge and ReClor with a letter to pick stay only in the calibration table below, as
measurements: ARC's core is 41.5% and ReClor's 14.8%, and ReClor looks for its answer in an attached
passage - reasoning over a text, not knowledge held in the weights.

If the zones help as much when the answer is sitting in the context as when it is only in the weights,
the mechanism is not doing what it is claimed to do. That is the test the old corpus could not run.

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

## What this does not fix

A better corpus makes a verdict possible; it does not make one favourable. Two things are known to be
wrong independently of it, both measured on the current mask with the letter-choice metric the reset withdrew, which leaves them as directions:

- **The address moves with the shape of the prompt.** Reordering the options - same question, same
  meaning - changes accuracy by 6.3 points and memory by a whole bit, while bf16 and uniform
  quantization repeat exactly. Reading the mask from other texts was tried three ways and all were
  worse. Until the address is stable, no corpus can give a clean answer.
- **The floor costs quality on its own.** On the core, uniform D4 - the floor every layout so far has
  used - reads 0.843 where uniform D6 reads 0.975. The floor stays D4, since raising it removes the
  saving the mechanism exists for; what has to change is D4 itself, or a thin static skeleton of
  blocks held above it.
