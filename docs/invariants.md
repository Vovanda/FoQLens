---
title: Invariants
---

# Invariants

What holds no matter how the question is asked. This page is the output of the exploration step
([goals.md](goals.md)) and the input to the hypotheses that follow it: a hypothesis worth
preregistering is a claim about something that already looks invariant, and a model of the mechanism
([quantization-filter.md](quantization-filter.md)) is worth keeping only as far as the invariants agree with it.

Started 2026-09-13.

## What counts as an invariant of the phenomenon

A statement about the phenomenon earns a place on this page when all four hold:

1. **It survives two independent conditions.** Two corpora that do not share their questions, or two
   metrics that do not share their failure mode, or two models. One condition is an observation, not
   an invariant - that is exactly the error this page exists to prevent.
2. **It is stated as a number with a bound**, not as a direction. "The mask repeats bit for bit on the
   same batch" is an invariant; "the mask is stable" is a wish.
3. **It names how it was measured**, down to the run, so that it can be attacked.
4. **It names what would break it.** A property nobody can imagine falsifying is not a finding.

Invariants of the *code* live in the docstrings of its modules, each with a test (`CLAUDE.md`,
engineering standards). This page holds the invariants of the bench - how every measurement is made -
and of the phenomenon: the model, the masks, the layouts.

## The bench

- **The judge of every model is the model at its source quality** (bf16): one measure against which
  quantized models are compared with each other and with models that carry a quantization filter on
  different bases. Whatever layout is under test, the same model judges. Tested: the judge's verdicts
  are the same under a D4 layout as at bf16, and garbage as D2 writes it is graded Garbage and never
  accepted (`tests/test_it_gpu.py`). The judge is held to labelled answers at each level it reads
  ([corpus](corpus.md)); a base whose answers it has not been held to is labelled before its verdicts count.

  Why this measure:

  1. **The judge knows every question of the corpus.** The corpus is the questions the source model
     knows, and the judge is that model, shown the reference as well - so it can tell a right answer
     from a wrong one. The exception is the 10% of questions it did not know: there it leans on the
     reference alone, and they are counted apart.
  2. **It is the strongest model on the bench**, the source one at bf16, and the same for every model
     compared.
  3. **A conclusion is decided on the discordant questions.** The difference between two models comes
     only from the questions where their verdicts differ. "One is better than the other" is accepted
     when the confidence interval of the difference excludes zero and Claude's reading of a sample of
     the discordant questions agrees with the judge's direction.

## The phenomenon

*None yet.* Everything observed so far was measured on the first corpus, which the bench no longer
trusts.
