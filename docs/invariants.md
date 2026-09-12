---
title: Invariants
---

# Invariants of the phenomenon

What holds no matter how the question is asked. This page is the output of the exploration step
([goals.md](goals.md)) and the input to the hypotheses that follow it: a hypothesis worth
preregistering is a claim about something that already looks invariant, and a model of the mechanism
([lens.md](lens.md)) is worth keeping only as far as the invariants agree with it.

Started 2026-09-13, after every earlier verdict of the bench was withdrawn - all of them rested on
one corpus, one metric and one way of reading the mask, so none of them could tell a property of the
phenomenon from a property of the setup.

## What counts as an invariant here

A statement earns a place on this page when all four hold:

1. **It survives two independent conditions.** Two corpora that do not share their questions, or two
   metrics that do not share their failure mode, or two models. One condition is an observation, not
   an invariant - that is exactly the error this page exists to prevent.
2. **It is stated as a number with a bound**, not as a direction. "The mask repeats bit for bit on the
   same batch" is an invariant; "the mask is stable" is a wish.
3. **It names how it was measured**, down to the run, so that it can be attacked.
4. **It names what would break it.** A property nobody can imagine falsifying is not a finding.

Invariants of the *code* are a different thing and stay where they are: each module states its own in
its docstring and has a test for it (`CLAUDE.md`, engineering standards). This page is about the
phenomenon - the model, the masks, the layouts - not about the implementation.

## Confirmed

*Empty.* Nothing has yet been measured under two independent conditions.

## Candidates

Observed once, under conditions the bench no longer trusts. Each has to be re-measured on a corpus and
a metric that can carry a verdict before it moves up; several may simply be artifacts of the old setup.

| Candidate | Observed | Measured on | What would break it |
| --- | --- | --- | --- |
| The mask follows the surface form of the prompt, not only its meaning | reordering the four answer options moved accuracy by 6.3 points and memory by a whole bit, while bf16 and uniform quantization repeated exactly | `runs/reference/address-stability`, 395 questions, 6 orders | the same reordering on a corpus with no options to reorder - if the address is stable there, the effect was the options and not the mask |
| The cost of a layout depends on the text the mask is read from | the same settings spent 5.24 to 6.24 bits depending only on which ordering the mask was read off | same run | a corpus where memory holds while the text varies |
| A coarse floor costs more quality than the zones recover | on the questions the model answered under every ordering, uniform D4 read 0.843 against uniform D6's 0.975 | `runs/reference/shuffle-answers`, core of 121 questions | the same gap measured with an answer-based metric, on a different corpus |
| What a model is competent at varies by more than a factor of twenty between corpora | core from 66.9% (ARC-Easy) to 3.0% (school mathematics) | `runs/reference/corpus` | another model with the same ordering of corpora would strengthen it; a different ordering would break the generality |

## Anti-invariants

Things shown *not* to hold, which are as useful and easier to establish.

| Statement | Why it fails |
| --- | --- |
| Accuracy on a four-option question measures what the model knows | it is right under every ordering of the options on 31% of the old set, and on 3% of its mathematics subject; the rest moves with the arrangement |
| A comparison on one order of the options is a comparison | the original order was the best of six for the lens layout and only for it |
