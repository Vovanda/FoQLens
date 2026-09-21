---
title: "E006 - Retention of knowledge by the filter's map"
date: 2026-09-21
weight: 21
hypotheses: [H3, H3.1, H3.3, H3.4, H1.1, H0]
statuses: [done]
params:
  fixed: ""
  run: "2026-09-21 - a sample of 150, the fields of five oracles, 197 points of the scale on ten questions, five scales over all 150"
  results: "2026-09-21 - 150 questions, five fields, five scales, the paraphrases recounted at the original's allowance"
  verdict: "An ideal map takes 26 hard questions of 50 at 0.516 of the top rung's memory; the flat D6 at 0.775 takes 7. The bounds of the scale are computed from the ladder's own measurement, not searched for. Three controls at the same memory to the byte show that the address is what decides"
---

# E006 - Retention of knowledge by the filter's map

I wanted to know whether the map of the quantization filter keeps the model's knowledge: does a layout built
for one question answer as the whole network does. The saving follows from how a map is built - everything
the question does not need is read coarsely. I look at the hard questions, where a uniform rung has already
lost the knowledge, apart from the ordinary ones, where a map has to be cheap.

The documents of the experiment: [results](results.ru.md), [what came out](findings.ru.md),
[the fields](fields.ru.md), [the field scale and the search for its optima](bands.ru.md),
[what this says about the hypotheses](hypotheses.ru.md), [the journal](journal.ru.md).

## What this experiment changes against E005

In E005 a map was built by searching for a threshold by the answer, every oracle in its own units, and the
oracles could not be compared with one another. Here every oracle's field is brought to one quantity - the
share of the answer's allowance a coarse reading of a section eats - and read into rungs by one rule. That is
what made it possible to ask about the scale of the reading itself, and not only about the map.

The sample is different too: 50 hard and 50 ordinary questions with 50 paraphrases, instead of 103 hard ones.
The ordinary half is there to catch a layout that lifts everything: on the hard questions it looks excellent
and costs as much as the top rung.

## Where I stand on 21.09.2026

The ceiling is there: a map built for a question takes three times as many hard questions as the flat D6 and
costs a third less. It belongs to that question - three different ways of spoiling it at the same memory to
the byte give 0.59-0.62 against 0.86 for its own map. The field splits into the network's own part and the
question's part, and they add back without loss. The scale needs no search: the one derived from the ladder
beats what a search over 197 points found.

Next comes the mechanism. Today every field but the error energy is computed from the right answer, and the
error energy is the dearest of them: 0.679 of the memory against 0.516 for an overlay. The next step is to
choose the field to go on with and to look for a rule that computes it ahead of the answer.
