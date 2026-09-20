---
title: Glossary
---

# Glossary

One notion, one word - in the documents, in the code and in conversation. Here is the definition of each and its properties; if a word is used otherwise, the error is in that text, not here.

## Block

A piece of a module's weights: 64 output rows. On E2B-it there are 14,708 of them. A level of precision is set on a block.

## Group

The attention of one layer or its MLP. On E2B-it there are 70. The unit the oracles work in: they build a map over groups, and a block takes the level of its group.

## Rung

The depth a weight is read to: D2, D4, D6, D8 - two, four, six, eight bits a weight. D2 is the base precision, D8 the top rung.

**Properties.** A rung need not cut the error sixteenfold, as the noise model would have it: measured, D4 leaves 0.53 of the base's error and D6 leaves 0.055. The coefficient is measured from the energies and enters the cost of a group.

## Layout

What the model actually reads on this pass: the level of every block. A uniform layout has every block at one rung; a layout by a map takes its levels from the map.

## Field of importance

A number a group: how much the query needs it. What an oracle returns.

**What an oracle is obliged to output.** A field in 0 ... 1, normalized on its own scale. Nats, Taylor scores and error energy are the oracle's own business and never leave it: otherwise two oracles are incomparable and an intersection of their maps means nothing. One shared rule reads a field into rungs, not one rule each.

## Map

A field cut into levels: one rung a group. One field, one map.

**How it is built.** The cost of a group at a rung is its importance times the share of the error that rung leaves. A group reads the coarsest rung whose cost fits the question's threshold. One threshold a question, searched by the answer.

**Properties.** A map comes out in one of three shapes, and two of them are not zones: the whole network at base precision (the query needs no precision), the whole network at the top rung (no threshold was found), and a map proper. Averages over them are taken apart.

## Oracle

A way of measuring importance while knowing the right answer. Impossible at inference - it costs from tens to thousands of passes over the network - and needed as a ceiling: it shows which layout exists for a query and what it costs.

**How many.** Four:

1. **the sweep by trying** - every group in turn is raised from the base to the top (the lift) or dropped from the top to the base (the drop), and the likelihood of the model's own answer is measured;
2. **the gradient of the answer** - a backward pass: importance is the gradient of the loss by a block's output times the strength of the signal through it; a variant takes the real difference of the weights between the base and the top instead of modelled noise (the quantization gap);
3. **the error energy** - a forward pass: how much a block's output changes when it is read at the base instead of the top; a variant counts from D4 rather than from the base;
4. **the reference** - not an estimate but a measurement: every group is put at every rung with the others at the top, and the coarsest one that holds the answer is taken.

**Properties.** One oracle gives one field and one map; the sweep, the gradient and the energy give two fields each. The pooled field is a fifth map, put together from the four after they are brought to one scale; it measures nothing of its own.

## Antinode

A group where the peaks of different maps of one query coincide. A measure of how far the oracles agree on that query.

## Address of a query

A number a block, taken on a forward pass: what tells one query from another. The hybrid of neuron activity and head energy is read over the first layers at base precision.

**Properties.** It reads a paraphrase as the same query more often than their shared tokens do. It is the forward half of the sensitivity; the backward half a forward pass does not see, and the projection of the first layers carries it.

## Zone

An area of the network a query engages: precision rises there and the rest is read at the base.

**Properties.** The areas are there in the maps that were read - 15-30 peaks falling to the base two or three layers out. Rules of growth with a radius and a profile do not fit them yet; an exact description is being worked out.

## Common map

One map for every query, put together by averaging the fields. The floor for anything that knows the query: a query's map has to beat it, or the gain comes from the shape of the layout rather than from knowing the query.

## Regulator

What decides a layout at inference, without looking at the answer.

**Variants.** A - predict the map in advance, from the address of the query. B - decide as the pass runs: before every layer, look at what enters it and set the levels of its blocks.

## Retention

The share of questions a layout answers right among those the whole network answers right. Counted over the same questions in the same batches.

## Base precision

The rung everything a map does not raise is read at. Today D2 over the published base bartowski Q2_K.

**Properties.** The sensitive classes of modules are never dropped below it: their base is four-bit, or the answers fall apart. The base precision need not be D2; it is a setting of the bench.

## Ladder

The rungs the model can read: D2, D4, D6, D8. A uniform ladder is a run where the whole network is read at one rung; it is the baseline any map is compared against.

## Refinement

An addition of precision over the base: 2 bits a weight that cut the step of the quantization fourfold. A rung is the base plus so many refinements.

## The .refocustensors format

The model's file on disk: the base and the refinements lie in separate planes, so a level is read without repacking. What is held on the card is what a layout read, not the whole file.

## Frozen corpus

The questions everything is measured on. Frozen: its composition, its wrappers and its reference answers do not change, or the numbers of two runs are incomparable.

**Properties.** A known question is one the whole model answers right; retention is counted on those. The unknown ones are kept in a row of their own.

## The draw

The part of the corpus the bench runs layouts over - 5% of the questions. The calibration is another 20%, the projection and the bridge are fitted on it, and it does not enter retention. Questions beyond the share are laid out by a named list, and the previous draw does not move.

## Price of memory

What a layout costs: bits a weight, or a share of what the top rung spends. The second is handier - the uniform D4 is 0.558, the uniform D6 0.775, the base precision 0.342.

## The question's threshold

The number a field is cut into levels by: a group reads the coarsest rung whose cost fits it. One a question.

## Tolerance

How far the answer may move from the whole network's answer while a threshold still counts. Today 0.03 nats or a share of what the whole network itself spends, whichever is larger.

**Properties.** A flat tolerance is unreachable where the whole network is itself unsure, and the oracle then raises everything.

## Budget

A memory set in advance that a layout has to fit into. Unlike a threshold it bounds the spending rather than the quality: under a budget a map cannot degenerate into the whole network.

## Sensitivity

How much the loss of the answer listens to a block's output, times the strength of the signal through it. What the oracles estimate; a block's level follows from it.

## Projection

A ridge regression from the first layers onto the rest: what is visible early predicts the address further on. It is also what returns to the address the half of the sensitivity a forward pass does not see.

## Block graph

What is next to what: edges between blocks by the closeness of the signal. The signal's path is the variant where the weight of an edge comes from the model's weights: a block votes for those it feeds.

## Layer-wise regulator

Variant B: before every layer it looks at what enters it and sets the levels of its blocks. It predicts no map in advance.

**Controls.** The price of memory and the ceiling. The source of the scores is the activity of the blocks or the votes along the graph of connections.

## Experiment

A folder `experiments/E0NN-slug`: the preregistration before the run, the card, the results after. The raw runs are in `runs/`.

## Preregistration

What is measured and why, with the predictions and the criterion, written before the run. Its hypotheses and criteria are not rewritten afterwards.

## Judge

A model that reads an answer and decides whether it is right in meaning. Used where there are tens of thousands of answers; on small samples I read them myself, and my reading outweighs both the judge and the exact match.
