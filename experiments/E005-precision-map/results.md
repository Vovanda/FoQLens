# E005 - The precision map of a query: results

Runs of 2026-09-19..20.

**The configuration the final numbers were read at.** The base bartowski Q2_K, the rungs D2 / D4 / D6 / D8, the small
corpus. A map is built like this: the field of importance is taken in its own units, the cost of a group at a rung is
its importance times the share of the base's error that rung leaves (measured: D4 leaves 0.53, D6 leaves 0.055), and a
group reads the coarsest rung whose cost fits the question's threshold. There is one threshold a question, searched by
the answer by bisection; the tolerance is 0.03 nats or 0.2 of what the top rung itself spends, whichever is larger.

**What these numbers do not hold.** The scale of the rungs with the bounds 0.8 / 0.9 / 1.0, set on 20.09, was not used
here: it works in another build of the maps, where a field is first laid on 0 ... 1. No run was made with it, and the
cost of a map under it is unknown.

## The sample

103 questions only the top rung answers: it is right by the judge and by the exact match, and both uniform D6 and D4
fail both checks. They were picked out of E003 over the whole corpus and laid out on top of the share; HotpotQA is left
out - its prompts are three times longer and the oracle spends 55 seconds a question on them.

On such questions the difference shows without averaging: the uniform rungs miss them by construction of the sample,
and the only question is whether a map gets them.

## The number

The share of right answers and the memory against the whole network, 103 questions:

| layout | right | bytes |
|---|---|---|
| the whole network at D8 | 0.961 | 1.000 |
| the lift oracle's map | 0.864 | 0.574 |
| the drop oracle's map | 0.845 | 0.491 |
| the reference (the direct sweep) | 0.835 | 0.620 |
| **the pooled field's map** | **0.816** | **0.523** |
| the error energy's map | 0.806 | 0.717 |
| the gradient's map | 0.796 | 0.729 |
| the common map (knows no question) | 0.544 | 0.513 |
| uniform D2 | 0.282 | 0.342 |
| uniform D6 | 0.039 | 0.775 |
| uniform D4 | 0.000 | 0.558 |

The pooled field's map holds 0.816 at 0.523 bytes. The uniform D4 at a comparable 0.558 holds nothing, and the uniform
D6 at one and a half times the memory holds 0.039.

The common map, averaged over the fields and knowing no question, holds 0.544 at the same price. The gap from 0.544 to
0.816 is what knowing the question is worth, over and above the shape of the layout.

## What the answers themselves look like

A uniform rung neither falls silent nor writes nonsense - it names the neighbouring fact. Read in a row: Mars for
Mercury, Austria for Hungary, Henry VIII for Edward III, Santiago for Wellington, Liszt for Schumann, rum for absinthe,
birds for bats, 1765 for 1757, Jaws for The Godfather Part II. The knowledge slides onto something alike.

The map answers what was asked. Of forty read by hand it was wrong twice, both times on ARC, where no answer could be
read out of the reasoning. In three places it answered where the whole network itself was wrong.

## The shapes of a map

A map is of one of three shapes, and two of them are not zones at all: the whole network at base precision (the
question holds without precision anywhere) and the whole network at the top rung (no threshold was found and everything
was raised). Both answer as the whole network does for reasons that have nothing to do with the oracle, and counting
them in one average is what made the earlier numbers flatter every oracle.

On the hard sample, after the tolerance was raised: the gradient, the error energy, the quantization gap and the pooled
field have no degenerate maps at all, the drop oracle has 7 of 103, the lift 29, the reference 16. Between 13 and 18
questions of 103 hold at base precision for most oracles and only one for the reference - the sample does ask for
precision.

## How a map is built

- The rungs cut the error unevenly: D4 leaves 0.53 of the base rung's error, D6 leaves 0.055. The noise model would
  give 0.063 and 0.004.
- A map is 15-30 narrow peaks over the middle of the network, falling to the base two or three layers out.
- The spread of the levels: 10% common to every question, 32% the corpus, 37% the question's own offset, 21% its
  pattern.
- Antinodes: three maps of one question agree on 3.0 groups of 70 on average against 0.84 by chance, and they stand in
  the same MLP groups - a property of the network rather than of the question.
- The maps of two questions are never the same: no identical pair among 85, the levels agree on 0.54 of the groups on
  average, and a group is raised by between 6% and 91% of the questions.

## The settings were set by hand

The tolerance (0.03 nats plus a share of 0.2) was set by hand, without a sweep; the rungs' coefficients are measured
from the energies. The effect holds at these values, so it is not an artefact of fitting.

What is not measured yet: the cost of a map under the scale with the bounds 0.8 / 0.9 / 1.0, which makes the upper
rungs dearer and should give coarser and cheaper maps. That is a run of its own.

## What it means for the mechanism

The target is proved and measured: a layout built for the question exists, costs about half the memory of the whole
network, and holds knowledge where a uniform rung loses it. The oracles build it by looking at the answer and are
impossible at inference - this is a ceiling, not a mechanism.

There is no cheap predictor of such a map yet. A bridge from the address is not distinguishable from a constant map by
the answers. The layer-wise regulator (variant B) predicts no map in advance - it decides the layout as the pass runs;
on the `activity` source it goes level with the uniform D4, and the votes along the signal's path are not wired in as a
source yet.

## What comes next: E006

The oracles are brought to one shape - every one of them returns a field in 0 ... 1 on its own scale, and one shared
rule reads a field into rungs. Their maps then become comparable directly, which they are not today: the units are
each their own, and the bringing together is done only inside the pooled field.

On comparable maps the intersections are counted: where the maps of different oracles raise the same groups, where
they part, and what a layout built on their agreement gives. Antinodes have so far been counted only as a statistic of
agreeing peaks; no answers were ever run on a layout built from them.
