# E006 - Retention of knowledge by the filter's map: results

Runs of 21.09.2026, a sample of 150: a hundred questions and fifty paraphrases of them.

**The setup the numbers come from.** The base is bartowski Q2_K, the rungs D2 / D4 / D6 / D8, the model
gemma-4-E2B-it. An oracle's field is carried onto demand - a group's cost over its question's threshold,
folded as `d / (1 + d)` into 0...1. A map is that field read by a scale: three bounds, and every group takes
the rung of the interval its value falls in.

**The measure.** Whether a layout's reply is the top rung's reply to the same question, in the same batches,
after one case, one spacing and no punctuation at either end. No judge: the question is not whether the answer
is right but whether the layout answers as the whole network does.

---

## 1. The sample

150 questions: 100 from the frozen corpus - 50 hard, which only the top rung answers, and 50 ordinary of the
same shares by corpus - and 50 paraphrases of them, 25 to each side. The sides were drawn from the judge's
verdicts in E003, where it read every uniform rung over the whole corpus. Frozen before the first run,
`runs/E006-filter-map-retention/sample/`.

Ten questions - 5 hard and 5 ordinary - were set aside for the search for a scale; the other ninety were not
opened while it went on and are counted in a row of their own.

## 2. The measuring stick: the uniform ladder on these questions

| rung | cost of D8's | same reply | hard |
| --- | --- | --- | --- |
| base precision | 0.342 | 26 of 100 | 8 of 50 |
| D4 | 0.558 | 32 of 100 | 3 of 50 |
| D6 | 0.775 | 44 of 100 | 7 of 50 |

The uniform D4 answers worse than base precision. Over a hundred questions six answers apart is at the edge of
what counts as a difference, and nothing is built on it.

## 3. The headline: a map against a uniform rung

| layout | cost of D8's | same reply | hard |
| --- | --- | --- | --- |
| **product 0.35/0.60/0.99** | **0.516** | **58 of 100** | **26 of 50** |
| error_energy 0.50/0.667/0.95 | 0.679 | 59 of 100 | 24 of 50 |
| pooled 0.35/0.60/0.88 | 0.588 | 56 of 100 | 22 of 50 |
| product 0.50/0.80/1.00 | 0.474 | 52 of 100 | 23 of 50 |
| product 0.50/0.95/1.00 | 0.431 | 47 of 100 | 19 of 50 |
| lift_per_weight 0.50/0.667/0.95 | 0.605 | 53 of 66 live | 21 of 29 |
| drop 0.35/0.60/0.99 | 0.569 | 48 of 88 live | 21 of 43 |
| uniform D6 | 0.775 | 44 of 100 | 7 of 50 |

A map takes three times as many hard questions as the uniform D6 and costs a third less. The whole gain sits
on the hard ones: the ordinary questions are taken by bare base precision, and there is nothing to reproduce.

Cheaper is possible and costs quality: the same `product` at 0.50/0.95/1.00 costs 0.431 instead of 0.516 and
loses eleven answers and seven hard ones. Eight and a half points of memory for eleven answers is a bad trade.
The saving worth having is the gap to the uniform rung.

**Degenerate maps are counted apart.** A map that lifted the whole network to the top rung matches it for
free: `lift_per_weight` has 34 of a hundred, `drop` 12, and `product`, `pooled` and the error energy none.

## 4. A map belongs to its question: three spoilings at the same memory

The memory is equal to the byte - 0.6098 against 0.6099.

| layout | same reply as D8 |
| --- | --- |
| the question's own map | 0.86 |
| another question's map of the same cost | 0.59 |
| the same figure carried elsewhere in the network | 0.61 |
| the rungs dealt at random | 0.62 |

Three different spoilings give one answer. The budget, the set of rungs and the shape of the layout are all
kept, so what the gain rests on is which groups got the precision.

## 5. The scale is computed, not searched for

A scale is the three bounds a field is read into rungs at. The search went over three regions on ten
questions: the low one (the lower bound 0.35-0.65), the high one (from 0.70) and the scales with the top rung
switched off (the top bound at 1) - 197 points over five fields.

What the search showed about a scale: the bounds move independently and in both directions; the extremes make
a rung degenerate, and the top bound at 1 switches the top rung off altogether; a section whose field stands
at the ceiling goes up at any scale.

**Nothing the search found beat the scale the formula gives** - `1 / (1 + r)` over the measured error ratios
of the rungs, which on this bench is 0.500 / 0.667 / 0.950. For `error_energy`, `pooled` and
`lift_per_weight` it stands above everything found. The ten questions of the search turned out to be the easy
ones: 80% on them against 56% on the other ninety.

Hence the working rule: the bounds are not tuned, they are computed. The bench changes, the error ratios
change with it, and the formula gives new bounds by itself.

## 6. What a field is made of: the network's part and the question's

The mean of a field over every question - the network's own part - is reproduced on a half of the sample that
was not there when it was taken out (0.77-0.996), and is not reproduced from shuffled data (-0.03...0.18). It
is how the network is built, not an artefact of averaging.

The network's part and the question's add back into the field exactly, a discrepancy of 0.000 - provided the
question's part keeps its sign: 51-59% of the groups stand below the network's part, and cutting them to zero
loses 0.12-0.18 of the field.

The oracles see different network parts: `drop` asks for 8 groups of 70 at 0.301 of the memory, the error
energy for 53 at 0.663. The only group common to all is `33.mlp`.

What looked like agreement between the oracles was agreement about the network's part: the error energy with
`pooled` stands at 0.61 on the raw fields and 0.17 on the question's parts.

## 7. Standing up to a rewording

A question against its paraphrase, by three measures: the share of groups at the same rung, the Jaccard over
the groups either raises, the memory the two differ by.

| map | rungs | shape | apart, bits |
| --- | --- | --- | --- |
| lift | 67% | 53% | 1.61 |
| drop | 66% | 50% | 1.42 |
| drop without the network's part | 81% | 46% | 0.69 |
| **mean without the network's part** | **97%** | **88%** | **0.10** |

The order is the same by all three measures: taking the network's part out removes noise and keeps the signal.
The steadiest map costs 0.250 of the top rung's memory; whether it answers has not been run.

## 8. The paraphrases: a uniform rung is stronger than the maps

On the 50 paraphrases the uniform D6 takes 33 of 50 and 16 hard of 25 at 0.775; the best map is the error
energy, 27 of 50 and 11 hard at 0.674. On the hundred originals it was the other way round: the maps 58
against 44.

The explanation at hand was the allowance. It is `max(0.03 nats, 0.2 of the whole model's own loss)`, and on a
paraphrase the model is less certain: its loss 0.25 against 0.11, higher on 74% of the pairs. The threshold
follows the allowance, the demand falls fourfold, and the map thins out.

Checked by recounting: the fields of the paraphrases were computed again at the allowance of their originals.
The error energy gained two hard questions and got dearer, 0.674 against 0.587; for `lift_per_weight` only the
count of degenerate maps changed, 10 against 7. The uniform D6 stayed ahead.

So the allowance is not the reason: a paraphrase is easier for a uniform rung - it takes 66% of them against
44% of the originals - and a map has nothing to win back. The allowance is still set badly, and that is a
piece of work of its own.

## 9. What this experiment does not show

- The oracles see the right answer. This is a ceiling: it says the layout exists and what it costs, and says
  nothing about how to find it at inference.
- Of the fields tested, only the error energy is computed ahead, without an answer - and it is the dearest:
  0.679 against 0.516 for an overlay.
- Whether a map can be predicted from the address was not tested here; in E005 the bridge from the address was
  not distinguishable from a constant map.

## Verdict

An ideal map takes 26 hard questions of 50 at 0.516 of the top rung's memory; the uniform D6 at 0.775 takes 7.
The bounds of the scale are computed from the ladder's own measurement, not searched for. Three controls at
memory equal to the byte show that the address is what decides.

The hypotheses this bears on, one by one, are in `hypotheses.ru.md`; H0, H1 and H2 have no answer here, and
what to draw for them is written in `topics-recipe.ru.md`.
