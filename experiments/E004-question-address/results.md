# E004 - The address of a query: results

Runs of 2026-09-19. The configuration the final numbers were read at: gemma-4-E2B-it over bartowski Q2_K, the small
corpus, TriviaQA / NQ / SQuAD, a ridge of 0.1.

## The sources of an address

Identification is the share of questions whose nearest address in another wrapper is still their own; the paraphrases
are mine, with proper names kept, and the control is a bag of the query's tokens.

| source | across wrappers | across a paraphrase | fit |
|---|---|---|---|
| neuron activity | 0.995-1.000 | 0.892 | yes |
| head energy | 0.964-0.993 | 0.692 | weaker |
| the two pooled | 0.820-0.886 | 0.308 | no, it holds to the wording |
| gradient | 0.473-0.490 | - | no |

Chance is 0.003-0.02 and the bag of tokens 0.717.

Neuron activity carries meaning over and above the words: it identifies a paraphrase more often than their shared
tokens do (0.892 against 0.717). Head energy works at the level of words, and `pooled` below them.

## The hybrid

The hybrid of activity and head energy, each part normalized to its own length, both read in one pass:

| check | hybrid | activity | heads |
|---|---|---|---|
| across wrappers | 0.997-1.000 | 0.995-1.000 | 0.964-0.993 |
| across a paraphrase | **0.917** | 0.892 | 0.692 |
| the deep address, N = 6-8 | 0.877-0.893 | 0.816-0.855 | 0.357-0.629 |

The hybrid is the best address of those checked. By geometry it is nearly the activity (agreement 0.96); the heads add
2-8 points on the deep address.

## The cheap reading

The first 12 layers at base precision give the same address as a full pass at bf16: the cosine with its own address is
0.97 for the activity and 0.87-0.93 for the heads, and about zero with the others.

A ridge projection of the first N layers predicts the address of the rest: at N = 4-8 identification is 0.82-0.89, and
84-92% of the weights are left under the zones. Reading a window of layers instead of the first N gained nothing. The
working point is the hybrid at N = 6.

## The depth cannot be chosen from the query

Hard questions are not read deeper than easy ones: the area under the shift's curve is 0.44, and even the oracle's
margin, which takes the true deep address, gives 0.53-0.70. The silhouette rule - read until the top blocks of the
address hold over two depths - is no better by identification than a fixed depth of the same average cost.

Probes over thirty trivial and twenty-five hard questions: the average stop is 6.70 against 6.48, so the hard ones stop
even earlier.

## What it means for the mechanism

The address of a query is read cheaply and identifies the meaning rather than the wording - the left half of the
regulator is there. What the address does not give: the depth of the reading cannot be chosen from it, and a precision
map could not be predicted from it - in E005 a bridge from the address is not distinguishable from a constant map by
the answers.
