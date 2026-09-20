# E004 - The address of a query: preregistration

Written 2026-09-19, before the runs.

## 1. Why

The regulator decides a layout from the query, so it needs an address - a number a block that tells one query from
another. The address has to be cheap to take (otherwise there is no saving left) and to hold to the meaning of the
query rather than to its wording: otherwise a paraphrase gives another layout and the mechanism is a trap on words.

## 2. Design

- **Model and base**: gemma-4-E2B-it over bartowski Q2_K, the address read at base precision D2.
- **Corpus**: the frozen small corpus, TriviaQA / NQ / SQuAD; the paraphrases of 60 TriviaQA questions are mine, with
  proper names kept.
- **Sources**: neuron activity at the input of `down_proj`, the energy of the attention heads, the two pooled, the
  gradient.
- **Measure**: identification - the share of questions whose nearest address stays their own across a change of wrapper
  (0-shot against 2-shot) and across a paraphrase.
- **Control**: a bag of the query's tokens. A source that identifies a paraphrase no more often than the bag of tokens
  holds to the words.
- **The cheap reading**: a ridge projection of the first N layers onto the address of the rest, fitted on calibration.

## 3. Predictions

- **P1.** There is a source that identifies its query across a change of wrapper more often than a threshold of 0.5.
  Falsified if every source is below it.
- **P2.** There is a source that identifies a paraphrase more often than the bag of tokens. Falsified if none beats the
  control.
- **P3.** The first layers at base precision give the same address as a full pass: the cosine with its own address is
  markedly higher than with the others.
- **P4.** The query itself tells whether it has to be read deeper - that is, the depth can be chosen adaptively.
  Falsified if the area under the curve is about 0.5.

## 4. Criterion

Every measure is read over the same questions. A small sample is named by its number of questions beside the result.

## 5. Run

The masks of the sources are computed once and kept (`runs/masks/`), the checks read them from there; the output is
`runs/address/`.
