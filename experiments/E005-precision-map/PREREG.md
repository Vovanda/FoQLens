# E005 - The precision map of a query: preregistration

Written 2026-09-19, before the oracles were run.

## 1. Why

The regulator is to decide, from the query, which blocks to read finer so that the answer gains at a given memory.
Before a cheap mechanism is built, it must be known whether such a layout exists at all and what it costs. An oracle
builds one expensively, by looking at the model's own answer: it cannot run at inference, but it gives the ceiling. If
even an oracle's map is no better than a uniform rung of the same memory, there is nothing to build a mechanism out of.

## 2. Design

- **Mechanism**: [docs/precision-regulator.md](../../docs/precision-regulator.md), the precision field - the value of
  the field is the level of a group.
- **Model and base**: gemma-4-E2B-it over bartowski Q2_K, the rungs D2 (base precision), D4, D6, D8.
- **Corpus**: the frozen small corpus; 5% of the questions laid out, 20% for calibration.
- **Oracles**: the sweep by trying (every group lifted from the base and dropped from the top), the gradient of the
  model's own answer and the quantization gap, the error energy of D2 and D4, their pooled field, and the reference -
  a direct measurement rung by rung.
- **Compared**: the uniform ladder on the same questions in the same batches - D2, D4, D6, D8. The shape control is the
  common map, averaged over the fields: it knows no question, and a query's map has to beat it.
- **Judges**: the exact match against the corpus's answer and my own reading; where they disagree my reading decides.

## 3. Predictions

- **P1.** A query's map at no more memory than the uniform D4 holds more right answers than the uniform D4. Falsified
  if the uniform D4 is no worse.
- **P2.** A query's map beats the common map of the same price - that is, the gain comes from knowing the question and
  not from the shape of the layout. Falsified if the gap is within the error.
- **P3.** Different oracles give similar maps: their peaks agree more often than chance. Falsified if the agreement is
  at chance.

## 4. Criterion

The shares are read over the same questions in the same batches. Degenerate maps - the whole network at base precision
or the whole network at the top rung - are counted in a row of their own: they hold the answer for reasons that have
nothing to do with the oracle.

## 5. Run

The sweep by trying, the masks of the error energy and the gradient, the precision fields, the answers at the maps. The
output is `runs/oracles/`, and for the hard sample `runs/oracles/hard-e2b-it/` and `runs/masks-hard/`.
