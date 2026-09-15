# E016 - Uniform quantization on the corpus: preregistration

Written 2026-09-15, before any run.

## 1. Why

How the model's knowledge degrades under uniform quantization: which share of what gemma-4-E2B-it
knows at bf16 each level D8, D6, D4 and D2 keeps, in which regime of the corpus knowledge goes first,
and at which step the model breaks down. No zones and no quantization filter take part: every block is
read at the same depth. The step at which coarsening slides into nonsense is the question
[H6](../../docs/hypotheses.md) asks of uniform coarsening.

## 2. Design

- **Quantization**: one stored copy read at 2 / 4 / 6 / 8 bits (residual slices, `foqlens.quant`) at
  commit `edc81f2`; every block at the same level, set with `set_all`.
- **Corpus**: the frozen files `corpus/e2b-it/*.json` at commit `edc81f2`. Asked are the kept questions,
  18,576, and the unknown share, 2,064 - 20,640 per level. The questions spent on choosing the prompt
  are never asked. Each corpus in the setup frozen by the prompt tuning.
- **Levels**: D8, D6, D4, D2, the same questions in the same batches (batches are built from the
  prompts, which do not depend on the level).
- **bf16 is not generated again.** The kept questions are known at bf16 by construction, and the static
  decoder is nearly invariant to the composition of a batch (the same verdict on 0.990 of ARC's 916
  questions, 2026-09-15). The stage 1 answers at bf16 were judged before the present judge (`41c12ab`),
  so the present judge reads them again - a judging pass only, no generation. That gives its ceiling:
  the share of right bf16 answers it accepts.
- **Regimes**, counted apart: the answer in the passage (SQuAD v2); knowledge in the weights
  (TriviaQA, NQ-open, ARC-Challenge, ARC-Easy); two sources and a step (HotpotQA). The 955 kept
  questions a guess between two answers gets right half the time are counted apart from the main share.
  The unknown share is counted apart.
- **Judges**: the model at bf16 grading an exam (`foqlens.judging`) decides the verdict and gives the
  kind of answer, Correct / Nearly / Partial / Related / Wrong, written into every answer line; exact
  match and F1 are written beside it; Claude reads a sample of the discordant questions.

## 3. Predictions

The retention of a level is the share of kept questions the judge accepts at that level, divided by
the share it accepts at bf16.

- **P1.** D8 keeps at least 98% on every corpus. Falsified if the upper bound of the interval of any
  corpus's retention is below 0.98.
- **P2.** D6 keeps at least 95% over the whole set. Falsified if the upper bound is below 0.95.
- **P3.** D4 loses for real: its retention over the whole set lies between 70% and 95%, and the
  interval of its difference from bf16 excludes zero. Falsified if the retention falls outside.
- **P4.** D2 breaks down: its retention over the whole set is below 10% - two bits without calibration
  destroy the model. Falsified if the lower bound is 0.10 or more.
- **P5.** At D4 knowledge in the weights goes first: SQuAD v2 loses less than the four closed-book
  corpora pooled, and the interval of the difference of the losses excludes zero. Falsified if SQuAD
  loses as much or more.
- **P6.** On the unknown share no level is accepted more often than bf16 beyond the interval. Every
  question a level answers there and bf16 does not is read by Claude.

The kinds of answer are described, not predicted: how Correct, Nearly, Partial and Wrong spread over
the levels.

## 4. Criterion

Paired over the same questions: the difference of acceptance between a level and bf16, per corpus and
over the set, with a 95% interval from 10,000 bootstrap resamples of the questions. A conclusion stands
when its interval excludes zero and Claude's reading of a sample of the discordant questions - up to
100 per level, spread over the corpora by their size - agrees with the judge's direction on at least
75% of them.

## 5. Run

UPD 2026-09-15: a level is baked into the weights - read once to its depth, the same weight the
unpacking reads on every call - and so cannot judge itself; its answers are judged at bf16 by a second
process. The smoke round at D4 unpacking on every call took ~20 minutes against ~4.5 at bf16 and ran
out of memory on HotpotQA.

```
uv run python scripts/rejudge_answers.py --answers runs/reference/stage1/e2b-it/answers --level bf16 \
    --frozen corpus/e2b-it --out runs/E016-uniform-quantization
uv run python scripts/stage1_answers.py --frozen corpus/e2b-it --level d8 --out runs/E016-uniform-quantization
uv run python scripts/rejudge_answers.py --answers runs/E016-uniform-quantization/e2b-it/unjudged --level d8 \
    --frozen corpus/e2b-it --out runs/E016-uniform-quantization
(the same for d6, d4, d2)
```

Before the run, one round at D4 names the time per level; at the speed of stage 1 at bf16 a level
would take about an hour. Blind analysis: all levels first, only a crash check while they go.
