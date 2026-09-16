# Results - uniform quantization on the corpus

Measured 2026-09-16. gemma-4-E2B-it answers the frozen corpus, 20,640 questions a level: 18,576 it knows at
bf16 and 2,064 of the unknown share. Answers at `0e08906`, the reasoning judge at `5b54c9e`
([the corpus](../../docs/corpus.md)), raw lines in `runs/E016-uniform-quantization/`. Intervals are 95%, paired
over the same questions, 10,000 bootstrap resamples.

This curve is the baseline every test of the quantization filter is compared with: uniform quantization at
the same memory is what a deployment would otherwise do.

## How much knowledge each level keeps

Retention is the share the judge accepts at a level over the share it accepts at bf16, on the kept questions.

| Corpus | Questions | bf16 accepts | D8 | D6 | D4 | D2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18,576 | 0.919 | 0.988 [0.986, 0.990] | 0.966 [0.962, 0.969] | 0.857 [0.851, 0.862] | 0.000 |
| SQuAD v2 | 6,123 | 0.926 | 0.995 [0.993, 0.998] | 0.988 [0.984, 0.991] | 0.944 [0.936, 0.952] | 0.000 |
| HotpotQA | 5,648 | 0.930 | 0.983 [0.979, 0.988] | 0.957 [0.950, 0.963] | 0.826 [0.815, 0.837] | 0.000 |
| TriviaQA | 4,111 | 0.954 | 0.989 [0.985, 0.992] | 0.961 [0.955, 0.968] | 0.779 [0.766, 0.793] | 0.000 |
| ARC-Easy | 1,362 | 0.862 | 0.996 [0.981, 1.010] | 0.962 [0.941, 0.983] | 0.875 [0.848, 0.902] | 0.000 |
| NQ-open | 812 | 0.765 | 0.984 [0.971, 0.997] | 0.952 [0.931, 0.972] | 0.758 [0.721, 0.795] | 0.003 |
| ARC-Challenge | 520 | 0.846 | 0.930 [0.892, 0.967] | 0.864 [0.820, 0.906] | 0.866 [0.825, 0.906] | 0.000 |

- **D8 and D6 hardly lose.** D8 keeps 98.8%, D6 96.6%.
- **D4 is where knowledge goes, and it goes from the weights first.** With the answer in the passage D4 loses
  5.6%; with the answer only in the weights, the four closed-book corpora pooled, 19.8%; the difference is
  14.3 points [12.9, 15.6]. HotpotQA, two passages and a step, loses 17.4%. The facts go first: TriviaQA and
  NQ-open lose 22-24%.
- **What D4 loses turns into wrong answers, not garbage.** Correct falls from 86.9% to 74.4% of the answers,
  Wrong rises from 1.9% to 13.0%, Garbage stays under 1%. The model still answers - it names something else.
- **D2 is nonsense.** 99.5% of its answers are Garbage: random symbols and repeated fragments. It ends a reply
  on its own in 13.9% of the answers against 99.6-99.9% at the other levels and otherwise runs to the limit of
  512 tokens. It is not a bug of the bench: the level baked into the weights gives the same logits as the one
  read on every call, and an independent round-to-nearest of the same grid gives our weights up to ties at
  the rounding boundary. It is what naive symmetric round-to-nearest does at two bits - four values ±0.25 and
  ±0.75 of the group's maximum, no zero and no search of the step.

The judge's reading was checked against Claude's on the discordant questions, those whose verdict at a level
differs from bf16's, 100 a level: Claude agrees with the judge's direction on 83 at D4, 77 at D6 and 63 at D8.
At D8 a third of the difference from bf16 is the judge grading near-identical answers differently, in both
directions: the loss of D8 does not stand by the criterion of 75% agreement, the losses of D6 and D4 do.

## Where the coarse model answers and the precise one refuses

On the unknown share - questions bf16 got wrong at stage 1 - bf16 was asked again on the present kernels and
judge, so that the levels are compared with a bf16 of the same run (82% of its replies repeat stage 1's).

| Regime | Questions | bf16 | D8 | D6 | D4 |
| --- | --- | --- | --- | --- | --- |
| answer in the passage | 680 | 0.084 | +0.003 [-0.007, +0.013] | +0.004 [-0.007, +0.018] | +0.024 [+0.003, +0.044] |
| answer in the weights | 756 | 0.090 | +0.009 [-0.005, +0.024] | +0.008 [-0.009, +0.025] | +0.053 [+0.029, +0.078] |
| two passages | 628 | 0.134 | +0.016 [-0.008, +0.040] | +0.072 [+0.043, +0.102] | +0.118 [+0.081, +0.153] |
| all | 2,064 | 0.101 | +0.009 [-0.000, +0.019] | +0.026 [+0.015, +0.038] | +0.063 [+0.047, +0.078] |

Share accepted at bf16, and the difference of each level from it.

Coarser levels get right questions bf16 does not - 202 of them at D4. The clearest case is HotpotQA: bf16 says
the passages hold no answer in 12.1% of these questions, D8 in 12.3%, D6 in 9.4%, D4 in 4.6%, and the accepted
answers rise 13.4 → 15.0 → 20.5 → 25.2%. The precise model refuses where it is not sure; the coarse one
answers, and its answer is sometimes right. Coarsening adds no knowledge; it removes the caution. This is the
premise of [H4](../../docs/hypotheses.md): a draft guess is more useful than a refusal, since a guess can be
refined and "I don't know" cannot - and it came up on data not built to show it. A class of tasks that shows
it on purpose is a separate experiment.

## Against the preregistration

| | Prediction | Measured | |
| --- | --- | --- | --- |
| P1 | D8 keeps ≥ 98% on every corpus | ARC-Challenge 0.930 [0.892, 0.967]; the rest ≥ 0.983 | falsified on ARC-Challenge by the judge; the reading does not confirm it at D8 |
| P2 | D6 keeps ≥ 95% over the set | 0.966 [0.962, 0.969] | holds |
| P3 | D4 keeps 70-95% and differs from bf16 | 0.857 [0.851, 0.862] | holds |
| P4 | D2 keeps < 10% | 0.000 | holds |
| P5 | at D4 SQuAD loses less than closed-book | 0.056 against 0.198, difference 0.143 [0.129, 0.156] | holds |
| P6 | no level is accepted more than bf16 on the unknown share | D6 +0.026, D4 +0.063, intervals above zero | falsified |

The 955 kept questions a guess between two answers gets right half the time were not counted apart.
