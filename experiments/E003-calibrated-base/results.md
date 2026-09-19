# Results - the ladder over a calibrated base

Base precision D2 now keeps 76.7% of bf16's knowledge instead of 51.4%: I took bartowski's Q2_K blocks as the base of
every module, as they lie in the file, and laid my refinements up to D8 over them. D4 keeps as much as the ladder on my
own base; D6 and D8 keep 0.7 and 1.3 points less. Measured as the [preregistration](PREREG.md) says: the whole frozen
corpus, the judge of E002.

## Why bartowski's base

While the bench built the model's copy in memory, quantizing bf16 at every load, the base could only be our own
quantization. Now the model is read from its own file, where the base of every module lies as ggml blocks, as in a GGUF
([the model format](../../docs/refocustensors.md)). So any published k-quant file can be the base as it lies, with my
refinements up to D8 over it. bartowski's Q2_K is k-quant in all 315 controlled tensors and calibrated with an
imatrix, so I took it. unsloth's UD-Q2_K_XL keeps more, but 40 of its tensors are IQ with no block step, and no
refinement can be laid over them.

## What every level keeps

| Model | Excellent on known | Kept | Good | Bad | Incoherent | Excellent on unknown |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 (E001) | 91.9% | 100% | 3.4% | 3.7% | 0.9% | 8.1% |
| D8 | 89.6% | 97.5% | 3.4% | 5.6% | 1.4% | 13.8% |
| D6 | 88.4% | 96.2% | 3.5% | 6.5% | 1.5% | 14.2% |
| D4 | 83.5% | 90.8% | 3.3% | 11.0% | 2.1% | 17.3% |
| D2 | 70.4% | 76.7% | 2.9% | 22.6% | 4.0% | 18.7% |

Excellent is Correct and Nearly, good is Partial, bad is Wrong and Related, incoherent is Noise and Garbage. 18,576
known questions, 2,064 unknown.

Beside it, E002's ladder on my base and the published files on the same questions; both ladders are counted by one
query over the verdicts, hence two decimals (E002's card rounds them to 51.4, 91.0, 96.8, 98.8):

| Level | E003, bartowski's base | E002, my base | Published file |
| --- | --- | --- | --- |
| D2 | 76.65% | 51.37% | UD-Q2_K_XL 79.9% |
| D4 | 90.82% | 90.94% | Q4_K_M 95.0% |
| D6 | 96.17% | 96.88% | - |
| D8 | 97.50% | 98.80% | - |

## What I expected and what came out

- **T1**: D2 keeps at least 65% - it keeps 76.7%. It falls 3.3 points short of the 80% I hoped for; UD-Q2_K_XL keeps
  79.9%, but cannot be a base as a whole: 40 of its tensors are IQ with no block step.
- **T2**: D4, D6 and D8 no more than 2 points below E002's ladder - they are 0.1, 0.7 and 1.3 points below. My
  refinements hold over a foreign base, and by the preregistration it can be frozen for the zones.
- **P1**: the calibrated base raises every level - no. D2 rose by 25.3 points, D4 stayed level, D6 and D8 fell below
  E002 by more than half a point.

Incoherent answers at D2 fell to 4.0% from 20.5%: on bartowski's base the model has all but stopped writing noise. D2
rose most on ARC (Challenge 40.6 -> 75.1, Easy 43.8 -> 82.3), HotpotQA (41.1 -> 74.3) and SQuAD (65.2 -> 92.1), less
on TriviaQA (51.4 -> 60.8) and NQ (37.8 -> 47.8).

The loss at D8 sits in the answers from the weights (TriviaQA 102.8 -> 100.6, NQ 81.6 -> 79.9) and on HotpotQA
(99.5 -> 97.6); on SQuAD, where the answer is in the passage, there is none. Why the refinements over a calibrated base
reach D8 less well I have not looked into yet.

On unknown questions the share of excellent answers again grows as the model is coarsened: 13.8% at D8, 18.7% at D2 -
the trend of E001 and E002.

## What it took

The levels answered in 78, 57, 54 and 56 minutes. D2 is slower because it writes longer: 52.8 tokens an answer against
34-36 at the other levels. The judge took 1 hour 49 minutes for all four.

The kernel round - 5% of the corpus, 1,029 questions at D4 of my copy, the level baked against read by blocks through
the tensor-core kernel. Baked took 157 seconds, by blocks 199, 1.27 times longer: the kernel cannot make a uniform level
faster, it already runs on cuBLAS at the speed of bf16; the kernel is for zone layouts. The answer agreed on 988 of
1,029 questions (96.0%), word for word on 928 (90.2%). Short answers agreed in full (TriviaQA 228 of 228, NQ 45 of 45,
SQuAD 338 of 340); long written solutions part (ARC-Challenge 12 of 28): the kernel's mma adds with truncation,
cuBLAS otherwise, and a greedy reply that meets a near tie goes its own way.

## What next

bartowski's base can be frozen for the zones: D2 keeps 76.7% at base precision, D4-D8 are within 1.3 points of my
ladder. The baseline of the zones is then this ladder.
