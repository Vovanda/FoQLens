# D2 answers cut short, asked again at 1,024 tokens

2026-09-17, gemma-4-E2B-it, the k-quant D2 floor (k, v, o, down and the per-layer modules on a Q4_K base, the rest
on Q2_K). An exploration outside the ladder's table, where every level answered with a written reply capped at 512
tokens. Here the known questions whose D2 answer ran into that cap are asked again with a cap of 1,024. The judge is
the same, bf16.

| Corpus | Cut at 512 | Now end | Excellent at 512 | Excellent at 1,024 | of them longer than 512 |
| --- | --- | --- | --- | --- | --- |
| ARC-Challenge | 164 | 120 | 5 | 51 | 40 |
| ARC-Easy | 12 | 3 | 0 | 2 | 0 |
| HotpotQA | 135 | 60 | 0 | 22 | 7 |

**The model thinks it through.** At 512 tokens the judge called 144 of the 164 cut ARC-Challenge answers garbage.
At a cap of 1,024, 40 solutions reach the right answer, with a median length of 594 tokens. D2 has this knowledge
and needs more than 512 tokens of reasoning to reach it.

**On HotpotQA 7 answers of 135 needed the room.** Another 15 new excellent answers are shorter than 512 tokens. In
the first run they looped. Asked again in a batch of cut questions only, greedy decoding took another path: bf16
computation shifts with the batch's composition.

**D2's retention rises by 0.4 points.** 70 answers are newly excellent: 75 at the cap of 1,024 less the 5 already
excellent at 512. On 18,576 known questions this raises D2's retention from 51.4% to 51.8%.

**An open question.** The answer's cap might be tied to the level of precision, with more room to reason for a
coarse level. The proposed mechanism: near the end of the window, look at how the probability of ending the answer
changed over the last steps, and check that the reply does not loop. The window is extended once if that
probability rose without being high and there is no repetition. If it stays flat, the model is producing nonsense
without an answer and gets no extension.

Runs: `runs/E017-uniform-quantization-floor/reask-1024`, code `stage1_answers --cut-of --written-tokens` (e5a5abb).
