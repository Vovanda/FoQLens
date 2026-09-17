# E002 - Quantization mechanics: preregistration

Written 2026-09-17.

E002 tests no hypothesis, so it has no predictions and no criteria. Every arm is a measurement on the frozen corpus
and the judge, as in E001:

- the corpus - the frozen files `corpus/e2b-it`: 18,576 known questions and 2,064 questions of the unknown share;
- the judge - the reasoning bf16 judge (`5b54c9e`), one for every arm;
- the measure - the share of excellent answers, Correct and Nearly; retention - that share divided by bf16's share on
  the known questions (0.919).
