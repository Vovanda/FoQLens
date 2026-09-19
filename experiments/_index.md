---
title: Experiments
---

# Experiments

One folder per experiment, `E0NN-slug`: its preregistration `PREREG.md`, its card `_index.md` and its `results.md`;
the raw runs are in `runs/E0NN-slug/`. The main preregistration is in [`prereg/`](../prereg/), the hypotheses in
[docs/hypotheses.md](../docs/hypotheses.md).

The first runs went on four MMLU subjects answered with an option letter, and nobody checked whether the model knew
the questions: it mostly guessed, and those runs are deleted. What that taught is in [corpus.md](../docs/corpus.md).

Status: `planned` - preregistration in progress; `fixed` - preregistration committed, not run; `done`.

| Id | Experiment | Fixed | Hypotheses | Status |
| --- | --- | --- | --- | --- |
| [E001](E001-uniform-quantization/_index.md) | Uniform quantization on the corpus: the share of knowledge each level keeps | 2026-09-15 | H6 | done |
| [E002](E002-base-precision-d2/_index.md) | Uniform quantization with a working base precision D2: the k-quant ladder D8-D2, the baseline of the filter | 2026-09-17 | - | done |
| [E003](E003-calibrated-base/_index.md) | The ladder over a calibrated base: our refinements over bartowski's imatrix Q2_K, read from the model file, on tensor cores | 2026-09-18 | - | done |
