# E003 - The ladder over a calibrated base: preregistration

Written 2026-09-18, before the full run.

E003 tests no hypothesis. It measures the bench on its new parts - the model read from its `.refocustensors` file and
the tensor-core kernel - with a base precision taken from a published file, on the frozen corpus and the judge, as
E002 did.

## Why

The base precision D2 of our k-quant copy keeps 51.4% of bf16's knowledge, unsloth's UD-Q2_K_XL 80.0% (E002). The
model format lets a module's base be another quantizer's blocks, read as they lie, with our refinements to D8 and the
exact tail to the source over them. bartowski's Q2_K of E2B-it is k-quant in every controlled tensor and was quantized
with an imatrix (`quantize.imatrix.*` in its metadata), so the whole model can stand on it. The aim is the cheapest way
to raise the retention of the base precision, ideally to 80%, before the zones are tested on it.

## Design

- **The model**: E2B-it cut over bartowski's Q2_K (`scripts/cut_model.py e2b-it --base bartowski-Q2_K`, the file
  pinned in `gguf_weights.PUBLISHED`) at commit `c070122`: every base is the published file's bytes, the source reads
  back bit for bit (`tests/test_foreign_base_gpu.py`).
- **The corpus**: the frozen files `corpus/e2b-it`, 18,576 known questions and 2,064 of the unknown share.
- **The levels**: D2, D4, D6, D8, each baked from the file's copy, answered as E002's ladder was.
- **The judge**: the reasoning bf16 judge (`5b54c9e`), as in E002.
- **The measure**: the share of excellent answers, Correct and Nearly; retention - that share over bf16's on the known
  questions (0.919, E001).
- **Compared with**: E002's ladder on our own base and UD-Q2_K_XL, on the same questions.
- **The kernel on the corpus**: one round (5% of every corpus) of D4 of our own copy, baked against read by blocks
  through the tensor-core kernel (`stage1_answers.py --read blocks`, kernel `97e655a`): the time of the round and how
  many answers agree.

## Prediction

**P1** (Volodya's): the calibrated base raises the quality at every level - D2, D4, D6 and D8 each keep more than
E002's ladder at the same level. D8 is 1.2 points under the ceiling at 98.8%, so a level counts as raised when it is
above E002's or level with it within 0.5 point.
## Thresholds, stated before the full run

A smoke round (5%) at D2 was run first and gave 77.1% on its 932 known questions; the thresholds were stated to Volodya
before it.

- **T1**: D2 keeps at least 65% of bf16's excellent answers.
- **T2**: D4, D6 and D8 are each no more than 2 points below E002's ladder at the same level (91.0%, 96.8%, 98.8%):
  our refinements hold over a calibrated base.

The kernel round is an observation with no threshold, to see how the kernels came out on the corpus: the time of the
round and how many answers agree with the baked ones. The kernel's correctness is held by its tests.

If T2 fails, the base is not frozen for the zones and the loss is traced before anything else.

## Run

```
uv run python scripts/stage1_answers.py --level <d2|d4|d6|d8> --base bartowski-Q2_K --frozen corpus/e2b-it --gpu-share 0.9 --out runs/E003-calibrated-base
uv run python scripts/rejudge_answers.py --answers runs/E003-calibrated-base/e2b-it/unjudged --levels d2 d4 d6 d8 --frozen corpus/e2b-it --out runs/E003-calibrated-base
uv run python scripts/stage1_answers.py --level d4 --read <baked|blocks> --frozen corpus/e2b-it --rounds 1 --out runs/E003-calibrated-base/kernel-<baked|blocks>
```

About 55 minutes a level answered and 16 a level judged, some 5 hours in all.
