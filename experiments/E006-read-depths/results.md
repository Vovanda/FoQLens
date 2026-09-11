# Results - read depths and the resident bench on E2B (docs/plan.md, step 5)

Measured 2026-09-11 on the bench of `03a9881` with `scripts/depth_perplexity.py`. Mean perplexity over 40 question texts (biology, math, chemistry, physics, 10 each). Raw numbers: [runs/E006-read-depths/e2b/summary.json](../../runs/E006-read-depths/e2b/summary.json).

## Storage

One `SlicedWeight` per controlled module, after MoBiQuant (arXiv 2602.20191): 4 slices of 2 bits, slice 1 quantizes the weight, each next slice quantizes what the earlier ones left with a 4x finer step; groups of 64 weights inside a row. Reading the first k slices gives the read depths D2, D4, D6, D8. The full copy costs 1 byte per weight plus one fp32 scale per 64 weights.

## Quality by level

| Level | Bits per weight | Perplexity |
| --- | --- | --- |
| bf16 | 16 | 12.700 |
| int8 (row absmax) | 8 | 12.687 |
| **D8** | 8 | **12.687** |
| **D6** | 6 | 12.828 |
| **D4** | 4 | 14.745 |
| nf4 (bnb, group 64) | 4 | 15.515 |
| D2 | 2 | 9.4e6 - diverges |

- D8 is indistinguishable from int8 and bf16; D6 costs 0.13.
- At 4 bits the uniform sliced base beats nf4 (14.75 vs 15.52). An nf4 base with uniform residual slices on top was also measured and lost at every depth (D6 13.36, D8 12.80), so it was not kept.
- 2 bits without calibration break the model. MoBiQuant calibrates its base (OmniQuant); ours is round-to-nearest.
- A single text is not enough for such comparisons: on the one text of the stand test D4 looked worse than nf4 (6.90 vs 6.11).

## Resident bench

After `Controller.drop_bf16()` every module keeps only its sliced copy.

| | GPU memory allocated |
| --- | --- |
| bf16 bench after install | 8828 MiB |
| resident bench | 7199 MiB |

- D8 read from the resident copy gives exactly the same perplexity (12.687); the logits are bit-exact (tested).
- The saving, 1.63 GiB, is the bf16 bytes of the controlled weights (3.51 GiB) minus the sliced copy (1.75 GiB) and its scales.
- What remains is dominated by the per-layer embedding table (4.4 GiB), which is a lookup and a candidate to leave the GPU.
- The saving is the same whatever layout is read: every slice stays stored. Directed memory savings would need blocks that are never read deep to drop their deeper slices.
