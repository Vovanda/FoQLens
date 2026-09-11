# Results - backbone + topic on E2B (prereg/ADDENDUM-05.md)

Run 2026-09-11 after ADDENDUM-05 was committed (`e1f8a6d`), on the bench of `573a1af`. 395 questions (biology 95, math 100, history 100, geography 100); outside the aperture every block is removed (ZERO); 121 policies. Raw numbers: [runs/backbone/e2b/summary.json](../runs/backbone/e2b/summary.json).

Uniform bf16: -1.012 mean right-letter log-probability, accuracy 0.547. Every block removed: -1.386 (the uniform guess), accuracy 0.208.

Two regimes are read separately. At apertures **0.9-0.99** the model still answers (the backbone keeps accuracy 0.43-0.55). At **0.8 and 0.5** every policy is at chance (accuracy 0.21-0.29): the log-probabilities there only say how a broken model spreads its guess.

## Verdict against the predictions

Count of cells whose 95% interval lies entirely above (+) or below (-) zero; each pair has 6 cells (share x source) per aperture, 30 in the working regime and 12 in the broken one. At 95% about one cell in twenty is expected on either side by chance.

| Prediction | biology-math | history-geography | Result |
| --- | --- | --- | --- |
| B1 backbone alone beats random blocks | + at 7 of 7 apertures | + at 7 of 7 apertures | **supported**, large: +0.20 at 0.99 up to +4.3 at 0.5 |
| B2 backbone + own topic beats backbone + other topic | working: 3+ / 3-; broken: 4+ / 0- | working: 0+ / 0-; broken: 0+ / 1- | **formally met for biology-math only** (e.g. a = 0.97, share 0.8, pooled: +0.107 [+0.039, +0.177]); in the working regime the positive cells are balanced by negative ones - no address |
| B3 backbone + own topic beats backbone + random fill | working: 0+ / 8-; broken: 1+ / 1- | working: 2+ / 11-; broken: 5+ / 0- | **not supported** where the model works: the topic fill is more often worse than a random fill |

## What the absolute levels show

Mean right-letter log-probability; "best fill" is the best of the 15 backbone + fill policies at that aperture.

| Aperture | random blocks | backbone alone | best fill |
| --- | --- | --- | --- |
| 0.99 | -1.280 | **-1.012** (= bf16) | -1.009 (share 0.8, random fill) |
| 0.98 | -1.478 | -1.037 | -1.042 (share 0.95, other, pooled) |
| 0.97 | -1.723 | -1.063 | -1.050 (share 0.95, own, pooled) |
| 0.95 | -1.996 | -1.110 | -1.080 (share 0.95, own, pooled) |
| 0.90 | -2.920 | -1.245 | -1.169 (share 0.95, own, pooled) |
| 0.80 | -4.663 | -2.101 | -1.743 (share 0.95, own, gradient) |
| 0.50 | -7.006 | -3.071 | -4.205 (share 0.8, other, gradient) |

- **Static importance carries almost everything.** Removing 5% of the weights at random costs 0.98 nat; removing the 5% with the lowest generic importance costs 0.10. At 1% removed the backbone is indistinguishable from the full model.
- **Filling with the topic adds nothing measurable** where the model works: the best fill beats the backbone alone by at most 0.08 nat (a = 0.9), and the fill that wins is not consistently the own topic. The backbone vs backbone + fill contrast was not preregistered; this is a description, not a test.
- Halving the backbone share (0.5) is clearly worse than 0.8 or 0.95 at apertures 0.9-0.98, whatever the fill: the topic masks are a poor substitute for generic importance.

## What it means

The ADDENDUM-05 fallback applies: over a generic importance backbone, an untrained topic address (background-subtracted pooled or gradient masks, averaged over the topic) adds nothing to known importance-based allocation. The topic is in the masks - a linear probe reads it - but the blocks it points at are not the ones a topic's answers depend on.

Open routes that are not yet closed by this: widening the sharp windows to structural neighbours (dilation), a mask computed online from the query's own first layers, and a learned score (step 4).

## Bench

- Mask phase: GPU utilization 27% mean, peak 15.2 GB allocated - the gradient pass is the bottleneck (the chunked cross-entropy item).
- Evaluation: 64% mean / 88% median, peak 9.9 GB allocated.
- The first attempt of this run spilled 6.4 GB into shared system memory and was stopped; `573a1af` fixed the cause (weight copies only for levels in use, towers dropped, allocator capped at free VRAM).
