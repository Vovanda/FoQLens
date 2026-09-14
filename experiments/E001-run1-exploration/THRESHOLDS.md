# Thresholds 01 - the instrument and the pass criteria for the confirmatory pass

Fixed 2026-09-11, after the run 1 exploration ([results](results.md)) and **before any run on the held-out domains or on E4B**. Required by the [preregistration](../../prereg/PREREGISTRATION.ru.md) (section 5, item 4) and [E001 PREREG](PREREG.md) (section 2). Not edited after its commit.

## 1. The instrument

**Mode B (pooled) with the background subtracted** - the only one of the three center modes that met the preregistered step 1 criterion in exploration (10/10 pairs; mode A 7/10, mode C 8/10 with the background subtracted). Modes A and C are not computed in the confirmatory pass.

- Block score: L2 norm of the block's output, averaged over every token except `<bos>`, from one eager-attention bf16 pass (`scripts/step1_masks.py --modes pooled`).
- Background: the mean mask over all queries of the confirmatory run, subtracted from every mask.

## 2. Why a permutation test and not a number

Calibration ([E001](_index.md)) showed that the cosine scale depends on the vector kind: raw masks sit at 0.97-0.99 for any two texts, background-subtracted ones spread over ±0.5. A fixed cosine margin would be arbitrary. The threshold is therefore taken from the data itself:

- **margin** of a domain pair = min over the two domains of cos_in − cos_between;
- **permutation test**: the domain labels inside the pair are shuffled 1000 times (seed 0); p = (1 + #{permuted margin ≥ observed}) / 1001;
- **pass** = the preregistered cosine direction holds (both cos_in above cos_between) **and** p < 0.05.

## 3. The confirmatory pass

Queries: `prompts/heldout/history.jsonl` (MMLU `prehistory`), `prompts/heldout/geography.jsonl`, and `prompts/math.jsonl` as the unrelated partner for step 2. First on E2B, then on E4B, same criteria.

| Claim | Criterion | Confirmed when |
| --- | --- | --- |
| Step 1 - masks separate by topic | pair history–geography passes (section 2) | on both E2B and E4B |
| Step 2 - zones overlap | cos(history, geography) lies between cos(history, math) and cos_in(history) | on both E2B and E4B |

Step 1 concentration is reported (Gini, entropy of raw masks) but carries no pass criterion here: exploration showed the raw concentration is the same for every topic, so it cannot confirm topical zones either way.

## 4. What happens after

- Both claims confirmed → step 3 (quality against budget, two baselines) with mode B.
- Step 1 not confirmed → the instrument is changed, as the preregistration fixes: the output gradient per block, then block ablation. The topology predictions stay unread until an instrument passes.
- Step 2+ is not part of this pass: exploration showed the mixed domain needs to be rebuilt first.
