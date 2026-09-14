# Preregistration addendum 01 - how centers are chosen, and the run 1 data

Fixed 2026-09-11, **before any run** of step 0 or step 1. It adds to the [preregistration](../../prereg/PREREGISTRATION.ru.md) (English translation: [PREREGISTRATION.md](../../prereg/PREREGISTRATION.md)) and changes none of its predictions. Like the preregistration, this file is not edited after its commit.

## 1. Three ways to choose the centers

The preregistration fixed one instrument for step 1: centers are the top-k tokens by activation norm. The original plan ([docs/plan.md](../../docs/plan.md), step 1) allowed two: "by activation norm **or by the share of attention** on them". A check of the bench on 2026-09-11 showed that on E2B the token norms at the middle layer are packed tightly (median 75.2, max 77.4 on one sentence), so choosing by norm may be close to choosing at random. Two more variants are therefore added. All three are computed in the same pass, over the same queries.

| Variant | Centers | Block score |
| --- | --- | --- |
| **A - norm** (the preregistered one) | top-4 tokens by hidden-state norm at the middle decoder layer | mean over the centers of the block's output L2 norm |
| **B - pooled** | every token | mean over all tokens of the block's output L2 norm |
| **C - attention** | top-4 tokens by the attention they receive at the middle decoder layer | as in A |

Common to all three:

- The first token (`<bos>`) is never used: it carries no meaning and is a common attention sink.
- The middle decoder layer is layer 17 of 35 (`hidden_states[18]`, `attentions[17]`).
- **Variant C normalization.** The attention a token receives is summed over heads and queries and divided by the number of heads times the number of queries allowed to look at it by the causal mask. Without this, early tokens win simply because more positions can attend to them - the mirror image of the positional pull seen with the norm. Inputs are shorter than the 512-token sliding window, so the causal count is exact.
- The whole step 1 run uses eager attention, which is the only implementation that returns attention weights. All three variants therefore share the same numerics.

## 2. How three variants are read without fishing

Three instruments are three chances to "confirm" by luck. The rule:

1. **Exploration** (debugging domains): all three variants are computed and reported, each against the same preregistered predictions. The bets on step 1 refer to variant A and are resolved on A.
2. **Choice**: after exploration one variant is chosen as the instrument, and the choice goes into the thresholds commit (preregistration, section 5, item 4), together with the calibrated thresholds, **before** the confirmatory pass.
3. **Confirmation** (held-out domains, E4B): only the chosen variant is tested. A conclusion about zones rests on the confirmatory pass, not on the best of three in exploration.

## 3. Run 1 data

The source and the reasons for it are in [docs/data-sources.md](../../docs/data-sources.md). What matters for the predictions:

- Questions: MMLU-Redux-2.0 at a pinned revision, `error_type == "ok"` only, question text without options or a chat template.
- **History in the confirmatory phase is the MMLU subject `prehistory`.** The other MMLU history subjects are built on long quoted sources (median ~1200 characters against 78-201 elsewhere), and separation would measure length. Prehistory partly overlaps with biology (human evolution); the confirmatory step 2 prediction for history–geography vs history–math stays as preregistered, and this overlap is noted for reading it.
- Biophysics (30 questions) is hand-written by Claude, a party to the step 1 bet, and committed before any run.
- Calibration pairs come from PAWS at a pinned revision: paraphrases are `label == 1`, unrelated pairs are random cross pairs.
