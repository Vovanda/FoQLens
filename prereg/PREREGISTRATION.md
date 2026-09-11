# FoQLens preregistration (English translation)

> **This is a translation.** The binding version is the Russian original [`PREREGISTRATION.ru.md`](PREREGISTRATION.ru.md), committed on 2026-09-11 in `be66b77` before any run and kept byte-for-byte since (renamed only). If the two ever disagree, the original wins.

Fixed 2026-09-11, before the first run. Basis - the [plan](../docs/plan.md), sections "Expected topology", "Step −1", "Bets", "Mandatory control". This file is not edited after the commit: clarifications and calibrated thresholds go into separate files in separate commits, and this one stays as it was.

Predictions are **directions, not numbers**. This is an exploratory experiment: the scale of cosines and concentrations is unknown before the first run, and a number named now would be pulled out of thin air. Numbers appear after calibration and the first pass, and are tested on held-out topics.

## 1. Conditions

- **Models.** Exploration - `google/gemma-4-E2B`, confirmation - `google/gemma-4-E4B`. Base checkpoints, not `-it`: what is measured is how pretraining organized knowledge, not the instruction-tuning layer. Queries are given as text without a chat template.
- **The step 1 instrument** is a naive score without training: the heaviest tokens of the query (by activation norm) as centers, a block's score is its response on these tokens. What a "block" is and how exactly the response is computed is fixed in the run code, committed before the launch.
- **Exploration phase - debugging domains:** biology, math, chemistry, physics, biophysics (mixed: biology + physics).
- **Domains are school subjects**, simple questions the model is known to tell apart.
- **Confirmation phase - held-out domains:** history, geography. Not opened and not run while the score is being debugged. Kept in `prompts/heldout/`.
- **Scale calibration:** known paraphrases and known-unrelated texts. Run before the main measurements; its result and thresholds go in a separate commit before the confirmatory pass.

| Role | Exploration | Confirmation |
| --- | --- | --- |
| Base (steps 0-1) | biology, math | history, geography |
| Related pair (step 2) | biology–chemistry | history–geography |
| Unrelated pair (step 2) | biology–math | history–math |
| Mixed and its components (step 2+) | biophysics = biology + physics | - |

## 2. Predictions by step

Notation: `cos_in(A)` - mean cosine between vectors of one domain, `cos(A,B)` - mean cosine between vectors of domains A and B.

### Step 0 - topics separate in representations

A check that the model is fit, not a result. The only step whose result is looked at immediately.

- At a middle layer, representations of biology and math queries cluster: `cos_in(bio)` and `cos_in(math)` > `cos(bio, math)`, and two-cluster clustering matches the domain labels better than chance.
- They do not separate on E2B → move to E4B. That is a conclusion about the model, not the scheme.

### Step 1 - masks are separable and concentrated (load-bearing)

Instrument predictions:

- **Separability.** Over score vectors: `cos_in(A)` > `cos(A,B)` on **all** pairs of debugging domains.
- **Concentration.** The mask is concentrated, not flat: the score mass sits in a small share of blocks (the Gini coefficient / entropy of the score distribution over blocks differs noticeably from uniform).

If this fails, the conclusion is only about the score (the activation norm cannot catch the topic), not about the topology. The next move is fixed in advance: **background subtraction** (score minus the mean score over all queries), then a change of instrument by increasing cost - the output gradient per block, block ablation.

### Step 2 - zones overlap

- Exploration: `cos(bio, chem)` lies **between** `cos(bio, math)` and `cos_in(bio)`.
- Confirmation: `cos(history, geography)` lies **between** `cos(history, math)` and `cos_in(history)`.

### Step 2+ - mask geometry (on step 1 data)

The mixed domain is biophysics, its components are biology and physics. Read **only with a working instrument** (step 1 passed). All items below are refining: a failure does not bring down the statement but simplifies it.

| Test | Prediction |
| --- | --- |
| Additivity | **Not additive.** The biophysics mask does not fit a linear combination of the biology and physics masks: the residual of the best fit has blocks raised on biophysics and on neither component. |
| Support shape | **Two bundles, not a blob.** The concentration of the biophysics mask is roughly that of biology, the mass is wider. |
| Bundles coincide with pure zones | There is a **third group** - top blocks of biophysics that are among the top blocks of neither biology nor physics (a junction zone). |
| Isthmus | **Exists.** Between the bundles there are blocks with a score above the background and below the centers, with no drop to zero. |
| Isthmus ablation | Coarsening the isthmus while the bundles stay precise: quality on pure tasks **does not drop**, on mixed ones it **drops**. |
| Reverse ablation | No prediction - exploration. Its result counts neither for nor against. |
| Linearity of representation → mask | **Non-linear.** In representation space biophysics lies between biology and physics; in block space the biophysics mask is bimodal, not in the middle. |

### Step 3 - quality against budget

- The quality curve against the **mean** number of bits for the directed scheme lies **above both** baselines:
  1. uniform quantization at the same mean bits;
  2. a random mask of the same concentration (the same share of blocks read deep, chosen at random).
- Beats uniform but not random → the gain comes from non-uniformity itself, the score does not work. Counted as a **negative** result, even if the numbers against uniform look good.

## 3. Expected topology of the ideal system

Seven properties the system will show if the statement is right. Detailed reasons are in the [plan](../docs/plan.md#expected-topology-of-the-ideal-system).

| # | Property | Tested by | Type |
| --- | --- | --- | --- |
| 1 | Not additive | additivity test, step 2+ | refining |
| 2 | Strongly overlapping | step 2 | refining |
| 3 | With isthmuses that carry a function | isthmus + ablation, step 2+ | refining |
| 4 | Sparse | concentration, step 1 | **load-bearing** |
| 5 | Hierarchical | some blocks rise on almost all natural-science domains (bio, chemistry, physics, biophysics), some on one only | refining |
| 6 | The representation → mask mapping is non-linear | linearity test, step 2+ | refining |
| 7 | Uneven in zone size | the procedure is not defined at the time of preregistration; its result is exploratory | refining |

**Load-bearing** - mask separability (step 1) and concentration (property 4). If they fail with a known-working score, there are no zones and nothing to hold on to. "Simplified, fixable" does not apply to them.

## 4. Bets on the outcome of step 1

- **Vladimir:** the trivial score is enough for a measurable, substantial gain already in the naive setup.
- **Claude:** there will be separation, but dirty - a norm-based score will light up both the topic and simply large blocks.

Resolved by background subtraction: separability after subtraction **did not change** → Vladimir is right; **improved sharply** → Claude is right.

## 5. Order of work

1. This file - the first commit, with the predictions.
2. Query sets and run code - commits before the launch.
3. Step 0 - run and look at once (the exception).
4. Scale calibration → thresholds in a separate commit.
5. All exploration runs at once; during them check only that the script did not crash. Raw results go to files, all opened together afterwards.
6. The confirmatory pass on held-out domains and E4B with the thresholds from item 4.
7. Results - in separate commits.
