# Addendum 03 - step 3 as an aperture sweep, and what the curve should look like

Fixed 2026-09-11, **before the full step 3 run**. Not edited after its commit.

## 1. The wider experiment

Step 3 is no longer a single budget. The mask says *where* to read weights sharp; the **aperture** says *how much*: the share of weights read at bf16, the rest at nf4. Aperture 0 is a closed lens (uniform nf4, 4 bits), aperture 1 fully open (uniform bf16, 16 bits); at aperture a the mean is 4 + 12 a bits, so a = 1/3 costs the same 8 bits as uniform int8.

- **Apertures:** 0.05, 0.1, 0.2, 1/3, 0.5, plus the ends (uniform nf4 and bf16) and uniform int8 at 8 bits.
- **Layouts at every aperture:** directed by the question's own mask - two sources, the naive pooled score and gradient × activation, each with the background (mean mask over all questions) subtracted - and random blocks of the same aperture.
- **Questions:** 591 MMLU-Redux questions with answers in six domains ([docs/data-sources.md](../docs/data-sources.md)); the mask of a question is computed from its full prompt (question and options) in a first bf16 pass.
- **Quality:** accuracy, and the log-probability of the right letter (smoother, used for the tests below).

This is the upper-bound version: the mask comes from a full bf16 pass. The one-pass online version (address from the first layers) follows only if this one works. Memory is not saved in this bench (bf16 and packed copies side by side); the real saving is step 5.

## 2. The main criterion, fixed now

**The mechanism works at aperture a** if the question-paired difference of the right-letter log-probability, directed minus random at that aperture, is positive with a 95% paired-bootstrap interval (10 000 resamples over questions, seed 0) that excludes zero. Reported per source and per aperture; at a = 1/3 directed is also compared with uniform int8 the same way.

## 3. Predictions about the shape of the curve

- **P1 (both authors).** Quality rises with the aperture for both directed and random layouts.
- **P2 (Claude).** The gap directed − random is zero at a = 0 and a = 1 by construction and peaks at small to middle apertures, where the mask already covers what matters and random blocks do not yet. A gap that keeps growing up to a = 0.5 would refute this shape.
- **P3 (Vladimir).** The thicker the mask, the clearer the effect: the directed advantage grows with the aperture over the tested range.

P2 and P3 disagree on 0.05-0.5; the sweep decides between them.
