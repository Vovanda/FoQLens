# Addendum 04 - mask injection: the mask of topic A on the questions of topic B

Fixed 2026-09-11, **before the injection run**. Proposed by Vladimir; not edited after its commit.

## 1. Why

The most direct test that a mask carries a topic: if it does, the same mask at the same aperture keeps the answers of its own topic and breaks the answers of another. It is run where the regions are polar - blocks outside the aperture removed (ZERO): the edge tests showed a random 5% hole already breaks E2B, so a mask that carries nothing cannot hide there.

## 2. Design

- **Pairs:** biology–math (polar) and history–geography (hard: the model separates them only faintly in representations).
- **Masks:** for every question, from a first bf16 pass, from two sources (naive pooled score, gradient × activation), background (mean over all questions of the run) subtracted.
- **Topic mask:** the mean mask of a topic's questions; for a question of that topic the question itself is left out (leave-one-out), so no question sees its own mask through the topic mean.
- **Policies at every aperture** (0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.5; outside the aperture ZERO):
  - `self` - the question's own mask;
  - `own` - the topic mask of the question's topic;
  - `other` - the topic mask of the paired topic;
  - `random` - random blocks of the same aperture.
  Plus uniform bf16 and uniform ZERO as the ends.
- **Quality:** right-letter log-probability and accuracy on the MMLU questions of the four topics.
- **Comparison:** paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability difference; 95% interval.

## 3. Predictions

- **I1.** `own` beats `other` at the same aperture: the interval of own − other lies above zero, for the polar pair at least at the apertures where random layouts break.
- **I2.** The effect is larger for the polar pair (biology–math) than for the hard pair (history–geography).
- **I3.** `self` and `own` are close: a topic mask is enough, the question's own mask adds little.

If own and other are indistinguishable everywhere, the masks do not carry the topic in a way that matters for the computation, whatever the representation probes say.
