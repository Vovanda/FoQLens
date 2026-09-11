# Addendum 07 - bubbles: the topic address in the shape of the picture

Fixed 2026-09-11, **before the run**. Not edited after its commit.

## 1. Why

The runs so far read a mask as a flat ranking of blocks and opened the top blocks one by one; a topic could only show up as scattered points. The project's picture needs bubbles: centers, radii and a stepped falloff ([docs/bubbles.md](../docs/bubbles.md)). This run tests the address in that shape.

## 2. Design

- Questions and pairs as in ADDENDUM-04 to 06: biology-math and history-geography, 395 questions.
- **Plane:** the raw gradient x activation masks of all questions of the run, embedded in 2D by co-activation (docs/bubbles.md, section 1). The plane is built without the topic labels.
- **Bubbles of a field:** smoothed on a 64 x 64 grid; one bubble per hill above the 95th percentile of the smoothed field, at most 16; base radius from the hill's area above half height.
- **Fields:** the question's own topic mask and the paired topic's mask (background subtracted, leave-one-out), pooled and gradient sources; the generic importance backbone (mean raw gradient mask) as a reference.
- **Layout:** rings of log-sharpness into D4 (background), D6 and D8 (centers) at a fixed mean of 5 bits per weight; regulator s = 0.2, 0.35, 0.5, 0.65, 0.8, 1.0. s = 0 is the uniform limit: no bubbles, the budget spread at random.
- **Random bubbles:** as many bubbles as the own topic's, with the same radii, centered on random blocks; the same centers at every s.
- **References:** uniform bf16, D4 and D8.
- **Comparison:** paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval. A cell is one pair x source x s: 12 cells per pair.

## 3. Predictions

A prediction **holds for a pair** when at least 3 of its 12 cells have the interval above zero and none below. With no effect, 3 or more of 12 cells above zero by chance has a probability of about 0.3%.

- **S1 - the shape finds the topic.** Own bubbles beat random bubbles of the same count and radii.
- **S2 - the address.** Own bubbles beat the paired topic's bubbles.
- **S3 - bubbles beat no mask.** Own bubbles beat the uniform layout at the same budget.

If own and random bubbles are indistinguishable for both pairs, the co-activation plane does not place a topic's blocks together.
