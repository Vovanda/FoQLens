# Addendum 14 - E013: the settings of the regulator that cost the fewest errors

Fixed 2026-09-12, **before the run**. Not edited after its commit.

## 1. Why

Everything so far asked whether the address beats a control. That is the scientist's question; the engineer's is the one a regulator actually has to answer: **set to what, does the model make the fewest mistakes, and what does that cost in memory?**

Random zones are dropped from the question. They are not an honest control here: zones of the same count and radii around random blocks overlap less, so they store and read more ([E011](../E011-depth-caps/results.md)) - a control that spends more and answers worse says nothing about the address. The honest reference for a setting is uniform quantization at the same memory, which is what the regulator would otherwise fall back to.

## 2. Given

- **The mechanism**: [docs/quantization-filter.md](../../docs/quantization-filter.md) at commit `c91af59`; sum of lifts, the default profile, attention level D4 behind an empty floor.
- **Model and data**: Gemma 4 E2B at the pinned revision; the 395 questions of E010 (biology 95, math 100, history 100, geography 100). Held-out topics are not opened.
- **The map**: floor D4, D2 and ZERO x focus area 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9 x focus strength 0.5 and 1 - denser where E010 found the effect lives, and out to where the lenses cover nearly everything.
- **Reported with every cell**: the error rate, the right-letter log-probability, the mean bits, and the share of the net lifted over the floor.

## 3. The measure

- **The error rate** is the share of questions whose highest-scoring letter is not the right one - what a user of the model feels. The log-probability stays as the second number, since it is what the earlier experiments compared on.
- **The cost** is the mean bits per weight the layout reads, weighted by block size.
- **The references**, once: bf16, uniform D4, D6, D8, and each bare floor. A cell is compared against uniform quantization interpolated at the same bits - the fallback a regulator has if it does not address anything.

## 4. Expected

- **R1 - there is a setting that matches bf16 within a point of error rate at under half its memory.** Expected: yes, near focus area 0.8 behind a D4 floor, around 6.5 bits.
- **R2 - the error rate is flat over a wide interval of the settings, not peaked.** Expected: within the interval focus area 0.7 ... 0.9 behind a D4 floor the error rate varies by less than two points, so the regulator has room rather than one lucky point.
- **R3 - below some size the lenses stop helping and start hurting.** Expected: behind an empty floor the error rate is worse than a uniform guess (75%) at focus area below 0.7, since a net with holes answers confidently and wrongly.
- **R4 - against uniform quantization at the same memory the lenses are at best level.** Expected: the difference in error rate is within two points at every cell; E010 already showed the log-probability difference is within noise.
- **The verdict names the intervals**: for each floor, the range of focus area and strength whose error rate is within one point of the best cell, and the memory that range costs.

## 5. Criterion

- Comparison of a cell against bf16 and against uniform at the same bits: paired bootstrap over the questions (10 000 resamples, seed 0) of the error indicator, 95% interval.
- R1 and R3 hold when they hold for the pair of topics they name at every cell in the stated range; R2 is read on the point estimates.
- The run writes `runs/E013-regulator-map/e2b/summary.json`; `results.md` gives the map of error rate against memory and names the working intervals.
