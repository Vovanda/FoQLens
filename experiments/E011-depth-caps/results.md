# Results - when a depth cap saves storage (experiments/E011-depth-caps/ADDENDUM-12.md)

Run 2026-09-12 on the bench of `c1bd457`, after ADDENDUM-12 (`4114696`); the mechanism is [docs/lens.md](../../docs/lens.md) at `c91af59`. The 395 questions of [E010](../E010-lens-layout/results.md), their gradient masks and weight map; focus strength 1. Everything but the mask pass is computed on the CPU: a cap never changes what a question reads, so quality is not evaluated again. Raw numbers: [runs/E011-depth-caps/e2b/summary.json](../../runs/E011-depth-caps/e2b/summary.json).

The full sliced copy costs 8 bits per weight. A 30% saving means storing at most 5.6.

## What a layout reads and what it has to store

| Floor, focus area | A question reads | All 395 store | Backbone caps | Random zones |
| --- | --- | --- | --- | --- |
| D4, 0.2 | 4.00 | 4.04 | 4.00 | 7.86 |
| D4, 0.5 | 4.28 | 6.47 | 4.06 | 8.00 |
| D4, 0.8 | 6.65 | 8.00 | 4.41 | 8.00 |
| ZERO, 0.2 | 0.02 | 0.26 | 0.16 | 7.86 |
| ZERO, 0.5 | 0.82 | 6.07 | 0.16 | 8.00 |
| ZERO, 0.8 | 6.32 | 8.00 | 4.41 | 8.00 |

Reading and storing part company as soon as the zones are large enough to lift anything: behind an empty floor a question reads 0.82 bits and the store needs 6.07 - seven times more. At focus area 0.8 nothing is saved at all: every block is read at D8 by someone.

## Verdict against the predictions

| Prediction | Verdict |
| --- | --- |
| S1 storage grows with the number of questions and saturates | **holds** - D4 at 0.5: 4.08 at one question, 4.95 at five, 5.97 at twenty, 6.30 at fifty, 6.47 at 395; the rise is steepest in the first few dozen |
| S2 one topic is cheaper than four | **holds, but the reason is not the count of topics** - biology alone stores 4.02, geography alone 6.43, and biology with math together 4.03 |
| S3 a 30% saving survives only for a narrow profile | **holds** - at D4 and focus area 0.5 the whole set passes 5.6 bits after 10 questions, geography after 2, while biology and math stay under it at every size measured |
| S4 backbone caps store less than own caps | **holds** - 4.06 against 6.47 at D4 and 0.5, 0.16 against 6.07 behind an empty floor; E010 already showed own zones read better (L6), so this is the price of that |

## Why one topic costs seven times another (exploratory)

The shape of a topic's zones, not their number, decides what its store costs:

| Topic | Zones per question | Mean radius | Spread of centers | One question lifts | The topic together lifts |
| --- | --- | --- | --- | --- | --- |
| biology | 3.6 | 0.116 | 0.260 | 0.5% | 1.3% |
| math | 2.9 | 0.137 | 0.255 | 0.6% | 0.6% |
| history | 3.0 | 0.123 | 0.181 | 3.4% | 6.0% |
| geography | 5.3 | 0.100 | 0.412 | 37.9% | 89.1% |

Geography's masks are not local: more zones, spread twice as far, and a single question already lifts 38% of the map. Its questions then share almost nothing, and the store grows to the full copy. Biology and math sit in the same small region of the map, so together they cost as little as either alone - which is also why E010 found an address for that pair and none for history-geography.

This was not preregistered and is read as exploration.

## What it means

Addressing precision by the query saves reading, not storage - unless the queries are few or their zones lie in the same place. Two ends of the scale:

- **A device with a narrow profile** - one topic, or a pair whose zones coincide - stores 4.0 bits against the full copy's 8 and reads 4.3: the caps are nearly free, and the lenses cost only what they read.
- **A general model serving everything** - 395 mixed questions - stores 6.5 of 8 bits behind a D4 floor. Caps still save a fifth, but the promise of "memory follows the lenses" belongs to reading, not to the file on disk.

The way to make reading the thing that costs is a kernel that fetches only the slices a block's level asks for ([issue #3](https://github.com/Vovanda/FoQLens/issues/3)); until then a lens layout saves bandwidth on paper only. The backbone's caps are the cheap store - one set of zones for every question, 4.06 bits - and E010 measured what that costs in quality: own zones read better (L6, both pairs).
