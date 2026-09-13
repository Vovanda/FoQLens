# Precision share, focus area and expert zones

> **Legacy layout.** This page describes the fixed-budget layout of ADDENDUM-07 to 10: the mean bits are set in advance and the zones are fitted to them. It is replaced by the layout of [lens.md](lens.md): the whole network at a base precision, the expert zones read more precisely, memory the result of the settings.

How precision is laid out over the weights. Two parameters, both in [0, 1] and checked where they enter:

- **precision_share** - how much precision the model gets: the share of the precision range spent, the memory and compute it pays for;
- **focus_area** - where that precision goes: the area read sharp, from the centers of the query's **expert zones** ([problem statement](problem-statement.md)) to the whole weight map.

Fixed 2026-09-11, names from a tier list 2026-09-12 ([ADDENDUM-10](../experiments/E009-zones-matrix/ADDENDUM-10.md)). The current layout is in [lens.md](lens.md); here the parameters of the legacy layout are named by what they do.

## Precision share - how much

The precision share `p` is the share of the precision range spent. With the levels of the zones - D4 for the background, D8 for the centers - a layout at precision share `p` holds a mean of `4 + 4p` bits per weight; with two levels it is the share of the weights read at the higher one.

| `p` | Meaning |
| --- | --- |
| 0 | every block at the floor level |
| 1 | every block at the top level |
| between | the budget: 0.25 is 5 bits per weight on D4 ... D8 |

- Every layout at one precision share spends the same bits (up to one block): this is what makes layouts comparable. A mask that simply spent more would win for nothing.
- The precision share is the resource knob. A future governor that lowers the model's quality under load or heat turns the precision share down; the focus area stays with the query.

## What was wrong in the early runs

The mask was a flat list of block scores, and the top blocks were opened one by one until the budget was spent. There were no centers and no radii: a topic could only show up as scattered single blocks, and dilation ([E007-dilation](../experiments/E007-dilation/_index.md)) only regrouped the same budget around those points. The problem statement asks for something else: zones sharp at their centers, precision falling off around them, the background coarse ([problem statement](problem-statement.md)). That needs centers, distances and radii.

## Expert zones - in what shape

### 1. The weight map

Blocks are placed on the **weight map** by co-activation: every block gets its vector of raw mask values over the questions of a calibration set; blocks that light up together are close. The vectors are embedded in a low-dimensional space (2D for the picture; the dimension is a parameter), and the distance `d(b, b')` between blocks is the distance on the map.

### 2. Zones from the context

The query's mask `m(b)` is a field over the weight map. Its peaks, found after smoothing on the map, are the expert zones: `n` of them, each with a center `c_i`, an amplitude and a base radius `r_i` (the width of the peak at half height). A stronger, wider peak gives a bigger zone. `n` is not fixed - the context decides it.

### 3. The focus area `f`

One number for all zones of the query:

```
R_i(f) = r_i * f / (1 - f)        f in [0, 1]
```

| `f` | Radius | Meaning |
| --- | --- | --- |
| 0 | 0 | the zones' centers only: a hard edge, all the precision at the centers |
| 0.5 | `r_i` | the zones as found |
| 1 | infinity | the whole weight map: no mask, the precision share spent evenly |

- Monotone in `f`, and the same factor for every zone, so the relative sizes of the zones are kept.
- The focus area does not change the budget: at one precision share it only moves the bits between the zones and the rest.

### 4. Sharpness and levels

```
phi(b) = max_i exp(-d(b, c_i) / R_i)        in (0, 1]
```

The sharpness is cut into rings - D8 at the centers, then D6, D4 for the background - read on `log phi = max_i(-d / R_i)`, which keeps the order of far blocks when `R` is small. The edge of the rings is searched so that the mean bits equal the budget of the precision share. D2 is not used as the background: uncalibrated 2 bits break the model ([E006-read-depths](../experiments/E006-read-depths/_index.md)).

### 5. Memory follows the zones

The level of a block is also the depth it stores (depth caps of the resident bench, [E006-read-depths](../experiments/E006-read-depths/_index.md)): a block in the background keeps only its first slices, a center keeps all four.

## What is compared

For every cell of precision share x focus area:

- the zones of the query's own topic;
- **random zones** - `n` random points of the map with the same radii and the same falloff;
- the zones of the paired topic;
- the precision share spent evenly, without a mask (focus area 1), the reference of its row.

The idea works where own-topic zones beat random zones and the other topic at the same precision share. The cells run from the most promising (mid precision share, zones as found) to the edges. Random zones are no longer a reference: they overlap less, so they cost more ([plan](plan.md)).

## Names used before

| Earlier documents | Now |
| --- | --- |
| aperture (ADDENDUM-03 to 06, results before the zones) | precision_share - with two levels, the share of weights read sharp |
| precision `p` (ADDENDUM-09) | precision_share |
| bubbles (ADDENDUM-07) | expert zones |
| regulator `s` (ADDENDUM-07), focus `f` (ADDENDUM-08, results-zones), spread `s` (ADDENDUM-09) | focus_area, `f = 1 - s_ADDENDUM-07 = f_ADDENDUM-08 = s_ADDENDUM-09` |
| plane of blocks | weight map |
