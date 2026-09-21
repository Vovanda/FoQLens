---
title: Zone strategies
---

# Zone strategies

The level rules in [precision-regulator.md](precision-regulator.md) say how a zone becomes levels. They do not say what a zone covers, where it grows from, along what, how far or how fast. Each of these is a replaceable strategy. The derivations behind them are in [the mathematics of the filter](bench-math.md); the code is `foqlens.strategies`, every part picked by name.

## What a zone covers: the query's subgraph

A query engages some blocks of the network more than its background does. Linked on the block graph, they form **the query's subgraph**: connected, of any shape - a tree, a graph with cycles, bridges from layer to layer. A zone is how the regulator covers it. A zone that covers blocks the query does not engage spends memory on them; a zone that misses the subgraph leaves the query's blocks at the base.

A ball does not cover the subgraph. A ball along a distance that ignores the query takes every block within reach, engaged or not. On a graph that expands, it grows by jumps ([math, section 8](bench-math.md#8-the-shape-of-a-map-what-is-measured)). Measured in one configuration only - the pooled address, the mutual-NICDM graph of 16 neighbours, rule 1 with the width from sweeps, the linear profile, the smoke of TriviaQA (13 questions), base D2: static zones lift a median 0.038 of the network at $f = 0.1$ and 0.497 at $f = 0.2$; every question gets 16 zones, the cap `MAX_ZONES`, not a number of its own.

What a strategy is judged by, before any answer:

- the share of the network it lifts grows smoothly with $f$, with no jump to half the network;
- the bytes of the layout against the uniform ladder: zones over the whole network still allocate while their profile keeps a gradient (at $f = 0.2$ the medium lifts half the network by rungs D4-D8 for 0.90 of uniform D4), and stop allocating when the gradient flattens to the ceiling (at $f = 0.4$: 0.996 of the network, 0.99 of uniform D8) - what decides is the quality of answers at the same bytes;
- the number of zones comes from the query, not from the cap;
- the zones repeat on the same query in other words (the stability of [math, section 11](bench-math.md#11-what-is-measured-and-what-is-open)), and differ from what the same procedure builds on shuffled topic labels.

## Where zones grow from

A source scores how much the query needs a block. The working source is read in one forward pass over the first layers at base precision, and a projection fitted on calibration questions carries it onto every block ([math, sections 4-6](bench-math.md#4-the-address-signal-background-excess)).

| Rank | Source | Measured |
| --- | --- | --- |
| 1 | hybrid: neuron activity at the input of `down_proj` plus head energy at the input of `o_proj`, each scaled to length 1 | finds its question across wrappers 0.997-1.000, a paraphrase 0.917 (words alone 0.717); the first 4-8 layers at D2 predict its address in the rest in 0.82-0.89 of cases |
| 2 | neuron activity alone | 0.995-1.000; paraphrase 0.892 |
| 3 | output energy of a block (pooled) | 0.820-0.886; paraphrase 0.308, below the words |
| comparison | Taylor score | 0.47-0.49: not an address; needs a backward pass |
| - | attention entropy of a head | not built: the sink and the length set it more than the query |

## Along what zones grow

The distance a zone reaches by. The graph itself is shared: mutual neighbours on the rescaled co-activation distance, joined into one component ([math, section 7](bench-math.md#7-the-distance-between-blocks)).

| Surface | What it follows | Role |
| --- | --- | --- |
| the geodesic of the graph as it is | nothing of the query | the control of shape: a ball |
| the medium M4, harmonic conductance | the query's activity: an edge with a quiet end closes | follows the subgraph |
| the medium M4b, conductance by the jump | the border between active and quiet | follows the subgraph's border |
| the signal's path M3 | the weights: an edge from a block that writes to one that reads | follows the flow through the layers |

In the medium a zone cannot cross a block whose activity is below a threshold set by its reach: its figure is a part of the active component around its center, whatever that shape ([math, section 8](bench-math.md#8-the-shape-of-a-map-what-is-measured)).

## How a zone is cut from the subgraph

| Cut | What a zone is | State |
| --- | --- | --- |
| ball on a surface | every block within $R_i$ along the surface (rule 1) | in the code |
| the hill | the connected component around the center where the excess stays above half the peak's height; the number of zones is the number of such components | computed today only to size the radius |
| a local cluster by diffusion | the set a random walk from the center stays in: low conductance to the rest of the graph (the "free path of a gas" law) | to read and build |

## How far zones grow

- The reach $f$: rule 1 today, $R = fD$ along the surface. On the fixed geodesic the lifted share is not linear in $f$ (above). `LogReach` takes the radius at which a ball growing exponentially covers the share $f$ ([math, section 8](bench-math.md#8-the-shape-of-a-map-what-is-measured)); inverting the coverage curve measured on calibration is the exact form. A share of the weight mass was dropped on 2026-09-17: it presets the memory before the query.
- A strength of its own for each zone: its ceiling $\kappa_i$ from the heads through the projection. The code has it (`ProjectedStrength`); the mechanisms do not use it yet.

## How fast zones grow

Linear from the center to the edge today. Any strictly falling law shows up in the levels as another set of stops (rule 3), so a law is tested through the profile ([math, section 8](bench-math.md#8-the-shape-of-a-map-what-is-measured)). The laws:

- a resistance of the medium - M4 and M4b above;
- attenuation along the path with a threshold below which the signal goes no further - the medium with a cut;
- a free path along the vertices and edges of the graph, as particles in a gas - the diffusion cut above.

## The order of checks

From the simplest to verify: the coverage measure itself; the smoothness of the address on the graph (the condition for zones to beat a per-block allocation, [math, section 9](bench-math.md#9-the-best-allocation-knapsack-and-lagrangian-new)); the per-block control; static zones; topic zones; the medium; the signal's path; the laws of growth not yet built; combinations of the above - topic centers with the query's medium, the medium on the signal's graph; the profile of the rings by sensitivity. Only what passes the checks above goes to answers, against the uniform ladder at the same memory.
