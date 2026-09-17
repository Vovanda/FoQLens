---
title: Zone strategies
---

# Zone strategies

The level rules in [quantization-filter.md](quantization-filter.md) do not say where zones grow from, along what, how fast or how far. Each of these is a replaceable strategy of its own. Below are the variants I want to build and test, in order of priority. None is tested on answer quality yet: the runs decide the laws and their parameters.

## Where zones grow from

A source scores how much the query needs a block, and its peaks become the centers of the zones. A working source is computed in one forward pass over the first layers at base precision: at a ZERO base the address has to be known before the weights are loaded.

| Priority | Source | Why |
| --- | --- | --- |
| 1 | Output energy of a block; MLP neuron activity at the input of `down_proj` | forward pass, the amplitude of the signal |
| 2 | Output energy of a head at the input of `o_proj`, carried onto the blocks | forward pass |
| 3 | Taylor | for comparison only: it needs a backward pass |
| 4 | Attention entropy of a head against its background | the sign is unclear, the entropy depends on the input length |

## Along what zones grow

The distance `d` between blocks: what makes two blocks neighbours.

| Priority | Metric | Why |
| --- | --- | --- |
| 1 | A nearest-neighbour graph of block co-activations, and the path along it | the data exists, a zone follows the graph |
| 2 | Coupling through the weights: an edge from a block that writes to the stream to a block that reads from it | the path of the signal in the model itself; very many edges |

The naive variant: the first two principal axes (PCA) of co-activation, with zones found as hills on a flat picture. The bench started with it. How much of the distance the two axes keep is not measured.

## How far zones grow

| Priority | What | Why |
| --- | --- | --- |
| 1 | A zone covers a share `f` of the network: 0 - the center only, 1 - the whole network | the control is linear and needs no found radius |
| 2 | A strength of its own for each zone: on a question the model knows poorly the center is read below the ceiling | the main hypothesis needs it |

## How fast zones grow

Today precision falls off linearly from the center to the edge. Variants of the law:

- as particles in a gas: a free path along the vertices and edges of the block graph;
- as a signal along neurons: attenuation along the way and a threshold below which the signal goes no further;
- a resistance of the medium: the conductance of a link depends on the query's activity at its ends.
