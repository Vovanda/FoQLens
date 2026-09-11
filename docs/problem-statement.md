# Problem statement - expert zones as an outcome, not an input

The precision-controller problem developed into an architectural branch. Assembled 2026-09-10, moved into FoQLens 2026-09-11. Based on the author's private note on the precision controller (not included in the repo).

**Status: a coherent problem statement with two named holes, not a solution.** Both holes are hard research questions, not implementation details.

## What differs from everything that exists

All known work quantizes by **units fixed in advance**: layer, channel, token, expert. The boundaries are fixed by the architecture; the controller only picks the bit depth inside them.

Here **the boundary is not given**. Precision is assigned to a region in weight space, and **the expert is not carved out in advance but emerges** as what stays sharp when everything around it is coarsened. An expert zone is the result of how precision is distributed, not a router's input.

This removes the need for a router rather than replacing it with a better one.

## A magnifier, not two levels

Quantization should be **non-uniform and directed**: sharp at the center, coarser toward the periphery, with a transition region around the sharpened zone.

A hard boundary would give a cliff at the junction - exactly what MoE is criticized for. A gradient of sharpness gives **bridges between zones** without a separate mechanism.

**Caveat (2026-09-11):** a smooth falloff is not a guarantee but a subject of measurement. Quality may drop in steps by kind of knowledge rather than smoothly.

## Zones overlapping themselves - the main argument for compactness

In MoE an expert owns its parameters entirely, there is no overlap: two related skills pay for their common foundation twice.

If zones overlap, the shared part is stored once and kept precise for both. **Compactness comes not from compression but from refusing to duplicate.**

The same explains the bridges: you can move from poetry to analysis because the zones share a piece, not because someone connected two experts.

## How the shape develops

Start with a parameterized oval → arbitrary ovals → arbitrary shapes with bridges.

Important: the shape is given by **a dozen numbers, not a map over every block**. Otherwise there would be more control parameters than weights.

## The model defines the shape itself

A fork to keep in mind: **learning the mask directly does not work** - bit depth is discrete, no gradient flows through it. Two ways around:

1. **A soft mask** - precision is continuous during training and rounded at inference. Closer to the magnifier: a continuous mask is exactly the gradient of sharpness.
2. **A shape predictor** from the input, trained on the final quality.

A side effect of the first option works in favor of the statement: if the mask is continuous and learnable, **zones are not assigned but converge by themselves** - overlaps appear where they pay off, not where they were drawn.

## Coarsening cuts precision, not structure

Example: "how many fingers does a person have". What is needed is not just a coarse answer but a coarse answer **with the caveat about exceptions preserved**: 20 is the center of the funnel, genetic variations are its edge, and the edge must survive.

The argument against uniform coarsening: it removes the tail of the distribution first, because the tail is stored thinly. The naive "simple question - cut everything" gives a confident "twenty" with no edge.

A directed magnifier avoids this: the biology zone is sharp, the rest is coarse, the tail inside the zone is intact.

## Two jobs for the controller, not one

From the same example: "how many fingers" and "prove the theorem" need not a different degree of sharpness but **sharpness in different places**.

So the controller decides:
- **how much** to sharpen - the scale;
- **where** to sharpen - the address.

Existing work has only the first.

## Where the address comes from

The key link: **the address is already computed by the model itself**. The activations of the first layers effectively say what the query is about - this signal exists in the pass. Contextual sparsity already reads it to skip neurons (GRIFFIN, CoreInfer - [prior art](prior-art.md)); no known work reads it to lay out precision over the weights.

Scheme: run the input through a few first layers at coarse precision → from the intermediate representation, get where to point the magnifier → compute the remaining layers with that mask.

**Estimate first, sharpen later.** This is literally the way of thinking from ["Don't overthink"](https://sawking.tech/blog/ne-nado-dumat-lishnego) (in Russian): step back, look at the whole coarsely, then look closely where it matters.

Consequence: there is no separate controller to train - the mask is derived from what the model has already computed. The controller becomes a function of the intermediate state.

## Two holes - the subject of research

1. **In what space are the regions defined.** By what metric are weights "close"? Adjacency by index in a matrix means nothing. Quantization today works on tensors and groups with a shared scale per group - there is simply no structure that describes an arbitrary connected shape.
2. **How to map a representation into a mask over weights.** The address is needed in weight space; activations live in representation space. The transition between them is undefined.

The second hole narrows the first: not "in what space are regions at all" but "how to get a region from a representation". That is progress in the statement, not its closure.
