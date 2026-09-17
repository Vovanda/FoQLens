---
title: Hypotheses
---

# Hypotheses

The claims FoQLens tests, each with the experiments that test it. The text of every hypothesis and its predictions lives in the preregistration ([PREREGISTRATION.ru.md](../prereg/PREREGISTRATION.ru.md), English: [PREREGISTRATION.md](../prereg/PREREGISTRATION.md)) and in the addenda of the experiments; this page only gives them ids (assigned 2026-09-12) and tracks their status.

Each hypothesis is tested at its step of the roadmap ([goals.md](goals.md)); the steps done so far built the bench, the corpus and the baseline of uniform quantization. A step in the table below is numbered as in the preregistration; the roadmap is in [goals.md](goals.md).

The list is a starting point: the hypotheses tested next are stated after the exploration ([invariants.md](invariants.md)) and preregistered before their runs.

| Id | Hypothesis | Where fixed | Type | Status | Experiments |
| --- | --- | --- | --- | --- | --- |
| H0 | Topics separate in the model's representations | preregistration, step 0 | fitness check | **not tested** | - |
| H1 | Per-block masks are separable by topic and concentrated | preregistration, step 1; property 4 | **load-bearing** | **not tested** | - |
| H1.1 | When the same query takes another form, the mask moves only slightly: the answer stays in the zone of raised quality | found 2026-09-13 in the shuffle check | part of H1 | **not tested** | - |
| H2 | Zones of related topics overlap more than of unrelated ones | preregistration, step 2; property 2 | refining | **not tested** | - |
| H2+ | Mask geometry: not additive, a junction zone, isthmuses with a function, hierarchy, a non-linear representation → mask map | preregistration, step 2+; properties 1, 3, 5, 6, 7 | refining | **not tested** | - |
| H3 | Precision laid out by the query's mask beats uniform quantization and a random mask of the same concentration at the same memory; the random-mask arm is dropped, 2026-09-13 ([plan](plan.md)) | preregistration, step 3 | core: what H4 stands on | **not tested** | - |
| H3.1 | The address: the query's own topic beats the paired topic's at the same memory | preregistration, step 3 | part of H3 | **not tested** | - |
| H3.2 | The shape: own expert zones beat random zones of the same count and size | preregistration, step 3 | part of H3 | **dropped 2026-09-13** with the random arm; the shape question is H3.4 | - |
| H3.3 | Static importance: generic block importance beats random blocks at the same memory | preregistration, step 3 | context for H3; its control is random blocks, not a baseline | **not tested** | - |
| H3.4 | The place: own zones beat the same figure carried elsewhere in the network, at the same cost | found 2026-09-13 | part of H3 | **not tested** | - |
| H4 | **Main.** A draft read mostly at base precision, then refined with sharper zones, ends better than the same model at native precision - where there are iterations: an agent or a model's reasoning. Full text below | author, 2026-09-12; restated by the author 2026-09-13 | main | **not tested**; its premise - the coarse model answers where the precise one refuses - came up in E001 | E001 |
| H5 | **Horizon.** A network trained with zoning and read with FoQZones beats a Mixture of Experts trained the classical way on the same data, holding no more in memory at any moment. Full text below | author, 2026-09-12; restated by the author 2026-09-13 | horizon | **not tested** | - |
| H6 | **Additional.** How good asking for a shorter answer and coarsening the weights each are at representing knowledge in compressed form, and below which step coarsening slides into nonsense. Full text below | author, 2026-09-13, from the author's article | additional | **not tested** | E001 |

## H4, H5 and H6 as stated by the author on 2026-09-13

### H4 - a draft, then a refinement

On a hard question the zones of the first pass may land imprecisely, and most of the answer is computed on weights at base precision. The first answer is then approximate: a draft guess at what the result should look like, and not necessarily a bad one. Such an answer is more useful than a refusal: a guess can be refined, "I don't know" cannot. The first answer goes back into the input of the next step together with the refinement, and the mask of the next step is taken with it. So the gain is expected where such iterations exist: in an agent system, or in a model's reasoning, which takes the same path inside one answer. Once the question is refined, the zones of the next step land more precisely, and the weights outside them, read at base precision, affect the answer less than at native precision. Hence the answer to the refined question is better than that of the same model at native precision.

**The author's premise.** I believe a model holds more knowledge than it shows at full precision. Part of it goes unrealized: the weights lead generation to other, more probable answers, and it never reaches the right one. Coarsening shifts those probabilities and slightly merges close meanings: distinctions the model keeps apart at native precision partly coincide at base precision. The model stops drifting into details, it gains room for an abstract answer, and generation sometimes reaches an answer that was already in the weights. Coarsening creates no new knowledge. The first sign is a trend on the questions bf16 does not know: the judge grades Correct or Nearly 8.1% of bf16's answers, 11.0% of D8's, 12.7% of D6's, 16.4% of D4's ([E001](../experiments/E001-uniform-quantization/results.md)) and 22.2% of the new D2's ([E002](../experiments/E002-base-precision-d2/_index.md)).

Hence the confidence that an answer will come. The address of the zones comes from the coarse model: the first layers are computed coarsely and decide where precision goes, and in the mode where only the zones are loaded there is no other way - the weights outside them are not in memory. In the coarse representations meanings have already merged, so the zone of a question covers the weights of the needed topic together with everything merged with it. When the regulator raises precision in the zone to native, all of it is restored at full quality, and as the question is refined the weights outside the zones weigh on the answer less and pull it aside less. Unneeded topics are in the zone too. That is a fair deal: the main saving is what lies outside the zones, and the surplus may even help the answer.

To check: whether the judge errs on unknown questions; the same trend on more questions in a separate experiment; whether the zone from the coarse pass covers the zone of the same question found at native precision.

**A consequence for the shape of the filter.** For a draft to come out at all and not go into detail, each zone needs a strength of its own. On a question the model knows poorly, the zones stay weak and their centers are read below the top rung, at D6 for example. Once the question is refined, the zones it needs grow stronger. How the strength of a zone is set is open.

**Test.** An agent chain - guess, refinement, next step - on a hard multi-step task, such as designing a software architecture. The same chains run on the zone model, on the same model at native precision, and on uniform quantization at the same memory. The first answer is not scored on its own; the result of the chain and its cost are.

**What it leans on.** [LASER](https://arxiv.org/abs/2312.13558) - replacing selected late MLP weight matrices of a trained model by their low-rank approximation raises answer accuracy by 20-30 points on some tasks. [SparseGPT](https://arxiv.org/abs/2301.00774) - at the largest scales, 50% sparsity costs nothing in perplexity and slightly raises accuracy. [MoBiQuant](https://arxiv.org/abs/2602.20191), §3 - lower precision sometimes scores higher ([reading notes](reading-notes.md)). The premise above is the author's and cites nothing.

### H5 - a network trained for zones

A model trained with zoning, read with FoQZones at inference, works better than a Mixture of Experts trained the classical way on the same data split by area, and at any moment holds no more in memory. In training, a rule is built in by which basic things accumulate in neighbouring places of the weights and the knowledge of each area settles into its own zone. In an MoE an expert owns its parameters entirely, and each expert stores the common base of related areas again. In the zoned model the basics lie in one common place once, the zones of areas overlap, and the junction of two areas gets precision without a separate mechanism.

**A necessary condition.** FoQZones can drop zeros: at a ZERO base the weights outside the zones are not loaded. Only then does the model hold in memory, at any moment, just the zones of the query and the common base, and cost no more than the MoE. Today ZERO only zeroes the output of a block while its weights stay in memory, and loading only the zones needs the address before the weights, from the first layers.

**How to get it - a sketch.** The starting layout of zones comes from a trained network in which the metric is guessed: how many zones each area has, how large they are and how they overlap - their structure, not their places, since block N of a new network is not block N of the old one. A new network is trained from scratch on data split by area. The zone of a sample's own area is read precisely and the rest at base precision, as at inference, and a smoothness loss over the map, as in TopoLM, pulls what is common into common places. The map is taken from the trained network, and the next network is trained under it, until the layout stops changing.

**Reproducibility.** Tested on a small transformer trained from scratch: two or three areas, both models on the same data, the same total number of parameters. Compared are quality per area and at the junctions, and the memory loaded per query.

**What it leans on.** The [problem statement](problem-statement.md): overlapping zones, the shared part stored once. [Emergent Modularity](https://arxiv.org/abs/2305.18390) - neurons group into functional experts during pretraining. [MoEfication](https://arxiv.org/abs/2110.01786) and [EMoE](https://arxiv.org/abs/2310.10908) - co-activation groups of a dense model become experts with hard edges. [TopoLM](https://arxiv.org/abs/2410.11516) - a smoothness loss over a map pulls similar units into neighbouring places. [DEMix](https://arxiv.org/abs/2108.05036) - experts by domain label, which is what training with a hard zero outside the zone reduces to. The classical MoE it is compared with: [Shazeer et al.](https://arxiv.org/abs/1701.06538), [Switch Transformer](https://arxiv.org/abs/2101.03961).

### H6 - internal compression of knowledge

An additional hypothesis: if it fails, the others and the bench stand as they are. The subject is how a model holds knowledge in its weights and unfolds it into an answer at the depth asked for - from the weights alone, with no source in the context. Knowledge can be unfolded more coarsely in two ways: by asking the model for a shorter answer, or by coarsening its weights. The aim is to find out how good each way is at representing knowledge in compressed form: what it costs at each step of coarsening, whether the cost can be neglected, and whether this holds at every step or below some step a slide into nonsense cannot be avoided. It comes from the author's article [«Квантование - всё, что вам нужно»](https://sawking.tech/blog/kvantovaniie-vsio-chto-vam-nuzhno) (in Russian).

**The author's assumptions.**

1. A coarsened model unfolds knowledge worse than the native one asked to be brief: at the same length it keeps what matters less often and distorts what it keeps more often.
2. Coarsening is a destructive operation, done blindly: each weight is rounded independently of the rest.
3. Merges of meanings under coarsening are unavoidable and uncontrolled: words from different places of the model fuse into one unit of meaning that nobody put there. A false link - say, "dog" sliding towards "neuroscience" - spoils an answer.
4. Some merges may turn out to be implicit bridges between areas. Bridges add no knowledge: coarsening only loses information. But at the junction of areas, where the hardest questions lie, a coarsened model may fall behind the native one in answer quality less than within a single area, and reach non-obvious solutions through these implicit links.

**Bets.**

- *The author.* A compressed model can be used without fearing that coarsening is bound to spoil its answers. Merges have useful applied sides: on hard interdisciplinary questions the bridges between areas outweigh the downsides of quantization, and the coarsened model stays fit for them. The level of abstraction of an answer can be steered by directed quantization: the regulator decides in which areas and how far to coarsen the model; which meanings merge as a result, nobody decides.
- *Claude.* At D6 the damage disappears into noise: perplexity 12.83 against 12.70 at native precision. At D4 it does not: 16% more perplexity is a systematic cost. Inside a coarsened area rounding is blind, so the loss at the junction of areas will not be smaller than within a single area.
- *How it is settled.* The drop in quality from native precision at D6 and D4, separately for junction questions and single-area questions, checked fact by fact. Smaller at the junction - the author is right; not smaller - Claude is.
- *The author's hope.* A compressed model may reach another level of emergence: drawing true links between areas that the native model does not draw in its answer. The sign: such links in the coarsened model's answers, checked against facts, come up more often than false ones.

**How each way is assessed.** The reference is the full answer of the native model, broken into essential facts. Every compressed answer is checked against it on the two axes of summarization: coverage - the share of essential facts kept - and faithfulness - whether the kept ones are distorted. For the request this is a curve over answer length, for coarsening a curve over bits per weight. Coarsening runs in uniform steps (D8, D6, D4, D2) as the baseline, and directed, by the zones of the query, at the same memory.

The request for a shorter answer takes two forms:

- *a ladder* - "the same answer, each row shorter": one prompt gives 6-10 compressions at once; the model gets no word count and cuts by itself;
- *a direct limit* - "answer in n words": answers in parallel with the limit n and tighter ones, several per limit, until at least one hits the length needed; the pick is by length only, never by content.

Which form keeps more of what matters at the same length is an open question inside H6.

**How the two ways are compared.** The coarsened model answers several times; the median length of its answers, L words, is the target. It is compared with an answer to the request of the same length - a row of the ladder or an answer under the limit. The comparison runs on three levels: which facts each way kept, dropped and distorted; how close the answers themselves are (token-level KL in both directions); how close the model's internal states are on one and the same answer, layer by layer, on the answer tokens only. A correct compressed answer is not unique: if the full answer is k independent facts and j of them fit, there are C(k, j) correct variants, and the more independent details a question has, the more "roughly correct" answers there are. So the agreement of the two ways on the facts they keep is compared with 1 / C(k, j).

**Question groups.** For each pair of areas, three groups: the junction (say, the neurobiology of dogs), the first area without the second, and the second without the first. A mixed domain has to be built: the hand-written biophysics set came out biology-like. Separately, a measurement of the merges themselves that needs neither corpus nor judge: which concepts change their nearest neighbours in the hidden states after coarsening, and how close the merged pairs are by the native model, compared with random pairs.

**What each outcome gives.**

- *Coarsening is no worse than the request.* Base precision carries the substance of an answer, the draft of H4 is a full summary, and weight precision is a knob for abstraction that saves memory besides. The request "answer briefly" gives no advantage, and there is no reason to hope it keeps information better.
- *Coarsening is better than the request.* At the same length the coarsened model keeps what matters more often; for a summary it pays to coarsen the model rather than ask it to be brief.
- *The request is better.* Coarsening loses what the model would have kept itself: base precision carries less substance, the zones have to cover more, and the lost facts show what to keep in them.
- *A breakdown below step X.* That is the lower bound of the regulator: a base below X is ruled out whatever the zones.
- *The loss at the junction is smaller than within an area.* The bridges exist, and for interdisciplinary questions the coarsened model is fit for use with all its memory saving.

**A condition on the bench.** The model has to follow a request to compress its answer. In a one-off probe of 2026-09-13 base E2B did not: it kept a word limit on 2 questions of 12 and built the ladder on none. H6 needs the `-it` checkpoint; the corpus is selected on it too (UPD 2026-09-15 in the [preregistration](../prereg/PREREGISTRATION.md)), and what needs no instructions stays on the base one.

**Aside, not a priority.** Summarizing a text given in the context - direct compression - can be checked alongside, with the same fact-by-fact comparison.

**What it leans on.** The author's article above. Faithfulness of summaries: [Maynez et al.](https://arxiv.org/abs/2005.00661). Summarization by instruction and the unreliability of automatic judges: [arXiv 2311.09184](https://arxiv.org/abs/2311.09184). Length control: [arXiv 2501.00233](https://arxiv.org/abs/2501.00233). Where information sits in the context: [Lost in the Middle](https://arxiv.org/abs/2307.03172).
