---
title: Hypotheses
---

# Hypotheses

The claims FoQLens tests, each with the experiments that test it. The text of every hypothesis and its predictions lives in the preregistration ([PREREGISTRATION.ru.md](../prereg/PREREGISTRATION.ru.md), English: [PREREGISTRATION.md](../prereg/PREREGISTRATION.md)) and in the addenda of the experiments; this page only gives them ids (assigned 2026-09-12) and tracks their status. Experiments list their hypotheses in the `hypotheses` field of their card ([index](../experiments/_index.md)).

| Id | Hypothesis | Where fixed | Type | Status | Experiments |
| --- | --- | --- | --- | --- | --- |
| H0 | Topics separate in the model's representations | preregistration, step 0 | fitness check | **supported** on E2B (ARI 0.98) | E001 |
| H1 | Per-block masks are separable by topic and concentrated | preregistration, step 1; property 4 | **load-bearing** | **not confirmed** with the naive score (held-out pair fails); the gradient instrument was never checked on its own | E001, E002 |
| H2 | Zones of related topics overlap more than of unrelated ones | preregistration, step 2; property 2 | refining | exploration passed, confirmation failed | E001 |
| H2+ | Mask geometry: not additive, a junction zone, isthmuses with a function, hierarchy, a non-linear representation → mask map | preregistration, step 2+; properties 1, 3, 5, 6, 7 | refining | mixed in exploration; the mixed domain came out biology-like | E001 |
| H3 | Precision laid out by the query's mask beats uniform quantization and a random mask of the same concentration at the same memory | preregistration, step 3 | main | **open** | E003-E009 |
| H3.1 | The address: the query's own topic beats the paired topic's at the same memory | ADDENDUM-04 I1, -05 B2, -06 W1, -07 S2, -09 M2 | part of H3 | flat masks: not supported; expert zones (legacy fixed budget): biology-math yes, history-geography no | E004, E005, E007, E008, E009 |
| H3.2 | The shape: own expert zones beat random zones of the same count and size | ADDENDUM-07 S1, -09 M1 | part of H3 | holds in the legacy fixed-budget layout (both pairs in E009) | E008, E009 |
| H3.3 | Static importance: generic block importance beats random blocks at the same memory | ADDENDUM-05 B1 | context for H3 | **supported**, large | E005 |
| H4 | **Main.** Emergent: a model read through the query's lenses is more efficient and more accurate than the original bf16 model - the glass damps what does not belong to the query, the isthmus sharpens the junction of topics; for an agent that reasons over many steps the effect adds up. Test: an agent on a hard multi-step task (designing a software architecture, for example), lens model against the same model in bf16, once the FoQLens model exists | author, 2026-09-12; a single-pass part fixed in ADDENDUM-11 | main | **open**. Support from the literature: removing higher-order components of weight matrices can improve reasoning (LASER, Sharma et al., 2023, arXiv 2312.13558). Exploratory look at E009 (legacy fixed budget, frosted glass, one forward pass): no layout beats bf16 - at 6-7 bits within noise (up to +0.008), below it at 4.5-5 bits | E010 |

| H5 | **Stronger than H4.** The lens model is more accurate and more efficient than a Mixture of Experts made from the same model: MoE experts have hard edges, the junction of two topics falls between them and a router error costs a whole expert; lenses are continuous, shared by related topics, and sharpen the junction. Test on the bench: the MoE end of the lens scale - opaque glass, hard-edged lenses on fixed clusters of the weight map, the top-k clusters chosen by the query's mask - against the query's lenses at the same memory; in full on the agent task of H4 | author, 2026-09-12; the bench part to be fixed in ADDENDUM-11 | main | **open** | E010 |

Engineering results that test no hypothesis (for example E006, read depths from one sliced copy) carry no hypothesis id.
