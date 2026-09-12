---
title: Hypotheses
---

# Hypotheses

The claims FoQLens tests, each with the experiments that test it. The text of every hypothesis and its predictions lives in the preregistration ([PREREGISTRATION.ru.md](../prereg/PREREGISTRATION.ru.md), English: [PREREGISTRATION.md](../prereg/PREREGISTRATION.md)) and in the addenda of the experiments; this page only gives them ids (assigned 2026-09-12) and tracks their status.

**As of 2026-09-13 every hypothesis is reset to untested, without exception.**

The bench measured quality by which of four letters the model ranks highest, on a set it answers by
guessing more than half the time, with a mask taken from the gradient of the language-model loss on
the prompt - that is, from what the text activates, not from what the answer needs. No verdict of any
sign can rest on that, so none is kept: the favourable readings, the damning ones and the instrument
checks all go back to untested together.

**And the list itself is not fixed.** It was written before the bench could measure anything, and the exploration that comes now - what the model is competent at, what a mask actually tracks, what the regulator's settings do to a usable answer - is expected to produce different questions than these. New ones are expected to sharpen the model of the mechanism rather than repeat these: the exploration is looking for invariants ([invariants.md](invariants.md)), and a hypothesis is worth fixing when it is a claim about something that already looks invariant. The hypotheses below are kept as history and as a starting point; the ones that get tested will be stated after the exploration, together with the experiments that test them, and preregistered in the usual way before any of them is run.

Re-testing starts from data and a bench that can be trusted. What has to be true of the corpus and the metric before anything is claimed
again: [docs/corpus.md](corpus.md).

| Id | Hypothesis | Where fixed | Type | Status | Experiments |
| --- | --- | --- | --- | --- | --- |
| H0 | Topics separate in the model's representations | preregistration, step 0 | fitness check | **not tested** | E001 |
| H1 | Per-block masks are separable by topic and concentrated | preregistration, step 1; property 4 | **load-bearing** | **not tested** | E001, E002 |
| H1.1 | The address is the meaning of the query, not the surface form of the prompt | found 2026-09-13 in the shuffle check | part of H1 | **not tested** | E013 |
| H2 | Zones of related topics overlap more than of unrelated ones | preregistration, step 2; property 2 | refining | **not tested** | E001 |
| H2+ | Mask geometry: not additive, a junction zone, isthmuses with a function, hierarchy, a non-linear representation → mask map | preregistration, step 2+; properties 1, 3, 5, 6, 7 | refining | **not tested** | E001 |
| H3 | Precision laid out by the query's mask beats uniform quantization and a random mask of the same concentration at the same memory | preregistration, step 3 | main | **not tested** | E003-E013 |
| H3.1 | The address: the query's own topic beats the paired topic's at the same memory | ADDENDUM-04 I1, -05 B2, -06 W1, -07 S2, -09 M2, -11 L2 | part of H3 | **not tested** | E004, E005, E007, E008, E009, E010 |
| H3.2 | The shape: own expert zones beat random zones of the same count and size | ADDENDUM-07 S1, -09 M1, -11 L1 | part of H3 | **not tested** | E008, E009, E010 |
| H3.3 | Static importance: generic block importance beats random blocks at the same memory | ADDENDUM-05 B1 | context for H3 | **not tested** | E005 |
| H3.4 | The place: own zones beat the same figure carried elsewhere on the map, at the same cost | ADDENDUM-15, ADDENDUM-16 P3 | part of H3 | **not tested** | E014, E015 |
| H4 | **Main.** Emergent: a model read through the query's lenses is more efficient and more accurate than the original bf16 model - the glass damps what does not belong to the query, the isthmus sharpens the junction of topics; for an agent that reasons over many steps the effect adds up. Test: an agent on a hard multi-step task (designing a software architecture, for example), lens model against the same model in bf16, once the FoQLens model exists | author, 2026-09-12; a single-pass part fixed in ADDENDUM-11 | main | **not tested** | E010 |
| H5 | **Stronger than H4.** The lens model is more accurate and more efficient than a Mixture of Experts made from the same model: MoE experts have hard edges, the junction of two topics falls between them and a router error costs a whole expert; lenses are continuous, shared by related topics, and sharpen the junction. Test on the bench: the MoE end of the lens scale - opaque glass, hard-edged lenses on fixed clusters of the weight map, the top-k clusters chosen by the query's mask - against the query's lenses at the same memory; in full on the agent task of H4 | author, 2026-09-12; the bench part comes after the lenses work | main | **not tested** | - |

Engineering results that test no hypothesis (for example E006, read depths from one sliced copy) carry no hypothesis id.
