---
title: Experiments
---

# Experiments

One folder per experiment, `E0NN-slug`: its preregistration (addenda, thresholds), its card (`_index.md`: dates, commits, hypotheses, status, verdict), and `results.md` once it has run. The main preregistration stays in [`prereg/`](../prereg/); the hypotheses are in [docs/hypotheses.md](../docs/hypotheses.md).

**Reset 2026-09-13.** The corpus could not carry a verdict: the core - questions bf16 gets right in
every order of the options - was 31%, and 3% on maths. The masks were read from the same questions,
the weight map from those masks, the zones from that map, so no link of the chain could carry a verdict.

Deleted: the result texts and the runs of E004, E005, E007-E011, E013, E014, and the never-run
E002, E003, E012, E015. Kept: the preregistrations and the code. Next: a corpus the model knows and
a metric that does not reward a lucky letter ([docs/corpus.md](../docs/corpus.md)).

Status: `planned` - prereg in progress; `fixed` - prereg committed, not run; `done`; `exploratory` - run without a preregistration, so it can point a direction but not settle one; `legacy` - run, but on an approach since replaced; `not-run` - fixed and never run on its own; `withdrawn` - run, but its verdict does not stand; `reset` - run on the question set of the reset, nothing is claimed from it; `deleted` - prepared on that set and deleted with it, kept in git history.

| Id | Experiment | Fixed | Hypotheses | Status |
| --- | --- | --- | --- | --- |
| [E001](E001-run1-exploration/_index.md) | Run 1: separation, masks, overlap, geometry, confirmation | 2026-09-11 | H0, H1, H2, H2+ | reset |
| E002 | Gradient x activation as the block score | 2026-09-11 | H1 | deleted |
| E003 | Step 3 as an aperture sweep | 2026-09-11 | H3 | deleted |
| [E004](E004-injection/_index.md) | Mask injection: topic A's mask on topic B's questions | 2026-09-11 | H3.1 | reset |
| [E005](E005-backbone/_index.md) | Backbone + topic fill | 2026-09-11 | H3.1, H3.3 | reset |
| [E006](E006-read-depths/_index.md) | Read depths from one sliced copy, resident bench | - | - | done |
| [E007](E007-dilation/_index.md) | Dilation of the topic fill | 2026-09-11 | H3.1 | reset |
| [E008](E008-zones-fixed-budget/_index.md) | Expert zones at a fixed 5-bit budget | 2026-09-11 | H3.1, H3.2 | reset |
| [E009](E009-zones-matrix/_index.md) | Expert zones over precision share x focus area | 2026-09-11 | H3.1, H3.2 | reset |
| [E010](E010-lens-layout/_index.md) | Lens layout: a floor over the network, zones lifted over it ([docs/lens.md](../docs/lens.md)) | 2026-09-12 | H3, H3.1, H3.2, H4 | reset |
| [E011](E011-depth-caps/_index.md) | Depth caps: what storage costs when many queries share one copy | 2026-09-12 | - | reset |
| E012 | The edges of the address: a scan behind an empty floor | 2026-09-12 | H3.1, H3.2 | deleted |
| [E013](E013-regulator-map/_index.md) | The map of the regulator: which settings cost the fewest errors | 2026-09-12 | H3 | reset |
| [E014](E014-moved-zones/_index.md) | The zones carried elsewhere: the honest control at the same cost | - | H3, H3.4 | reset |
| E015 | Three topic pairs the bench has not used | 2026-09-13 | H3, H3.4, H4 | deleted |
