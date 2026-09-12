---
title: Experiments
---

# Experiments

One folder per experiment, `E0NN-slug`: its preregistration (addenda, thresholds), its card (`_index.md`: dates, commits, hypotheses, status, verdict), and `results.md` once it has run. Raw summaries are in `runs/E0NN-slug/`. The main preregistration stays in [`prereg/`](../prereg/); the hypotheses are in [docs/hypotheses.md](../docs/hypotheses.md).

Status: `planned` - prereg in progress; `fixed` - prereg committed, not run; `done`; `legacy` - run, but on an approach since replaced; `not-run` - fixed and never run on its own.

| Id | Experiment | Fixed | Hypotheses | Status | Verdict |
| --- | --- | --- | --- | --- | --- |
| [E001](E001-run1-exploration/_index.md) | Run 1: separation, masks, overlap, geometry, confirmation | 2026-09-11 | H0, H1, H2, H2+ | done | H0 yes; H1 and H2 pass in exploration, fail on the held-out pair |
| [E002](E002-gradient-score/_index.md) | Gradient x activation as the block score | 2026-09-11 | H1 | not-run | never checked on its own; used as a mask source from E004 on |
| [E003](E003-aperture-sweep/_index.md) | Step 3 as an aperture sweep | 2026-09-11 | H3 | not-run | replaced by E004 and E005 |
| [E004](E004-injection/_index.md) | Mask injection: topic A's mask on topic B's questions | 2026-09-11 | H3.1 | done | own does not beat other |
| [E005](E005-backbone/_index.md) | Backbone + topic fill | 2026-09-11 | H3.1, H3.3 | done | importance carries the budget; the topic fill adds nothing |
| [E006](E006-read-depths/_index.md) | Read depths from one sliced copy, resident bench | - | - | done | D8 as good as int8; 1.63 GiB freed on E2B |
| [E007](E007-dilation/_index.md) | Dilation of the topic fill | 2026-09-11 | H3.1 | done | structural groups are real; no address |
| [E008](E008-zones-fixed-budget/_index.md) | Expert zones at a fixed 5-bit budget | 2026-09-11 | H3.1, H3.2 | legacy | address for biology-math only |
| [E009](E009-zones-matrix/_index.md) | Expert zones over precision share x focus area | 2026-09-11 | H3.1, H3.2 | legacy | M1-M3 for biology-math, M1 for history-geography |
| [E010](E010-lens-layout/_index.md) | Lens layout: a floor over the network, zones lifted over it ([docs/lens.md](../docs/lens.md)) | 2026-09-12 | H3, H3.1, H3.2, H4 | done | own zones beat generic importance on both pairs; not the same memory without a mask |
| [E011](E011-depth-caps/_index.md) | Depth caps: what storage costs when many queries share one copy | 2026-09-12 | - | done | caps pay off only for a narrow profile: one topic stores 4.0 bits of 8, 395 mixed questions 6.5 |
| [E012](E012-address-edges/_index.md) | The edges of the address: a scan behind an empty floor | 2026-09-12 | H3.1, H3.2 | not-run | the question moved to E013: random zones spend more memory, so they are not an honest reference |
| [E013](E013-regulator-map/_index.md) | The map of the regulator: which settings cost the fewest errors | 2026-09-12 | H3 | fixed | - |
