# FoQLens

Hi, I'm Vladimir Savkin. I hold master's degrees in mathematical software and in fundamental computer science, and I work as a systems architect - distributed systems, lately with formal verification. FoQLens is my research, and it is a hobby: I am not a professional scientist. That is why the bench keeps me honest - every prediction is committed before the run that tests it. More about me: [CV](https://sawking.tech/cv).

**Can a language model spend its precision where the question is, instead of everywhere?**

**[vovanda.github.io/FoQLens](https://vovanda.github.io/FoQLens/)** - the idea, with the controls of the
regulator to move: base precision, the size and strength of the zones, how overlaps combine, and what
the setting costs in bits per weight. The metaphor they act on is drawn on real weights: the map of
Gemma 4 E2B, 14 708 blocks, blocks that light up together lying close.
[The documents](https://vovanda.github.io/FoQLens/docs.html) - the mechanism, the preregistration and
every run - are on the same site.

The larger goal is a universal **precision regulator** - one mechanism that sets how finely a model works right now and makes it adaptive: to the task, to the machine it runs on and to the value of the query ([the idea](https://sawking.tech/blog/kvantovaniie-vsio-chto-vam-nuzhno)). One set of weights serves every device and every load: it runs lean on a phone or on a hot, busy server, and opens to full precision exactly where a query needs it. Under pressure it degrades gracefully - the background coarsens first, what the query needs stays sharp.

Mixture of Experts is its rigid special case: experts with hard edges fixed at training, opened by a router. The regulator makes experts continuous - zones emerge from the query itself, related topics share them, and the junction between two topics is sharpened instead of falling between two experts.

**The main hypothesis:** on a hard question the first pass gives a draft, most of it read at base precision - an approximate guess, and more useful than a refusal, since a guess can be refined. The draft goes back into the input with the refinement; the zones of the next step land more precisely, and the weights outside them affect the answer less. So where there are iterations - an agent, or a model's reasoning - the model should end better than the same model at native precision. It is tested once the FoQLens model exists: an agent on a hard multi-step task, such as designing a software architecture ([H4](docs/hypotheses.md)). Further out, a horizon: a network trained with zoning, read the same way, should beat a Mixture of Experts trained the classical way on the same data, holding no more in memory at any moment ([H5](docs/hypotheses.md)).

**FoQLens** (Focus + Quantization + Lens) is the model that uses this regulator. Its mechanism is **FoQZones** - focusable quantization zones: the precision of the weights is allocated by the meaning of the query, and the lens is its metaphor. This repository is the R&D inside FoQLens - a bench that tests the core of the idea: keep the weights that matter for *this particular query* at high precision and read the rest coarsely - and let the model itself say which weights those are ([goals](docs/goals.md)).

## The problem

Quantization makes models smaller and faster by storing weights with fewer bits. Every existing scheme - static or dynamic - decides precision per **fixed unit**: the whole model, a layer, a channel, a token, an MoE expert. The boundaries come from the architecture; a controller only chooses how many bits each unit gets.

A question about biology and a question about a proof do not need *more or less* precision. They need precision **in different places**.

## The idea

1. Score every block of weights (64 output rows) by how much it matters for the query - from the model's own activations and gradients, no trained router.
2. Place the blocks on a **weight map**, where blocks that light up together lie close. The query's mask has peaks on this map: its **expert zones**.
3. Read the whole network at a **base precision** and read each expert zone more precisely: sharpest at its center, falling off toward its edge.

Three controls, each doing one thing:

- **base precision** (`floor` in the scripts) - the precision of everything outside the zones, down to nothing at all;
- **focus_area** - the size of the zones;
- **focus_strength** - how far the zone centers rise above the base.

Memory is the result of the settings, and no budget is preset: a query that needs little gets small zones and pays little. The mechanism in formulas, the single source of truth for it: [docs/quantization-filter.md](docs/quantization-filter.md).

If the idea holds, **expert zones emerge** as the regions that stay sharp when everything around them is coarsened - and related topics share part of their zone instead of paying for it twice, as MoE experts do.

## Where it leads

- **On-device models.** One weights file for a phone, glasses or a laptop: precision follows the battery, the heat and the free memory, and the zones stay where the query is.
- **Cost per query in data centers.** A simple question gets small zones and costs little; a hard one gets wider ones. The price of a token follows the question.
- **Agents and reasoning.** Wherever a draft is refined step by step - the main hypothesis above.
- **Graceful degradation.** Under load or heat the model gets coarser first in what the query does not need.

These are the directions the regulator opens; each is tested by a step of the [plan](docs/goals.md) before it is claimed.

## What the bench checks

The steps, numbered as in the [preregistration](prereg/), are ordered so each one can kill the next; where each stands is in the [roadmap](docs/goals.md):

| Step | Question | Kills the idea if |
| --- | --- | --- |
| 0 | Do topics separate in the model's representations at all? | they don't (then the model is too weak, not the idea) |
| 1 | Are per-block masks similar within a topic and different between topics? Are they concentrated? | masks look the same for every query |
| 2 | Do related topics (biology-chemistry) share more of their zones than unrelated ones (biology-math)? | no overlap structure |
| 2+ | Geometry: are masks additive, is there a junction zone, does ablating it break mixed questions only? | - (refining, not load-bearing) |
| 3 | How much of what the model knows do the query's zones keep, against uniform quantization at the same memory? The other topic's zones and generic importance test the address | the zones keep no more than uniform quantization at the same memory |
| 7 | In an agent chain of a draft and refinements, does the zone model end better than the same model at native precision and than uniform quantization at the same memory? (the main hypothesis, once the model exists) | - |

All predictions were [preregistered](prereg/) in git before each run, as directions ("A > B"), not numbers. The full reasoning is in [`docs/`](docs/).

## Status

**Every hypothesis is untested.** The first corpus chosen - four MMLU subjects with a letter to pick - was
rejected: the letter does not tell whether the model knows the answer or guesses it
([corpus.md](docs/corpus.md)).

**The corpus of what the model knows is built and frozen:** the full model answered every question of five
datasets in its own words, with no options anywhere, and a question stays if the answer is right; three
regimes in it - the answer in a passage, only in the weights, split across two passages - so that the claim
can fail ([docs/corpus.md](docs/corpus.md)). Next come uniform quantization levels on it as the baseline,
then a mask that is asked about the answer instead of the text. Then the hypotheses, from the first.

**The question all of it serves:** does the regulator work - the structure where a coarse reading is enough, the
details where sharpness is needed, a gain over iterations. Saving memory is a secondary goal, plan B: even without
a gain in quality the mechanism saves memory at the same usability.

**Engineering:** one stored copy of the weights read at 2 / 4 / 6 / 8 bits (residual slices after
MoBiQuant), with no separate bf16 copy, and each block able to store only the depth it is read to. The slices
are still unpacked before the multiplication; a kernel that reads only the bits it needs is written and
tested, and not yet wired into decoding.

## Reproduce

Requirements: an NVIDIA GPU with 16 GB+ of memory, [uv](https://docs.astral.sh/uv/). uv fetches Python 3.12 by itself.

```sh
git clone https://github.com/Vovanda/FoQLens.git
cd FoQLens
uv sync                                         # torch (CUDA 12.8), transformers, bitsandbytes
uv run python scripts/download_models.py        # Gemma 4 E2B at its pinned revision, ~10 GB
uv run pytest                                   # unit + bench health tests on the GPU
```

A run takes 0.8 of the GPU by default - that share of the VRAM, and rest between batches - so the card stays usable; `--gpu-share 1` or `FOQLENS_GPU_SHARE=1` gives the full speed.

Model weights are not stored in the repository. `scripts/download_models.py` fetches them from Hugging Face (Apache 2.0, no token needed) at the commits pinned in `foqlens.model.REVISIONS`, and `foqlens.model.load()` reads exactly those commits. Add `e4b` to the download for the confirmation model (~16 GB).

## Layout

- [`docs/goals.md`](docs/goals.md) - goals by step and their status.
- [`docs/problem-statement.md`](docs/problem-statement.md) - the problem statement: expert zones as an outcome, not an input.
- [`docs/quantization-filter.md`](docs/quantization-filter.md) - the quantization filter and its zones: how precision is laid out over the weights.
- [`docs/hypotheses.md`](docs/hypotheses.md) - the hypotheses under test, with their status and experiments.
- [`docs/plan.md`](docs/plan.md) - the step-by-step plan, mask geometry tests, method.
- [`experiments/`](experiments/_index.md) - one folder per experiment (`E0NN-slug`): its preregistration, card and results; raw summaries in `runs/E0NN-slug/`.
- [`docs/prior-art.md`](docs/prior-art.md) - what dynamic quantization already has and where FoQLens differs.
- [`docs/reading-notes.md`](docs/reading-notes.md) - notes from the papers read, with the passages cited and what FoQLens takes from them.
- [`docs/station.md`](docs/station.md) - the machine the runs are made on, its limits, and a log of what the runs cost it.
- [`docs/visual-metaphor.md`](docs/visual-metaphor.md) - how FoQLens is drawn.
- [`docs/data-sources.md`](docs/data-sources.md) - where the questions of the first corpus came from (MMLU-Redux-2.0).
- [`docs/corpus.md`](docs/corpus.md) - the corpus: how it is selected and its three regimes.
- [`docs/data.md`](docs/data.md) - how the runs are stored and read: JSON Lines files, and DuckDB over them.
- [`prereg/`](prereg/) - the main preregistration; the addenda of each experiment sit in its folder.
- [`src/foqlens/`](src/foqlens/) - the bench:
  - `model`, `quant`, `precision` - loading, quantizers and residual slices, the per-block precision controller;
  - `scoring`, `pipeline` - mask sources (a new score is a new `MaskSource`);
  - `evaluate`, `quality` - quality metrics (a new metric is a new `QualityMetric`) and evaluation;
  - `budget`, `layouts`, `weight_map`, `zones`, `neighbours` - from masks to layouts: zone sources, fields and level rules as replaceable parts;
  - `gpu_share`, `gpu_monitor` - the share of the GPU a run takes, and its utilization.
- [`scripts/`](scripts/) - model download and one script per run; they only wire the bench together.
- [`tests/`](tests/) - unit tests and bench health tests.

## Order of work

Preregistration → run code → runs → results, each in its own commit, so the history shows the predictions came before the data. Analysis is blind: all runs first, then everything is opened at once. The only exception is step 0, a check that the model is fit for the bench at all.
