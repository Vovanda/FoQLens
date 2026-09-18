# FoQLens

Hi, I'm Vladimir Savkin. I studied mathematics and programming and hold a master's degree in fundamental computer science and information technology, and I work as a systems architect - distributed systems, lately with formal verification. FoQLens is my research, and it is a hobby: I am not a professional scientist. That is why the bench keeps me honest - every prediction is committed before the run that tests it. More about me: [CV](https://sawking.tech/cv).

**Does a language model need a fixed precision for every question?**

FoQLens raises the precision of the weights only where the question needs it.

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
2. Grow the query's **expert zones** from the peaks of that score, along a distance between blocks. Which score, which distance and how the zones grow are strategies still to be tested ([docs/zone-strategies.md](docs/zone-strategies.md)).
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
| 0 | Do topics separate in the model's representations at all? | they don't (then the model is too weak) |
| 1 | Are per-block masks similar within a topic and different between topics? Are they concentrated? | masks look the same for every query |
| 2 | Do related topics (biology-chemistry) share more of their zones than unrelated ones (biology-math)? | no overlap structure |
| 2+ | Geometry: are masks additive, is there a junction zone, does ablating it break mixed questions only? | - (refining, not load-bearing) |
| 3 | How much of what the model knows do the query's zones keep, against uniform quantization at the same memory? The other topic's zones and generic importance test the address | the zones keep no more than uniform quantization at the same memory |
| 7 | In an agent chain of a draft and refinements, does the zone model end better than the same model at native precision and than uniform quantization at the same memory? (the main hypothesis, once the model exists) | - |

All predictions were [preregistered](prereg/) in git before each run, as directions ("A > B"). The full reasoning is in [`docs/`](docs/).

## Status

**The bench and the data are ready; the hypotheses are tested from the next step on.**

**The corpus of what the model knows is built and frozen:** the full model answered every question of six
datasets in its own words, with no options anywhere, and a question stays if the answer is right; three
regimes in it - the answer in a passage, only in the weights, across two passages and a step. The corpus has
20,640 questions: 18,576 the model knows and 2,064 it does not
([docs/corpus.md](docs/corpus.md)).

**How the model holds its knowledge under uniform quantization is measured:** D8 keeps 98.8% of what the full
model knows, D6 96.8%, D4 91.0%, D2 51.4%. Knowledge goes from the weights first: with the answer in the
passage D4 loses 2.5%, with the answer only in the weights 13.0%. This is the baseline for every test of the
filter ([E002](experiments/E002-base-precision-d2/results.md)). On the first measurement naive rounding made
D2 incoherent ([E001](experiments/E001-uniform-quantization/results.md)); with the quantization method changed to
k-quant base precision D2 keeps half of the knowledge, and the filter is tested from D2.

**The premise of the main hypothesis came up on its own.** On the questions the full model does not know, the
coarser model answers where the precise one refuses: in E001 on HotpotQA bf16 says the passages hold no answer in
12.1% of them, D4 in 4.6%, and the accepted answers rise from 13.4% to 25.2%. Coarsening removes the caution and adds
no knowledge, and the guess is sometimes right. That is what [H4](docs/hypotheses.md) stands on: a draft guess can
be refined, "I don't know" cannot. It came up on data not built to show it; a class of tasks that shows it on
purpose is an experiment of its own.

**Next, the filter:** a mask that is asked about the model's answer, and the zones built from it.

**The question all of it serves:** does the regulator work - the structure where a coarse reading is enough, the
details where sharpness is needed, a gain over iterations. Saving memory is a secondary goal, plan B: even without
a gain in quality the mechanism saves memory at the same usability.

**Engineering:** one stored copy of the weights read at 2 / 4 / 6 / 8 bits: a k-quant base (Q2_K, Q4_K for the
sensitive classes) with 2-bit refinements over it, after MoBiQuant with departures
([E002](experiments/E002-base-precision-d2/results.md)). The zones' top rung is D8; the model's file may also hold
a tail to the source weights of any type, so the model reads back its source bit for bit without the checkpoint
([docs/refocustensors.md](docs/refocustensors.md)). A layout of depths is read by a CUDA kernel on tensor cores
straight from the copy's bytes, every block of rows to its depth: a decoding step of E2B-it at a mixed layout takes
28.8 ms at a batch of 32 against 422.6 unpacked and 20.4 at bf16 ([docs/kernels.md](docs/kernels.md)); a prefill
unpacks the copy for a GEMM.

## Reproduce

Requirements: an NVIDIA GPU with 16 GB+ of memory, [uv](https://docs.astral.sh/uv/). uv fetches Python 3.12 by itself.

```sh
git clone https://github.com/Vovanda/FoQLens.git
cd FoQLens
uv sync                                              # torch (CUDA 12.8), transformers, bitsandbytes
uv run python scripts/download_models.py e2b e2b-it   # Gemma 4 E2B and E2B-it at their pinned revisions, ~20 GB
uv run python scripts/cut_model.py e2b-it            # the bench's model: E2B-it as .refocustensors, ~9.3 GB
uv run pytest                                        # unit + bench health tests on the GPU
```

A run takes 0.8 of the GPU by default - that share of the VRAM, and rest between batches - so the card stays usable; `--gpu-share 1` or `FOQLENS_GPU_SHARE=1` gives the full speed.

Model weights are not stored in the repository. `scripts/download_models.py` fetches them from Hugging Face (Apache 2.0, no token needed) at the commits pinned in `foqlens.model.REVISIONS`, and `foqlens.model.load()` reads exactly those commits. `scripts/cut_model.py` cuts a downloaded model into its [.refocustensors](docs/refocustensors.md) folder outside the repository, and the bench runs from that folder. Add `e4b` to the download for the confirmation model (~16 GB).

## Layout

- [`docs/goals.md`](docs/goals.md) - goals by step and their status.
- [`docs/problem-statement.md`](docs/problem-statement.md) - the problem statement: expert zones come out of it.
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
  - `model`, `quant`, `kquant`, `refinements`, `precision` - loading, quantizers, the k-quant copy with refinements (`refinements`), the per-block precision controller; `refocustensors`, `safetensors_io`, `tail_cost` - the model file, its reading and writing tensor by tensor, the cost of its exact tail; `gguf_weights` - a published GGUF's weights for comparison;
  - `scoring`, `pipeline` - mask sources (a new score is a new `MaskSource`);
  - `evaluate`, `quality` - quality metrics (a new metric is a new `QualityMetric`) and evaluation;
  - `budget`, `layouts`, `weight_map`, `zones`, `neighbours` - from masks to layouts: zone sources, fields and level rules as replaceable parts;
  - `gpu_share`, `gpu_monitor` - the share of the GPU a run takes, and its utilization.
- [`scripts/`](scripts/) - model download and one script per run; they only wire the bench together.
- [`tests/`](tests/) - unit tests and bench health tests.

## Order of work

Preregistration → run code → runs → results, each in its own commit, so the history shows the predictions came before the data. Analysis is blind: all runs first, then everything is opened at once. The only exception is step 0, a check that the model is fit for the bench at all.
