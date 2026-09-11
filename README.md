# FoQLens

**Can a language model spend its precision where the question is, instead of everywhere?**

FoQLens is a research bench that tests one idea: keep the weights that matter for *this particular query* at high precision and read the rest coarsely - and let the model itself say which weights those are.

Name: Focus + Quantization + Lens. The lens is a visual metaphor, not the mechanism.

## The problem

Quantization makes models smaller and faster by storing weights with fewer bits. Every existing scheme - static or dynamic - decides precision per **fixed unit**: the whole model, a layer, a channel, a token, an MoE expert. The boundaries come from the architecture; a controller only chooses how many bits each unit gets.

A question about biology and a question about a proof do not need *more or less* precision. They need precision **in different places**.

## The idea

1. Run the first layers coarsely. Their activations already tell what the query is about - a signal that is computed on every pass and read by nobody.
2. Turn that intermediate representation into a score per block of weights: how much this block matters for this meaning.
3. Read high-scoring blocks at high precision (bf16), everything else coarsely (int8 / nf4), at a fixed mean bit budget.

No router is trained and no experts are defined in advance. If the idea holds, **expert zones emerge** as the regions that stay sharp when everything around them is coarsened - and related topics share part of their zone instead of paying for it twice, as MoE experts do.

## What the bench checks

The steps are ordered so each one can kill the next:

| Step | Question | Kills the idea if |
| --- | --- | --- |
| 0 | Do topics separate in the model's representations at all? | they don't (then the model is too weak, not the idea) |
| 1 | Are per-block masks similar within a topic and different between topics? Are they concentrated? | masks look the same for every query |
| 2 | Do related topics (biology-chemistry) share more of their zones than unrelated ones (biology-math)? | no overlap structure |
| 2+ | Geometry: are masks additive, is there a junction zone, does ablating it break mixed questions only? | - (refining, not load-bearing) |
| 3 | At the same mean bits, is quality above **both** uniform quantization and a random mask of the same concentration? | it beats uniform but not random: then any non-uniformity helps, not the address |

All predictions were [preregistered](prereg/) in git before the first run, as directions ("A > B"), not numbers. The full reasoning is in [`docs/`](docs/).

## Status

- Preregistration committed before any run.
- The bench is ready: bf16 / int8 / nf4 precision set per layer, per module or per block of 64 weight rows over Gemma 4 E2B, with a mean-bits account. Health tests prove the control is real: all-bf16 is bit-exact with the original model, and switching layer N leaves every earlier layer's output untouched.
- Steps 0-3 have not been run yet.

What this bench does **not** show: memory savings. Coarse copies are stored alongside the full weights and dequantized on the fly - the numbers are exactly those of a weight read at that precision, which is all quality measurements need.

## Reproduce

Requirements: an NVIDIA GPU with 16 GB+ of memory (E2B with the controller installed takes ~12.3 GB), [uv](https://docs.astral.sh/uv/). uv fetches Python 3.12 by itself.

```sh
git clone https://github.com/Vovanda/FoQLens.git
cd FoQLens
uv sync                                         # torch (CUDA 12.8), transformers, bitsandbytes
uv run python scripts/download_models.py        # Gemma 4 E2B at its pinned revision, ~10 GB
uv run pytest                                   # unit + bench health tests on the GPU
```

Model weights are not stored in the repository. `scripts/download_models.py` fetches them from Hugging Face (Apache 2.0, no token needed) at the commits pinned in `foqlens.model.REVISIONS`, and `foqlens.model.load()` reads exactly those commits. Add `e4b` to the download for the confirmation model (~16 GB).

## Layout

- [`docs/goals.md`](docs/goals.md) - goals by step and their status.
- [`docs/problem-statement.md`](docs/problem-statement.md) - the problem statement: expert zones as an outcome, not an input.
- [`docs/plan.md`](docs/plan.md) - the step-by-step plan, mask geometry tests, method.
- [`docs/prior-art.md`](docs/prior-art.md) - what dynamic quantization already has and where FoQLens differs.
- [`docs/visual-metaphor.md`](docs/visual-metaphor.md) - how FoQLens is drawn.
- [`prereg/`](prereg/) - the preregistration: predictions fixed before any run.
- [`src/foqlens/`](src/foqlens/) - the bench: model loading, quantizers, precision controller.
- [`scripts/`](scripts/) - model download.
- [`tests/`](tests/) - unit tests and bench health tests.

## Order of work

Preregistration → run code → runs → results, each in its own commit, so the history shows the predictions came before the data. Analysis is blind: all runs first, then everything is opened at once. The only exception is step 0, a check that the model is fit for the bench at all.
