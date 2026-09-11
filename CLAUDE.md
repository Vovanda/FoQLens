<!-- mcp-project: FoQLens -->

# FoQLens

FoQLens is a model whose precision regulator reads the weights through a filter with lenses in the query's expert zones; MoE is a special case of it. This repository is the R&D bench inside FoQLens that tests the idea: weight precision is allocated by the meaning of the query, expert zones emerge instead of being set by a router. The problem statement and the plan are in `docs/`, the main preregistration in `prereg/`, and every experiment in `experiments/E0NN-slug/` - its addenda, a card (`_index.md`, YAML front matter for Hugo: dates, commits, hypotheses, status, verdict) and `results.md` - with its raw summaries in `runs/E0NN-slug/`. A new experiment takes the next id; hypotheses are tracked in `docs/hypotheses.md`.

Everything in the repository is written in English: docs, code comments, test messages, commit messages.

## Bench discipline (do not break)

- **The preregistration is untouchable.** The binding version is the Russian original `prereg/PREREGISTRATION.ru.md`, committed in `be66b77`; it is never edited and its history is never rewritten. `prereg/PREREGISTRATION.md` is an English translation. Clarifications go into a new file in a new dated commit; the old one stays. What is untouchable in every preregistration file is the hypothesis and its predictions, criteria and bets: when files move, link paths inside them are updated in a separate commit whose diff touches only the links.
- **Commit order:** preregistration → run code → runs → results, in separate commits.
- **Blind analysis.** All runs at once; during the runs only check that the script did not crash. The exception is step 0.
- **Held-out topics** (`prompts/heldout/`) are not opened and not run while the score is being debugged.
- **Two baselines** at step 3: uniform quantization and a random mask of the same concentration.

## Engineering standards

We write high-performance code and hold it to the usual engineering principles - not research scripts.

- **Performance is a requirement, not a later fix.** Hot paths run on the GPU without host-device synchronization (no `.item()`, `.tolist()`, `torch.unique` or Python branching on tensor values inside forward passes); work is batched; anything reusable across passes is computed once and cached. Every long run reports GPU utilization; a run below its GPU share is a bug to fix before the numbers are trusted. The share (`--gpu-share`, default 0.8, `foqlens/gpu_share.py`) caps both the VRAM and the utilization, so that the card stays usable for people - a game, another researcher's job.
- **SOLID.** One responsibility per module and class; extend through new classes behind the same interface (a new mask source is a new scorer, not an `if` in the old one); callers depend on interfaces, not concrete classes; small, focused interfaces.
- **Clean architecture.** The domain - quantization, precision control, scoring, metrics (`foqlens/*`) - knows nothing about datasets, files or the command line. Infrastructure - model loading, dataset building, run output - sits around it. Scripts in `scripts/` only wire the two together and hold no logic worth testing.
- **Tests for every behavior**, including performance-critical rewrites: a batched or cached path must match the reference path within the stated tolerance.
- **Invariants are written down.** Every module states its invariants in its docstring as `Invariant: ...`, exact or approximate with the measured bound, and each has a test. bf16 inference is not batch-invariant (token states move up to ~2% with the batch size), so exact claims are made only where they hold - same inputs in the same batch, per-sample layouts against the same batch - and comparisons between configurations always run on the same batches.
- No magic numbers in code: named constants with a reason, or a parameter.

## Stack

- Gemma 4 E2B (debugging) / E4B (confirmation), base checkpoints `google/gemma-4-E2B`, `google/gemma-4-E4B` at the revisions pinned in `foqlens.model.REVISIONS`. Do not take the 26B-A4B MoE.
- transformers + bitsandbytes, lm-evaluation-harness. Not ollama / llama.cpp / LM Studio - they hide activations and do not allow per-block precision.
- Environment - `uv` in `.venv`, Python 3.12. Do not use the global Python.
- Model weights are never committed; `scripts/download_models.py` fetches them.
- Hardware: RTX 3090 Ti 24 GB.
