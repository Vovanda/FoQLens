<!-- mcp-project: FoQLens -->

# FoQLens

A directed quantization bench: weight precision is allocated by the meaning of the query, expert zones emerge instead of being set by a router. The problem statement and the plan are in `docs/`, the predictions in `prereg/`.

Everything in the repository is written in English: docs, code comments, test messages, commit messages.

## Bench discipline (do not break)

- **The preregistration is untouchable.** The binding version is the Russian original `prereg/PREREGISTRATION.ru.md`, committed in `be66b77`; it is never edited and its history is never rewritten. `prereg/PREREGISTRATION.md` is an English translation. Clarifications go into a new file in a new dated commit; the old one stays.
- **Commit order:** preregistration → run code → runs → results, in separate commits.
- **Blind analysis.** All runs at once; during the runs only check that the script did not crash. The exception is step 0.
- **Held-out topics** (`prompts/heldout/`) are not opened and not run while the score is being debugged.
- **Two baselines** at step 3: uniform quantization and a random mask of the same concentration.

## Stack

- Gemma 4 E2B (debugging) / E4B (confirmation), base checkpoints `google/gemma-4-E2B`, `google/gemma-4-E4B` at the revisions pinned in `foqlens.model.REVISIONS`. Do not take the 26B-A4B MoE.
- transformers + bitsandbytes, lm-evaluation-harness. Not ollama / llama.cpp / LM Studio - they hide activations and do not allow per-block precision.
- Environment - `uv` in `.venv`, Python 3.12. Do not use the global Python.
- Model weights are never committed; `scripts/download_models.py` fetches them.
- Hardware: RTX 3090 Ti 24 GB.
