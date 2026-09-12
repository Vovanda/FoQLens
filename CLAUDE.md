<!-- mcp-project: FoQLens -->

# FoQLens

FoQLens is a model whose precision regulator reads the weights through a filter with lenses in the query's expert zones; MoE is a special case of it. This repository is the R&D bench inside FoQLens that tests the idea: weight precision is allocated by the meaning of the query, expert zones emerge instead of being set by a router. The problem statement and the plan are in `docs/`, the main preregistration in `prereg/`, and every experiment in `experiments/E0NN-slug/` - its addenda, a card (`_index.md`, YAML front matter for Hugo: dates, commits, hypotheses, status, verdict) and `results.md` - with its raw summaries in `runs/E0NN-slug/`. A new experiment takes the next id; hypotheses are tracked in `docs/hypotheses.md`.

Everything in the repository is written in English: docs, code comments, test messages, commit messages.

## Bench discipline (do not break)

- **The preregistration is untouchable.** The binding version is the Russian original `prereg/PREREGISTRATION.ru.md`, committed in `be66b77`; it is never edited and its history is never rewritten. `prereg/PREREGISTRATION.md` is an English translation. Clarifications go into a new file in a new dated commit; the old one stays. What is untouchable in every preregistration file is the hypothesis and its predictions, criteria and bets: when files move, link paths inside them are updated in a separate commit whose diff touches only the links.
- **Commit order:** preregistration → run code → runs → results, in separate commits.
- **Blind analysis.** All runs at once; during the runs only check that the script did not crash. The exception is step 0.
- **Held-out topics** (`prompts/heldout/`) are not opened and not run while the score is being debugged.
- **The baseline** at step 3 is uniform quantization at the same memory - what a deployment would otherwise do. A random mask is not a baseline: random zones overlap less, so they cost more, and beating deliberate damage proves nothing (dropped 2026-09-13, see `docs/plan.md`). Where a shape control is needed, it is the query's own zones carried elsewhere at the same cost.
- **The question to answer** is the engineer's one: over what interval of the regulator's settings is the model still usable, and what memory does that interval actually save - if any.

## Engineering standards

We write high-performance code and hold it to the usual engineering principles - not research scripts.

- **Performance is a requirement, not a later fix.** Hot paths run on the GPU without host-device synchronization (no `.item()`, `.tolist()`, `torch.unique` or Python branching on tensor values inside forward passes); work is batched; anything reusable across passes is computed once and cached. Every long run reports GPU utilization; a run below its GPU share is a bug to fix before the numbers are trusted. The share (`--gpu-share`, default 0.8, `foqlens/gpu_share.py`) caps both the VRAM and the utilization, so that the card stays usable for people - a game, another researcher's job.
- **SOLID.** One responsibility per module and class; extend through new classes behind the same interface (a new mask source is a new scorer, not an `if` in the old one); callers depend on interfaces, not concrete classes; small, focused interfaces.
- **Clean architecture.** The domain - quantization, precision control, scoring, metrics (`foqlens/*`) - knows nothing about datasets, files or the command line. Infrastructure - model loading, dataset building, run output - sits around it. Scripts in `scripts/` only wire the two together and hold no logic worth testing.
- **Tests for every behavior**, including performance-critical rewrites: a batched or cached path must match the reference path within the stated tolerance.
- **Invariants are written down.** Every module states its invariants in its docstring as `Invariant: ...`, exact or approximate with the measured bound, and each has a test. bf16 inference is not batch-invariant (token states move up to ~2% with the batch size), so exact claims are made only where they hold - same inputs in the same batch, per-sample layouts against the same batch - and comparisons between configurations always run on the same batches.
- No magic numbers in code: named constants with a reason, or a parameter.

## The site (`index.html`, `docs.html`, `site/*`)

The same standards as the bench, in the shape the front end takes them. They are the ones already in
force on the sibling project ([Work-Life-Schedule](https://github.com/Vovanda/Work-Life-Schedule)).

- **Semantics first.** `<button>` for an action, a real element for what it is. An icon button carries
  an `aria-label`, a decorative shape carries `aria-hidden`, and focus is always visible.
- **No colour outside the palette.** Every colour is a token in `:root`, and both themes are
  assignments of those tokens - a hex inside a rule is a bug. The same holds for the field drawn on
  the canvas: its palettes live at the top of `site/field.js`, one set per theme.
- **The theme is the device's** unless the reader says otherwise, and it is applied before the first
  paint - a page that picks it later shows one palette and swaps to the other in front of the reader.
- **Layout by flex and grid, not by margins.** Geometry stays the same across states: the mark of the
  current page is drawn with `outline` or an inset shadow, never by a border that shifts everything.
- **Flat selectors**, nesting no deeper than two levels. `!important` means the order of the rules is
  broken and is to be fixed there - the one exception is `.preload`, which switches every transition
  off while the page lays itself out.
- **Animate `transform` and `opacity` only**, and respect `prefers-reduced-motion`.
- **Do not move with a script what CSS can do.** What the script may do is publish a measurement the
  stylesheet cannot take itself (the gutter, the panel's shift).
- **One thing does one thing.** A function computes or draws, never both. The top of the file reads
  like a table of contents; sections carry `/* ==== NAME ==== */` headers, and a new rule goes into
  its section, not onto the end of the file.
- **Do not duplicate knowledge.** The ladder lives in `LADDER`, a colour in a token, the controls in
  `DEFAULTS`. One entity, one word, everywhere: the level outside the lenses is `floor` in the panel,
  in the scripts and in `docs/lens.md`; the glass is the picture of it, not a second name for it.
- **A comment says why.** What a line does is visible in the line.
- **Static files are versioned** (`site/field.js?v=N`) and the version is bumped with every change to
  `site/*`: without it a browser serves a stale copy and the page you are shown is not the page you
  wrote. One version per page, never a half-bumped one.
- **The page only reads.** Nothing on the site writes data of the bench.
- **Never say it is done without looking.** Smoke it at 390, 768, 1200 and 1700 px in both themes;
  `node --check site/*.js` for syntax, `pytest tests/test_site_unit.py` for the frame the two pages
  share. A screenshot after the change, not after the complaint.

## Stack

- Gemma 4 E2B (debugging) / E4B (confirmation), base checkpoints `google/gemma-4-E2B`, `google/gemma-4-E4B` at the revisions pinned in `foqlens.model.REVISIONS`. Do not take the 26B-A4B MoE.
- transformers + bitsandbytes, lm-evaluation-harness. Not ollama / llama.cpp / LM Studio - they hide activations and do not allow per-block precision.
- Environment - `uv` in `.venv`, Python 3.12. Do not use the global Python.
- Model weights are never committed; `scripts/download_models.py` fetches them.
- Hardware: RTX 3090 Ti 24 GB.
