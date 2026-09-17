---
name: new-experiment
description: Open a new FoQLens experiment - the next id, its folder with PREREG.md and the card, the registry rows - and fix the preregistration before any code or run. Use whenever a new run, control or exploration is about to start.
---

# new-experiment

A new experiment is opened, and its preregistration committed, **before** any run code for it exists.
Order of commits, never merged: preregistration → run code → runs → results (CLAUDE.md, bench discipline).

## 1. The id

The next id is the highest id in `experiments/` plus one:

```bash
ls experiments | grep -o '^E[0-9]\{3\}' | sort | tail -1
```

## 2. The name

The slug is named by the question the experiment answers, in the project's words: zones, the
quantization filter, the corpus, the regimes. The lens is only the metaphor - never in a slug or a title.
Offer Volodya a tier list (S..D) of 3-4 slugs with a reason each, recommend one; he picks.

## 3. The folder

```
experiments/E0NN-slug/
  PREREG.md     the preregistration - one per folder; a later refinement is PREREG-2.md
  _index.md     the card: Hugo front matter + a few lines
  results.md    only after the run
runs/E0NN-slug/ written by the run script (its --out default), never by hand
```

**PREREG.md** - written first in Russian for Volodya, English into the repo after his word:

```markdown
# E0NN - <title>: preregistration

Written <date>, before any run.

## 1. Why
The question in the project's terms, and what the answer changes for the regulator.

## 2. Design
- **Mechanism**: [docs/quantization-filter.md](../../docs/quantization-filter.md) at commit `<sha>`.
- **Corpus**: the frozen corpus file at commit `<sha>`; which regimes.
- **Compared**: the layouts, with floor / focus_area / focus_strength; the baseline is uniform
  quantization at the same memory. A shape control, where one is needed, is the query's own zones
  carried elsewhere at the same cost - never random zones.
- **Judges**: the SQuAD score, the full model, Claude - Claude's verdict decides where they disagree.

## 3. Predictions
Directions, each with its threshold and what outcome falsifies it. Id each one (P1, P2, ...).

## 4. Criterion
Paired bootstrap over the same questions in the same batches; what interval counts.

## 5. Run
The command, the output folder runs/E0NN-slug/, the expected time.
```

**_index.md** - the card:

```markdown
---
title: "E0NN - <title>"
date: <YYYY-MM-DD>
weight: <NN>
hypotheses: [<H..>]
statuses: [planned]
params:
  fixed: "<prereg commit sha, once committed>"
  run: ""
  results: ""
  verdict: ""
---

# E0NN - <title>

One paragraph: the question and the design. Preregistration: [PREREG](PREREG.md).
```

## 4. Register it, in the same commit

- `experiments/_index.md`: a row - id linked to the card, the title, the date fixed, hypotheses, status.
- `docs/hypotheses.md`: the id in the Experiments column of every hypothesis it tests.
- Every word as the rest of the project has it (Устав 18): zones, floor, focus_area, focus_strength,
  the quantization filter, uniform quantization at the same memory.

## 5. Check before the commit

- Relative links resolve:

  ```bash
  .venv/Scripts/python.exe - <<'PY'
  import re, subprocess
  from pathlib import Path
  files = [f for f in subprocess.run(["git","ls-files","-co","--exclude-standard"],capture_output=True,text=True).stdout.split()
           if f.endswith((".md",".html")) and not f.startswith(("runs/",".claude/"))]
  link = re.compile(r"\]\(([^)\s#]+)(#[^)]*)?\)|href=\"([^\"#:?$]+)(#[^\"]*)?\"")
  print([f"{f}: {t}" for f in files for m in link.finditer(Path(f).read_text(encoding="utf-8"))
         for t in [m.group(1) or m.group(3)] if t and not t.startswith(("http","mailto"))
         and not (Path(f).parent / t).resolve().exists()])
  PY
  ```
- `pytest tests/test_site_unit.py tests/test_site_shelf_unit.py` if the site shelf was touched.
- The commit holds the preregistration, the card and the registry rows only - `docs(prereg): E0NN ...`.
  Then the card's `fixed` gets that sha in the next commit, with the run code.

## 6. During and after the run

- Blind analysis: all runs first, only a crash check while they go; held-out topics are not opened.
- Runs go through `Bench.load`, so the thermal guard and the hourly break apply; the summary records
  the GPU (utilization, temperature, power).
- Results: `results.md` states what was measured against each prediction; a conclusion goes into the
  docs only after Volodya approves it (Устав 11).
