---
name: run-slices
description: Ask a run's files a question - answers, the judge's runs and verdicts, Claude's readings, the frozen corpus - with one DuckDB query over the jsonl where it lies (foqlens.runs). Use before counting anything about a run; never loop over the files in Python.
---

# run-slices

The files are the source of truth and nothing is copied into a database: DuckDB reads
`answers/<level>/<corpus>.jsonl`, the judge's runs `judge/<level>/<corpus>/<run>.jsonl` and
`verdicts/<level>/<corpus>.jsonl` on the spot (`docs/data.md`, `src/foqlens/runs.py`). A slice is one
query; a Python loop over the lines is how field names get guessed wrong and floats get compared for equality.

## The entry point

```python
from pathlib import Path
from foqlens.runs import RunFiles

run = RunFiles(answers=Path("runs/E016-uniform-quantization/e2b-it/answers"),
               judged=Path("runs/E016-uniform-quantization/e2b-it/judge"),
               readings=None, frozen=Path("corpus/e2b-it"))   # judged, readings and frozen are optional
run.query("select level, count(*) from verdicts group by level")
run.agreement_with_readings("bf16")   # per corpus: exact match and judge against Claude's readings
```

The views: `answers` (the answers files), `judged` (every line of every judge run, with `run`), `verdicts`
(the latest run per level, corpus and question), `readings`, `frozen`. Every answers-like view has
`accepted`: the verdict whichever judge wrote the line, null where no judge read it.

## What a row holds

`corpus, id, revision, model, level, prompt, reply, answer, reasoning, exact_match, f1, tokens, stopped,
judge_kind, judge_accepted, judge_reply` - and on lines of the one-token judge (before 2026-09-16)
`judge_with_reference, judge_without_reference, judge_grades` instead of the three `judge_*` verdict fields.

Four traps, each one already paid for:

- **`reply` is what the model wrote**; `answer` is what was extracted from it. An ARC or HotpotQA reply
  without its answer line has an empty `answer` - D2 has it on every such question.
- **Count verdicts with `accepted`**, never with the fields of one judge: old and new lines read together
  only through it.
- **The one-token judge's `judge_*` are probabilities.** Where you must read them, compare decisions -
  `judge_with_reference > JUDGE_YES` - never the floats: two kernels agree on the verdict and differ in the
  seventh digit.
- **`judge_kind = 'N/A'`** is a reply the parser could not read, not a No; a later run usually has it.
  `stopped` false means the answer ran into the token cap.

## Recipes

Health of every level - how many answers, how often the model never stopped, how long they ran:

```sql
select level, count(*) n, round(avg((not stopped)::int), 3) not_stopped,
       round(avg((trim(reply) = '')::int), 4) empty_reply,
       median(tokens) med_tokens, quantile_cont(tokens, 0.95) p95_tokens
from answers group by level order by level
```

The judge's verdicts on a level, kept questions only, and what is still unread:

```sql
select v.level, count(*) n, round(avg(v.accepted::int), 4) accepted,
       sum((v.judge_kind = 'N/A')::int) not_read, sum((v.judge_kind = 'Garbage')::int) garbage
from verdicts v join frozen f using (corpus, id) where f.part = 'kept'
group by v.level order by v.level
```

Two judges over the same answers - how many decisions moved:

```sql
select v.level, count(*) n, sum((v.accepted <> a.accepted)::int) moved
from verdicts v join answers a using (corpus, id, level) group by v.level order by v.level
```

The lines of an answers file an earlier run left N/A, for `rejudge_answers.py --rows`:

```sql
with lines as (select id, row_number() over () as line from read_json_auto(?))
select string_agg(line::varchar, ',' order by line) picked_rows   -- `rows` is a reserved word
from lines join verdicts v using (id) where v.level = ? and v.corpus = ? and v.judge_kind = 'N/A'
```

The collapse signature of issue #14 - whole batches answering with one and the same text:

```sql
select level, max(n) largest_group, arg_max(len, n) len_of_that_reply
from (select level, reply, count(*) n, length(reply) len from answers group by level, reply)
group by level order by largest_group desc
```

## Rules

- One query, no Python loop over the jsonl; parameters go through `query(sql, params)`.
- A slice that a run of the bench needs again belongs in `foqlens/runs.py` as a method with a test
  (`tests/test_runs_unit.py`), not in a script.
- Thresholds come from the domain - `selection.JUDGE_YES`, `selection.KNOWING` - never typed in by hand.
- The views hold no copy: a slice reads the files as they are when it is asked.
