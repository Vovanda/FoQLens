---
name: run-slices
description: Ask a run's files a question - answers, verdicts, the frozen corpus - with one DuckDB query over the jsonl where it lies (foqlens.runs). Use before counting anything about a run; never loop over the files in Python.
---

# run-slices

The files are the source of truth and nothing is copied into a database: DuckDB reads
`answers/<level>/<corpus>.jsonl` and `verdicts/<level>/<corpus>.jsonl` on the spot (`docs/data.md`,
`src/foqlens/runs.py`). A slice is one query; a Python loop over the lines is how field names get
guessed wrong and floats get compared for equality.

## The entry point

```python
from pathlib import Path
from foqlens.runs import RunFiles
from foqlens.selection import JUDGE_YES, KNOWING

run = RunFiles(answers=Path("runs/E016-uniform-quantization/e2b-it/answers"),
               readings=None, frozen=Path("corpus/e2b-it"))   # readings and frozen are optional
run.query("select level, count(*) from answers group by level")
run.agreement_with_readings("bf16")   # per corpus: exact match and judge against Claude's readings
```

A second run joins as another view on the same connection:

```python
run.con.execute("create view other as select * from read_json_auto("
                "'../FoQLens-e016-math-judge/answers/*/*.jsonl', union_by_name=true)")
```

## What a row holds

`corpus, id, revision, model, level, prompt, reply, answer, reasoning, exact_match, f1,
judge_with_reference, judge_without_reference, tokens, stopped, judge_grades`

Three traps, each one already paid for:

- **`reply` is what the model wrote**; `answer` is what was extracted from it. Counting "empty answers"
  on `answer` reports every row as empty.
- **`judge_*` are probabilities, not verdicts.** Compare decisions - `judge_with_reference > JUDGE_YES` -
  never the floats themselves: two kernels agree on the verdict and differ in the seventh digit, so raw
  equality reported 99.7% "flips" where the decisions moved on 0.25%.
- **`judge_grades` is a list** of the grade probabilities; `stopped` false means the answer ran into the
  token cap.

## Recipes

Health of every level - how many answers, how often the model never stopped, how long they ran:

```sql
select level, count(*) n, round(avg((not stopped)::int), 3) not_stopped,
       round(avg((trim(reply) = '')::int), 4) empty_reply,
       median(tokens) med_tokens, quantile_cont(tokens, 0.95) p95_tokens
from answers group by level order by level
```

Two judge passes over the same answers - how many verdicts actually moved:

```sql
select a.level, count(*) n,
       sum(((a.judge_with_reference > ?) <> (m.judge_with_reference > ?))::int) flips_with_ref,
       sum(((a.judge_without_reference > ?) <> (m.judge_without_reference > ?))::int) flips_no_ref,
       round(max(abs(a.judge_with_reference - m.judge_with_reference)), 4) max_gap
from answers a join other m using (corpus, id, level) group by a.level order by a.level
```

The collapse signature of issue #14 - whole batches answering with one and the same text:

```sql
select level, max(n) largest_group, arg_max(len, n) len_of_that_reply
from (select level, reply, count(*) n, length(reply) len from answers group by level, reply)
group by level order by largest_group desc
```

A level against the frozen corpus, kept questions only:

```sql
select a.level, f.part, count(*) n, round(avg((a.judge_with_reference > ?)::int), 4) judged_yes
from answers a join frozen f using (corpus, id) where f.part = 'kept'
group by a.level, f.part order by a.level
```

## Rules

- One query, no Python loop over the jsonl; parameters go through `query(sql, params)`.
- A slice that a run of the bench needs again belongs in `foqlens/runs.py` as a method with a test
  (`tests/test_runs_unit.py`), not in a script.
- Thresholds come from the domain - `selection.JUDGE_YES`, `selection.KNOWING` - never typed in by hand.
- The views hold no copy: a slice reads the files as they are when it is asked.
