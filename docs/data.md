---
title: The data of the runs
---

# The data of the runs: files, and DuckDB over them

**The principle.** Everything the bench writes - the model's answers, the judges' verdicts, Claude's
readings - lies in JSON Lines files: a line per answer, a file per corpus and level. The files are the
source of truth. A run only appends to them, and the lines stand in the order the model wrote them.
There is no database in the repository - only a query engine that reads the files where they lie and
keeps nothing.

**The judge's runs.** A verdict is not written over the answer it reads. Every pass of the judge over one
answers file is a run of its own, `judge/<level>/<corpus>/<run>.jsonl`, named by when it started, with its
summary beside it: the commit of the code, the file it read, the lines picked and the questions judged.
A later run asks again only what it is given - the N/A of an earlier run, say - and the earlier run stays.
The verdict on a question is the latest run's that read it: the view `verdicts`, over the view `judged`
of every run.

**Why not a database.**

- A run that crashes appends where it stopped and skips the answers already written; it needs no
  transactions.
- Blind analysis and evidence are plain folders: a broken run is moved aside with one command and kept
  as evidence.
- Git sees the data: a run is committed as it is, and its diff can be read.
- The order of the lines is data too: a line remembers the batch it was written in. A table of a
  database has no order.

**Why DuckDB.** We chose it, and we recommend it for this kind of work:

- it reads JSON Lines where they lie - a glob over folders, the schema inferred, old and new lines with
  different fields read together; there is no second copy of the data to drift from the files;
- it is embedded - a package in the environment, no server;
- it is a columnar engine built for analytics: aggregates over tens of thousands of answers take seconds;
- its SQL is PostgreSQL's dialect (DuckDB took its parser), with conveniences on top: `GROUP BY ALL`,
  `QUALIFY`, `PIVOT`, functions over lists, regular expressions with Unicode script classes;
- it hands its results to pandas or Polars as they are.

ClickHouse was turned down (a server, and not this scale), and so was SQLite (row-oriented, and the files
would have to be loaded into tables). The engine can be replaced: Polars, pandas and ClickHouse local read
the same files. The queries sit in one module, [`foqlens.runs`](../src/foqlens/runs.py); the rest of the
bench asks for a slice and knows no SQL.

**What it gave on the first evening.**

- The slices of stage 1 equal what its selection wrote, `passed-bf16.json` - the agreement of the exact
  match and of the judge with Claude's readings, per corpus (`tests/test_runs_unit.py`).
- All of E001 - the knowledge kept per level and corpus, the judges' agreement, topics, fragility, the
  language of the answer - is counted in seconds.
- The slice "lost answers next to bf16's" showed Chinese characters, and the order of the lines showed
  that whole batches broke. That is how a fault of the bench was found
  ([#14](https://github.com/Vovanda/FoQLens/issues/14)): the fused attention kernel collapsed padded batches.
