"""Infrastructure: slices over the run files, queried where they lie (docs/data.md).

The files stay the source of truth - answers/<level>/<corpus>.jsonl, the judge's runs
judge/<level>/<corpus>/<run>.jsonl, verdicts/<level>/<corpus>.jsonl, the frozen corpus files - and nothing is
copied into a database: DuckDB reads them on the spot, each time a slice is asked. The queries live here; the
rest of the bench asks for a slice and knows no SQL.

Invariant: a slice reads the files as they are when it is asked - the views hold no copy of the rows.
Invariant: `verdicts` holds one line per question, from the latest run of the judge that read it; the runs
before it stay in `judged`.
Invariant: the agreement of the exact match and of the judge with Claude's readings of stage 1 equals
passed-bf16.json, which stage 1's selection wrote (tests/test_runs_unit.py).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from foqlens.selection import JUDGE_YES, KNOWING

# The frozen corpus files hold every id of a corpus in a few lists; DuckDB's default object limit is 16 MiB.
FROZEN_OBJECT_BYTES = 64 * 2**20


class RunFiles:
    """One run's files as views: `answers`, and `judged` with `verdicts`, `readings` and `frozen` where their folders are given."""

    def __init__(self, answers: Path, readings: Path | None = None, frozen: Path | None = None, judged: Path | None = None):
        self.con = duckdb.connect()
        lines = f"read_json_auto('{answers.as_posix()}/*/*.jsonl', union_by_name=true)"
        self.con.execute(f"create view answers as select *, {self._accepted(lines)} as accepted from {lines}")
        if judged is not None:
            runs = f"read_json_auto('{judged.as_posix()}/*/*/*.jsonl', union_by_name=true, filename=true)"
            self.con.execute(f"""create view judged as select *, {self._accepted(runs)} as accepted,
                regexp_extract(replace(filename, '\\', '/'), '([^/]+)\\.jsonl$', 1) as run from {runs}""")
            self.con.execute("""create view verdicts as select * from judged
                qualify row_number() over (partition by level, corpus, id order by run desc) = 1""")
        if readings is not None:
            self.con.execute(f"create view readings as select * from "
                             f"read_json_auto('{readings.as_posix()}/*/*.jsonl', union_by_name=true)")
        if frozen is not None:
            self.con.execute(f"""create view frozen as
                with f as (select corpus, kept, unknown_share, tuning from read_json_auto(
                    '{frozen.as_posix()}/*.json', maximum_object_size={FROZEN_OBJECT_BYTES}))
                select corpus, unnest(kept) id, 'kept' part from f
                union all select corpus, unnest(unknown_share), 'unknown_share' from f
                union all select corpus, unnest(tuning), 'tuning' from f""")

    def _accepted(self, lines: str) -> str:
        """`accepted`: the judge's verdict whichever judge wrote the line (selection.Answer.accepted)."""
        columns = {row[0] for row in self.con.execute(f"describe select * from {lines}").fetchall()}
        verdicts = (["judge_accepted"] if "judge_accepted" in columns else []) + \
                   ([f"judge_with_reference > {JUDGE_YES}"] if "judge_with_reference" in columns else [])
        # lines no judge has read carry neither: their verdict is unknown, not No
        return f"coalesce({', '.join(verdicts)}, null)" if verdicts else "null::boolean"

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        """Any slice over the views, as rows of column -> value."""
        relation = self.con.execute(sql, params or [])
        columns = [d[0] for d in relation.description]
        return [dict(zip(columns, row)) for row in relation.fetchall()]

    def agreement_with_readings(self, level: str) -> dict[str, dict[str, float]]:
        """Per corpus: how many answers Claude read, and how often the exact match and the judge agree with the reading."""
        known = ", ".join(f"'{reading}'" for reading in sorted(KNOWING))
        rows = self.query(f"""
            select a.corpus, count(*) n_read,
                   avg(((a.exact_match = 1) = (r.reading in ({known})))::int) exact_match,
                   avg((a.accepted = (r.reading in ({known})))::int) judge
            from answers a join readings r using (corpus, id, level)
            where a.level = ? group by a.corpus order by a.corpus""", [level])
        return {r["corpus"]: {"read": r["n_read"], "exact_match": r["exact_match"], "judge": r["judge"]} for r in rows}
