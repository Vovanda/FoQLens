"""Infrastructure: slices over the run files, queried where they lie (docs/data.md).

The files stay the source of truth - answers/<level>/<corpus>.jsonl, the judge's runs
judge/<level>/<corpus>/<run>.jsonl, verdicts/<level>/<corpus>.jsonl, the frozen corpus files - and nothing is
copied into a database: DuckDB reads them on the spot, each time a slice is asked. The queries live here; the
rest of the bench asks for a slice and knows no SQL.

Invariant: a slice reads the files as they are when it is asked - the views hold no copy of the rows.
Invariant: `verdicts` holds one line per question, from the latest run of the judge that read it; the runs
before it stay in `judged`.
Invariant: every kind the judge gives falls into one grade group or is N/A, so a level's grade shares and its
unread share sum to 1 on a part of the frozen corpus.
Invariant: the agreement of the exact match and of the judge with Claude's readings of stage 1 equals
passed-bf16.json, which stage 1's selection wrote (tests/test_runs_unit.py).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from foqlens.corpora import CORPORA
from foqlens.judging import GRADE_GROUPS, NOT_READ
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

    def grades(self, part: str) -> dict[str, dict[str, float]]:
        """Per level, on one part of the frozen corpus: how many questions, and the share of each grade group
        (judging.GRADE_GROUPS) and of replies the judge left unread."""
        return {r.pop("level"): r for r in self._grades(part, "v.level")}

    def grades_by_regime(self, part: str) -> dict[tuple[str, str], dict[str, float]]:
        """As grades, per level and regime of the corpus (corpora.Corpus.regime)."""
        return {(r.pop("level"), r.pop("regime")): r for r in self._grades(part, "v.level, c.regime")}

    def _grades(self, part: str, keys: str) -> list[dict]:
        shares = ", ".join(f"avg((v.judge_kind in ({', '.join(repr(k) for k in sorted(kinds))}))::int) {group}"
                           for group, kinds in GRADE_GROUPS.items())
        regimes = " union all ".join(f"select '{name}' corpus, '{corpus.regime}' regime" for name, corpus in CORPORA.items())
        return self.query(f"""
            with c as ({regimes})
            select {keys}, count(*) n, {shares}, avg((v.judge_kind = '{NOT_READ}')::int) not_read
            from verdicts v join frozen f using (corpus, id) join c using (corpus)
            where f.part = ? group by {keys} order by {keys}""", [part])

    # A reply read as the same answer: one case, one spacing, no punctuation at either end. Read by hand on the maps
    # of 20.09, two of six disagreements were a capital letter and a full stop - measured word for word, a layout is
    # charged for how it set the answer out rather than for what it answered.
    SAME_FORM = r"lower(trim(regexp_replace(regexp_replace({}, '\s+', ' ', 'g'), '^[^\w]+|[^\w]+$', '', 'g')))"

    def same_reply(self, reference: str) -> list[dict]:
        """Per level, how far its reply is the reference level's on the same question - what a precision field is
        measured by (Volodya 20.09: no judge, the question is whether the layout answers as the whole network does).

        Four readings of one comparison, loosest last: word for word; the same answer set out differently (one case,
        one spacing, no punctuation at either end); one answer inside the other, which is what a truncation or an
        addition looks like; and how close the two strings are (Jaro-Winkler), averaged over the questions.
        """
        mine, theirs = self.SAME_FORM.format("a.reply"), self.SAME_FORM.format("r.reply")
        return self.query(rf"""
            select a.level, count(*) n,
                   avg(({mine} = {theirs})::int) same,
                   avg((a.reply = r.reply)::int) same_word_for_word,
                   avg((contains({mine}, {theirs}) or contains({theirs}, {mine}))::int) one_inside_the_other,
                   avg(jaro_winkler_similarity({mine}, {theirs})) likeness
            from answers a join answers r using (corpus, id)
            where r.level = ? and a.level != r.level group by a.level order by same desc""", [reference])

    def same_reply_by_corpus(self, reference: str) -> list[dict]:
        """As same_reply, per level and corpus."""
        return self.query(rf"""
            select a.level, a.corpus, count(*) n,
                   avg(({self.SAME_FORM.format('a.reply')} = {self.SAME_FORM.format('r.reply')})::int) same
            from answers a join answers r using (corpus, id)
            where r.level = ? and a.level != r.level group by a.level, a.corpus order by a.level, a.corpus""",
                          [reference])

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
