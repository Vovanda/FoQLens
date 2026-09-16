"""The present judge on 250 plainly right answers and three degradations of each, graded by Claude beforehand."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from foqlens import corpora
from foqlens import model as fm
from foqlens.attention import SPLIT
from foqlens.graph_decode import StaticDecoder
from foqlens.io import answers_path, read_answers
from foqlens.judging import GRADES, NOT_READ, ModelJudge
from foqlens.precision import install

S = Path(sys.argv[1])
R = Path("runs/reference/stage1/e2b-it")
LETTER = {"C": "Correct", "N": "Nearly", "P": "Partial", "R": "Related", "W": "Wrong"}
ORDER = [word for word, _ in GRADES] + [NOT_READ]

sample = json.loads((S / "sample.json").read_text(encoding="utf-8"))
degraded = json.loads((S / "degraded.json").read_text(encoding="utf-8"))
items = []  # (corpus, id, question, references, answer, Claude's grade)
for c, ids in sample.items():
    rows = {r.id: r for r in corpora.read(c)[0]}
    answers = {a.id: a for a in read_answers(answers_path(R / "answers", "bf16", c))}
    assert len(degraded[c]) == len(ids), c
    for i, variants in zip(ids, degraded[c]):
        row = rows[i]
        items.append((c, i, row.question, list(row.answers), answers[i].answer, "Correct"))
        items += [(c, i, row.question, list(row.answers), text, LETTER[g]) for text, g in variants]

model, tokenizer = fm.load(fm.E2B_IT, attn_implementation="sdpa")
ctl = install(model)
fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
judge = ModelJudge(model, tokenizer, ctl, fmt, StaticDecoder(attention=SPLIT))
verdicts = judge.verdicts([q for _, _, q, *_ in items], [a for *_, a, _ in items], [r for _, _, _, r, _, _ in items])

table = defaultdict(Counter)
accepted = defaultdict(list)
for (c, i, q, r, a, g), v in zip(items, verdicts):
    table[g][v.kind] += 1
    accepted[g].append(v.accepted)
print(f"{len(items)} answers. Rows: Claude's grade; columns: the judge's kind of answer; last: share the judge accepts")
print(f"{'':9} " + " ".join(f"{w:>8}" for w in ORDER) + "   accepted")
for g in ORDER:
    if accepted[g]:
        print(f"{g:9} " + " ".join(f"{table[g][w]:8}" for w in ORDER) + f"   {np.mean(accepted[g]):.2f}  (n={len(accepted[g])})")
by_corpus = defaultdict(lambda: defaultdict(list))
for (c, *_, g), v in zip(items, verdicts):
    by_corpus[c][g].append(v.accepted)
print("\naccepted by corpus: " + " | ".join(ORDER))
for c, gs in by_corpus.items():
    print(f"  {c:22} " + " ".join(f"{np.mean(gs[g]):5.2f}" if gs[g] else "    -" for g in ORDER))
(S / "verdicts.json").write_text(json.dumps(
    [{"corpus": c, "id": i, "answer": a, "claude": g, "judge": v.kind, "accepted": v.accepted, "reply": v.reply}
     for (c, i, q, r, a, g), v in zip(items, verdicts)], indent=0, ensure_ascii=False), encoding="utf-8")
