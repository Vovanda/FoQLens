"""Draw 50 plainly right answers per corpus from the frozen corpus and lay them out for Claude to degrade."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from foqlens import corpora
from foqlens.io import answers_path, read_answers, read_verdicts
from foqlens.prompt_variants import SETUPS
from foqlens.selection import FrozenCorpus

R = Path("runs/reference/stage1/e2b-it")
OUT = Path(sys.argv[1])
PER_CORPUS = 50
SEED = 0

sample = {}
lines = []
for c in SETUPS:
    frozen = FrozenCorpus.from_json(json.loads(Path(f"corpus/e2b-it/{c}.json").read_text(encoding="utf-8")))
    rows = {r.id: r for r in corpora.read(c)[0]}
    answers = {a.id: a for a in read_answers(answers_path(R / "answers", "bf16", c))}
    read = read_verdicts(answers_path(R / "verdicts", "bf16", c))
    two_way = set(frozen.two_way)
    pool = [i for i in frozen.kept if i not in two_way and rows[i].answers and answers[i].exact_match == 1.0
            and i in read and str(read[i].reading) == "right"]
    picked = sorted(np.random.default_rng(SEED).choice(len(pool), size=PER_CORPUS, replace=False).tolist())
    sample[c] = [pool[k] for k in picked]
    lines.append(f"== {c} (pool {len(pool)})")
    for n, i in enumerate(sample[c], 1):
        refs = " / ".join(list(dict.fromkeys(rows[i].answers))[:3])
        lines.append(f"{n}. {i} | Q: {' '.join(rows[i].question.split())[:220]} | REF: {refs[:80]} | A: {answers[i].answer}")
(OUT / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
(OUT / "sheet.txt").write_text("\n".join(lines), encoding="utf-8")
print({c: len(v) for c, v in sample.items()})
