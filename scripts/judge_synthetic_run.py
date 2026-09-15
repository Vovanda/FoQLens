"""The committed judge on 250 plainly right answers and three degradations of each, graded by Claude beforehand."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
from foqlens import corpora
from foqlens import model as fm
from foqlens.evaluate import letter_logprobs_batch
from foqlens.io import answers_path, read_answers
from foqlens.judging import GRADES, ModelJudge, judge_prompt
from foqlens.precision import install
from foqlens.quant import Level
from foqlens.selection import JUDGE_YES

S = Path(sys.argv[1])
R = Path("runs/reference/stage1/e2b-it")
LETTER = {"C": "Correct", "N": "Nearly", "P": "Partial", "R": "Related", "W": "Wrong"}
ORDER = [word for word, _ in GRADES]
BATCH = 16

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
judge = ModelJudge.build(model, tokenizer, ctl, fmt, batch_size=BATCH)
word_ids = [tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in ORDER]

with torch.no_grad():
    p_yes = judge.p_yes([q for _, _, q, *_ in items], [a for *_, a, _ in items], [r for _, _, _, r, _, _ in items])
    ctl.set_all(Level.BF16)
    # The reason: the word after the verdict the judge gave, one decoding step on the same prompt.
    prompts = [judge_prompt(q, a, r, fmt) + (" Yes," if p > JUDGE_YES else " No,")
               for (_, _, q, r, a, _), p in zip(items, p_yes)]
    reason = np.concatenate([np.exp(letter_logprobs_batch(model, tokenizer, prompts[s:s + BATCH], word_ids))
                             for s in range(0, len(prompts), BATCH)])

said = [ORDER[k] for k in reason.argmax(axis=1)]
table = defaultdict(Counter)
accepted = defaultdict(list)
for (c, i, q, r, a, g), p, w in zip(items, p_yes, said):
    table[g][w] += 1
    accepted[g].append(p > JUDGE_YES)
print(f"{len(items)} answers. Rows: Claude's grade; columns: the judge's reason word; last: share the judge accepts")
print(f"{'':9} " + " ".join(f"{w:>8}" for w in ORDER) + "   accepted")
for g in ORDER:
    if accepted[g]:
        print(f"{g:9} " + " ".join(f"{table[g][w]:8}" for w in ORDER) + f"   {np.mean(accepted[g]):.2f}  (n={len(accepted[g])})")
by_corpus = defaultdict(lambda: defaultdict(list))
for (c, *_, g), p in zip(items, p_yes):
    by_corpus[c][g].append(p > JUDGE_YES)
print("\naccepted by corpus: " + " | ".join(ORDER))
for c, gs in by_corpus.items():
    print(f"  {c:22} " + " ".join(f"{np.mean(gs[g]):5.2f}" if gs[g] else "    -" for g in ORDER))
(S / "verdicts.json").write_text(json.dumps(
    [{"corpus": c, "id": i, "answer": a, "claude": g, "p_yes": float(p), "judge": w}
     for (c, i, q, r, a, g), p, w in zip(items, p_yes, said)], indent=0, ensure_ascii=False), encoding="utf-8")
