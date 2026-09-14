"""Which subjects this model actually knows - calibrating the corpus instead of guessing it.

A layout can only be measured on questions the model answers from knowledge. On the four subjects the
bench has used, bf16 is right in all six orders of the options on 31% of the questions, and on the
maths subject on 3% - those questions are answered by calculating, and the metric gives the model one
token to do it in (runs/reference/shuffle-answers). Everything else is a coin toss that no layout can
move, and it drowns every comparison in noise.

So: ask bf16 every question of a candidate subject in several orders of its options, and count the
core - the questions it gets right in *every* order. A subject with a large core is a subject where a
layout's damage is visible; a subject with a small one measures nothing.

Also reported per subject: the share of questions whose options are all numbers or formulas, which is
the tell of a question answered by calculation rather than recall.

The free-answer corpora (foqlens.corpora) are answered in the model's own words and scored as SQuAD
scores them, with a passage or without one; there the core is the exact-match share.

Writes <out>/<model>/summary.json; every invocation replaces it, so a run of other subjects takes its own --out.

    uv run python scripts/corpus_calibration.py
    uv run python scripts/corpus_calibration.py --subjects nutrition marketing --limit 20
    uv run python scripts/corpus_calibration.py --subjects triviaqa nq_open --limit 102 --out runs/reference/corpus-closed-book
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from foqlens import corpora
from foqlens import model as fm
from foqlens.evaluate import LETTERS, LetterChoice, Question, letter_ids, letter_logprobs_batch, mc_prompt
from foqlens.extractive import NO_ANSWER
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import write_json
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.quant import Level

REDUX = "edinburgh-dawg/mmlu-redux-2.0"
REDUX_REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
# Everyday knowledge and reasoning rather than school calculation - what a 2B model is asked to hold.
# The four the bench already uses are included so the new numbers can be read against the known ones.
CANDIDATES = (
    "miscellaneous", "nutrition", "human_aging", "marketing", "sociology", "world_religions",
    "public_relations", "high_school_psychology", "logical_fallacies", "global_facts",
    "high_school_biology", "high_school_geography", "prehistory", "high_school_mathematics",
)


# Corpora built for base checkpoints, where reasoning is asked for rather than school recall. Google
# publishes no numbers for the pretrained E2B - only instruction-tuned ones, in thinking mode - so
# what this model can do on them is measured here rather than looked up.
EXTERNAL = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    "openbookqa": ("allenai/openbookqa", "main/test-00000-of-00001.parquet"),
    "hellaswag": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    "reclor": ("metaeval/reclor", "val.json"),
}


# The free-answer corpora (foqlens.corpora): the same prompt shape and the same score with a passage and
# without one, so the regimes differ only in where the answer lies.
QA_SHOTS = 2  # examples shown to a base checkpoint so it answers in the expected shape
QA_PACE = 8   # answers written between two rests of the pacer: one prompt at a time, a few seconds of GPU


def _revision(repo: str) -> str:
    """The dataset's current commit, recorded in the summary so a run can be repeated exactly."""
    return HfApi().dataset_info(repo).sha


def answer_questions(bench, rows: list[corpora.Row], shots: tuple) -> list[dict]:
    """What the model writes for each passage, scored as SQuAD scores it."""
    from foqlens.evaluate import Question
    from foqlens.extractive import ContextQA, qa_prompt

    passages = {}
    questions = []
    for r in rows:
        prompt = qa_prompt(r.context, r.question, shots)
        passages[prompt] = r.answers
        questions.append(Question("x", prompt, 0))
    metric = ContextQA(passages=passages, shots=shots)
    rows = []
    for start in range(0, len(questions), QA_PACE):
        with bench.throttle.batch():
            rows += metric.score(bench.model, bench.tokenizer, questions[start:start + QA_PACE])
    return rows


def external_rows(name: str, limit: int | None) -> tuple[list[dict], str]:
    """One external corpus as {question, choices, answer}, keeping only its four-option questions."""
    repo, path = EXTERNAL[name]
    revision = _revision(repo)
    local = hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    if name == "reclor":
        raw = json.load(open(local, encoding="utf-8"))
        rows = [{"question": r["context"] + "\n" + r["question"], "choices": r["answers"],
                 "answer": int(r["label"])} for r in raw]
    else:
        table = pq.read_table(local).to_pylist()
        if name == "hellaswag":
            rows = [{"question": "Which ending fits?\n" + r["ctx"], "choices": list(r["endings"]),
                     "answer": int(r["label"])} for r in table if str(r["label"]).isdigit()]
        else:
            stem = "question_stem" if name == "openbookqa" else "question"
            rows = []
            for r in table:
                labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
                if r["answerKey"] not in labels:
                    continue
                rows.append({"question": r[stem], "choices": texts, "answer": labels.index(r["answerKey"])})
    rows = [r for r in rows if len(r["choices"]) == len(LETTERS)]
    return (rows[:limit] if limit else rows), f"{repo}@{revision[:8]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--subjects", nargs="+", default=list(CANDIDATES) + list(EXTERNAL) + list(corpora.CORPORA),
                        help="MMLU-Redux subjects, the multiple-choice corpora of EXTERNAL and the free-answer "
                             "corpora of foqlens.corpora")
    parser.add_argument("--orders", type=int, default=6, help="orders of the options a question is asked in")
    parser.add_argument("--limit", type=int, default=None, help="questions per subject, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--out", type=Path, default=Path("runs/reference/corpus"))
    return parser.parse_args(argv)


def subject_rows(subject: str, limit: int | None) -> list[dict]:
    """The clean questions of one MMLU-Redux subject, as build_prompts.py reads them."""
    path = hf_hub_download(REDUX, f"{subject}/data-00000-of-00001.arrow", repo_type="dataset",
                           revision=REDUX_REVISION)
    with open(path, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    rows = table.select(["question", "choices", "answer", "error_type"]).to_pylist()
    clean = [r for r in rows if r["error_type"] == "ok" and len(r["choices"]) == len(LETTERS)]
    return clean[:limit] if limit else clean


def answered_by_calculating(row: dict) -> bool:
    """Every option a number or a formula - the tell of a question the model would have to compute."""
    return all(any(ch.isdigit() for ch in c) and sum(ch.isalpha() for ch in c) <= len(c) / 2
               for c in row["choices"])


def ask(bench, ids, rows: list[dict], order: np.ndarray, batch: int) -> np.ndarray:
    """Whether bf16 gets each question right with the options in this order."""
    questions = [Question("x", mc_prompt(r["question"].strip(), [r["choices"][i].strip() for i in order]),
                          int(np.flatnonzero(order == r["answer"])[0])) for r in rows]
    right = np.empty(len(questions), dtype=bool)
    for start in range(0, len(questions), batch):
        chunk = questions[start:start + batch]
        with bench.throttle.batch():
            logprobs = letter_logprobs_batch(bench.model, bench.tokenizer, [q.prompt for q in chunk], ids)
        right[start:start + batch] = logprobs.argmax(axis=1) == [q.answer for q in chunk]
    return right


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    orders = [np.arange(len(LETTERS))] + [rng.permutation(len(LETTERS)) for _ in range(args.orders - 1)]
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)
    bench.ctl.set_all(Level.BF16)
    ids = letter_ids(bench.tokenizer)

    with GpuMonitor() as gpu:
        subjects = {}
        progress = Progress(len(args.subjects), "subject")
        for subject in args.subjects:
            if subject in corpora.CORPORA:
                rows, source = corpora.read(subject, args.limit)
                kind = "context_qa" if corpora.CORPORA[subject].passage else "closed_book"
                shots = tuple((r.context, r.question, (r.answers or (NO_ANSWER,))[0]) for r in rows[:QA_SHOTS])
                scored = answer_questions(bench, rows[QA_SHOTS:], shots)
                em = np.array([s["exact_match"] for s in scored])
                f1 = np.array([s["f1"] for s in scored])
                subjects[subject] = {
                    "source": str(source), "kind": kind, "questions": len(scored),
                    "exact_match": float(em.mean()), "f1": float(f1.mean()), "core": float(em.mean()),
                    "unanswerable": float(np.mean([not r.answers for r in rows[QA_SHOTS:]])),
                }
                print(f"  {subject:30} exact {em.mean():5.1%}  F1 {f1.mean():5.1%}  n={len(scored)}", flush=True)
                print(progress.step(subject), flush=True)
                continue
            rows, source = ((external_rows(subject, args.limit)) if subject in EXTERNAL
                            else (subject_rows(subject, args.limit), f"{REDUX}@{REDUX_REVISION[:8]}"))
            right = np.array([ask(bench, ids, rows, order, args.batch) for order in orders])  # [orders, questions]
            core = right.all(axis=0)
            calc = np.array([answered_by_calculating(r) for r in rows])
            subjects[subject] = {
                "source": source,
                "kind": LetterChoice.name,
                "questions": len(rows),
                "accuracy": float(right.mean()),
                "core": float(core.mean()),
                "never_right": float((~right.any(axis=0)).mean()),
                "calculation": float(calc.mean()),
                "core_among_knowledge": float(core[~calc].mean()) if (~calc).any() else None,
            }
            print(f"  {subject:30} core {core.mean():5.1%}  accuracy {right.mean():5.1%}  "
                  f"calculation {calc.mean():5.1%}  n={len(rows)}", flush=True)
            print(progress.step(subject), flush=True)

    # Each subject names its own source and kind; the orders exist only where options were reordered.
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "seed": args.seed,
        "gpu": gpu.summary(),
        "pacer": bench.throttle.stats(),
        "subjects": dict(sorted(subjects.items(), key=lambda kv: -kv[1]["core"])),
    }
    if any(s["kind"] == LetterChoice.name for s in subjects.values()):
        summary["orders"] = [o.tolist() for o in orders]
    out = args.out / args.model / "summary.json"
    write_json(out, summary)
    print(f"written {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
