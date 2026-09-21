"""The sample an experiment is measured on (config.Sample, foqlens.sample): the hard questions and the ordinary ones.

Hard is what only the top rung answers while the lower ones fail it or replace it with something else; ordinary is
what the lower rungs answer too. Both sides are drawn from a run the judge has already read, so no model runs here.
The ordinary side takes the shares by corpus the hard side came out with.

    uv run python scripts/draw_sample.py --config configs/e006-sample.toml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from foqlens import config, corpora, runlog, sample
from foqlens.runlog import stage
from foqlens.runs import RunFiles

LOG = logging.getLogger("foqlens.draw_sample")


def sets(cfg: config.Sample) -> dict[str, list[tuple[str, str]]]:
    """The hard and the ordinary questions of `cfg`, each as [corpus, id] pairs."""
    run = RunFiles(answers=Path(cfg.answers), judged=Path(cfg.judged) if cfg.judged else None)
    corpora = list(cfg.corpora)
    hard_rows = run.questions_by_verdicts([cfg.top], list(cfg.lower), corpora, view=cfg.view)
    ordinary_rows = run.questions_by_verdicts([cfg.top, *cfg.lower], [], corpora, view=cfg.view)
    hard_all = [(r["corpus"], r["id"]) for r in hard_rows]
    ordinary_all = [(r["corpus"], r["id"]) for r in ordinary_rows]
    LOG.info("candidates", extra={"hard": len(hard_all), "ordinary": len(ordinary_all)})
    hard = sample.pick(hard_all, cfg.hard, cfg.seed)
    ordinary = sample.pick(ordinary_all, cfg.ordinary, cfg.seed, wanted=sample.shares(hard))
    drawn = {"hard": hard, "ordinary": ordinary}
    if cfg.paraphrased:
        for side in ("hard", "ordinary"):
            drawn[f"{side}-paraphrased"] = sample.pick(drawn[side], cfg.paraphrased, cfg.seed + 1)
    return drawn


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = config.read(args.config, config.Sample)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    run = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    runlog.setup(out / "logs" / f"draw_sample-{run}.jsonl", {"run": run, "script": "draw_sample"})
    with stage(LOG, "draw"):
        drawn = sets(cfg)
    with stage(LOG, "texts"):
        texts = {corpus: {row.id: row.question for row in corpora.read(corpus)[0]} for corpus in cfg.corpora}
    for name, questions in drawn.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps([list(q) for q in questions], ensure_ascii=False, indent=1), encoding="utf-8")
        written = out / f"{name}-questions.jsonl"
        written.write_text("".join(json.dumps({"corpus": c, "id": i, "question": texts[c][i]}, ensure_ascii=False)
                                   + "\n" for c, i in questions), encoding="utf-8")
        counts: dict[str, int] = {}
        for corpus, _ in questions:
            counts[corpus] = counts.get(corpus, 0) + 1
        LOG.info("written", extra={"set": name, "questions": len(questions), "by_corpus": counts, "path": str(path)})
    return out


if __name__ == "__main__":
    main()
