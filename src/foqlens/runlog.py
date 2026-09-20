"""The log of a run: what stage it is in, how long each stage took, how far a long loop has got.

Every module logs to its own logger under "foqlens" (logging.getLogger(__name__)) and knows nothing of where the events
go. Where they go is configuration: `setup` loads configs/logging.toml into logging.config.dictConfig - stderr as text
and a JSON Lines file beside the run's output, the format Loki and Elastic ingest - and a new sink is a handler there,
not code. Every event carries the run's context (RunContext) and its own fields through `extra`, so a sink that keeps
structure (JSON, Loki, OTLP) gets them as fields, not as words inside the message.

A stage logs its start and its end with the seconds it took. The steps of a loop are counted by foqlens.progress and
logged by the module: INFO where the steps are few (layouts, rounds), DEBUG where they are many (batches).

Invariant: a stage logs exactly one start and one end line with its duration: INFO "done", or ERROR "failed" with the
traceback when its body raises - the exception goes on.
Invariant: every event of a run set up here carries the run's context fields.
"""

from __future__ import annotations

import logging
import logging.config
import time
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

CONFIG = Path("configs/logging.toml")


class RunContext(logging.Filter):
    """Adds the run's context - run, script and whatever setup was given - to every event as fields."""

    fields: Mapping[str, str] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.fields.items():
            setattr(record, key, value)
        return True


def setup(log_path: Path, context: Mapping[str, str], config: Path = CONFIG) -> None:
    """Configure the sinks from `config` (dictConfig in TOML), the file sink writing to `log_path`, every event carrying
    `context` (run, script: the labels a run is found by). Once per script."""
    with open(config, "rb") as file:
        table = tomllib.load(file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    table["handlers"]["file"]["filename"] = str(log_path)
    RunContext.fields = dict(context)
    logging.config.dictConfig(table)


@contextmanager
def stage(log: logging.Logger, name: str) -> Iterator[None]:
    """Log the start of a stage and its end with the seconds it took; a failure as an ERROR with its traceback."""
    log.info("start %s", name, extra={"stage": name})
    began = time.perf_counter()
    try:
        yield
    except BaseException:
        seconds = time.perf_counter() - began
        log.exception("failed %s after %.1f s", name, seconds, extra={"stage": name, "seconds": round(seconds, 1)})
        raise
    seconds = time.perf_counter() - began
    log.info("done %s in %.1f s", name, seconds, extra={"stage": name, "seconds": round(seconds, 1)})
