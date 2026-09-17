"""Infrastructure: the station's log - what the runs cost the card - built from the run summaries.

A record is any run summary with a "gpu" field written by GpuMonitor: flat, or one per phase of the
run ({"masks": ..., "eval": ...}). A session of the GPU tests writes a record of its own into
runs/station/tests/. A record entered by hand - a run stopped before it wrote its summary - lives in
runs/station/ in the same shape, with a "note" saying so; it may carry "date" and "estimated".

Invariant: an entry's time is the sum of its phases' wall time; a summary written before the monitor
kept a clock (2026-09-14) counts its samples at SAMPLE_S each, and the entry is marked estimated.
Invariant: a mean over phases is weighted by their time, and a peak is the maximum over them.
"""

from __future__ import annotations

from dataclasses import dataclass

BEGIN = "<!-- station-log:begin -->"
END = "<!-- station-log:end -->"
# GpuMonitor's default interval; each nvidia-smi call adds ~0.07 s to it, hence "estimated"
SAMPLE_S = 1.0


@dataclass(frozen=True)
class Entry:
    date: str
    run: str
    seconds: float
    estimated: bool
    utilization: float | None
    temperature_mean: float | None
    temperature_peak: float | None
    power_peak: float | None
    cooling_breaks: int | None
    note: str


def phases(gpu: dict) -> list[dict]:
    """The phases of a summary's GPU field: a flat summary is one phase, a dict of summaries several."""
    if any(isinstance(v, dict) for v in gpu.values()):
        return [p for p in gpu.values() if isinstance(p, dict)]
    return [gpu] if gpu else []


def seconds_of(phase: dict) -> float:
    return phase["wall_s"] if "wall_s" in phase else phase.get("samples", 0) * SAMPLE_S


def entry(record: dict, run: str, date: str) -> Entry | None:
    """One line of the log; None for a summary that never watched the card."""
    parts = [p for p in phases(record.get("gpu") or {}) if seconds_of(p) > 0]
    if not parts:
        return None

    def mean(key: str) -> float | None:
        known = [p for p in parts if key in p]
        return sum(p[key] * seconds_of(p) for p in known) / sum(seconds_of(p) for p in known) if known else None

    def peak(key: str) -> float | None:
        return max((p[key] for p in parts if key in p), default=None)

    started = next((p["started"] for p in parts if "started" in p), None)
    return Entry(
        date=started[:10] if started else record.get("date", date),
        run=record.get("run", run),
        seconds=sum(seconds_of(p) for p in parts),
        estimated=record.get("estimated", False) or any("wall_s" not in p for p in parts),
        utilization=mean("utilization_mean"),
        temperature_mean=mean("temperature_mean_c"),
        temperature_peak=peak("temperature_peak_c"),
        power_peak=peak("power_peak_w"),
        cooling_breaks=(record.get("pacer") or {}).get("cooling_breaks"),
        note=record.get("note", ""),
    )


def session_record(gpu: dict, outcomes: dict[str, int], files: list[str]) -> dict:
    """The record a session of the GPU tests leaves for the log: which test files ran, its monitor and how it ended."""
    return {"run": f"GPU tests: {', '.join(sorted(files))}", "gpu": gpu,
            "note": ", ".join(f"{n} {k}" for k, n in outcomes.items() if n)}


def duration(seconds: float, estimated: bool) -> str:
    hours, minutes = divmod(round(seconds / 60), 60)
    text = f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"
    return ("~" if estimated else "") + text


def value(v: float | None, unit: str) -> str:
    return "-" if v is None else f"{v:.0f}{unit}"


def log_table(entries: list[Entry]) -> str:
    rows = ["| Date | Run | Time | Utilization, mean | Temperature, mean / peak | Power, peak | Cooling breaks | Note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in entries:
        rows.append(f"| {e.date} | {e.run} | {duration(e.seconds, e.estimated)} | {value(e.utilization, '%')} | "
                    f"{value(e.temperature_mean, '')} / {value(e.temperature_peak, ' °C')} | {value(e.power_peak, ' W')} | "
                    f"{'-' if e.cooling_breaks is None else e.cooling_breaks} | {e.note} |")
    return "\n".join(rows)


def peaks_table(entries: list[Entry]) -> str:
    def top(key) -> Entry | None:
        known = [e for e in entries if key(e) is not None]
        return max(known, key=key, default=None)

    def when(e: Entry | None) -> str:
        return "-" if e is None else f"{e.date}, {e.run}"

    hot, power, longest = top(lambda e: e.temperature_peak), top(lambda e: e.power_peak), top(lambda e: e.seconds)
    total = sum(e.seconds for e in entries)
    rows = ["| | Value | When |", "| --- | --- | --- |",
            f"| Temperature | {value(hot and hot.temperature_peak, ' °C')} | {when(hot)} |",
            f"| Power | {value(power and power.power_peak, ' W')} | {when(power)} |",
            f"| Longest run | {'-' if longest is None else duration(longest.seconds, longest.estimated)} | {when(longest)} |",
            f"| GPU time in total | {duration(total, any(e.estimated for e in entries))} | {len(entries)} entries |"]
    return "\n".join(rows)


def render(entries: list[Entry]) -> str:
    ordered = sorted(entries, key=lambda e: (e.date, e.run))
    return f"## Log\n\n{log_table(ordered)}\n\n## Peaks\n\n{peaks_table(ordered)}"


def splice(doc: str, body: str) -> str:
    """The document with what stands between BEGIN and END replaced by `body`."""
    start, end = doc.find(BEGIN), doc.find(END)
    if start < 0 or end < start:
        raise ValueError(f"the document has no {BEGIN} ... {END} section")
    return f"{doc[:start + len(BEGIN)]}\n{body}\n{doc[end:]}"
