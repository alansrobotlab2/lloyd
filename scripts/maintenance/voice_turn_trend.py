#!/usr/bin/env python3
"""Per-stage latency percentiles over the spoken turns that were actually kept.

Reads `<data root>/voice/turns.jsonl` (`app/voice_turns.py`, written one row per
spoken reply by `agent-services/livekit_worker.py`) and prints, for every stage of
`voice.timeline.STAGES` plus the `eos→audio` total and `max_gap_s`: **N, then the
percentiles**. N is the whole point of the tool. Before #2273 the only numbers
anyone had about voice latency were `e2e_voice.py`'s — sampled from a synthetic
rig — and the reason a rig number could stand in for production is that no one
could say how many production turns a claim rested on. An aggregate over a handful
of turns is not a trend, and an aggregate over zero turns is not a green verdict,
so every line here carries its denominator and the verdict is withheld below a
declared floor rather than computed anyway.

Split by `label`, because the store's own population is not one population. A
typed turn spoken out loud (`typed:user`) has no `speech_end`, no `vad_close` and
no `asr_done`: measured over `~/lloyd-data/logs/lloyd-agent-worker.log` on
2026-10-06, 7 of the 8 `[latency]` lines the worker had ever emitted were typed
turns and all 7 printed `eos→audio=?`. Pool them and the post-VAD percentiles are
silently computed over a handful of microphone turns while the TTS-side ones have
hundreds — the survivorship version of the zero-denominator mistake. So each
label group gets its own table, its own N and its own verdict.

Bounds are declared here, in the script that grades them, not in prose:
`MAX_GAP_P90_BOUND_S` is the voice latency plan's 8 s bound on the silence inside a
tool turn (the 2026-09-24 turn that prompted it was silent for 44 s), and
`EOS_TO_AUDIO_P90_BOUND_S` is deliberately `None` — a bound invented before the
first week of rows is a remembered number, not a measured one, so the report says
`not declared yet (owed #2273)` beside every `eos→audio` percentile until a real
week supplies it.

    python scripts/maintenance/voice_turn_trend.py [--store PATH] [--min-rows N]
                                                  [--json]

Exit 0 with every graded population green or abstained; 1 if any graded
population is red — which is what makes a bad night visible in an autonomy run log
without anyone reading the table.

Not here, on purpose: no artifact file (the store is the artifact, and a second
unbounded store beside it is the next retention-sweep finding), no model calls, no
GPU. Writing a dated report beside the other eval baselines is the owed half of
#2273, once the floor is reachable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# Appended, not inserted: `app/` holds top-level modules (`app/paths.py`,
# `app/config.py`) that would shadow a package of the same name for anything else
# in this process, and this script has no view of them to begin with.
for _p in (REPO_ROOT, REPO_ROOT / "agent-services"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from app import voice_turns  # noqa: E402
from voice.timeline import STAGES  # noqa: E402  (stdlib-only; no LiveKit, no venv)

#: The declared floor, in rows, per label population. 30 because that is the
#: number the item's own acceptance clause declares, and it is a floor and not an
#: estimate: at ~0.5 spoken rows a day (measured 2026-10-06 over
#: `~/lloyd-data/logs/lloyd-agent-worker.log`, 8 `[latency]` lines across
#: 2026-09-22 → 2026-10-05), 30 rows of ANY population is weeks away and 30
#: microphone turns is months. Reporting `abstain` with N printed is the honest
#: output for the next stretch of nights, and says so in the run log instead of
#: printing a p90 computed over four turns.
MIN_ROWS_FOR_VERDICT = 30

#: The plan's bound on the longest silence inside one spoken reply, seconds. From
#: the voice latency plan, and the incident that set it: the 2026-09-24 tool turn
#: ran twenty tool calls and said nothing for 44 s.
MAX_GAP_P90_BOUND_S: Optional[float] = 8.0
#: `None` is not "no bound matters"; it is "this bound is owed". The item's step 5
#: says set it from the first real week's numbers, so until rows exist the report
#: prints it as undeclared rather than inventing 2.0 s from the rig's folklore.
EOS_TO_AUDIO_P90_BOUND_S: Optional[float] = None

#: What each population's verdict can be.
VERDICT_ABSTAIN = "abstain"
VERDICT_GREEN = "green"
VERDICT_RED = "red"

#: The three flags a row carries, in the order the report prints them. The keys are the
#: row's JSON fields (`app/voice_turns.py` writes them from the worker's own state); the
#: labels are what a human reads beside the count. One list owns both so a fourth flag is
#: one edit, and the printed line always carries all three even at zero.
_FLAG_FIELDS = ("interrupted", "queued_behind", "tools_ran")
_FLAG_LABELS = {"interrupted": "interrupted",
                "queued_behind": "queued behind",
                "tools_ran": "tools ran"}


def _percentile(values: list[float], q: float) -> Optional[float]:
    """The `q`-th percentile of `values`, or None for an empty sample.

    Linear interpolation between the two nearest ranks — the same definition
    `numpy.percentile` uses with its default `method="linear"`, so a number in this
    table and a number recomputed from the same rows in numpy agree. That matters
    because the table is the thing a latency fix gets graded on: an off-by-one
    interpolation is a "regression" nobody can reproduce.

    `q` is a fraction (0.5, 0.9, 0.95), not a percent.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _numbers(rows: Iterable[dict], key: str) -> list[float]:
    """Every `key` in `rows` that is really a number.

    A bool is rejected explicitly: `isinstance(True, int)` is true in Python, and
    a row with `"eos_to_audio": true` would otherwise enter the sample as a 1.0 and
    move a p50. A missing or null value is not a zero and is not entered at all —
    it is counted as missing by the caller's N.
    """
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.append(float(value))
    return out


def _stage_numbers(rows: Iterable[dict], stage: str) -> list[float]:
    """One stage's deltas, from rows whose `stages` is actually an object.

    A row written by a future schema that put the stages somewhere else yields
    nothing here rather than raising, and shows up as a missing stage — the shape
    that keeps a schema drift visible in the missing-stage rate instead of killing
    the nightly job at 03:00.
    """
    out = []
    for row in rows:
        stages = row.get("stages")
        if not isinstance(stages, dict):
            continue
        value = stages.get(stage)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.append(float(value))
    return out


def _share(count: int, total: int) -> float:
    """`count / total`, or 0.0 for an empty population.

    A denominator of zero is reported as 0.0 % and never as a division error, and
    never as a verdict: the caller prints N beside it, and N is the number that
    says whether the share means anything.
    """
    return 0.0 if total <= 0 else count / total


def _missing_stage_cells(rows: list[dict]) -> tuple[int, int]:
    """`(absent, possible)` stage cells across `rows` — the missing-stage rate.

    `possible` is `12 × rows`, the cells a complete microphone turn would fill. A
    spoken turn that never replied (the gate rejected it, a barge-in cancelled it)
    has most of them absent, and a report that only ever counted the turns that
    produced all twelve would be grading its own survivors.
    """
    possible = len(STAGES) * len(rows)
    present = sum(len(row.get("stages") or {}) if isinstance(row.get("stages"), dict)
                  else 0 for row in rows)
    return max(possible - min(present, possible), 0), possible


def _flags(rows: list[dict]) -> dict:
    """How many rows carry each flag, as counts and shares of the population."""
    out = {}
    for key in ("interrupted", "queued_behind", "tools_ran"):
        n = sum(1 for row in rows if row.get(key) is True)
        out[key] = {"n": n, "share": _share(n, len(rows))}
    return out


def _stage_row(stage: str, rows: list[dict]) -> dict:
    """One table row: the sample size first, then the percentiles of that sample."""
    values = _stage_numbers(rows, stage)
    return {"n": len(values), "missing": len(rows) - len(values),
            "p50": _percentile(values, 0.5), "p90": _percentile(values, 0.9),
            "p95": _percentile(values, 0.95)}


def summarise(rows: list[dict], *, min_rows: int = MIN_ROWS_FOR_VERDICT) -> dict:
    """Everything one label population says about itself, as data.

    The verdict only ever grades a bound that is declared AND has a sample: an
    `eos→audio` bound of `None` cannot make a night red, and a `max_gap_s`
    percentile over zero rows cannot either. Below `min_rows` the population
    abstains — the word, printed, never a silent absence of a line.
    """
    n = len(rows)
    stages = {stage: _stage_row(stage, rows) for stage in STAGES}
    eos = _numbers(rows, "eos_to_audio")
    gap = _numbers(rows, "max_gap_s")
    absent, possible = _missing_stage_cells(rows)
    tools = [row for row in rows if row.get("tools_ran") is True]
    no_tools = [row for row in rows if row.get("tools_ran") is not True]

    graded: list[dict] = []
    for field, bound, sample in (("eos_to_audio", EOS_TO_AUDIO_P90_BOUND_S, eos),
                                 ("max_gap_s", MAX_GAP_P90_BOUND_S, gap)):
        p90 = _percentile(sample, 0.9)
        if bound is None or p90 is None:
            continue
        graded.append({"field": field, "p90": p90, "bound": bound,
                       "breached": p90 > bound})

    verdict = VERDICT_ABSTAIN if n < min_rows else (
        VERDICT_RED if any(item["breached"] for item in graded) else VERDICT_GREEN)
    return {
        "n": n,
        "min_rows_for_verdict": min_rows,
        "verdict": verdict,
        "stages": stages,
        "eos_to_audio": {"n": len(eos), "missing": n - len(eos),
                         "p50": _percentile(eos, 0.5), "p90": _percentile(eos, 0.9),
                         "p95": _percentile(eos, 0.95)},
        "max_gap_s": {"n": len(gap), "missing": n - len(gap),
                      "p50": _percentile(gap, 0.5), "p90": _percentile(gap, 0.9),
                      "p95": _percentile(gap, 0.95)},
        "missing_stage_cells": {"absent": absent, "possible": possible,
                               "rate": _share(absent, possible)},
        "flags": _flags(rows),
        "by_tools": {"ran": _split_by_tools(tools), "absent": _split_by_tools(no_tools)},
        "breaches": [item for item in graded if item["breached"]],
        "graded": graded,
    }


def _split_by_tools(rows: list[dict]) -> dict:
    """`eos→audio` over the rows that ran tools and the ones that did not.

    The two are different work — a tool turn's `eos→audio` includes however long
    the tools took — and the item's acceptance names them separately: a fix aimed
    at the gap during tool work (#1163-class) has to show up in the population it
    claims to move, not in the pooled number where it can hide behind the other.
    """
    eos = _numbers(rows, "eos_to_audio")
    return {"n": len(rows), "eos_to_audio_n": len(eos),
            "p50": _percentile(eos, 0.5), "p90": _percentile(eos, 0.9)}


def populations(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows grouped by `label`, which is the only split the store earns.

    `""` (a row with no label) keeps its own group rather than joining another
    one's — an unlabelled population is a fact about the writer, and pooling it
    into `voice` would make the microphone table a table about something else.
    """
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get("label") or ""), []).append(row)
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _fmt(value: Optional[float], spec: str = "{:.3f}") -> str:
    """A number, or `-` for a sample that does not exist. Never `None`, and never
    a `0` standing in for one."""
    return "-" if value is None else spec.format(value)


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100.0:.1f}%"


def report_lines(store: Path, summaries: dict[str, dict], *, rows_read: int,
                 malformed: int, min_rows: int = MIN_ROWS_FOR_VERDICT) -> tuple[list[str], int]:
    """The printed report and the exit code (1 if any graded population is red).

    Every line that carries a percentile carries its N on the same line, and the
    header states the denominators the tables are missing: the malformed count and
    the missing-stage rate. An absent store reads as `0 row(s) read` and an
    abstention, not as an error — a box that has not spoken yet is not broken, and
    a nightly job that reddens on a quiet week stops being read.
    """
    lines = [f"[voice-turn-trend] store: {store}",
             f"[voice-turn-trend] {rows_read} row(s) read, "
             f"{malformed} malformed line(s) skipped"]
    if not summaries:
        lines.append(f"[voice-turn-trend] verdict: {VERDICT_ABSTAIN} — no rows in "
                     f"the store; nothing is graded, and no bound is green")
        return lines, 0

    lines.append(f"[voice-turn-trend] floor for a verdict: {min_rows} row(s) per "
                 f"label population")
    exit_code = 0
    for label, summary in summaries.items():
        n = summary["n"]
        lines.append("")
        lines.append(f"── population {label or '(no label)'}: {n} row(s) ──")
        lines.append(f"  {'stage':<16} {'N':>5} {'miss':>6} "
                     f"{'p50':>8} {'p90':>8} {'p95':>8}")
        for stage in STAGES:
            stage_row = summary["stages"][stage]
            lines.append(
                f"  {stage:<16} {stage_row['n']:>5} "
                f"{_pct(_share(stage_row['missing'], n)):>6} "
                f"{_fmt(stage_row['p50']):>8} {_fmt(stage_row['p90']):>8} "
                f"{_fmt(stage_row['p95']):>8}")
        eos = summary["eos_to_audio"]
        lines.append(f"  {'eos→audio':<16} {eos['n']:>5} "
                     f"{_pct(_share(eos['missing'], n)):>6} "
                     f"{_fmt(eos['p50']):>8} {_fmt(eos['p90']):>8} "
                     f"{_fmt(eos['p95']):>8}")
        gap = summary["max_gap_s"]
        lines.append(f"  {'max_gap_s':<16} {gap['n']:>5} "
                     f"{_pct(_share(gap['missing'], n)):>6} "
                     f"{_fmt(gap['p50']):>8} {_fmt(gap['p90']):>8} "
                     f"{_fmt(gap['p95']):>8}")
        cells = summary["missing_stage_cells"]
        lines.append(f"  missing stages: {cells['absent']} of {cells['possible']} "
                     f"cells ({_pct(cells['rate'])})")
        # All three flags every night, each with the population's N beside it: the line
        # proves the field is being read, and `interrupted 0 of 40` is a different fact
        # from a metric nobody wired up — the zero-denominator class rule again.
        flags = summary["flags"]
        lines.append("  " + " | ".join(
            f"{_FLAG_LABELS[name]} {flags[name]['n']} of {n} "
            f"({_pct(flags[name]['share'])})"
            for name in _FLAG_FIELDS))
        ran, absent = summary["by_tools"]["ran"], summary["by_tools"]["absent"]
        lines.append(f"  tools ran: n={ran['n']} eos→audio p50 {_fmt(ran['p50'])}s "
                     f"p90 {_fmt(ran['p90'])}s | no tools: n={absent['n']} "
                     f"p50 {_fmt(absent['p50'])}s p90 {_fmt(absent['p90'])}s")
        lines.append(f"  bounds: max_gap_s p90 <= {_fmt(MAX_GAP_P90_BOUND_S, '{:.1f}')}s"
                     + (f" | eos→audio p90 <= "
                        f"{_fmt(EOS_TO_AUDIO_P90_BOUND_S, '{:.1f}')}s"
                        if EOS_TO_AUDIO_P90_BOUND_S is not None else
                        " | eos→audio p90: not declared yet (owed #2273)"))
        if summary["verdict"] == VERDICT_ABSTAIN:
            lines.append(f"  verdict: {VERDICT_ABSTAIN} — n={n} is below the declared "
                         f"floor of {min_rows}; no bound graded, this is not a "
                         f"healthy night, it is an unmeasured one")
        else:
            if summary["verdict"] == VERDICT_RED:
                exit_code = 1
                breaches = ", ".join(
                    f"{item['field']} p90 {_fmt(item['p90'])}s > bound "
                    f"{_fmt(item['bound'], '{:.1f}')}s" for item in summary["breaches"])
                lines.append(f"  verdict: {VERDICT_RED} — {breaches}")
            else:
                lines.append(f"  verdict: {VERDICT_GREEN} — n={n}, "
                             f"{len(summary['graded'])} bound(s) graded, none breached")
    return lines, exit_code


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None,
                    help="the turns.jsonl to read (default: this tree's data root)")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS_FOR_VERDICT,
                    help=f"rows a population needs before it is graded "
                         f"(default {MIN_ROWS_FOR_VERDICT}); for a rig or a test, "
                         f"not for the nightly job — the floor is the verdict")
    ap.add_argument("--json", action="store_true",
                    help="print one JSON object instead of the table")
    args = ap.parse_args(argv)

    store = Path(args.store) if args.store else voice_turns.turns_path()
    rows: list[dict] = []
    malformed = 0
    for _number, row in voice_turns.read_rows(store):
        if row is None:
            malformed += 1
        else:
            rows.append(row)

    # One pass of arithmetic per population, shared by the table and by `--json`,
    # so a number on stdout and a number in the JSON object are one measurement
    # and not two computations that can disagree about the same store.
    summaries = {label: summarise(group, min_rows=args.min_rows)
                 for label, group in populations(rows).items()}
    red = any(summary["verdict"] == VERDICT_RED for summary in summaries.values())

    if args.json:
        print(json.dumps({
            "store": str(store), "rows_read": len(rows),
            "malformed_lines": malformed,
            "min_rows_for_verdict": args.min_rows,
            "bounds": {"max_gap_s_p90_s": MAX_GAP_P90_BOUND_S,
                       "eos_to_audio_p90_s": EOS_TO_AUDIO_P90_BOUND_S},
            "populations": summaries,
        }, ensure_ascii=False))
    else:
        lines, _code = report_lines(store, summaries, rows_read=len(rows),
                                    malformed=malformed, min_rows=args.min_rows)
        for line in lines:
            print(line)
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
