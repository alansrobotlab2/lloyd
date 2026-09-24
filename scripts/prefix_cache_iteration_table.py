#!/usr/bin/env python3
"""#520's per-iteration prefix-cache table — committed, boot-aware, per platform.

Backlog #859. Until now this measurement existed only as a heredoc pasted into
a backlog body, and that is the defect #618 filed: an ad-hoc aggregate with no
boot boundary in it. `stats.cache_read` is vLLM's
`prompt_tokens_details.cached_tokens`, and every engine restart empties the
KV cache, so a day-filtered table blends two engines into one column. Run over
2026-09-09 — the day commit `2677dea` landed and restarted the backend mid-day
— the same query returned 76-79%; run over 2026-09-10 and 2026-09-11 it
returned 84-90%. Three answers, one query, and the difference was written up as
a standing defect (#603) rather than as the restart it was.

So the boot cut is the primary output here, not a footnote: this script refuses
to print a number without saying which engine boot it is measuring and how many
sessions it threw away for predating that boot.

What it also pins (#604, and the reason the heredoc could never see the bug):
the **turn-level** stats row has no `iteration` key — 20,179 of 41,287 persisted
rows on 2026-09-11 — while #604 and #618 both filtered on `iteration == 0`,
which matches nothing. A check that filters on the wrong sentinel returns zero
offenders and reads as "fixed" over a store that had 665 rows above 100% cached.
Those rows are therefore counted and printed on their own line, never folded
into an iteration bucket and never dropped.

Usage:
    python3 scripts/prefix_cache_iteration_table.py
    python3 scripts/prefix_cache_iteration_table.py --boot 2026-09-11T06:00:00
    python3 scripts/prefix_cache_iteration_table.py --boot-from-landing
    python3 scripts/prefix_cache_iteration_table.py --json
    python3 scripts/prefix_cache_iteration_table.py --sessions-dir /tmp/fixtures
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.paths import SESSIONS_DIR  # noqa: E402 — the data root, not the tree

SUPERVISOR_CONF = ROOT / "agent-services" / "supervisor" / "supervisord.conf"
# The same resolution as scripts/automod/state.py, restated rather than
# imported so this read-only report does not pull the automod package in.
STATE_DIR = Path(os.environ.get(
    "LLOYD_AUTOMOD_STATE", Path.home() / ".local" / "state" / "lloyd-automod"))
PROMOTIONS = STATE_DIR / "promotions.jsonl"

# Whose restart empties the prefix cache. The backend restarts on every landing
# and the engine restarts on an engine change; a vLLM restart drops every KV
# block with it, so the LATEST start across these is when cache went cold.
BOOT_UNITS = ("lloyd-backend", "agent-llm-primary", "agent-llm-secondary")
USER_HZ = 100

#: Iterations at or past this share one bucket. #520's cliff sat at 8, and the
#: long Bash-heavy turns that reach 9+ are few enough that splitting them out
#: would report a sample count of 3 as a trend.
MERGE_FROM = 8

#: The engine's own guarantee for one request; a per-iteration row that breaks
#: it is a parser or engine defect and says so.
def _ratio(cached: int, prompt: int) -> float:
    return cached / prompt if prompt else 0.0


def _int(source: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


@dataclass
class Bucket:
    """One iteration's folded samples across every session in scope."""

    iteration: int
    samples: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @property
    def fraction(self) -> float:
        return _ratio(self.cached_tokens, self.prompt_tokens)

    @property
    def label(self) -> str:
        return f"{self.iteration}+" if self.iteration == MERGE_FROM else str(
            self.iteration)


@dataclass
class Fall:
    from_iteration: int
    to_iteration: int
    fall: float

    @property
    def label(self) -> str:
        return f"{self.from_iteration} -> {self.to_iteration}"


def largest_fall_between(buckets: list[Bucket]) -> Fall | None:
    """The biggest cached-fraction drop between ADJACENT iteration numbers.

    Deliberately not "the biggest drop between any two buckets": buckets 3 and
    8 can be populated by disjoint session sets, and the difference between
    them is a traffic-mix artefact — which is exactly the reasoning that
    manufactured #603's false premise. The metric that predicted the `keep=6`
    cliff was iteration k against iteration k-1, so that is what this is, and
    a pair with a missing neighbour is not compared at all.
    """
    by_iteration = {b.iteration: b for b in buckets}
    best: Fall | None = None
    for iteration in sorted(by_iteration):
        nxt = by_iteration.get(iteration + 1)
        if nxt is None:
            continue
        drop = by_iteration[iteration].fraction - nxt.fraction
        if drop > 0 and (best is None or drop > best.fall):
            best = Fall(iteration, iteration + 1, drop)
    return best


@dataclass
class Report:
    boot_cut: datetime | None
    boot_source: str
    sessions_total: int = 0
    sessions_kept: int = 0
    sessions_excluded: int = 0
    excluded_session_ids: list[str] = field(default_factory=list)
    rows_excluded: int = 0
    iteration_rows: int = 0
    deduped_rows: int = 0
    #: Turn-level rows: the ones with no `iteration` key. Never a bucket.
    turn_rows_no_key: int = 0
    turn_rows_no_key_prompt: int = 0
    turn_rows_no_key_cached: int = 0
    #: The sentinel #604/#618 filtered on. Same class of row, reported apart.
    turn_rows_iteration_zero: int = 0
    #: Rows reading over 100% cached, split by class so a per-iteration hit is
    #: not hidden inside an aggregate count.
    turn_rows_offending: int = 0
    iteration_rows_offending: int = 0
    worst_offender_ratio: float = 0.0
    platforms: list[str] = field(default_factory=list)
    overall: list[Bucket] = field(default_factory=list)
    by_platform: dict[str, list[Bucket]] = field(default_factory=dict)

    @property
    def largest_fall(self) -> Fall | None:
        """The biggest cached-fraction drop in the overall table."""
        return largest_fall_between(self.overall)

    @property
    def iteration_8_plus(self) -> Bucket | None:
        return next((b for b in self.overall if b.iteration == MERGE_FROM), None)

    def fall_for(self, platform: str) -> Fall | None:
        return largest_fall_between(self.by_platform.get(platform, []))


# ── reading the transcripts ─────────────────────────────────────────────────
def _timestamps(doc: dict[str, Any]) -> list[datetime]:
    out: list[datetime] = []
    created = doc.get("created_at")
    if isinstance(created, str):
        parsed = _parse_ts(created)
        if parsed:
            out.append(parsed)
    for msg in doc.get("messages") or []:
        if isinstance(msg, dict):
            parsed = _parse_ts(str(msg.get("timestamp") or ""))
            if parsed:
                out.append(parsed)
    return out


def _parse_ts(text: str) -> datetime | None:
    """Every timestamp in one frame: naive LOCAL wall clock.

    `sessions/*.json` really does carry two frames on the same day (measured
    2026-09-11): the chat and background writers stamp naive local
    (`created_at: 2026-09-11T11:02:45`, file born 11:03:32 -0700) while the
    Inner Voice writer stamps honest UTC with a `Z`
    (`created_at: 2026-09-11T17:46:04Z`, file born 10:51:35 -0700). Comparing
    them raw raises `TypeError: can't compare offset-naive and offset-aware
    datetimes`, and "fixing" that by dropping `tzinfo` silently cuts the table
    at the wrong hour — seven hours off, which is precisely the class of error
    #618 was filed about. So the offset is honoured where present and naive
    stamps are read as the local clock they were written from.
    """
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _stats_rows(doc: dict[str, Any]):
    for msg in doc.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        stats = msg.get("stats")
        if isinstance(stats, dict) and (
                _int(stats, "input_tokens") or _int(stats, "cache_read")):
            yield stats


def collect(sessions_dir: Path | str, *,
            boot: datetime | None = None,
            boot_source: str = "") -> Report:
    """Fold every session's stats rows into the per-iteration table.

    `boot` is the cut: a session whose EARLIEST timestamp predates it is
    excluded whole, because its early iterations were served by the engine that
    has since restarted and its later ones by the one that replaced it — the
    blend is the defect, so a session that straddles the boot cannot vote in
    either direction. With `boot=None` nothing is cut, and `render` says so in
    as many words, because an un-cut table is the thing #618 was filed about.
    """
    report = Report(boot_cut=boot, boot_source=boot_source or "no boot cut applied")
    if boot is not None and boot.tzinfo is not None:
        # Same frame as the transcripts; see `_parse_ts`.
        boot = boot.astimezone().replace(tzinfo=None)
        report.boot_cut = boot
    # (platform, iteration) -> session_id -> the row with the largest prompt.
    folds: dict[tuple[str, int], dict[str, tuple[int, int]]] = {}
    started: dict[str, datetime | None] = {}

    for path in sorted(Path(sessions_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        rows = list(_stats_rows(doc))
        if not rows:
            continue
        report.sessions_total += 1
        session_id = str(doc.get("session_id") or path.stem)
        stamps = _timestamps(doc)
        start = min(stamps) if stamps else None
        started[session_id] = start

        if boot is not None and start is not None and start < boot:
            report.sessions_excluded += 1
            report.excluded_session_ids.append(session_id)
            report.rows_excluded += len(rows)
            continue
        report.sessions_kept += 1
        platform = str(doc.get("platform") or "unknown")
        if platform not in report.platforms:
            report.platforms.append(platform)

        seen: dict[int, tuple[int, int]] = {}
        for stats in rows:
            prompt = _int(stats, "input_tokens", "prompt_tokens")
            cached = _int(stats, "cache_read", "prompt_tokens_cached")
            iteration = stats.get("iteration")
            if not isinstance(iteration, int) or isinstance(iteration, bool):
                # The turn-level row. Counted, reported, never bucketed.
                report.turn_rows_no_key += 1
                report.turn_rows_no_key_prompt += prompt
                report.turn_rows_no_key_cached += cached
                if cached > prompt:
                    report.turn_rows_offending += 1
                    report.worst_offender_ratio = max(
                        report.worst_offender_ratio, _ratio(cached, prompt))
                continue
            if iteration == 0:
                report.turn_rows_iteration_zero += 1
                if cached > prompt:
                    report.turn_rows_offending += 1
                    report.worst_offender_ratio = max(
                        report.worst_offender_ratio, _ratio(cached, prompt))
                continue
            report.iteration_rows += 1
            if cached > prompt:
                report.iteration_rows_offending += 1
                report.worst_offender_ratio = max(
                    report.worst_offender_ratio, _ratio(cached, prompt))
            previous = seen.get(iteration)
            if previous is None or prompt > previous[0]:
                seen[iteration] = (prompt, cached)
            else:
                report.deduped_rows += 1

        for iteration, (prompt, cached) in seen.items():
            for key in ((platform, iteration), ("overall", iteration)):
                fold = folds.setdefault(key, {})
                existing = fold.get(session_id)
                if existing is None or prompt > existing[0]:
                    fold[session_id] = (prompt, cached)

    def bucketed(table_iterations: list[int], platform: str) -> list[Bucket]:
        out: list[Bucket] = []
        for iteration in table_iterations:
            if iteration == MERGE_FROM:
                # Fold every deeper iteration in. The ceiling is the deepest
                # iteration ACTUALLY seen — `table_iterations` has already
                # collapsed 9, 10, 11 into this one row, so taking its max here
                # would drop them again.
                members = list(range(MERGE_FROM, max(iterations) + 1))
            else:
                members = [iteration]
            samples = prompt = cached = 0
            for member in members:
                for sess_prompt, sess_cached in folds.get(
                        (platform, member), {}).values():
                    samples += 1
                    prompt += sess_prompt
                    cached += sess_cached
            if samples:
                out.append(Bucket(iteration, samples, prompt, cached))
        return out

    iterations = sorted({it for (_, it) in folds})
    merged = [i for i in iterations if i >= MERGE_FROM]
    table_iterations = sorted(set(i for i in iterations if i < MERGE_FROM)
                              | ({MERGE_FROM} if merged else set()))
    report.overall = bucketed(table_iterations, "overall")
    for platform in report.platforms:
        report.by_platform[platform] = bucketed(table_iterations, platform)
    report.overall.sort(key=lambda b: b.iteration)
    for buckets in report.by_platform.values():
        buckets.sort(key=lambda b: b.iteration)
    return report


def bucket_for(buckets: list[Bucket], iteration: int) -> Bucket | None:
    return next((b for b in buckets if b.iteration == iteration), None)


# ── rendering ───────────────────────────────────────────────────────────────
def render(report: Report) -> str:
    lines: list[str] = []
    lines.append("prefix-cache per-iteration table — backlog #520, via #859")
    lines.append("")
    lines.append("BOOT CUT")
    if report.boot_cut is None:
        lines.append("  NONE — sessions from before an engine restart are "
                     "blended in. Do not draw a verdict from this table "
                     "(#618).")
    else:
        lines.append(f"  boot: {report.boot_cut.isoformat()}   "
                     f"source: {report.boot_source}")
        lines.append(f"  sessions excluded (predate the boot): "
                     f"{report.sessions_excluded}   rows excluded: "
                     f"{report.rows_excluded}")
        if report.excluded_session_ids:
            shown = ", ".join(report.excluded_session_ids[:12])
            more = (f" (+{len(report.excluded_session_ids) - 12} more)"
                    if len(report.excluded_session_ids) > 12 else "")
            lines.append(f"  excluded session ids: {shown}{more}")
    lines.append(f"  sessions in scope: {report.sessions_kept} of "
                 f"{report.sessions_total}   platforms: "
                 f"{', '.join(report.platforms) or 'none'}")
    lines.append(f"  per-iteration rows folded: {report.iteration_rows} "
                 f"(duplicate stats blocks deduped to the largest prompt per "
                 f"session x iteration: {report.deduped_rows})")
    lines.append("")

    offenders = report.turn_rows_offending + report.iteration_rows_offending
    lines.append("TURN-LEVEL ROWS (not an iteration bucket)")
    lines.append(f"  no iteration key: {report.turn_rows_no_key} rows   "
                 f"prompt {report.turn_rows_no_key_prompt}   cached "
                 f"{report.turn_rows_no_key_cached}")
    lines.append(f"  iteration == 0: {report.turn_rows_iteration_zero} rows")
    if offenders:
        lines.append(f"  rows over 100% cached: {offenders} "
                     f"(turn-level {report.turn_rows_offending}, "
                     f"per-iteration {report.iteration_rows_offending}) — "
                     f"worst ratio {report.worst_offender_ratio:.1f}x")
    else:
        lines.append("  rows over 100% cached: 0")
    lines.append("  a turn-level row carries no `iteration` key; the "
                 "`iteration == 0` filter in #604/#618 matched none of these, "
                 "which is how the store read as fixed")
    lines.append("")

    def table(title: str, buckets: list[Bucket]) -> None:
        lines.append(title)
        lines.append("  iteration   prompt tokens   cached    samples")
        for bucket in buckets:
            lines.append(f"  {bucket.label:>9}   {bucket.prompt_tokens:>13}   "
                         f"{bucket.fraction * 100:5.1f}%   {bucket.samples:>7}")
        if not buckets:
            lines.append("  (no samples)")
            return
        fall = largest_fall_between(buckets)
        if fall:
            lines.append(f"  largest fall: {fall.fall * 100:.2f} points at "
                         f"{fall.label}")
        else:
            lines.append("  largest fall: none — no adjacent iteration pair "
                         "drops cached fraction")
        eight = bucket_for(buckets, MERGE_FROM)
        if eight:
            lines.append(f"  iteration 8+ bucket: {eight.fraction * 100:.1f}% "
                         f"cached, samples {eight.samples}")
        lines.append("")

    table("OVERALL", report.overall)
    if report.by_platform:
        lines.append("PER PLATFORM")
        lines.append("")
    for platform in report.platforms:
        table(f"  [{platform}]", report.by_platform.get(platform, []))
    return "\n".join(lines).rstrip() + "\n"


def as_dict(report: Report) -> dict[str, Any]:
    overall_fall = report.largest_fall
    return {
        "boot_cut": report.boot_cut.isoformat() if report.boot_cut else None,
        "boot_source": report.boot_source,
        "sessions_total": report.sessions_total,
        "sessions_kept": report.sessions_kept,
        "sessions_excluded": report.sessions_excluded,
        "excluded_session_ids": report.excluded_session_ids,
        "rows_excluded": report.rows_excluded,
        "iteration_rows": report.iteration_rows,
        "deduped_rows": report.deduped_rows,
        "turn_rows": {
            "no_iteration_key": report.turn_rows_no_key,
            "no_iteration_key_prompt": report.turn_rows_no_key_prompt,
            "no_iteration_key_cached": report.turn_rows_no_key_cached,
            "iteration_zero": report.turn_rows_iteration_zero,
            "over_100_percent": (report.turn_rows_offending
                                 + report.iteration_rows_offending),
            "worst_ratio": report.worst_offender_ratio,
        },
        "overall": [
            {"iteration": b.iteration, "samples": b.samples,
             "prompt_tokens": b.prompt_tokens, "cached_tokens": b.cached_tokens,
             "cached_fraction": round(b.fraction, 4)}
            for b in report.overall],
        "largest_fall": (
            {"from": overall_fall.from_iteration,
             "to": overall_fall.to_iteration,
             "fall": round(overall_fall.fall, 4)} if overall_fall else None),
        "by_platform": {
            platform: {
                "buckets": [
                    {"iteration": b.iteration, "samples": b.samples,
                     "prompt_tokens": b.prompt_tokens,
                     "cached_tokens": b.cached_tokens,
                     "cached_fraction": round(b.fraction, 4)}
                    for b in buckets],
                "largest_fall": (
                    {"from": f.from_iteration, "to": f.to_iteration,
                     "fall": round(f.fall, 4)}
                    if (f := report.fall_for(platform)) else None),
            }
            for platform, buckets in report.by_platform.items()},
    }


# ── boot resolution ─────────────────────────────────────────────────────────
def parse_supervisor_status(text: str) -> dict[str, int]:
    """`program -> pid` for every RUNNING unit in `supervisorctl status` output."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] != "RUNNING":
            continue
        # `supervisorctl status` renders `... pid 2298786, uptime 3:02:11`, so
        # the number is the token AFTER `pid`, not a token starting with it.
        if "pid" not in parts:
            continue
        at = parts.index("pid") + 1
        if at >= len(parts):
            continue
        digits = parts[at].rstrip(",")
        if digits.isdigit():
            out[parts[0]] = int(digits)
    return out


def process_start_time(pid: int, *, proc_root: Path = Path("/proc")) -> datetime | None:
    """When a process started, from `/proc`.

    Field 22 of `/proc/<pid>/stat` is the start time in clock ticks since
    boot; `/proc/stat`'s `btime` is the boot epoch. Deliberately not
    `ps -o lstart`, whose locale makes it a parse of a formatted string, and
    not `uptime` arithmetic, which drifts by the poll-to-read gap.
    """
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
        btime_line = next(
            line for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime "))
        btime = int(btime_line.split()[1])
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    # `comm` is parenthesised and may contain spaces — split after the last ')'.
    tail = stat.rsplit(")", 1)[-1].split()
    if len(tail) < 20:
        return None
    try:
        ticks = int(tail[19])          # field 22 overall == 20th after `comm`
    except ValueError:
        return None
    return datetime.fromtimestamp(btime + ticks / USER_HZ)


def resolve_boot_from_proc(status_text: str, *,
                           proc_root: Path = Path("/proc")
                           ) -> tuple[datetime | None, str]:
    """The latest start among the units whose restart empties the prefix cache."""
    pids = parse_supervisor_status(status_text)
    best: tuple[datetime, str] | None = None
    for unit in BOOT_UNITS:
        name = next((p for p in pids if p == unit or p.endswith(f":{unit}")), None)
        if name is None:
            continue
        started = process_start_time(pids[name], proc_root=proc_root)
        if started and (best is None or started > best[0]):
            best = (started, name)
    if best is None:
        return None, ""
    return best[0], f"{best[1]} process start (/proc)"


def resolve_boot(*, from_landing: bool = False) -> tuple[datetime | None, str]:
    """Where the engine went cold. Two sources, both recorded verbatim."""
    if from_landing:
        stamp = latest_landing()
        return (stamp, f"{PROMOTIONS} landing timestamp") if stamp else (None, "")
    try:
        status = subprocess.run(
            ["supervisorctl", "-c", str(SUPERVISOR_CONF), "status"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None, ""
    return resolve_boot_from_proc(status)


def latest_landing() -> datetime | None:
    """The most recent landing that restarted the stack, from the automod ledger.

    The ledger carries every loop event (gate rungs, triage verdicts, ...), so
    only `promoted` rows count, and not one recorded `restart: false`: that
    landing replaced no process and left the cache warm.
    """
    if not PROMOTIONS.exists():
        return None
    latest: datetime | None = None
    for line in PROMOTIONS.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "promoted" \
                or row.get("restart") is False:
            continue
        stamp = _parse_ts(str(row.get("created_at") or ""))
        if stamp and (latest is None or stamp > latest):
            latest = stamp
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sessions-dir", default=str(SESSIONS_DIR))
    parser.add_argument("--boot", default=None,
                        help="ISO timestamp to cut at; overrides discovery")
    parser.add_argument("--boot-from-landing", action="store_true",
                        help="cut at the newest automod landing that restarted the stack")
    parser.add_argument("--no-boot-cut", action="store_true",
                        help="measure every session regardless of restarts "
                             "(the blend #618 was filed about)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    source = ""
    boot: datetime | None = None
    if args.boot:
        boot = _parse_ts(args.boot)
        source = f"--boot {args.boot}"
        if boot is None:
            print(f"could not parse --boot {args.boot!r} as an ISO timestamp",
                  file=sys.stderr)
            return 2
    elif args.no_boot_cut:
        boot, source = None, "--no-boot-cut: none applied"
    else:
        boot, source = resolve_boot(from_landing=args.boot_from_landing)
        if boot is None:
            print(
                "cannot resolve a boot cut: supervisorctl or /proc gave no "
                "start time for " + ", ".join(BOOT_UNITS) + ".\n"
                "Pass --boot <ISO> (the landing time), or "
                "--boot-from-landing, or --no-boot-cut to measure anyway. "
                "A blended table is what produced #603's false premise, so "
                "this script will not print one silently.",
                file=sys.stderr)
            return 2

    report = collect(args.sessions_dir, boot=boot, boot_source=source)
    if args.json:
        print(json.dumps(as_dict(report), indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
