"""The nightly schedule for the behaviour-fitness run.

`scripts/automod/fitness.py` runs the benches of `eval/fitness/manifest.yaml`
and appends what they measured to `app.paths.FITNESS_SERIES`. This source is the
caller and nothing else: the benches, their values, the noise floors and the
attribution stay in the script. Ships **disabled** (`workers.sources.fitness
.enabled: false`) — the noise floors are measured by hand first, and a delta
with no floor under it reports nothing.

Four rules:

- **Once a night, inside a quiet window.** `enqueue_if_due` enqueues only while
  the local hour is inside `window_start_hour`..`window_end_hour` and the last
  enqueue is older than `min_interval_seconds`. The benches share the primary
  with every worker turn; a run at noon would measure the afternoon's traffic.
- **`--pool-job`, not `--force`.** The run is one of the pool's own in-flight
  rows, so it cannot ask for a paused pool. In this mode the script waits before
  each engine bench until the pool holds nothing but this job and the KV gate is
  released, and records a bench it could not get a quiet engine for as `skipped`
  with no value. The source is `LONG_LIVED`, so the pool's KV gate also holds the
  claim itself while the primary is over budget. A hand run with the pool paused
  is still the cleaner measurement, and the noise floors are taken that way.
- **Off the event loop.** One `asyncio` subprocess; nothing here reads a bench
  artifact or the series.
- **Its timer beats the pool's.** The subprocess is killed at the pool's cap
  (`source_max_duration_seconds`) minus `POOL_MARGIN_SECONDS`, and the script is
  handed a budget shorter again, so it stops starting benches before either
  timer fires and the run is never orphaned mid-bench by the pool.

Failures are RETURNED, not raised: a retry of a nightly measurement an hour later
is a second night's engine time, and a failed run is a row someone reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.fitness")

NAME = "fitness"
#: Below every stream that does work for someone, and below the frontend-probe
#: canary (80): an instrument can always wait.
DEFAULT_PRIORITY = 85
#: Up to an hour of engine benches, so the pool's KV gate must hold the claim
#: while the primary is under pressure. `tests/test_kv_gate.py` pins the set.
LONG_LIVED = True
DEDUP_KEY = "fitness:nightly"
WM_LAST_ENQUEUED = "last_enqueued"

#: One run a night; 20 h rather than 24 so a run that started late in last
#: night's window does not push tonight's out of its own.
DEFAULT_MIN_INTERVAL_SECONDS = 20 * 3600
#: The quiet window, local hours, start inclusive and end exclusive.
DEFAULT_WINDOW_START_HOUR = 1
DEFAULT_WINDOW_END_HOUR = 4
#: The subprocess dies this long before the pool's own timer would fire.
POOL_MARGIN_SECONDS = 60
#: And the script stops STARTING benches this long before the subprocess dies.
BUDGET_MARGIN_SECONDS = 120
#: How long the script queues for `regression.lock` behind a regression check.
LOCK_WAIT_SECONDS = 600

MODULE = "scripts.automod.fitness"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_PREFIX = "FITNESS_RESULT "


def in_window(hour: int, start: int, end: int) -> bool:
    """Whether local `hour` is inside [start, end), wrapping past midnight."""
    start, end = int(start) % 24, int(end) % 24
    if start == end:
        return False
    return start <= hour < end if start < end else (hour >= start or hour < end)


def timers(cap_seconds: int) -> tuple[int, int]:
    """`(subprocess timeout, script budget)` under the pool's cap."""
    timeout = max(1, int(cap_seconds) - POOL_MARGIN_SECONDS)
    return timeout, max(1, timeout - BUDGET_MARGIN_SECONDS)


def build_argv(budget_seconds: int) -> list[str]:
    return [sys.executable, "-m", MODULE, "run", "--once", "--pool-job",
            "--budget-seconds", str(int(budget_seconds)),
            "--lock-wait", str(LOCK_WAIT_SECONDS)]


def _local_hour() -> int:
    return datetime.now().astimezone().hour


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict, *,
                         local_hour: Optional[int] = None,
                         now: Optional[datetime] = None) -> None:
    hour = _local_hour() if local_hour is None else local_hour
    if not in_window(hour,
                     src_cfg.get("window_start_hour", DEFAULT_WINDOW_START_HOUR),
                     src_cfg.get("window_end_hour", DEFAULT_WINDOW_END_HOUR)):
        return
    stamp = now or datetime.now(timezone.utc)
    interval = float(src_cfg.get("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS))
    last = await asyncio.to_thread(queue.wm_get, NAME, WM_LAST_ENQUEUED)
    if last:
        try:
            age = (stamp - datetime.fromisoformat(last)).total_seconds()
        except ValueError:
            age = None
        if age is not None and age < interval:
            return
    new_id = await asyncio.to_thread(
        queue.enqueue, source=NAME, kind="fitness",
        payload={"command": f"python -m {MODULE} run --once --pool-job", **intake_flags(src_cfg)},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY)
    if new_id is not None:
        await asyncio.to_thread(queue.wm_set, NAME, WM_LAST_ENQUEUED, stamp.isoformat())
        logger.info("Enqueued nightly fitness run id=%d", new_id)


async def _run_cli(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run the script off the event loop; (exit code, combined output). The child
    inherits this process's environment unchanged, so it resolves the live data
    root and state dir exactly as the backend does."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return int(proc.returncode or 0), out.decode("utf-8", "replace")


def parse_result(output: str) -> dict[str, Any] | None:
    """The script's one machine-readable line, or None."""
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                doc = json.loads(line[len(RESULT_PREFIX):])
            except ValueError:
                return None
            return doc if isinstance(doc, dict) else None
    return None


def _tail(text: str, n: int = 6) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def _failed(summary: str, **extra: Any) -> dict[str, Any]:
    return {"status": "failed", "summary": summary, "error": summary, **extra}


def _cap_seconds() -> int:
    from workers.pool import source_max_duration_seconds
    return int(source_max_duration_seconds(NAME)[0])


async def execute(item: QueueItem) -> dict[str, Any]:
    timeout, budget = timers(_cap_seconds())
    argv = build_argv(budget)
    command = " ".join(["python", *argv[1:]])
    try:
        code, output = await _run_cli(argv, timeout)
    except asyncio.TimeoutError:
        return _failed(f"fitness run not finished: exceeded {timeout} s and was "
                       f"killed (rows already appended stand)", command=command)
    except OSError as exc:
        return _failed(f"fitness run not started ({type(exc).__name__})", command=command)

    base = {"command": command, "exit_code": code, "response": _tail(output, 30)}
    doc = parse_result(output)
    if doc is None:
        # Exit 2 is a refusal (manifest), 3 is a held lock: neither measured.
        return _failed(f"fitness run measured nothing (exit {code}): {_tail(output, 2)}",
                       **base)
    measured, rows = int(doc.get("measured") or 0), int(doc.get("rows") or 0)
    meta = {"run_id": doc.get("run_id"), "measured": measured, "rows": rows,
            "not_measured": doc.get("not_measured") or []}
    line = f"fitness run {doc.get('run_id')}: {measured} of {rows} benches measured"
    if meta["not_measured"]:
        line += " (" + "; ".join(str(x) for x in meta["not_measured"]) + ")"
    if not measured:
        return _failed(line, meta=meta, **base)
    if (item.payload or {}).get("file_items"):
        intake = await asyncio.to_thread(_file_items, bool(item.payload.get("file_lowest")))
        meta["intake"] = intake
        line += f"; intake {intake.get('decision')}" + (
            f" #{intake['item_id']}" if intake.get("item_id") else "")
    return {"status": "success", "summary": line, "meta": meta, **base}


# ── fitness-driven intake (scripts/automod/fitness_intake.py) ────────────────
#
# At the foot so no line above moves. After a run that measured something, and
# only when `workers.sources.fitness.file_items` is true (ships false), at most
# one `improvement` item a day is filed against the bench that most regressed
# outside its noise floor. The switches travel in the queue row's payload
# because `execute` is not handed the source's config.

def intake_flags(src_cfg: dict) -> dict[str, bool]:
    """The payload keys the intake hook reads; `{}` when it is off, so a row
    enqueued with the switch off is byte-identical to one from before it existed."""
    if not bool((src_cfg or {}).get("file_items", False)):
        return {}
    return {"file_items": True, "file_lowest": bool(src_cfg.get("file_lowest", False))}


def _file_items(file_lowest: bool) -> dict[str, Any]:
    """Runs in a thread. Fail-open: a broken intake costs a warning and its
    journal row, never the measurement that was just recorded."""
    try:
        from scripts.automod import fitness_intake as FI
        out = FI.run(file_lowest=file_lowest)
        return {k: out.get(k) for k in ("decision", "bench", "item_id", "why") if out.get(k)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("fitness intake failed (%s: %s)", type(exc).__name__, exc)
        return {"decision": "error", "why": f"{type(exc).__name__}: {exc}"[:200]}

