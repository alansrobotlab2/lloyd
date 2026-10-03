"""The schedule for the frontend-probe canary (#1981).

`scripts/automod/frontend_probe_canary.py` seeds known-broken frontend builds and
reports what fraction the runtime probe detects. It landed (#1872) with a CLI, an
artifact writer and no caller: every test redirects the automod state dir to a temp
dir, so the live state dir's `frontend_probe_canary/latest.json` could only
ever be written by a runner, and there was none. The owed-check job re-measured the
number by hand on every pass, and the detection rate could decay unobserved. This
source is the runner and nothing else — the measurement, its bar and its artifact
stay in the script.

Three rules, each one a way this could have been quietly wrong:

- **The default state dir.** The command carries no `LLOYD_AUTOMOD_STATE` and the
  child inherits the pool's environment, which sets none, so the artifact lands in
  the live state dir. The result names the path it read
  (`STATE_DIR / frontend_probe_canary / latest.json`, resolved at call time).
- **`--seeds 12`, not the script's default floor of 10.** `--seeds` is a floor the
  run refuses to report below, and the shipped table holds twelve must-detect seeds.
  At 10 a scheduled run would tolerate a seed silently dropping out of `n` on top of
  the one miss a 90% bar already allows.
- **A run that did not measure is never a success, and never carries a rate.** The
  CLI's exit code is the verdict: 0 passed, 1 measured below the bar (the result
  carries the measured numbers, as a failure), 2 not measured (no numbers at all —
  `latest.json` may still hold an OLDER run, and quoting it here would report a
  rate this run never took). An exit-0 run whose artifact is missing or predates
  the run is likewise a failure: the number has to come from this run's file.

Failures are RETURNED, not raised: a retry of a deterministic measurement measures
the same thing, and a failed run is a row someone reads on `/api/workers/health`.

No session: seeded builds and a headless probe are arithmetic, not a judgement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.frontend-probe-canary")

NAME = "frontend-probe-canary"
# Below every stream that does work for someone (research 70, clustering 65):
# an instrument check can always wait.
DEFAULT_PRIORITY = 80
LONG_LIVED = False
DEDUP_KEY = "frontend-probe-canary:scheduled"

#: The declared interval: at most one run per day. The pool polls sooner than that;
#: (`interval_seconds`, the scheduler's key); `enqueue_if_due` keeps its own
#: watermark against `min_interval_seconds` so a restart or a faster poll cannot
#: double up.
DEFAULT_INTERVAL_SECONDS = 24 * 3600  # `min_interval_seconds` in config
WM_LAST_ENQUEUED = "last_enqueued"

#: `len(frontend_probe_canary.must_detect_seeds())` today. If the table grows the
#: floor is merely conservative; if it shrinks the CLI exits 2 ("nothing measured")
#: and the run fails loudly, which is the right answer to a seed going missing.
#: 12 -> 16 on #2130: four layout seeds joined the table. The number is pinned
#: against the table by `test_the_declared_interval_is_a_day_and_the_seed_floor_is_the_whole_table`,
#: so a seed that stops being must-detect has to be deleted here as well as there.
SEEDS = 16
MODULE = "scripts.automod.frontend_probe_canary"
#: What the item's payload carries, and what a human types to reproduce the run.
COMMAND = f"python -m {MODULE} --seeds {SEEDS}"

DEFAULT_TIMEOUT_SECONDS = 600
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: The fields a later job quotes instead of re-measuring.
ARTIFACT_FIELDS = ("detected", "seeds_measured", "rate",
                   "rate_counting_blind_as_misses", "run_at")


def build_argv() -> list[str]:
    """`COMMAND` with the interpreter this process runs on — the lloyd venv."""
    return [sys.executable, *COMMAND.split()[1:]]


def artifact_path() -> Path:
    """Where the CLI writes, resolved now: `S.STATE_DIR` is read at call time so
    this names the same file `write_run_artifact` does."""
    from scripts.automod import frontend_probe_canary as C, state as S
    return S.STATE_DIR / C.ARTIFACT_DIRNAME / "latest.json"


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    interval = float(src_cfg.get("min_interval_seconds", DEFAULT_INTERVAL_SECONDS))
    last = await asyncio.to_thread(queue.wm_get, NAME, WM_LAST_ENQUEUED)
    if last:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
        except ValueError:
            age = None
        if age is not None and age < interval:
            return
    new_id = await asyncio.to_thread(
        queue.enqueue, source=NAME, kind="canary",
        payload={"command": COMMAND,
                 "timeout_seconds": int(src_cfg.get("timeout_seconds",
                                                    DEFAULT_TIMEOUT_SECONDS))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY)
    if new_id is not None:
        await asyncio.to_thread(queue.wm_set, NAME, WM_LAST_ENQUEUED,
                                datetime.now(timezone.utc).isoformat())
        logger.info("Enqueued frontend-probe canary id=%d", new_id)


async def _run_cli(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run the canary off the event loop; (exit code, combined output).

    No `env=`: the child inherits this process's environment unchanged, which is
    what puts the artifact in the default state dir.
    """
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


def _read_artifact(path: Path, started: float) -> dict[str, Any] | None:
    """This run's artifact, or None: missing, unreadable, or older than the run."""
    try:
        if path.stat().st_mtime < started - 1.0:
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _tail(text: str, n: int = 6) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def _failed(summary: str, **extra: Any) -> dict[str, Any]:
    return {"status": "failed", "summary": summary, "error": summary, **extra}


async def execute(item: QueueItem) -> dict[str, Any]:
    p = item.payload or {}
    timeout = float(p.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    path = artifact_path()
    started = time.time()
    try:
        code, output = await _run_cli(build_argv(), timeout)
    except asyncio.TimeoutError:
        return _failed(f"frontend-probe canary not measured: the run exceeded "
                       f"{timeout:.0f} s and was killed", command=COMMAND)
    except OSError as exc:
        return _failed(f"frontend-probe canary not measured: the CLI could not be "
                       f"started ({exc})", command=COMMAND)

    base = {"command": COMMAND, "exit_code": code, "response": _tail(output, 30)}
    if code not in (0, 1):
        # Exit 2 is the CLI's "nothing measured". Any other code is a crash, which
        # measured nothing either. No rate: `latest.json` may hold an older run.
        return _failed(f"frontend-probe canary not measured (exit {code}): "
                       f"{_tail(output, 2)}", **base)

    doc = await asyncio.to_thread(_read_artifact, path, started)
    if doc is None:
        return _failed(f"frontend-probe canary not measured: exit {code} but no "
                       f"artifact from this run at {path}", **base)
    measured = {k: doc.get(k) for k in ARTIFACT_FIELDS}
    line = (f"detected {measured['detected']}/{measured['seeds_measured']} "
            f"(rate {measured['rate']}, counting blind seeds as misses "
            f"{measured['rate_counting_blind_as_misses']})")
    # `meta` is what the run record keeps (`normalize_result`), so the numbers are
    # readable from the runs table as well as from the artifact.
    if code == 1 or not doc.get("passes"):
        return _failed(f"frontend-probe canary below the bar: {line}",
                       artifact_path=str(path), meta=dict(measured), **measured, **base)
    return {"status": "success", "summary": f"frontend-probe canary passed: {line}",
            "artifact_path": str(path), "meta": dict(measured), **measured, **base}
