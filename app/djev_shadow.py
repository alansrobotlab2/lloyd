"""Record what djev would have decided, beside what production did decide.

Two seams record on an ordinary day: write-time backlog dedupe in
`agent_mcp/backlog.py`, and the entity SAME/DIFFERENT gate in
`scripts/memory/entity_semantic_gate.py`. The third — the recall's `rerank`
seam in `agent_mcp/vault.py` — is NOT one of them while djev is the recall's
ranker. Since #1336 djev IS that ranker (`RECALL_RERANKER = "djev"`), the
dispatch's first arm takes every call, and the hook below it sits in an `elif`
that production never reaches: **structurally dark**, not quiet. Its one live
window is the kill switch pulled to `"qmd"` with the engine still answering — a
ranker rollback, which is precisely when a djev's-order-beside-fusion's-order
row is worth having. Measured on the live log, `rerank`'s last row is
2026-09-21T08:57:40Z while `dedupe` and `entity` still run daily; `seam_log_stats`
below is what says so on `djev_status` instead of leaving it to this paragraph.

The point of one recorder rather than three hooks is that all three want the
same thing (a labelled row per real decision) and all three sit on paths where
being slow is worse than being uninstrumented.

**Nothing here can change a production decision.** `shadow()` returns `None`,
always, having done one `put_nowait`. It has no return value a caller could
branch on by accident.

WHY A BOUNDED QUEUE THAT DROPS
------------------------------
djev is `--max-num-seqs 1`: strictly serialized, with unbounded queueing in
front of it. That is the old secondary's trap, where post-session jobs queued
behind agent turns and the slot's own docstring had to warn about it. A shadow
call must never sit in front of a production decision, so the queue is bounded
and **drops on overflow**, recording the drop. Losing an observation costs a
row; blocking the recall path costs the recall.

WHY THE WORKER DOES THE READING
-------------------------------
A seam enqueues the ids and the text it already holds. Anything that needs
work — the dedupe seam's candidate heads, which are a disk read per candidate
— arrives as a zero-argument callable the WORKER runs. The hot path pays one
`put_nowait` and nothing else.

TWO PROCESS SHAPES, AND BOTH HAVE TO WORK
-----------------------------------------
In the aggregator the worker thread lives as long as the process, and
`main.lifespan` calls `flush()` with a short bound on the way down. In a
script — the entity-resolution sweep — the process exits minutes after its
last call and a daemon thread goes with it, so the sweep calls `flush()` in a
`finally`. What a landing restart still loses is counted into
`dropped_at_shutdown` on the next process's first row: the stack restarts
several times a night, and a silent loss reads exactly like a quiet seam.

MUTED UNDER THE EVAL
--------------------
`eval/run_eval.py` and the regression runner set `LLOYD_DJEV_SHADOW=0`. A
pinned-corpus run recorded as production traffic would poison the very
`label_mass` distribution this log exists to collect the floors from — and
`run_eval.py` calls `_vault_recall` directly, which is where the `rerank` seam
sits. It cannot reach the recorder from there today; a mute whose correctness
depended on that dispatch staying as it is would be a mute with a hidden
precondition, so the mute stays and the seam's condition is stated above rather
than inferred from a row count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import djev

logger = logging.getLogger("lloyd-djev-shadow")

#: Outside the repo, like the automod state dir: a landing rewrites the tree
#: while this is being appended to, and `graphify-out/`'s lesson is that
#: derived state inside the tree dirties it and stops the loop.
STATE_DIR = Path.home() / ".local" / "state" / "lloyd-djev"
SHADOW_LOG = STATE_DIR / "shadow.jsonl"
#: Survives the process that could not drain its queue. One integer.
PENDING_DROPS = STATE_DIR / "dropped_at_shutdown.json"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    #: Deliberately small. The seams fire 140-300 times a day between them and
    #: a decision takes ~40 ms plus prefill, so a healthy backlog is single
    #: digits. A queue deep enough to hold an hour of traffic would hide the
    #: overflow this is meant to report.
    "queue_max": 256,
    #: Above `djev.DEFAULT_TIMEOUT_S`, because a shadow call has no caller
    #: waiting on it and its states are the large ones (a 12-candidate rerank
    #: pool, an item body).
    "timeout_seconds": 20.0,
    #: Rows are small; this is a guard against a runaway seam, not retention.
    "max_log_mb": 256,
}

#: Per-seam switches live under `djev.shadow.seams`. A seam absent from the
#: config is ON: adding a seam and forgetting its flag must not be the same as
#: switching it off, which is the failure mode of every allow-list default.
SEAMS = ("rerank", "dedupe", "entity")

_lock = threading.Lock()
_queue: "queue.Queue[_Job] | None" = None
_worker: threading.Thread | None = None
_counters: dict[str, int] = {"enqueued": 0, "dropped": 0, "written": 0,
                             "errors": 0, "skipped": 0}


@dataclass
class _Job:
    seam: str
    #: A string, or a zero-argument callable the WORKER runs. See the module
    #: docstring: the hot path must not read disk.
    state: Any
    questions: Any
    actual: Any = None
    meta: dict = field(default_factory=dict)
    floor: float | None = None
    schema: str = ""
    enqueued_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def config() -> dict:
    """`djev.shadow` over the defaults. Lazy and fail-open, like
    `backlog_similar.dedupe_config`."""
    cfg = dict(DEFAULTS)
    cfg["seams"] = {}
    try:
        from app.config import CONFIG
        block = ((CONFIG or {}).get("djev") or {}).get("shadow") or {}
        if isinstance(block, dict):
            cfg.update({k: v for k, v in block.items() if k in DEFAULTS})
            seams = block.get("seams")
            if isinstance(seams, dict):
                cfg["seams"] = {str(k): bool(v) for k, v in seams.items()}
    except Exception:  # noqa: BLE001 — fail open to the defaults
        pass
    return cfg


def _muted() -> bool:
    """`LLOYD_DJEV_SHADOW=0` — the eval's mute, read per call.

    Per call rather than at import: `run_eval.py` sets it in its own process
    before importing the handler, but the regression runner sets it for a
    child, and a module-level read would freeze whichever came first.
    """
    return os.environ.get("LLOYD_DJEV_SHADOW", "1").strip() in ("0", "false", "no")


def seam_enabled(seam: str, cfg: dict | None = None) -> bool:
    cfg = cfg or config()
    if not cfg.get("enabled", True):
        return False
    return bool((cfg.get("seams") or {}).get(seam, True))


def enabled(seam: str = "") -> bool:
    """Would a call for this seam be recorded at all?"""
    if _muted() or not djev.enabled():
        return False
    return seam_enabled(seam) if seam else bool(config().get("enabled", True))


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

def _ensure_worker(cfg: dict) -> "queue.Queue[_Job]":
    global _queue, _worker
    with _lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=int(cfg.get("queue_max", 256)))
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="djev-shadow",
                                       daemon=True)
            _worker.start()
        return _queue


def shadow(*, seam: str, state: Any, questions: Any, actual: Any = None,
           meta: dict | None = None, floor: float | None = None,
           schema: str = "") -> None:
    """Record djev's answer to this decision, later, off the caller's thread.

    Returns `None` unconditionally. `state` and `questions` may each be a
    zero-argument callable, which the worker runs — that is where a disk read
    belongs.
    """
    try:
        cfg = config()
        if _muted() or not cfg.get("enabled", True) or not seam_enabled(seam, cfg):
            _bump("skipped")
            return None
        if not djev.enabled():
            _bump("skipped")
            return None
        # A seam names itself and nothing else. The floor and the schema hash
        # come from `eval/djev/schemas.py`, which is where they were measured
        # and frozen — three seams each remembering to pass their own floor is
        # three places for it to go stale after one recalibration.
        if floor is None or not schema:
            fl, sh = _schema_for(seam)
            floor = fl if floor is None else floor
            schema = schema or sh
        q = _ensure_worker(cfg)
        job = _Job(seam=seam, state=state, questions=questions, actual=actual,
                   meta=dict(meta or {}), floor=floor, schema=schema)
        try:
            q.put_nowait(job)
            _bump("enqueued")
        except queue.Full:
            # The whole point of the bound. Counted, never waited on.
            _bump("dropped")
    except Exception as exc:  # noqa: BLE001 — a recorder may never raise
        logger.debug("djev shadow enqueue failed: %s", exc)
    return None


_schema_cache: dict[str, tuple[float | None, str]] = {}


def _schema_for(seam: str) -> tuple[float | None, str]:
    """`(label_mass_floor, schema_hash)` for a seam, or `(None, "")`.

    Cached, and fail-open: `eval/` is not a package the aggregator otherwise
    imports, and a recorder that raised because a sibling tree moved would
    take the seam with it.
    """
    if seam in _schema_cache:
        return _schema_cache[seam]
    try:
        from eval.djev import schemas
        out = (schemas.floor_for(seam), schemas.hash_for(seam))
    except Exception:  # noqa: BLE001
        out = (None, "")
    _schema_cache[seam] = out
    return out


def _bump(key: str, n: int = 1) -> None:
    with _lock:
        _counters[key] = _counters.get(key, 0) + n


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _resolve(value: Any) -> Any:
    return value() if callable(value) else value


def _digest(state: Any, questions: Any) -> str:
    try:
        blob = json.dumps([state, questions], sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        blob = repr((state, questions))
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]


def _take_pending_drops() -> int:
    """The count a previous process could not drain, read once and cleared."""
    try:
        if not PENDING_DROPS.exists():
            return 0
        n = int(json.loads(PENDING_DROPS.read_text()).get("dropped", 0))
        PENDING_DROPS.unlink(missing_ok=True)
        return n
    except Exception:  # noqa: BLE001
        return 0


def _record_pending_drops(n: int) -> None:
    if n <= 0:
        return
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        prior = 0
        if PENDING_DROPS.exists():
            prior = int(json.loads(PENDING_DROPS.read_text()).get("dropped", 0))
        PENDING_DROPS.write_text(json.dumps({"dropped": prior + n,
                                             "at": time.time()}))
    except Exception as exc:  # noqa: BLE001
        logger.debug("djev shadow: could not record %d shutdown drops: %s", n, exc)


_first_row_pending = threading.Event()
_first_row_pending.set()


def _write(row: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if _first_row_pending.is_set():
            _first_row_pending.clear()
            lost = _take_pending_drops()
            if lost:
                row["dropped_at_shutdown"] = lost
        with SHADOW_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        _bump("written")
    except Exception as exc:  # noqa: BLE001
        _bump("errors")
        logger.debug("djev shadow: write failed: %s", exc)


def _process(job: _Job, cfg: dict) -> None:
    started = time.time()
    try:
        state = _resolve(job.state)
        questions = _resolve(job.questions)
    except Exception as exc:  # noqa: BLE001 — a seam's own lookup failing is
        # news about that seam, not a reason to lose the row.
        _write({"ts": time.time(), "seam": job.seam, "schema": job.schema,
                "meta": job.meta, "actual": job.actual,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "queue_ms": round((started - job.enqueued_at) * 1e3, 1)})
        return
    if not questions:
        _bump("skipped")
        return
    answers = djev.ask_sync(state, questions, seam=job.seam, floor=job.floor,
                            timeout=float(cfg.get("timeout_seconds", 20.0)))
    _write({
        "ts": time.time(),
        "seam": job.seam,
        "schema": job.schema,
        "inputs_digest": _digest(state, questions),
        "n_questions": len(questions),
        "meta": job.meta,
        # What production actually decided. The whole value of the row.
        "actual": job.actual,
        "djev": answers.as_dict() if answers is not None else None,
        "queue_ms": round((started - job.enqueued_at) * 1e3, 1),
        "worker_ms": round((time.time() - started) * 1e3, 1),
    })


def _run() -> None:
    """The worker owns the queue it was STARTED with, and exits when the
    module moves off it.

    The first cut read the module global on every iteration and looped on
    `None` with no sleep — `job = _queue.get(...) if _queue else None`
    followed by `if job is None: continue`. `reset_for_tests()` sets
    `_queue = None` while the previous test's daemon thread is still alive,
    and that thread then spun a core at 100% for the life of the process.
    Caught as one pytest worker at 99% CPU starving the other seven.

    Binding `q` once is also the right shape rather than only the safe one: a
    worker draining a queue nobody enqueues into any more has nothing to do.
    """
    cfg = config()
    q = _queue
    if q is None:
        return
    while True:
        try:
            job = q.get(timeout=5.0)
        except queue.Empty:
            if _queue is not q:
                return
            continue
        except Exception:  # noqa: BLE001
            return
        if job is None:
            continue
        try:
            _process(job, cfg)
        except Exception as exc:  # noqa: BLE001 — one bad row never ends the
            # worker; a dead worker turns every later seam silent.
            _bump("errors")
            logger.debug("djev shadow: job failed: %s", exc)
        finally:
            try:
                q.task_done()
            except Exception:  # noqa: BLE001
                pass


def flush(timeout: float = 10.0) -> int:
    """Drain what is queued, up to `timeout`. Returns what was left behind.

    Called from `main.lifespan`'s shutdown with a short bound, and from the
    entity sweep's `finally` — a daemon thread takes the queue with it when a
    script exits, so without this the sweep's ~302 rows a day would be the one
    seam that never recorded anything. Whatever is still queued when the bound
    expires is written to `PENDING_DROPS` and reported on the next process's
    first row.
    """
    q = _queue
    if q is None:
        return 0
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if q.unfinished_tasks == 0:
            return 0
        time.sleep(0.05)
    left = q.unfinished_tasks
    if left > 0:
        _record_pending_drops(left)
        logger.debug("djev shadow: %d rows left at flush", left)
    return left


def stats() -> dict:
    """The recorder's block of `djev_status`. Offline; touches no socket.

    Named for its real reader rather than a list of possible ones: #1372 is a
    change about descriptions that outlived their facts, and this sentence used
    to name a second reader — a state route — that never called it. A field
    added here for a surface that never reads it is stale on arrival, so the
    reader list stays exact. Grep this function's name for the call sites;
    `tests/test_djev_doc_claims.py` pins the sentence.
    """
    q = _queue
    with _lock:
        counts = dict(_counters)
    return {
        **counts,
        "queue_depth": q.qsize() if q is not None else 0,
        "queue_max": int(config().get("queue_max", 256)),
        "worker_alive": bool(_worker and _worker.is_alive()),
        "log": str(SHADOW_LOG),
        "log_rows": _log_rows(),
        # The whole-file count above cannot answer "is THIS seam recording",
        # which is the question #1372 was filed for: 224 rows looked healthy on
        # 2026-09-22 while `rerank` had recorded nothing for a day and a half.
        "seam_log": seam_log_stats(),
        "pending_shutdown_drops": _peek_pending_drops(),
    }


#: A seam whose newest row is older than this reads as `dark` on the status
#: route (#1372 clause 5). Deliberately a day and not a week: `entity` fires
#: about once a day and `dedupe` several, so the seam that legitimately goes
#: silent the longest still fits inside it, and 2026-09-22 — `rerank` at zero
#: rows against `dedupe` 33 and `entity` 25 — sits a whole day outside it.
DARK_AFTER_S = 86400


def seam_log_stats(now: float | None = None) -> dict:
    """Per-seam rows and newest row, read from `SHADOW_LOG`.

    `stats()` has always reported one whole-file `log_rows`, which cannot answer
    the question that actually matters: is THIS seam recording? On 2026-09-22 the
    file held 224 rows and looked healthy while `rerank` had recorded nothing
    since 2026-09-21T08:57:40Z — and the two seams whose counts made the total
    look alive were the two that were fine.

    Offline like `_log_rows`, and same about a read failure: `rows` is `None`,
    not `0`, and `dark` is `None` rather than a verdict, because an unreadable
    log and an empty one are different answers and only one of them is a
    finding. Malformed rows are skipped the way `_write`'s reader skips them.

    `now` is injectable so the age boundary is testable without waiting a day
    for a fixture to go stale.
    """
    now = time.time() if now is None else now
    rows: dict[str, int | None] = {s: 0 for s in SEAMS}
    newest: dict[str, float | None] = {s: None for s in SEAMS}
    unreadable = False
    try:
        if SHADOW_LOG.exists():
            with SHADOW_LOG.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    seam = row.get("seam")
                    if seam not in rows:
                        continue          # a seam this build doesn't know: its
                                          # rows are not this build's verdict
                    ts = row.get("ts")
                    if not isinstance(ts, (int, float)):
                        continue          # no timestamp, no age claim
                    rows[seam] += 1
                    if newest[seam] is None or ts > newest[seam]:
                        newest[seam] = float(ts)
    except Exception:  # noqa: BLE001
        unreadable = True

    out: dict[str, dict] = {}
    for seam in SEAMS:
        last = newest[seam]
        out[seam] = {
            "rows": None if unreadable else rows[seam],
            "last_row_utc": (None if last is None else
                             datetime.fromtimestamp(last, timezone.utc)
                             .isoformat(timespec="seconds")),
            "age_s": None if last is None else max(0, int(now - last)),
            "dark": None if unreadable else (
                rows[seam] == 0 or (now - last) > DARK_AFTER_S),
        }
    return out


def _peek_pending_drops() -> int:
    try:
        if PENDING_DROPS.exists():
            return int(json.loads(PENDING_DROPS.read_text()).get("dropped", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _log_rows() -> int | None:
    """Row count, or `None` when the log cannot be read. Not zero: an
    unreadable log and an empty one are different answers."""
    try:
        if not SHADOW_LOG.exists():
            return 0
        with SHADOW_LOG.open("rb") as fh:
            return sum(1 for _ in fh)
    except Exception:  # noqa: BLE001
        return None


def reset_for_tests() -> None:
    global _queue, _worker
    _schema_cache.clear()
    with _lock:
        _queue = None
        _worker = None
        for k in _counters:
            _counters[k] = 0
    _first_row_pending.set()
