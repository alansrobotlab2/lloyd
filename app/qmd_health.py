"""What the qmd daemon says about the answer it just gave.

The fork's REST `/query` returns `meta: {reranked, rerankFallback, ms, phases}`
beside `results` (qmd/WORKLOG.md section 7). `reranked` is `false` when the
caller asked for the cross-encoder and the daemon could not run it -- in
practice, no VRAM for a ranking context on the GPU it shares with the desktop
and the TTS server. Until 2026-09-19 that case was an HTTP 200 and an ordinary
result list: `vault_recall` quietly served fusion-order results, worth 0.16 MRR
on the pinned eval, and the only trace was one line on the daemon's stderr.

Retrieval still WORKS in that state, so nothing here raises or blocks a recall.
It counts, logs, and rings the guardian's `announce()` bell once per cooldown --
news, not an incident, by the same route `app/prefix_miss.py` takes.

Stdlib only, like the rest of `app/`'s leaf modules: the aggregator imports
this on the recall path.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable

#: Floor between announcements. A bad evening is one toast, not one per recall.
ANNOUNCE_COOLDOWN_SECONDS = 1800
#: Floor between log lines for the same condition.
LOG_COOLDOWN_SECONDS = 300

_lock = threading.Lock()
_stats = {
    "responses": 0,            # responses that carried `meta`
    "without_meta": 0,         # a daemon older than the fork's section-7 build
    "reranked": 0,
    "rerank_fallbacks": 0,
    "last_fallback_at": None,  # epoch seconds
    "last_fallback_reason": None,
    "last_ms": None,
    "last_phases": None,
}
_last_log_at = 0.0
_last_announce_at = 0.0


def stats() -> dict:
    """A copy, for the aggregator's `/state`."""
    with _lock:
        return dict(_stats)


def _default_announce(title: str, body: str) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return  # a test that walks this path must not toast the desktop
    try:
        from app.prefix_miss import _announce
        _announce(title, body, False)
    except Exception as exc:  # noqa: BLE001 -- the bell is a courtesy, never a dependency
        print(f"[qmd] could not announce: {exc!r}", file=sys.stderr, flush=True)


def note_response(rerank_requested: bool, meta: dict | None, *,
                  announce: Callable[[str, str], None] | None = None,
                  now: float | None = None) -> bool:
    """Fold one daemon response in. True when it was a rerank fallback.

    `announce` and `now` are injectable so the cooldowns can be pinned without a
    clock or a desktop."""
    global _last_log_at, _last_announce_at
    now = time.time() if now is None else now
    if not isinstance(meta, dict):
        with _lock:
            _stats["without_meta"] += 1
        return False

    fell_back = bool(rerank_requested) and meta.get("reranked") is False
    reason = str(meta.get("rerankFallback") or "reranker unavailable")
    ring = log = False
    with _lock:
        _stats["responses"] += 1
        _stats["last_ms"] = meta.get("ms")
        _stats["last_phases"] = meta.get("phases")
        if meta.get("reranked") is True:
            _stats["reranked"] += 1
        if fell_back:
            _stats["rerank_fallbacks"] += 1
            _stats["last_fallback_at"] = now
            _stats["last_fallback_reason"] = reason
            if now - _last_log_at >= LOG_COOLDOWN_SECONDS:
                _last_log_at, log = now, True
            if now - _last_announce_at >= ANNOUNCE_COOLDOWN_SECONDS:
                _last_announce_at, ring = now, True
        total = _stats["rerank_fallbacks"]
    if log:
        print(f"[qmd] DEGRADED: the daemon answered WITHOUT reranking ({reason}); "
              f"{total} such answers since this process started", file=sys.stderr, flush=True)
    if ring:
        title = "Vault recall is running unreranked"
        body = (f"qmd could not run its cross-encoder: {reason}. Results are in fusion "
                f"order (about a third of MRR gone) until GPU 0 has room again. "
                f"`curl localhost:8181/health` shows the daemon's own count.")
        fn = announce or _default_announce
        # Off the recall path: the fan-out may toast and write a journal line.
        threading.Thread(target=fn, args=(title, body), daemon=True).start()
    return fell_back


def _reset_for_tests() -> None:
    global _last_log_at, _last_announce_at
    with _lock:
        for key, value in list(_stats.items()):
            _stats[key] = 0 if isinstance(value, int) else None
    _last_log_at = _last_announce_at = 0.0
