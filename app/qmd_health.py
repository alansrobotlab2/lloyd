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
    # The recall's ranker (#1336): djev on GPU 2, falling back to qmd's
    # cross-encoder when djev does not answer.
    "djev_ranked": 0,
    "djev_fallbacks": 0,
    "last_djev_fallback_at": None,
    "last_djev_fallback_reason": None,
}
_last_log_at = 0.0
_last_announce_at = 0.0
_last_ranker_log_at = 0.0
_last_ranker_announce_at = 0.0


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


#: The reasons a djev recall fallback can be booked with (#2112). The recall used
#: to pass the first for EVERY cause, so an engine that ANSWERED and whose answer
#: was unusable — a ranking the server split across canvas chunks — was logged and
#: toasted as an engine that was down. A caller that means "down" still says this
#: exactly: the first is `note_ranker`'s own default, now named.
RANK_FALLBACK_NO_ANSWER = "djev did not answer"
RANK_FALLBACK_CANVAS_SPLIT = ("the engine split the pool across canvas chunks, "
                              "whose scores are not comparable")


def note_ranker(ok: bool, reason: str = "", *,
                announce: Callable[[str, str], None] | None = None,
                now: float | None = None) -> None:
    """Fold one recall ranking in: djev answered (`ok`), or the recall fell back.

    A fallback still answers — it takes qmd's cross-encoder path, ~2 s instead of
    ~0.5 s — so, like a rerank fallback, it is counted, logged and announced,
    never raised. Same cooldowns, its own clock.

    `reason` is WHICH cause the caller hit, and it is what the degraded line and
    the toast interpolate verbatim, so it is the only place a reader ever learns
    the difference between an engine that did not answer and one that answered
    with something unusable. The recall started passing the real one in #2112."""
    global _last_ranker_log_at, _last_ranker_announce_at
    now = time.time() if now is None else now
    ring = log = False
    with _lock:
        if ok:
            _stats["djev_ranked"] += 1
            return
        _stats["djev_fallbacks"] += 1
        _stats["last_djev_fallback_at"] = now
        _stats["last_djev_fallback_reason"] = reason or "djev did not answer"
        if now - _last_ranker_log_at >= LOG_COOLDOWN_SECONDS:
            _last_ranker_log_at, log = now, True
        if now - _last_ranker_announce_at >= ANNOUNCE_COOLDOWN_SECONDS:
            _last_ranker_announce_at, ring = now, True
        total = _stats["djev_fallbacks"]
    if log:
        print(f"[qmd] DEGRADED: djev did not rank a recall ({reason or 'no answer'}); "
              f"fell back to qmd's cross-encoder; {total} such recalls since this process started",
              file=sys.stderr, flush=True)
    if ring:
        title = "Vault recall fell back to the cross-encoder"
        body = (f"djev on GPU 2 did not rank a recall ({reason or 'no answer'}). Recalls are "
                f"taking qmd's cross-encoder path (~2 s instead of ~0.5 s) until it answers. "
                f"`djev_status` shows the engine.")
        fn = announce or _default_announce
        threading.Thread(target=fn, args=(title, body), daemon=True).start()


#: The three things a run's artifact can honestly say about who ordered its
#: document pool (#3754). The words live HERE, beside the counters they are read
#: from, for the reason #1486 states for the seed record: the recall that ran and
#: the artifact that records the run read ONE accessor, so a baseline cannot
#: report a ranker different from the one that served it.
RANKER_DJEV = "djev"
#: At least one recall in this process asked for djev and took qmd's
#: cross-encoder instead — the #3675 state, where `djev.enabled: true` in
#: config.yaml made every recall name an engine that was down.
RANKER_QMD_CROSS_ENCODER = "qmd-cross-encoder"
#: No ranker verdict was booked at all. Its own token, for the same reason
#: `app/uptake.py` has one for seeding: the absence of a measurement is not a
#: measurement of the alternative. A process with `djev.enabled: false` lands here
#: rather than at `qmd-cross-encoder` because these two counters only ever see the
#: djev arm — nothing booked means nothing was observed, not that qmd answered.
RANKER_UNRECORDED = "unrecorded"


def observed_ranker_record() -> dict:
    """Who actually ordered the document pool in this process, as an artifact record.

    `{"ranker": <one of the three tokens above>, "djev_ranked": int,
    "djev_fallbacks": int, "last_fallback_reason": str | None}`.

    Read from the counters `note_ranker` moves — never from a requested knob. The
    difference is the whole point (#3675): `app/djev.py`'s `enabled()` asks config
    whether djev is switched on, not whether it answers, so the nightly eval of
    2026-11-01 06:06 ran entirely inside that outage and every one of its recalls
    was answered by qmd's cross-encoder while the knob it could have recorded said
    the run was not djev-reranked. An artifact that records intent cannot be asked
    which engine produced its numbers; this one can.

    A mixed night is stamped as the fallback, not as the majority. The band this
    feeds is a tolerance measured across nights, and a night whose recalls were
    ordered two ways is comparable to NEITHER of the two pure nights — reading it
    as "mostly djev" would pool it with the djev nights, which is the defect
    #1547 was filed against. The two counts ride beside the token so a reader can
    see the mix rather than take it on faith.

    Scope, stated because it is load-bearing: the counters are process-lifetime, so
    this answers for every recall this process has served, and the caller writing
    an artifact is a process whose only recalls are the ones in its own run
    (`eval/run_eval.py` is a fresh interpreter per night). A long-lived caller
    would have to book a before/after pair; no caller does that today.
    """
    with _lock:
        ranked = _stats["djev_ranked"]
        fallbacks = _stats["djev_fallbacks"]
        reason = _stats["last_djev_fallback_reason"]
    if fallbacks > 0:
        ranker = RANKER_QMD_CROSS_ENCODER
    elif ranked > 0:
        ranker = RANKER_DJEV
    else:
        ranker = RANKER_UNRECORDED
    return {"ranker": ranker,
            "djev_ranked": ranked,
            "djev_fallbacks": fallbacks,
            "last_fallback_reason": reason if fallbacks else None}


def _reset_for_tests() -> None:
    global _last_log_at, _last_announce_at, _last_ranker_log_at, _last_ranker_announce_at
    with _lock:
        for key, value in list(_stats.items()):
            _stats[key] = 0 if isinstance(value, int) else None
    _last_log_at = _last_announce_at = 0.0
    _last_ranker_log_at = _last_ranker_announce_at = 0.0
