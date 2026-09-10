"""Prefix-cache misses on long re-admissions: count them, and say so.

Every agent-loop iteration is a fresh request that re-submits the whole
conversation. While the engine still holds that prefix, an iteration pays only
for what was appended since the last one — 5-7k tokens on a 207k prompt,
measured 2026-09-10 on the FP8 build. When the prefix has been evicted in
between, the whole prompt is prefilled again from cold, one 8,192-token chunk
per engine step, and every other request on the engine gets one token per
step until it finishes. That is the 09-09 "5 tok/s" stall: every one of the
27 episodes in that day's engine log was a 100-200k prompt re-prefilled from
cold beside a chat. The FP8 cutover (a 1.74x pool, 3200-token pages) made each
one cheaper; nothing measured whether they still happen. This does.

The data was always there — `cache_read` per iteration, on each assistant
row's `stats` since `5531f21` — and what was missing was a number and a bell.
Three rules decide whether the number means anything, each learned the
expensive way (memory `vllm-admission-stall`):

  * **Iterations 1 and 2 are not counted.** The engine reports 0 cached for
    the first two requests of any prefix, and a turn's first iteration
    legitimately re-admits history the engine may not have seen for hours.
  * **A turn that reads zero on every counted iteration is unmeasured, not
    cold.** Before 5531f21 nothing parsed `cached_tokens` and every session
    read 0 everywhere; a regression of that parser would look exactly like
    every iteration missing. So a miss is reported only once the same turn
    has shown a non-zero read. Earlier candidates wait and are confirmed
    then, and a turn that never shows one reports `None`, not a count.
  * **Only a miss counts toward `reprefill_tokens`.** The plan's first draft
    summed the uncached tail of every iteration, but a healthy iteration's
    tail is the tool result it just appended plus up to a page of alignment,
    so a 60-iteration round with a perfect cache would read ~300k and the
    alert would fire on health. What is summed is the uncached part of the
    iterations that missed — which is also what lloyd-be's fleet baseline
    summed (194 misses, 20.6M tokens over 09-08/09).

Both writers call `record_iteration` — `app/routers/messages.py` for chat-path
turns, `app/run_recorder.py` for direct background runs — for the same reason
they share `app/transcript_entries.py`: two private definitions of "a miss"
would come to disagree about the same turn.

The bell is `announce()` on the guardian's one fan-out: news, not an incident
(journal and toast; voice only if `harness.prefix_miss.announce_voice`). It
fires once per turn, when that turn's misses pass
`announce_reprefill_tokens` while at least one other request was running
during the miss, and not more often than `announce_cooldown_seconds` across
the process. A cold prefill with the engine otherwise idle hurts nobody and
is not news.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("lloyd-server")

#: First iteration that can miss. See the module docstring.
MIN_ITERATION = 3
#: Below this a cold re-prefill costs about a second and is not the stall.
MIN_INPUT_TOKENS = 100_000
#: An iteration that reused less than this share of its prompt missed.
MAX_REUSE_FRACTION = 0.5
#: A turn whose misses re-prefilled more than this, beside a neighbour, is
#: announced. One fully cold 100k+ re-admission crosses it.
ANNOUNCE_REPREFILL_TOKENS = 100_000
#: Process-wide floor between announcements. A bad hour is one toast.
ANNOUNCE_COOLDOWN_S = 1800.0

LogFn = Callable[[str, dict], None]


def _cfg() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict((CONFIG.get("harness") or {}).get("prefix_miss") or {})
    except Exception:
        return {}


def enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def usage_numbers(usage: dict | None) -> tuple[int, int]:
    """(prompt tokens, cached prompt tokens) from a harness usage dict.

    Accepts both spellings the writers already accept: `_merge_usage` maps
    vLLM's `prompt_tokens` to `input_tokens` and lifts
    `prompt_tokens_details.cached_tokens` to `cache_read`.
    """
    u = usage or {}

    def _first(*keys: str) -> int:
        for key in keys:
            value = u.get(key)
            if isinstance(value, (int, float)) and value:
                return int(value)
        return 0

    return (_first("input_tokens", "prompt_tokens"),
            _first("cache_read", "prompt_tokens_cached"))


def label_for_session(session_id: str) -> str:
    """Who an announcement names: a background run's source, else a chat.

    A background session id has four underscore-separated parts
    (`20260910_120001_autocode_9f2a`) and a chat's has three — the same
    fast path `sessions_io.is_background_session_name` takes.
    """
    parts = (session_id or "").split("_")
    if len(parts) >= 4 and parts[2]:
        return parts[2]
    return "a chat turn"


@dataclass
class Miss:
    iteration: int
    input_tokens: int
    cache_read: int
    duration_ms: int = 0
    #: time.monotonic() when the iteration's stream ended.
    at: float = 0.0

    @property
    def uncached(self) -> int:
        return max(0, self.input_tokens - self.cache_read)

    @property
    def reuse(self) -> float:
        return self.cache_read / self.input_tokens if self.input_tokens else 0.0

    def as_event(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "input_tokens": self.input_tokens,
            "cache_read": self.cache_read,
            "uncached_tokens": self.uncached,
            "reuse": round(self.reuse, 3),
            "duration_ms": self.duration_ms,
        }


@dataclass
class TurnMissTracker:
    """One turn's misses. Arithmetic only; `record_iteration` does the I/O."""

    session_id: str = ""
    turn_id: str = ""
    label: str = ""
    min_iteration: int = MIN_ITERATION
    min_input_tokens: int = MIN_INPUT_TOKENS
    max_reuse_fraction: float = MAX_REUSE_FRACTION
    #: True once any iteration of this turn has read a non-zero cache_read,
    #: which is what proves the field is being parsed at all.
    cache_seen: bool = False
    #: Iterations at or past `min_iteration` folded in so far.
    counted: int = 0
    confirmed: list[Miss] = field(default_factory=list)
    #: Candidates seen before `cache_seen` — misses or the parser bug, and
    #: the next non-zero read decides which.
    pending: list[Miss] = field(default_factory=list)
    announced: bool = False

    @classmethod
    def for_turn(cls, session_id: str = "", turn_id: str = "",
                 label: str = "") -> "TurnMissTracker":
        cfg = _cfg()
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            label=label or label_for_session(session_id),
            min_iteration=int(cfg.get("min_iteration", MIN_ITERATION)),
            min_input_tokens=int(cfg.get("min_input_tokens", MIN_INPUT_TOKENS)),
            max_reuse_fraction=float(cfg.get("max_reuse_fraction",
                                             MAX_REUSE_FRACTION)),
        )

    def observe(self, iteration: int, usage: dict | None, *,
                duration_ms: int = 0, now: float | None = None) -> list[Miss]:
        """Fold one iteration in. Returns the misses this call confirmed —
        its own, and any earlier candidates a first non-zero read vindicates."""
        inp, cached = usage_numbers(usage)
        if cached > 0:
            self.cache_seen = True
        released: list[Miss] = []
        if self.cache_seen and self.pending:
            released, self.pending = self.pending, []
            self.confirmed.extend(released)
        if iteration < self.min_iteration:
            return released
        self.counted += 1
        if inp < self.min_input_tokens or cached >= self.max_reuse_fraction * inp:
            return released
        miss = Miss(iteration, inp, cached, int(duration_ms or 0),
                    time.monotonic() if now is None else now)
        if self.cache_seen:
            self.confirmed.append(miss)
            released.append(miss)
        else:
            self.pending.append(miss)
        return released

    @property
    def measured(self) -> bool:
        """Whether this turn's count means anything. A turn with no counted
        iteration cannot miss by definition, so it is measured at zero."""
        return self.cache_seen or self.counted == 0

    @property
    def reprefill_tokens(self) -> int:
        return sum(m.uncached for m in self.confirmed)

    def summary(self) -> dict[str, int | None]:
        """The two numbers a turn's stats and its usage row carry. `None`
        means unmeasured — never a zero standing in for "could not tell"."""
        if not self.measured:
            return {"reprefill_tokens": None, "prefix_misses": None}
        return {"reprefill_tokens": self.reprefill_tokens,
                "prefix_misses": len(self.confirmed)}


# ── I/O: event log, server log, announcement ─────────────────────────

_tasks: set[asyncio.Task] = set()
_last_announce_at: float = 0.0


def record_iteration(tracker: TurnMissTracker, iteration: int,
                     usage: dict | None, *, duration_ms: int = 0,
                     log: LogFn | None = None) -> list[Miss]:
    """Fold one `assistant_message` in, log its misses, and maybe announce.

    Never raises and never blocks the stream: the fold is arithmetic, the log
    is one append, and the announcement — which may need an HTTP read and
    runs notify-send — is a task the caller does not wait on.
    """
    if not enabled():
        return []
    try:
        released = tracker.observe(iteration, usage, duration_ms=duration_ms)
    except Exception as exc:  # noqa: BLE001 — accounting is not the turn
        logger.debug("prefix_miss: observe failed: %s", exc)
        return []
    for miss in released:
        # INFO, deliberately: the guardian reads logs/server.err during an
        # observation window, and a slow iteration is not an error.
        logger.info(
            "prefix_miss: %s %s iteration %d re-prefilled %d of %d prompt "
            "tokens (%.0f%% cached)", tracker.label, tracker.session_id,
            miss.iteration, miss.uncached, miss.input_tokens, 100 * miss.reuse)
        _log(log, "brain1.prefix_miss", {
            **miss.as_event(),
            "turn_reprefill_tokens": tracker.reprefill_tokens,
            "turn_prefix_misses": len(tracker.confirmed),
        })
    if released and not tracker.announced:
        _spawn(maybe_announce(tracker, released[-1], log=log))
    return released


def finish(tracker: TurnMissTracker, *, log: LogFn | None = None
           ) -> dict[str, int | None]:
    """The turn's summary, saying so once if it could not be measured."""
    summary = tracker.summary()
    if enabled() and not tracker.measured:
        logger.info(
            "prefix_miss: %s read cache_read=0 on all %d counted iterations — "
            "unmeasured. That is the unread-field signature (see 5531f21), "
            "not %d misses.", tracker.session_id, tracker.counted,
            len(tracker.pending))
        _log(log, "brain1.prefix_miss_unmeasured", {
            "counted_iterations": tracker.counted,
            "candidates": [m.as_event() for m in tracker.pending],
        })
    return summary


def _log(log: LogFn | None, name: str, data: dict) -> None:
    if log is None:
        return
    try:
        log(name, data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("prefix_miss: event log write failed: %s", exc)


def _spawn(coro) -> None:
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:  # no loop — a synchronous caller, e.g. a unit test
        coro.close()
        return
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def announcement(tracker: TurnMissTracker, miss: Miss,
                 neighbours: int) -> tuple[str, str]:
    """Toast head and body. Pure, so it can be pinned."""
    others = f"{neighbours} other request{'' if neighbours == 1 else 's'}"
    title = f"Prefix-cache miss: {tracker.label or 'a turn'}"
    body = (f"Iteration {miss.iteration} re-prefilled {_k(miss.uncached)} of "
            f"{_k(miss.input_tokens)} prompt tokens from cold "
            f"({miss.reuse:.0%} cached) beside {others}, which get one token "
            f"per prefill chunk until it finishes. "
            f"{_k(tracker.reprefill_tokens)} re-prefilled this turn.")
    return title, body


def _k(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


async def _neighbours_during(miss: Miss) -> int | None:
    from app import engine_pressure

    t0 = miss.at - max(0, miss.duration_ms) / 1000.0
    n = engine_pressure.neighbours_during(t0, now=miss.at)
    if n is not None:
        return n
    # No sample landed inside the iteration (sampler off, or a short one).
    # Read now instead: the miss's own request has finished, so whatever is
    # running is somebody else.
    gauges = await engine_pressure.scrape_once()
    if gauges and gauges.get("requests_running") is not None:
        return int(gauges["requests_running"])
    return None


async def maybe_announce(tracker: TurnMissTracker, miss: Miss, *,
                         log: LogFn | None = None) -> bool:
    """Announce this turn's misses once, if they cost a neighbour enough."""
    global _last_announce_at
    cfg = _cfg()
    try:
        if not cfg.get("announce", True) or tracker.announced:
            return False
        threshold = int(cfg.get("announce_reprefill_tokens",
                                ANNOUNCE_REPREFILL_TOKENS))
        if tracker.reprefill_tokens <= threshold:
            return False
        cooldown = float(cfg.get("announce_cooldown_seconds",
                                 ANNOUNCE_COOLDOWN_S))

        def _cooling() -> bool:
            return bool(_last_announce_at) and \
                time.monotonic() - _last_announce_at < cooldown

        if _cooling():
            return False
        neighbours = await _neighbours_during(miss)
        if not neighbours:
            return False
        # Re-checked after the await: another turn may have announced.
        if tracker.announced or _cooling():
            return False
        tracker.announced = True
        _last_announce_at = time.monotonic()
        title, body = announcement(tracker, miss, neighbours)
        voice = bool(cfg.get("announce_voice", False))
        channels = await asyncio.to_thread(_announce, title, body, voice)
        logger.info("prefix_miss: announced %r via %s", title, channels)
        _log(log, "brain1.prefix_miss_announced", {
            "title": title, "body": body, "neighbours": neighbours,
            "turn_reprefill_tokens": tracker.reprefill_tokens,
            "channels": channels,
        })
        return True
    except Exception as exc:  # noqa: BLE001 — a bell is not the turn
        logger.info("prefix_miss: announcement failed: %s", exc)
        return False


def _announce(title: str, body: str, voice: bool) -> dict:
    """Through the guardian's one fan-out, as `announce`: journal and toast,
    and speech only when asked. Same import route `scripts/automod/promote.py`
    takes for its landing announcement."""
    import sys
    from pathlib import Path

    gdir = Path(__file__).resolve().parents[1] / "agent-services" / "guardian"
    if not (gdir / "notify.py").is_file():
        return {}
    if str(gdir) not in sys.path:
        sys.path.insert(0, str(gdir))
    import gstate
    import notify as notify_mod
    import policy

    notifier = notify_mod.Notifier(
        ledger=gstate.AutomodState(Path(policy.AUTOMOD_STATE)).ledger,
        state_dir=Path(policy.GUARDIAN_STATE),
        vault_root=policy.VAULT_ROOT,
        voice=voice,
        voice_window=policy.VOICE_REPEAT_SECONDS,
    )
    return notifier.announce(title, body, level="info")
