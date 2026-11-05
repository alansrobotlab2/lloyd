"""Pre-warm the KV prefix of the one predictable job prompt (#3771).

Why this exists
---------------
Iteration 1 is the cold request of every session and it is the worst row in the
store. ``scripts/prefix_cache_iteration_table.py --boot 2026-10-28T00:00:00``
(2561 sessions in scope, measured 2026-11-05) prints worker iteration 1 at
**22.7% cached over 1839 samples** against iteration 2 at 80.7% and 8+ at 92.1%;
autonomy iteration 1 at 22.7% over 305 samples against iteration 2 at 85.0%.
Those prompts are the *predictable* ones: position 0 is the worker system prompt
``app.routers.turn_options.build_turn_options`` builds for a worker session —
``build_system_prompt(platform="worker")`` and nothing appended to it — and the
per-run part, the skill body and the item content, is the user message that comes
after it. (The denied-tools block is *not* in that prompt on this path: the worker
pool appends it to the user text, ``workers/sources/_common.py:1280``. It is in the
system prompt only for the in-process shape, at ``_common.py:710`` — the two shapes
do not share a head, which is the whole content of `warm_payload`'s docstring.)
A predictable prompt has a predictable prefill, so the prefill can be
paid at a chosen moment instead of in front of the first real request. That is
all this does: it sends the head once, early, and lets the runs arrive to blocks
the engine already holds.

Not an offline KV serialization
------------------------------
vLLM has no supported load-from-disk path for KV here, so the locally-runnable
form of Ramp's "KV cartridge" is a synthetic warm request — one completion of one
token over the head, ``max_tokens=1``, whose only product is the blocks the
engine keeps afterwards. The engine was always going to do that prefill; this
changes *when*, so the mechanism is a scheduling decision and the honest price is
the amortisation ratio, not a saved prefill.

The mechanism is the voice path's, which has measured it on this engine
(``app/routers/voice.py`` ``_prewarm``): first answer token 4.16 s cold against
0.45 s warm, and an 88k-token prompt prefilled in 7.6 s (15.6 s after a restart).
The difference is the trigger. Voice can predict a human; a job cannot, so the
triggers here are the two deterministic events that *empty* the cache — an engine
boot, and a prompt-surface rewrite that changes the head — plus the head-sha
change itself, which catches every other writer.

What bounds the cost
--------------------
A warm that serves no run is pure added cost, so: one job source
(``DEFAULT_SOURCE``), one request per *head version* (``_WARMED`` is keyed on the
head's sha256, so the same head cannot be warmed twice), both firing paths gated
on the primary being idle (``engine_busy``), and ``max_tokens=1``. The running
tally — warms issued, warm tokens, and via the head sha the runs that hit them —
is one JSONL row per attempt at ``_pipeline/worker-prewarm.jsonl``. Whether the
ratio earns the switch is #3771's owed measurement; ``worker_prewarm:
{enabled: false}`` in config.yaml turns it off without a code change, and the
item's acceptance is that a hits-per-warm below 1 leaves it off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("lloyd-server.worker-prewarm")

#: What may ask for a warm. ``engine-ready``: the primary answered ``/metrics``
#: after a boot, so its cache is empty by definition. ``prompt-surface-write``: a
#: writer finished rewriting a loaded prompt surface (SOUL.md / MEMORY.md /
#: USER.md), which changes the head for every later job. ``head-sha-changed``: the
#: poll recomputed the head and the sha moved — the catch-all for any writer that
#: does not announce itself, including the UTC date rollover in
#: ``app/prompt_builder._skill_rank_counts``.
PREWARM_TRIGGERS = ("engine-ready", "prompt-surface-write", "head-sha-changed")

#: The one job source this warms, and the whole blast radius. ``autotriage`` is
#: 1465 of the pool's runs in the 7 days to 2026-11-05T02:00Z (``workers.db``,
#: ``runs`` table), which is most of the worker platform's 1839 iteration-1
#: samples. Warming a second source is #3771's owed ruling, not this module's to
#: take. Its plain triage turn passes no ``extra_disallowed``
#: (``workers/sources/autotriage.py:1165``), which is why the head built here with
#: the default envelope is the head that turn sends; the sweep path at
#: ``workers/sources/autotriage.py:1484`` adds ``SWEEP_DISALLOWED`` and so has a
#: different head, and a different head simply does not hit this warm.
DEFAULT_SOURCE = "autotriage"

#: The platform whose head this warms. ``build_turn_options`` takes a turn's
#: platform from the *session*, not from the request body (``platform =
#: identity[0]``, `app/routers/turn_options.py:193`), and the identity comes from
#: ``SessionSnapshot.identity`` — so this constant reaches the builder only
#: through a snapshot that also says ``exists=True``, which is what the warm
#: snapshot now does. It is the value autotriage asks ``new_worker_session`` to
#: write for its own sessions — ``new_worker_session`` hands ``create_session``
#: an explicit ``platform="worker"`` (``workers/sources/_common.py:1079``).
#: Getting it wrong would warm a head no
#: worker is ever shown, because the platform is also what arms the turn's tool
#: policy and decides whether ``USER.md`` is in the prompt.
PLATFORM = "worker"

#: The ``source`` half of the identity the warm has to answer with. It is the
#: name autotriage hands ``run_prompt_in_session``
#: (`workers/sources/autotriage.py:1164`, whose ``NAME`` is ``"autotriage"`` at
#: that module's :34) and which therefore lands in the session file the run's
#: iteration 1 is built from. It matters because the automod ban
#: (`messages._ban_automod_for_workers`, #709) carves out exactly one source —
#: ``autocode`` — and reads it from ``SessionSnapshot.identity``: a warm that
#: answered ``("worker", "")`` would be a worker turn whose source is unknown,
#: which is a policy this job's own turns do not have.
SOURCE = "autotriage"

#: The model the warm asks for, and the reason it is a name rather than a
#: default. ``resolve_model`` reads the request body before the session
#: (`app/routers/turn_options.py:173`), and the body a worker turn actually
#: posts carries ``run_prompt_in_session``'s own default, ``"primary"``
#: (`workers/sources/_common.py:1152`), so the real job resolves its engine that
#: way and the warm has to resolve it the same way to warm the same engine's
#: cache — the primary is the only one worth warming, since the secondary runs
#: ``--parallel 1`` and keeps no prefix to reuse.
MODEL = "primary"

#: How often the poll recomputes the head sha. One build is a handful of vault
#: reads; the invalidators it watches for are nightly, not per-minute.
DEFAULT_POLL_INTERVAL_S = 900.0
#: How long after a boot to wait for the primary to answer ``/metrics`` before
#: giving up on the ``engine-ready`` warm. Past it the head is still warmed by the
#: poll or the next boot, whichever comes first.
DEFAULT_READY_WAIT_S = 300.0
DEFAULT_READY_POLL_S = 5.0
#: ``_WARMED`` is an in-process register of head versions, one entry per warm. The
#: cap is a leak guard, not a policy: the sha of the head last warmed is the one
#: that matters, and the older entries only exist to stop a re-warmed head (a
#: rewrite that reverts) finding its old row and being refused.
_WARMED_CAP = 32
#: The user turn the warm sends, and the only payload it sends besides the head.
#: The head is the system prompt and the loop puts it at position 0 by itself
#: (`app/harness/loop.py:251`), so a warm request needs some user turn only to make
#: it a request; one character of one is the cheapest thing the ChatML envelope
#: accepts. What the warm therefore buys is the head's blocks — position 0 of the
#: rendered prompt and by far its largest span, against a real job whose per-run
#: item content is hundreds of tokens after it. It is NOT a strict extension of the
#: warm the way the voice warm is (`app/routers/voice.py:437` drops a trailing
#: history message, and there is no history here to drop), which is exactly the
#: distinction #520's committed probe turns on: over an engine that read an
#: identical resend as 0 cached and a strict extension as 4800/60.0%, this is a
#: third shape, and it is why the item's owed A/B measures this one rather than
#: borrowing the voice path's number.
_SEED_MESSAGE = {"role": "user", "content": "."}

# ── registers ────────────────────────────────────────────────────────────

#: Head shas with a warm running right now — never two over one prefix.
_INFLIGHT: set[str] = set()
#: Head shas that a *successful* send covered, oldest first, mapped to the trigger
#: that covered them. A failure deliberately does not appear here: a warm that could
#: not be sent warmed nothing, and registering it would refuse the retry and turn one
#: engine hiccup into a head that never gets warmed at all. Emptied by the first
#: ``engine-ready`` after a boot — see ``_BOOT_PENDING``.
_WARMED: dict[str, str] = {}
#: The last head sha this process computed. The poll fires on a move of this, so a
#: nightly rewrite costs one warm and a quiet night costs none.
_SEEN_SHA: str = ""
#: Has the boot warm not happened yet? Consumed by the first ``engine-ready``
#: trigger, which is the only event entitled to bypass the once-per-head rule — a
#: restart is what empties the engine's cache, so a head warmed before one is warm
#: for nobody. Without this flag the bypass would be permanent, and every
#: ``engine-ready`` POST would be allowed to re-pay the same prefill.
_BOOT_PENDING: bool = True
_task: Optional[asyncio.Task] = None


def reset_state() -> None:
    """Forget every register. For tests, and for a poller pointed elsewhere."""
    global _SEEN_SHA, _BOOT_PENDING, _task
    _INFLIGHT.clear()
    _WARMED.clear()
    _SEEN_SHA = ""
    _BOOT_PENDING = True
    _task = None


# ── config ───────────────────────────────────────────────────────────────


def _cfg() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict(CONFIG.get("worker_prewarm") or {})
    except Exception:  # noqa: BLE001 — an unreadable config is a default, not a crash
        return {}


def enabled() -> bool:
    """On by default, off on one config line.

    The value of the mechanism is unproven on this engine for the job path — that
    is #3771's owed A/B — but the cost of being wrong is bounded to one 1-token
    request per head version, issued only while the primary is idle, which is
    smaller than the turn it is trying to help. On is also the only setting that
    can produce the 7 days of traffic the item needs to price itself.
    """
    return bool(_cfg().get("enabled", True))


def source() -> str:
    return str(_cfg().get("source") or DEFAULT_SOURCE)


def poll_interval_s() -> float:
    return float(_cfg().get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_S))


def log_path() -> Path:
    from app.paths import PIPELINE_DIR
    return PIPELINE_DIR / "worker-prewarm.jsonl"


# ── the payload ──────────────────────────────────────────────────────────


def warm_sha(text: str) -> str:
    """The head's identity. The log row and the registers are keyed on this."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Why the head build last fell back to the worker path. A warm built on the wrong
#: bytes is wasted, never wrong, so this is a detail for whoever reads the journal
#: and not a state anything acts on; `head_build_error()` is how the test sees it.
_head_build_error: str = ""


def head_build_error() -> str:
    """Why the head build last fell back (``""`` when it has not since fallen back)."""
    return _head_build_error


def _worker_fallback_payload(src: str | None) -> tuple[str, Any]:
    """The head as ``_worker_run_options`` builds it — a degraded warm, not a warm.

    Reached only when the backend's own builder raises. It is a different string
    from the real head and is documented as such at its call site, because the
    worker path appends the denied-tools block to the system prompt
    (`workers/sources/_common.py:710`) where the job's real head does not: warming
    it can only waste a prefill, never corrupt one.
    """
    from workers.sources._common import _worker_run_options

    options = _worker_run_options(max_turns=1, source=src or source(),
                                  session_id="")
    options.extra_body = {**options.extra_body, "max_tokens": 1}
    return options.system_prompt, options


def warm_payload(src: str | None = None) -> tuple[str, Any]:
    """``(the stable head, the options that send it)``, from the builder the job uses.

    Autotriage is a session-backed source: `workers/sources/autotriage.py:1163`
    dispatches through ``run_prompt_in_session``, and `_common._run_turn` POSTs the
    contract to ``/api/message/stream`` with **no system prompt in the body**. The
    head the job actually sends is therefore built in the backend, by
    ``app.routers.turn_options.build_turn_options`` — which calls
    ``build_system_prompt(platform="worker")`` and nothing else — and the denied-tools
    block that list carries goes into the *user* text (`_common.py:1280`), not into
    that prompt. So this calls the backend builder, and holds no string of its own.

    It used to call ``_worker_run_options``, which is the *other* shape's builder:
    for a source that runs its turn in-process, where there is no backend to append
    to the user text and so the block is appended to the system prompt at
    `_common.py:710`. Those two heads are not the same bytes, so a warm built that
    way warmed a prefix the real job never sends. The node that is supposed to see
    this (`tests/test_worker_prewarm.py`) previously took both sides of its sha256
    equality from ``_worker_run_options``, which is why the mistake survived; the
    real job's side is now captured out of a real ``_run_turn`` POST.

    The snapshot is a fresh turn's, not a session's: the warm belongs to no run, and
    every field that would make it one (todos, plan, goal, the frozen memory) is a
    per-run variable, which is precisely what must not be in a prefix. A real
    worker session created by ``new_worker_session`` has those fields empty too, which
    is what makes the two sides comparable; ``tests/test_worker_prewarm.py`` pins
    that, and reddens if a fresh worker session ever starts carrying state the warm
    would miss.

    It is ``exists=True``, and that is not a formality. Two of the controls that
    change the rendered bytes are armed from ``SessionSnapshot.identity``, and that
    property (``app/routers/turn_options.py:129-131``) answers
    ``("mission-control", "")`` for any snapshot that does not exist — whatever
    ``raw_platform`` was handed it — while ``build_turn_options:193`` takes
    ``platform = identity[0]``. The first version of this module passed
    ``raw_platform=PLATFORM`` and nothing else, so the kwarg was inert: the warm
    rendered a *chat* head (the 60,315-byte one, carrying the ``## USER.md`` section a
    worker turn drops, against the autotriage head's 43,988) and armed neither
    ``_ban_grant_minting`` (:225) nor ``_ban_automod_for_workers`` (:228), both keyed
    on that identity's platform being in ``NON_USER_PLATFORMS`` — so its tools array
    differed too, which is a different rendered prefix and a guaranteed miss.
    ``source=SOURCE`` is load-bearing for the same reason: the automod ban carves out
    exactly one source, ``autocode``, and reads it from that same tuple.

    The returned options are the built turn's, minus the generation and the identity:
    ``max_tokens=1``, ``max_turns=1``, no session (so no scratchpad, no component
    note and no phantom session row for a request that belongs to no run), and
    ``base_url`` re-pointed at the primary, which is the engine the item names and
    the only one with a cache to warm (`--parallel 1` on the secondary).
    """
    global _head_build_error
    from app.routers.turn_options import SessionSnapshot, build_turn_options

    try:
        built = build_turn_options(
            SessionSnapshot(session_id="", path=Path(""), exists=True,
                            raw_platform=PLATFORM, source=SOURCE),
            {"model": MODEL}, "stream")
    except Exception as exc:  # noqa: BLE001 — a failed warm is worse than no warm
        _head_build_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "worker_prewarm: the head build through build_turn_options failed "
            "(%s); the worker-path head differs from the real job's, so a warm "
            "from it can only be wasted", _head_build_error)
        return _worker_fallback_payload(src)
    _head_build_error = ""

    options = built.options
    options.extra_body = {**options.extra_body, "max_tokens": 1}
    options.max_turns = 1
    options.session_id = ""
    options.session_persist = False
    options.turn_tail = ""
    options.on_event = None
    options.context_meter = None
    options.chat_messages_handle = None
    options.cancel_event = None
    return built.system_prompt, options


# ── admission ────────────────────────────────────────────────────────────


def engine_busy() -> bool:
    """Is the primary serving something right now? Fail-open, like its source.

    Reads the sampled gauge (`app/engine_pressure.py`), never the engine at the
    moment of asking — the same reason the worker pool's KV gate does not call
    ``/metrics`` inline. A warm that lands beside a human's turn steals prefill
    from the one request on this box somebody is waiting on, which is the item's
    named risk; a warm skipped because the gauge was stale costs one head version.
    """
    from app import engine_pressure

    sample = engine_pressure.latest()
    if sample is None:
        return False
    return bool((sample.running or 0) > 0 or (sample.waiting or 0) > 0)


def _refusal(trigger: str, head_sha: str, *, enabled: bool,
             busy: bool) -> Optional[str]:
    """Why this request does not start a warm, or None when it does.

    The voice path's ``_prewarm_refusal`` (`app/routers/voice.py:361`) is the
    shape: a pure function over its arguments plus the module registers, so the
    rules are testable without an engine. Four reasons, in the order they are
    checked:

      * ``disabled`` — the switch. Checked first because a warm that is switched
        off must not reach the engine, the vault or the prompt builder.
      * ``in flight`` — a warm over THIS head sha is running. Keyed on the sha and
        not on the source because two triggers can fire over one head (a boot and
        the nightly write landing together), and the second of them is the same
        prefill the first is already paying for.
      * ``busy`` — the engine is serving a request. A warm is the least important
        thing the primary could be doing.
      * ``already warmed`` — this head sha has been sent once. This is the clause
        that makes the trigger storm harmless: an unchanged head cannot be warmed
        twice, however many events announce it.

    A voice warm has a fifth reason this one does not (a debounce, because a wake
    word fires per sentence); a job prompt has no such clock, and the sha register
    is the stronger rule anyway.
    """
    if not enabled:
        return "disabled"
    if head_sha in _INFLIGHT:
        return "in flight"
    if busy:
        return "busy"
    if head_sha in _WARMED:
        return "already warmed"
    return None


# ── the row ──────────────────────────────────────────────────────────────


def _row(**fields: Any) -> dict[str, Any]:
    """One machine-readable line of the warm ledger, and its in-memory twin.

    ``outcome`` is ``sent`` / ``refused`` / ``failed``, and it is on every row: the
    item's price is warms issued against runs that hit them, and a ledger of
    successes only would read as a perfect ratio over a handful of attempts.
    ``sha256`` is the join key to the runs — `scripts/prefix_cache_iteration_table.py`
    attributes a run's cached fraction, so warm count and warm tokens come from
    these rows and hits-per-warm comes from joining the sha to the runs after it.
    """
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "worker_prewarm",
        **fields,
    }
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 — a ledger that cannot be written is not a reason to skip a warm
        logger.warning("worker_prewarm: could not append to %s: %s", path, exc)
    logger.info("worker_prewarm: %s", json.dumps(row, sort_keys=True))
    return row


# ── firing ───────────────────────────────────────────────────────────────


async def _send_warm(options: Any) -> dict[str, Any]:
    """One completion of one token over the head. Returns the engine's usage."""
    from app.harness import run_query

    usage: dict[str, Any] = {}
    async for evt in run_query([dict(_SEED_MESSAGE)], options):
        if isinstance(evt, dict) and evt.get("type") == "assistant_message":
            usage = dict(evt.get("usage") or {})
    return usage


def _usage_tokens(usage: dict[str, Any]) -> tuple[int | None, int | None]:
    """(prompt tokens, cached prompt tokens) as the engine reported them.

    ``None`` when the engine said nothing, which is a real state and not a zero:
    `app/prefix_miss.py:25` records that this engine reads ``cached_tokens`` as 0
    for the first two requests of a prefix, so a warm's own cache_read of 0 is
    what a cold head looks like and not evidence of anything.
    """
    prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
    cached = usage.get("cache_read")
    if cached is None:
        cached = usage.get("prompt_tokens_cached")
    return (int(prompt) if prompt else None,
            int(cached) if cached is not None else None)


async def prewarm(trigger: str, *, src: str | None = None) -> dict[str, Any]:
    """One warm attempt, and one ledger row either way. Never raises.

    The two triggers that are not the poll arrive over HTTP
    (``POST /api/worker/prewarm``, ``app/routers/worker_prewarm.py``), which is how
    a writer in another process — the nightly consolidation, an ``automod_vault_land``
    — reports that it finished rewriting the surface every worker prompt is built
    from. ``engine-ready`` comes from `start()` in this module.
    """
    global _SEEN_SHA, _BOOT_PENDING
    t0 = time.monotonic()
    if trigger not in PREWARM_TRIGGERS:
        return _row(trigger=trigger, outcome="refused", reason="unknown trigger",
                    source=src or source(), sha256=None, prompt_chars=None)
    try:
        head, options = warm_payload(src)
    except Exception as exc:  # noqa: BLE001 — no head, no warm, and a row saying so
        # The landing drain raises here (`workers/sources/_common.py:624`), which is
        # the item's existing idleness discipline arriving for free: a promotion in
        # flight restarts the backend, so a warm now warms a prefix into an engine
        # that is about to be asked nothing.
        return _row(trigger=trigger, outcome="failed",
                    reason=f"head build failed: {type(exc).__name__}: {exc}",
                    source=src or source(), sha256=None, prompt_chars=None)

    sha = warm_sha(head)
    _SEEN_SHA = sha
    if trigger == "engine-ready" and _BOOT_PENDING:
        # The one bypass the once-per-head rule allows. The engine restarted, so its
        # cache is empty by definition and every entry in the register names a head
        # that is warm for nobody; keeping them would refuse exactly the warm that is
        # worth paying for. Consumed here, once, so a caller that spams this trigger
        # cannot re-pay the same prefill at will.
        _BOOT_PENDING = False
        _WARMED.clear()
    why = _refusal(trigger, sha, enabled=enabled(), busy=engine_busy())
    if why is not None:
        return _row(trigger=trigger, outcome="refused", reason=why,
                    source=src or source(), sha256=sha, prompt_chars=len(head))

    _INFLIGHT.add(sha)
    try:
        usage = await _send_warm(options)
    except Exception as exc:  # noqa: BLE001 — a failed warm is a cost, not an incident
        _INFLIGHT.discard(sha)
        return _row(trigger=trigger, outcome="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    source=src or source(), sha256=sha, prompt_chars=len(head),
                    seconds=round(time.monotonic() - t0, 3))
    _INFLIGHT.discard(sha)
    _WARMED[sha] = trigger
    while len(_WARMED) > _WARMED_CAP:
        _WARMED.pop(next(iter(_WARMED)))
    prompt_tokens, cached_tokens = _usage_tokens(usage)
    return _row(trigger=trigger, outcome="sent", source=src or source(),
                sha256=sha, prompt_chars=len(head), max_tokens=1,
                prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                seconds=round(time.monotonic() - t0, 3))


def head_sha(src: str | None = None) -> str:
    """The sha of the head as it stands right now. The poll's only question."""
    try:
        head, _ = warm_payload(src)
    except Exception:  # noqa: BLE001 — unreadable right now is "unchanged"
        return _SEEN_SHA
    return warm_sha(head)


async def _wait_engine_ready() -> bool:
    """True once the primary answers ``/metrics``, within the boot budget."""
    from app import engine_pressure

    deadline = time.monotonic() + DEFAULT_READY_WAIT_S
    while time.monotonic() < deadline:
        if await engine_pressure.scrape_once() is not None:
            return True
        await asyncio.sleep(DEFAULT_READY_POLL_S)
    return False


async def _run() -> None:
    """Boot warm, then the head-sha poll."""
    try:
        if await _wait_engine_ready():
            await prewarm("engine-ready")
    except Exception:  # noqa: BLE001 — the poll below is the retry
        logger.exception("worker_prewarm: engine-ready warm failed")
    while True:
        await asyncio.sleep(poll_interval_s())
        try:
            if head_sha() != _SEEN_SHA:
                await prewarm("head-sha-changed")
        except Exception:  # noqa: BLE001 — a poll that dies stops the trigger
            logger.exception("worker_prewarm: head-sha poll iteration failed")


async def start() -> None:
    """Startup hook. Idempotent; a no-op when the switch is off."""
    global _task
    if not enabled():
        logger.info("worker_prewarm disabled in config — no warm request for %s",
                    source())
        return
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_run(), name="lloyd-worker-prewarm")
    logger.info("worker_prewarm: warming %s's prompt head at boot, "
                "then every %.0fs on a head-sha change",
                source(), poll_interval_s())


async def stop() -> None:
    """Shutdown hook."""
    global _task
    task, _task = _task, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
