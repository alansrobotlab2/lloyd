"""Backlog #458 — async IV judgment calls survive momentary engine saturation.

Measured 2026-09-01→09-12: 351 of 9,006 LLM-call rows in
`inner_voice_observations` ended with no verdict, every one folded into
`action='noop'` with `error` set, and they are episodic — three one-hour
windows hold 24 % of them, the worst hour 36/44 = 81.8 %. The engine stops
answering for an hour and every judgment submitted in that hour dies.
Per-call latency of calls that did answer is p50 1,731 ms against a 12 s
async budget, so the drops are queue starvation, not a slow tail — and
`latency_ms` is a client-side timer started before the POST, so engine
queue time is charged against the same deadline as inference time and a
dropped row cannot tell the two conditions apart.

This file pins what the fix in `app/inner_voice/observer.py` guarantees,
clause by clause (the ids are #458's acceptance clauses):

1. async timeout + successful retry  -> the row carries a verdict, not a noop
2. an async call is retried exactly once — two attempts, never three
3. the two synchronous terminal calls keep first-deadline-returns behaviour
4. a dropped async call records `engine_unresponsive` vs `call_slow`,
   readable from the `error` column of inner_voice_observations
5. an unresponsive engine is detected by the pre-flight probe in ~1 s, and
   the abandoned call records latency_ms <= 2,000 ms, not the full 12 s

Engine seams: the POST seam is stubbed at
`_post_chat_completion_with_tools` (the one module-level function that
touches httpx on this path); the probe seam is stubbed at `_probe_engine`
everywhere except the clause-5 test, which binds a silent loopback socket
and lets httpx's real per-request read timeout fire against it —
httpx.MockTransport ignores the per-request `timeout` argument entirely,
so it cannot exercise the 1 s budget. The only socket ever opened is to
127.0.0.1 on an ephemeral port that nothing but this test's own listener
owns. Nothing here writes the real usage.db.

Run: .venvs/lloyd/bin/python -m pytest tests/test_iv_async_timeout_retry.py -q
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import sys
import time
import types
from pathlib import Path

import httpx

LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLOYD_HOME))

import usage_store  # noqa: E402
from app.inner_voice import observer as obs_mod  # noqa: E402

# The values production passes: 12 s for spawned non-terminal judgments
# (`async_timeout_seconds`), 5 s for the synchronous terminal ones
# (`timeout_seconds`), and the new `probe_timeout_seconds` default of 1.0.
CFG = {
    "timeout_seconds": 5.0,
    "async_timeout_seconds": 12.0,
    "probe_timeout_seconds": 1.0,
    "max_tokens": 64,
    # These pins are about the deadline semantics, which thinking changes
    # only by choosing a longer default; tests/test_iv_one_review.py pins that.
    "thinking": False,
}

ASYNC_TIMEOUT = CFG["async_timeout_seconds"]  # 12.0 — what production passes async
SYNC_TIMEOUT_DEFAULT = CFG["timeout_seconds"]  # 5.0 — no override, terminal path


def _lever_body(name: str = "inject", reason: str = "primary stalled", content: str = "wake up"):
    """A vLLM response body shaped the way _extract_tool_call consumes it."""
    assert name in obs_mod.LEVER_NAMES, f"stub lever {name!r} is not a real lever"
    return {
        "choices": [
            {"message": {"tool_calls": [{"function": {
                "name": name,
                "arguments": json.dumps({"reason": reason, "content": content}),
            }}]}}
        ],
        "usage": {"prompt_tokens": 9770, "completion_tokens": 42},
    }


class _Stub:
    """Seam recorder: per-attempt POST behaviour, per-call probe verdicts."""

    def __init__(self, post_behaviours=(), probe_results=()):
        self.post_behaviours = list(post_behaviours)
        self.probe_results = list(probe_results)
        self.posts = 0
        self.probes = 0
        self.probe_timeouts = []

    async def post(self, **kwargs):
        self.posts += 1
        if self.posts > len(self.post_behaviours):
            raise AssertionError(
                f"attempt {self.posts}: engine POST beyond the "
                f"{len(self.post_behaviours)} stubbed behaviour(s)"
            )
        item = self.post_behaviours[self.posts - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    async def probe(self, base_url, timeout_seconds):
        self.probes += 1
        self.probe_timeouts.append(timeout_seconds)
        if self.probes > len(self.probe_results):
            raise AssertionError(
                f"probe call {self.probes} beyond the "
                f"{len(self.probe_results)}-call script — an unscripted "
                "probe must not silently reuse the last verdict"
            )
        return self.probe_results[self.probes - 1]


def _install(monkeypatch, stub: _Stub):
    monkeypatch.setattr(obs_mod, "_post_chat_completion_with_tools", stub.post)
    monkeypatch.setattr(obs_mod, "_probe_engine", stub.probe)
    monkeypatch.setattr(
        obs_mod, "_resolve_endpoint",
        lambda alias=None: ("http://engine.test:9999", "stub-model"),
    )
    return stub


def _no_post_at_all():
    async def refuse(**kwargs):
        raise AssertionError("request was queued behind an engine the probe found dead")
    return refuse


_HEALTHY = (True, "replied 200")
_SILENT = (False, "no response in 1.0s")


# ---------------------------------------------------------------------------
# Clause 1 — async timeout, retry answers: the row carries a verdict
# ---------------------------------------------------------------------------


def test_async_timeout_then_successful_retry_yields_a_verdict_not_a_noop(monkeypatch):
    """First attempt hits the 12 s deadline, the retry answers `inject`.

    Before #458 this event recorded action='noop', error='timeout after
    12.0s' — the judgment was discarded whole even though the engine was
    demonstrably alive (it answered the retry). Now the verdict the retry
    produced is the decision: error None, action from the lever.
    """
    stub = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ReadTimeout("deadline"), _lever_body("inject")],
            probe_results=[_HEALTHY],
        ),
    )
    d = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert stub.posts == 2, "expected first attempt to fail and the retry to run"
    assert d.error is None, f"the retry's verdict must not carry an error: {d.error!r}"
    assert d.action == "inject"
    assert d.content == "wake up"
    assert d.reason == "primary stalled"
    assert d.input_tokens == 9770 and d.output_tokens == 42
    # Pre-flight exactly once (the retry answered, so there was no drop to
    # classify afterwards) — and at the probe budget, not the judgment's.
    assert stub.probes == 1
    assert stub.probe_timeouts == [CFG["probe_timeout_seconds"]]


# ---------------------------------------------------------------------------
# Clause 2 — retried exactly once
# ---------------------------------------------------------------------------


def test_async_call_that_times_out_twice_records_one_drop_and_two_attempts(monkeypatch):
    """Both attempts time out: two engine POSTs, never a third; one drop."""
    stub = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")],
            # pre-flight + post-drop classification; a third probe would
            # mean a third attempt or a mislabelled drop
            probe_results=[_HEALTHY, _HEALTHY],
        ),
    )
    d = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert stub.posts == 2, "an async call gets exactly one retry"
    assert d.action == "noop" and d.error is not None
    assert d.error.startswith("timeout after 12.0s"), d.error


def test_http_error_on_an_async_call_never_earns_a_retry(monkeypatch):
    """Only a deadline earns the second attempt.

    An HTTP-level error is the engine ANSWERING, and answering badly is not
    queue saturation that a re-submit would outrun — without this pin, the
    retry could silently widen into N attempts on every 5xx or reset.
    """
    stub = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ConnectError("connection refused")],
            probe_results=[_HEALTHY],
        ),
    )
    d = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert stub.posts == 1
    assert d.error is not None and d.error.startswith("http_error")


# ---------------------------------------------------------------------------
# Clause 3 — the synchronous terminal calls are untouched
# ---------------------------------------------------------------------------


def test_sync_terminal_call_returns_on_first_deadline_and_never_probes(monkeypatch):
    """`result` and the terminal `assistant_message` pass async_call=False.

    The primary is blocked on those two, so they must not pay for the
    pre-flight probe or a second attempt: one POST, one drop at the tight
    `timeout_seconds` budget (5.0 s here — what they get by not passing a
    timeout_override), no probe traffic, and no discriminator suffix on the
    error string.
    """
    stub = _install(
        monkeypatch,
        _Stub(post_behaviours=[httpx.ReadTimeout("deadline"), _lever_body()]),
    )
    d = asyncio.run(
        obs_mod._call_observer(user_prompt="terminal", cfg=dict(CFG))
    )
    assert stub.posts == 1, "the critical path must not queue a second attempt"
    assert stub.probes == 0, "the critical path must not pay for the probe"
    assert d.error == "timeout after 5.0s", (
        "sync drop keeps its exact pre-#458 string — no probe suffix, "
        f"no retry: {d.error!r}"
    )
    assert d.action == "noop"


# ---------------------------------------------------------------------------
# Clause 3, dispatch level — the flags the call SITES pass. The unit-level
# tests pin what _call_observer does with the flag; only this test pins
# that install_observer's own handlers set it correctly.
# ---------------------------------------------------------------------------


def test_observer_dispatch_marks_terminal_sync_and_off_path_async(monkeypatch):
    """The terminal review must reach `_call_observer` with async_call
    False; a mid-turn review, with it True; `result` and `tool_result` not at
    all (IV plan R2 removed both judgments).

    Dropping `and not is_terminal` at the terminal assistant_message site,
    or flipping `async_call` at any site, would put the probe and a full
    second attempt (about +13 s) on the primary's critical path — or take
    the retry off the off-critical-path calls — with every unit test green.

    Drives the real `install_observer` hooks over three events: a
    non-terminal tool_result, a terminal assistant_message (no
    tool_calls), and `result`. Only `_call_observer` itself is replaced,
    and it records the flags instead of phoning the engine.
    """
    from app.harness.hooks import HookRegistry

    calls: list[dict] = []

    async def recording_call(**kwargs):
        calls.append(kwargs)
        return obs_mod.ObserverDecision(action="noop", reason="recorded")

    async def no_goal_card(*a, **kw):
        return None

    monkeypatch.setattr(obs_mod, "_call_observer", recording_call)
    monkeypatch.setattr(obs_mod, "extract_goal_card", no_goal_card)
    monkeypatch.setattr(
        obs_mod, "record_inner_voice_observation", lambda **kw: len(calls),
    )
    cfg = obs_mod._observer_cfg()
    # sample_every=1 so the tool_result reaches the LLM instead of being
    # sampled out — sampling is a different guard with its own tests.
    # fast_path_enabled=False so the terminal assistant_message is judged
    # by the LLM rather than short-circuited by the fast path.
    cfg.update({"async_nonterminal": True, "review_every_iterations": 1})
    monkeypatch.setattr(obs_mod, "_observer_cfg", lambda: dict(cfg))

    async def scenario():
        hooks = HookRegistry()
        state = obs_mod.install_observer(
            hooks=hooks, session_id="dispatch-458", turn_id="dispatch_turn",
            user_request="pin the async_call flags",
            chat_messages_handle=[], cancel_event=asyncio.Event(),
            primary_model="stub-model",
        )
        await hooks.fire_on_event(
            {"type": "assistant_message", "text": "Checking the clauses.",
             "tool_calls": [{"function": {"name": "Bash"}}], "iteration": 1},
        )
        await hooks.fire_on_event(
            {"type": "tool_result", "name": "Bash", "content": "ok",
             "is_error": False},
        )
        await hooks.fire_on_event(
            {"type": "assistant_message", "text": "All clauses verified.",
             "tool_calls": [], "iteration": 2},
        )
        await hooks.fire_on_event(
            {"type": "result", "stop_reason": "end_turn",
             "response_text": "done"},
        )
        obs_mod.close_observer(state)

    asyncio.run(scenario())

    by_site: dict[str, list] = {}
    for c in calls:
        by_site.setdefault(
            "async" if c.get("async_call") else "sync", [],
        ).append(c)

    # The off-critical-path tool_result judgment: async flag True AND the
    # 12 s deadline — the flag and the budget travel together.
    assert by_site.get("async"), f"mid-turn review never judged as async: {calls}"
    for c in by_site["async"]:
        assert c["timeout_override"] == CFG["async_timeout_seconds"]
    # Both synchronous terminal calls (terminal assistant_message +
    # result): async_call False and NO timeout override, i.e. the tight
    # 5 s `timeout_seconds` still decides, with no probe or retry able to
    # attach.
    assert len(by_site.get("sync", [])) == 1, (
        f"expected exactly the one terminal review to run sync, got "
        f"{by_site.get('sync')}"
    )
    for c in by_site["sync"]:
        assert c.get("timeout_override") is None, c
        assert c.get("async_call") in (False, None), c


# ---------------------------------------------------------------------------
# Clause 4 — machine-readable discriminator between the two conditions
# ---------------------------------------------------------------------------


def test_drop_says_engine_unresponsive_when_the_probe_cannot_reach_the_engine(monkeypatch):
    """Probe fails pre-flight: abandoned for ~1 s, no POST ever queued,
    and the row says `engine_unresponsive`."""
    stub = _install(
        monkeypatch,
        _Stub(probe_results=[_SILENT]),
    )
    monkeypatch.setattr(obs_mod, "_post_chat_completion_with_tools", _no_post_at_all())
    d = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert stub.posts == 0, "a probe failure must not also spend the full deadline"
    assert d.error is not None and "engine_unresponsive" in d.error, d.error
    assert stub.probe_timeouts == [CFG["probe_timeout_seconds"]]


def test_drop_says_call_slow_when_the_engine_answers_but_the_request_overruns(monkeypatch):
    """Probe answers 200 before and after two deadline cuts: the engine is
    live and working — the condition is `call_slow`, not a dead engine."""
    stub = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")],
            probe_results=[_HEALTHY, _HEALTHY],
        ),
    )
    d = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert d.error is not None and "call_slow" in d.error, d.error
    # pre-flight probe + post-drop classification probe
    assert stub.probes == 2


def test_the_two_stubbed_conditions_produce_different_discriminators(monkeypatch):
    """The clause's decisive assertion: unresponsive vs slow answer with
    two DIFFERENT readable values, from otherwise identical drops."""
    dead = _install(monkeypatch, _Stub(probe_results=[_SILENT]))
    monkeypatch.setattr(obs_mod, "_post_chat_completion_with_tools", _no_post_at_all())
    d_dead = asyncio.run(
        obs_mod._call_observer(
            user_prompt="e", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    live = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")],
            probe_results=[_HEALTHY, _HEALTHY],
        ),
    )
    d_live = asyncio.run(
        obs_mod._call_observer(
            user_prompt="e", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert dead.posts == 0 and live.posts == 2
    assert d_dead.error and d_live.error
    assert "engine_unresponsive" in d_dead.error
    assert "call_slow" in d_live.error
    assert d_dead.error != d_live.error


def test_discriminator_is_readable_from_the_observations_table(tmp_path, monkeypatch):
    """The drop path's tagged error reaches the `error` COLUMN, unchanged.

    Drives the real `_call_observer` (both attempts timed out against the
    stub, post-drop probe live → `call_slow`) and persists its decision
    through `_persist` into a scratch db; the real usage.db is never
    opened. A tag that only lived on the in-memory ObserverDecision would
    pass every unit test above and still leave the production store as
    blind as it was — `scripts/iv_grade.py` reads rows, not objects.
    """
    db = tmp_path / "usage-test-458.db"
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    # tests/integration/test_iv_guards.py replaces this name on the module
    # with an in-memory recorder at import time (its own clause-6
    # protection), so restore the real writer here rather than depend on
    # which file pytest imported first.
    monkeypatch.setattr(
        obs_mod, "record_inner_voice_observation",
        usage_store.record_inner_voice_observation,
    )
    stub = _install(
        monkeypatch,
        _Stub(
            post_behaviours=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")],
            probe_results=[_HEALTHY, _HEALTHY],
        ),
    )
    decision = asyncio.run(
        obs_mod._call_observer(
            user_prompt="event", cfg=dict(CFG),
            timeout_override=ASYNC_TIMEOUT, async_call=True,
        )
    )
    assert stub.posts == 2 and "call_slow" in (decision.error or "")
    state = types.SimpleNamespace(
        sequence=0,
        session_id="test-458",
        turn_id="turn-1",
        decisions_this_turn=[],
        observer_model="stub-model",
        primary_model="stub-model",
    )
    asyncio.run(obs_mod._persist(state, decision, "tool_result", related_tool="Bash"))
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT action, error FROM inner_voice_observations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "noop"
    assert rows[0]["error"] == decision.error
    assert "call_slow" in rows[0]["error"]


# ---------------------------------------------------------------------------
# Clause 5 — unresponsive engine detected in ~1 s, not at the 12 s deadline
# ---------------------------------------------------------------------------


def test_probe_budget_default_is_1s():
    """The whole clause-5 bound rests on the probe budget: 1.0 s default
    (config key `probe_timeout_seconds`, unset in config.yaml, so the
    setdefault value is what production runs with)."""
    assert obs_mod.DEFAULT_PROBE_TIMEOUT_SECONDS == 1.0
    assert obs_mod._observer_cfg()["probe_timeout_seconds"] == 1.0


def test_unresponsive_engine_is_abandoned_within_2s_of_wall_clock(monkeypatch):
    """Clause 5, over real TCP with httpx's own read timeout doing the
    deciding: a socket that accepts connections and never replies.

    Deliberately NOT httpx.MockTransport — measured on httpx 0.28.1,
    MockTransport ignores the per-request `timeout` argument outright (a
    handler sleeping 5 s under a 1 s budget returned its response after
    5,005 ms), so a handler that raises ReadTimeout "itself" would prove
    the abandonment path but never the budget. A silent listening socket
    exercises the real pool: the handshake completes in the kernel
    backlog, nothing answers, and httpx's read timeout fires at ~1,000 ms.

    The 12 s judgment deadline (12,000 ms) must never be reached and no
    engine POST may ever be queued; `latency_ms` and wall clock are both
    capped at the clause's 2,000 ms.
    """
    silent = socket.socket()
    silent.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    silent.bind(("127.0.0.1", 0))  # ephemeral port, never listened past backlog
    silent.listen(1)               # connects succeed; no response ever follows
    port = silent.getsockname()[1]
    monkeypatch.setattr(
        obs_mod, "_resolve_endpoint",
        lambda alias=None: (f"http://127.0.0.1:{port}", "stub-model"),
    )
    monkeypatch.setattr(
        obs_mod, "_post_chat_completion_with_tools", _no_post_at_all(),
    )

    async def run():
        try:
            return await obs_mod._call_observer(
                user_prompt="event", cfg=dict(CFG),
                timeout_override=ASYNC_TIMEOUT, async_call=True,
            )
        finally:
            await obs_mod.aclose_clients()

    try:
        t0 = time.perf_counter()
        d = asyncio.run(run())
        wall_ms = (time.perf_counter() - t0) * 1000
    finally:
        silent.close()

    assert d.error is not None and "engine_unresponsive" in d.error, d.error
    # The probe's own budget must be what fired — read off the row, not a
    # coincidence of the 2 s cap.
    assert "no response in 1.0s" in d.error, d.error
    assert d.latency_ms <= 2000, (
        f"abandoned after {d.latency_ms} ms; the clause caps the drop at "
        "2,000 ms, not the 12,000 ms judgment deadline"
    )
    assert wall_ms <= 2000, f"wall clock {wall_ms:.0f} ms exceeds the 2,000 ms cap"
    assert d.action == "noop"
