"""Shared test harness for the intra-turn context-relief latch (#800).

Imported by `test_context_relief_hysteresis.py` and by the latch clauses of
`test_overflow_recovery.py`. Deliberately not a `test_*.py` file: pytest does not
collect it, so a broken harness is a collection error in both suites rather than a
mysterious red run.

The harness exists so both files share ONE set of token tiers. The whole defect is
about where the ladder's thresholds sit, and a second file deriving them its own way
is how a latch gets tested against a number the loop never uses.

What it drives is the REAL `run_query` on a REAL `ContextMeter`. Four things are
replaced, all inside the loop module's own namespace, so nothing else in the process
is disturbed and a loop that stopped using a seam fails the suite instead of quietly
making contact with the machine:

  - `app.harness.loop._build_pool` — the loop discovers tools before it streams and
    discovery opens a real MCP connection to 127.0.0.1. Replaced with a pool that
    advertises one plain tool. Same seam `test_context_meter.py` patches.
  - `app.harness.loop.stream_chat` — the HTTP boundary. Replaced with a generator
    yielding scripted deltas, and raising `ContextOverflowError` when asked to.
    It reports NO usage block, which is what keeps the meter's reading under the
    test's control: `ContextMeter.observe_usage(None, ...)` (`context_meter.py:167`)
    keeps the baseline, so the level then moves only with the history the turn
    actually grows by and only with what relief actually removes.
  - `app.harness.loop._relieve_context` — WRAPPED, not replaced. The wrapper records
    what the loop asked for and calls the real ladder underneath. The latch decision
    lives inside that function, so stubbing it out would test the stub.
  - `RunOptions.session_id` is empty: with it set, the clearing and truncation rungs
    write into a real per-session spill directory. Empty, both replace content in
    memory, so these tests leave nothing behind and never touch `/tmp`.

How the level is set, and why it is exact. `ContextMeter.used` is
`_reported + estimate(chat_messages[_reported_at_len:])` (`context_meter.py:115`).
Seeding therefore means: build the history the turn runs on, report a prompt size
equal to the desired level at `n_messages = len(history)` — which zeroes the tail
term — and the level IS that number. From then on the loop's own `observe_append`
in the per-iteration path raises it by the estimate of everything the iteration
appends, and the ladder's own `meter.resync(chat_messages)` call in each of its
four rungs lowers it by what that rung removed — no resync, no fall. No number in
these tests is a mock: the tier is one synthetic
engine report, and every movement after it is measured off the real message list.
"""
import asyncio
import json

import pytest

from app.compaction import truncation_threshold
from app.harness import loop as loop_mod
from app.harness import tool_search_cache
from app.harness.context_meter import ContextMeter, context_window_for
from app.harness.errors import ContextOverflowError
from app.harness.loop import run_query
from app.harness.options import RunOptions

# ── the tiers, from the same config the loop reads ─────────────────────────────
WINDOW = context_window_for("primary")            # 262,144
THRESHOLD = truncation_threshold(WINDOW)          # 210,144 (`app/compaction.py`)
TRIGGER = int(THRESHOLD * 0.8)                    # 168,115 — the re-arm level
TARGET = int(THRESHOLD * 0.6)                     # 126,086 — the relief target
WALL = 300_000                    # a prompt size the engine refuses outright

# Where a seeded turn sits. `BAND` is the interval the firings actually land in:
# over `target`, under `trigger`. Re-measured 2026-09-20 over `logs/server.err.1`
# (file spans 09:32:58 → 18:54:28; the 2,474 `intra_turn` firings run 10:04:29 →
# 18:51:02, 8.78 h, 282 an hour): 87.1% of them sat below the re-arm level those
# same lines imply (their printed `target 109,274` ⇒ threshold 182,123 ⇒ trigger
# 145,698 — the writing process resolved a smaller window than the 262,144 HEAD
# resolves for `primary`, which is why the tiers above and the log disagree on the
# absolute number but not on the band). The item's triage reading one day earlier
# was 1,966 firings / 240 an hour / 97.0% below trigger. Rung 1
# (tool-result clearing) cannot fire below `trigger`, so inside the band the pass is
# the reasoning prune — `reasoning:` is named in all 2,474 of those lines.
BAND = TARGET + 10_000            # 136,086
ABOVE_TRIGGER = TRIGGER + 40_000  # 208,115 — so high that no pass can reach target

# ── the history the turn runs on ───────────────────────────────────────────────
REASONING_CHARS = 4_000           # ~1,000 tokens of thinking per assistant turn
SEED_RESULTS = 18                 # > the 15-result `keep_recent_tools` floor
SEED_RESULT_CHARS = 12_000        # > the 2,000-char `min_chars_to_clear` floor
# What one iteration appends: the fake pool's answer. 60,000 chars is ~15,000
# tokens, so a turn climbs the ~32,000 tokens from `BAND` to `TRIGGER` in two
# iterations — the interval is short enough that a test can watch the re-arm
# happen, and the number is ordinary tool output rather than a contrivance.
GROWTH_RESULT_CHARS = 8_000


class _FakePool:
    """One advertised tool, shaped as `test_context_meter.py` shapes it: `run_query`
    refuses a toolless turn (`ToolDiscoveryError`) and builds the advertised catalog
    out of `pool.discovered`. One is plenty — `tool_search_enabled` is off in the
    shared options, so the catalog goes out whole either way.

    `call_tool` returns `GROWTH_RESULT_CHARS` of text. That IS the turn's growth:
    the loop appends it as a tool message and the meter measures it.
    """

    @property
    def discovered(self):
        return [("fake", [{"name": "Bash", "description": "Run a command.",
                           "inputSchema": {"type": "object",
                                           "properties": {}}}])]

    def __init__(self, growth_chars: int = GROWTH_RESULT_CHARS):
        self.growth_chars = growth_chars

    async def call_tool(self, name, args, *, session_id: str = "", **_kw):
        return {"content": "x" * self.growth_chars, "is_error": False}


class _Script:
    """Scripted `stream_chat`: one canned iteration per call.

    Each turn is `(text, tool_calls)`. No usage block is emitted — see the module
    docstring for why that is the design rather than an omission. The last turn
    repeats if the loop asks for more iterations, so a `max_turns` guard never
    changes a tier.

    `overflow_times=N` rejects N requests the way the client does at the wall
    (`ContextOverflowError`, carrying the size the engine says it was asked for), and
    `overflow_at=K` says WHICH request — 1-based. Rejecting the FIRST is unrealistic
    and useless: the meter has never been measured, so the recovery has no anchor to
    aim at. Real rounds succeed for a while and then cross the wall, so the default
    is request 2.
    """

    def __init__(self, turns, *, overflow_times: int = 0, overflow_at: int = 2,
                 requested: int = WALL):
        self.turns = list(turns)
        self.overflow_times = overflow_times
        self.overflow_at = overflow_at
        self.requested = requested
        self.overflow_count = 0
        self.calls = 0
        self.last_messages = None

    def __call__(self, **kwargs):
        idx = min(self.calls, len(self.turns) - 1)
        self.calls += 1
        self.last_messages = [dict(m) for m in (kwargs.get("messages") or [])]
        return self._gen(*self.turns[idx])

    async def _gen(self, text, tool_calls):
        if (self.calls >= self.overflow_at
                and self.overflow_count < self.overflow_times):
            self.overflow_count += 1
            raise ContextOverflowError(
                "prompt too long", requested_input_tokens=self.requested)
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc.get("arguments") or {})},
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def TC(n: int) -> dict:
    """A tool call with a fresh id. Ids must be unique or the loop reuses a message
    slot instead of appending, silently changing the size of the turn."""
    return {"id": f"c{n}", "name": "Bash", "arguments": {"command": "ls"}}


def seeded_history(results: int = SEED_RESULTS, *,
                   result_chars: int = SEED_RESULT_CHARS,
                   reasoning: int = REASONING_CHARS) -> list[dict]:
    """A history carrying `results` large COMPLETED tool results, each followed by
    another call, each assistant turn carrying `reasoning` chars of preserved
    thinking.

    Tool messages count in the ladder's pre-check only once the assistant message
    advertising them ALSO has tool calls — a result with no follow-up action is
    history, not activity, and the loop says so. `reasoning` is what rung 2 takes
    back, and `results` beyond `keep_recent_tools` is what rung 1 takes back, so the
    two sizes are what decide how much one pass can buy.
    """
    msgs = [{"role": "user", "content": "gather a lot of context"}]
    body = "x" * result_chars
    for i in range(results):
        msgs.append({"role": "assistant", "content": "",
                     "reasoning": "r" * reasoning,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                         "name": "Bash",
                         "arguments": json.dumps({"command": "find / -name x"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": body})
    msgs.append({"role": "assistant", "content": "",
                 "tool_calls": [{"id": "cZ", "type": "function", "function": {
                     "name": "Bash", "arguments": json.dumps({"command": "ls"})}}]})
    return msgs


def meter_at(level: int, history: list[dict],
             window: int = WINDOW) -> ContextMeter:
    """A REAL meter reading exactly `level`, anchored on a synthetic engine report.

    `level` is what the engine is said to have reported when the conversation was
    `history` long. The tail term of `used` is then `estimate([])` = 0, so the
    reading is `level` with no rounding and no subclass — the assertion below is the
    harness checking its own arithmetic, and a change to `ContextMeter`'s formula
    fails here rather than silently re-scaling every tier test.
    """
    m = ContextMeter(window)
    m.observe_usage({"input_tokens": int(level)}, len(history))
    m.observe_append(history)
    assert m.measured and m.used == int(level), (m.used, m.measured, level)
    return m


class _Run:
    """What one driven turn left behind.

    `intra_passes` / `intra_refusals` distinguish "the loop asked the ladder and it
    ran" from "the loop asked and was refused". That distinction is the entire subject
    of the latch tests, so it comes from the ladder's own report (`latched`) rather
    than from an inference about log lines.
    """

    def __init__(self, script, meter, calls, events, raised: BaseException | None = None):
        self.script = script
        self.meter = meter
        self.calls = calls
        self.events = events
        # The exception the turn died on, if it did. `run_query` re-raises after the
        # context-overflow recovery ceiling, so "the turn ended at the wall" is an
        # exception rather than a stop reason, and the ladder calls recorded up to
        # that point are the other half of the evidence.
        self.raised = raised

    def ladder_calls(self) -> list[dict]:
        return list(self.calls)

    def intra_calls(self) -> list[dict]:
        return [c for c in self.calls if c["reason"] == "intra_turn"]

    def intra_passes(self) -> list[dict]:
        return [c for c in self.intra_calls() if not c["latched"]]

    def intra_refusals(self) -> list[dict]:
        return [c for c in self.intra_calls() if c["latched"]]

    def by_reason(self, reason: str) -> list[dict]:
        return [c for c in self.calls if c["reason"] == reason]

    def stop_reason(self) -> str:
        """The `result` event's `stop_reason`, or a marker saying there was none.

        `events.result(...)` yields a FLAT dict (`{"type": "result",
        "stop_reason": ..., "usage": ...}`), not an envelope with a `data` key, so
        the lookup reads the top level and falls back to `data` for a StreamEvent.
        Getting this wrong returns a "?" that reads like "the loop reported nothing"
        when the loop in fact named the stop reason.
        """
        for e in self.events:
            if not isinstance(e, dict):
                e = {"type": getattr(e, "type", None), "data": getattr(e, "data", None)}
            if e.get("type") != "result":
                continue
            data = e.get("data") or e
            return data.get("stop_reason", "<result event with no stop_reason>")
        return "<no result event>"


@pytest.fixture
def _clean_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def relief_harness(monkeypatch):
    """Return `go(turns, **kwargs) -> _Run`, driving the real `run_query`.

    `level` is where the turn's prompt starts; `history` is the message list it runs
    on; `overflow_times`/`overflow_at` which requests the engine rejects. Any other
    keyword goes to `RunOptions`.
    """

    def go(turns, *, level: int = BAND, history: list[dict] | None = None,
           growth_chars: int = GROWTH_RESULT_CHARS,
           spy: bool = True, overflow_times: int = 0, overflow_at: int = 2,
           max_turns: int = 8, inject_once: bool = False, **opt_kw) -> _Run:
        msgs = history if history is not None else seeded_history()
        meter = meter_at(level, msgs)
        script = _Script([(t[0], t[1]) for t in turns],
                         overflow_times=overflow_times, overflow_at=overflow_at)
        pool = _FakePool(growth_chars=growth_chars)
        if inject_once:
            # The observer's `inject` lever, which is the only way to reach the
            # terminal-inject branch (`run_query` continues only when a hook
            # appended a message). Real `HookRegistry`, real callback, appending
            # to the same list the loop is reading — the shape the voice observer
            # uses (`app/harness/tests/test_loop_inject_ordering.py`).
            opt_kw["hooks"] = _inject_once_hook(msgs)

        monkeypatch.setattr("app.harness.loop._build_pool",
                            lambda _options, *_a, **_kw: _resolved(pool))
        monkeypatch.setattr("app.harness.loop.stream_chat", script)

        calls: list[dict] = []
        if spy:
            _real_ladder = loop_mod._relieve_context

            def _ladder(messages, **kwargs):
                before = kwargs["meter"].used
                report = _real_ladder(messages, **kwargs)
                calls.append({
                    "reason": kwargs.get("reason"),
                    "used": before,
                    "after": kwargs["meter"].used,
                    "iteration": kwargs.get("iteration"),
                    "rungs": list(report.get("rungs") or []),
                    "latched": bool(report.get("latched")),
                    "freed_tokens": report.get("freed_tokens", 0),
                    "passes": report.get("passes"),
                    "rearm": report.get("rearm"),
                    # None means the caller handed in no latch, i.e. the call is not
                    # gated by the per-turn pass count. That is what the pre-request,
                    # terminal-inject and overflow rungs must be. The absence of a
                    # latch is the whole boundary: `_relieve_context` takes no other
                    # bound on how far down the ladder it may go.
                    "latch": kwargs.get("latch"),
                    "target": kwargs.get("target"),
                })
                return report

            monkeypatch.setattr(loop_mod, "_relieve_context", _ladder)

        opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                          preserve_thinking_iterations=6, max_turns=max_turns,
                          chat_messages_handle=msgs, context_meter=meter, **opt_kw)

        async def _drain():
            # The seeded history already carries the user turn; passing it again as
            # the `messages` argument would append a second copy and move the level
            # the whole harness is anchored to.
            return [e async for e in run_query([], opts)]

        raised: BaseException | None = None
        try:
            events = asyncio.run(_drain())
        except BaseException as exc:       # noqa: BLE001 — re-surfaced to the test
            # `run_query` re-raises once the context-overflow recovery ceiling is
            # spent, so a turn that ends at the wall ends by raising. The tests that
            # care about that outcome need the ladder calls recorded up to the wall
            # AND the exception, which is why the driver keeps both instead of
            # letting it propagate into an errored test.
            raised = exc
            events = []
        return _Run(script, meter, calls, events, raised=raised)

    return go


def _inject_once_hook(chat_messages: list, text: str = "[INNER VOICE] Do it now."):
    """An OnEvent hook that injects on the first terminal assistant_message.

    Used by the terminal-inject clause of #800: that branch is reachable only when a
    hook appended a message to `chat_messages` on an assistant turn with no tool
    calls, so the driver needs a real registry rather than a stub — the branch reads
    `len(chat_messages)` before and after `fire_on_event`.
    """
    from app.harness.hooks import HookRegistry

    state = {"fired": False}

    async def cb(evt: dict) -> None:
        if evt.get("type") != "assistant_message" or evt.get("tool_calls"):
            return
        if state["fired"]:
            return
        state["fired"] = True
        chat_messages.append({"role": "user", "content": text})

    hooks = HookRegistry()
    hooks.add_on_event(cb)
    return hooks


async def _resolved(pool):
    """`_build_pool` is awaited by the loop, so the stub returns a coroutine."""
    return pool

# Where the loop's own reading lands per iteration. `estimate_tokens` is ~4
# chars/token plus per-message overhead, so a tool answer of `growth_chars` moves
# `meter.used` by roughly `growth_chars // 4`. The tests take the growth as a
# parameter because the two things they must show need different speeds: a turn
# that stays inside the band for several iterations (a small climb, like the real
# log's ~2,600 tokens per iteration), and a turn that crosses the release level in
# two (a large climb).
SMALL_GROWTH_CHARS = 8_000          # ~2,069 tokens per iteration
FAST_GROWTH_CHARS = 60_000          # ~15,069 tokens per iteration
