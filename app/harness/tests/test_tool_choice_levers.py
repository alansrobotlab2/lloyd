"""P6: `tool_choice` as a cache-safe lever (review 2026-09-24).

Two uses, both driving the REAL `run_query` through `_replay`:

(a) The echo guard in `tool_choice` mode discards an attempt that printed a
    shell fence without calling a tool and re-sends the identical request with
    `tool_choice: "required"`, instead of appending a nudge. `nudge` stays the
    shipped mode.
(b) At `max_turns`, one more request with `tool_choice: "none"` asks where the
    work stands. It is decided per engine (`max_turns_wrapup_base_urls`),
    never dispatches a tool, and leaves `stop_reason` at `max_turns` so the
    finalizer still skips.
"""
from __future__ import annotations

import asyncio

from app.harness.hooks import HookRegistry
from app.harness.options import RunOptions
from app.harness.tests._replay import (
    ReplayEngine, ReplayPool, Step, drive, install, of_type, tool_call,
)

PRIMARY = "http://127.0.0.1:8096"
SECONDARY = "http://127.0.0.1:8091"
FENCE = "Run this:\n```bash\nls -la\n```\n"


def _opts(**kw) -> RunOptions:
    base = dict(model="primary", base_url=PRIMARY, session_id="p6-test",
                tool_call_summaries=False)
    base.update(kw)
    return RunOptions(**base)


def _run(monkeypatch, steps, **kw):
    engine = ReplayEngine(steps)
    install(monkeypatch, engine, ReplayPool())
    evts = asyncio.run(drive(_opts(**kw)))
    return engine, evts


# ── (a) echo guard ─────────────────────────────────────────────────────────

def test_echo_guard_reissues_the_identical_request_with_required(monkeypatch):
    engine, evts = _run(monkeypatch, [
        Step(text=FENCE, reasoning="think"),
        Step(tool_calls=[tool_call("c1", "Bash", command="ls -la")]),
        Step(text="done"),
    ], echo_guard_mode="tool_choice")

    first, second = engine.requests[0], engine.requests[1]
    # Byte-identical but for tool_choice: the discarded attempt is gone.
    assert second.messages == first.messages
    assert second.tools == first.tools
    assert first.kwargs["tool_choice"] == "auto"
    assert second.kwargs["tool_choice"] == "required"
    # Reset after the request that used it.
    assert engine.requests[2].kwargs["tool_choice"] == "auto"

    retries = of_type(evts, "iteration_retry")
    assert len(retries) == 1
    r = retries[0]
    assert r["reason"] == "echo_guard" and r["attempt"] == 1
    assert r["discarded_text_chars"] == len(FENCE)
    assert r["discarded_thinking_chars"] == len("think")

    result = of_type(evts, "result")[0]
    assert FENCE not in result["response_text"]
    assert result["response_text"] == "done"
    # The reissue does not spend an iteration.
    assert result["num_turns"] == 2


def test_nudge_mode_appends_the_nudge_as_before(monkeypatch):
    engine, evts = _run(monkeypatch, [
        Step(text=FENCE),
        Step(text="only showing it"),
    ])  # default mode is nudge
    assert all(r.kwargs["tool_choice"] == "auto" for r in engine.requests)
    second = engine.requests[1].messages
    assert second[-2]["role"] == "assistant" and FENCE in second[-2]["content"]
    assert second[-1]["role"] == "user"
    assert "did not actually call a tool" in second[-1]["content"]
    assert not of_type(evts, "iteration_retry")


def test_no_reissue_when_the_observer_injected(monkeypatch):
    hooks = HookRegistry()
    chat: list[dict] = []
    fired = {"n": 0}

    async def _inject(evt):
        if evt.get("type") == "assistant_message" and fired["n"] == 0:
            fired["n"] += 1
            chat.append({"role": "user", "content": "[INNER VOICE] keep going"})

    hooks.add_on_event(_inject)
    engine = ReplayEngine([Step(text=FENCE), Step(text="ok")])
    install(monkeypatch, engine, ReplayPool())
    evts = asyncio.run(drive(_opts(echo_guard_mode="tool_choice", hooks=hooks,
                                   chat_messages_handle=chat)))
    assert not of_type(evts, "iteration_retry")
    assert all(r.kwargs["tool_choice"] == "auto" for r in engine.requests)
    # The inject is what the second request answers, after the kept attempt.
    assert engine.requests[1].messages[-1]["content"] == "[INNER VOICE] keep going"
    assert FENCE in engine.requests[1].messages[-2]["content"]


# ── (b) max-turns wrap-up ──────────────────────────────────────────────────

_LOOPING = [
    Step(tool_calls=[tool_call("c1", "Read", path="/a")]),
    Step(tool_calls=[tool_call("c2", "Read", path="/b")]),
]


def _wrap(**kw):
    return dict(max_turns=2, max_turns_wrapup=True,
                max_turns_wrapup_base_urls=(PRIMARY,), **kw)


def test_one_toolless_wrap_up_at_the_budget(monkeypatch):
    engine, evts = _run(monkeypatch, _LOOPING + [
        Step(text="Read /a and /b; the fix is not written yet."),
    ], **_wrap())
    assert len(engine.requests) == 3
    last = engine.requests[-1]
    assert last.kwargs["tool_choice"] == "none"
    # Same tools array: the wrap-up is a cache hit plus one user message.
    assert last.tools == engine.requests[0].tools
    assert last.messages[-1]["role"] == "user"
    assert "used all 2 iterations" in last.messages[-1]["content"]
    assert [r.kwargs["tool_choice"] for r in engine.requests[:2]] == ["auto", "auto"]

    result = of_type(evts, "result")[0]
    assert result["wrapped_up"] is True
    assert result["stop_reason"] == "max_turns"
    assert result["response_text"].endswith("the fix is not written yet.")


def test_wrap_up_tool_calls_are_never_dispatched(monkeypatch):
    engine = ReplayEngine(_LOOPING + [
        Step(text="summary", tool_calls=[tool_call("c9", "Bash", command="rm x")]),
    ])
    pool = ReplayPool()
    install(monkeypatch, engine, pool)
    evts = asyncio.run(drive(_opts(**_wrap())))
    assert "c9" not in pool.started
    assert [c["name"] for c in pool.calls] == ["Read", "Read"]
    assert all(e["call_id"] != "c9" for e in of_type(evts, "tool_call"))
    wrap_msg = of_type(evts, "assistant_message")[-1]
    assert wrap_msg["tool_calls"] == [] and wrap_msg["text"] == "summary"
    assert of_type(evts, "result")[0]["wrapped_up"] is True


def test_stop_reason_stays_max_turns_and_the_finalizer_skips(monkeypatch):
    engine, evts = _run(monkeypatch, _LOOPING + [Step(text="where it stands")],
                        **_wrap(final_schema={"type": "object"}))
    result = of_type(evts, "result")[0]
    assert result["stop_reason"] == "max_turns"
    assert result["structured"] is None
    assert result["structured_error"]  # the deliberate skip names itself
    # No finalizer request went out after the wrap-up.
    assert len(engine.requests) == 3


def test_off_by_config(monkeypatch):
    engine, evts = _run(monkeypatch, _LOOPING + [Step(text="never asked")],
                        max_turns=2)
    assert len(engine.requests) == 2
    assert all(r.kwargs["tool_choice"] == "auto" for r in engine.requests)
    result = of_type(evts, "result")[0]
    assert result["wrapped_up"] is False
    assert result["stop_reason"] == "max_turns"


def test_an_unverified_engine_never_wraps_up(monkeypatch):
    """The llama.cpp secondary is not verified to honour `none`: a run on a
    base_url outside the list ends at the budget exactly as before."""
    engine, evts = _run(monkeypatch, _LOOPING + [Step(text="never asked")],
                        **_wrap(), base_url=SECONDARY)
    assert len(engine.requests) == 2
    assert of_type(evts, "result")[0]["wrapped_up"] is False


def test_config_resolves_wrap_up_to_the_listed_slots(monkeypatch):
    from app import mcp_discovery
    cfg = {
        "harness": {"max_turns_wrapup": {"enabled": True, "models": ["primary"]},
                    "echo_guard": {"mode": "tool_choice"}},
        "models": {"primary": {"base_url": PRIMARY + "/"},
                   "secondary": {"base_url": SECONDARY}},
    }
    monkeypatch.setattr(mcp_discovery, "CONFIG", cfg)
    kw = mcp_discovery.max_turns_wrapup_kwargs()
    assert kw == {"max_turns_wrapup": True,
                  "max_turns_wrapup_base_urls": (PRIMARY,)}
    assert mcp_discovery._get_harness_kwargs()["echo_guard_mode"] == "tool_choice"
    cfg["harness"]["max_turns_wrapup"]["enabled"] = False
    assert mcp_discovery.max_turns_wrapup_kwargs() == {}


def test_shipped_config_is_nudge_and_wrap_up_on_the_primary_only():
    """What config.yaml ships: echo guard `nudge`, wrap-up on `primary` only
    (the llama.cpp secondary is not verified to honour `none`)."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config.yaml").read_text())
    h = cfg["harness"]
    assert h["echo_guard"]["mode"] == "nudge"
    assert h["max_turns_wrapup"] == {"enabled": True, "models": ["primary"]}
