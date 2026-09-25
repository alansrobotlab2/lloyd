"""The caption ratchet is corrected at call two, and cache hits are counted.

Two unrelated defects, both of which made a real number invisible.

**Captions.** Every advertised tool carries an injected `summary` the
transcript renders instead of a bare tool name. The failure mode is a ratchet
rather than a constant: `arguments` is replayed to the engine as history, so
the first call that omits the caption becomes the model's own most recent
example of calling that tool and the session locks into omitting it. Session
`20260907_184351_ivec8d` is the shape — 5/5 captioned on first use of each
tool, 0/31 after; sessions `20260907_235236_backlogs_a8fd` and
`20260908_000804_backlogi_3828` emitted 49 consecutive uncaptioned Bash calls.
Round SM_20260908_165950 ran 24/25 before its first miss and 15/56 after, so
most of a 34-minute round reads as a wall of `Bash`.

Correcting it at call two is cheap. At call thirty the wrong example has been
reinforced thirty times, which is why the nudge is one-shot and early rather
than repeated.

**Cache.** vLLM reports prefix-cache hits under a nested
`prompt_tokens_details.cached_tokens`, and `_merge_usage` copied only
top-level ints — so `cache_read` was absent from every usage block the harness
ever produced, and `messages.py` then recorded a hardcoded zero on top. Every
session on 2026-09-08 reported `cache_read: 0` on every iteration. That is not
a cold cache; it is nobody reading the number. It matters because the
position-0 rule exists to keep the prompt prefix cached across a long turn and
this is the only signal that says whether it holds.
"""
from __future__ import annotations

from app.harness import events
from app.harness.loop import _merge_usage, _with_caption_nudge


# ── the caption nudge ───────────────────────────────────────────────────────

def test_the_nudge_is_appended_to_the_result_content():
    evt = events.tool_result(call_id="c1", name="Bash", content="ok", is_error=False)
    nudged = _with_caption_nudge(evt)
    assert nudged["content"].startswith("ok")
    assert "`summary`" in nudged["content"]


def test_the_nudge_does_not_mutate_the_original_event():
    """The same dict reaches `fire_on_event` and history; an in-place edit
    would make the observer read the reminder as something the tool said."""
    evt = events.tool_result(call_id="c1", name="Bash", content="ok", is_error=False)
    _with_caption_nudge(evt)
    assert evt["content"] == "ok"


def test_the_nudge_preserves_the_error_flag():
    evt = events.tool_result(call_id="c1", name="Bash", content="boom", is_error=True)
    assert _with_caption_nudge(evt)["is_error"] is True


def test_the_nudge_asks_for_the_next_call_not_this_one():
    """A model cannot edit a call it already made; the point is to reset the
    example before the ratchet sets."""
    evt = events.tool_result(call_id="c1", name="Bash", content="", is_error=False)
    text = _with_caption_nudge(evt)["content"]
    assert "from here on" in text


def test_result_event_carries_the_caption_rate():
    evt = events.result(stop_reason="stop", tool_calls_total=56, tool_calls_captioned=15)
    assert evt["tool_calls_total"] == 56
    assert evt["tool_calls_captioned"] == 15


def test_result_event_defaults_to_zero_for_a_turn_with_no_tools():
    evt = events.result(stop_reason="stop")
    assert evt["tool_calls_total"] == 0 and evt["tool_calls_captioned"] == 0


# Driven through the real `run_query` on the replay seams
# (`app/harness/tests/_replay.py`); these used to be source-substring pins.

def _caption_pool():
    from app.harness.tests import _replay as R

    return R.ReplayPool({
        "Read": True, "Bash": False,
        # Declares a real, required `summary` of its own, so it is not in
        # `summary_tools` and its `summary` is a genuine argument.
        "session_inject_context": {
            "inputSchema": {"type": "object",
                            "properties": {"summary": {"type": "string"}},
                            "required": ["summary"]},
            "annotations": {"readOnlyHint": False}},
    })


async def _caption_turn(monkeypatch, steps, **opts):
    from app.harness.options import RunOptions
    from app.harness.tests import _replay as R

    pool = _caption_pool()
    R.install(monkeypatch, R.ReplayEngine(steps), pool)
    out = await R.drive(RunOptions(model="m", max_turns=8,
                                   tool_search_enabled=False, **opts))
    return out, pool


def _nudged(out):
    from app.harness.loop import _CAPTION_NUDGE
    from app.harness.tests import _replay as R

    return [e["call_id"] for e in R.of_type(out, "tool_result")
            if _CAPTION_NUDGE in e["content"]]


async def test_only_summary_tools_are_counted(monkeypatch):
    """`session_inject_context` has a real `summary` of its own and is not in
    `summary_tools`; counting it would report a caption nobody injected — and
    its `summary` must reach the tool as an argument, not be lifted off."""
    from app.harness.tests import _replay as R

    out, pool = await _caption_turn(monkeypatch, [
        R.Step(tool_calls=[
            R.tool_call("c1", "Read", "reading x", file_path="/x"),
            R.tool_call("c2", "session_inject_context", "a real argument"),
        ]),
        R.Step(text="done"),
    ])
    result = R.of_type(out, "result")[0]
    assert (result["tool_calls_total"], result["tool_calls_captioned"]) == (1, 1)
    by_id = {c["call_id"]: c for c in pool.calls}
    assert "summary" not in by_id["c1"]["args"]
    assert by_id["c1"]["summary"] == "reading x"
    assert by_id["c2"]["args"]["summary"] == "a real argument"
    assert _nudged(out) == []


async def test_the_nudge_fires_once_per_turn(monkeypatch):
    """One result carries it — the first miss. Thirty would be noise."""
    from app.harness.tests import _replay as R

    out, _pool = await _caption_turn(monkeypatch, [
        R.Step(tool_calls=[R.tool_call("c1", "Read", "captioned", file_path="/a")]),
        R.Step(tool_calls=[R.tool_call("c2", "Bash", command="ls")]),
        R.Step(tool_calls=[R.tool_call("c3", "Bash", command="pwd"),
                           R.tool_call("c4", "Read", file_path="/b")]),
        R.Step(text="done"),
    ])
    assert _nudged(out) == ["c2"]
    result = R.of_type(out, "result")[0]
    assert (result["tool_calls_total"], result["tool_calls_captioned"]) == (4, 1)


async def test_in_a_parallel_batch_the_nudge_lands_on_the_first_miss_in_wire_order(
        monkeypatch):
    """The first result back is arbitrary; the ratchet is about the first call
    the model MADE without a caption."""
    from app.harness.tests import _replay as R

    pool = _caption_pool()
    pool.delay_by_call_id = {"c1": 0.04, "c2": 0.03, "c3": 0.0}
    R.install(monkeypatch, R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "has one", file_path="/a"),
                           R.tool_call("c2", "Read", file_path="/b"),
                           R.tool_call("c3", "Read", file_path="/c")]),
        R.Step(text="done"),
    ]), pool)
    from app.harness.options import RunOptions

    out = await R.drive(RunOptions(model="m", max_turns=4, tool_search_enabled=False,
                                   parallel_tool_calls_enabled=True))
    assert pool.completed[0] == "c3", "the fixture did not reorder"
    assert _nudged(out) == ["c2"]
    result = R.of_type(out, "result")[0]
    assert (result["tool_calls_total"], result["tool_calls_captioned"]) == (3, 1)


# ── prefix-cache accounting ─────────────────────────────────────────────────

def test_nested_cached_tokens_are_read():
    usage = {"prompt_tokens": 1000, "completion_tokens": 20,
             "prompt_tokens_details": {"cached_tokens": 768}}
    assert _merge_usage({}, usage)["cache_read"] == 768


def test_nested_created_cache_tokens_are_read():
    usage = {"prompt_tokens_details": {"created_cache_tokens": 55}}
    assert _merge_usage({}, usage)["cache_create"] == 55


def test_openai_keys_are_still_normalised():
    out = _merge_usage({}, {"prompt_tokens": 7, "completion_tokens": 3})
    assert out["input_tokens"] == 7 and out["output_tokens"] == 3


def test_a_zero_cache_hit_is_reported_as_zero_not_dropped():
    """A cold prefix and an unread field must not look the same."""
    out = _merge_usage({}, {"prompt_tokens_details": {"cached_tokens": 0}})
    assert out["cache_read"] == 0 and "cache_read" in out


def test_absent_details_leave_the_key_alone():
    assert "cache_read" not in _merge_usage({}, {"prompt_tokens": 5})


def test_a_non_dict_details_field_is_ignored():
    assert "cache_read" not in _merge_usage({}, {"prompt_tokens_details": None})


def test_both_recorders_pass_the_measured_cache_numbers():
    """Hardcoded zeros made the usage table agree with the harness by accident."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "routers" / "messages.py").read_text(encoding="utf-8")
    assert "cache_read=0," not in src, "a record_usage call still hardcodes the cache"
    # One per recorder in the router: the streaming turn's. The sync
    # `POST /api/message` was the second and is deleted (P13.6); the third
    # writer is `app/run_recorder.py`.
    assert src.count("cache_read=cache_read_tokens") == 1
