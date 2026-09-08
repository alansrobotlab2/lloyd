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


def test_the_loop_counts_only_tools_that_were_asked_for_a_caption():
    """`session_inject_context` has a real `summary` of its own and is not in
    `summary_tools`; counting it would report a caption nobody injected."""
    import inspect

    from app.harness import loop

    src = inspect.getsource(loop.run_query)
    assert 'tc["function"]["name"] in summary_tools' in src


def test_the_nudge_fires_at_most_once_per_turn():
    """One result carries it. Thirty would be its own kind of noise."""
    import inspect

    from app.harness import loop

    src = inspect.getsource(loop.run_query)
    assert "elif not caption_nudged:" in src
    assert "caption_nudged = True" in src


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
    assert src.count("cache_read=cache_read_tokens") == 2
