"""Tests for app/compaction.py — client-side conversation compaction.

Run: .venvs/lloyd/bin/python -m tests.test_compaction
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.compaction import (  # noqa: E402
    DEFAULT_CONTEXT_WINDOW,
    OUTPUT_TOKENS_RESERVED,
    TOKENS_PER_CHAR,
    TRUNCATION_BUFFER_TOKENS,
    TURNS_TO_KEEP,
    estimate_conversation_tokens,
    estimate_tokens,
    load_and_compact_session,
    truncate_conversation,
    truncation_threshold,
)


def _run(coro):
    """Sync runner for the now-async load_and_compact_session."""
    return asyncio.get_event_loop().run_until_complete(coro) if False \
        else asyncio.run(coro)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def test_estimate_tokens_returns_int():
    out = estimate_tokens("hello world, this is a test")
    assert isinstance(out, int), f"expected int, got {type(out).__name__}"
    assert out > 0


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore[arg-type]


def test_estimate_tokens_proportional():
    short = estimate_tokens("a" * 100)
    long = estimate_tokens("a" * 1000)
    assert long > short
    assert long == 1000 // TOKENS_PER_CHAR


def test_estimate_conversation_no_double_count():
    # System prompt should be counted once — not duplicated with hidden
    # hardcoded 20k padding like the old version.
    sys_prompt = "s" * 400  # 100 tokens
    msgs = [{"role": "user", "content": "u" * 400}]  # 100 tokens
    got = estimate_conversation_tokens(msgs, sys_prompt)
    assert got == 200, f"expected 200, got {got} (double-count regression?)"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def _mk(role: str, size: int) -> dict:
    """Build a message with `size` chars of content → size/4 tokens."""
    return {"role": role, "content": "x" * size}


def test_truncate_under_threshold_is_noop():
    msgs = [_mk("user", 400), _mk("assistant", 400)]  # 200 tokens
    out, dropped = truncate_conversation(msgs, max_tokens=1_000_000)
    assert out == msgs
    assert dropped == 0


def test_truncate_drops_old_turns_above_threshold():
    # 10 turns of 4000 chars each = 10 turns * (1000 user + 1000 asst) = 20000 tokens
    msgs: list[dict] = []
    for i in range(10):
        msgs.append(_mk("user", 4000))
        msgs.append(_mk("assistant", 4000))

    out, dropped = truncate_conversation(msgs, max_tokens=5_000, turns_to_keep=2)

    assert dropped > 0, "expected truncation to drop tokens"
    # Synthetic omission note + kept turns should total < input
    assert len(out) < len(msgs)
    # First message should be the synthetic note
    first_content = out[0].get("content", "")
    if isinstance(first_content, list):
        first_text = first_content[0].get("text", "")
    else:
        first_text = first_content
    assert "compaction" in first_text.lower(), f"expected synthetic note, got: {first_text[:80]}"


def test_truncate_keeps_last_turn_minimum():
    # One gigantic turn larger than threshold — must still keep it (never
    # return empty).
    msgs = [_mk("user", 100_000), _mk("assistant", 100_000)]
    out, _ = truncate_conversation(msgs, max_tokens=100)
    # Should contain the final turn (2 msgs) plus possibly a synthetic note.
    assert len(out) >= 2
    assert out[-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# load_and_compact_session (session-level helper)
# ---------------------------------------------------------------------------


def _write_session(path: Path, messages: list[dict]) -> None:
    path.write_text(json.dumps({"messages": messages}))


def test_load_and_compact_missing_file():
    out = _run(load_and_compact_session(Path("/nonexistent/path.json"), model="qwen"))
    assert out["history"] == []
    assert out["tokens_before"] == 0
    assert out["truncated"] is False


def test_load_and_compact_no_truncation_needed():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        out = _run(load_and_compact_session(p, model="qwen"))
        assert len(out["history"]) == 2
        assert out["truncated"] is False
        assert out["tokens_before"] == out["tokens_after"]


def test_load_and_compact_triggers_truncation():
    # 50 turns of 4000 chars each = 50 * 2000 tokens = 100_000 tokens
    # With DEFAULT_CONTEXT_WINDOW=128_000 and threshold ~= 128k - 20k - 32k = 76k,
    # this should trigger truncation.
    messages: list[dict] = []
    for _ in range(50):
        messages.append(_mk("user", 4000))
        messages.append(_mk("assistant", 4000))

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, messages)
        # Force truncate mode so the LLM summarization path doesn't fire
        # in unit tests (it would try to hit a real vLLM endpoint).
        out = _run(load_and_compact_session(
            p, model="qwen-unknown", mode_override="truncate",
        ))
        assert out["truncated"] is True
        assert out["tokens_after"] < out["tokens_before"]
        # Synthetic compaction note should be the first history entry
        first = out["history"][0]
        text = ""
        if isinstance(first.get("content"), list):
            text = first["content"][0].get("text", "")
        elif isinstance(first.get("content"), str):
            text = first["content"]
        assert "compaction" in text.lower()


def test_load_and_compact_accepts_str_path():
    # Type hint is `Path | str` — str should also work.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, [{"role": "user", "content": "hi"}])
        out = _run(load_and_compact_session(str(p), model="qwen"))
        assert len(out["history"]) == 1


def test_load_and_compact_filters_ui_only_roles():
    # Subliminal entries and other UI-only roles should be stripped —
    # they're not sent to the model.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, [
            {"role": "user", "content": "real user msg"},
            {"role": "subliminal", "content": "injected context"},
            {"role": "assistant", "content": "real reply"},
        ])
        out = _run(load_and_compact_session(p, model="qwen"))
        roles = [m["role"] for m in out["history"]]
        assert "subliminal" not in roles
        assert "user" in roles
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# Microcompaction pre-pass
# ---------------------------------------------------------------------------


def _tool_pairs(n: int, *, chars: int = 3_000, prefix: str = "f") -> list[dict]:
    """n assistant/tool pairs of Read calls with `chars`-sized results."""
    msgs: list[dict] = []
    for i in range(n):
        cid = f"call_{i:03d}"
        msgs.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"file_path": f"/{prefix}{i}.py"}),
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": f"contents of file {i}\n" * (chars // 20),
        })
    return msgs


def _marker_text(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        return c[0].get("text", "")
    return ""


def test_microcompact_does_nothing_without_context_pressure():
    """The defect that motivated the budget gate.

    Session 20260905_024955_iv5f05 peaked at 106,802 tokens against a
    210,144 threshold — 40% of budget — and the pre-pass still cleared 93
    of 97 tool results, because the trigger counted tools and never looked
    at tokens. The turn hit max_turns with no output; the next turn began
    with its evidence erased.
    """
    msgs: list[dict] = [{"role": "user", "content": "read these files"}]
    msgs += _tool_pairs(30)
    msgs.append({"role": "user", "content": "now what?"})

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, msgs)
        out = _run(load_and_compact_session(p, model="qwen", mode_override="truncate"))
        assert out["microcompacted"] == 0, (
            f"cleared {out['microcompacted']} results on a conversation using "
            f"{out['tokens_before']} of {out['threshold']} tokens"
        )
        assert out["tokens_after"] == out["tokens_before"]


def test_microcompact_clears_only_down_to_target_when_over_budget():
    """Above the trigger it clears oldest-first and stops at the target —
    not everything but the last N."""
    from app.harness.microcompact import microcompact

    msgs = _tool_pairs(40, chars=4_000)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    before = est(msgs)
    target = int(before * 0.6)

    out, cleared = microcompact(
        msgs, token_budget=target, estimate_fn=est,
        keep_recent_tools=15, legacy_count_rule=False,
    )
    after = est(out)
    assert cleared > 0, "should have cleared something"
    assert after <= target, f"still over target: {after} > {target}"
    # Proportional, not scorched-earth: the old rule would have cleared
    # all 25 candidates outright.
    assert cleared < 25, f"cleared {cleared}; expected the minimum needed"
    # The recent window is untouched.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    for m in tool_msgs[-15:]:
        assert "cleared from context" not in _marker_text(m)


def test_microcompact_never_clears_below_the_recent_floor():
    from app.harness.microcompact import microcompact

    msgs = _tool_pairs(18, chars=4_000)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    # An unreachable budget: it must still refuse to touch the last 15.
    out, cleared = microcompact(
        msgs, token_budget=1, estimate_fn=est,
        keep_recent_tools=15, legacy_count_rule=False,
    )
    assert cleared == 3, (
        f"expected exactly the 3 candidates outside the floor, cleared {cleared}"
    )


def test_microcompact_skips_results_too_small_to_be_worth_clearing():
    """Clearing a 200-byte result saves less than the marker costs."""
    from app.harness.microcompact import microcompact

    msgs = _tool_pairs(30, chars=200)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    out, cleared = microcompact(
        msgs, token_budget=1, estimate_fn=est,
        keep_recent_tools=5, min_chars_to_clear=2_000, legacy_count_rule=False,
    )
    assert cleared == 0, f"cleared {cleared} sub-threshold results"


def test_microcompact_marker_names_the_call_and_the_spill_file():
    """"[content cleared — retrieve via Read if needed]" is an instruction
    the model cannot follow: it names neither the file nor the call."""
    from app.harness.microcompact import microcompact
    from app.paths import SESSIONS_DIR

    sid = "test_microcompact_marker"
    msgs = _tool_pairs(20, chars=4_000)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    spill_dir = SESSIONS_DIR / f"{sid}.tool-results"
    try:
        out, cleared = microcompact(
            msgs, token_budget=1, estimate_fn=est,
            keep_recent_tools=5, session_id=sid, legacy_count_rule=False,
        )
        assert cleared > 0
        marker = _marker_text([m for m in out if m.get("role") == "tool"][0])
        assert "Read" in marker, marker
        assert "file_path=" in marker, marker
        assert str(spill_dir) in marker, marker
        # And the content is genuinely on disk, not merely named.
        assert list(spill_dir.glob("*.txt")), "marker names a file that was never written"
    finally:
        if spill_dir.exists():
            for f in spill_dir.iterdir():
                f.unlink()
            spill_dir.rmdir()


def test_microcompact_refuses_to_clear_what_it_could_not_persist():
    """Staying over budget is recoverable; deleting evidence is not."""
    from app.harness import microcompact as mc_mod

    msgs = _tool_pairs(20, chars=4_000)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    original = mc_mod.persist_for_compaction
    mc_mod.persist_for_compaction = lambda *a, **k: None  # simulate disk failure
    try:
        out, cleared = mc_mod.microcompact(
            msgs, token_budget=1, estimate_fn=est,
            keep_recent_tools=5, session_id="whatever", legacy_count_rule=False,
        )
        assert cleared == 0, f"cleared {cleared} results it failed to persist"
        assert out == msgs or all(
            "cleared from context" not in _marker_text(m)
            for m in out if m.get("role") == "tool"
        )
    finally:
        mc_mod.persist_for_compaction = original


def test_microcompact_preserves_the_spill_path_it_promises_to_keep():
    """The module docstring said twice that a spilled result keeps its
    file path. `_replace_tool_content` overwrote the whole block with a
    path-free marker, so the one genuinely lossless case was lossy."""
    from app.harness.microcompact import microcompact
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG

    spilled = (
        f"{PERSISTED_OUTPUT_TAG}\n"
        "Output too large (54.7 KB, 56,034 chars). "
        "Full output saved to: /sessions/s.tool-results/abc.txt\n\n"
        "Preview (first 2.0 KB):\n" + ("x" * 2000) + "\n</persisted-output>"
    )
    msgs = _tool_pairs(20, chars=100)
    msgs[1]["content"] = spilled

    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    out, cleared = microcompact(
        msgs, token_budget=None, estimate_fn=est,
        keep_recent_tools=5, legacy_count_rule=False,
    )
    first = _marker_text([m for m in out if m.get("role") == "tool"][0])
    assert cleared == 1, f"spill-aware pass should clear exactly this one, got {cleared}"
    assert "/sessions/s.tool-results/abc.txt" in first, first
    assert "xxxx" not in first, "preview should be dropped"


def test_microcompact_legacy_count_rule_still_available():
    """Direct callers that pass no budget keep the old behavior."""
    from app.harness.microcompact import microcompact

    msgs = _tool_pairs(30, chars=3_000)
    out, cleared = microcompact(
        msgs, keep_recent_tools=5, count_threshold=20, legacy_count_rule=True,
    )
    assert cleared == 25, f"expected legacy 30-5=25, got {cleared}"


# ---------------------------------------------------------------------------
# Threshold math
# ---------------------------------------------------------------------------


def test_truncation_threshold_math():
    window = 200_000
    t = truncation_threshold(window)
    assert t == window - OUTPUT_TOKENS_RESERVED - TRUNCATION_BUFFER_TOKENS
    assert t > 0
    # Tiny window → floor at 1000
    assert truncation_threshold(100) == 1_000


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mid-turn microcompaction (app/harness/loop.py)
# ---------------------------------------------------------------------------


def test_intra_turn_microcompact_is_silent_without_pressure():
    """The call site that did the most damage, and the one easiest to miss.

    `loop.py` clears tool results *during* a turn. Until 2026-09-05 its
    only trigger was `tool_count >= 15`, so turn 20260905_024955_iv5f05 —
    70 tool calls at a peak of 106,802 tokens against a 210,144 threshold
    — was held to 5 inline results for its whole length.
    """
    from app.harness.loop import _intra_turn_microcompact

    class _Opts:
        model = "primary"
        session_id = "intra_turn_probe"
        intra_turn_microcompact_trigger_fraction = 0.8
        intra_turn_microcompact_target_fraction = 0.6
        intra_turn_microcompact_min_chars = 2_000

    msgs = _tool_pairs(40, chars=4_000)
    before = list(msgs)
    _intra_turn_microcompact(
        msgs, options=_Opts(), total_usage={"input_tokens": 40_000},
        keep_recent=15, tool_count=40, iteration=40,
    )
    assert msgs == before, "cleared tool results with the window 80% empty"


def test_intra_turn_microcompact_fires_when_actually_near_the_wall():
    """And it must still work — this is a real defense on a long turn."""
    from app.harness.loop import _intra_turn_microcompact
    from app.compaction import get_context_window, truncation_threshold

    class _Opts:
        model = "primary"
        session_id = "intra_turn_probe2"
        intra_turn_microcompact_trigger_fraction = 0.8
        intra_turn_microcompact_target_fraction = 0.6
        intra_turn_microcompact_min_chars = 2_000

    threshold = truncation_threshold(get_context_window("primary"))
    msgs = _tool_pairs(40, chars=4_000)
    original_id = id(msgs)
    # vLLM reports a prompt well past the trigger; the estimator alone
    # would not have caught it, which is why the real figure is consulted.
    _intra_turn_microcompact(
        msgs, options=_Opts(), total_usage={"input_tokens": int(threshold * 0.95)},
        keep_recent=15, tool_count=40, iteration=40,
    )
    assert id(msgs) == original_id, "must mutate in place for the observer's handle"
    cleared = sum(
        1 for m in msgs
        if m.get("role") == "tool" and "cleared from context" in _marker_text(m)
    )
    assert cleared > 0, "near the wall it must still clear"
    # The recent floor is respected even under pressure.
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert all("cleared from context" not in _marker_text(m) for m in tool_msgs[-15:])


_TESTS = [
    test_estimate_tokens_returns_int,
    test_estimate_tokens_empty_string,
    test_estimate_tokens_proportional,
    test_estimate_conversation_no_double_count,
    test_truncate_under_threshold_is_noop,
    test_truncate_drops_old_turns_above_threshold,
    test_truncate_keeps_last_turn_minimum,
    test_load_and_compact_missing_file,
    test_load_and_compact_no_truncation_needed,
    test_load_and_compact_triggers_truncation,
    test_load_and_compact_accepts_str_path,
    test_load_and_compact_filters_ui_only_roles,
    test_microcompact_does_nothing_without_context_pressure,
    test_microcompact_clears_only_down_to_target_when_over_budget,
    test_microcompact_never_clears_below_the_recent_floor,
    test_microcompact_skips_results_too_small_to_be_worth_clearing,
    test_microcompact_marker_names_the_call_and_the_spill_file,
    test_microcompact_refuses_to_clear_what_it_could_not_persist,
    test_microcompact_preserves_the_spill_path_it_promises_to_keep,
    test_microcompact_legacy_count_rule_still_available,
    test_intra_turn_microcompact_is_silent_without_pressure,
    test_intra_turn_microcompact_fires_when_actually_near_the_wall,
    test_truncation_threshold_math,
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for t in _TESTS:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")

    print()
    print(f"{passed}/{len(_TESTS)} passed")
    if failed:
        print()
        print("Failures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
