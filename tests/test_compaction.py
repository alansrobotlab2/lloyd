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


def test_microcompact_marker_names_a_route_the_turn_can_take():
    """The same false promise in the pass that fires most.

    `microcompact` writes the marker half of the corpus's denials were answering
    — deep-research's second turn opens with its own first turn's markers — and
    it holds the spilled file at the path it names, so the file is real and only
    the offer is false. With `Read` denied the marker still names the path: the
    path is evidence for whoever reads the transcript, and it is the argument the
    notice is answering. What changes is the verb.
    """
    from app.harness.microcompact import microcompact
    from app.paths import SESSIONS_DIR

    sid = "test_microcompact_marker_denied"
    spill_dir = SESSIONS_DIR / f"{sid}.tool-results"
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731

    def _first_marker(disallowed):
        try:
            out, cleared = microcompact(
                _tool_pairs(20, chars=4_000), token_budget=1, estimate_fn=est,
                keep_recent_tools=5, session_id=sid, legacy_count_rule=False,
                disallowed_tools=disallowed,
            )
            assert cleared > 0
            return _marker_text([m for m in out if m.get("role") == "tool"][0])
        finally:
            if spill_dir.exists():
                for f in spill_dir.iterdir():
                    f.unlink()
                spill_dir.rmdir()

    denied = _first_marker(["Read", "Bash", "Grep"])
    assert "Read that path" not in denied, "the marker still orders a refused call"
    assert "re-run the call" in denied, denied[:200]
    assert ".tool-results" in denied, "the path is still named, as it should be"
    # The chat turn that owns Read keeps the wording every other test here pins.
    assert "Read that path if you need it again." in _first_marker(["Bash"])


def test_the_pre_turn_pass_carries_the_turn_s_deny_list(monkeypatch):
    """The kwarg has to survive the hop, or the fix stops at the live turn.

    A worker session's own earlier markers are in its stored history, and this
    is the pass that rewrites them before the model reads them again — so the
    deny list has to travel with the call, not just with the turn that produced
    the spill. Asserted at the forwarding rather than on a cleared marker:
    clearing for real needs a history over the trigger fraction, which would
    make the assertion about the threshold instead of about the argument.
    """
    import app.compaction as comp_mod
    import app.harness.microcompact as mc_mod

    seen: dict = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return list(args[0]), 0

    monkeypatch.setattr(mc_mod, "microcompact", _spy)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "worker.json"
        _write_session(p, [
            {"role": "user", "content": "research X"},
            {"role": "assistant", "content": "done"},
        ])
        _run(comp_mod.load_and_compact_session(
            p, model="qwen", disallowed_tools=["Read", "Bash"]))

    assert seen.get("disallowed_tools") == ["Read", "Bash"], seen
    # And a caller that passes nothing — every chat, ambient and UI call site —
    # forwards nothing, which the pass reads as everything-allowed.
    seen.clear()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "chat.json"
        _write_session(p, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        _run(comp_mod.load_and_compact_session(p, model="qwen"))
    assert seen.get("disallowed_tools") is None, seen


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


# ---------------------------------------------------------------------------
# D1 (review 2026-09-24): transcript rows are pointers, and the turn-start
# stack reads them
# ---------------------------------------------------------------------------


def _pointer_session(tmp_path, monkeypatch, n: int, chars: int = 5_000):
    """A session JSON whose `n` Read results were written by the real
    transcript shaping, so each over-2k row is a `<persisted-output>`
    pointer at a file in the (scratch) spill dir."""
    from app import transcript_entries as te
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    sid = "20260924_120000_d1"
    fulls: list[str] = []
    rows: list[dict] = [te.build_user_entry("read these", timestamp="T")]
    for i in range(n):
        cid = f"call_{i:03d}"
        full = "".join(f"file {i} line {j}\n" for j in range(chars // 16))
        fulls.append(full)
        tc = te.build_tool_call(cid, "Read", json.dumps({"file_path": f"/f{i}.py"}))
        rows.append(te.build_tool_call_entry(tc, timestamp="T"))
        shaped = te.shape_tool_result_for_transcript(
            full, call_id=cid, session_id=sid, tool_name="Read")
        rows.append(te.build_tool_result_entry(cid, shaped, timestamp="T"))
    rows.append(te.build_user_entry("now what?", timestamp="T"))
    p = tmp_path / f"{sid}.json"
    _write_session(p, rows)
    return p, sid, fulls


def test_turn_start_history_carries_the_pointer_and_the_file_holds_the_full_result(
        tmp_path, monkeypatch):
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG
    p, sid, fulls = _pointer_session(tmp_path, monkeypatch, 1, chars=20_000)
    out = _run(load_and_compact_session(p, model="qwen"))
    tools = [m for m in out["history"] if m.get("role") == "tool"]
    assert len(tools) == 1
    text = _marker_text(tools[0])
    assert text.startswith(PERSISTED_OUTPUT_TAG), text[:200]
    path = tmp_path / f"{sid}.tool-results" / "call_000.txt"
    assert f"Full output saved to: {path}" in text
    assert "file 0 line 0" in text, "the preview rides along while it is recent"
    # The part the old 2 KB cut threw away is on disk, whole.
    assert path.read_text() == fulls[0]
    assert fulls[0][-40:] not in text


def test_old_pointer_rows_shrink_to_their_header_and_recent_ones_keep_the_preview(
        tmp_path, monkeypatch):
    """No pressure at all: the spill-aware pass drops an old pointer's preview
    because nothing is lost — the path is kept — and never touches the
    `keep_recent_tools` window."""
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    out = _run(load_and_compact_session(p, model="qwen", mode_override="truncate"))
    tools = [_marker_text(m) for m in out["history"] if m.get("role") == "tool"]
    assert len(tools) == 20
    old, recent = tools[:5], tools[5:]          # keep_recent_tools = 15
    for i, text in enumerate(old):
        assert f"{sid}.tool-results/call_{i:03d}.txt" in text, text
        assert "preview dropped" in text
        assert f"file {i} line 0" not in text
        assert len(text) < 400
    for i, text in enumerate(recent, start=5):
        assert f"file {i} line 0" in text
        assert f"{sid}.tool-results/call_{i:03d}.txt" in text
    assert out["microcompacted"] == 5


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


def _primed_meter(reported: int, msgs: list[dict]):
    """A real `ContextMeter` that has seen one engine report of `reported`
    tokens for `msgs` — the state rung 1 reads its budget from (D6)."""
    from app.harness.context_meter import ContextMeter, context_window_for

    m = ContextMeter(context_window_for("primary"))
    m.observe_usage({"input_tokens": reported}, len(msgs))
    m.observe_append(msgs)
    return m


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
        msgs, options=_Opts(), meter=_primed_meter(40_000, msgs),
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
        msgs, options=_Opts(), meter=_primed_meter(int(threshold * 0.95), msgs),
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


# ---------------------------------------------------------------------------
# Rung 1 deny-list mode (D10)
# ---------------------------------------------------------------------------


def _named_pairs(names: list[str], *, chars: int = 6_000) -> list[dict]:
    """One assistant/tool pair per name, each result `chars` long."""
    msgs: list[dict] = [{"role": "user", "content": "go"}]
    for i, name in enumerate(names):
        cid = f"call_{i:03d}"
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function", "function": {
                "name": name, "arguments": json.dumps({"q": f"q{i}"})}}],
        })
        msgs.append({"role": "tool", "tool_call_id": cid,
                     "content": f"{name} result {i}\n" * (chars // 20)})
    return msgs


def _cleared_names(msgs: list[dict]) -> list[str]:
    names = {tc["id"]: tc["function"]["name"]
             for m in msgs if m.get("role") == "assistant"
             for tc in m.get("tool_calls") or []}
    return [names[m["tool_call_id"]] for m in msgs
            if m.get("role") == "tool" and "cleared from context" in _marker_text(m)]


def test_microcompact_deny_list_clears_domain_results():
    """An MCP domain result (here `vault_search`, and its namespaced form) was
    never in the allow-list, so under pressure it went straight to truncation.
    Deny-list mode clears it like any other stale result."""
    from app.harness.microcompact import DEFAULT_NON_COMPACTABLE, microcompact

    msgs = _named_pairs(["vault_search", "mcp__lloyd-mcp__http_fetch"] * 10)
    kw = dict(keep_recent_tools=4, token_budget=1, legacy_count_rule=False,
              estimate_fn=lambda m: estimate_conversation_tokens(m, ""),
              min_chars_to_clear=2_000, session_id="")

    _, allow_cleared = microcompact(msgs, **kw)
    assert allow_cleared == 0, "allow-list mode must still ignore domain tools"

    out, cleared = microcompact(msgs, non_compactable_tools=DEFAULT_NON_COMPACTABLE, **kw)
    assert cleared == 16, cleared           # 20 results, the newest 4 kept
    assert set(_cleared_names(out)) == {"vault_search", "mcp__lloyd-mcp__http_fetch"}


def test_microcompact_deny_list_protects_todowrite_and_toolsearch():
    """What the model steers by — its todo list, the catalogue ToolSearch loaded —
    is never cleared, bare or namespaced, however stale."""
    from app.harness.microcompact import DEFAULT_NON_COMPACTABLE, microcompact

    names = ["TodoWrite", "mcp__lloyd-mcp__ToolSearch", "vault_search"] * 6
    msgs = _named_pairs(names)
    out, cleared = microcompact(
        msgs, keep_recent_tools=0, token_budget=1, legacy_count_rule=False,
        estimate_fn=lambda m: estimate_conversation_tokens(m, ""),
        min_chars_to_clear=2_000, session_id="",
        non_compactable_tools=DEFAULT_NON_COMPACTABLE,
    )
    assert cleared == 6, cleared
    assert set(_cleared_names(out)) == {"vault_search"}


def test_the_intra_turn_pass_reads_the_deny_list_from_config():
    """The key reaches rung 1 through the seam every turn path builds its
    options from (`intra_turn_compaction_kwargs`), and its absence keeps the
    allow-list — so removing the key is the off switch."""
    import app.mcp_discovery as disc
    from app.compaction import get_context_window, truncation_threshold
    from app.harness.loop import _intra_turn_microcompact
    from app.harness.options import RunOptions

    base = {"trigger_fraction": 0.8, "target_fraction": 0.6}
    undo = _stub(disc, "CONFIG", {"compaction": {"microcompact": dict(
        base, non_compactable_tools=["TodoWrite", "ToolSearch"])}})
    try:
        on = disc.intra_turn_compaction_kwargs()
    finally:
        undo()
    undo = _stub(disc, "CONFIG", {"compaction": {"microcompact": dict(base)}})
    try:
        off = disc.intra_turn_compaction_kwargs()
    finally:
        undo()
    assert on["intra_turn_microcompact_non_compactable"] == ("TodoWrite", "ToolSearch")
    assert "intra_turn_microcompact_non_compactable" not in off

    threshold = truncation_threshold(get_context_window("primary"))

    def _run(kwargs):
        opts = RunOptions(model="primary", session_id="", **kwargs)
        msgs = _named_pairs(["vault_search"] * 20 + ["TodoWrite"] * 2)
        cleared = _intra_turn_microcompact(
            msgs, options=opts, meter=_primed_meter(int(threshold * 0.95), msgs),
            keep_recent=4, tool_count=22, iteration=22,
        )
        return cleared, _cleared_names(msgs)

    cleared, names = _run(on)
    assert cleared > 0 and set(names) == {"vault_search"}, (cleared, names)
    assert _run(off) == (0, [])


# ---------------------------------------------------------------------------
# A recovery notice must not offer a tool the turn cannot use (#1066)
# ---------------------------------------------------------------------------


def _clean_spill_dir(sid: str) -> None:
    from app.paths import SESSIONS_DIR

    d = SESSIONS_DIR / f"{sid}.tool-results"
    if d.exists():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


def _spill_block(disallowed):
    """One spilled `<persisted-output>` block, with the turn's deny list."""
    from app.harness.tool_result_spill import maybe_spill

    # One fixed session id: the two calls in the "unchanged" test below must
    # produce byte-identical blocks, and a per-call id would put a different
    # spill path in each and make that comparison always fail.
    sid = "spill_notice_probe"
    try:
        return maybe_spill(
            "y" * 60_000, tool_name="Grep", tool_use_id="call_spill",
            session_id=sid, disallowed_tools=disallowed,
        )
    finally:
        _clean_spill_dir(sid)


def test_spill_notice_stops_offering_read_to_a_turn_that_cannot_use_it():
    """The notice ordered a call the same turn's policy refused.

    10 of the 25 deep-research denials in the 09-14→09-17 window were the
    harness instructing the denied call: `Read` is on that source's deny list
    (`workers/sources/deep_research.py` denies it), and both notices tell the model to
    Read the path. Reproduced in `sessions/20260917_004805_deepresearch_4032.json`
    — a cleared-under-pressure notice, then a `Read` denial on
    `…tool-results/chatcmpl-tool-97eab20b3d3646c1.truncated.json`, then a `Bash`
    denial against the same directory.
    """
    for denied in (["Read", "Bash"], ["mcp__lloyd-mcp__Read"]):
        block = _spill_block(denied)
        assert "Read the full file" not in block, f"{denied}: the false promise stands"
        assert "narrower query" in block, "the recovery it offers is not a real one"
        assert "saved to:" in block, "the file itself is still named, as it should be"


def test_the_spill_notice_is_unchanged_for_a_turn_that_has_read():
    """The chat agent owns Read, and the wording it relies on must not move —
    this rung fires on every oversized result a chat turn produces."""
    with_read = _spill_block(["Bash"])
    assert "Read the full file with the Read tool" in with_read
    assert "narrow your next query" in with_read
    # Omitting the argument is the other shape of "allowed": a caller that
    # passes nothing must not read as a turn with everything denied.
    assert _spill_block(None) == with_read


def test_context_pressure_notice_stops_offering_read_to_a_turn_that_cannot_use_it():
    """`_truncate_largest_tool_results` is the notice the corpus actually shows,
    so it gets the same treatment as the spill block — and the same test: a
    turn with `Read` allowed keeps its wording."""
    from app.harness.loop import _truncate_largest_tool_results

    def _notice(disallowed, sid):
        msgs = _tool_pairs(4, chars=30_000)
        try:
            truncated, _freed = _truncate_largest_tool_results(
                msgs, target_chars=10_000, session_id=sid, min_chars=4_096,
                disallowed_tools=disallowed,
            )
            assert truncated > 0, "nothing was truncated, so the notice never rendered"
            for m in msgs:
                if m.get("role") == "tool" and "cleared under context pressure" in _marker_text(m):
                    return _marker_text(m)
            raise AssertionError("no cleared-under-pressure notice was produced")
        finally:
            _clean_spill_dir(sid)

    denied = _notice(["Read"], "truncate_read_denied_probe")
    assert "Read that path" not in denied
    assert "narrower query" in denied, "the recovery it offers is not a real one"
    assert ".tool-results/" in denied, "the spilled evidence is still named"

    allowed = _notice([], "truncate_read_allowed_probe")
    assert "Read that path if you need it again." in allowed, (
        "the wording a turn that owns Read depends on moved")


# ---------------------------------------------------------------------------
# Which mechanism fired, recorded per turn (#1078)
# ---------------------------------------------------------------------------
#
# Until this section, the only trace of the turn-start stack was one `logger.info`
# line inside `if truncated or summarized or microcompacted:` — so the two
# questions the item names were both unanswerable. A reader of
# `logs/server.err*` could not tell "the summarize layer ran and declined" from
# "the turn never reached it", and 9 days of logs with `summarized=True`
# appearing 0 times could not say whether `compaction.mode: summarize` was dead
# configuration or alive and simply not needed.


def _big_history(turns: int = 50, chars: int = 4_000) -> list[dict]:
    """A conversation over the compaction threshold without touching the LLM."""
    msgs: list[dict] = []
    for _ in range(turns):
        msgs.append(_mk("user", chars))
        msgs.append(_mk("assistant", chars))
    return msgs


def _events_for(tmp_dir, session_stem: str) -> list[dict]:
    """This session's rows, read from the directory the CALLER points at.

    Every caller reaches here after `_stub(event_log, "EVENT_LOGS_DIR", ...)` to
    that same path, so reading the module attribute would give the same answer —
    but then the argument would name one path while the code read another, which
    is how a helper ends up asserting about a directory no test wrote to. Reading
    the argument means a caller that forgets its stub gets an empty list and a
    failing assertion, not a file someone else's test happened to leave behind.
    """
    path = Path(tmp_dir) / f"{session_stem}.events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def _turn_start_events(tmp_dir, session_stem: str) -> list[dict]:
    return [e for e in _events_for(tmp_dir, session_stem)
            if e["event"] == "compaction.turn_start"]


def _record(out: dict) -> dict:
    """The record this turn's writers would store — one shared projection."""
    from app.compaction_record import turn_start_record

    record = turn_start_record(out)
    assert record is not None, (
        "the stack ran on this history, so it must produce a record; None is "
        "reserved for a turn that never reached it"
    )
    return record


def test_a_truncating_turn_records_the_layer_that_acted_with_its_numbers():
    """The clause's first half: which of microcompact/summarize/truncate acted,
    and tokens before and after. `tokens_after` is the truncated estimate, so
    the pair is the real amount removed rather than a rung count — the number
    #600's gate needs to say a mechanism actually moved the history.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, _big_history())
        out = _run(load_and_compact_session(
            p, model="qwen-unknown", mode_override="truncate"))

    record = _record(out)
    assert out["truncated"] is True
    assert record["mechanisms"] == ["truncate"], record["mechanisms"]
    assert record["tokens_before"] == out["tokens_before"]
    assert record["tokens_after"] == out["tokens_after"]
    assert record["tokens_freed"] == out["tokens_before"] - out["tokens_after"]
    assert record["tokens_freed"] > 0
    assert record["ran"] is True


def test_a_summarize_pass_that_ran_and_declined_is_not_recorded_as_unreached():
    """The clause's second half, and the item's central distinction.

    The summarize layer was entered, called the summary model, got nothing back,
    and the turn fell through to truncate. `summarized` is False — which is the
    same flag value a turn under the threshold carries — so the flag alone cannot
    answer "did the layer have its chance". `summarize_attempted` and
    `summarize_outcome` can, and they are why `summarized=True` appearing zero
    times in a log window stopped being evidence of anything.
    """
    import app.compaction_llm as llm_mod
    import tempfile

    async def _empty_summary(*args, **kwargs):
        return ""

    undo = _stub(llm_mod, "summarize_history", _empty_summary)
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            _write_session(p, _big_history())
            out = _run(load_and_compact_session(
                p, model="qwen-unknown", mode_override="summarize"))
    finally:
        undo()

    record = _record(out)
    assert out["summarized"] is False
    assert record["summarize_attempted"] is True
    assert record["summarize_outcome"] == "empty_summary"
    # Declining to summarize is not the same as the layer not running, and the
    # history was still rewritten — by the fallback, which is named.
    assert record["mechanisms"] == ["truncate"]


def test_a_summarize_layer_that_could_not_be_imported_is_its_own_outcome():
    """The one branch of this layer that ever logged anything
    (`compaction.summarize_fallback`) said "failed to import, falling back to
    truncate". That is a third reason for `summarized: False`, and it now has a
    name in the record instead of a log line that rotates away.
    """
    import sys
    import tempfile

    saved = sys.modules.get("app.compaction_llm")
    sys.modules["app.compaction_llm"] = None  # makes `from ... import` raise
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            _write_session(p, _big_history())
            out = _run(load_and_compact_session(
                p, model="qwen-unknown", mode_override="summarize"))
    finally:
        if saved is None:
            sys.modules.pop("app.compaction_llm", None)
        else:
            sys.modules["app.compaction_llm"] = saved

    record = _record(out)
    assert record["summarize_attempted"] is True
    assert record["summarize_outcome"] == "import_failed"
    assert out["truncated"] is True, "the fallback is what saved this turn"


def test_a_history_under_the_threshold_records_under_threshold_not_an_absence():
    """`summarize_attempted` False with `under_threshold` is a measurement: the
    stack ran, read the size, and left the history alone. A turn that never
    called this function produces no record at all (`turn_start_record(None)` is
    None), which is what keeps NULL meaning unmeasured.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        out = _run(load_and_compact_session(p, model="qwen"))

    record = _record(out)
    assert record["mechanisms"] == []
    assert record["summarize_attempted"] is False
    assert record["summarize_outcome"] == "under_threshold"
    assert record["tokens_before"] == record["tokens_after"]


def test_a_truncate_mode_turn_records_the_mode_as_the_reason_it_skipped():
    """Under `mode: truncate` the summarize layer is skipped by configuration.
    That reads identically to "under threshold" in every flag the old record
    carried, and it is the answer to the dead-configuration question itself: if
    every row says `mode_truncate`, the summarize mode is not running.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        _write_session(p, _big_history())
        out = _run(load_and_compact_session(
            p, model="qwen-unknown", mode_override="truncate"))

    record = _record(out)
    assert record["summarize_attempted"] is False
    assert record["summarize_outcome"] == "mode_truncate"


def test_ran_and_declined_is_a_different_answer_from_never_reached():
    """The two turns this clause separates, side by side in one assertion,
    through the object that actually reaches the database.
    """
    from app.compaction_record import TurnCompaction

    declined = TurnCompaction(session_id="s", turn_id="t")
    declined.note_turn_start({
        "tokens_before": 5_000, "tokens_after": 5_000,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
    })
    never_reached = TurnCompaction(session_id="s", turn_id="t")

    assert declined.to_record()["turn_start"]["summarize_outcome"] == \
        "under_threshold"
    assert never_reached.to_record() is None, (
        "a turn whose writers never called the stack stores NULL, not a decline"
    )


def test_a_rewrite_emits_one_event_carrying_the_session_id():
    """The event half of the record, and the half that survives a rotation.

    The session id is the session file's stem, which is what makes the join
    possible: the relief ladder's 7,253 summary lines in the retained window
    carried none, and `grep "context relief" | grep -cE "session="` returned 0 of
    7,253 — attribution had to come from a new record, not from the log.
    """
    from app import event_log

    stem = "20260924_070000_mission-control_a1b2"
    with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as d:
        undo = _stub(event_log, "EVENT_LOGS_DIR", Path(ev))
        try:
            p = Path(d) / f"{stem}.json"
            _write_session(p, _big_history())
            out = _run(load_and_compact_session(
                p, model="qwen-unknown", mode_override="truncate"))
            events = _turn_start_events(Path(ev), stem)
        finally:
            undo()

    assert len(events) == 1, f"expected exactly one event, got {events}"
    assert events[0]["session_id"] == stem
    assert events[0]["data"]["mechanisms"] == ["truncate"]
    assert events[0]["data"]["tokens_freed"] == \
        out["tokens_before"] - out["tokens_after"]


def test_a_declined_turn_start_emits_no_event():
    """Events are rewrites; decisions are the usage column.

    The two stores answer different questions and must not be summed: a count of
    rows with a non-NULL `compaction` counts turns the stack saw, and a
    per-session event count counts rewrites. An event on every turn would make
    the second number the first, and the ladder's firing rate would disappear
    into turn volume.
    """
    from app import event_log

    stem = "20260924_070000_mission-control_c3d4"
    with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as d:
        undo = _stub(event_log, "EVENT_LOGS_DIR", Path(ev))
        try:
            p = Path(d) / f"{stem}.json"
            _write_session(p, [{"role": "user", "content": "hi"}])
            _run(load_and_compact_session(p, model="qwen"))
            events = _events_for(Path(ev), stem)
        finally:
            undo()

    assert events == [], f"a turn that rewrote nothing must emit nothing: {events}"


def test_a_microcompact_only_rewrite_below_the_summarize_threshold_still_emits():
    """Why the retained window held 7,253 relief passes and 0 `[compaction]`
    lines: the pre-pass runs ahead of the threshold check (`token_budget=None`
    below the trigger, but the pass itself is unconditional), so a turn can be
    rewritten while the summarize layer is never entered. A record keyed on the
    summarize layer's decision would miss every one of those, and the mechanism
    that fires most would be the one mechanism not recorded.
    """
    from app import event_log
    import app.harness.microcompact as mc_mod

    def _clear_one_third(messages, **kwargs):
        return list(messages), 3

    stem = "20260924_070000_mission-control_e5f6"
    with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as d:
        undo_ev = _stub(event_log, "EVENT_LOGS_DIR", Path(ev))
        undo_mc = _stub(mc_mod, "microcompact", _clear_one_third)
        try:
            p = Path(d) / f"{stem}.json"
            _write_session(p, [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ])
            out = _run(load_and_compact_session(p, model="qwen"))
            events = _turn_start_events(Path(ev), stem)
        finally:
            undo_mc()
            undo_ev()

    assert out["microcompacted"] == 3
    record = _record(out)
    assert record["mechanisms"] == ["microcompact"]
    assert record["summarize_attempted"] is False, (
        "this turn never reached the summarize layer, and the event exists anyway"
    )
    assert len(events) == 1, events
    assert events[0]["data"]["mechanisms"] == ["microcompact"]


def _stub(module, name, replacement):
    """setattr-with-restore, for the tests in this file that cannot take a
    `monkeypatch` fixture because they are also listed in `_TESTS` and run under
    `python -m tests.test_compaction`.
    """
    saved = getattr(module, name)

    def _undo():
        setattr(module, name, saved)
    setattr(module, name, replacement)
    return _undo


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
    test_microcompact_deny_list_clears_domain_results,
    test_microcompact_deny_list_protects_todowrite_and_toolsearch,
    test_the_intra_turn_pass_reads_the_deny_list_from_config,
    test_truncation_threshold_math,
    test_microcompact_marker_names_a_route_the_turn_can_take,
    test_the_pre_turn_pass_carries_the_turn_s_deny_list,
    test_spill_notice_stops_offering_read_to_a_turn_that_cannot_use_it,
    test_the_spill_notice_is_unchanged_for_a_turn_that_has_read,
    test_context_pressure_notice_stops_offering_read_to_a_turn_that_cannot_use_it,
    test_a_truncating_turn_records_the_layer_that_acted_with_its_numbers,
    test_a_summarize_pass_that_ran_and_declined_is_not_recorded_as_unreached,
    test_a_summarize_layer_that_could_not_be_imported_is_its_own_outcome,
    test_a_history_under_the_threshold_records_under_threshold_not_an_absence,
    test_a_truncate_mode_turn_records_the_mode_as_the_reason_it_skipped,
    test_ran_and_declined_is_a_different_answer_from_never_reached,
    test_a_rewrite_emits_one_event_carrying_the_session_id,
    test_a_declined_turn_start_emits_no_event,
    test_a_microcompact_only_rewrite_below_the_summarize_threshold_still_emits,
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
