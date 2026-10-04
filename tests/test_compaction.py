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
# #2168: relief's reduction survives its turn as a read-time sidecar
#
# Every node below runs the REAL producer (`record_reduced_calls`, the file
# relief's caller writes) and the REAL consumer (`load_and_compact_session`),
# and reads the history the turn-start pre-pass was handed through a spy on
# `microcompact` — so "before the first relief pass" is witnessed, not
# asserted from the final history alone. The falsifier each node leans on is
# the knob-off number for this fixture, pinned above at
# `test_old_pointer_rows_shrink_to_their_header_and_recent_ones_keep_the_preview`:
# 20 spilled rows, `keep_recent_tools` 15, so the pre-pass clears exactly 5
# (`call_000` … `call_004`). A sidecar naming `call_000` and `call_001` must
# turn that 5 into 3, and 5 back into 5 the moment the sidecar is foreign,
# deleted, missing or the knob is off.
# ---------------------------------------------------------------------------


def _record_reduced(sid: str, ids: list[str]) -> None:
    """Write the sidecar with the writer relief's caller uses, not a fixture
    dict — the read-time pass must agree with the real file shape."""
    from app.harness.microcompact import record_reduced_calls
    record_reduced_calls(sid, ids)


def _handed_to_the_pass(monkeypatch) -> list[dict]:
    """Wrap the module attribute `app.compaction` imports at call time, so a
    node can see the message list the pre-pass receives. Each entry is the
    history as handed, one dict per `microcompact` call."""
    import app.harness.microcompact as mc_mod
    real = mc_mod.microcompact
    seen: list[list[dict]] = []

    def _spy(msgs, **kw):
        seen.append({m.get("tool_call_id"): _marker_text(m)
                     for m in msgs if m.get("role") == "tool"})
        return real(msgs, **kw)

    monkeypatch.setattr(mc_mod, "microcompact", _spy)
    return seen


def test_a_reduced_id_from_the_last_turn_arrives_reduced_and_the_pass_clears_it_again_no_more(
        tmp_path, monkeypatch):
    """#2168 clause 1: turn N reduced `call_000` and `call_001`; turn N+1's
    rebuilt history must carry both in reduced shape when the pre-pass starts,
    and that pass must book no clear for them."""
    from app.harness.microcompact import PREVIEW_DROPPED
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced(sid, ["call_000", "call_001"])
    handed = _handed_to_the_pass(monkeypatch)

    out = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))

    assert len(handed) == 1, "the pre-pass is one relief pass over the history"
    for i, cid in enumerate(("call_000", "call_001")):
        text = handed[0][cid]
        assert PREVIEW_DROPPED in text, (
            f"{cid} must already be reduced when the pass starts, got {text[:200]}")
        assert f"file {i} line 0" not in text, (
            f"{cid}'s preview is the bytes the sidecar was for")
        assert f"{sid}.tool-results/{cid}.txt" in text, text
    assert out["microcompacted"] == 3, (
        "the pass must find nothing left to do to those two: 5 rows are stale, "
        f"2 arrive reduced, so 3 clears — got {out['microcompacted']}")
    # Byte-exact claim that the two recorded ids were not rewritten again: the
    # rows that changed are exactly the three the pass had to reduce itself.
    final = {m.get("tool_call_id"): _marker_text(m)
             for m in out["history"] if m.get("role") == "tool"}
    changed = {cid for cid in final if final[cid] != handed[0][cid]}
    assert changed == {"call_002", "call_003", "call_004"}, sorted(changed)


def test_the_sidecar_reduces_the_prompt_and_never_the_session_file(
        tmp_path, monkeypatch):
    """#2168 clause 2: for the same run, `<sid>.json` is byte-identical with the
    knob on and off, because the reduction is a read-time act and the row on disk
    keeps the whole pointer block the UI renders and `Read` reopens."""
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced(sid, ["call_000", "call_001"])
    before = p.read_bytes()
    assert b"file 0 line 0" in before, (
        "the row on disk starts carrying its preview — the equality below is "
        "worth nothing if it never had one")

    on = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))
    after_on = p.read_bytes()
    off = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=False))

    assert p.read_bytes() == before, "no writer may touch <sid>.json"
    assert after_on == before, "the knob-on load wrote to the session file"
    assert b"file 0 line 0" in after_on, (
        "a single reduction pass that trimmed the row would still read back "
        "unchanged on the second run only if it never wrote — check the first")
    assert on["microcompacted"] == 3 and off["microcompacted"] == 5, (
        "the two runs must differ in the prompt while not differing on disk")


def test_a_read_time_reduction_keeps_the_saved_to_line_and_leaves_a_lost_file_inline(
        tmp_path, monkeypatch):
    """#2168 clause 3: what the read-time pass leaves behind must still name the
    file, and it must never reduce a row whose spill file has gone. The second
    half is asserted on the history the pass RECEIVED: the spill-aware pass may
    still trim `call_001` itself (it does that today, file or no file), and this
    clause is about the sidecar not trading a row's content for a dead path."""
    from app.harness.microcompact import PREVIEW_DROPPED
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    gone = tmp_path / f"{sid}.tool-results" / "call_001.txt"
    gone.unlink()
    _record_reduced(sid, ["call_000", "call_001"])
    handed = _handed_to_the_pass(monkeypatch)

    _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))

    kept = handed[0]["call_000"]
    assert PREVIEW_DROPPED in kept, kept[:200]
    assert (
        f"Full output saved to: {tmp_path}/{sid}.tool-results/call_000.txt" in kept
    ), f"the reduced row lost its route back: {kept}"
    skipped = handed[0]["call_001"]
    assert PREVIEW_DROPPED not in skipped, (
        "call_001's spill file was deleted, so the sidecar must not reduce it")
    assert "file 1 line 0" in skipped, skipped[:200]


def test_the_sidecar_is_keyed_by_session_id(tmp_path, monkeypatch):
    """#2168 clause 4: another session's sidecar is inert here, and with this
    session's own sidecar deleted the pass is byte-for-byte today's pass."""
    from app.harness.microcompact import PREVIEW_DROPPED, reduction_sidecar_path
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced("20260101_000000_some_other_session",
                    ["call_000", "call_001"])
    handed = _handed_to_the_pass(monkeypatch)

    foreign = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))
    assert PREVIEW_DROPPED not in handed[0]["call_000"], (
        "a sidecar named for another session id must not reach these rows")
    assert foreign["microcompacted"] == 5, (
        f"a foreign sidecar changed the clears: {foreign['microcompacted']}")

    off = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=False))
    _record_reduced(sid, ["call_000", "call_001"])
    with_it = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))
    assert with_it["microcompacted"] == 3, "the sidecar must take effect when present"
    reduction_sidecar_path(sid).unlink()
    deleted = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))

    assert deleted["microcompacted"] == 5, (
        "with the sidecar gone the previews are the pass's to drop, as today")
    assert [_marker_text(m) for m in deleted["history"] if m["role"] == "tool"] \
        == [_marker_text(m) for m in off["history"] if m["role"] == "tool"], (
        "a deleted sidecar must reproduce the knob-off history byte for byte")


def test_the_shipped_default_does_not_consult_the_sidecar_at_all(tmp_path, monkeypatch):
    """#2168 clause 5: the knob ships off, and off means the pre-pass never reads
    the sidecar — not that it reads it and applies nothing."""
    from app.harness.options import RunOptions
    import app.harness.microcompact as mc_mod

    assert RunOptions.__dataclass_fields__[
        "microcompact_reduction_sidecar"].default is False, (
        "the shipped default is what clause 6 turns on after the deploy gate; "
        "if this flips, the owed measurement has already been skipped")

    calls: list[str] = []
    real_apply = mc_mod.apply_reduction_sidecar

    def _counting(msgs, session_id):
        calls.append(session_id)
        return real_apply(msgs, session_id)

    monkeypatch.setattr(mc_mod, "apply_reduction_sidecar", _counting)
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced(sid, ["call_000", "call_001"])
    handed = _handed_to_the_pass(monkeypatch)

    out = _run(load_and_compact_session(p, model="qwen", mode_override="truncate"))

    assert calls == [], "with the shipped default the sidecar is never opened"
    # And the node is testing the shipped value, not a parameter fixture:
    # `load_and_compact_session` carries no default of its own, so a turn that
    # passes nothing reads `RunOptions` — which is where clause 6 flips it on.
    import inspect
    assert inspect.signature(load_and_compact_session).parameters[
        "microcompact_sidecar"].default is None
    assert out["microcompacted"] == 5, (
        f"the pass changed with the knob untouched: {out['microcompacted']}")
    assert "file 0 line 0" in handed[0]["call_000"], handed[0]["call_000"][:200]

    # The ship-on lever, which the assertions above cannot supply on their own: every
    # one of them still holds if the pre-pass's read of
    # `options.microcompact_reduction_sidecar` is replaced by a literal False — the
    # shortcut whoever reaches for after the owed deploy gate answers "stay off" — and
    # the knob then survives as a comment nobody can turn back on. Same spy, same
    # sidecar, one run with the knob True: the helper must be reached, and reached with
    # the two ids it names actually reduced.
    _on_handed = _handed_to_the_pass(monkeypatch)
    out_on = _run(load_and_compact_session(
        p, model="qwen", mode_override="truncate", microcompact_sidecar=True))
    assert calls == [sid], (
        "with the knob ON the sidecar is still never opened, so the read is not wired "
        "to the knob and there is nothing for the deploy gate to turn on")
    assert sum(1 for txt in _on_handed[0].values()
               if "preview dropped" in txt) == 2, _on_handed[0]
    assert out_on["microcompacted"] == 3, out_on


def test_the_relief_pass_that_reduces_a_spill_records_what_it_reduced(
        tmp_path, monkeypatch):
    """#2168, the producer half across the loop seam: rung 1 reducing a spilled
    row is what the sidecar exists to remember, so the ids must come out of
    `_intra_turn_microcompact` with the knob on and not at all with it off."""
    from app.harness.loop import _intra_turn_microcompact
    from app.harness.microcompact import read_reduced_calls

    class _Opts:
        model = "primary"
        session_id = "sidecar_probe"
        intra_turn_microcompact_trigger_fraction = 0.01
        intra_turn_microcompact_target_fraction = 0.01
        intra_turn_microcompact_min_chars = 2_000
        microcompact_reduction_sidecar = True

    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    msgs = _pointer_messages("sidecar_probe", 20)
    _intra_turn_microcompact(
        msgs, options=_Opts(), meter=None, keep_recent=15,
        tool_count=20, iteration=20)

    recorded = read_reduced_calls("sidecar_probe")
    assert recorded == tuple(f"call_{i:03d}" for i in range(5)), (
        "rung 1 reduced the five stale pointer rows, so the sidecar holds "
        f"exactly those ids, got {recorded}")
    assert (tmp_path / "sidecar_probe.microcompact-reduced.json").is_file(), (
        "the sidecar sits beside the session's own record, keyed by its id")

    class _Off(_Opts):
        microcompact_reduction_sidecar = False

    (tmp_path / "sidecar_probe.microcompact-reduced.json").unlink()
    msgs2 = _pointer_messages("sidecar_probe", 20)
    _intra_turn_microcompact(
        msgs2, options=_Off(), meter=None, keep_recent=15,
        tool_count=20, iteration=20)
    assert any(m.get("role") == "tool" and "preview dropped" in _marker_text(m)
               for m in msgs2), "relief still relieves with the knob off"
    assert read_reduced_calls("sidecar_probe") == (), (
        "with the knob off nothing is recorded, so nothing can be replayed")


def _pointer_messages(sid: str, n: int, chars: int = 5_000) -> list[dict]:
    """The same 20 spilled Read results as `_pointer_session`, but as the live
    `chat_messages` list rung 1 works on rather than as a session file: spilled
    by the real shaping, so every row is a `<persisted-output>` pointer."""
    from app import transcript_entries as te
    msgs: list[dict] = [{"role": "user", "content": "read these"}]
    for i in range(n):
        cid = f"call_{i:03d}"
        full = "".join(f"file {i} line {j}\n" for j in range(chars // 16))
        tc = te.build_tool_call(cid, "Read", json.dumps({"file_path": f"/f{i}.py"}))
        msgs.append({"role": "assistant", "content": "", "tool_calls": [tc]})
        shaped = te.shape_tool_result_for_transcript(
            full, call_id=cid, session_id=sid, tool_name="Read")
        msgs.append({"role": "tool", "tool_call_id": cid, "content": shaped})
    return msgs



# ---------------------------------------------------------------------------
# Threshold math
# ---------------------------------------------------------------------------

#: #2168 clause 6 — the rows behind the measurement this item's premise is made of,
#: committed as bytes so a later reader can re-run the query. This file is the copy
#: the node opens: the gate runs with HOME at the round home, where the vault tree
#: does not exist, so a node reading a vault witness would skip — and a skipping
#: node pins nothing. A byte-identical copy, 4,112,384 bytes, is committed in the vault's
#: witness store as
#: `backlog/data/usage-microcompact-2026-10-04.db`, with its provenance note beside it,
#: and `sqlite3 -readonly` over that file answers the clause's own re-derive:
#: `select count(*) from sqlite_master` -> 2. (That note records the copy's md5 as a
#: content-identity pin. It is not a commit in either repository and no command here
#: resolves it as one — a digest is how you say "these are the same bytes", which is
#: exactly the claim a commit id cannot make across two repos.) The
#: other frozen extract already on that store's `usage.db` path is #1918/#1919's:
#: 3,787 rows spanning 2026-09-22 to 2026-09-30 with columns `ts`, `session_id`,
#: `prefix_misses` and `reprefill_tokens` — no `compaction` column, so it cannot
#: answer this query, and rewriting it would move every figure its own provenance
#: note quotes.
WITNESS_DB = Path(__file__).resolve().parent / "fixtures" / "usage-microcompact-2026-10-04.db"

#: The item's own report, verbatim, and the window its rows are cut to.
WITNESS_REPORT = {
    "clearing_turns": 271,
    "sessions": 159,
    "repeat_clearing_turns": 112,
    "repeat_reprefill": 37_512_065,
    "total_reprefill": 84_856_726,
    "rows": 4_852,
    "objects": 2,
    "window": ("2026-09-27T00:00:26", "2026-10-04T08:17:44"),
}


def test_the_witness_rows_behind_the_repeat_clear_measure_reproduce_the_quoted_report():
    """#2168 clause 6: the premise's numbers come from bytes in this repo, because
    `~/lloyd-data/usage.db` is a rolling store and the window they were measured
    over slides out from under the next reader.

    271 turns in 159 sessions carry `turn_start.microcompacted > 0` across
    2026-09-27T00:00:26 → 2026-10-04T08:17:44; 112 of them are repeat clearing
    turns (a session's second or later clearing turn) carrying 37,512,065 of the
    84,856,726 reprefilled tokens — the re-clear this round's sidecar exists to
    stop paying for.
    """
    import sqlite3

    assert WITNESS_DB.is_file(), f"{WITNESS_DB.name} is the premise's witness"
    db = sqlite3.connect(f"file:{WITNESS_DB}?mode=ro", uri=True)
    try:
        # `select count(*) from sqlite_master` over these bytes — the clause's own
        # figure: the `usage` table and its `ts` index and nothing else, so no
        # cost, model or prompt column was carried into the witness.
        assert db.execute("select count(*) from sqlite_master").fetchone()[0] \
            == WITNESS_REPORT["objects"]
        rows, lo, hi = db.execute(
            "select count(*), min(ts), max(ts) from usage").fetchone()
        assert (rows, (lo, hi)) == (WITNESS_REPORT["rows"], WITNESS_REPORT["window"]), \
            f"witness window moved: {rows}, {lo}, {hi}"
        got = db.execute(
            "WITH mc AS (SELECT session_id, ts, COALESCE(reprefill_tokens,0) rep, "
            "json_extract(compaction,'$.turn_start.microcompacted') mcn "
            "FROM usage WHERE ts>='2026-09-27' "
            "AND json_extract(compaction,'$.turn_start.microcompacted')>0), "
            "ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY session_id "
            "ORDER BY ts) rn FROM mc) "
            "SELECT COUNT(*), COUNT(DISTINCT session_id), "
            "SUM(CASE WHEN rn>1 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN rn>1 THEN rep ELSE 0 END), SUM(rep) FROM ranked;"
        ).fetchone()
    finally:
        db.close()

    assert got == (
        WITNESS_REPORT["clearing_turns"], WITNESS_REPORT["sessions"],
        WITNESS_REPORT["repeat_clearing_turns"], WITNESS_REPORT["repeat_reprefill"],
        WITNESS_REPORT["total_reprefill"],
    ), (
        "the committed witness no longer answers the query the item quoted: "
        f"turns={got[0]} sessions={got[1]} repeats={got[2]} "
        f"repeat_reprefill={got[3]} total_reprefill={got[4]}")
    share = got[3] / got[4]
    assert 0.43 < share < 0.46, (
        "the repeat clears are the avoidable part of the window: 37,512,065 of "
        f"84,856,726 is 44.2%, this extract says {share:.1%}")


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


def _spill_block(disallowed, tool_name="Grep"):
    """One spilled `<persisted-output>` block, with the turn's deny list and
    the tool that produced the result.

    The tool defaults to `Grep` because every test written before #2067 asks
    about a spill the knob list really describes, and #1066's fix must not
    move for it.
    """
    from app.harness.tool_result_spill import maybe_spill

    # One fixed session id: the two calls in the "unchanged" test below must
    # produce byte-identical blocks, and a per-call id would put a different
    # spill path in each and make that comparison always fail.
    sid = "spill_notice_probe"
    try:
        return maybe_spill(
            "y" * 60_000, tool_name=tool_name, tool_use_id="call_spill",
            session_id=sid, disallowed_tools=disallowed,
        )
    finally:
        _clean_spill_dir(sid)


#: Every argument the without-Read sentence used to name. None of them is a
#: parameter of `http_fetch` (`agent_mcp/http_tools.py:390` takes `url`,
#: `extract_mode`, `max_chars`), so for a fetch each one is a call the model
#: cannot make — the #1066 false promise wearing another tool's clothes.
_KNOB_PHRASES = ("narrower query", "hops", "min_confidence", "--glob",
                 "--type", "head_limit")


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


def test_a_spilled_fetch_on_a_read_denied_turn_is_told_there_is_no_route():
    """The pointer named an argument the fetching tool does not have (#2067).

    `recovery_notice` chose its sentence from the deny list alone, so a
    spilled `http_fetch` on a turn with `Read` denied got the graph/grep knob
    list — and `http_fetch` takes `url`, `extract_mode` and `max_chars`, none
    of which selects a region of a page.

    The population is the 202 committed witness rows, one per spilled
    `http_fetch` result whose block carried both the without-Read sentence and
    that knob list: `wc -l < backlog/data/20261001_191646_deepresearch_715a.json`
    in the vault, where those bytes have history the session store has never
    had. Over those bytes `raw_chars` is
    median 13,894, p90 35,547, max 72,348, and the text that reached the
    transcript (`result_chars`) is 2,527–2,529 in all 202 — the page was
    closed at its first ~2.5 KB every time. The same pairing scan over every
    tool found 446 such blocks that day, 202 `http_fetch` and 163
    `http_search`; the item's median 12,395 is over all 295 spilled fetches,
    Read-owning turns included, so it is not the figure these 202 rows yield.
    The `item_quoted_example` row is the one the item names: that session's own
    `messages` hold the fetch asked for at `max_chars: 20000` whose block ends
    in the unsatisfiable advice.
    """
    for denied in (["Read", "Bash"], ["mcp__lloyd-mcp__Read"]):
        for fetch in ("http_fetch", "mcp__lloyd-mcp__http_fetch"):
            block = _spill_block(denied, tool_name=fetch)
            for phrase in _KNOB_PHRASES:
                assert phrase not in block, (
                    f"{fetch} on {denied}: the block still names {phrase!r}, "
                    f"an argument that tool does not accept")
            # Clauses 1 and 4: the file is still named, and the sentence that
            # replaces the knob list says plainly that re-running gets you
            # nothing — and names the path it is the only copy of.
            assert "saved to:" in block, "the file itself is still named"
            assert "cannot recover the part past this preview" in block, (
                "no explicit no-route sentence was printed")
            assert block.count("call_spill.txt") == 2, (
                "the no-route sentence must name the saved path itself, not "
                "only lean on the header line that already does")


def test_a_spilled_fetch_points_at_read_and_says_so_when_the_turn_has_read():
    """The other branch of the same tool test: a chat turn owns `Read`, and
    for a fetch that is not merely the route it is the *only* one, so the
    sentence says that instead of offering a narrower query as well."""
    block = _spill_block(["Bash"], tool_name="http_fetch")
    assert "Read the full file with the Read tool" in block
    for phrase in _KNOB_PHRASES:
        assert phrase not in block, f"a fetch was told about {phrase!r} anyway"


def test_a_grep_or_graph_spill_keeps_the_narrowing_it_really_has():
    """#2067 must not undo #1066: for a tool that does take an argument
    selecting a smaller slice, "re-run it narrower" is the true advice, on a
    denied turn as much as on any other.

    Only tools the knob list actually names: `Grep` takes `--glob`, `--type`
    and `head_limit`, `graph_affected` takes `prefix`, `depth` and `limit`
    (`app/harness/tool_result_spill.py`, the comment above `HEAD_ONLY_TOOLS`).
    `mcp__lloyd-mcp__Grep` is here for the other half of that claim — the MCP
    spelling of a tool with the knobs must keep them too, so the bare-name cut
    cannot quietly widen the fix past the tools that need it.
    """
    for tool in ("Grep", "graph_affected", "mcp__lloyd-mcp__Grep"):
        block = _spill_block(["Read", "Bash"], tool_name=tool)
        assert "narrower query" in block, f"{tool} lost advice that is real for it"
        assert "Read the full file" not in block


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



# ---------------------------------------------------------------------------
# #1481 — observation stubs; idempotent re-relief. #1514 — the free route.
# ---------------------------------------------------------------------------


def _planted_pairs(n: int, first_line: str, chars: int = 6_000) -> list[dict]:
    """`n` Read pairs; result 0 opens with a distinctive first line."""
    msgs = _tool_pairs(n, chars=chars)
    body = first_line + "\n" + ("filler line for the planted result\n" * (chars // 35))
    msgs[1] = dict(msgs[1], content=body)
    return msgs


def test_a_cleared_result_leaves_an_observation_stub_with_its_first_line_verbatim(
        tmp_path, monkeypatch):
    """Clause 1: id + bounded verbatim prefix, never a summary."""
    from app.harness import microcompact as mc
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    first = "QUARTZ-HERON-2291 ships on relay port 7185 — do not paraphrase me"
    msgs = _planted_pairs(20, first)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    out, cleared = mc.microcompact(
        msgs, token_budget=1, estimate_fn=est, keep_recent_tools=5,
        session_id="sid1481", legacy_count_rule=False,
        observation_stubs=True, observation_head_chars=300)
    assert cleared == 15
    stub = _marker_text(out[1])
    assert stub.startswith(mc.OBSERVATION_STUB_PREFIX + "call_000 "), stub[:120]
    assert first in stub, "the first line must appear byte-for-byte"
    # Bounded: the head is at most its cap, and the stub is not the result.
    head = stub.split("verbatim:\n", 1)[1].split("\n… recall_observation", 1)[0]
    assert len(head) <= 300 and msgs[1]["content"].startswith(head)
    assert len(stub) < len(msgs[1]["content"]) / 5
    assert 'recall_observation(id="call_000")' in stub
    # The id resolves to the full content, persisted before the clear.
    saved = tmp_path / "sid1481.tool-results" / "call_000.txt"
    assert saved.read_text() == msgs[1]["content"]
    assert mc.observation_id_for_path(saved, "sid1481") == "call_000"
    assert mc.observation_id_for_path(saved, "another-session") == ""


def test_stubs_off_writes_todays_marker(tmp_path, monkeypatch):
    from app.harness import microcompact as mc
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    msgs = _planted_pairs(20, "FIRST LINE")
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    out, _ = mc.microcompact(msgs, token_budget=1, estimate_fn=est,
                             keep_recent_tools=5, session_id="s",
                             legacy_count_rule=False)
    text = _marker_text(out[1])
    assert not text.startswith(mc.OBSERVATION_STUB_PREFIX)
    assert "FIRST LINE" not in text and "Read that path if you need it again.]" in text
    assert "sessions" not in text.split("full content at")[0]


def test_a_second_relief_pass_changes_no_byte_and_counts_no_clear(tmp_path, monkeypatch):
    """Clause 2: re-running the pass over its own output is a no-op — for
    stubs, plain markers and reduced `<persisted-output>` blocks alike (the
    last one used to grow a duplicate `[preview dropped …]` line)."""
    from app.harness import microcompact as mc
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731

    msgs = _planted_pairs(20, "HEAD")
    pointer = (f"{PERSISTED_OUTPUT_TAG}\nOutput too large (54.7 KB, 56,034 chars). "
               f"Full output saved to: {tmp_path}/s.tool-results/call_001.txt\n\n"
               "Preview (first 2.0 KB):\n" + "p" * 2000 + "\n...\n"
               "Read the full file with the Read tool.\n</persisted-output>")
    msgs[3] = dict(msgs[3], content=pointer)
    for stubs in (False, True):
        once, n1 = mc.microcompact(msgs, token_budget=1, estimate_fn=est,
                                   keep_recent_tools=5, session_id="s",
                                   legacy_count_rule=False, observation_stubs=stubs)
        assert n1 == 15
        twice, n2 = mc.microcompact(once, token_budget=1, estimate_fn=est,
                                    keep_recent_tools=5, session_id="s",
                                    legacy_count_rule=False, observation_stubs=stubs)
        assert n2 == 0, f"stubs={stubs}: {n2} re-cleared"
        assert json.dumps(twice) == json.dumps(once)
        assert _marker_text(twice[3]).count("preview dropped") <= 1
    # The helper itself is idempotent too.
    reduced = mc._persisted_block_only(pointer)
    assert mc._persisted_block_only(reduced) == reduced


def test_a_pointer_block_becomes_a_stub_whose_head_is_the_preview(tmp_path, monkeypatch):
    """Through the real turn-start pass, `compaction.microcompact.
    observation_stubs` on: an old transcript pointer (D1) becomes a stub whose
    id is its spill file and whose head is its verbatim preview."""
    from app.config import CONFIG
    from app.harness import microcompact as mc
    p, sid, fulls = _pointer_session(tmp_path, monkeypatch, 20)
    comp = dict(CONFIG.get("compaction") or {})
    comp["microcompact"] = {**(comp.get("microcompact") or {}),
                            "observation_stubs": True, "observation_head_chars": 200}
    monkeypatch.setitem(CONFIG, "compaction", comp)
    out = _run(load_and_compact_session(p, model="qwen", mode_override="truncate"))
    tools = [_marker_text(m) for m in out["history"] if m.get("role") == "tool"]
    stubs = [t for t in tools if t.startswith(mc.OBSERVATION_STUB_PREFIX)]
    assert len(stubs) == 5 == out["microcompacted"]
    assert stubs[0].startswith(f"{mc.OBSERVATION_STUB_PREFIX}call_000 ")
    assert fulls[0][:150] in stubs[0]
    assert f"{len(fulls[0]):,} chars cleared" in stubs[0]
    assert "preview dropped" not in stubs[0]


def test_the_session_record_is_named_where_a_cleared_result_is_described(
        tmp_path, monkeypatch):
    """#1514: with `name_session_record` the marker names sessions/<sid>.json
    and the spill dir; off, it does not; a turn with Grep and Read denied is
    not offered the route."""
    from app.harness import microcompact as mc
    from app.harness.tool_result_spill import session_record_route
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    (tmp_path / "s.json").write_text("{}")
    route = session_record_route("s")
    assert f"{tmp_path}/s.json" in route and f"{tmp_path}/s.tool-results/" in route
    assert session_record_route("s", ["Grep", "Read"]) == ""
    assert "Read it" in session_record_route("s", ["Grep"])
    msgs = _planted_pairs(20, "HEAD")
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    on, _ = mc.microcompact(msgs, token_budget=1, estimate_fn=est, keep_recent_tools=5,
                            session_id="s", legacy_count_rule=False,
                            name_session_record=True)
    assert route in _marker_text(on[1])
    off, _ = mc.microcompact(msgs, token_budget=1, estimate_fn=est, keep_recent_tools=5,
                             session_id="s", legacy_count_rule=False)
    assert "s.json" not in _marker_text(off[1])


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


def test_turning_the_knob_on_is_what_the_shipped_default_is_not(tmp_path, monkeypatch):
    """#2168, the seam the last review called unverified: the knob's ON resolution.

    A default-off knob is only worth shipping if turning it on is a thing that
    happens. The clause-5 node proves the off half (the pre-pass never opens the
    sidecar); on its own that is satisfied by a read hard-wired to False — the shape
    whoever arrives after the owed deploy gate says "stay off" would leave behind, and
    it would keep every off-side assertion true while making the knob impossible to
    turn back on. So this node runs the SAME session, the SAME sidecar and the SAME
    counting spy twice, once per resolution, and requires the two to differ:

      knob False -> the spy is never called, the two named rows arrive with their
                    preview, the pass clears 5;
      knob True  -> the spy is called once with this session id, both rows arrive
                    reduced, the pass clears 3.

    What makes the node worth its run is that the two halves are COMPARED, not
    separately asserted: an implementation that behaves the same whatever the knob
    says cannot pass it, and no node that only looks at the off side can tell that
    implementation from a correct one. Measured, not argued — two mutations, run over
    the whole file, each turning 7 nodes red and leaving 52 green against 59 green
    clean:

      * the pre-pass's knob read at `app/compaction.py:667` replaced by a literal
        `False`, so both resolutions clear 5;
      * `apply_reduction_sidecar` returning its input untouched, so the read happens
        and changes nothing.

    Both break this node and the other six sidecar nodes, which is the honest result:
    they are two ways of producing the same broken behaviour, and a node that claimed
    to tell them apart would be claiming more than a comparison of clear counts can
    see. The one thing no off-only node could catch either way is here: the knob's
    `True` resolution reached the helper and reduced the ids it names.
    """
    from app.harness.microcompact import read_reduced_calls
    import app.harness.microcompact as mc_mod

    calls: list[str] = []
    real_apply = mc_mod.apply_reduction_sidecar

    def _counting(msgs, session_id):
        calls.append(session_id)
        return real_apply(msgs, session_id)

    monkeypatch.setattr(mc_mod, "apply_reduction_sidecar", _counting)
    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced(sid, ["call_000", "call_001"])
    handed = _handed_to_the_pass(monkeypatch)

    off = _run(load_and_compact_session(p, model="qwen", mode_override="truncate",
                                        microcompact_sidecar=False))
    assert calls == [], "the off resolution opened the sidecar"
    assert off["microcompacted"] == 5, off
    assert "file 0 line 0" in handed[0]["call_000"], handed[0]["call_000"][:200]

    on = _run(load_and_compact_session(p, model="qwen", mode_override="truncate",
                                       microcompact_sidecar=True))
    assert calls == [sid], (
        "the on resolution never reached the sidecar, so the deploy gate has nothing "
        "to measure and the knob is a comment")
    assert on["microcompacted"] == 3, on
    assert on["microcompacted"] < off["microcompacted"], (
        "the two resolutions cleared the same number of rows: the knob changes nothing")
    assert "file 0 line 0" not in handed[-1]["call_000"], handed[-1]["call_000"][:200]
    assert read_reduced_calls(sid) == ("call_000", "call_001"), (
        "reading through the knob must not consume what it read")


def test_a_second_turn_start_pass_books_no_re_clear_for_the_ids_the_sidecar_named(
        tmp_path, monkeypatch):
    """#2168, the seam the last review asked to see run twice: load → pass → load →
    pass over ONE persisted session, with the knob on.

    The item's harm is a repeat, so the witness has to be a repeat too. Clause 1 pins
    that a sidecar named by the previous turn arrives pre-reduced; that is one round
    trip. What the pass does on its SECOND read of the same file is what a session
    actually lives in, and it has to answer three things at once:

      * the second load still hands both named rows in reduced shape — the sidecar was
        not consumed by being read;
      * the second pass books the same 3 clears, none of them for `call_000`/`call_001`
        — no re-clear, which is the whole point of the sidecar;
      * the sidecar's own bytes are unchanged across both runs, so a read-time
        instrument cannot quietly become a writer and start owning state relief did
        not ask it to own.

    The falsifier is the same session and the same sidecar read with the knob off:
    5 clears, because the read that would have spared two of them never happened. Two
    mutations, each run over the whole file, turn this node red along with the other
    six sidecar nodes: the pre-pass's knob read at `app/compaction.py:667` replaced by
    a literal `False`, and `apply_reduction_sidecar` returning its input untouched.
    """
    from app.harness.microcompact import (
        read_reduced_calls, reduction_sidecar_path)
    import app.harness.microcompact as mc_mod

    p, sid, _ = _pointer_session(tmp_path, monkeypatch, 20)
    _record_reduced(sid, ["call_000", "call_001"])
    sidecar = reduction_sidecar_path(sid)
    before = sidecar.read_bytes()
    real_apply = mc_mod.apply_reduction_sidecar
    calls: list[str] = []

    def _counting(msgs, session_id):
        calls.append(session_id)
        return real_apply(msgs, session_id)

    monkeypatch.setattr(mc_mod, "apply_reduction_sidecar", _counting)
    handed = _handed_to_the_pass(monkeypatch)

    first = _run(load_and_compact_session(p, model="qwen", mode_override="truncate",
                                         microcompact_sidecar=True))
    second = _run(load_and_compact_session(p, model="qwen", mode_override="truncate",
                                          microcompact_sidecar=True))

    assert calls == [sid, sid], calls
    assert first["microcompacted"] == second["microcompacted"] == 3, (first, second)
    for run in (0, 1):
        both_reduced = all("preview dropped" in text
                           for text in (handed[run]["call_000"],
                                        handed[run]["call_001"]))
        assert both_reduced, (
            f"run {run + 1} did not receive the two named rows pre-reduced")
    assert sidecar.read_bytes() == before, (
        "the read-time pass rewrote the sidecar: an instrument that only reads has no "
        "business becoming this session's writer of reduction state")
    assert read_reduced_calls(sid) == ("call_000", "call_001")

    off = _run(load_and_compact_session(p, model="qwen", mode_override="truncate",
                                        microcompact_sidecar=False))
    assert off["microcompacted"] == 5, (
        "with nothing recorded the same session clears all five again, which is the "
        "number the sidecar exists to take off the next turn")
