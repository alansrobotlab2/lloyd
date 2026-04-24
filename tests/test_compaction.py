"""Tests for app/compaction.py — client-side conversation compaction.

Run: .venvs/lloyd/bin/python -m tests.test_compaction
"""
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
    format_conversation_for_local,
    is_local_model,
    load_and_compact_session,
    truncate_conversation,
    truncation_threshold,
)


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
# Local/cloud detection
# ---------------------------------------------------------------------------


def test_is_local_model_cloud_vs_local():
    assert is_local_model("") is False
    assert is_local_model("https://api.anthropic.com") is False
    assert is_local_model("https://api2.anthropic.com/v1") is False
    assert is_local_model("http://127.0.0.1:8096") is True
    assert is_local_model("http://localhost:8080") is True
    assert is_local_model("https://my-custom-llm.example.com") is True


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
# Chat-template formatting
# ---------------------------------------------------------------------------


def test_format_qwen_chatml_has_correct_markers():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = format_conversation_for_local(history, current_user_text="how are you?", model="qwen-30b")

    # ChatML markers
    assert "<|im_start|>user\nhi<|im_end|>" in out
    assert "<|im_start|>assistant\nhello<|im_end|>" in out
    assert "<|im_start|>user\nhow are you?<|im_end|>" in out
    # Ends with generation cue (no closing marker after it)
    assert out.endswith("<|im_start|>assistant\n")


def test_format_picks_template_by_model_name():
    history = [{"role": "user", "content": "test"}]

    qwen_out = format_conversation_for_local(history, current_user_text="q", model="qwen-coder")
    assert "<|im_start|>" in qwen_out
    assert "<|start_header_id|>" not in qwen_out

    llama_out = format_conversation_for_local(history, current_user_text="q", model="llama-3.3-70b")
    assert "<|start_header_id|>" in llama_out
    assert "<|eot_id|>" in llama_out


def test_format_strips_trailing_user_when_current_provided():
    # Persisted history ends with a user turn (just-appended by messages.py).
    # Caller passes prefetched current_user_text — the trailing persisted
    # copy should be dropped so we don't duplicate.
    history = [
        {"role": "user", "content": "old msg"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "latest msg"},
    ]
    out = format_conversation_for_local(history, current_user_text="latest msg with prefetch", model="qwen")

    # The persisted "latest msg" should NOT appear — only the prefetched version.
    assert "latest msg with prefetch" in out
    assert out.count("latest msg") == 1, f"expected exactly one copy of user msg, got:\n{out}"


def test_format_structured_content_extracts_text():
    # Structured content on a non-trailing user turn should survive
    # template formatting (text extracted from the content blocks).
    history = [
        {"role": "user", "content": [{"type": "text", "text": "structured"}]},
        {"role": "assistant", "content": "reply"},
    ]
    out = format_conversation_for_local(history, current_user_text="new", model="qwen")
    assert "structured" in out, f"expected 'structured' in output, got:\n{out}"


# ---------------------------------------------------------------------------
# load_and_compact_session (session-level helper)
# ---------------------------------------------------------------------------


def _write_session(path: Path, messages: list[dict]) -> None:
    path.write_text(json.dumps({"messages": messages}))


def test_load_and_compact_missing_file():
    out = load_and_compact_session(Path("/nonexistent/path.json"), model="qwen")
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
        out = load_and_compact_session(p, model="qwen")
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
        out = load_and_compact_session(p, model="qwen-unknown")  # unknown → default window
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
        out = load_and_compact_session(str(p), model="qwen")
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
        out = load_and_compact_session(p, model="qwen")
        roles = [m["role"] for m in out["history"]]
        assert "subliminal" not in roles
        assert "user" in roles
        assert "assistant" in roles


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

_TESTS = [
    test_estimate_tokens_returns_int,
    test_estimate_tokens_empty_string,
    test_estimate_tokens_proportional,
    test_estimate_conversation_no_double_count,
    test_is_local_model_cloud_vs_local,
    test_truncate_under_threshold_is_noop,
    test_truncate_drops_old_turns_above_threshold,
    test_truncate_keeps_last_turn_minimum,
    test_format_qwen_chatml_has_correct_markers,
    test_format_picks_template_by_model_name,
    test_format_strips_trailing_user_when_current_provided,
    test_format_structured_content_extracts_text,
    test_load_and_compact_missing_file,
    test_load_and_compact_no_truncation_needed,
    test_load_and_compact_triggers_truncation,
    test_load_and_compact_accepts_str_path,
    test_load_and_compact_filters_ui_only_roles,
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
