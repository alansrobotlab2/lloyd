"""Preserved thinking across the two engines behind the harness.

Assistant turns carry their own reasoning back into history so the model
does not re-derive its conclusions every iteration. The mechanism fails
*silently* when it fails — the model simply sees prior turns in which it
apparently thought nothing — so the shape of the message is worth pinning.

The subtlety is that the two engines disagree about the field name:

* vLLM 0.28 accepts both spellings on the wire but only renders the
  template from `reasoning`.
* llama.cpp applies the model's jinja template directly, and Qwen3.6's
  reads `message.reasoning_content`.

The secondary slot runs llama.cpp (Qwen3.6-35B-A3B GGUF) and the primary
runs vLLM, so a message carrying only one spelling is preserved on one
engine and quietly dropped on the other.
"""

from app.harness.loop import _assistant_message_for_history, _prune_reasoning


def _assistant(reasoning: str = "thought"):
    return _assistant_message_for_history(
        text="answer", tool_calls=[], reasoning=reasoning
    )


def test_reasoning_is_sent_under_both_spellings():
    msg = _assistant("deriving the fix")
    assert msg["reasoning"] == "deriving the fix"
    assert msg["reasoning_content"] == "deriving the fix"


def test_no_reasoning_field_when_there_was_no_reasoning():
    msg = _assistant_message_for_history(text="hi", tool_calls=[], reasoning="")
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg


def test_tool_calls_survive_alongside_reasoning():
    msg = _assistant_message_for_history(
        text="",
        tool_calls=[{"id": "c1", "function": {"name": "Bash", "arguments": "{}"}}],
        reasoning="picking a tool",
    )
    assert msg["content"] is None
    assert msg["tool_calls"][0]["id"] == "c1"
    assert msg["reasoning_content"] == "picking a tool"


def test_prune_keeps_only_the_recent_window():
    msgs = [_assistant(f"r{i}") for i in range(5)]
    _prune_reasoning(msgs, keep=2)
    assert [("reasoning" in m) for m in msgs] == [False, False, False, True, True]


def test_prune_drops_both_spellings_together():
    """Dropping only `reasoning` would leave the full text on the wire for
    llama.cpp, so the token bound would not actually bound anything."""
    msgs = [_assistant(f"r{i}") for i in range(4)]
    _prune_reasoning(msgs, keep=1)
    for msg in msgs[:-1]:
        assert "reasoning" not in msg
        assert "reasoning_content" not in msg
    assert msgs[-1]["reasoning"] == "r3"
    assert msgs[-1]["reasoning_content"] == "r3"


def test_prune_counts_messages_carrying_only_reasoning_content():
    """History rebuilt by an older build (or a llama.cpp-shaped session)
    can carry `reasoning_content` alone. Skipping those would make them
    invisible to the window count and immune to pruning."""
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": f"r{i}"}
            for i in range(3)]
    _prune_reasoning(msgs, keep=1)
    assert "reasoning_content" not in msgs[0]
    assert "reasoning_content" not in msgs[1]
    assert msgs[2]["reasoning_content"] == "r2"


def test_prune_ignores_user_and_tool_messages():
    msgs = [
        {"role": "user", "content": "q"},
        _assistant("r0"),
        {"role": "tool", "content": "out", "tool_call_id": "c1"},
        _assistant("r1"),
    ]
    _prune_reasoning(msgs, keep=1)
    assert "reasoning" not in msgs[1]
    assert msgs[3]["reasoning"] == "r1"
    assert msgs[0] == {"role": "user", "content": "q"}
