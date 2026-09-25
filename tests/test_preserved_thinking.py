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

One exception, from backlog #1510: a trace
`app.thinking_fidelity.flag_fabricated_reasoning` flags is not carried,
because what preserved thinking replays is whatever the last iteration
emitted, and the primary sometimes emits reasoning about a request nobody
made — which would then come back as the model's own prior thought for
`preserve_thinking_iterations: 6` more iterations. The tests below pin the
exception at the message-builder level; the same pair driven through the
whole agent loop, including the audit row a withheld trace still leaves
behind, is in `app/harness/tests/test_preserve_thinking.py`.
"""

import logging

from app.harness.loop import _assistant_message_for_history, _prune_reasoning

#: A trace of the shape #1510 is about, on a turn that asked for arithmetic. Four
#: of the six markers are in it; `no actual task` comes first so the marker named
#: by the withhold log line is deterministic.
FABRICATED = (
    "The user's message has no actual task in it — just a system reminder. My "
    "previous turn had no substantive reasoning. Now they ask me to reproduce my "
    "complete previous thinking using the audit tool."
)


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


# ── clause 3 at the message builder: withhold the fabricated prior thought ──
#
# Everything above pins the mechanism; the three below pin its one exception. The
# asymmetry is deliberate: preserving an ordinary trace is worth real tokens and
# real coherence, so the exception has to be narrow, and it has to be *visible*
# when it fires. The engine-side A/B that would say whether MTP speculative
# decoding contributes to these traces is an operator's (`MTP_ENABLED=0` needs an
# `agent-llm-primary` restart); withholding on the fidelity check is what could
# be done here without touching an engine.


def test_a_fabricated_trace_is_not_carried_into_history():
    """Neither spelling, because either one alone reaches a template.

    The two keys are how preserved thinking survives across the two engines, so a
    withhold that cleared one and left the other would leak the trace to whichever
    engine reads the survivor — the same silent asymmetry the rest of this file
    exists to prevent, running in the other direction.
    """
    msg = _assistant_message_for_history(
        text="17*23 is 391.", tool_calls=[], reasoning=FABRICATED)

    assert msg["content"] == "17*23 is 391.", msg
    assert "reasoning" not in msg, msg
    assert "reasoning_content" not in msg, msg


def test_the_withhold_says_which_marker_fired_and_in_what_session(caplog):
    """An invisible withhold is indistinguishable from reasoning that never arrived.

    This is the only surface that knows a trace was *dropped* rather than absent,
    and the question an operator asks next — how often, on which marker — is
    answerable from this line and nowhere else, precisely because the transcript
    keeps the trace and the request does not show it.
    """
    with caplog.at_level(logging.INFO, logger="lloyd-harness-loop"):
        _assistant_message_for_history(
            text="ok", tool_calls=[], reasoning=FABRICATED,
            session_id="20260925_001902_iv029b", iteration=2)

    lines = [r.getMessage() for r in caplog.records
             if "withheld a fabricated reasoning trace" in r.getMessage()]
    assert len(lines) == 1, caplog.text
    line = lines[0]
    assert "marker 'no actual task'" in line, line
    assert "20260925_001902_iv029b" in line, line
    assert "iteration=2" in line, line
    assert 'the role="thinking" row still records it' in line, line


def test_an_unflagged_trace_is_built_exactly_as_before():
    """Nothing else about the message moves: same keys, same values, same text.

    Asserted as whole-dict equality rather than key lookups so that a new field —
    a marker, a flag, a rewritten `content` — fails here instead of passing
    quietly. That is what "without changing what unflagged preserved thinking does
    today" has to mean to be checkable. The tool-call path alongside reasoning is
    `test_tool_calls_survive_alongside_reasoning` above.
    """
    thought = "split 17*23 as 17*20 + 17*3"

    msg = _assistant_message_for_history(
        text="answer", tool_calls=[], reasoning=thought)

    assert msg == {
        "role": "assistant",
        "content": "answer",
        "reasoning": thought,
        "reasoning_content": thought,
    }, msg
