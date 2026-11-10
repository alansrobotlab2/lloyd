"""The per-turn half of the prompt layout (P1): what rides on the user message.

`prompt_builder.build_system_prompt` decides what the system prompt holds;
this decides what the turn's user message carries at its tail instead. Three
things can land there, and all exist to keep the system prompt — and so the
whole previous conversation behind it — byte-stable from one turn to the next:

  * the `<now>` clock line when `harness.prompt_layout.clock` is on;
  * the `<session_state>` block (goal, plan, todos) when
    `harness.prompt_layout.session_state` is `user_tail`;
  * the `<memory_delta>` note when `harness.prompt_layout.freeze_memory` is on
    and the memory files moved since the session's snapshot
    (`app/memory_snapshot.py`).

None of the three is a property of this module, and no current value is
stated here: what ships is `harness.prompt_layout` in `config.yaml`. The
`<session_state>` part is empty whenever the placement switch is anything but
`user_tail` (the block rides inside the system prompt instead), the
`<memory_delta>` part whenever `freeze_memory` is off (no snapshot is taken,
so nothing has diverged to report), and the clock whenever `clock` is off. With all three off `turn_tail` returns "" and every caller's
`prefetched_text` is exactly what it was.

The four hand-built turn paths (stream, ambient, sync in `messages.py`;
`voice.py`) call `turn_tail` + `append_turn_tail`, and `_run_turn` records the
tail in the turn's `subliminal` row (`_messages_subliminal._split_subliminal`)
so the chat UI still shows everything the model saw.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("lloyd-server")

#: Tags a turn tail can open with. `_split_subliminal` recognises the tail by
#: these, so a user message that happens to end in its own text is not split.
#: `<now>` is first: the tail opens with it when the clock is on, and the
#: splitter takes the first tag it finds as the start of the whole tail.
TAIL_TAGS = ("<now>", "<session_state>", "<memory_delta>")
TAIL_SEP = "\n\n"


def _cfg() -> dict:
    try:
        from app.config import CONFIG

        return dict((CONFIG.get("harness") or {}).get("prompt_layout") or {})
    except Exception:  # noqa: BLE001
        return {}


def replay_injected_context() -> bool:
    """`harness.prompt_layout.replay_injected_context` (default false)."""
    return bool(_cfg().get("replay_injected_context", False))


def clock_enabled() -> bool:
    """`harness.prompt_layout.clock` (default false in code; config.yaml says)."""
    return bool(_cfg().get("clock", False))


def clock_line(now: float | None = None) -> str:
    """`<now>Tue 2026-11-10 11:05 PST</now>`, or "" when it cannot be stamped.

    The system prompt carries no clock on purpose (a timestamp there would bust
    the cached prefix every turn), and until 2026-11-10 nothing else did on a
    chat or voice turn — only ambient envelopes were stamped — so "yesterday",
    "this week" or a deadline was composed in prose (#1197's failure). The tail
    rides on the current user message only; past user rows replay bare, so the
    stamp never enters the cached history.
    """
    import time as _time
    from datetime import datetime

    from app.sessions_io import ambient_clock_stamp

    t = _time.time() if now is None else now
    stamp = ambient_clock_stamp(t)
    if not stamp:
        return ""
    day = datetime.fromtimestamp(t).astimezone().strftime("%a")
    return f"<now>{day} {stamp}</now>"


def turn_tail(todos: list[dict] | None = None, plan: dict | None = None,
              goal: dict | None = None, memory_note: str = "") -> str:
    """The text a turn appends to its user message, or "". Never raises."""
    parts: list[str] = []
    try:
        if clock_enabled():
            line = clock_line()
            if line:
                parts.append(line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt_layout: clock line failed: %s", exc)
    try:
        from app import prompt_builder as pb

        if pb.session_state_layout() == "user_tail":
            block = pb.build_session_state_block(todos, plan, goal)
            if block:
                parts.append(block)
    except Exception as exc:  # noqa: BLE001 — a lost block is today's prompt
        logger.warning("prompt_layout: session state block failed: %s", exc)
    if memory_note:
        parts.append(memory_note)
    return TAIL_SEP.join(parts)


def append_turn_tail(prefetched_text: str, tail: str) -> str:
    if not tail:
        return prefetched_text
    return f"{prefetched_text}{TAIL_SEP}{tail}"


_CLOCK_RE = re.compile(r"(?:\n\n)?<now>[^<\n]*</now>")


def without_clock(text: str) -> str:
    """`text` with the `<now>` line removed: the clock moves every minute, so a
    digest of the injected block that kept it (the component manifest, #581)
    would read as "the retrieval changed" on every turn."""
    return _CLOCK_RE.sub("", text or "")


def mem_kwargs(memories_text: str | None) -> dict:
    """`build_system_prompt` kwargs for a frozen memory body.

    Empty when nothing is frozen, so the call is the one it always was — and
    a stub standing in for `build_system_prompt` in a test that predates the
    kwarg is not handed an argument it does not take.
    """
    return {} if memories_text is None else {"memories_text": memories_text}


def replay_injected(convo: list[dict], messages: list[dict]) -> list[dict]:
    """Re-join each user row with its turn's subliminal row.

    When `replay_injected_context` is on, a past user turn is replayed as what
    the model was actually sent — the `<context>` prefix, the text, and any
    turn tail — rather than the bare text. A behaviour choice, not a cache
    one: either way the history is byte-stable across turns. Rows without a
    `turn_id`, or with no subliminal row, are left alone. Never raises.
    """
    try:
        by_turn: dict[str, dict] = {}
        for m in messages:
            if m.get("role") != "subliminal":
                continue
            tid = (m.get("subliminal") or {}).get("turn_id")
            if tid:
                by_turn[tid] = m
        if not by_turn:
            return convo
        out: list[dict] = []
        for m in convo:
            sub = by_turn.get(m.get("turn_id") or "") if m.get("role") == "user" else None
            if sub is None:
                out.append(m)
                continue
            out.append({**m, "content": [{"type": "text",
                                          "text": _rejoin(m, sub)}]})
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt_layout: replay_injected failed: %s", exc)
        return convo


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join((b.get("text") or "") for b in content if isinstance(b, dict))
    return ""


def _rejoin(user_row: dict, sub_row: dict) -> str:
    text = _text_of(user_row.get("content"))
    injected = _text_of(sub_row.get("content"))
    meta = sub_row.get("subliminal") or {}
    tail_chars = int(meta.get("tail_chars") or 0)
    tail = injected[-tail_chars:] if tail_chars else ""
    prefix = injected[:-tail_chars] if tail_chars else injected
    if tail_chars and prefix.endswith(TAIL_SEP):
        prefix = prefix[: -len(TAIL_SEP)]
    if meta.get("kind") == "ambient_envelope":
        # The envelope already embeds the producer text.
        return append_turn_tail(prefix, tail)
    return TAIL_SEP.join(p for p in (prefix, text, tail) if p)
