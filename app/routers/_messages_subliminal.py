"""Subliminal-injection capture (#306).

Three ephemeral injection sites send text to the harness that the session
JSON never sees:

  1. `prefetch_context()` `<context>` block prepended to user message
  2. `build_ambient_turn()` `<ambient ...>...</ambient>` envelope wrapping
     producer text
  3. 20-turn `<system-reminder>` memory-preservation nudge
  4. P1's turn tail (`<session_state>`, `<memory_delta>`) appended after
     the text (`app/prompt_layout.py`)

These helpers extract the injected prefix from `prefetched_text` vs the
original `text` so we can persist it as a `role="subliminal"` entry for
UI visibility. Pulled out of `messages.py` to keep the router slim.
"""

from __future__ import annotations

import re

from app.sessions_io import SessionTurn


# Ordered classifiers: first match wins. `memory_nudge` check precedes
# `prefetch` so a nudge+prefetch combo surfaces as "memory_nudge" (the
# more notable framing) when it leads the prefix.
_SUBLIMINAL_KINDS = (
    ("memory_nudge",     "<system-reminder>"),
    ("ambient_envelope", "<ambient "),
    ("prefetch",         "<context>"),
    ("session_state",    "<session_state>"),
    ("memory_delta",     "<memory_delta>"),
)

# Tag → source-name map for the summary badge. Ordering matches rendering
# order in `prefetch._format_context()`. A section that is missing from here is
# a part of the turn Mission Control never shows — that is how
# `<next-session-notes>` shipped invisible (#3081) — so
# `tests/test_prefetch.py::test_every_rendered_context_section_has_a_badge`
# now fails the suite on it, against `prefetch.CONTEXT_SECTION_TAGS`.
# `state` and `memory_delta` are the deliberate extras: they ride the P1 turn
# tail (`app/prompt_layout.py`), so they are rendered by nobody in
# `_format_context` and the guard does not expect them.
_SUBLIMINAL_SOURCE_TAGS = (
    ("ambient",  "<ambient-signals>"),
    ("notes",    "<next-session-notes>"),
    ("skills",   "<skill "),
    ("backlog",  "<backlog-refs>"),
    ("facts",    "<facts>"),
    ("vault",    "<vault-context>"),
    ("sessions", "<recent-sessions>"),
    ("hint",     "<skill-hint>"),
    ("ide",      "<ide_state>"),
    ("state",    "<session_state>"),
    ("memory_delta", "<memory_delta>"),
)


def _extract_subliminal_prefix(prefetched_text: str, text: str) -> str:
    """Return the injected-only portion of `prefetched_text`, or "" if none.

    Two shapes are handled:
      - Prefetch/nudge path: `prefetched_text` ends with a newline
        separator + text (see prefetch.prefetch_context and the
        memory-nudge branch). "\\n\\n" is the norm; a single "\\n" is
        accepted too so a nudge that fired on a turn with no <context>
        block doesn't get the user's own text swept into the entry.
      - Ambient envelope path: text is embedded inside an <ambient> wrapper
        (see build_ambient_turn). The whole prefetched_text is "injection".
    If `prefetched_text == text` no injection happened.
    """
    if prefetched_text == text:
        return ""
    for sep in ("\n\n", "\n"):
        suffix = sep + text
        if prefetched_text.endswith(suffix):
            prefix = prefetched_text[: -len(suffix)]
            return prefix if prefix.strip() else ""
    # Ambient envelope (or any other shape where text is not a clean suffix)
    return prefetched_text


def _split_subliminal(prefetched_text: str, text: str) -> tuple[str, str]:
    """`(prefix, tail)`: what was injected before and after the user's text.

    P1 can append a turn tail (`<session_state>`, `<memory_delta>`; see
    `app/prompt_layout.py`) after the text, which `_extract_subliminal_prefix`
    alone would read as the ambient shape and record the user's own words as
    injection. The tail is recognised by its opening tag, so it is split off
    only when it really is one. With no tail this is exactly
    `(_extract_subliminal_prefix(...), "")`.
    """
    from app.prompt_layout import TAIL_SEP, TAIL_TAGS

    body, tail = prefetched_text, ""
    for tag in TAIL_TAGS:
        marker = TAIL_SEP + tag
        idx = prefetched_text.find(marker)
        while idx != -1:
            head = prefetched_text[:idx]
            if head == text or head.endswith("\n" + text):
                body, tail = head, prefetched_text[idx + len(TAIL_SEP):]
                break
            idx = prefetched_text.find(marker, idx + 1)
        if tail:
            break
    if not tail:
        return _extract_subliminal_prefix(prefetched_text, text), ""
    return _extract_subliminal_prefix(body, text), tail


def _classify_subliminal(prefix: str) -> str:
    """Return 'prefetch' | 'ambient_envelope' | 'memory_nudge' | 'other'."""
    lead = prefix.lstrip()
    for kind, marker in _SUBLIMINAL_KINDS:
        if lead.startswith(marker):
            return kind
    return "other"


def _detect_subliminal_sources(prefix: str) -> list[str]:
    """Return the source-sections this injection actually carried, in order.

    A badge is claimed only when the section's marker OPENS a line of `prefix`
    (`(?m)^`), not merely when its characters appear somewhere in it. Every
    section this names is emitted at column 0 — `prefetch._format_context` ends
    with `"<context>\\n" + "\\n".join(parts) + "\\n</context>"`
    (`app/prefetch.py:1387`) and the `<ide_state>` splice lands on text ending in
    a newline (`app/prefetch.py:1712-1720`) — so a line-start test costs no real
    section, while a bare `marker in prefix` charged the row for any tag a
    producer merely QUOTED. Read back 2026-11-01 across every persisted session
    transcript under `~/lloyd-data/sessions` (`*.json` + `*.json.gz`: 19,211 files,
    15,544 subliminal rows — a growing corpus, so count it again rather than carry
    it): 75 of those rows lose a badge under this rule, 183 badges in all (`facts`
    52, `ambient` 45, `hint` 37, `sessions` 33, `notes` 14, `skills` 2), and for
    every one of the 183 the marker appears at NO line start of that row's own text.
    Shapes: a `<vault-context>` bullet whose text is about this very badge list
    quoting `<next-session-notes>` in backticks, and a prose line enumerating
    "`<facts>`, `<ambient-signals>`" as tag names. Same strip #3081 landed to make
    truthful.

    The property bought is "this marker opens a line", NOT "this section was
    rendered": prose that quotes a tag at column 0 — a fenced code block in an
    injected skill body, say — still mints that badge. It cannot be tightened
    further from here: `prefetch._section_spans` builds its regex FROM
    `CONTEXT_SECTION_TAGS` and so cannot see the two turn-tail extras `state` and
    `memory_delta` (`tests/test_prefetch.py::test_the_renderer_cannot_emit_a_section_the_tuple_does_not_name`).
    """
    return [name for name, marker in _SUBLIMINAL_SOURCE_TAGS
            if re.search(r"(?m)^" + re.escape(marker), prefix)]


def _subliminal_text(prefix: str, tail: str = "") -> str:
    """The row's text: the prefix, then the tail, as the model read them."""
    if not tail:
        return prefix
    return f"{prefix}\n\n{tail}" if prefix else tail


def _build_subliminal_entry(turn: SessionTurn, prefix: str, timestamp: str,
                            tail: str = "") -> dict:
    """Shape the subliminal message entry. Kept pure for testability.

    `tail` (P1) is what rode after the user's text; it is shown after the
    prefix so the row holds everything the model saw, and its length is
    recorded (`tail_chars`, omitted when empty) so a replay can put the two
    halves back on either side of the text.
    """
    shown = _subliminal_text(prefix, tail)
    meta = {
        "kind":     _classify_subliminal(prefix or tail),
        "sources":  _detect_subliminal_sources(shown),
        "chars":    len(shown),
        "turn_id":  turn.turn_id,
    }
    if tail:
        meta["tail_chars"] = len(tail)
    return {
        "id": f"subl_{turn.turn_id}",
        "role": "subliminal",
        "content": [{"type": "text", "text": shown}],
        "timestamp": timestamp,
        "subliminal": meta,
    }
