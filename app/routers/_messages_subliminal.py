"""Subliminal-injection capture (#306).

Three ephemeral injection sites send text to the harness that the session
JSON never sees:

  1. `prefetch_context()` `<context>` block prepended to user message
  2. `build_ambient_turn()` `<ambient ...>...</ambient>` envelope wrapping
     producer text
  3. 20-turn `<system-reminder>` memory-preservation nudge

These helpers extract the injected prefix from `prefetched_text` vs the
original `text` so we can persist it as a `role="subliminal"` entry for
UI visibility. Pulled out of `messages.py` to keep the router slim.
"""

from __future__ import annotations

from app.sessions_io import SessionTurn


# Ordered classifiers: first match wins. `memory_nudge` check precedes
# `prefetch` so a nudge+prefetch combo surfaces as "memory_nudge" (the
# more notable framing) when it leads the prefix.
_SUBLIMINAL_KINDS = (
    ("memory_nudge",     "<system-reminder>"),
    ("ambient_envelope", "<ambient "),
    ("prefetch",         "<context>"),
)

# Tag → source-name map for the summary badge. Ordering matches rendering
# order in `prefetch._format_context()`.
_SUBLIMINAL_SOURCE_TAGS = (
    ("ambient",  "<ambient-signals>"),
    ("skills",   "<skill "),
    ("backlog",  "<backlog-refs>"),
    ("facts",    "<facts>"),
    ("vault",    "<vault-context>"),
    ("sessions", "<recent-sessions>"),
    ("hint",     "<skill-hint>"),
)


def _extract_subliminal_prefix(prefetched_text: str, text: str) -> str:
    """Return the injected-only portion of `prefetched_text`, or "" if none.

    Two shapes are handled:
      - Prefetch/nudge path: `prefetched_text` ends with "\\n\\n" + text
        (see prefetch.prefetch_context and the memory-nudge branch).
      - Ambient envelope path: text is embedded inside an <ambient> wrapper
        (see build_ambient_turn). The whole prefetched_text is "injection".
    If `prefetched_text == text` no injection happened.
    """
    if prefetched_text == text:
        return ""
    suffix = "\n\n" + text
    if prefetched_text.endswith(suffix):
        prefix = prefetched_text[: -len(suffix)]
        return prefix if prefix.strip() else ""
    # Ambient envelope (or any other shape where text is not a clean suffix)
    return prefetched_text


def _classify_subliminal(prefix: str) -> str:
    """Return 'prefetch' | 'ambient_envelope' | 'memory_nudge' | 'other'."""
    lead = prefix.lstrip()
    for kind, marker in _SUBLIMINAL_KINDS:
        if lead.startswith(marker):
            return kind
    return "other"


def _detect_subliminal_sources(prefix: str) -> list[str]:
    """Return the list of detected source-sections in this injection."""
    return [name for name, marker in _SUBLIMINAL_SOURCE_TAGS if marker in prefix]


def _build_subliminal_entry(turn: SessionTurn, prefix: str, timestamp: str) -> dict:
    """Shape the subliminal message entry. Kept pure for testability."""
    return {
        "id": f"subl_{turn.turn_id}",
        "role": "subliminal",
        "content": [{"type": "text", "text": prefix}],
        "timestamp": timestamp,
        "subliminal": {
            "kind":     _classify_subliminal(prefix),
            "sources":  _detect_subliminal_sources(prefix),
            "chars":    len(prefix),
            "turn_id":  turn.turn_id,
        },
    }
