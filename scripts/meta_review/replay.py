#!/usr/bin/env python3
"""Inner Voice (#345) — pretty-print a session's event log.

Reads `~/lloyd/event_logs/<session_id>.events.jsonl` and renders each
event chronologically with terminal coloring + abbreviated payloads.
Used for ad-hoc forensics: "what did Brain 1 do during this session?"

Usage:
    python -m scripts.meta_review.replay <session_id>
    python scripts/meta_review/replay.py <session_id>
    python scripts/meta_review/replay.py <session_id> --expand-blobs
    python scripts/meta_review/replay.py <session_id> --filter brain1.tool_call
    python scripts/meta_review/replay.py <session_id> --raw    # raw JSON

Stage 0 ships this minimally. Later stages can grow filters, sequence
diagrams, intervention traces, etc. — but the core requirement is just
"can you read your own log."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/meta_review/replay.py` from any cwd. We resolve
# the lloyd repo root from this file's location so the script doesn't
# rely on an installed package.
_THIS = Path(__file__).resolve()
_LLOYD_ROOT = _THIS.parent.parent.parent
EVENT_LOGS_DIR = _LLOYD_ROOT / "event_logs"
BLOBS_DIR = EVENT_LOGS_DIR / "blobs"


# ── Terminal coloring (ANSI). Plain by default if not a TTY. ──

class _Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _coloring_enabled() -> bool:
    return sys.stdout.isatty()


def _c(s: str, code: str) -> str:
    if not _coloring_enabled():
        return s
    return f"{code}{s}{_Color.RESET}"


# Rough event → color map. Keep this aesthetic, not semantic — colors
# help the eye scan, they aren't the ground truth.
_EVENT_COLORS: dict[str, str] = {
    "brain1.user_prompt_received":   _Color.BLUE,
    "brain1.options_built":          _Color.GRAY,
    "brain1.query_started":          _Color.GRAY,
    "brain1.stream_event":           _Color.GRAY,
    "brain1.thinking_block_emitted": _Color.MAGENTA,
    "brain1.tool_call_proposed":     _Color.CYAN,
    "brain1.tool_result_received":   _Color.CYAN,
    "brain1.result_message":         _Color.GREEN,
    "inner_voice.persona_invoked":         _Color.YELLOW,
    "inner_voice.persona_response_raw":    _Color.YELLOW,
    "inner_voice.persona_response_parsed": _Color.YELLOW,
    "inner_voice.persona_parse_failure":   _Color.RED,
    "inner_voice.aggregation_decision":    _Color.YELLOW,
    "inner_voice.intervention_dispatched": _Color.RED,
    "inner_voice.cancel_event_fired":      _Color.RED,
    "inner_voice.consensus_termination_proposal": _Color.YELLOW,
    "inner_voice.consensus_termination_decision": _Color.YELLOW,
    "inner_voice.pre_tool_use_evaluated":  _Color.CYAN,
}


def _resolve_blob(sha: str) -> str | None:
    path = BLOBS_DIR / f"{sha}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        if "$blob" in value and isinstance(value["$blob"], str):
            content = _resolve_blob(value["$blob"])
            return content if content is not None else f"<missing blob {value['$blob'][:8]}>"
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _abbreviate(value: Any, max_len: int = 200) -> str:
    """Render a JSON value into a one-line preview, truncating long strings."""
    s = json.dumps(value, ensure_ascii=False, default=str)
    if len(s) > max_len:
        return s[:max_len] + f" …(+{len(s) - max_len} chars)"
    return s


def _format_event(obj: dict[str, Any], expand_blobs: bool) -> str:
    ts = obj.get("ts", "?")
    ev = obj.get("event", "?")
    turn_id = obj.get("turn_id")
    data = obj.get("data") or {}
    if expand_blobs:
        data = _expand(data)

    color = _EVENT_COLORS.get(ev, _Color.RESET)
    head = f"{_c(ts, _Color.DIM)}  {_c(ev, color + _Color.BOLD)}"
    if turn_id:
        head += _c(f"  turn={turn_id}", _Color.DIM)

    body = _abbreviate(data, max_len=240)
    return f"{head}\n  {_c(body, _Color.GRAY if not _coloring_enabled() else color)}"


def _iter_events(session_id: str) -> list[dict[str, Any]]:
    path = EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
    if not path.exists():
        print(f"error: event log not found at {path}", file=sys.stderr)
        sys.exit(2)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {line_no}: {e}", file=sys.stderr)
    return out


def _event_persona(event: dict[str, Any]) -> str | None:
    """Return the persona name an event is associated with, or None.

    Inner-voice events keep `persona` as a top-level data field; we look
    there first, then fall back to a few alternates we've seen in practice.
    """
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    for k in ("persona", "persona_name", "agent"):
        v = data.get(k)
        if isinstance(v, str):
            return v
    # `inner_voice.ensemble_selected` carries `personas: [...]`. Treat the
    # event as belonging to each of those — return the first one (the
    # caller's --persona filter does substring contain anyway).
    personas = data.get("personas")
    if isinstance(personas, list) and personas:
        first = personas[0]
        if isinstance(first, str):
            return first
    return None


def _event_belongs_to_turn(event: dict[str, Any], turn_id: str) -> bool:
    """Match `--turn <id>` against an event. Accepts a prefix match so the
    user can pass the leading 8 chars without quoting the full UUID.
    """
    ev_turn = event.get("turn_id")
    if not isinstance(ev_turn, str):
        return False
    return ev_turn == turn_id or ev_turn.startswith(turn_id) or turn_id.startswith(ev_turn)


def _print_summary(events: list[dict[str, Any]]) -> None:
    """One-line-per-event-name count, ordered by frequency. Useful when
    skimming a long log to see what fired and how often.
    """
    from collections import Counter
    counts = Counter(e.get("event", "?") for e in events)
    print(_c(f"# event summary ({len(events)} total)", _Color.BOLD))
    for ev, n in counts.most_common():
        color = _EVENT_COLORS.get(ev, _Color.RESET)
        print(f"  {_c(f'{n:>6}', _Color.DIM)}  {_c(ev, color)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session_id", help="Session id; events.jsonl must exist for it")
    p.add_argument("--filter", default=None,
                   help="Substring filter on event name (e.g. 'tool_call')")
    p.add_argument("--persona", default=None,
                   help="Filter to events tagged with this persona (e.g. 'completion_checker')")
    p.add_argument("--turn", default=None,
                   help="Filter to a specific turn id (prefix match accepted)")
    p.add_argument("--expand-blobs", action="store_true",
                   help="Resolve $blob refs to inline strings")
    p.add_argument("--raw", action="store_true",
                   help="Dump raw JSON (one event per line) instead of pretty-print")
    p.add_argument("--summary", action="store_true",
                   help="Print event-name counts instead of the full event stream")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap to first N events (0 = all)")
    args = p.parse_args(argv)

    events = _iter_events(args.session_id)
    if args.filter:
        events = [e for e in events if args.filter in e.get("event", "")]
    if args.persona:
        # Substring match on persona name — `--persona completion` matches
        # `completion_checker` etc.
        events = [
            e for e in events
            if (_event_persona(e) or "").lower().find(args.persona.lower()) >= 0
        ]
    if args.turn:
        events = [e for e in events if _event_belongs_to_turn(e, args.turn)]
    if args.limit > 0:
        events = events[: args.limit]

    if args.summary:
        _print_summary(events)
        return 0

    if args.raw:
        for e in events:
            if args.expand_blobs and isinstance(e.get("data"), (dict, list)):
                e["data"] = _expand(e["data"])
            print(json.dumps(e, ensure_ascii=False, default=str))
        return 0

    title = f"# {args.session_id}  ({len(events)} events"
    qualifiers: list[str] = []
    if args.filter:  qualifiers.append(f"filter={args.filter!r}")
    if args.persona: qualifiers.append(f"persona={args.persona!r}")
    if args.turn:    qualifiers.append(f"turn={args.turn!r}")
    if qualifiers:
        title += f"; {', '.join(qualifiers)}"
    title += ")"
    print(_c(title, _Color.BOLD))
    print(_c(f"# {EVENT_LOGS_DIR / (args.session_id + '.events.jsonl')}", _Color.DIM))
    print()
    for e in events:
        print(_format_event(e, expand_blobs=args.expand_blobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
