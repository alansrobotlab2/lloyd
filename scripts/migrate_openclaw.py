#!/usr/bin/env python3
"""
Migrate ~/.openclaw/agents/*/sessions/*.jsonl to ~/lloyd/sessions/ format.

OpenClaw uses JSONL event streams. We extract message events and convert
to Lloyd's session JSON format. All agents are migrated; the source agent
is preserved in the 'platform' field.

Usage:
    python3 scripts/migrate_openclaw.py [--dry-run] [--agent AGENT_NAME]
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

OPENCLAW_BASE = Path.home() / ".openclaw/agents"
LLOYD_SESSIONS = Path(__file__).parent.parent / "sessions"


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def ts_to_iso(ts) -> str:
    """Convert a timestamp (ms int, s float, or ISO string) to ISO string."""
    if isinstance(ts, (int, float)):
        # OpenClaw stores ms epoch integers
        seconds = ts / 1000 if ts > 1e10 else ts
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    s = str(ts).replace("Z", "").replace("+00:00", "")
    return s


def make_lloyd_session_id(uuid: str, iso_ts: str) -> str:
    """Convert openclaw UUID + ISO timestamp to Lloyd's YYYYMMDD_HHMMSS_XXXXXX format."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        date_part = dt.strftime("%Y%m%d_%H%M%S")
        short = uuid.replace("-", "")[:6]
        return f"{date_part}_{short}"
    except Exception:
        return "openclaw_" + uuid.replace("-", "")[:12]


# ── Content conversion ────────────────────────────────────────────────────────

def convert_content(content) -> tuple[list[dict], list[dict]]:
    """Convert openclaw content blocks to (lloyd_content_blocks, tool_calls).

    Returns:
        lloyd_content: list of {type, text/thinking} blocks
        tool_calls: list of Lloyd-format tool_call dicts (from toolCall blocks)
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}], []

    lloyd_content = []
    tool_calls = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            lloyd_content.append({"type": "text", "text": block.get("text", "")})

        elif btype == "thinking":
            lloyd_content.append({"type": "thinking", "thinking": block.get("thinking", "")})

        elif btype == "toolCall":
            args = block.get("arguments", {})
            tool_calls.append({
                "id": block.get("id", ""),
                "call_id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })

        # toolResult blocks are emitted as separate tool-role messages by openclaw
        # in a follow-up message event; skip them here.

    if not lloyd_content:
        lloyd_content = [{"type": "text", "text": ""}]

    return lloyd_content, tool_calls


def extract_preview(messages: list[dict]) -> str:
    """Extract a clean preview from the first real user message."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if block.get("type") != "text":
                continue
            text = block["text"]

            # Strip XML-style injected blocks (<daily_notes>, <active_mode>, etc.)
            clean = re.sub(r'<[a-zA-Z_][^>]*>[\s\S]*?</[a-zA-Z_][^>]*>', '', text).strip()
            # Strip sender metadata code blocks (```json ... ```)
            clean = re.sub(r'Sender \(untrusted metadata\):\s*```[\s\S]*?```', '', clean).strip()
            # Strip bracketed timestamp/boilerplate lines like [Sun 2026-03-29 12:41 PDT] ...
            clean = re.sub(r'\[[^\]]{8,50}\][^\n]*\n?', '', clean).strip()
            # Strip remaining leading metadata lines (bare JSON, dashes, etc.)
            lines = [l for l in clean.splitlines() if l.strip() and not l.strip().startswith('{') and not l.strip() == '}']
            if lines:
                return lines[0][:60]

    return ""


# ── Per-file migration ────────────────────────────────────────────────────────

def migrate_jsonl(path: Path, agent_name: str) -> dict | None:
    """Parse a single openclaw JSONL file and return a Lloyd session dict, or None."""
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as e:
        print(f"  Read error {path.name}: {e}")
        return None

    session_event = next((e for e in events if e.get("type") == "session"), None)
    if not session_event:
        return None

    # For .jsonl.deleted.* files, stem is still the UUID portion before .jsonl
    raw_stem = path.name.split(".jsonl")[0]
    session_uuid = session_event.get("id", raw_stem)
    session_ts = ts_to_iso(session_event.get("timestamp", ""))
    lloyd_id = make_lloyd_session_id(session_uuid, session_ts)

    # Detect model (prefer model_change event, fall back to first assistant message)
    model = "unknown"
    for e in events:
        if e.get("type") == "model_change":
            model = e.get("modelId", model)
            break
    if model == "unknown":
        for e in events:
            if e.get("type") == "message":
                m = e.get("message", {})
                if m.get("model"):
                    model = m["model"]
                    break

    messages = []
    last_ts = session_ts

    for event in events:
        if event.get("type") != "message":
            continue

        msg = event.get("message", {})
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        raw_ts = msg.get("timestamp") or event.get("timestamp", session_ts)
        ts = ts_to_iso(raw_ts)
        last_ts = ts

        lloyd_content, tool_calls = convert_content(msg.get("content", []))
        mid = (event.get("id") or "")[:8] or session_uuid[:8]

        lloyd_msg: dict = {
            "id": mid + ("_tc" if tool_calls else ""),
            "role": role,
            "content": lloyd_content,
            "timestamp": ts,
        }
        if tool_calls:
            lloyd_msg["tool_calls"] = tool_calls

        messages.append(lloyd_msg)

    if not messages:
        return None

    preview = extract_preview(messages)
    message_count = len([m for m in messages if m["role"] in ("user", "assistant")])

    return {
        "session_id": lloyd_id,
        "model": model,
        "created_at": session_ts,
        "last_active": last_ts,
        "preview": preview,
        "message_count": message_count,
        "platform": f"openclaw/{agent_name}",
        "openclaw_uuid": session_uuid,
        "messages": messages,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def migrate(dry_run: bool = False, agent_filter: str | None = None, include_deleted: bool = False) -> None:
    if not OPENCLAW_BASE.exists():
        print(f"OpenClaw agents dir not found: {OPENCLAW_BASE}", file=sys.stderr)
        sys.exit(1)

    LLOYD_SESSIONS.mkdir(exist_ok=True)

    total = converted = skipped = errors = empty = deleted_skipped = 0

    for agent_dir in sorted(OPENCLAW_BASE.iterdir()):
        if not agent_dir.is_dir():
            continue
        if agent_filter and agent_dir.name != agent_filter:
            continue
        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.exists():
            continue

        agent_name = agent_dir.name
        jsonl_files = sorted(sessions_dir.glob("*.jsonl"))

        if include_deleted:
            # Also pick up *.jsonl.deleted.* files (soft-deleted sessions)
            deleted_files = sorted(sessions_dir.glob("*.jsonl.deleted.*"))
            jsonl_files = jsonl_files + deleted_files
        else:
            n_deleted = len(list(sessions_dir.glob("*.jsonl.deleted.*")))
            if n_deleted:
                deleted_skipped += n_deleted

        if not jsonl_files:
            continue

        print(f"Agent '{agent_name}': {len(jsonl_files)} sessions")
        total += len(jsonl_files)

        for jsonl_path in jsonl_files:
            try:
                session = migrate_jsonl(jsonl_path, agent_name)
            except Exception as e:
                print(f"  Error {jsonl_path.name}: {e}")
                errors += 1
                continue

            if session is None:
                empty += 1
                continue

            out_path = LLOYD_SESSIONS / f"{session['session_id']}.json"

            # Check if already migrated (same openclaw_uuid in existing file)
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text())
                    if existing.get("openclaw_uuid") == session["openclaw_uuid"]:
                        skipped += 1
                        continue
                except Exception:
                    pass
                # Different session mapped to same timestamp ID — use UUID suffix
                out_path = LLOYD_SESSIONS / f"{session['session_id']}_{session['openclaw_uuid'][:6]}.json"
                if out_path.exists():
                    skipped += 1
                    continue

            if dry_run:
                print(f"  [dry-run] {out_path.name}  msgs={session['message_count']}  {session['preview'][:40]!r}")
            else:
                out_path.write_text(json.dumps(session, indent=2, ensure_ascii=False))
            converted += 1

    print(f"\nTotal JSONL files : {total}")
    print(f"Converted         : {converted}")
    print(f"Skipped (exist)   : {skipped}")
    print(f"Empty/no-messages : {empty}")
    print(f"Errors            : {errors}")
    if deleted_skipped:
        print(f"Deleted (skipped) : {deleted_skipped}  (re-run with --include-deleted to migrate these)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate openclaw sessions to lloyd format")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--agent", metavar="NAME", help="Migrate only this agent (e.g. main)")
    parser.add_argument("--include-deleted", action="store_true", help="Also migrate .jsonl.deleted.* files")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run, agent_filter=args.agent, include_deleted=args.include_deleted)
