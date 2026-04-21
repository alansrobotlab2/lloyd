#!/usr/bin/env python3
"""
Migrate ~/.hermes/sessions/ request dumps to ~/lloyd/sessions/ format.

Hermes stores one JSON file per API request (full conversation history at
that point). We group by session_id, take the file with the most messages,
and convert to Lloyd's session JSON format.

Usage:
    python3 scripts/migrate_hermes.py [--dry-run]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERMES_SESSIONS = Path.home() / ".hermes/sessions"
LLOYD_SESSIONS = Path(__file__).parent.parent / "sessions"


def parse_session_id(filename: str) -> tuple[str, str]:
    """Extract session_id and request timestamp from filename.

    Filename: request_dump_YYYYMMDD_HHMMSS_XXXXXX_YYYYMMDD_HHMMSS_XXXXXX.json
    Returns (session_id, request_ts_str)
    """
    name = filename.removeprefix("request_dump_").removesuffix(".json")
    parts = name.split("_")
    # Session ID = first 3 underscore-delimited tokens (YYYYMMDD_HHMMSS_XXXXXX)
    session_id = "_".join(parts[:3])
    req_ts = "_".join(parts[3:])
    return session_id, req_ts


def session_id_to_iso(session_id: str) -> str:
    """Convert '20260401_084104_c838f1' to '2026-04-01T08:41:04'."""
    try:
        return datetime.strptime(session_id[:15], "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return datetime.now().isoformat()


def short_id(role: str, index: int, text: str) -> str:
    return hashlib.md5(f"{role}{index}{text[:20]}".encode()).hexdigest()[:8]


def get_preview(raw_messages: list[dict]) -> str:
    for msg in raw_messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"][:60]
            elif isinstance(content, str):
                return content[:60]
    return ""


def convert_messages(raw_messages: list[dict], base_time: str) -> list[dict]:
    """Convert OpenAI-format messages to Lloyd format."""
    result = []
    for i, msg in enumerate(raw_messages):
        role = msg.get("role", "")
        if role == "system":
            continue

        raw_content = msg.get("content", "") or ""
        if isinstance(raw_content, list):
            text = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in raw_content
            )
        else:
            text = str(raw_content)

        tool_calls = msg.get("tool_calls")
        mid = short_id(role, i, text)

        if role == "assistant" and tool_calls:
            lloyd_msg: dict = {
                "id": mid + "_tc",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "call_id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in tool_calls
                ],
                "timestamp": base_time,
            }
            result.append(lloyd_msg)
        elif role == "tool":
            result.append({
                "id": mid + "_result",
                "role": "tool",
                "content": [{"type": "text", "text": text}],
                "tool_call_id": msg.get("tool_call_id", ""),
                "timestamp": base_time,
            })
        else:
            lloyd_msg = {
                "id": mid,
                "role": role,
                "content": [],
                "timestamp": base_time,
            }
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                lloyd_msg["content"].append({"type": "thinking", "thinking": reasoning})
            lloyd_msg["content"].append({"type": "text", "text": text})
            result.append(lloyd_msg)

    return result


def migrate(dry_run: bool = False) -> None:
    if not HERMES_SESSIONS.exists():
        print(f"Hermes sessions dir not found: {HERMES_SESSIONS}", file=sys.stderr)
        sys.exit(1)

    LLOYD_SESSIONS.mkdir(exist_ok=True)

    # Group dump files by session_id
    sessions: dict[str, list[Path]] = {}
    for f in HERMES_SESSIONS.glob("request_dump_*.json"):
        try:
            sid, _ = parse_session_id(f.name)
            sessions.setdefault(sid, []).append(f)
        except Exception:
            continue

    print(f"Found {len(sessions)} hermes sessions ({sum(len(v) for v in sessions.values())} dump files)")

    converted = skipped = errors = 0

    for session_id, files in sorted(sessions.items()):
        out_path = LLOYD_SESSIONS / f"{session_id}.json"
        if out_path.exists():
            skipped += 1
            continue

        # Pick the dump file with the most messages (most complete history)
        best_file: Path | None = None
        best_count = -1
        best_dump_ts = ""
        best_model = "primary"

        for f in files:
            try:
                data = json.loads(f.read_text())
                msgs = data.get("request", {}).get("body", {}).get("messages", [])
                if len(msgs) > best_count:
                    best_count = len(msgs)
                    best_file = f
                    best_dump_ts = data.get("timestamp", "")
                    best_model = data.get("request", {}).get("body", {}).get("model", best_model)
            except Exception:
                continue

        if best_file is None:
            errors += 1
            continue

        try:
            data = json.loads(best_file.read_text())
            raw_messages = data.get("request", {}).get("body", {}).get("messages", [])
            model = data.get("request", {}).get("body", {}).get("model", "primary")
            dump_ts = data.get("timestamp", "")

            created_at = session_id_to_iso(session_id)
            last_active = dump_ts or created_at
            messages = convert_messages(raw_messages, created_at)
            preview = get_preview(raw_messages)
            message_count = len([m for m in messages if m["role"] in ("user", "assistant")])

            session_doc = {
                "session_id": session_id,
                "model": model,
                "created_at": created_at,
                "last_active": last_active,
                "preview": preview[:60],
                "message_count": message_count,
                "platform": "hermes",
                "messages": messages,
            }

            if dry_run:
                print(f"  [dry-run] would write {out_path.name} ({message_count} msgs, preview: {preview[:40]!r})")
            else:
                out_path.write_text(json.dumps(session_doc, indent=2, ensure_ascii=False))
            converted += 1

        except Exception as e:
            print(f"  Error converting {session_id}: {e}")
            errors += 1

    print(f"Converted: {converted}  Skipped (exist): {skipped}  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate hermes sessions to lloyd format")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
