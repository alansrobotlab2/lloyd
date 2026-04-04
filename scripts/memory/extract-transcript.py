#!/usr/bin/env python3
"""
Extract clean user+assistant transcript from Hermes session JSON files.
Uses a watermark (state.json) to only output new content since last run.
Exits with empty output if nothing new.

Output format (stdout):
---
SESSION: <session-id> | <timestamp>
[HH:MM] USER: <text>
[HH:MM] ASSISTANT: <text>
---
"""

import json
import os
import sys
import glob
from datetime import datetime, timezone

SESSIONS_DIR = os.path.expanduser("~/.hermes/sessions")
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "state.json")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"lastRunTs": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def format_time(ts_str):
    """Format timestamp to HH:MM."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%H:%M")
    except (ValueError, AttributeError):
        return "??"


def process_session_file(filepath, last_run_ts):
    """
    Process a single Hermes session JSON file.
    Returns list of (timestamp, role, text) tuples.
    Also returns session_id and session_ts.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None, []

    session_id = data.get("session_id", "unknown")
    session_ts = data.get("session_start", "")

    # Use file mtime to check if session is newer than watermark
    file_mtime = os.path.getmtime(filepath)
    if file_mtime <= last_run_ts and last_run_ts > 0:
        return session_id, session_ts, []

    entries = []
    for msg in data.get("messages", []):
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text blocks from structured content
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        parts.append(text)
            text = "\n".join(parts)
        elif isinstance(content, str):
            text = content.strip()
        else:
            continue

        if not text:
            continue

        # Skip very short assistant messages
        if role == "assistant" and len(text) < 10:
            continue

        # Skip system-injected user messages
        if role == "user":
            stripped = text.strip()
            if stripped.startswith("[cron:") or stripped.startswith("[System Message]"):
                continue

        # Hermes sessions don't have per-message timestamps; use session_ts
        entries.append((session_ts, role, text))

    return session_id, session_ts, entries


def main():
    dry_run = '--dry-run' in sys.argv
    state = load_state()
    last_run_ts = state.get("lastRunTs", 0)
    now_ts = datetime.now(timezone.utc).timestamp()

    # Find all Hermes session files
    pattern = os.path.join(SESSIONS_DIR, "session_*.json")
    all_files = glob.glob(pattern)

    if not all_files:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    # Filter to files modified after last run
    if last_run_ts > 0:
        recent_files = [f for f in all_files if os.path.getmtime(f) > last_run_ts]
    else:
        recent_files = all_files

    if not recent_files:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    # Process each file
    output_blocks = []
    for filepath in sorted(recent_files):
        session_id, session_ts, entries = process_session_file(filepath, last_run_ts)
        if not entries:
            continue

        lines = [f"SESSION: {session_id or 'unknown'} | {session_ts or 'unknown'}"]
        for ts, role, text in entries:
            time_str = format_time(ts)
            role_label = "USER" if role == "user" else "ASSISTANT"
            # Truncate very long messages
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"
            lines.append(f"[{time_str}] {role_label}: {text}")
        output_blocks.append("\n".join(lines))

    if not output_blocks:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    # Output the transcript with 50KB safety cap
    separator = "\n---\n"
    full_output = separator.join(output_blocks)
    output_bytes = full_output.encode("utf-8")
    if len(output_bytes) > 51200:
        print("[extract-transcript] WARNING: output truncated to 50KB", file=sys.stderr)
        full_output = "[WARNING: transcript truncated to 50KB — oldest content dropped]\n" + output_bytes[-51200:].decode("utf-8", errors="replace")
    print(full_output)

    # Update watermark only after successful output
    if not dry_run:
        save_state({"lastRunTs": now_ts})


if __name__ == "__main__":
    main()
