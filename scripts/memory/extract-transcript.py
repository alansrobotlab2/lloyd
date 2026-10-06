#!/usr/bin/env python3
"""
Extract clean user+assistant transcript from Lloyd session JSON files.
Uses a watermark (app.paths.MEMORY_CAPTURE_STATE_PATH, under DATA_ROOT) to only
output new content since last run. Exits with empty output if nothing new.

Reads ~/lloyd-data/sessions/*.json

Output format (stdout):
---
SESSION: <session-id> | <timestamp>
[HH:MM] USER: <text>
[HH:MM] ASSISTANT: <text>
---
"""

import json
import os
import sys; print("SCRIPT START", file=sys.stderr)
import sys
import glob
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from app.paths import MEMORY_CAPTURE_STATE_PATH, SESSIONS_DIR  # noqa: E402

LLOYD_SESSIONS_DIR = str(SESSIONS_DIR)
# The watermark is runtime state, so it hangs off DATA_ROOT like every other store
# (#2294). It used to be `os.path.join(os.path.dirname(__file__), "state.json")`,
# which `app.paths` resolves nothing about and no checkout tracks: `load_state` then
# took its `FileNotFoundError` branch forever, `lastRunTs` stayed 0, and every pass
# re-read the entire sessions corpus (the 2026-09-22 data move, `6426668b`, rewrote
# this file's SESSIONS path and left this line). The second half of the bug was where
# a write would have landed: `.gitignore` un-ignores `scripts/**/*.json`, so the first
# real pass made untracked-but-visible dirt in the production checkout, which a
# round's `git add -A` reaper commits under the round's own subject. Under DATA_ROOT
# the file is outside the repo, so both halves stop, and two checkouts (a worktree, a
# scratch `LLOYD_DATA`) stop colliding on one machine-wide file.
STATE_FILE = str(MEMORY_CAPTURE_STATE_PATH)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"lastRunTs": 0}


def save_state(state):
    # Importing `app.paths` deliberately creates nothing under DATA_ROOT (#712), and
    # this is a standalone script that never calls `ensure_dirs()`, so the directory
    # holding the watermark is this writer's to make. Without it a first pass against
    # a root that does not exist yet — a fresh `LLOYD_DATA`, a worktree's
    # `.lloyd-data/` — raised FileNotFoundError out of `open(…, "w")` and the
    # watermark was never written, which is the same silent-zero failure this file
    # just moved to fix, reached a different way.
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
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


def _extract_text_from_content(content) -> str:
    """Extract plain text from content (string or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def process_lloyd_session(filepath, last_run_ts):
    """
    Process a Lloyd session JSON file (~/lloyd-data/sessions/*.json).
    Returns (session_id, session_ts, entries).
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None, []

    session_id = data.get("session_id", "unknown")
    session_ts = data.get("created_at", "")

    file_mtime = os.path.getmtime(filepath)
    if file_mtime <= last_run_ts and last_run_ts > 0:
        return session_id, session_ts, []

    entries = []
    for msg in data.get("messages", []):
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        text = _extract_text_from_content(msg.get("content", ""))
        if not text:
            continue

        # Skip short assistant messages
        if role == "assistant" and len(text) < 10:
            continue

        # Skip injected system context in user messages
        if role == "user":
            stripped = text.strip()
            if stripped.startswith("[cron:") or stripped.startswith("[System Message]"):
                continue
            # Skip large context injections (daily notes, system context blocks)
            if stripped.startswith("<daily_notes>") or stripped.startswith("<memory>"):
                continue

        msg_ts = msg.get("timestamp", session_ts)
        entries.append((msg_ts, role, text))

    return session_id, session_ts, entries


def main():
    import sys
    print("TEST: main() called", file=sys.stderr)
    dry_run = '--dry-run' in sys.argv
    sessions_dir = LLOYD_SESSIONS_DIR
    pattern = os.path.join(sessions_dir, "*.json")
    process_fn = process_lloyd_session

    state = load_state()
    last_run_ts = state.get("lastRunTs", 0)
    now_ts = datetime.now(timezone.utc).timestamp()

    all_files = glob.glob(pattern)
    if not all_files:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    recent_files = [f for f in all_files if os.path.getmtime(f) > last_run_ts] if last_run_ts > 0 else all_files

    if not recent_files:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    output_blocks = []
    print(f"DEBUG: Processing {len(recent_files)} recent files", file=sys.stderr)
    for filepath in sorted(recent_files):
        session_id, session_ts, entries = process_lloyd_session(filepath, last_run_ts)
        print(f"DEBUG: {os.path.basename(filepath)} -> {len(entries)} entries", file=sys.stderr)
        if not entries:
            continue

        lines = [f"SESSION: {session_id or 'unknown'} | {session_ts or 'unknown'}"]
        for ts, role, text in entries:
            time_str = format_time(ts)
            role_label = "USER" if role == "user" else "ASSISTANT"
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"
            lines.append(f"[{time_str}] {role_label}: {text}")
        output_blocks.append("\n".join(lines))

    if not output_blocks:
        if not dry_run:
            save_state({"lastRunTs": now_ts})
        sys.exit(0)

    separator = "\n---\n"
    full_output = separator.join(output_blocks)
    output_bytes = full_output.encode("utf-8")
    if len(output_bytes) > 51200:
        print("[extract-transcript] WARNING: output truncated to 50KB", file=sys.stderr)
        full_output = "[WARNING: transcript truncated to 50KB — oldest content dropped]\n" + output_bytes[-51200:].decode("utf-8", errors="replace")
    print(full_output)

    if not dry_run:
        save_state({"lastRunTs": now_ts})


if __name__ == "__main__":
    main()
