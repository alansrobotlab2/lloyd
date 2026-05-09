#!/usr/bin/env python3
"""
One-time backfill: export Lloyd session JSONs to vault markdown for QMD indexing.

Converts ~/lloyd/sessions/*.json → ~/lloyd/_pipeline/vault-derived/sessions/{date}/*.md
Skips autonomy sessions and sessions that already have a markdown export.

Usage:
    python3 scripts/memory/backfill-session-markdown.py [--dry-run]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
VAULT_SESSIONS_DIR = Path.home() / "lloyd" / "_pipeline" / "vault-derived" / "sessions"
PST = ZoneInfo("America/Los_Angeles")


def export_session(filepath: Path) -> tuple[str, bool, str]:
    """Export one session JSON to vault markdown. Returns (filename, success, reason)."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        return filepath.name, False, f"JSON parse error: {e}"

    # Skip autonomy sessions
    if data.get("platform") == "autonomy":
        return filepath.name, False, "autonomy session"

    session_id = data.get("session_id", filepath.stem)
    created_at = data.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        dt = datetime.now()
    date_str = dt.astimezone(PST).strftime("%Y-%m-%d")

    # Check if already exported
    safe_id = session_id.replace("/", "--")[:30]
    out_dir = VAULT_SESSIONS_DIR / date_str
    out_path = out_dir / f"{safe_id}.md"
    if out_path.exists():
        return filepath.name, False, "already exported"

    messages = data.get("messages", [])
    if not messages:
        return filepath.name, False, "no messages"

    # Build markdown
    lines = []
    lines.append(f"# {session_id}")
    lines.append(f"# {dt.isoformat()}")
    model = data.get("model", "")
    if model:
        lines.append(f"# model: {model}")
    lines.append("")

    content_lines = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "\n".join(t for t in text_parts if t)
            tool_uses = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
        elif isinstance(content, str):
            text = content
            tool_uses = []
        else:
            continue

        if role == "user":
            stripped = text.strip()
            if any(stripped.startswith(p) for p in (
                "<context>", "<system-reminder>", "<memory>", "<daily_notes>",
                "[cron:", "[System Message]", "[autonomy:",
            )):
                continue
            if not stripped or len(stripped) < 2:
                continue
            display = stripped[:600] if len(stripped) > 600 else stripped
            lines.append(f"user: {display}")
            content_lines += 1

        elif role == "assistant":
            for tu in tool_uses:
                name = tu.get("name", "?")
                args = tu.get("input", {})
                arg_parts = []
                for k, v in (args.items() if isinstance(args, dict) else []):
                    if isinstance(v, str):
                        arg_parts.append(f"{k}={v[:200]}")
                    elif isinstance(v, (bool, int, float)):
                        arg_parts.append(f"{k}={v}")
                    else:
                        arg_parts.append(f"{k}=...")
                lines.append(f"tool_call: {name}({', '.join(arg_parts)})")
                content_lines += 1

            if text.strip() and len(text.strip()) > 10:
                display = text.strip()[:500]
                lines.append(f"lloyd: {display}")
                content_lines += 1

        elif role == "tool":
            result_text = text.strip()[:300] if text else "(empty)"
            is_error = msg.get("is_error", False)
            status = "ERROR" if is_error else "OK"
            lines.append(f"  → [{status}] {result_text}")

    if content_lines == 0:
        return filepath.name, False, "no content"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return filepath.name, True, f"→ {out_path.relative_to(Path.home())}"


def main():
    dry_run = "--dry-run" in sys.argv

    files = sorted(SESSIONS_DIR.glob("*.json"))
    # Skip autonomy_ prefix files and old hermes migration files
    files = [f for f in files if not f.name.startswith("autonomy_")]

    print(f"Found {len(files)} session files to process")
    exported = 0
    skipped = 0
    errors = 0

    for f in files:
        name, success, reason = export_session(f) if not dry_run else (f.name, False, "dry-run")
        if success:
            exported += 1
            print(f"  ✓ {name}: {reason}")
        else:
            if reason not in ("autonomy session", "already exported", "dry-run", "no messages", "no content"):
                errors += 1
                print(f"  ✗ {name}: {reason}")
            else:
                skipped += 1

    print(f"\nDone: {exported} exported, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
