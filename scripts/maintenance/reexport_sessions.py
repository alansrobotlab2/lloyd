"""Re-export chat sessions whose qmd export stopped at turn 1.

Until 2026-11-10 a chat's markdown export was written once, on the first
capture pass, so the `sessions` qmd collection holds each chat's opening and
nothing after it. `app/post_capture.py` now re-exports on every pass; this
script repairs the exports written before that.

Only sessions with a surviving JSON are touched — an export whose JSON is gone
(every chat before the 2026-09-22 wipe) is the only copy left and is never
rewritten. Unchanged exports are not rewritten either (the exporter compares
first), so a re-run is a no-op.

    python -m scripts.maintenance.reexport_sessions            # dry run: counts
    python -m scripts.maintenance.reexport_sessions --apply    # write
"""
from __future__ import annotations

import argparse
import json
import sys

from app import post_capture
from app.paths import SESSIONS_DIR
from app.sessions_io import is_conversation_session


def _user_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("user: "))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the exports")
    args = ap.parse_args(argv)

    seen = short = rewritten = failed = 0
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        sid = path.stem
        try:
            data = json.loads(path.read_text())
        except Exception:
            failed += 1
            continue
        if not is_conversation_session(sid, data):
            continue
        seen += 1
        if not args.apply:
            # Render into memory by pointing the exporter at nothing: compare
            # the user-line count against the export already on disk.
            from datetime import datetime
            from zoneinfo import ZoneInfo
            try:
                dt = datetime.fromisoformat(data.get("created_at", ""))
            except Exception:
                continue
            date_str = dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
            out = (post_capture.VAULT_SESSIONS_DIR / date_str
                   / f"{sid.replace('/', '--')[:30]}.md")
            have = _user_lines(out.read_text()) if out.exists() else 0
            want = sum(1 for m in data.get("messages", []) if m.get("role") == "user")
            if have < want:
                short += 1
            continue
        try:
            out = post_capture._export_session_markdown(sid, data)
        except Exception as exc:
            print(f"{sid}: export failed: {exc}", file=sys.stderr)
            failed += 1
            continue
        if out is not None:
            rewritten += 1

    if args.apply:
        print(f"chats={seen} exported_or_unchanged={rewritten} failed={failed}")
    else:
        # `want` counts every user row, machine-injected ones included, so this
        # is an upper bound on the exports that would change.
        print(f"chats={seen} possibly_short={short} unreadable={failed} (dry run; --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
