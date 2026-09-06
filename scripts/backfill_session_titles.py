#!/usr/bin/env python
"""Backfill titles onto sessions that predate `app/session_titles.py`.

New sessions get titled off turn completion, so the chat history fills in
on its own — but only for conversations that see another turn. Everything
already in the list would stay nameless forever without this.

Runs against the same secondary slot the live path uses, and that slot is
single-tenant: every title here is a call agent turns, voice summaries and
post-session capture queue behind. So it is a script you run deliberately,
newest-first, with a limit and a delay — not something wired into boot.

    .venvs/lloyd/bin/python scripts/backfill_session_titles.py --limit 50
    .venvs/lloyd/bin/python scripts/backfill_session_titles.py --dry-run

Idempotent: a session that already has a title is skipped, so an
interrupted run resumes by being run again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import session_titles as st  # noqa: E402
from app.paths import SESSIONS_DIR  # noqa: E402


def _candidates(limit: int) -> list[Path]:
    """Newest-first sessions that are due for a title.

    Newest-first because the chat history list shows the 50 most recent —
    those are the rows a person is actually trying to read.
    """
    if not SESSIONS_DIR.exists():
        return []
    out: list[Path] = []
    for path in sorted(
        SESSIONS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if len(out) >= limit:
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if st.should_title(data):
            out.append(path)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50,
                    help="how many sessions to title (newest first)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between calls, to leave the secondary "
                         "responsive to live work")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be titled, call no model")
    args = ap.parse_args()

    paths = _candidates(args.limit)
    if not paths:
        print("nothing to title")
        return 0

    print(f"{len(paths)} session(s) due for a title")
    titled = skipped = 0
    for i, path in enumerate(paths, 1):
        session_id = path.stem
        if args.dry_run:
            data = json.loads(path.read_text(encoding="utf-8"))
            preview = (data.get("preview") or "")[:60]
            print(f"[{i}/{len(paths)}] {session_id}  {preview}")
            continue

        # The chat history is ordered by when a conversation was last
        # active, and writing a title rewrites the file. Restore the mtime
        # so a backfill can't reshuffle the list even if something
        # downstream still sorts on it.
        before = path.stat()
        started = time.monotonic()
        title = await st.maybe_title_session(session_id)
        elapsed = time.monotonic() - started
        try:
            os.utime(path, (before.st_atime, before.st_mtime))
        except OSError:
            pass
        if title:
            titled += 1
            print(f"[{i}/{len(paths)}] {session_id}  {elapsed:5.1f}s  {title}")
        else:
            skipped += 1
            print(f"[{i}/{len(paths)}] {session_id}  {elapsed:5.1f}s  (no usable title)")
        if i < len(paths):
            await asyncio.sleep(args.delay)

    if not args.dry_run:
        print(f"\ntitled {titled}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
