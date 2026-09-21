#!/usr/bin/env python3
"""
One-time backfill: export Lloyd session JSONs to vault markdown for QMD indexing.

Converts ~/lloyd/sessions/*.json → ~/lloyd/_pipeline/vault-derived/sessions/{date}/*.md
for a conversation and → .../sessions-background/{date}/*.md for everything else,
skipping sessions that already have a markdown export.

**Classification is imported, not decided here.** The corpus this writes into is
the one `agent-services/scripts/qmd-watcher.sh` watches and *embeds*, so writing
a machine transcript here puts the machine talking to itself into the collection
that exists to answer questions about what the user and Lloyd discussed. The
split is therefore the same predicate the two listings and the live exporter use
— `app.sessions_io.is_conversation_session`, as used by
`app/post_capture.py::_export_session_markdown` — and the two directories come
from `app.paths` rather than from a private copy.

**It used to decide this itself, with one word, and leaked both ways.** The
guard here compared `data.get("platform")` against the single literal
`"autonomy"` and skipped only that, with the chat directory hard-coded as the
only destination. `worker` joined the non-user
platforms on 2026-09-07 and this path never heard about it, so worker
transcripts were embedded: measured over the exported corpus on 2026-09-17, 469
`worker` exports sat in the chat corpus, 446 of them dated after that widening.
(Retracting them is a deletion over live data and stays a human step — see
#1064. This change stops the file from writing more.) The other direction was
just as real: a `worker` or `autonomy` session with a chat-shaped name, or a
genuine conversation recorded under a platform the skip-list had never heard of,
was classified by a literal instead of by the rule.

The architecture page promises this cannot happen —
`architecture/background-runs.md`: "No component is allowed to restate this rule
... `tests/test_session_platform_checks.py` scans for literal `== "autonomy"`
comparisons on a platform field and fails on any hit". That held only because
the guard's scan roots were `("app", "agent_mcp", "workers")` and this file
lives in `scripts/`. #1064 clause 5 widens them.

Usage:
    python3 scripts/memory/backfill-session-markdown.py [--dry-run]
        [--sessions-dir DIR] [--chat-dir DIR] [--background-dir DIR]

The three directory flags exist so the command line above can be run against a
fixture. `tests/test_session_platform_checks.py` used to exercise this file by
`importlib`-ing it inside pytest, which skipped the two things most likely to be
broken about a script: the `sys.path` insertion that makes `import app` work when
the file is run BY PATH (the documented form puts `scripts/memory/` on the path,
not the repo root), and `main()` itself. The review rung named that gap on
2026-09-21. The defaults are the live directories, so an operator's run is
unchanged; only a test passes a flag.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _import_root_on_path() -> str:
    """Make `app.*` importable when this script is run by path.

    The documented invocation is `python3 scripts/memory/backfill-session-markdown.py`,
    which puts `scripts/memory/` on `sys.path` and not the repo root, so the
    `from app...` imports below would resolve only when pytest happened to have
    the checkout on the path. `scripts/extract-trajectories.py`,
    `scripts/groundskeeper/retention-sweep.py` and `scripts/entity_resolution.py`
    do the same insertion for the same reason.
    """
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


_import_root_on_path()

# The one definition of "is this a conversation", and the one definition of the
# two corpora. Both imported: a restated platform rule here is exactly how 469
# machine transcripts got embedded, and a private copy of a corpus path would
# write into a directory the watcher no longer watches.
from app.paths import VAULT_BACKGROUND_SESSIONS_DIR, VAULT_SESSIONS_DIR  # noqa: E402
from app.sessions_io import is_conversation_session  # noqa: E402

SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
PST = ZoneInfo("America/Los_Angeles")


def export_session(filepath: Path,
                   chat_dir: Path | None = None,
                   background_dir: Path | None = None,
                   ) -> tuple[str, bool, str]:
    """Export one session JSON to vault markdown. Returns (filename, success, reason).

    The two destination arguments exist so `main()` can honour
    `--chat-dir`/`--background-dir`, which is what lets the documented command line
    be run against a fixture; left unset they are the live corpora, read here
    rather than as parameter defaults so a caller that rebinds the module attribute
    gets the rebinding. WHICH of the two to use is never an argument — that is
    `is_conversation_session`'s answer, below.
    """
    chat_dir = VAULT_SESSIONS_DIR if chat_dir is None else chat_dir
    background_dir = (VAULT_BACKGROUND_SESSIONS_DIR if background_dir is None
                      else background_dir)
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        return filepath.name, False, f"JSON parse error: {e}"

    session_id = data.get("session_id", filepath.stem)
    # One predicate, given the name as well as the data. A four-part id is never
    # a conversation whatever its `platform` claims — that is the rule that kept
    # the `platform: mission-control` background runs out of the chat corpus
    # here, and the same one `/api/sessions` applies before it opens a file.
    # Nothing is skipped for being a background run any more: it is exported to
    # the corpus that is not embedded, which is what the live exporter does.
    conversation = is_conversation_session(session_id, data)
    created_at = data.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        dt = datetime.now()
    date_str = dt.astimezone(PST).strftime("%Y-%m-%d")

    # Check if already exported
    safe_id = session_id.replace("/", "--")[:30]
    out_dir = (chat_dir if conversation else background_dir) / date_str
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
    # `os.path.relpath`, not `relative_to`: the latter raises when the corpus is
    # not under home, which is every test that points these two directories at a
    # tmpdir. The reason string is display text for a counter, not a path any
    # caller parses.
    return filepath.name, True, f"→ {os.path.relpath(out_path, Path.home())}"


def main(argv=None):
    """Run the backfill over `--sessions-dir` into the two corpora.

    SEAM(process): `python3 scripts/memory/backfill-session-markdown.py
    --sessions-dir … --chat-dir … --background-dir …` -> a separate interpreter
    started by relative path from a repo root -> two corpus directories. It is run
    by hand, never imported, and the corpus each transcript lands in is decided at
    argparse time from `app.paths`; a test that only calls `export_session` proves
    the predicate and leaves the route from a flag to a directory unproven. Crossed
    for real by `tests/test_session_platform_checks.py::
    test_the_backfill_command_line_itself_runs_and_classifies`, which runs this
    command line as a subprocess against a fixture and then asserts which directory
    each transcript landed in.

    Defaults are the live directories, so `python3
    scripts/memory/backfill-session-markdown.py` does exactly what it did before
    the flags existed. The flags exist for the test that runs THIS command line
    against a fixture — see the module docstring for why importing `main` was not
    the same test.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be exported, write nothing")
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR,
                    help="where the session JSONs live")
    ap.add_argument("--chat-dir", type=Path, default=VAULT_SESSIONS_DIR,
                    help="corpus for conversations (the one qmd embeds)")
    ap.add_argument("--background-dir", type=Path, default=VAULT_BACKGROUND_SESSIONS_DIR,
                    help="corpus for everything that is not a conversation")
    args = ap.parse_args(argv)
    dry_run = args.dry_run

    files = sorted(args.sessions_dir.glob("*.json"))
    # Skip autonomy_ prefix files and old hermes migration files
    files = [f for f in files if not f.name.startswith("autonomy_")]

    print(f"Found {len(files)} session files to process")
    exported = 0
    skipped = 0
    errors = 0

    for f in files:
        name, success, reason = (
            (f.name, False, "dry-run") if dry_run
            else export_session(f, args.chat_dir, args.background_dir))
        if success:
            exported += 1
            print(f"  ✓ {name}: {reason}")
        else:
            if reason not in ("already exported", "dry-run", "no messages", "no content"):
                errors += 1
                print(f"  ✗ {name}: {reason}")
            else:
                skipped += 1

    print(f"\nDone: {exported} exported, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
