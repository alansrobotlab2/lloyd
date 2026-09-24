#!/usr/bin/env python3
""""Where did this land?" — does a session that wrote a vault note name it at the end?

Backlog #1035 (split from #567). Three hand-rolled classifiers over one window
reported 8, 12 and 14 failures because each used its own path regex, and #571
had already said of a sibling metric that a number which moves with the proxy
is not a gate. This is the one classifier, pinned by
`tests/test_artifact_named_in_final_message.py`:

  * **Scope** — a session counts only if a `Write`/`Edit`/`vault_write` target
    is a `.md` under `<vault>/knowledge/` or `<vault>/projects/`. `skills/`,
    `backlog/`, `autonomy/`, `lloyd/` and `memory/` writes are bookkeeping, not
    a deliverable anybody asks the whereabouts of.
  * **Pass** — the session's last assistant text names an obsidian-absolute
    path (`~/obsidian/…`, `/home/<user>/obsidian/…`) or a dir-qualified one
    (`knowledge/…`, `projects/…`) to a `.md` that exists on disk. Filenames may
    contain spaces: two legacy notes under `knowledge/youtube/` do, and a
    whitespace-excluding character class failed sessions that had reported
    them correctly.
  * **Excluded, not failed** — a session whose last assistant row is empty or
    is the harness's no-summary placeholder (`app/routers/messages.py`,
    `synthetic_empty_terminal`). That is #832's class; counting it here would
    charge #832's bug to this metric.

The window is the last `--days` (7) anchored on the newest session mtime, not
on the clock, so two runs over the same tree print the same numbers. Offline
read of logs; nothing under `app/` or `agent_mcp/` imports it.

    .venvs/lloyd/bin/python -m eval.artifact_named_in_final_message [--json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.paths import VAULT_ROOT, production_data_root  # noqa: E402

DEFAULT_DAYS = 7
# The live box's transcripts whichever tree this runs from; a worktree's data
# root has none. Same choice as `eval/tool_failure_baseline.py`.
DEFAULT_SESSIONS_DIR = production_data_root() / "sessions"

WRITE_TOOLS = ("Write", "Edit", "vault_write")
SCOPE_DIRS = ("knowledge", "projects")
EMPTY_TERMINAL_OWNER = "#832"

# The placeholder text as written by the chat path. The row also carries
# `synthetic_empty_terminal`, but sessions written before that flag existed
# carry only the text.
_PLACEHOLDER = re.compile(r"did not produce a summary", re.I)

# Where a named path may start. Everything from there to the next `.md` on the
# same line is a candidate — spaces included — and each candidate is judged by
# whether it exists, which is what keeps a space-tolerant match from accepting
# prose that merely ends in `.md`.
_START = re.compile(
    r"(?:~|/home/[^/\s]+)/obsidian/"
    r"|(?<![\w/.-])(?:" + "|".join(SCOPE_DIRS) + r")/")
_END = re.compile(r"\.md(?![\w-])")
# Characters no vault filename contains here but markdown puts around a path.
_NOT_PATH = re.compile(r"[`\"*<>|\[\]()\n]")


def _text(entry: dict) -> str:
    content = entry.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type", "text") == "text")
    return ""


def vault_relative(path: str, vault: pathlib.Path) -> str | None:
    """A write target or a named path as a vault-relative string, or None."""
    p = path.strip()
    for prefix in (str(vault).rstrip("/") + "/", "~/obsidian/"):
        if p.startswith(prefix):
            return p[len(prefix):]
    m = re.match(r"/home/[^/]+/obsidian/", p)
    if m:
        return p[m.end():]
    if p.startswith("/") or p.startswith("~"):
        return None
    return p  # vault_write's own argument is vault-relative


def in_scope(rel: str | None) -> bool:
    return bool(rel) and rel.endswith(".md") and rel.split("/", 1)[0] in SCOPE_DIRS


def write_targets(session: dict, vault: pathlib.Path) -> list[str]:
    """Vault-relative `.md` targets under the scope dirs, in call order."""
    out: list[str] = []
    for e in session.get("messages") or []:
        if not isinstance(e, dict) or e.get("role") != "assistant":
            continue
        for tc in e.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name") not in WRITE_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                continue
            if not isinstance(args, dict):
                continue
            raw = args.get("file_path") if fn["name"] != "vault_write" else args.get("path")
            rel = vault_relative(str(raw or ""), vault)
            if in_scope(rel):
                out.append(rel)
    return out


def named_existing_paths(text: str, vault: pathlib.Path) -> list[str]:
    """Every vault path in `text` that resolves to a file on disk."""
    found: list[str] = []
    for start in _START.finditer(text):
        tail = text[start.start():]
        stop = _NOT_PATH.search(tail)
        tail = tail[:stop.start()] if stop else tail
        for end in _END.finditer(tail):
            rel = vault_relative(tail[:end.end()], vault)
            if rel and (vault / rel).is_file():
                found.append(rel)
                break
    return found


def classify(session: dict, vault: pathlib.Path) -> str:
    """`out_of_scope`, `empty_terminal`, `pass` or `fail`."""
    if not write_targets(session, vault):
        return "out_of_scope"
    assistants = [e for e in session.get("messages") or []
                  if isinstance(e, dict) and e.get("role") == "assistant"]
    last = assistants[-1] if assistants else {}
    text = _text(last)
    if (not text.strip() or last.get("synthetic_empty_terminal")
            or _PLACEHOLDER.search(text)):
        return "empty_terminal"
    return "pass" if named_existing_paths(text, vault) else "fail"


def build_record(sessions_dir: pathlib.Path, days: int = DEFAULT_DAYS,
                 vault: pathlib.Path = VAULT_ROOT) -> dict:
    # Non-recursive: `sessions/` also holds `<id>.tool-results/` spill dirs.
    paths = [p for p in sorted(sessions_dir.glob("*.json")) if p.is_file()]
    mtimes = {p: p.stat().st_mtime for p in paths}
    window_end = max(mtimes.values()) if mtimes else 0.0
    cutoff = window_end - days * 86400.0
    counts: Counter = Counter()
    fails_by_source: Counter = Counter()
    failures: list[dict] = []
    empty: list[str] = []
    scanned = 0
    for path in paths:
        if mtimes[path] < cutoff:
            continue
        try:
            session = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(session, dict):
            continue
        scanned += 1
        verdict = classify(session, vault)
        counts[verdict] += 1
        if verdict == "fail":
            source = session.get("source") or session.get("platform") or "chat"
            fails_by_source[source] += 1
            failures.append({"session": path.stem, "source": source})
        elif verdict == "empty_terminal":
            empty.append(path.stem)
    wrote = counts["pass"] + counts["fail"] + counts["empty_terminal"]
    return {
        "days": days,
        "sessions_dir": str(sessions_dir),
        "vault": str(vault),
        "sessions_scanned": scanned,
        "wrote_a_note": wrote,
        "pass": counts["pass"],
        "fail": counts["fail"],
        "empty_terminal": {"count": counts["empty_terminal"],
                           "owner": EMPTY_TERMINAL_OWNER, "sessions": empty},
        "fail_by_source": dict(sorted(fails_by_source.items())),
        "failures": failures,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR))
    ap.add_argument("--vault", default=str(VAULT_ROOT))
    ap.add_argument("--json", action="store_true", help="print the whole record")
    args = ap.parse_args(argv)
    if args.days <= 0:
        print(f"--days must be positive, got {args.days}", file=sys.stderr)
        return 2
    rec = build_record(pathlib.Path(args.sessions_dir).expanduser(), args.days,
                       pathlib.Path(args.vault).expanduser())
    if args.json:
        json.dump(rec, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    et = rec["empty_terminal"]
    print(f"{rec['wrote_a_note']} sessions wrote a knowledge/projects note "
          f"(last {rec['days']} d of {rec['sessions_scanned']}): "
          f"{rec['pass']} named it, {rec['fail']} did not; "
          f"{et['count']} ended empty ({et['owner']}'s class, excluded)")
    for src, n in rec["fail_by_source"].items():
        print(f"  fail  {src}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
