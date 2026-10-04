"""Name a stray in the live checkout on the Bash call that made it.

The start directory of a worker's Bash is outside the tree (#1906), and that does not
stop a command that goes there itself. On 2026-10-02 the nightly trajectory task ran
`cd ~/lloyd && sqlite3 workers.db "select … from autonomy_runs …"` against a path and a
table it had guessed; `sqlite3` created the file to open it. The session saw the empty
file twenty seconds later, confirmed git ignored it, and moved on. The guardian named
it 47 minutes after that and hourly from then, and could not say who wrote it — the
attribution was worked out by hand from the transcripts seven hours later.

So the tool asks the tree the question itself, on either side of the command: which
ignored paths exist in the live checkout (`live_strays.ignored`, about 7 ms). A path
that appeared is put in front of the model on the same result, with where it belongs,
and journaled with the session and the command — and with `actionable_by`, which names
which instrument could act on that path, because the guardian's hourly check and this
bracket read the same journal over two different path sets and a row that does not say
which one can see it reads as an alert nobody actioned (#2172).

No command shape is matched — a
redirect, `sqlite3`, a Python one-liner and a tool nobody has thought of are one case,
which is what a pattern in `protected_paths` could not offer.

Background sessions only: a chat session is a person deciding what is written where.
Not a refusal and not a guard — the command has already run. Fail-open throughout: a
tree that could not be read, a journal that could not be written, costs the note and
nothing else.

What it cannot do: `run_in_background` returns before the command has run, so a
background command is not measured; and two sessions writing in the same second can
each be shown the other's file, which is why the note says "appeared during this
call" and tells the reader to leave what it did not create.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-bash-tree-strays")

#: At most this many paths are spelled out in the note; the journal row has them all.
NOTE_PATHS = 8


def _parent_of(session_id: str) -> str | None:
    try:
        from agent_mcp import _subagent_registry
        parent = _subagent_registry.parent_scope(session_id)
        return parent[0] if parent else None
    except Exception:  # noqa: BLE001
        return None


def watched(session_id: str | None) -> bool:
    """Whether this session's Bash calls are measured: an unattended turn, or a
    subagent of one."""
    if not session_id:
        return False
    try:
        from app.harness.service_control import is_background_session
        return is_background_session(str(session_id), parent_of=_parent_of)
    except Exception:  # noqa: BLE001
        return False


def _live_root() -> Path:
    from app import session_cwd
    return session_cwd.live_root()


def _journal_path() -> Path:
    from app.paths import TREE_STRAY_JOURNAL_PATH
    return Path(TREE_STRAY_JOURNAL_PATH)


def _snapshot_sync() -> "tuple[Path, set[str]] | None":
    from app import live_strays
    root = _live_root()
    seen = live_strays.ignored(root)
    return None if seen is None else (root, seen)


async def before(session_id: str | None, args: dict[str, Any]) -> "tuple[Path, set[str]] | None":
    """The tree as it stands before the command, or None when this call is not
    measured (not a background session, a background command, an unreadable tree)."""
    if not isinstance(args, dict) or args.get("run_in_background"):
        return None
    if not watched(session_id):
        return None
    try:
        return await asyncio.to_thread(_snapshot_sync)
    except Exception:  # noqa: BLE001
        logger.warning("tree stray snapshot failed", exc_info=True)
        return None


def note(root: Path, paths: list[str]) -> str:
    """The paragraph appended to the tool result."""
    from app.paths import DATA_ROOT
    shown = ", ".join(paths[:NOTE_PATHS])
    more = f" (+{len(paths) - NOTE_PATHS} more)" if len(paths) > NOTE_PATHS else ""
    return (
        f"[stray in the code tree] Appeared in {root} during this call: {shown}{more}. "
        f"git ignores {'it' if len(paths) == 1 else 'them'} there, so "
        f"{'it is' if len(paths) == 1 else 'they are'} not code and cannot be committed. "
        f"{root} holds code only; runtime data — sessions, logs, _pipeline, workers.db, "
        f"usage.db — lives in {DATA_ROOT}. If this command created "
        f"{'it' if len(paths) == 1 else 'them'} (an output file, or a database opened by "
        f"a relative path), remove {'it' if len(paths) == 1 else 'them'} now and use the "
        f"path under {DATA_ROOT} or your own working directory. If another session did, "
        "leave it."
    )


#: The two instruments that can act on a journalled path, and the only two values
#: `actionable_by` takes. `guardian-strays` is the guardian's hourly
#: `datawatch.stray_in_tree` — the one that raises `RUNTIME DATA INSIDE THE CODE TREE`
#: and can move inert residue out; `bracket-only` is this module and nothing else.
#: Computed, never guessed: see `_actionable_by`.
GUARDIAN_STRAYS = "guardian-strays"
BRACKET_ONLY = "bracket-only"

#: Where the guardian's reach rule lives, relative to the tree being measured, and
#: the name this process loads it under — deliberately NOT `datawatch`, so the copy
#: this file execs cannot become the `datawatch` some other importer finds cached.
_GUARDIAN_SUBDIR = ("agent-services", "guardian")
_REACH_LOAD_NAME = "lloyd_tree_stray_reach"


def _actionable_by(root: Path, paths: list[str]) -> str:
    """Which instrument could act on these paths: `GUARDIAN_STRAYS` if any one of
    them is within `stray_in_tree`'s reach, `BRACKET_ONLY` if none is.

    WHY THE ROW NEEDS IT. Two instruments, two different path sets, one journal. The
    bracket journals every path that entered the checkout's ignored set
    (`app/live_strays.ignored`, nested paths included); the alerting check judges the
    tree's top level minus `KNOWN_GOOD_TOPLEVEL`, plus the retained `RUNTIME_NAMES`.
    Nothing on a row said which of the two could act, so on 2026-10-03 nightly
    reflection read three rows for a 0-byte `workers.db` that lived 15 seconds
    (against `policy.STRAY_CHECK_SECONDS = 3600.0`, an hourly poll it could never be
    caught by) and one row for `web/tsconfig.node.tsbuildinfo` — ignored by
    `.gitignore:20` (`*.tsbuildinfo`) under the tracked `web/`, so unreachable by the
    alert by construction — sitting under an `ALERT.md` `cleared:` stamp, and filed
    #2169 at priority high as a silent-clear alerting gap. That shape had been closed
    by `6bf40361` about 23 hours earlier. A reach marker is what stops two
    out-of-reach rows and one sub-poll-lifetime row becoming a false alarm.

    WHY IT ASKS THE GUARDIAN RATHER THAN REPEATING IT. The rule is read live from
    `agent-services/guardian/datawatch.py`'s `reachable_by_stray_check` — the same
    function `stray_in_tree` builds its candidates through — so the label cannot drift
    from the check the way two hand-maintained reach lists would. The direction is
    fixed by the runtime, not by taste: the guardian runs from a pinned snapshot under
    `/usr/bin/python3`, where `app` is not importable (`policy.quarantine_dir` reads
    repo constants *by path* for that reason, and `lloyd-guardian.service` execs the
    staged copy), so the bracket reaches toward the guardian and never the reverse.
    `guardian-stage.sh` copies every `*.py` in that directory, so the staged
    `datawatch.py` carries the predicate with no change to the stager.

    HOW IT IS READ, AND WHY NOT `import datawatch`. The module sits outside this
    process's import path and its own body runs `import policy`, which would execute
    file I/O on first call in a fail-open path and would bind a bare top-level
    `policy` name beside the unrelated `app/harness/policy.py`. So it is exec'd from
    its file under a private module name, with its own directory on `sys.path` only
    for the duration (that is what resolves its sibling imports), and the guardian's
    own modules that the exec added are dropped again — by the FILE they came from, so
    the standard library they drag in is left alone. This runs next to the command it is
    measuring and must leave the interpreter as it found it. Any failure raises and is
    the caller's to swallow — a tree whose reach rule cannot be read still gets its row,
    without the key.
    """
    guard = Path(root).joinpath(*_GUARDIAN_SUBDIR)
    source = guard / "datawatch.py"
    if not source.is_file():
        raise FileNotFoundError(f"{source} is not a file")
    import importlib.util
    spec = importlib.util.spec_from_file_location(_REACH_LOAD_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader for {source}")
    module = importlib.util.module_from_spec(spec)
    prefix = str(guard) + os.sep
    loaded: list[str] = []
    sys.path.insert(0, str(guard))
    try:
        before = set(sys.modules)
        spec.loader.exec_module(module)
        # Only the guardian's OWN modules are dropped again — identified by the file they
        # were loaded from, not by having appeared. exec'ing the module drags in half the
        # standard library on this interpreter's first pass (`ast`, `dataclasses`,
        # `enum`, `re`, `subprocess`), and popping those would leave a later importer a
        # second `enum` object beside the first one's classes. A module loaded from a
        # path outside the guardian directory was either already there or is stdlib, and
        # in neither case does this call own it.
        loaded = sorted(n for n in set(sys.modules) - before
                        if str(getattr(sys.modules[n], "__file__", "") or "").startswith(prefix))
        reach = module.reachable_by_stray_check
        # Sorted so the first reachable path is deterministic when several are in one
        # row; the short-circuit keeps this one subprocess on the common case.
        return GUARDIAN_STRAYS if any(reach(p, tree=str(root))
                                      for p in sorted(paths)) else BRACKET_ONLY
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(guard))
        for name in loaded:
            sys.modules.pop(name, None)


def _record(session_id: str | None, command: str, root: Path, paths: list[str],
            kind: str = "appeared") -> None:
    """Append one fact to the durable journal; never raises.

    `kind` is on BOTH row shapes rather than only the new one. `removed` rows are the
    #2110 half: the note this module writes tells the session to delete the stray, so
    until now the one action the instrument asked for was the one action that left no
    trace — the tree's `workers.db` is gone with an empty journal beside it. A file that
    mixes two event classes with no discriminator is a file a later reader guesses at,
    and nothing read this one programmatically when the field was added: it had no rows
    at all (`ls -1 ~/lloyd-data/safety/` → `denials.jsonl` only).

    `actionable_by` is on both kinds for the same reason: it is the other half of "which
    of these rows is an alert nobody actioned", and a reader that has to work out reach
    per row works it out wrong (#2172, and #2169 filed at priority high on exactly that
    mistake). It is computed inside this same fail-open try, in its own nested one, so a
    tree whose reach rule this process cannot read costs the LABEL and not the row —
    the row still lands without the key, and `after` still returns the note.
    """
    try:
        target = _journal_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "session": str(session_id) if session_id else None,
            "root": str(root),
            "paths": paths,
            "command": str(command or "")[:400],
        }
        try:
            row["actionable_by"] = _actionable_by(root, paths)
        except Exception as exc:  # noqa: BLE001 — a missing label costs the label only
            logger.warning("tree stray reach rule unavailable (%s: %s); "
                           "the row is journalled without it",
                           type(exc).__name__, str(exc)[:160])
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tree stray journal write failed (%s: %s); the note stands",
                       type(exc).__name__, str(exc)[:160])


def _removed(root: Path, seen: set[str], now: set[str]) -> list[str]:
    """Paths this call was shown and which are now gone from disk.

    `seen - now` alone is not the answer: a path leaves `ignored()` when it becomes
    TRACKED (`git add`) or when the ignore rule changes, and it is still sitting in the
    tree — journalling that as a deletion would put a file's removal on the record while
    the file is right there, which is a worse instrument than the silence it replaces.
    The fact a row may assert is only this one: it was visible to the before-snapshot,
    and `Path.exists()` on the same root says it is not there now. That covers a failed
    `rm` (still present, no row) as well as an untracked-file removal.

    `app.live_strays` deliberately offers no `disappeared()` helper (#2110 triage),
    because every other decision it makes is about not asserting a fact a measurement
    did not supply; this is the same judgement, so it stays here at the one call site
    that needs it rather than joining the module's API.
    """
    return sorted(p for p in set(seen) - set(now)
                  if p != ".git" and not (root / p).exists())


def _after_sync(text: str, snap: "tuple[Path, set[str]]", session_id: str | None,
                command: str) -> str:
    from app import live_strays
    root, seen = snap
    now = live_strays.ignored(root)
    appeared = live_strays.appeared(seen, now)
    gone = _removed(root, seen, now)
    if appeared:
        _record(session_id, command, root, appeared)
    if gone:
        # Journaled and silent on the result: the session deleted what the note told it
        # to delete, and a second paragraph saying so would read as a new finding. The
        # warning log is the counterparty's half of the fact. Ordered after the
        # appearance row when a single call both deleted one stray and created another,
        # so the file's row order is the shipped one plus an append.
        _record(session_id, command, root, gone, kind="removed")
        logger.warning("session %s: ignored path(s) removed from %s during a Bash call: %s",
                       session_id, root, ", ".join(gone[:NOTE_PATHS]))
    if not appeared:
        return text
    logger.warning("session %s: ignored path(s) appeared in %s during a Bash call: %s",
                   session_id, root, ", ".join(appeared[:NOTE_PATHS]))
    return f"{text}\n\n{note(root, appeared)}"


async def after(text: str, snap: "tuple[Path, set[str]] | None", session_id: str | None,
                command: Any) -> str:
    """`text`, plus the note when an ignored path appeared since `snap`."""
    if snap is None:
        return text
    try:
        return await asyncio.to_thread(_after_sync, text, snap, session_id,
                                       command if isinstance(command, str) else "")
    except Exception:  # noqa: BLE001
        logger.warning("tree stray check failed", exc_info=True)
        return text
