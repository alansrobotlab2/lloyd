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

A third thing it could not do until #2220: name where a stray came from when the tool
that made it was not Bash. `Write` and `Edit` are not bracketed, so the file they lay is
first measured by the NEXT Bash call, whose before-snapshot already holds it — and
`appeared(seen, now)` is a difference against that snapshot, so no origin row for such a
path was merely missing, it was unrepresentable. The journal therefore read as a file that
existed, then was deleted, with nothing between: the committed extract of the live journal
is 5 rows and the `removed` row for `knowledge/` (created by a `Write`, deleted 2026-10-04)
has no origin row beside it. That row is `kind: "present"` now, journalled once per tree
and path against a persisted per-tree acknowledgement state that is seeded silently the
first time a tree is read — because the same branch keyed on the after-set alone would
journal the checkout's 14 standing ignored paths, `.env` among them, as incidents. Still
not covered, and the state is what makes the gap explicit rather than silent: a stray that
entered the tree before this file existed at all, and any write by a session this bracket
skips (a sandboxed call, or `run_in_background`), remain the guardian's `stray_in_tree` to
find, within what `reachable_by_stray_check` can reach.
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

#: The per-tree set of ignored paths already known to be standing in the tree, which is the
#: only thing that decides the `present` kind below. It has to be a file rather than process
#: state: the layer is wired into Bash alone, so the session that MEASURES a stray laid by
#: `Write`/`Edit` is whichever background session happens to run the next command, possibly
#: in another process, and a set kept in the MCP process would also forget it on restart —
#: after which the tree's standing paths, `.env` among them, would be journalled as new
#: incidents by the next call. Seeded silently the first time a tree is read, for the same
#: reason the before-snapshot is taken before the command runs: the live checkout currently
#: answers 14 ignored paths, and 14 rows appearing in a safety journal is not a finding, it
#: is the instrument describing the tree.
ACK_FILE = "tree-stray-acks.json"


def _parent_of(session_id: str) -> str | None:
    try:
        from agent_mcp import _subagent_registry
        parent = _subagent_registry.parent_scope(session_id)
        return parent[0] if parent else None
    except Exception:  # noqa: BLE001
        return None


def watched(session_id: str | None) -> bool:
    """Whether this session's Bash calls are measured: an unattended turn, or a
    subagent of one.

    **What this bracket does when a measured session deletes a stray** — the ruling
    #2110's owed check decided on 2026-10-07, written here so the next author reading
    this bracket to ask "or should a deletion be refused?" does not re-open it:
    removing an in-tree runtime path is JOURNALED, never refused. `after` records the
    disappearance to `safety/tree-strays.jsonl` as a `kind: "removed"` row and hands
    the tool result back untouched, exactly as an appearance is reported unblocked.

    **The guardian is the only mover.** When a stray is meant to leave the tree, the
    moving is the guardian's: `_actionable_by` answers `GUARDIAN_STRAYS`
    ("guardian-strays") for a path only the guardian can reach, and stamps that on the
    row when the writer was not the mover. So a removal route outside this bracket — a
    human terminal, or an attended chat session, where `watched` is false and nothing
    here runs at all — is not this guard's to stop. It is the guardian's
    `stray_in_tree` to find, which is the same sentence the module header ends on for
    strays this bracket cannot see an origin for.

    **Why refusing here would have cost real work and enforced nothing.** Both removals
    on the journal are work a refusal would have blocked: the `workers.db` row, which is
    #2061's prescribed cleanup run by the watched session whose own tool result had just
    named that path, and the #2220 `knowledge/` cleanup of a file a `Write` had laid —
    journalled as a removal at all only because no origin row can exist for a non-Bash
    write. And a refusal would be hollow where it matters most: a `rm` typed in a
    terminal reaches no Lloyd code, so this function is not on the path of the very
    route a refusal is meant to deter.

    Stray paths are named here by BASENAME, never as a full retired-layout path:
    `tests/test_no_runtime_paths_in_code.py` sweeps every non-comment line of every
    tracked `.py` for such a path, and a docstring is a non-comment line.
    """
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


def _ack_path() -> Path:
    """The acknowledgement state, beside the journal and DERIVED from it.

    Not a second `app.paths` constant: the one monkeypatch that redirects the journal has
    to redirect the state with it, or a test that journals into `tmp_path` would be writing
    its acknowledgement state into the live safety directory — and the file that decides
    whether real rows get journalled is exactly the one no test should be allowed to move.
    """
    return _journal_path().with_name(ACK_FILE)


def _read_state() -> dict:
    """The acknowledgement state as written. Raises when it is missing, unreadable or not
    an object; the caller decides what that costs, and it is never "an empty state".

    That distinction is the reason this is not a two-line `try: return json.loads(...)
    except Exception: return {}`. The file is the only memory of which paths a tree was
    found to be holding, and a parse error handled as emptiness would rewrite it from
    nothing: the next call would re-seed, and a box whose state half-wrote would silently
    lose every acknowledgement it had — printing the standing 14 paths as incidents later,
    which is the failure this file exists to prevent. `app/live_strays.ignored` keeps the
    same distinction between UNKNOWN and EMPTY by returning a `None` union, for the same
    reason.
    """
    raw = json.loads(_ack_path().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ACK_FILE} holds {type(raw).__name__}, not an object")
    return raw


def _write_state(state: dict) -> None:
    """Replace the state atomically. One writer per call — `_after_sync`, once, at the end —
    because the three things that change it in a call (an appearance, a removal, a present
    row) are computed from one read, and a read-modify-write per row would let the last one
    clobber the other two.
    """
    path = _ack_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _saved_paths(state: dict, root: Path) -> set[str]:
    entry = state.get(str(root))
    return set(entry.get("paths", [])) if isinstance(entry, dict) else set()


def _snapshot_sync(session_id: "str | None" = None) -> (
        "tuple[Path, set[str], tuple[Path, frozenset[str]] | None,"
        " tuple[Path, frozenset[str]] | None] | None"):
    """The pre-call snapshots: the code tree, and the two watched data-home directories.

    The third and fourth elements are why this is a 4-tuple rather than three separate
    calls. "Did THIS call lay a checkout under the data root", and "did THIS call open an
    empty database", are both questions about a BEFORE, and `after()` is the only place the
    answer gets used; a snapshot taken there would already contain the artefact it is
    asking about, and the check would report nothing forever while looking exactly like a
    check that passes. All three reads therefore have to happen at the same instant as the
    tree, before the command runs, and ride through the harness in the same object the tool
    already threads from `before` to `after`. Each data-home element is None when its
    directory cannot be read, which costs that directory's note and nothing else.

    `session_id` is optional so a caller that only wants the tree keeps working; the
    scratch element is None without it, which is the fail-open direction — a snapshot
    taken without an id measures two directories rather than zero.
    """
    from app import live_strays
    root = _live_root()
    seen = live_strays.ignored(root)
    return None if seen is None else (root, seen, _data_root_top(),
                                      _scratch_top(session_id))


async def before(session_id: str | None, args: dict[str, Any]) -> "tuple | None":
    """The tree as it stands before the command, or None when this call is not
    measured (not a background session, a background command, an unreadable tree)."""
    if not isinstance(args, dict) or args.get("run_in_background"):
        return None
    if not watched(session_id):
        return None
    try:
        return await asyncio.to_thread(_snapshot_sync, session_id)
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
            kind: str = "appeared", actionable: str | None = None) -> None:
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
        # The two branches are separate statements and the computed one keeps its
        # stamp site as the plain assignment below, because that assignment is
        # what `tests/test_bash_tree_strays.py::_label_field` reads out of this
        # file's TEXT to learn the field's name: collapse the two into one
        # conditional expression and three doc-pinning nodes go red on a rename
        # that never happened.
        #
        # A caller that already knows the answer states it instead. `actionable_by`
        # asks which instrument could ACT on the paths, and for a checkout under
        # the data root that is settled by construction, not by asking the tree's
        # reach rule: `stray_in_tree` walks the CODE tree for runtime files, the
        # opposite direction, so the only honest label there is `BRACKET_ONLY`.
        # Leaving it to be computed would exec the guardian's rule against
        # `DATA_ROOT/agent-services/guardian` — a path that does not exist — and
        # land the row without the key, which is the shape #2172 was filed for.
        if actionable is None:
            try:
                row["actionable_by"] = _actionable_by(root, paths)
            except Exception as exc:  # noqa: BLE001 — costs the label, not the row
                logger.warning("tree stray reach rule unavailable (%s: %s); "
                               "the row is journalled without it",
                               type(exc).__name__, str(exc)[:160])
        else:
            row["actionable_by"] = actionable
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


def _record_present(session_id: str | None, command: str, root: Path,
                    paths: set[str]) -> None:
    """One row for an ignored path this call FOUND in the tree, having not put it there.

    WHY THIS KIND EXISTS. The bracket measures Bash and nothing else, so the file a `Write`
    tool call laid in the checkout is first seen by the NEXT Bash call — whose
    before-snapshot already holds it, which is exactly why `appeared(seen, now)`, a
    difference against that snapshot, cannot name it. The origin of such a path was
    therefore unrecordable: the committed witness of the live journal is 5 rows, kinds
    `{appeared: 3, removed: 2}`, and the `removed` row for `knowledge/` — created by a
    `Write` at message 361 of session `20261004_023233_autocode_8e79`, deleted at
    2026-10-04T10:42:21+00:00 — has no origin row beside it and never could have. A reader
    of that journal sees a file that came from nowhere and was quietly deleted; this kind is
    the row that says "somebody non-Bash wrote this, here is the tree and the call that
    found it".

    WHY NOT A NOTE. The sentence `note()` writes tells the model to remove the file now
    because this call made it. This call did not make it. Appending an accusation to a
    command that is innocent of it is how the note gets ignored on the calls where it is
    true, and it would fire on every one of them for the tree's standing paths. The row is
    the record; the result is untouched.

    WHY ONCE. `appeared` is a per-call fact and re-fires legitimately. A `present` path is
    the same file continuing to sit there, and a journal that grew a row for it on every
    background call would bury the rows that do name a writer — so the paths go into the
    acknowledgement state, which `_after_sync` writes once for the whole call.
    """
    _record(session_id, command, root, sorted(paths), kind="present")
    logger.warning("session %s: ignored path(s) found in %s, not created by this call: %s",
                   session_id, root, ", ".join(sorted(paths)[:NOTE_PATHS]))


def _after_sync(text: str, snap: "tuple[Path, set[str]]", session_id: str | None,
                command: str) -> str:
    from app import live_strays
    root, seen = snap
    now = live_strays.ignored(root)
    if now is None:
        # UNKNOWN, not empty. `ignored` answers None when git refuses to answer for the
        # tree; `appeared` would fail closed on it, and the new branch below would not —
        # subtracting the saved set from an unknown set reads as "the whole tree is new",
        # which is precisely the accusation the union type exists to prevent. The rows this
        # call cannot know about are not written, and neither is the state: a state
        # rewritten from an unknown set would silently reset the tree's acknowledgements.
        return text
    appeared = live_strays.appeared(seen, now)
    gone = _removed(root, seen, now)

    # What is unacknowledged is decided by the STATE, never by the after-set alone. With no
    # entry for this tree — the first measured call since the file was created — the tree's
    # standing ignored paths are acknowledged silently and nothing is journalled, which is
    # what keeps `test_a_file_that_was_already_there_is_not_this_calls` green and keeps the
    # 14 paths a real checkout holds out of the journal.
    try:
        state = _read_state()
    except FileNotFoundError:
        state = {}
    except Exception as exc:  # noqa: BLE001 — an unreadable state costs the new kind only
        logger.warning("tree stray state unreadable (%s: %s); no `present` row this call",
                       type(exc).__name__, str(exc)[:160])
        state = None
    fresh: set[str] = set()
    seed = state is not None and str(root) not in state
    if state is not None and not seed:
        fresh = set(now) - _saved_paths(state, root)
    if seed:
        state[str(root)] = {"paths": sorted(now)}

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
    # An appearance this call made is journalled as its own kind, with its note, and is not
    # also `present`; a path that vanished during the call is not standing in the tree, so
    # it has no business in the acknowledgement either.
    fresh -= set(appeared) | set(gone)
    if fresh:
        _record_present(session_id, command, root, fresh)
    if state is not None:
        # The call's one state write, after the rows: a row that landed is the fact, and a
        # state write that fails costs at worst a repeated `present` row on the next call.
        # An appearance joins the acknowledgement here, which is what stops a stray created
        # by this call being journalled a second time as `present` by the next one; a
        # removal drops out of it, because the acknowledgement is of a path STANDING in the
        # tree — leave it set and this becomes the instrument that can name a writer once
        # per filename for the life of the checkout.
        if seed:
            saved = set(state[str(root)]["paths"])
        else:
            # Parenthesised because it has to be: `-` binds tighter than `|` in Python, so
            # `saved | appeared - gone` would be `saved | (appeared - gone)` and a removal
            # would leave the acknowledgement standing — which reads, on the next call, as
            # "already acknowledged", and silently loses the row for the next stray laid at
            # that same path. Both set operations are explicit about their order.
            saved = (_saved_paths(state, root) | set(appeared)) - set(gone)
        state[str(root)] = {"paths": sorted(saved | fresh)}
        try:
            _write_state(state)
        except Exception as exc:  # noqa: BLE001 — the rows above are already the record
            logger.warning("tree stray state write failed (%s: %s)",
                           type(exc).__name__, str(exc)[:160])
    if not appeared:
        return text
    logger.warning("session %s: ignored path(s) appeared in %s during a Bash call: %s",
                   session_id, root, ", ".join(appeared[:NOTE_PATHS]))
    return f"{text}\n\n{note(root, appeared)}"


#: The entry that says a directory is a code checkout, and how deep below a
#: top-level folder of the runtime data root it may sit and still count as that
#: folder being the checkout. The same property and the same bound as the
#: guardian's (`vaultwatch.CHECKOUT_ENTRY`, `vaultwatch.CHECKOUT_MAX_DEPTH`), for
#: the same two shapes, both from 2026-10-20: `~/lloyd-data/triage-3087/.git` (a
#: `git worktree add` straight into the root) and
#: `~/lloyd-data/triage-2918/checkout/.git` (a folder holding the checkout one
#: level deeper). Kept as literals here rather than imported because this module
#: is the MCP process and `vaultwatch` is the pinned guardian snapshot — neither
#: imports the other today, and a shared import between them would be the first
#: seam between two trees that update on different schedules. Pinned to agree by
#: `tests/test_guardian_datawatch.py::the_checkout_spelling_is_the_same_both_ends`.
CHECKOUT_ENTRY = ".git"
CHECKOUT_MAX_DEPTH = 2

#: Heads the sentence naming an empty database this call just opened (#3140), beside
#: `CHECKOUT_MARK` because the two are the data-home half's notes and a reader compares
#: them. A test pins this exact string rather than importing it, same reason as that one.
EMPTY_DB_MARK = "[empty database in the runtime data root]"

#: The extensions that make an empty file a database rather than an empty file. A
#: 0-byte `notes.txt` is a person's placeholder; a 0-byte `workers.db` is a
#: store nobody wrote to, and the difference is the finding.
EMPTY_DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")

#: The journal's kind for "this call opened this empty database into existence".
#: Distinct from `appeared` / `removed` / `present` (a code-tree stray changing
#: state) and from `checkout-in-data-root` (a code checkout laid under the data
#: root) — the five are every event class this file writes, and `tests/
#: test_bash_tree_strays.py` reads them out of the source so a sixth that
#: collides with one of them cannot pass a difference test.
EMPTY_DB_KIND = "empty-db-created"


def _data_root_top() -> "tuple[Path, frozenset[str]] | None":
    """The runtime data root and the names standing in it, or None when unreadable.

    Top-level names ONLY, which is why this costs one `scandir` of ~47 entries and
    no stats: the question the data-home half asks is "did this call put a NEW
    folder here", and a new folder is by definition a name that was not there
    before. The recursion happens in `_new_checkouts`, over the new names alone —
    usually zero, so the ordinary Bash call pays one directory read and one set
    difference. The read itself is `_top_level`, now shared with `_scratch_top` so the
    two watched directories fail the same way too.
    """
    try:
        from app.paths import DATA_ROOT
        return _top_level(Path(DATA_ROOT))
    except OSError:
        return None


def _holds_checkout(dirpath: Path) -> bool:
    """Whether `dirpath` is, or directly holds, a git checkout.

    Depth-bounded by construction: the caller hands it a top-level folder, so
    `dirpath/.git` is depth 1 and a child's `.git` is depth 2, which is the bound
    this module shares with the guardian. Going deeper is not a cheaper check, it
    is a different claim — a repository cloned two levels down inside `_pipeline/`
    does not make that store scratch, and the tripwire that has to decide whether
    to stop guarding it reads the same bound.
    """
    try:
        if (dirpath / CHECKOUT_ENTRY).exists():
            return True
        with os.scandir(dirpath) as it:
            kids = [e for e in it if e.is_dir(follow_symlinks=False)]
    except OSError:
        return False
    return any((Path(e.path) / CHECKOUT_ENTRY).exists() for e in kids)


def _scratch_top(session_id: "str | None") -> "tuple[Path, frozenset[str]] | None":
    """Names of the directory THIS session's Bash call starts in, read before it ran.

    The second watched directory #3140 adds, and the one the module had never looked at:
    `app.session_cwd.resolved_for` is where `agent_mcp/builtin_bash.py::_start_cwd` sends a
    call that names no `cwd`, so a command written as bare `sqlite3 workers.db .tables` —
    no path, no `cd` — lays its file here, in the session's own scratch dir. Two of the
    four shadows this item names were laid exactly that way and stayed invisible to a
    watcher that only read the data root's top level, because this directory is a grandchild
    of that root (`~/lloyd-data/session-cwd/<id>`).

    The path is `session_cwd.path_for(session_id)` — the mint's own convention for the id,
    the same expression that decides the directory's name — and NOT `resolved_for`, which
    reads the session's stored `start_cwd`. The two are equal for every session the writer
    stamps (it lays the scratch at `path_for` and records that path), but reading the
    convention rather than the record keeps this read working for a session whose record is
    mid-write, and it can never be pointed at a directory a person chose: a watched
    session's `cwd` argument is trusted, and a note that reported a person's own working
    directory as though the harness had laid a store in it would be the wrong kind of sure.

    Names only, one `scandir`, exactly like `_data_root_top`. Sizes are not read here: they
    are read for the NEW names afterwards, which is usually none of them, so the ordinary
    call still costs one directory read per watched directory.
    """
    if not session_id:
        return None
    from app import session_cwd
    return _top_level(session_cwd.path_for(session_id))


def _top_level(root: Path) -> "tuple[Path, frozenset[str]] | None":
    """One directory's entry names, or None when the directory cannot be read.

    The shared half of `_data_root_top` and `_scratch_top`. Returning None rather than an
    empty set is the load-bearing part: an empty set would make every name in a directory
    this call could not list look NEW, and a note invented from a failed read is worse than
    no note. None costs that directory's note and nothing else — the other watched
    directory and the code-tree half are read independently.
    """
    try:
        with os.scandir(root) as it:
            return root, frozenset(e.name for e in it)
    except OSError:
        return None


def _new_empty_dbs(root: Path, before: frozenset[str]) -> list[str]:
    """Top-level NEW files of `root` that are 0 bytes and named like a database.

    Shape, not a command pattern: `sqlite3 x.db`, `touch x.db`, `: > x.db` and a Python
    one-liner all produce the same artefact and the same cost, and matching the command
    would silently stop firing on the next one (#1267's argument, restated in this file's
    header for the write side). A new file that has BYTES in it is a job that wrote
    something, and a new 0-byte file that does not end in a database extension is a
    placeholder; both are left alone.
    """
    try:
        with os.scandir(root) as it:
            entries = sorted((e for e in it if e.name not in before),
                             key=lambda e: e.name)
    except OSError:
        return []
    out: list[str] = []
    for e in entries:
        try:
            if not e.is_file(follow_symlinks=False):
                continue
            if not e.name.lower().endswith(EMPTY_DB_SUFFIXES):
                continue
            if e.is_symlink() or e.stat(follow_symlinks=False).st_size != 0:
                continue
        except OSError:
            continue
        out.append(e.name)
    return out


def _empty_db_note(found: "list[tuple[Path, list[str]]]") -> str:
    """The paragraph for the call that opened an empty database into existence.

    Three things, because those are the three each of the four shadows needed a human for:
    the path just created, what it is (an empty database, not a store with contents), and
    where the real stores live — the fact the author of the call was missing. The real root
    is read from `app.paths.DATA_ROOT` at call time and NOT taken from a watched root,
    because when the file sits in a session's scratch dir the two are different directories,
    and telling a session that its scratch dir is where the stores live would be false in
    the one case the leg most needs to be right about.

    One paragraph over every file across every watched directory, because the explanation
    does not change per root. Says nothing about removing the file: the data-root sweep is
    report-only by its own docstring ("a data-root file is a person's call"), and this
    module moved nothing out of the data root before this leg and moves nothing after it.
    """
    from app import paths
    names = [n for _, ns in found for n in ns]
    listed = "`, `".join(str(root / n) for root, ns in found for n in ns)
    return (
        f"\n\n{EMPTY_DB_MARK} this call created `{listed}` — "
        f"{len(names)} empty database{'' if len(names) == 1 else 's'} of 0 bytes, which is "
        f"not data: `sqlite3 <path>` writes the file before it reads it, so opening a store "
        f"that is not there leaves an empty one where the next reader will find it. The real "
        f"stores live in `{paths.DATA_ROOT}`. Nothing was written to here and nothing is "
        f"moved — removing what this call made is this session's own call to make.")


def _new_checkouts(root: Path, before: frozenset[str]) -> list[str]:
    """Top-level folders of `root` that this call created AND that are git checkouts.

    Newness is by name, so a checkout appearing inside a folder that was already
    here is NOT caught by this half — `~/lloyd-data/_pipeline/clone/.git` for
    instance. That asymmetry is deliberate and it is not a gap in the guard: what
    makes a folder dangerous here is its SIZE and its VANISHING, and only a
    top-level folder is what `vaultwatch` counts and what the folder rule fires
    on. The tripwire-side flag reads the same depth from the root regardless of
    newness, so the deeper clone is still refused an excuse — it simply is not
    announced on the call that made it.
    """
    try:
        with os.scandir(root) as it:
            entries = sorted((e for e in it if e.name not in before),
                             key=lambda e: e.name)
    except OSError:
        return []
    out: list[str] = []
    for e in entries:
        try:
            if e.is_dir(follow_symlinks=False) and _holds_checkout(Path(e.path)):
                out.append(e.name)
        except OSError:
            continue
    return out


def checkout_note(root: Path, names: list[str]) -> str:
    """The paragraph naming where a throwaway checkout does belong (#3095)."""
    from app.paths import SCRATCH_CHECKOUT_ROOT
    shown = ", ".join(names[:NOTE_PATHS])
    more = f" (+{len(names) - NOTE_PATHS} more)" if len(names) > NOTE_PATHS else ""
    return (
        f"[checkout in the runtime data root] Now standing under {root} because of this "
        f"call: {shown}{more}. A git checkout is not runtime data. A throwaway one "
        f"belongs under {SCRATCH_CHECKOUT_ROOT} "
        f"(e.g. `git worktree add --detach {SCRATCH_CHECKOUT_ROOT}/check-$$ HEAD`), which "
        "is scratch and nobody watches; this root is watched file by file, so when you "
        "delete this folder the data tripwire sees thousands of files vanish in one tick "
        "and halts promotions and pauses the worker pool over a directory that was never "
        "data. Remove it with `git worktree remove --force` and the folder with `rmdir`, "
        "and run the throwaway from the root above next time (#3095).")


def _data_root_check(session_id: str | None, command: str,
                     snap: "tuple[Path, frozenset[str]]") -> str:
    """The note for a checkout this call laid under the data root; journals it too.

    Shares `ACK_FILE` with the tree half, keyed by this root's own path, because
    the two questions are the same question — "have I already told a session about
    the thing standing at this path" — and a second state file would be a second
    place for the answer to disagree. Keyed separately from the code tree by
    construction: `state` is keyed on the root string, and this root is not the
    code tree. An entry drops out when the folder goes, so a fresh checkout at the
    same name is a fresh finding, exactly as the tree half does it.
    """
    root, before = snap
    found = _new_checkouts(root, before)
    # No early return on an empty `found`. The call that CREATED a checkout and the
    # call that REMOVED it are different calls, and the acknowledgement has to be
    # dropped on the second one — return here and the folder stays acknowledged
    # forever, so the next `triage-<id>` to land under this root is never reported
    # again, which is the exact drift the tree half documents for a removed stray.
    # The state file is read on that call and written only if the set moved, the
    # same cost `_removed` pays.
    state = _read_state()
    key = str(root)
    known = set((state.get(key) or {}).get("paths") or [])
    # A name drops out of the acknowledgement as soon as it is not on disk, which is
    # the same rule the tree half applies to a removed stray: the acknowledgement is
    # of a folder STANDING there, and leaving it set would make the next checkout at
    # that name — the next `triage-3087`, one folder per backlog item forever — read
    # as already reported and go unnoted.
    kept = {n for n in known if (root / n).exists()}
    fresh = [n for n in found if n not in kept]
    if fresh:
        _record(session_id, command, root, fresh, kind="checkout-in-data-root",
                actionable=BRACKET_ONLY)
        logger.warning("session %s: git checkout(s) created under the runtime data root "
                       "by a Bash call: %s", session_id, ", ".join(fresh))
    if state is not None and kept | set(found) != known:
        state[key] = {"paths": sorted(kept | set(found))}
        try:
            _write_state(state)
        except Exception as exc:  # noqa: BLE001 — the row above is already the record
            logger.warning("data-home checkout state write failed (%s: %s)",
                           type(exc).__name__, str(exc)[:160])
    return checkout_note(root, found) if fresh else ""


def _empty_db_check(session_id: "str | None", command: str,
                    watched: "list[tuple[Path, frozenset[str]]]") -> str:
    """The note for an empty database this call just opened, in either watched directory.

    The third direction of the same mistake, and the one nobody was watching: a NEW
    0-byte `.db`/`.sqlite`/`.sqlite3` is a file that exists only because `sqlite3 <path>`
    writes the file before it reads it. Four of them sat unseen for weeks — `kg.sqlite`
    and `_pipeline/workers.db` under the data root and two inside session scratch dirs —
    and each bought an hourly guardian alert plus a human pass (#3109, #3117, #3118),
    because the only thing that found them was `datawatch.data_root_shadows`, which runs
    on the guardian's tick. An hour after the call that could have been told. This leg
    spends one `stat` per NEW name and says it on that call instead.

    One row for the call, however many watched directories it laid files in: this record
    is the origin of a COMMAND, and splitting it by directory would make the number of
    rows a property of how many directories happen to be watched rather than of what the
    session did. The row's `root` is the deepest directory that contains every file it
    names (`_origin_row`), so a single shadow keeps the row rooted exactly where it was
    laid — `root` + `paths[0]` is the created file, and that composition is what a reader
    downstream resolves.
    """
    found: list[tuple[Path, list[str]]] = []
    for root, before in watched:
        fresh = _new_empty_dbs(root, before)
        if fresh:
            found.append((root, fresh))
    if not found:
        return ""
    names = [n for _, ns in found for n in ns]
    logger.warning("session %s: Bash call opened %d empty database(s): %s",
                   session_id, len(names),
                   ", ".join(str(r / n) for r, ns in found for n in ns))
    # Logged BEFORE the row is written, so a `_record` that cannot reach the journal still
    # leaves the warning: a check that finds something and silently records nothing is the
    # #957 defect. `_record` swallows its own write failures, so this cannot raise out of a
    # disk problem — only a bug in here can, and `after()` catches that.
    row_root, rels = _origin_row([r / n for r, ns in found for n in ns])
    _record(session_id, command, row_root, rels, kind=EMPTY_DB_KIND,
            actionable=BRACKET_ONLY)
    return _empty_db_note(found)


def _origin_row(created: list[Path]) -> "tuple[Path, list[str]]":
    """The one row's root and its paths-relative-to-it, for files found across directories.

    The deepest common parent, not `app.paths.DATA_ROOT`: with one file that is the watched
    directory itself, so a call that lays a single shadow journals exactly the row this leg
    would have journalled before a second watched directory existed — `root` + `paths[0]` is
    the file, in both cases. With files in two directories it is the root they share, which
    in production is the data root, because the scratch tree hangs under it.

    A path with no common parent (a file on another mount, which production does not have
    today but a relocated `SCRATCH_ROOT` could) degrades to `/` rather than raising: the row
    still resolves, and a leg that loses the note to a `ValueError` from a path helper would
    be failing open in the wrong direction — the note is what the session needs.
    """
    try:
        # Over the PARENTS: commonpath of a lone file is the file, and a row rooted at the
        # file it names is a row whose paths point at its own sibling.
        root = Path(os.path.commonpath([str(p.parent) for p in created]))
    except ValueError:
        root = Path("/")
    return root, [str(p.relative_to(root)) for p in created]


async def after(text: str, snap: "tuple[Path, set[str]] | None", session_id: str | None,
                command: Any) -> str:
    """`text`, plus the note when an ignored path appeared since `snap`."""
    if snap is None:
        return text
    cmd = command if isinstance(command, str) else ""
    try:
        # `[:2]` because the tree half's own signature is the pair it has always
        # taken; the third element belongs to the data-home half below.
        text = await asyncio.to_thread(_after_sync, text, snap[:2], session_id, cmd)
    except Exception:  # noqa: BLE001
        logger.warning("tree stray check failed", exc_info=True)
        return text
    # The data-home half, on the same result. Deliberately after the tree half and
    # outside its try: the two guards watch different roots, a failure in one must
    # not cost the other's note, and the tree half has early returns that mean
    # nothing at all about whether a checkout landed in the data root. The
    # snapshot is the one `before()` took, so a caller that passes a bare pair
    # simply does not ask this question.
    data_snap = snap[2] if len(snap) > 2 else None
    if data_snap is not None:
        try:
            text += await asyncio.to_thread(_data_root_check, session_id, cmd, data_snap)
        except Exception:  # noqa: BLE001
            logger.warning("data-home checkout check failed", exc_info=True)
    # The empty-database leg, last, and over BOTH watched directories at once so the two
    # notes it can produce come out as one paragraph. Its own try for the same reason the
    # checkout half has one: a directory that cannot be read, or a failure inside the leg,
    # costs this note and leaves the tree note and the checkout note standing.
    db_watch: "list[tuple[Path, frozenset[str]]]" = [
        p for p in (data_snap, snap[3] if len(snap) > 3 else None) if p]
    if db_watch:
        try:
            text += await asyncio.to_thread(_empty_db_check, session_id, cmd, db_watch)
        except Exception:  # noqa: BLE001
            logger.warning("data-home empty-database check failed", exc_info=True)
    return text
