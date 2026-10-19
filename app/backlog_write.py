"""One write path for a backlog item file: lock, read inside it, atomic replace.

The board is one file per item, and every writer of it is a read-modify-write:
parse the front matter, change a key or append an `activity_log` line, re-dump
the whole block, replace the file. Until #3036 each of the four lanes that do
that ended in a bare `path.write_text(...)`, with nothing between the parse and
the dump. Two things followed from that, and neither left a trace:

**The lost update.** `agent_mcp.main` (the MCP tool lane), `server.py` (the board
route), the automod status reconciler and the promote/finalizer are separate
processes, and they write the same item file by design — the autocode source
moves statuses on its worker tick while the aggregator answers
`backlog_write_task` and the board answers a modal save. Writer B starts from a
parse taken before A's write landed, so B's dump carries only the keys B knew:
A's `activity_log` entry, and any key only A held (`group`, `acceptance_clauses`,
`automod_landed`), is gone. No exception, no log line, and the automod ledger
still records the move B dropped — which is why the board and the ledger
disagree for some items today. `app/atomic_io`'s own docstring states the rule
this satisfies: the read has to happen *inside* the lock, because a lock around
only the write turns a lost update into a slightly later lost update.

**The torn read.** `Path.write_text` opens with `O_TRUNC`, so anyone reading
while a lane writes — the board's `_backlog_scan`, `load_task`, the reconciler's
`open_items` — can get an empty file or a prefix cut mid-scalar. A prefix that
stops inside a quoted scalar does not parse, and the reader marks the item
`_yaml_broken`; through the route that becomes a 409 on every later edit of that
item (#918 measured the shape on a real hand-off). An `os.replace` is invisible
to a reader: it sees the old inode or the new one, never either half.

Why `commit_lock` and not `locked_file`: `locked_file` puts its lock on a sibling
`<name>.lock` *inside* the board directory, and the board is tracked in the vault
— a stray lock there is a file in `git status`, in Obsidian's tree, and pushed by
`agent-obsidian-sync`. `commit_lock` takes the same flock from
`app.atomic_io.LOCK_DIR` instead, and keys it on the target's realpath so the
three spellings of one item path (module-joined, vault-relative, absolute) take
one lock. The temp goes to `atomic_io.SCRATCH_DIR` for the same reason: the
vault's `.gitignore` has no `*.tmp` rule, so a half-written temp left beside the
item gets committed and synced. `tests/test_memory_writer_lane.py`
::`test_a_crashed_writer_leaves_no_half_written_temp_inside_the_vault` is the
incident that established that rule for the memory files; this store inherits it.

The lock is advisory, so it only excludes writers that take it. That is why this
module is the *only* fence-and-dump path for an item file: every writer that
reaches an item file goes through `write_item`, and
`tests/test_backlog_write_lock.py` parses each of them to prove it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from app import atomic_io
from app import frontmatter as FM

logger = logging.getLogger("app.backlog_write")

#: The one field that is an append-only journal rather than a value. A writer
#: whose parse predates another lane's commit can legitimately overwrite any
#: value it did not touch — that is key-granular last-write-wins, and it is what
#: every lane already did — but it must not delete a row another lane appended.
#: Everything else in the caller's dict is written exactly as given, including
#: the keys it omits: `update_frontmatter` and `_take_back` delete a key by
#: leaving it out, and merging disk keys back in would resurrect them.
APPEND_ONLY_KEYS: tuple[str, ...] = ("activity_log",)


def item_text(fm: Mapping[str, Any], body: str) -> str:
    """Assemble one item file: fence, dumped front matter, fence, body.

    Byte-for-byte the shape all four lanes already emitted, and the inverse of
    `app.frontmatter.split_frontmatter` for a plain `---` + LF fence pair.
    `body` is what follows the closing fence *verbatim*: the automod writers pass
    the split's own body (which begins with the newline after the fence), the MCP
    and route writers pass `"\\n"` plus a stripped body. Keeping that choice at
    the call site is what makes this change byte-neutral for every item file;
    `tests/test_backlog_body_roundtrip.py` pins the route's shape.
    """
    dumped = yaml.dump(dict(fm), default_flow_style=False,
                       allow_unicode=True, sort_keys=False)
    return f"---\n{dumped}---\n{body}"


def merge_journal(current: Mapping[str, Any] | None,
                  fm: Mapping[str, Any]) -> dict[str, Any]:
    """Return `fm` with any `activity_log` row that is on disk but not in it kept.

    Disk order first, then the caller's own new rows: two lanes appending within
    one second come out in the order they were committed, which is the order the
    stamps claim. A caller that supplies no `activity_log` at all keeps the
    file's existing journal rather than clearing it — no lane has ever meant to
    delete one, and a writer that simply did not read it must not empty it.
    """
    out = dict(fm)
    if not current:
        return out
    for key in APPEND_ONLY_KEYS:
        on_disk = current.get(key)
        if not isinstance(on_disk, list) or not on_disk:
            continue
        mine = out.get(key)
        if mine is None:
            out[key] = list(on_disk)
            continue
        rows = [mine] if isinstance(mine, str) else list(mine) if isinstance(mine, list) else None
        if rows is None:
            # A journal the caller stored as something other than a list/string is
            # its own shape to keep; the merge does not guess at it.
            continue
        out[key] = list(on_disk) + [r for r in rows if r not in on_disk]
    return out


def _read_current(path: Path) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """(journal source, body verbatim, refusal reason) for the file as it stands.

    Three shapes come back: no file (everything None), a readable file (its front
    matter for the journal merge, its body for a caller that asked to keep it),
    and a file whose front matter cannot be read at all — a fenced block whose
    YAML raises. That last one is a refusal, not an empty dict: every lane that
    parses for itself refuses this shape before writing (`_yaml_broken` in the MCP
    and route lanes, `_unparsed_guard` in this store's), and the primitive that
    re-reads bytes it did not ask for has to refuse for the same reason — the
    alternative is replacing front matter this module could not read with the
    caller's dict, which is exactly the destruction #1020 was about.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None, None
    block = FM.split_frontmatter(text)
    if block is None:
        # No fence at all: a plain note a writer may legitimately give front
        # matter, exactly the one shape `_unparsed_guard` permits through.
        return None, text, None
    try:
        parsed = yaml.safe_load(block[0])
    except yaml.YAMLError as exc:
        return None, None, f"its front matter will not parse ({type(exc).__name__})"
    if isinstance(parsed, dict) and parsed.get("_yaml_broken") is True:
        # Only ever present if a writer stored a reader's marker; nothing writes
        # it, and if it appeared it means the file was dumped from a fallback
        # parse — the same round-trip every lane refuses.
        return None, None, "its front matter carries a `_yaml_broken` recovery marker"
    return (parsed if isinstance(parsed, dict) else {}), block[1], None


def write_item(path: Path | str, fm: Mapping[str, Any],
               body: str | None = None, *,
               lock, timeout: float | None = None) -> bool:
    """Replace one item file with `fm`, keeping what this call did not touch.

    The caller must be inside `commit_lock(path)` and pass the lock path it
    yielded as `lock`; the primitive deliberately cannot take the lock itself,
    because the whole point is that the *read* below happens inside the caller's
    critical section. A lock that is not the one this path resolves to is a bug
    — the writer locked a different file — and raises rather than writing.
    `tests/test_backlog_write_lock.py` pins that each named writer holds the
    block; the argument is the identity check, not a substitute for that.

    Inside the lock: read the file, keep any `activity_log` row on disk that the
    caller's dict does not carry (`merge_journal`), keep the on-disk body when
    `body` is None, then dump and `atomic_write_text` with the temp in
    `atomic_io.SCRATCH_DIR` and `fsync` before the rename.

    Returns False, having written nothing, when the file's current front matter
    cannot be read. A `TimeoutError` from a lock that will not come free within
    `timeout` (default: the lane's `atomic_io.DEFAULT_LOCK_WAIT`) propagates to
    the caller on purpose — every lane turns it into an error it surfaces; a
    writer that quietly skipped the move instead would keep the board and the
    ledger disagreeing, which is the failure this module exists to end.
    """
    path = Path(path)
    if lock is None:
        raise TypeError(
            "write_item needs the commit_lock path it must run under: take "
            f"`with commit_lock({path.name}) as lock:` and read inside it")
    expected = atomic_io.lock_file_for(path)
    if Path(lock) != expected:
        raise RuntimeError(
            f"write_item({path.name}) was given the lock for a different file "
            f"({Path(lock).name} != {expected.name}) — the caller locked the wrong item")
    journal, disk_body, refusal = _read_current(path)
    if refusal:
        logger.warning("backlog write refused on %s: %s — leaving the file as it is",
                       path, refusal)
        return False
    text = item_text(merge_journal(journal, fm), disk_body if body is None else body)
    atomic_io.atomic_write_text(path, text, fsync=True, tmp_dir=atomic_io.SCRATCH_DIR)
    return True
