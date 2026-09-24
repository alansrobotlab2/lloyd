"""Which on-disk writes are rewrites of a loaded memory file, and whether to refuse.

This module answers the *path* question — is this file one of the two that
`prompt_builder._load_memories` injects, and would this write cross its line — and
delegates the *number* and the wording to `prompt_surface.size_error`, which owns
both. That split is the point: the ceiling a `memory_add` refusal quotes and the
ceiling a vault round refuses at have to be one constant, and #1010 was filed
because two automated writers each carried their own idea of what was bounded
(namely: nothing).

Stdlib-only and dependency-free on purpose, like `prompt_surface` itself: it has to
be importable from the MCP server process that writes these files, from a vault-round
validator running in a fresh interpreter, and from a test, without pulling in the SDK
or `prompt_builder`.

Bytes throughout. Measuring the threshold in bytes and the file in characters is how
a budget guard goes quietly wrong — this guard's own first draft (`f51ecd3`, the
round that died ungated) passed a 20,687-character / 20,815-byte file because
`len(text)` was under the limit while `wc -c` was not.

One entry point, `memory_write_error(path, prospective)`, because every writer that
can reach these files — `memory_add`, `memory_replace`, `Write`, `Edit`,
`vault_write` — holds, or can trivially hold, the complete text the file would end
up as. That is the property the refusal has to be priced on: the string refused is
byte-identical to the string that would hit disk, so the writer cannot check one
text and write another. A second helper that re-derived the prospective size from
"current plus an entry" was the earlier shape, and it put the newline arithmetic in
two places — one of which was inside the caller's lock, where the real arithmetic
already lives.

Every call also resolves the path against `MEMORIES_DIR` first, so an unrelated file
that merely shares the name — a `USER.md` in a worktree, a sandbox, a test's
`tmp_path` — is nobody's business.

Why the guard is not just on `memory_add`: the nightly knowledge-write skill says in
its own text "Absolute paths, `Read`/`Write`/`Bash` only. `vault_read`/`vault_write`
reject…", so the job that took `lloyd/USER.md` from 48,068 B to 95,302 B in five days
(#507) never once called the tool that has a ceiling. A guard on the append tool
alone would have guarded the path nobody walks. `Bash` stays outside every tool
handler by design and is therefore outside this module: the route and what should be
done about it are #1010's remaining human clause, not a thing a Python import can
reach.
"""

from __future__ import annotations

import os
from pathlib import Path

# The ceilings themselves, and the refusal wording, live in `prompt_surface` — one
# definition shared by every writer and by the reporting tests (#1010).
from prompt_surface import memory_ceiling, size_error

# The directory `prompt_builder._load_memories` reads the two names from. The guard
# fires on a path only when it resolves INTO this directory, which is what keeps a
# fixture vault and the live vault from being the same question.
MEMORIES_DIR = Path.home() / "obsidian" / "lloyd"


def _loaded_memory_name(path: str | os.PathLike[str]) -> str | None:
    """The memory filename this path writes, or None if it is not a loaded memory."""
    p = Path(path)
    if memory_ceiling(p.name) is None:
        return None
    try:
        parent = os.path.realpath(str(p.parent))
        root = os.path.realpath(str(MEMORIES_DIR))
    except OSError:
        # realpath over an unreachable tree raises on some platforms. Refusing a
        # write because the guard could not resolve it would be the guard reading
        # its own missing input and reporting a verdict it cannot justify, so it
        # abstains — the whole class this file's neighbours catalogue.
        return None
    return p.name if parent == root else None


def _on_disk_bytes(path: Path) -> int:
    """Bytes the file holds right now. Measured here, never handed in by the caller.

    A `current_bytes` argument would be a bypass with a parameter: any caller that
    passed a generous number turned the guard off. The files are tens of kilobytes,
    so re-reading one beside a writer that just read it costs nothing, and the
    number a refusal depends on is then one this module measured.
    """
    try:
        return len(path.read_bytes())
    except OSError:
        return 0


def memory_write_error(path: str | os.PathLike[str], prospective: str) -> str | None:
    """Why this whole-file write must be refused, or None to allow it.

    Serves every handler that can rewrite these files: `memory_add`,
    `memory_replace`, `Write`, `Edit` and `vault_write`. `prospective` is the
    complete text the file would hold after the write — an append's caller composes
    it the same way it composes the bytes it is about to write, newline included —
    so the size quoted in the refusal is the size on disk after the write and not a
    sum of parts that could be assembled differently from the write itself.

    A write that *shrinks* an already-over-ceiling file is allowed even while the
    result is still over the line. That is the difference between a ceiling and a
    freeze: a guard that refuses every over-ceiling write makes its own repair
    impossible, because the only writer left is the one it is refusing. Above the
    line the ceiling is absolute; below it the file's own size is the ratchet.

    (The unlanded draft this replaces documented that shrink rule and did not
    implement it — it returned the refusal whenever the prospective text was over
    the ceiling, whatever the file held. A docstring promise the code does not keep
    is the defect that got that round's note landed as a false enforcement claim.)
    """
    filename = _loaded_memory_name(path)
    if filename is None:
        return None
    ceiling = memory_ceiling(filename) or 0
    size = len(prospective.encode("utf-8"))
    if size <= ceiling:
        return None
    if size < _on_disk_bytes(Path(path)):
        return None
    return size_error(filename, prospective)
