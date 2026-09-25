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
import re
from pathlib import Path

# The ceilings themselves, and the refusal wording, live in `prompt_surface` — one
# definition shared by every writer and by the reporting tests (#1010).
from prompt_surface import memory_ceiling, size_error

# The directory `prompt_builder._load_memories` reads the two names from. The guard
# fires on a path only when it resolves INTO this directory, which is what keeps a
# fixture vault and the live vault from being the same question.
MEMORIES_DIR = Path.home() / "obsidian" / "lloyd"

# ── topic files (review 2026-09-24, P4) ─────────────────────────────────────
# MEMORY.md is meant to become an *index*: one typed line per entry, the detail
# pulled on demand from `<MEMORIES_DIR>/memory/<slug>.md` through
# `memory_read(file="topics/<slug>")`. Topic files are never rendered into a
# prompt — `prompt_builder._load_memories` loads MEMORY.md and USER.md by name and
# nothing else — so their bound is a file-size sanity limit, not a prompt budget,
# and it lives here rather than in `prompt_surface` (whose ceilings are all about
# what a prompt carries).

#: The subdirectory of `MEMORIES_DIR` topic files live in.
TOPICS_SUBDIR = "memory"

#: How a tool names a topic file: `topics/<slug>`. The slug grammar is the whole
#: traversal defence — no `/`, no `.`, no case games — so a name that passes it can
#: only ever resolve to one file directly under `TOPICS_SUBDIR`.
TOPIC_PREFIX = "topics/"
TOPIC_SLUG_RE = re.compile(r"[a-z0-9-]{1,48}")

#: Largest legal topic file. A topic is pulled whole by `memory_read`, so a file
#: past this is a tool result the model pays for in full on every read.
TOPIC_FILE_CEILING_BYTES = 32_768


def topic_slug(file: str) -> str | None:
    """The slug of a `topics/<slug>` file name, or None if it is not one.

    `topics/<slug>.md` is accepted as the same name, because a model that has just
    read an index line ending `→ topics/voice` will as often write the extension as
    not; anything else outside the grammar is refused, not normalised.
    """
    if not isinstance(file, str) or not file.startswith(TOPIC_PREFIX):
        return None
    slug = file[len(TOPIC_PREFIX):]
    if slug.endswith(".md"):
        slug = slug[:-3]
    return slug if TOPIC_SLUG_RE.fullmatch(slug) else None


def topic_path(root: str | os.PathLike[str], slug: str) -> Path:
    """Where topic `slug` lives under a memory root (live or an eval overlay)."""
    return Path(root) / TOPICS_SUBDIR / f"{slug}.md"


def _topic_file_name(path: str | os.PathLike[str]) -> str | None:
    """`topics/<slug>` when `path` is a topic file under `MEMORIES_DIR`, else None."""
    p = Path(path)
    if p.suffix != ".md" or not TOPIC_SLUG_RE.fullmatch(p.stem):
        return None
    try:
        parent = os.path.realpath(str(p.parent))
        root = os.path.realpath(str(MEMORIES_DIR / TOPICS_SUBDIR))
    except OSError:
        return None
    return f"{TOPIC_PREFIX}{p.stem}" if parent == root else None


def topic_size_error(name: str, text: str) -> str | None:
    """Why `text` is too large to be topic file `name`, or None if it fits."""
    size = len(text.encode("utf-8"))
    if size <= TOPIC_FILE_CEILING_BYTES:
        return None
    return (
        f"{name} is {size:,} bytes, over the {TOPIC_FILE_CEILING_BYTES:,}-byte topic "
        f"file ceiling ({size - TOPIC_FILE_CEILING_BYTES:,} B over). Split it into "
        f"two topics and point the index line at both, or trim it in the same edit."
    )


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
        # A topic file is not loaded, but it is a memory file with a bound, and it
        # gets the same shrink rule: every writer that can reach it (the memory
        # tools, Write/Edit, vault_write) is refused the same growth.
        topic = _topic_file_name(path)
        if topic is None:
            return None
        size = len(prospective.encode("utf-8"))
        if size <= TOPIC_FILE_CEILING_BYTES or size < _on_disk_bytes(Path(path)):
            return None
        return topic_size_error(topic, prospective)
    ceiling = memory_ceiling(filename) or 0
    size = len(prospective.encode("utf-8"))
    if size <= ceiling:
        return None
    if size < _on_disk_bytes(Path(path)):
        return None
    return size_error(filename, prospective)
