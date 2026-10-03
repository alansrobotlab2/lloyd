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
handler by design and is therefore outside this module — guarded elsewhere, not
uncovered: `profile_argv` in `agent_mcp/_path_sandbox.py` passes one `--ro-bind` per
`PROTECTED_SHELL_RO_ROOTS` entry, which names both these files, to every shell child.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# The ceilings themselves, and the refusal wording, live in `prompt_surface` — one
# definition shared by every writer and by the reporting tests (#1010). `body` is
# imported so the index-line rule below measures the same lines the read-side
# validator measures, from one function and not two copies of its regex.
from app.prompt_surface import body, memory_ceiling, size_error

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

# ── the index's own bounds (MEMORY.md, #1895) ───────────────────────────────
# These two numbers existed only on the READ side until #1895: `git grep -n
# INDEX_LINE_MAX_CHARS` found the validator and the consolidator and no writer,
# while the write guard stopped at the 25,600 B ceiling. So a nightly append of
# ~870 B re-armed a red `test_the_live_memory_index_validates` and the only thing
# that noticed was the next day's Pre-Flight: vault git measured `lloyd/MEMORY.md`
# at 20,466 B (09-25) → 23,941 B (09-29 07:58Z), 3,461 B over the tight limit for
# ~15 h with no refusal anywhere. They live here because this module is the one
# entry point every writer already asks, and the two scripts now import them
# instead of carrying their own copy (#1010's "one constant" rule).

#: The only loaded memory file that is an index. The two bounds below are about
#: index mechanics — a pointer with a hook, and room to append into — and USER.md
#: is neither: it is a profile whose lines the curator retires a few a night.
#: Applying them there would refuse every non-shrinking write on the day they
#: ship, because `lloyd/USER.md` sits at 16,373 of its own 16,384 B carrying 17
#: top-level bullets over 300 characters (measured 2026-09-30). That is the
#: "a ceiling below the file it bounds is a freeze, not a tripwire" trap written
#: up at `prompt_surface`'s `USER_MD_CEILING_BYTES` comment, and a guard that
#: freezes the one file the curator must act on is worse than no guard.
INDEX_MEMORY_FILE = "MEMORY.md"

#: Longest legal index line, in characters. An index line is a pointer with a
#: hook, not the entry: past this it is the detail again, paid for every turn.
INDEX_LINE_MAX_CHARS = 300

#: Share of the ceiling a consolidated index may fill. The rest is the room the
#: nightly writers append into before the next dream pass tightens it again —
#: 20,480 B against the live 19,918 B file, which is under one night's median
#: append, and exactly why the append now has a guard on it.
MEMORY_TIGHTNESS = 0.80


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


def tight_limit(filename: str, tightness: float = MEMORY_TIGHTNESS) -> int | None:
    """The consolidation line for `filename`: `tightness` of its byte ceiling.

    Derived here and nowhere else, because three readers need the same arithmetic:
    the write guard below, `validate_memory_index.check`'s full mode, and
    `memory_ledger.py status`'s CURATE trigger. None for a name that is not a
    loaded memory file — there is no line to hold it to.
    """
    ceiling = memory_ceiling(filename)
    return None if ceiling is None else int(ceiling * tightness)


def overlong_index_lines(text: str) -> list[str]:
    """The top-level index lines of `text` over `INDEX_LINE_MAX_CHARS`.

    The one predicate behind both the write refusal and the validator's
    `long_lines` count, so the two can never disagree about which lines they mean:
    bullets at column zero of `prompt_surface.body(text)` — an indented line is
    detail hanging under an entry, not a line the index pays for — and `>` not
    `>=`, so a line of exactly the cap is legal and one character more is not.
    """
    return [ln for ln in body(text).split("\n")
            if ln.startswith(("- ", "* ")) and len(ln) > INDEX_LINE_MAX_CHARS]


def index_tight_limit_error(filename: str, text: str) -> str | None:
    """Why an index whose prospective text is `text` is too big to grow further.

    The refusal is the instruction, so it routes the writer instead of leaving it
    to improvise: §2a-bis of the knowledge-write skill forbids an ad hoc trim, and
    a refusal that only names a number is precisely what sends a writer to trim
    another entry to squeeze its own line in. Names the limit it refused at, in
    bytes, beside the file's own prospective size.
    """
    limit = tight_limit(filename)
    ceiling = memory_ceiling(filename) or 0
    size = len(text.encode("utf-8"))
    if limit is None or size <= limit:
        return None
    return (
        f"{filename} would be {size:,} bytes, over the {limit:,}-byte tight limit "
        f"({MEMORY_TIGHTNESS:.0%} of its {ceiling:,}-byte ceiling; {size - limit:,} B "
        f"over) that keeps the index consolidated. This is stop-and-write-a-topic-"
        f"file: put the detail in lloyd/{TOPICS_SUBDIR}/<slug>.md with "
        f'memory_add(file="topics/<slug>") and give the index ONE typed line of at '
        f"most {INDEX_LINE_MAX_CHARS} characters ending `→ topics/<slug>`. Do not "
        f"trim or shorten another entry to squeeze this one in — tightening the "
        f"index is dream-consolidation's (#47), and record "
        f"`{filename} at ceiling — consolidation owed` in the completion note."
    )


def index_line_length_error(filename: str, text: str) -> str | None:
    """Why an index whose prospective text holds an over-long line must be refused."""
    long = overlong_index_lines(text)
    if not long:
        return None
    over = len(long[0]) - INDEX_LINE_MAX_CHARS
    return (
        f"{filename} would hold an index line of {len(long[0]):,} characters, over "
        f"the {INDEX_LINE_MAX_CHARS}-character index-line cap ({over:,} characters over)"
        + (f", with {len(long) - 1} more line(s) over it too" if len(long) > 1 else "")
        + f". An index line is a pointer with a hook, not the entry: put the detail "
        f"in lloyd/{TOPICS_SUBDIR}/<slug>.md with memory_add(file=\"topics/<slug>\") "
        f"and keep one typed line of at most {INDEX_LINE_MAX_CHARS} characters ending "
        f"`→ topics/<slug>`. Do not split the entry across two index lines to fit it."
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

    A third bound, on `INDEX_MEMORY_FILE` alone (#1895): a *growing* write to the
    index is also refused once its prospective text passes `tight_limit(filename)`
    (80% of the ceiling — the line the dream pass is meant to keep it under), or
    whenever it would hold a top-level index line over `INDEX_LINE_MAX_CHARS`. The
    first because an index that quietly grows past its tight limit turns the
    read-side validator red and nobody sees that until the next Pre-Flight; the
    second because a long line is the detail coming back into the index one entry
    at a time. Both are scoped to MEMORY.md on purpose — see
    `INDEX_MEMORY_FILE` — and both sit under the shrink escape, so the curator's
    trim is still writable while the index is over the line.

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
    if size < _on_disk_bytes(Path(path)):
        # The shrink escape, tested first so it covers every bound below — the
        # ceiling, the tight limit and the line cap — or the guard would refuse
        # the only write that could bring the file back under them.
        return None
    if size > ceiling:
        return size_error(filename, prospective)
    if filename == INDEX_MEMORY_FILE:
        # The long-line rule first: it is the specific instruction (shorten this
        # line, route its detail to a topic file) where the tight-limit refusal is
        # the general one, and a writer shown a byte count over a line it could
        # simply shorten is a writer that trims an unrelated entry instead.
        return (index_line_length_error(filename, prospective)
                or index_tight_limit_error(filename, prospective))
    return None
