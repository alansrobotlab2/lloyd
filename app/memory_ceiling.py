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
    """Why `text` is too large to be topic file `name`, or None if it fits.

    The bound is `topic_ceiling(name)`, not the shared constant inline: since #2212
    a live ledger is measured at `ledger_ceiling()`, 3x its audited file's ceiling."""
    size = len(text.encode("utf-8"))
    ceiling = topic_ceiling(name)
    if size <= ceiling:
        return None
    return _topic_size_message(name, size, ceiling)


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

    A fourth bound, on a topic file's NAME rather than its bytes (#2173): a
    non-shrinking write whose stem is outside `TOPIC_SLUG_RE` is refused by
    `topic_slug_error`, which quotes `TOPIC_SLUG_MAX_CHARS`. The memory tools
    already refused such a name through their `file=` JSON-schema pattern, so the
    only writers that ever saw one were `Write`, `Edit` and `vault_write` — the
    three that take an absolute path — and those are exactly the lanes the
    knowledge-write skill sends `lloyd/memory/<slug>.md` down. The condition is the
    one `scripts/memory/validate_memory_index.py` reports ("slug is not
    `[a-z0-9-]{1,48}`"), which this module's docstring and that script's header have
    both said belongs here since the day they were written.

    (The unlanded draft this replaces documented that shrink rule and did not
    implement it — it returned the refusal whenever the prospective text was over
    the ceiling, whatever the file held. A docstring promise the code does not keep
    is the defect that got that round's note landed as a false enforcement claim.)
    """
    filename = _loaded_memory_name(path)
    if filename is None:
        # A topic file is not loaded, but it is a memory file with a bound, and it
        # gets the same shrink rule: every writer that can reach it (the memory
        # tools, Write/Edit, vault_write) is refused the same growth. The escape is
        # measured first so it covers BOTH bounds below — the byte ceiling and,
        # since #2173, the name — or the guard would refuse the only write that can
        # shorten a file whose name it also refuses, and the bad name would stand
        # for good because the lane able to trim it is the lane that said no.
        stem = _topics_dir_stem(path)
        if stem is None:
            return None
        size = len(prospective.encode("utf-8"))
        if size < _on_disk_bytes(Path(path)):
            return None
        # `_topic_file_name` is the name half of that split: it answers None exactly
        # when the stem is outside the grammar, which is the refusal below, and
        # otherwise gives the `topics/<slug>` the byte refusal has always quoted.
        topic = _topic_file_name(path)
        if topic is None:
            return topic_slug_error(stem)
        # One measurement, one owner: `topic_size_error` compares against
        # `topic_ceiling(topic)` and answers None when the write fits, so this call
        # cannot drift from the bound its own refusal quotes. A second `size >`
        # comparison here is where a ledger's derived bound would be missed.
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


# ── the topic-file NAME bound (#2173) ───────────────────────────────────────────
# These three definitions sit BELOW their only caller on purpose. Every line above
# `memory_write_error` is cited by line number in
# `lloyd/reviews/2026-09-14-user-md-audit.md` — `app/memory_ceiling.py:114`, `:162`
# and `:262` — and
# `tests/test_prompt_surface_budget.py::test_every_code_line_the_audit_cites_is_the_line_it_claims`
# reddens any edit that shifts them, which means the space this module can grow
# into without editing a review note that is not the round's to edit is the space
# under line 262. The next addition here has the same constraint.

#: Longest legal topic slug, in characters. A constant rather than a literal inside
#: `TOPIC_SLUG_RE` so the refusal quotes the same number the pattern enforces —
#: #1010's rule for the ceilings, applied to the slug. Until #2173 the 48 existed
#: only inside the regex and in the `file=` JSON-schema pattern on the memory tools,
#: so the absolute-path lanes (`Write`, `Edit`, `vault_write`) were never asked and
#: the nightly write of 2026-10-04 could create
#: `a-done-closure-over-a-refuted-premise-is-refuted-not.md`, 52 characters, which
#: only the next day's vault-reading test noticed.
#: `tests/test_memory_index_cap.py::test_the_slug_cap_the_refusal_names_is_the_cap_the_regex_enforces`
#: is what keeps this equal to the pattern's literal.
TOPIC_SLUG_MAX_CHARS = 48


def _topics_dir_stem(path: str | os.PathLike[str]) -> str | None:
    """The stem of `path` when it is a `.md` directly under the topics directory.

    Deliberately silent on whether that stem is a *legal* slug, which is the
    distinction `memory_write_error` needs: "not a topic file at all" is nobody's
    business and answers None, while "a topic file with an unusable name" is a
    refusal. `_topic_file_name` answered None to both, and that conflation is
    precisely how a 52-character stem got a clean bill from the one guard every
    writer asks.
    """
    p = Path(path)
    if p.suffix != ".md":
        return None
    try:
        parent = os.path.realpath(str(p.parent))
        root = os.path.realpath(str(MEMORIES_DIR / TOPICS_SUBDIR))
    except OSError:
        # Abstain rather than refuse: pricing a path this function could not
        # resolve would be the missing-input verdict its neighbours catalogue.
        return None
    return p.stem if parent == root else None


def topic_slug_error(stem: str) -> str:
    """Why a topic file named `stem` cannot be created or grown (#2173).

    Says which of the two ways of failing `TOPIC_SLUG_RE` it is, so a writer with a
    nine-character `Voice_Mode.md` is not told its name is too long — and names the
    cap either way, because a refusal that hides the bound gets the same name
    written again. The condition mirrors what
    `scripts/memory/validate_memory_index.py` reports for a topic file, and the
    grammar is quoted because it is the whole traversal defence: a slug is a
    single path segment with no `.`, so `memory_read` can never resolve out of
    `memory/` through it.
    """
    over = len(stem) - TOPIC_SLUG_MAX_CHARS
    why = (f"its {len(stem)} characters are {over} over the "
           f"{TOPIC_SLUG_MAX_CHARS}-character topic-slug cap" if over > 0 else
           f"only lowercase letters, digits and hyphens are allowed and this stem "
           f"holds something else (its {len(stem)} characters are inside the "
           f"{TOPIC_SLUG_MAX_CHARS}-character topic-slug cap)")
    return (
        f"{TOPIC_PREFIX}{stem}.md is not a writable topic file: {why}. "
        f"`TOPIC_SLUG_RE` is `[a-z0-9-]{{1,{TOPIC_SLUG_MAX_CHARS}}}`, and that "
        f"grammar is the traversal defence — `memory_read` cannot resolve a name "
        f"outside it, and `validate_memory_index.py` reports the standing file as an "
        f"error. Write the detail under a legal slug of at most "
        f"{TOPIC_SLUG_MAX_CHARS} characters and point the index line at that; a file "
        f"already standing under the bad name is still shrinkable, so write the new "
        f"file first, then trim and retire the old one."
    )


# ── the ledger's OWN bound (#2212) ──────────────────────────────────────────────
# Below `memory_write_error` on purpose, exactly like the #2173 name-bound block:
# `lloyd/reviews/2026-09-14-user-md-audit.md` cites `:114`, `:162` and `:262`, and
# `tests/test_prompt_surface_budget.py::
# test_every_code_line_the_audit_cites_is_the_line_it_claims` reddens any edit that
# shifts them. Everything the ledger bound needs therefore lives here and resolves at
# call time, so the cited lines keep their numbers.

#: How many times the audited file's own ceiling a live ledger may reach.
#:
#: Three, because the ledger's cost and the loaded file's cost are different
#: quantities. A loaded line costs prompt bytes on every turn; a ledger row is never
#: rendered into a prompt at all — `app/prompt_surface.py` composes the two loaded
#: files and nothing under `memory/topics/` — so its own size limit is bookkeeping,
#: not context, and tying it to the topic ceiling measured the wrong thing. The
#: multiplier is a fixed multiple, not a derivation: 3 x MEMORY.md's ceiling holds
#: ~113 rows at the ledger's 672 B mean row, while the same ceiling permits ~127
#: index lines at MEMORY.md's 197 B mean, so past ~113 loaded lines the ledger runs
#: out first. Which of the two moves there is no longer owed — #2212's owed-check
#: ruling of 2026-10-09: the multiplier stays at 3 and MEMORY.md's index budget does
#: not move, because lowering the audited file's ceiling to force a pairing would
#: shrink loaded memory, the thing this guard protects. Its re-open trigger: a
#: nightly finding memory-md-ledger over 70,000 B while MEMORY.md is under 24,000 B
#: raises the multiplier to 4. Both means re-derive live under `~/obsidian/` with
#: `wc -c lloyd/MEMORY.md`, and over `lloyd/memory/memory-md-ledger.md`:
#: `awk '/^- \[/ {n++; s += length($0)+1} END {print n, s/n}'`.
#: The input that, if missing, restores 32,768 B is `memory_ceiling(loaded_file)`.
LEDGER_MULTIPLIER = 3

#: ledger-topic stem -> the loaded file whose ceiling it is bounded by. Declared
#: here, not read from `scripts/memory/memory_ledger.py`, because that module already
#: imports this one and a reverse import is circular; the two are kept equal by
#: `tests/test_memory_ledger_bound.py::
#: test_the_stems_the_guard_treats_as_ledgers_are_the_ones_the_script_reads`, since
#: an unequal pair means the guard bounds a file nobody audits, or leaves the real
#: ledger at a bound it will outgrow. `memory-md-ledger` at 76,800 B and
#: `user-md-ledger` at 49,152 B.
MEMORY_LEDGERS: dict[str, str] = {
    "memory-md-ledger": "MEMORY.md",
    "user-md-ledger": "USER.md",
}


def ledger_ceiling(stem: str) -> int | None:
    """The byte bound of the live ledger whose topic stem is `stem`, or None if `stem`
    is not a ledger. Derived, never stored: 3 x `memory_ceiling()` of the loaded file
    the ledger audits, so changing MEMORY.md's ceiling moves its ledger's bound with
    it instead of leaving a second number to rot."""
    audited = MEMORY_LEDGERS.get(stem)
    if audited is None:
        return None
    file_ceiling = memory_ceiling(audited)
    if file_ceiling is None:
        # The input that, if missing, restores the shared topic ceiling. A renamed
        # loaded file must not silently make a ledger unbounded, and must not silently
        # clamp it to 32,768 B while its audit still writes to it either: the fallback
        # is the old behaviour, and the test names it.
        return TOPIC_FILE_CEILING_BYTES
    return LEDGER_MULTIPLIER * file_ceiling


def topic_ceiling(name: str) -> int:
    """The byte bound that applies to topic file `name` (`topics/<slug>`, `slug.md` or
    the bare stem): the ledger's derived bound for a live ledger, every other topic
    file's shared ceiling for everything else. One owner for the question "how big may
    this topic file be", asked by the write guard, by `topic_size_error`'s message and
    by `scripts/memory/validate_memory_index.py`; a topic file that is not a ledger
    gets `TOPIC_FILE_CEILING_BYTES`, unchanged from before #2212."""
    stem = name[len(TOPIC_PREFIX):] if name.startswith(TOPIC_PREFIX) else name
    stem = stem[: -len(".md")] if stem.endswith(".md") else stem
    return ledger_ceiling(stem) or TOPIC_FILE_CEILING_BYTES


def _topic_size_message(name: str, size: int, ceiling: int) -> str:
    """The refusal text for a topic write of `size` bytes against `ceiling`.

    Two messages, because the two files want opposite advice. An ordinary topic is
    loaded whole by `memory_read`, so splitting it is the right fix and the wording is
    the one `tests/test_memory_index_cap.py` and `agent_mcp/builtin_fs.py` quote, kept
    byte-identical to pre-#2212. A ledger's rows are read back by
    `scripts/memory/memory_ledger.py` from exactly one file per loaded file, so
    splitting it is the WORST thing a curator could do: rows in a second file are
    invisible to `status`, and `curate: true` would keep asking for them forever.
    """
    stem = name[len(TOPIC_PREFIX):] if name.startswith(TOPIC_PREFIX) else name
    audited = MEMORY_LEDGERS.get(stem)
    audited_ceiling = memory_ceiling(audited) if audited is not None else None
    # `audited_ceiling`, not `audited`: the ledger prose needs the number it quotes to
    # exist. An audited file that answers None has already had its bound fall back to
    # TOPIC_FILE_CEILING_BYTES in `ledger_ceiling`, and a refusal that then printed
    # "3 x MEMORY.md's None-byte ceiling" would be a message no reader could act on —
    # so a ledger that lost the file it audits is described as the ordinary topic file
    # the guard is actually measuring it as.
    if audited_ceiling is None:
        return (
            f"{name} is {size:,} bytes, over the {ceiling:,}-byte topic "
            f"file ceiling ({size - ceiling:,} B over). Split it into "
            f"two topics and point the index line at both, or trim it in the same edit."
        )
    return (
        f"{name} is {size:,} bytes, over its {ceiling:,}-byte ledger ceiling "
        f"({LEDGER_MULTIPLIER} x {audited}'s "
        f"{audited_ceiling:,}-byte ceiling, #2212) "
        f"({size - ceiling:,} B over). Do not split a live ledger: "
        f"`scripts/memory/memory_ledger.py` reads only this file for {audited}, so a "
        f"row in a second file is invisible to `status` and the audit silently stops "
        f"covering lines. Trim or retire rows against a loaded line that is gone."
    )
