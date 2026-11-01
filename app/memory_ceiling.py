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
from app.prompt_surface import ENTRY_TYPES, body, memory_ceiling, size_error, untyped_entry_count
from app.data_root import vault_root
# The directory `prompt_builder._load_memories` reads the two names from. The guard
# fires on a path only when it resolves INTO this directory, which is what keeps a
# fixture vault and the live vault from being the same question.
MEMORIES_DIR = vault_root() / "lloyd"

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
    `memory_ledger.py status`'s CURATE trigger. Only `INDEX_MEMORY_FILE` is held
    to it (#3629) — see `index_bounds_apply`. None for a name that is not loaded.
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
    a refusal that only names a number is what sends a writer to trim another
    entry to fit. None for a name other than `INDEX_MEMORY_FILE` (#3629): this
    text's advice is the reverse of the ceiling refusal such a write really gets.
    """
    limit = tight_limit(filename)
    ceiling = memory_ceiling(filename) or 0
    size = len(text.encode("utf-8"))
    if not index_bounds_apply(filename) or limit is None or size <= limit:
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
    if index_bounds_apply(filename):
        # #3562: the room reserve is asked HERE, above the shrink escape, because a
        # tightening pass is by definition a shrink and the escape below answers
        # None for every one of them — which is how #3542 shipped an index left with
        # 225 B of room and orphaned run 47's two topic files. The long-line refusal
        # still goes first even on this path, for the reason the chain below gives:
        # it names an edit this very write can make, where the reserve names a room
        # the writer has to go and free.
        reserve = index_room_reserve_error(path, filename, prospective)
        if reserve is not None:
            return index_line_length_error(filename, prospective) or reserve
    if size < _on_disk_bytes(Path(path)):
        # The shrink escape, tested first so it covers every bound below — the
        # ceiling, the tight limit and the line cap — or the guard would refuse
        # the only write that could bring the file back under them. It no longer
        # covers the room reserve above it, and that is deliberate: the reserve
        # guards the room a file is LEFT with, which is not a quantity a shrink can
        # escape by being small.
        return None
    if size > ceiling:
        return size_error(filename, prospective)
    if index_bounds_apply(filename):
        # The long-line rule first: it is the specific instruction (shorten this
        # line, route its detail to a topic file) where the tight-limit refusal is
        # the general one, and a writer shown a byte count over a line it could
        # simply shorten is a writer that trims an unrelated entry instead.
        #
        # `index_no_room_error` sits between the two because it is the SAME tight-limit
        # refusal re-texted for the case where that refusal's own advice is impossible
        # (#2838), not a new condition: it answers non-None only where
        # `index_tight_limit_error` already answers non-None, so the three-term chain
        # refuses exactly what the two-term chain refused before, at the same priority.
        return (index_line_length_error(filename, prospective)
                or index_no_room_error(path, filename, prospective)
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
#: Four now, and the reason it is a multiple at all is unchanged: the ledger's cost
#: and the loaded file's cost are different quantities. A loaded line costs prompt
#: bytes on every turn; a ledger row is never rendered into a prompt at all —
#: `app/prompt_surface.py` composes the two loaded files and nothing under
#: `memory/topics/` — so its own size limit is bookkeeping, not context, and tying it
#: to the topic ceiling measured the wrong thing. The multiplier is a fixed multiple,
#: not a derivation: 4 x MEMORY.md's ceiling holds ~120 rows at the live 847.5 B mean
#: row (89 rows measured 2026-10-19), which is the coverage `status` reports as
#: complete; 3 x held ~90 of them, and the ledger reached that wall rather than the
#: ~113 the stale 672 B mean row had predicted, because every re-check appends a
#: `— earlier: <previous stamp>` history to its row (rows carrying history mean
#: 1,086 B against 793 B for rows that have none).
#:
#: WHY 4 AND NOT A NEW POLICY. #2212's owed-check ruling of 2026-10-09 held that the
#: multiplier stays at 3 and MEMORY.md's index budget does not move, because lowering
#: the audited file's ceiling to force a pairing would shrink loaded memory, the thing
#: this guard protects. That ruling is not being revisited — it shipped its own re-open
#: trigger, "a nightly finding memory-md-ledger over 70,000 B while MEMORY.md is under
#: 24,000 B raises the multiplier to 4", and #3015 is the nightly that found both
#: halves measured on 2026-10-19: `memory-md-ledger` 75,863 B with 937 B of slack, and
#: `MEMORY.md` 20,449 B, after one curation pass stamped 10 `checked:` fields for
#: +2,235 B and the next pass would cross the bound and be refused with
#: `topic_size_error`. So 4 is the pre-authorized multiple the ruling named, taken on
#: its own terms: still bounded, still derived, and MEMORY.md's ceiling unmoved.
#:
#: What the multiple does NOT buy is stability, and the fix for that landed 2026-10-30:
#: §2 item 2 of `skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md` now bounds
#: one `checked:` re-stamp — the date, one note that REPLACES the row's previous note, one
#: earlier-date. Measured headroom, not an estimate: 86,023 B of ledger on 2026-10-30 leaves
#: 16,377 B, ~17 writes at the +924 B/write mean over the 11 writes since `2c7480d3`.
#:
#: Both means re-derive live under `~/obsidian/` with `wc -c lloyd/MEMORY.md`, and over
#: `lloyd/memory/memory-md-ledger.md`:
#: `awk '/^- \[/ {n++; s += length($0)+1} END {print n, s/n}'`.
#: The input that, if missing, restores 32,768 B is `memory_ceiling(loaded_file)`.
LEDGER_MULTIPLIER = 4

#: ledger-topic stem -> the loaded file whose ceiling it is bounded by. Declared
#: here, not read from `scripts/memory/memory_ledger.py`, because that module already
#: imports this one and a reverse import is circular; the two are kept equal by
#: `tests/test_memory_ledger_bound.py::
#: test_the_stems_the_guard_treats_as_ledgers_are_the_ones_the_script_reads`, since
#: an unequal pair means the guard bounds a file nobody audits, or leaves the real
#: ledger at a bound it will outgrow. Ask `ledger_ceiling()` for either bound rather
#: than remembering it: no literal ledger cap is quoted in this module, because a bound
#: repeated in prose is a second number to rot when the multiplier above moves, which
#: is what `tests/test_memory_ledger_bound.py::
#: test_the_curation_skill_witness_carries_no_ledger_cap_at_the_topic_ceiling` already
#: bans a curator's text from doing.
MEMORY_LEDGERS: dict[str, str] = {
    "memory-md-ledger": "MEMORY.md",
    "user-md-ledger": "USER.md",
}


def ledger_ceiling(stem: str) -> int | None:
    """The byte bound of the live ledger whose topic stem is `stem`, or None if `stem`
    is not a ledger. Derived, never stored: `LEDGER_MULTIPLIER` x `memory_ceiling()` of the loaded file
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
    # "N x MEMORY.md's None-byte ceiling" would be a message no reader could act on —
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


# ── the index refusal whose own advice the same bound refuses (#2838) ──────────
#
# `index_file_refusal` sends a writer to write a topic file AND to "give the index ONE
# typed line … ending `→ topics/<slug>`". That second half is only advice while the index
# has room for a line, and the tight limit is exactly the bound that removes it: at the
# size the index actually fails at — 20,463 B on disk against the 20,480 B limit, the
# number #2649's own autotriage record quotes — adding the shortest legal typed line
# (24 B) is refused, so the refusal ordered the one edit that could not be made. The
# writer that obeyed anyway left a detail file the index never linked, which is #2613 and
# #2649 (a link is a line, and a line is what was refused).
#
# This section answers that case instead. It is appended BELOW `memory_write_error` — the
# space under line 262 — for the reason the citation block at :350-358 gives: the three
# lines above it are cited by number in `lloyd/reviews/2026-09-14-user-md-audit.md`, and
# `test_every_code_line_the_audit_cites_is_the_line_it_claims` reddens an edit that shifts
# them.

#: The shortest index line the add-a-line route can ever ask for: a typed bullet that
#: ends in a topic pointer. Derived from the two grammars that define it — the shortest of
#: `prompt_surface.ENTRY_TYPES`, a one-character hook, and `TOPIC_SLUG_RE`'s one-character
#: minimum — and pinned against both regexes by
#: `tests/test_memory_index_cap.py::test_the_line_floor_is_the_line_the_add_a_line_route_orders`,
#: so a grammar change moves this number and the branch that compares it together instead
#: of leaving the branch measuring against a stale figure.
SHORTEST_TOPIC_LINE = f"- [{min(ENTRY_TYPES, key=len)}] x → {TOPIC_PREFIX}a\n"
TOPIC_LINE_MIN_BYTES = len(SHORTEST_TOPIC_LINE.encode("utf-8"))

#: Bytes a linked topic file must have free to be offered as the merge target. The
#: judgeable quantity would be the length of the detail being written, and the guard cannot
#: know it: the file it is checking and the file the detail belongs in may be two different
#: files. So the floor is set from the other bound in reach — a finding is what
#: `INDEX_LINE_MAX_BYTES` sizes an index line at (300 B), and a detail with considerably
#: more to say than that is a knowledge note, which is `consolidation_hint`'s route. The
#: number only ever decides between "this file can obviously take the merge" and "do not
#: offer this file", and every live linked topic sits kilobytes clear of its ceiling.
MIN_TOPIC_MERGE_BYTES = 512

#: A `topics/<slug>` pointer in prose, with the slug bound spelled from its constant.
TOPIC_LINK_RE = re.compile(rf"{TOPIC_PREFIX}[a-z0-9-]{{1,{TOPIC_SLUG_MAX_CHARS}}}")


def linked_topic_files(text: str) -> list[str]:
    """The topic files the top-level index lines of `text` link, in first-seen order.

    Top-level bullets only, which is the grain `app.uptake` parses and the grain
    `consolidate_memory_index.py` writes: an indented continuation quoting a topic path is
    prose about a link, not a link the index carries. Duplicates collapse — the unit here
    is the file, and the same file named by two index lines is one place a merge can go.

    Names come back canonicalised to `topics/<slug>` through `topic_slug`, which is both
    the string the index line itself carries and the form `memory_add(file=…)` takes; a
    pointer that is not a legal slug is skipped rather than offered, since the refusal
    spends the name it returns on an instruction to write to that exact file.
    """
    found: list[str] = []
    for line in body(text).split("\n"):
        if not line.startswith(("- ", "* ")):
            continue
        for match in TOPIC_LINK_RE.finditer(line):
            slug = topic_slug(match.group(0))
            name = f"{TOPIC_PREFIX}{slug}"
            if slug is not None and name not in found:
                found.append(name)
    return found


def _topic_room(name: str) -> tuple[int, int] | None:
    """Bytes `name`'s file has free under its own ceiling, or None when it has none.

    None covers the three ways a file cannot take a merge: it is absent (`memory_add`
    refuses to create one and the writer would hand-create the orphan this whole branch
    exists to stop), it is over its ceiling (`_topic_size_message` and its "Split it into
    two topics" route are the word for that), and it has bytes left but not enough to hold
    a finding. The last is judged against `MIN_TOPIC_MERGE_BYTES`, a floor of judgement
    rather than a bound the guard can derive: nothing here knows how long the detail is,
    because prospective content and the file the guard is checking may be different files
    entirely.
    """
    # `topic_path(MEMORIES_DIR, slug)`, not a join: it is the resolver the rest of this
    # tree uses, and the one that knows a topic lives under `TOPICS_SUBDIR` while the
    # index that links it lives at the root beside that directory.
    slug = name[len(TOPIC_PREFIX):] if name.startswith(TOPIC_PREFIX) else name
    path = topic_path(MEMORIES_DIR, slug)
    ceiling = topic_ceiling(name)
    if not path.is_file() or ceiling is None:
        return None
    free = ceiling - path.stat().st_size
    return (free, ceiling) if free >= MIN_TOPIC_MERGE_BYTES else None


def merge_target(linked: list[str]) -> tuple[str, int, int] | None:
    """The linked topic file with the most room, as `(name, free_bytes, ceiling)`.

    Deliberately a byte count and not a judgement of fit: the guard cannot know which
    linked topic the new detail belongs in, so it names the one that can certainly hold
    it and the refusal says out loud that any other linked topic with room is as good a
    target. Most-room also means the merge that costs the least attention: the file with
    13 KB free is a safer destination than the one with 300 B, and the difference is all
    this function is able to see.
    """
    best: tuple[str, int, int] | None = None
    for name in linked:
        room = _topic_room(name)
        if room is None:
            continue
        if best is None or room[0] > best[1]:
            best = (name, room[0], room[1])
    return best


def index_no_room_error(path, filename: str | None, prospective: str) -> str | None:
    """The tight-limit refusal, re-texted for when its advice cannot be carried out.

    Same condition as `index_tight_limit_error` — the prospective file over the tight
    limit — so it can only ever answer where that function already refuses, and never
    move a verdict. What it adds is the disk fact that turns the advice into a
    contradiction: the tight limit MINUS the bytes already on disk is the room a new index
    line has, and below `TOPIC_LINE_MIN_BYTES` no typed line fits at all, so "give the
    index ONE typed line" is an instruction the same guard will refuse on the next call.
    The room is read from disk and never handed in, because `_on_disk_bytes` measures it
    there — "Measured here, never handed in by the caller", :248-259 — and a room the
    caller could round down would be a way to pick which refusal this is.

    A file still holding a line's worth of room gets None and keeps
    `index_file_refusal`'s text unchanged: with 288 B free — `lloyd/MEMORY.md` at
    20,192 B on 2026-10-16, the last day this branch described the live index — a 105 B
    line is admissible, so the add-a-line route is still the right advice there and
    rewriting it would only cost a writer the sentence it needs. Both figures are dated
    now, because an undated "today" is what rotted: the 2026-10-25 nightly wrote that
    file to 20,465 B and left 15 B free, so this branch stopped being the live answer
    while the sentence above it still claimed to be, and that is what made the
    live-vault node of `tests/test_memory_index_cap.py` red (#3298). The boundary is
    `tight_limit - TOPIC_LINE_MIN_BYTES` = 20,456 B on disk, and the index crosses it
    every time the nightly writes a fat line or dream #47 trims a thin one — which is
    why no number here is a fact about today, and why the test reads the file.
    """
    if index_tight_limit_error(filename, prospective) is None:
        return None
    limit = tight_limit(filename) or 0
    ceiling = memory_ceiling(filename) or 0
    room = limit - _on_disk_bytes(Path(path))
    if room >= TOPIC_LINE_MIN_BYTES:
        return None
    size = len(prospective.encode("utf-8"))
    head = (
        f"{filename} would be {size:,} bytes, over the {limit:,}-byte tight limit "
        f"({MEMORY_TIGHTNESS:.0%} of its {ceiling:,}-byte ceiling) that keeps the index "
        f"consolidated, and only {room:,} of those bytes are free on disk — under the "
        f"{TOPIC_LINE_MIN_BYTES}-byte floor of any typed index line. No new index line "
        f"can be added while the file sits over that limit, so this refusal will not tell "
        f"you to add one.")
    return head + " " + _index_room_advice(path, filename, prospective)


def _index_room_advice(path: str | os.PathLike[str], filename: str,
                       prospective: str) -> str:
    """Where a detail goes when the index has no room for the line that would carry it.

    The merge route if a linked topic file can take it, the consolidation route if none can.
    Extracted by #3562 so `index_room_reserve_error` — which refuses a write that FITS under
    the tight limit but leaves too little room in it — sends a writer to the same place
    `index_no_room_error` does, with one copy of each instruction and its byte counts. The
    wording is what `index_no_room_error` returned before the extraction, byte for byte; the
    leading space is the caller's, because both heads end on a period.
    """
    target = merge_target(linked_topic_files(prospective))
    owed = f"Record `{filename} at ceiling — consolidation owed` in the completion note."
    if target is not None:
        name, free, topic_ceiling_bytes = target
        slug = name[len(TOPIC_PREFIX):]
        return (
            f"This is merge-into-a-linked-topic-file: append the detail to "
            f"`{name}` ({free:,} bytes free of its {topic_ceiling_bytes:,}-byte topic "
            f"ceiling), which `{filename}` already links — "
            f"`memory_add(file=\"{name}.md\")`, or `Edit` on "
            f"lloyd/{TOPICS_SUBDIR}/{slug}.md. Any other topic file the index links and "
            f"that has room is as good a target; a new topic file is not, because the new "
            f"line that would link it is the line that cannot be written (#2613/#2649 "
            f"leave the reason). No line can point at the new detail until "
            f"`dream-consolidation` (#47) reopens room under the tight limit, so the "
            f"loaded index will not advertise it — that is the cost of writing here, and "
            f"it is cheaper than the alternative: a detail file no index line links, "
            f"which nothing loads and nothing prunes. Do not trim or shorten an unrelated "
            f"entry to squeeze a line in — tightening `{filename}` is "
            f"`dream-consolidation`'s decision, not a write-side one. " + owed)
    return _consolidation_owed_tail(filename)


def _consolidation_owed_tail(filename: str) -> str:
    """The tail a refusal ends on when there is nowhere to merge the detail to.

    Extracted by #3562 because `index_room_reserve_error` reaches the same dead end — every
    linked topic full, so only `dream-consolidation` (#47) can reopen room — and the one
    instruction that unblocks a writer cannot have two copies. The text is what
    `index_no_room_error` returned before the extraction, byte for byte: the leading space is
    the caller's, because the heads end on a period.
    """
    return (
        f"This is a consolidation, not a write: no topic file `{filename}` links has enough "
        f"room under its own ceiling to take the merge, and a new topic file cannot be "
        f"created, because the new line that would link it is the line that cannot be "
        f"written (#2613/#2649 leave the reason). Do not trim or shorten an unrelated entry "
        f"to squeeze a line in either — reopen room by merging linked topic files together "
        f"in `dream-consolidation` (#47), which is also the run that reopens the tight "
        f"limit. Record `{filename} at ceiling — consolidation owed` in the completion note."
    )


#: How much room a write has to leave under `MEMORY.md`'s tight limit (#3562). The
#: ceiling chain — `memory_ceiling`, `tight_limit`, `TOPIC_LINE_MIN_BYTES` — answers one
#: question, *does this write fit*, and a write that fits has always been admitted however
#: little room it left behind. Room is the thing the NEXT writer spends: the nightly finishes
#: a topic file and needs one typed index line per file to hook it, and on 2026-10-29 the run
#: record (`~/lloyd-data/autonomy-runs/47/run_47_20261029_180839.md`) found two of them
#: shipped unhooked because the curation pass before it (#3542, vault `e9af9933`) had
#: tightened the index to leave *exactly* 225 B — inside the tight limit, above the 24-byte
#: floor, and under two hook lines, so the guard that owns the tight limit had nothing to
#: say. A never-linked detail file is unrecoverable from the index: nothing loads it, nothing
#: prunes it, and no line points at it.
#:
#: Two lines, then, and two of the WIDEST kind: `INDEX_LINE_MAX_CHARS` is the cap
#: `index_line_length_error` enforces, `+ 1` is the newline that ends the line, and the
#: reserve is twice that. Derived rather than typed for the reason #1895 gave about the tight
#: limit itself — a literal beside the cap it is two of rots the moment the cap moves, and
#: the figure in this comment would become a second hand-tuned number wearing a name. So no
#: size of the reserve is written down here: every term is a constant, and the two
#: relationships that matter are both stated against constants — it is under the cost of the
#: widest hook line the grammar allows, and above the `TOPIC_LINE_MIN_BYTES` floor whose
#: refusal this arm replaces. What the reserve is sized FOR is a measurement of the board,
#: and it lives in the test that measures the board
#: (`tests/test_memory_index_cap.py::test_the_reserve_is_two_of_the_widest_lines_the_index_grammar_allows`
#: + the live-hook-line node), where a figure goes red when the population moves instead of
#: disagreeing with the cap in silence.
#:
#: It is a floor, not a second ceiling: `index_room_reserve_error` fires ONLY at or under the
#: tight limit and never tells anyone to shrink the index — the `consolidation owed` escape
#: that `index_no_room_error` states stays the only route that reopens room, and a repair of
#: an over-limit file keeps the shrink escape below it. `dream-consolidation` (#47) is the
#: run that owns the number; this constant only refuses to let a pass take the room down to
#: a level where that run is the only thing that can write again.
INDEX_ROOM_RESERVE_BYTES = 2 * (INDEX_LINE_MAX_CHARS + 1)


def index_room_reserve_error(path: str | os.PathLike[str], filename: str,
                             prospective: str) -> str | None:
    """Refuse an index write that fits but leaves less than two hook lines of room (#3562).

    Returns None for anything that is not `MEMORY.md`, for a file already at or over its
    tight limit — `index_tight_limit_error` and `index_no_room_error` own those verdicts, and
    their text is unchanged by this predicate, byte for byte — for a write that leaves
    `INDEX_ROOM_RESERVE_BYTES` or more, and, since #3661, for a REPAIR write: one whose whole
    difference from the file on disk is hook lines for topic files that exist there and are
    unlinked (`index_repair_hook_slugs`). The exemption reaches this arm and no other — a
    repair write over the tight limit or under the line floor is refused by those two with
    their text unchanged, byte for byte, exactly as any other write is. `room` is
    `tight_limit(filename) - size`: the space the next typed index line would have to fit in.

    Called by `memory_write_error` BEFORE its shrink escape, which is the whole reason it is
    a separate function placed where it is: a tightening pass shrinks the file, so a rule
    beneath that escape answers None for exactly the writes that made this item, and the
    generator of the orphan keeps generating it. Every string of consequence is a callee's,
    so the 40-character rule on `memory_write_error`'s own code objects is untouched.
    """
    if not index_bounds_apply(filename):
        return None
    limit = tight_limit(filename)
    if limit is None:
        return None
    size = len(prospective.encode("utf-8"))
    room = limit - size
    if room < 0 or room >= INDEX_ROOM_RESERVE_BYTES:
        return None
    # #3661: a write that spends the room on a hook line and on nothing else is the write
    # this room exists to fund, so it is not refused. The shape is measured against the file
    # on disk by the callee and never handed in, for the reason `_on_disk_bytes` gives for
    # the byte count (:248-259): which lines a write adds is not a fact the caller gets to
    # assert, and an `is_repair` argument would be a bypass with a parameter.
    if index_repair_hook_slugs(path, prospective):
        return None
    head = (
        f"{filename} would be {size:,} bytes, inside the {limit:,}-byte tight limit but "
        f"leaving only {room:,} B of room, and a write has to leave "
        f"{INDEX_ROOM_RESERVE_BYTES:,} B — two typed index lines, the room the NEXT writer "
        f"spends — so this one is {INDEX_ROOM_RESERVE_BYTES - room:,} B short."
    )
    # The same advice `index_no_room_error` gives a sub-floor index, from the same helper:
    # merge into a linked topic file that has room, or report consolidation owed when none
    # does. Only the reason differs, and the reason is what `head` carries — a refusal whose
    # advice was copied would be a second text to keep honest, which is what #2637/#2838 left
    # as the standing warning against doing this twice.
    return head + " " + _index_room_advice(path, filename, prospective)


# ── #3629: which file the tight bounds are enforced against ─────────────────────
#
# The scope used to exist only as four copies of `filename == INDEX_MEMORY_FILE` inside
# this module, and the one reader that needed to *report* it had to guess. It guessed the
# way #3629 recorded it: `status` printed `tight limit NNNN` and `-> CURATE` for
# `USER.md` and for the index off identical arithmetic, and the nightly chain read the
# pair as a growth refusal. No writer ever applied that limit to `USER.md` — the ceiling
# is the only refusal that file can receive, which is what `index_bounds_apply` now says.
# One predicate answers the question now, the guard branches on it, and the instrument
# prints what it answers. Placed at the bottom for the reason the block inside
# `memory_write_error` gives: anything inserted above its `def` moves a line that
# `lloyd/reviews/2026-09-14-user-md-audit.md` cites by number.

def index_bounds_apply(filename: str) -> bool:
    """Whether the tight bounds are enforced against writes to `filename`.

    True for `INDEX_MEMORY_FILE` alone. The tight limit (`index_tight_limit_error`),
    its no-room re-texting (`index_no_room_error`), the room reserve
    (`index_room_reserve_error`) and the line cap (`index_line_length_error`) are index
    mechanics, scoped there for the reason `INDEX_MEMORY_FILE`'s own comment gives — and
    every other loaded memory file has exactly one enforced write bound: its ceiling,
    answered by `size_error`. `USER.md` is such a file, which is why its `CURATE` is a
    curation trigger and never a frozen file.

    A predicate and not a comment, because `scripts/memory/memory_ledger.py status` has
    to *say* which bound bites, and a report that re-spelled the file's name would be a
    second definition able to disagree with the guard it describes — the same reason
    `tight_limit` lives here rather than in each script that reads it (#1010).
    `tests/test_memory_index_cap.py::test_the_status_comment_no_longer_calls_the_tight_limit_a_write_bound_for_both_files`
    is the pin: it reddens if the entry point stops branching on this name, which is the
    moment the predicate would become the second definition it exists to avoid.
    """
    return filename == INDEX_MEMORY_FILE


# ── #3661: a repair write may spend the reserve ────────────────────────────────
#
# `INDEX_ROOM_RESERVE_BYTES` and the reverse-link arm of `validate_memory_index.py` are two
# individually-correct guardrails with no ordering between them, and #3661 is what that gap
# costs. `lloyd/memory/a-vault-land-beaten-to-the-commit-by-the.md` was written into the vault
# on 2026-10-31 by automod round #3417 and never indexed, so full mode reports
# `topic_files_unlinked: 1` and `"ok": false`; Step 2.5 of the knowledge-write skill (#2173)
# forbids the `nightly: knowledge write` commit while `ok` is false; and the remedy the
# validator itself names — "add a line ending `→ topics/<slug>`" — is precisely the write the
# reserve refuses, because the live index held 201 B of its 20,480 B tight limit and the hook
# line costs ~184 B, leaving 17 B: 585 B short of the reserve. Two refusals, no route. The
# night's writes stay uncommitted until the FOLLOWING night's `nightly-reflection: pre-flight
# … (unattributed dirty state)` sweep commits them without attribution — the unattributed
# commit Step 2.5 exists to prevent, and #3545's triage measured 6 of the 12 historical
# unlinked-topic-file adds coming out of exactly those sweeps.
#
# What breaks the deadlock is that the two guardrails protect different things. The reserve
# protects room, and room is only ever spent on a line that hooks a topic file: a write that
# spends it on that is not depleting the resource for the next writer, it is spending it on
# what the resource is for. Nothing loads an unlinked detail file and nothing prunes it, so
# the rule that file carries is worth more loaded than the 17 B of room is worth kept. Hence
# an exemption scoped to the one arm that prices room, and to one shape of write. It leaves
# #3562's own case a refusal on the shape, not on luck: the curation pass that left 225 B
# added no hook line, so `index_repair_hook_slugs` answers empty for it and the reserve
# speaks with its old text.
#
# The item's option (a) — deciding ownership in the VALIDATOR, so an orphan the calling job
# did not create is a warning with a named owner — is the bigger design and is NOT taken here.
# Triage records why it cannot be built yet: the orphan is UNTRACKED in the vault
# (`?? lloyd/memory/…`), so a "tracked in HEAD ⇒ not this job's file" test classifies it as
# this job's and keeps blocking; ownership has to key on the commit that added the file or on
# a disposition row, and that ruling is owed. The half that can ship without an ownership
# answer is this one, because "this write adds only hook lines for files that are on disk and
# unlinked" does not need to know who wrote the file.
#
# Appended below `memory_write_error` for the standing reason at :350-358: three lines above
# it are quoted by number in `lloyd/reviews/2026-09-14-user-md-audit.md`, and
# `tests/test_prompt_surface_budget.py::test_every_code_line_the_audit_cites_is_the_line_it_claims`
# reddens any edit that shifts them.


def unlinked_topic_slugs(index_text: str) -> set[str]:
    """The topic files on disk that no top-level line of `index_text` links.

    The same two collections the validator's reverse arm diffs, reached through
    `linked_topic_files` so the guard and the instrument can never disagree about which lines
    link a file — the reason `overlong_index_lines` is shared rather than copied (#1895). A
    slug outside `TOPIC_SLUG_RE` is excluded rather than offered: no legal hook line can name
    it, so a write that "hooked" it would repair nothing and must not be exempted for it. An
    absent `memory/` directory is an empty set, not an error: a vault with no topic files has
    nothing to repair, and this is a question about disk, asked of disk.
    """
    linked = {name[len(TOPIC_PREFIX):] for name in linked_topic_files(index_text)}
    try:
        present = {p.stem for p in (MEMORIES_DIR / TOPICS_SUBDIR).glob("*.md")}
    except OSError:
        return set()
    return {slug for slug in present - linked if TOPIC_SLUG_RE.fullmatch(slug)}


def index_repair_hook_slugs(path: str | os.PathLike[str],
                            prospective: str) -> frozenset[str]:
    """The slugs this prospective index write newly hooks — empty if it is not a repair write.

    A repair write's WHOLE difference from the file on disk is added lines, and every added
    line is a top-level index bullet every one of whose `→ topics/<slug>` pointers names a
    topic file that is on disk and unhooked by the index on disk. Each excluded shape is a
    leak this exemption must not open, which is why the answer is empty for all of them:

    * a pointer to a slug with no file on disk is the dangling link the validator errors on,
      not a repair — and the exemption must not be a way to buy a reserve-spending line;
    * a pointer to a slug the on-disk index already links is a second line to a linked topic
      spending reserve room to duplicate a link, #2613/#2649's shape;
    * an INDENTED bullet: `linked_topic_files` reads top-level lines only, so an indented
      pointer links nothing and the validator would still report the file unlinked — the
      exemption would have spent the room and fixed nothing;
    * any line added, changed or REMOVED besides those hook lines, including the trim that
      would free the bytes for them. Tightening the index is `dream-consolidation`'s (#47)
      decision and this arm's own refusal says so ("Do not trim or shorten an unrelated entry
      to squeeze a line in"); an exemption reachable by swapping an entry for a line would
      hand a writer #47's job paid for with the reserve's room.

    The difference is taken over whole lines of the complete text — front matter, headings and
    blank lines included, not `body()`'s bullets — because what is being waived is a price on
    bytes, and only the whole difference is this write's price. A bumped `updated:` stamp is
    therefore a changed line and a voided repair: the strict side of a rule whose strictness
    has to be total to be checkable.

    What is NOT checked here is the 300-character cap and the byte bounds: their owners
    (`index_line_length_error`, `index_tight_limit_error`, `index_no_room_error`, the ceiling)
    judge the whole prospective text and run right after this arm in `memory_write_error`, so a
    repair write that breaks one is refused by that owner with that owner's text — a second copy
    of a bound inside the one arm built to say yes is just two owners free to disagree. The type
    tag is the exception, and it is not copied either: `untyped_entry_count` is asked, the same
    owner the validator's `untyped_entries` arm calls. It belongs here because admitting an
    untyped line would have this arm spend reserve room on a line that re-reddens the very
    report it exists to clear — trading `no index line links it` for `top-level entries carry no
    [type] tag` is a worse freeze, not a remedy.

    A set rather than a bool, so a test or a future refusal can name WHICH file a write
    hooked instead of re-deriving it from the diff. `bool(...)` is the answer the caller wants.
    """
    try:
        current = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # No readable file is nobody's repair: with nothing on disk there is no difference to
        # price, and the guard's default is always the refusal.
        return frozenset()
    unmatched = current.split("\n")
    added: list[str] = []
    for line in prospective.split("\n"):
        if line in unmatched:
            unmatched.remove(line)
        else:
            added.append(line)
    if unmatched or not added:
        return frozenset()
    unlinked = unlinked_topic_slugs(current)
    hooked: set[str] = set()
    for line in added:
        if not line.startswith(("- ", "* ")):
            return frozenset()
        # The added line has to be a line the index ACCEPTS, and its owner for that is
        # `prompt_surface.untyped_entry_count`, which the validator's `untyped_entries` arm
        # calls too (#2173's full mode). Not a second copy of a bound — a call to the one that
        # exists — and the reason it belongs here rather than waiting for its owner is that an
        # untyped line is an ERROR in the report this exemption exists to turn green: admitting
        # one would trade the blocking `no index line links it` for `1 top-level entries carry
        # no [type] tag`, which is a worse freeze, not a remedy.
        if untyped_entry_count(line):
            return frozenset()
        slugs = {topic_slug(m.group(0)) for m in TOPIC_LINK_RE.finditer(line)}
        slugs.discard(None)
        if not slugs or not slugs <= unlinked:
            return frozenset()
        hooked |= slugs
    return frozenset(hooked)
