"""One definition of "where a markdown file's front matter ends".

Three programs split a backlog item into (front matter, body) —
`agent_mcp/backlog.py::parse_frontmatter` (the `backlog_*` MCP tools),
`app/routers/backlog.py::_backlog_parse_fm` (the Mission Control board API), and
`scripts/automod/backlog.py::_split_frontmatter` (the triage/implement loop) —
and until #1146 each cut the block at the first `---` *anywhere* in the text:
`content.split("---", 2)`. An item whose activity log quotes that very
expression — `backlog/460-iv-grader-output-is-never-persisted-the-metric-exist.md`
carries the string `text.split('---\\n',2)` inside a YAML scalar — was cut in the
middle of that scalar. The truncated YAML then fails with "found unexpected end
of stream", the reader marks the record `_yaml_broken`, and every writer refuses
it: `save_task` answers "fix the file by hand", `_reject_broken_fm` answers HTTP
409. Thirty-one items were locked that way on 2026-09-20, #1146 among them — the
item reporting the defect could not be written by the tool reporting defects.
All three now call `split_frontmatter` here.

The rule: front matter runs from the opening fence line to the first *line* that
is exactly `---`. A `---` that is part of a longer line — inside a quoted scalar,
in a code fence, in a markdown rule — does not end the block. This is the rule
CLAUDE.md states ("front matter is bounded by its closing `---`") and the one
`app/prompt_builder.py` and `workers/sources/bench_mine.py` already approximate with
`content.find("\\n---\\n", 3)`.

It is not merely tidier than the substring split; it is the only rule that can be
right, because YAML itself cannot express a `---` at column 0 inside a value (see
`_CLOSING_FENCE`). So for any file whose front matter parses, the first column-0
`---` after the opening fence *is* its closing fence — the anchored rule and the
old rule can only disagree on files the old rule was truncating.

Lives in `app/` importing only the standard library for the same reason
`app/backlog_tags.py` does: `scripts/automod/backlog.py` is the light module the
automod CLI loads, and it must not pull `mcp` and `httpx` in behind a
twenty-line helper.
"""

from __future__ import annotations

import os
import re

# The opening fence: the file's first line, beginning with three hyphens.
# Deliberately as loose as the `content.startswith("---")` these callers used,
# rather than requiring the line to be *exactly* `---`: a first line of `----`
# has no valid front matter either way, and tightening the opener would turn
# "parsed, however oddly" into `{}` — which the board's readers skip, so an item
# that used to show up (degraded) would silently disappear from the listing and
# from the board's task count. Only the closing rule is the defect.
_OPENING_FENCE = re.compile(r"^---[^\n]*\n")

# The closing fence: a whole line of exactly three hyphens (trailing blanks and a
# CRLF allowed — hand-edited items carry them), anywhere after the opening line,
# consuming its own terminator so the returned rest is the body proper.
#
# The `$` at the end carries as much of the rule as the `^` at the start. Written
# as `[ \t]*\r?\n?` — both terminators optional — the pattern also matched a
# *prefix* of a longer line, so `--- Some Heading` or a markdown rule with text
# after it ended the block: the reader then took the rest of that line for the
# first body line, the YAML it had kept stopped at whatever the heading line was
# interrupting, and a writer handed that truncated dict re-dumps it — losing every
# key that came after the heading, with the lost text appearing in the body. So the
# line must be *nothing but* the fence: three hyphens, optional blanks, optional CR,
# and then end of line or end of text (`(?:\n|$)` rather than `\n?`, so a closing
# fence on the file's last line with no trailing newline is still a fence).
#
# The `^` anchor is not a nicety either. YAML will not put a fence-lookalike at column 0
# inside a value in any quoting style: a continuation line unindented under
# `note: "before` is a scanner error, and a bare `---` line at column 0 is a
# document separator, which `yaml.safe_load` refuses as "expected a single
# document". The only `---` a valid front-matter block can contain is an
# *indented* one — which is exactly what `yaml.dump` emits for a multi-line
# scalar: `yaml.dump({"note": "a\n---\nb"})` is `note: 'a\n\n  ---\n\n  b'`, so a
# value holding a fence line survives the loop's own re-dump without ever
# starting a line with it. And a file that does carry a column-0 `--- ` line inside
# its block is malformed YAML on exactly those grounds, so the stricter close sends
# it to `_yaml_broken` — where clause 4 of #1221 wants a file like that refused
# rather than silently mis-sliced. Measured across every corpus these three readers
# touch (1,264 board items, 36 autonomy tasks, 194 skills, 3,507 other vault notes)
# no file's split moves at all between the two patterns, so this tightening changes
# no live item: it only stops a shape that has not happened yet from being
# mis-read when it does.
_CLOSING_FENCE = re.compile(r"^---[ \t]*\r?(?:\n|$)", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split `text` into `(front_matter_block, body)`, or None if it has none.

    `front_matter_block` is the YAML *between* the two fence lines with neither
    fence included — pass it to `yaml.safe_load`, or to
    `agent_mcp._shared.parse_frontmatter_text` for the graduated recovery these
    callers want. `body` is everything after the closing fence line, verbatim;
    whether to strip it is the caller's, as it was when each caller held its own
    split (`agent_mcp` and the API reader strip, the automod loop does not).

    None — no front matter at all — when the file does not open with a `---` line
    or has no later line that is exactly `---`. Callers then treat the whole text
    as the body, which is what the unanchored split returned when it came back
    with fewer than three parts.

    A writer reassembles with `f"---\\n{block}---\\n{body}"`, byte-for-byte the
    shape all three writers already emit and the inverse of this split for any
    file whose fences are plain `---` + LF. A fence carrying trailing blanks or a
    CRLF is normalised to `---` + LF by that reassembly.
    """
    opening = _OPENING_FENCE.match(text)
    if opening is None:
        return None
    fence = _CLOSING_FENCE.search(text, opening.end())
    if fence is None:
        return None
    return text[opening.end():fence.start()], text[fence.end():]


# ── The board's one memoised front-matter read ──────────────────────────────
#
# Two programs walk the whole board on every cold dashboard cycle:
# `app.routers.dashboard._backlog`'s scan and `scripts.automod.scorecard.compute`,
# which globs the board from several of its thirteen sections. Measured
# 2026-11-06 on the live board (3,799 numbered item files, ~8.5 KB each): one
# memo-cold walk of the front matter costs ~1.8 s of CPU, and `compute` paid it
# ~2.5 times over inside a single call (11,875 YAML documents parsed for 4,754
# files), which is 6.8 s of a ~15 s cycle. Each reader already had, or wanted, a
# parse cache; keeping one per module meant the second walker still paid full
# price for files the first had just parsed. So the store is here, once, and
# both readers key into it identically.
#
# What is shared is the *store and the read rule*, never the parse: a caller
# hands in `parse_block`, and the YAML still goes through the CALLING module's
# `_YamlLoader` attribute at the call site. `tests/test_dashboard_yaml_loader.py`
# (#1204) proves the loader by construction count under a monkeypatch of that
# attribute, and a shared parse here would have silently un-pinned it — the
# exact trap that module's docstring names.
#
# Stdlib only, like `split_frontmatter` above: `scripts/automod/scorecard.py`
# is a CLI that must not pull FastAPI in to read a front matter block.

#: Bytes per read. The ordinary item's front matter fits in the first chunk, so
#: the common case costs one small read; longer blocks pay one read per 4 KiB.
FM_CHUNK = 4096

#: How far to look for the closing fence. Bounds the **front matter**, not the
#: prefix worth reading: at a flat small prefix, five backlog items were once
#: silently dropped, and the ones dropped were the ones with long `activity_log`
#: blocks — i.e. the ones being worked on. Past `FM_LIMIT` with no closing fence
#: the file is malformed rather than large, and the answer is None.
FM_LIMIT = 65536


def front_matter_text(path, limit: int = FM_LIMIT):
    """The YAML block between a file's fence lines, or None if there is none.

    `split_frontmatter` cuts it — this module's one rule, not a second statement
    of it — over a prefix read a chunk at a time, stopping the moment
    `_CLOSING_FENCE` is in hand so the ordinary item costs one 4 KiB read. A file
    that does not open with a `---` line, never closes within `limit`, or cannot
    be decoded as UTF-8 has no front matter at all.

    This is the block `app.routers/dashboard.py::_frontmatter` parsed until #3860,
    and the two agree byte-for-byte on every file the live board holds: the old
    reader cut from the first newline to a `^---[ \\t]*$` found from byte 3, which
    differs from `split_frontmatter` only on a CRLF file, where the old rule found
    no closing fence at all and this one does. Measured 2026-11-06 over the 3,800
    `*.md` files in `~/obsidian/backlog`: 0 carry `\\r\\n` in their first 8 KiB, so
    no item's fields move on today's board. A CRLF item is now read rather than
    treated as front-matter-less, which is what the three writers already assume.
    """
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(FM_CHUNK)
            while head and len(head) < limit and _CLOSING_FENCE.search(head, 3) is None:
                chunk = f.read(FM_CHUNK)
                if not chunk:
                    break
                head += chunk
    except (OSError, UnicodeError):
        # Undecodable is not front matter, and must not take a whole dashboard
        # section down with it.
        return None
    parts = split_frontmatter(head)
    return None if parts is None else parts[0]


#: path+limit -> ((mtime_ns, size), parsed dict). Keyed like
#: `scripts.automod.state.ledger_rows` and `backlog._load_item_cached`: every
#: writer on this board rewrites the file (new mtime, usually a new size), so a
#: changed file is a different key rather than a stale entry. The signature is
#: taken BEFORE the read, the order the memo shipped with in
#: `app.routers.dashboard` before it moved here: a write landing in between
#: caches the NEWER parse under the OLDER signature, and the next reader's own
#: stat disagrees, so it reparses. Taking the stat afterwards would cache an
#: OLDER parse under a NEWER signature, which nothing notices until the file is
#: written a third time.
FM_MEMO: dict[str, tuple[tuple[int, int], dict]] = {}

#: A board this size is ~12,000 files across the top level and the archive
#: directories the health walk reads, so the normal working set fits with room.
#: On overflow the whole store is dropped rather than evicted per entry: a
#: dashboard pass is bounded by the walk anyway, and a clear is one call.
FM_MEMO_MAX = 50_000


def front_matter_of(path, parse_block, limit: int = FM_LIMIT, memo: bool = True) -> dict:
    """The front matter of `path`, built by `parse_block`, optionally kept per file.

    `parse_block` is the caller's own — it parses through the CALLING module's
    `_YamlLoader` — which is what keeps #1204's loader guard meaningful for every
    reader that shares this read rule and this store. Returns a shallow copy on a
    hit, so a caller that adds a key to the dict it was given cannot change the
    next reader's. `memo=False` is the cold path: same read, same parse, nothing
    stored.
    """
    key = sig = None
    if memo:
        # Signature first: see the note on FM_MEMO about which side of the read
        # it is taken on and why.
        try:
            st = os.stat(path)
        except OSError:
            st = None
        if st is not None:
            key, sig = f"{path}\0{limit}", (st.st_mtime_ns, st.st_size)
            hit = FM_MEMO.get(key)
            if hit is not None and hit[0] == sig:
                return dict(hit[1])

    block = front_matter_text(path, limit)
    if block is None:
        return {}
    fm = parse_block(block)
    if not isinstance(fm, dict):
        # A block that parses to a list or a bare string is not front matter;
        # returning it would blow up on the caller's first `.get`.
        fm = {}
    if key is not None and sig is not None:
        # Both, not just the key: they are assigned together off a successful stat, but
        # `sig` is what the store's value type says must be a real (mtime_ns, size) pair.
        # Storing `None` there would park an entry that can never match a later signature,
        # so the file would be re-parsed every time while looking memoised.
        if len(FM_MEMO) >= FM_MEMO_MAX:
            FM_MEMO.clear()
        FM_MEMO[key] = (sig, fm)
    return dict(fm)


def memo_size() -> int:
    """How many parses the store is holding.

    A reader that has to PROVE it emptied the store needs a number, not a truthiness
    check on a dict it may be holding a different reference to: `memo_clear()` is
    `FM_MEMO.clear()`, which mutates in place, so an alias written before a clear sees
    the same zero and an alias written to the wrong object sees a nonzero that has
    nothing to do with what the loader will hit. `tests/test_dashboard_cold_render.py`
    reads this after its clear for exactly that reason.
    """
    return len(FM_MEMO)


def memo_clear() -> None:
    """Forget every cached parse. For a reader that must measure the cold path."""
    FM_MEMO.clear()
