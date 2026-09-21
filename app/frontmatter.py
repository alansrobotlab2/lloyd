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
`prompt_builder.py` and `workers/sources/bench_mine.py` already approximate with
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
