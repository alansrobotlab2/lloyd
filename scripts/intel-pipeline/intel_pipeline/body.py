"""The one place a GitHub body becomes the text a vault entry carries (backlog #1225).

Two defects on the same field, `FeedItem.summary`, and one contract between the
scanner and the writer that fixes both:

- **The scanner clipped at a byte count** (`[:500]`, three times over): notes ended
  inside a word, and `knowledge/tools/isaaclab/releases.md` carried `> [!NO` — a
  callout marker cut in half, which Obsidian renders as literal text. `clip_body`
  cuts at the last whitespace before the limit, drops a block-marker line the limit
  fell inside, and says it cut with an ellipsis.
- **The writer pasted whatever came back**, so a PR whose author filled in nothing
  published the whole unfilled template — 42 copies of "Thank you for your interest
  in sending a pull request" in `isaaclab/prs.md` by 2026-09-24. `clean_body`
  decides whether a body says anything; one that does not is replaced by `None`
  and a reason, the skill's own convention for an empty section.

HTML comments go first in both, because they are never content: upstream templates
put their instructions inside `<!-- -->`, and a comment left in would spend the
clip budget on text no reader of the rendered note ever sees.
"""

import re
from typing import Optional, Tuple

# What the scanner keeps of a release body, commit message or issue/PR body.
SUMMARY_LIMIT = 500

# A body with less than this much content once scaffolding is removed says nothing
# a title does not. `fix typo` is 8; a one-sentence PR description is ~60+.
BODY_FLOOR = 40

ELLIPSIS = "…"

# An unclosed `<!--` (the body was cut, upstream or by an older clip) runs to the end.
_COMMENT_RE = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)

# Lines that open a block whose marker must never be emitted half-written: a quote
# or callout, a heading, a task-list checkbox.
_BLOCK_MARKER_RE = re.compile(r"^\s*(?:>|#|[-*+]\s*\[)")

# Scaffolding: headings and checkbox lines carry no description of their own.
_SCAFFOLD_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|#{1,6}$|[-*+]\s*\[[ xX]?\])")

# Upstream template instructions, matched case-insensitively. A line holding one is
# boilerplate wherever it sits, inside a comment or pasted bare.
TEMPLATE_PHRASES = (
    "thank you for your interest in sending a pull request",
    "please include a summary",
    "please make sure to check the contribution guidelines",
    "please try to keep prs small and focused",
)


def strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text or "")


def _collapse_blank_runs(text: str) -> str:
    return re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n\n", text).strip()


def clip_body(text: Optional[str], limit: int = SUMMARY_LIMIT) -> str:
    """`text` without comments, cut on a word boundary at or under `limit`.

    A body that fits is returned whole (comments removed). One that does not is cut
    at the last whitespace before `limit`; when that leaves the last line a partial
    quote, callout, heading or checkbox, the line goes too; then `…` is appended.
    A run of `limit` characters with no whitespace at all is the one case that is
    cut mid-token, since there is no boundary to find.
    """
    text = _collapse_blank_runs(strip_comments(text or ""))
    if len(text) <= limit:
        return text

    cut_at = max(text.rfind(ws, 0, limit + 1) for ws in (" ", "\n", "\t"))
    if cut_at <= 0:
        return text[:limit].rstrip() + ELLIPSIS
    head = text[:cut_at]
    # The cut fell inside the last line unless it fell on the newline ending it.
    if text[cut_at] != "\n":
        line_start = head.rfind("\n") + 1
        if _BLOCK_MARKER_RE.match(head[line_start:]):
            head = head[:line_start]
    head = head.rstrip()
    if not head:
        # The only line is a straddling marker line; a word-boundary cut of it
        # still beats an ellipsis on its own.
        head = text[:cut_at].rstrip()
    return head + ELLIPSIS


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(ELLIPSIS).strip()


def clean_body(text: Optional[str], title: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Decide what a vault entry carries for this body.

    Returns one of:
      (body, None)   paste `body`;
      (None, reason) the body says nothing — render `None — reason`;
      (None, None)   the body only restates the title — omit the body line.
    """
    # Looked for in the raw text: templates keep their instructions in a comment,
    # and the comment is exactly what says the body is a template.
    raw_low = (text or "").lower()
    saw_phrase = any(p in raw_low for p in TEMPLATE_PHRASES)
    saw_scaffold = False
    kept_lines = [line for line in strip_comments(text or "").splitlines()
                  if not any(p in line.lower() for p in TEMPLATE_PHRASES)]
    body = _collapse_blank_runs("\n".join(kept_lines))

    content = []
    for line in body.splitlines():
        if _SCAFFOLD_LINE_RE.match(line):
            saw_scaffold = True
            continue
        content.append(line)
    content_text = " ".join(" ".join(content).split())

    if title:
        norm_title = _normalise(title)
        norm_body = _normalise(body)
        if norm_title and norm_body.startswith(norm_title):
            # A commit's title is its message's first line, so this is the normal
            # shape of a commit. Keep what the message says beyond it, if anything.
            rest = _collapse_blank_runs(body.split("\n", 1)[1]) if "\n" in body else ""
            if len(" ".join(rest.split())) < BODY_FLOOR:
                return None, None
            return clean_body(rest)

    if len(content_text) < BODY_FLOOR:
        if saw_phrase:
            return None, "the upstream body is an unfilled PR template"
        if saw_scaffold:
            return None, "the upstream body is template headings with nothing under them"
        if not content_text or content_text.lower() == "no description":
            return None, "no description upstream"
        return None, f"the upstream body is under {BODY_FLOOR} characters"
    return body, None
