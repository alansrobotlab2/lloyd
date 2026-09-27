"""The one place an upstream body becomes the text a vault entry carries (#1225).

It was GitHub-only until #1561, which is exactly how the YouTube branch slipped past
it: a channel's own description reached `knowledge/…/youtube-digest.md` verbatim, its
promotional footer and all, cut wherever the byte count fell. Every shape the
pipeline now publishes goes through here — release, commit, issue/PR and video — and
`git grep "\\[:500\\]" scripts/intel-pipeline/` is empty because of it.

Two defects on the same field, `FeedItem.summary`, and one contract between the
scanner and the writer that fixes both:

- **The scanners clipped at a byte count** (`[:500]`, three times over, and the
  YouTube scanner's a fourth time until #1561): notes ended inside a word, and
  `knowledge/tools/isaaclab/releases.md` carried `> [!NO` — a callout marker cut in
  half, which Obsidian renders as literal text. `clip_body` cuts at the last
  whitespace before the limit, drops a block-marker line the limit fell inside, and
  says it cut with an ellipsis.
- **The writer pasted whatever came back**, so a PR whose author filled in nothing
  published the whole unfilled template — 42 copies of "Thank you for your interest
  in sending a pull request" in `isaaclab/prs.md` by 2026-09-24. `clean_body`
  decides whether a body says anything; one that does not is replaced by `None`
  and a reason, the skill's own convention for an empty section.

HTML comments go first in both, because they are never content: upstream templates
put their instructions inside `<!-- -->`, and a comment left in would spend the
clip budget on text no reader of the rendered note ever sees. `strip_link_footer`
and `ends_a_sentence` are the same argument made about a channel's links.
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

#: A rule alone on its line: `______` or `======`. Anchored to the whole line on
#: purpose — markdown turns a `______` sitting under a heading into a setext H2, and
#: a run of underscores inside text (`snake__case`, a table row) is not a rule.
_RULE_LINE_RE = re.compile(r"^\s*[_=]{5,}\s*$")

#: The labels a creator writes above the links they sell: `My Links 🔗`, `Follow me`,
#: `Business inquiries`, `Check out my Patreon`. Matched as a leading phrase after
#: any emoji or bullet (`\W*?` skips both), because the label line usually carries no
#: URL and a links-only rule would keep it. Deliberately a closed list of labels and
#: not a general "is this promo" classifier: a classifier that mis-fires eats the
#: description's opening paragraph, which is #856's failure mode on the other side.
_FOOTER_LABEL_RE = re.compile(
    r"^\W*?(?:my\s+links?|follow\s+(?:me|us)|subscribe|support\s+(?:the\s+)?(?:channel|us)"
    r"|business\s+inquiries?|inquiries?|contact\s+(?:us|me)?|check\s+out|resources?"
    r"|timestamps?|chapters?)\b", re.IGNORECASE)

_URL_RE = re.compile(r"(?:https?://|www\.)\S", re.IGNORECASE)

#: Ends a sentence: terminal punctuation, any closing quote or bracket after it. `…`
#: counts, because `clip_body` ends a cut body with exactly that and a body that ran
#: out of budget is still a body (#1561).
_SENTENCE_FINAL_RE = re.compile(r"[.!?…]\s*[\"')\]”’]*\s*$")

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


def _is_link_block_line(line: str) -> bool:
    """A line of a creator's link block: a handle/label, or a bare URL."""
    return bool(_URL_RE.search(line) or _FOOTER_LABEL_RE.match(line))


def _set_off(lines: list, idx: int) -> bool:
    """Was this line set off by a blank — or does it open the text at all?

    The only thing that tells a channel's promo rule from a markdown setext underline:
    `####Heading` or a paragraph immediately followed by `______` is ONE heading to a
    renderer, while a promo block is set off from the prose above it by a blank line.
    Same characters, so the blank above is the discriminator, and dropping the underline
    of a real setext heading would change how the surviving line renders. A description's
    first line has nothing above it to be welded to, so it counts as set off.
    """
    return idx == 0 or not lines[idx - 1].strip()


def strip_link_footer(text: str) -> str:
    """Remove a trailing separator-rule link block from channel-authored text (#1561).

    A YouTube description routinely ends with the channel's own promotional footer —
    a `______` rule, then `My Links 🔗`, then an arrow and a Twitter handle — and the
    scanner used to paste all of it into `knowledge/` as knowledge prose. The rule is
    the block's start and the block's whole tail is link lines: a label, or a URL.

    Two things have to hold before anything is removed, and the clause-1 test pins
    which side of each line a shape falls on:

    - The LAST rule line is the candidate, and every non-blank line under it must be
      part of a link block. A description that rules its own sections with `====` above
      a heading, or puts a `____` above a paragraph of prose, keeps everything, because
      the tail under its rule is prose and not links.
    - The rule must be set off from the line above it (`_set_off`). This is what tells a
      promo rule from a markdown setext underline — `####Heading` + `______` is one
      heading — and a strip that removed the underline would not be returning the text
      above the rule untouched, it would be re-rendering it.

    What is kept is returned as it was: the strip removes the block and does not rewrap
    or re-case the prose it was attached to.

    Returns "" for a body that WAS nothing but the footer; the caller decides what an
    empty body means (`ends_a_sentence` below is the writer's test for that).
    """
    if not text:
        return ""
    lines = text.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if not _RULE_LINE_RE.match(lines[idx]) or not _set_off(lines, idx):
            continue
        tail = [ln for ln in lines[idx + 1:] if ln.strip()]
        if not tail or all(_is_link_block_line(ln) for ln in tail):
            return "\n".join(lines[:idx]).rstrip()
        # A rule with prose under it is section furniture, not a footer. Anything
        # above that rule is the description too, so there is nothing to take off.
        return text.rstrip()
    return text.rstrip()


def ends_a_sentence(text: str) -> bool:
    """True when at least one line of `text` ends in sentence-final punctuation.

    The YouTube branch's emptiness test (#1561). Once the footer is stripped, what is
    left either reads as prose — some line ends a sentence — or is the channel's link
    block wearing a body's clothes. #1509's `is_trailer_only` cannot do this job: it
    matches git trailers, and a `____` rule with `My Links 🔗` under it is not a
    trailer, which is why that run logged `skipped (no body): 0` while publishing
    nothing but a footer.
    """
    return any(_SENTENCE_FINAL_RE.search(line) for line in (text or "").splitlines())
