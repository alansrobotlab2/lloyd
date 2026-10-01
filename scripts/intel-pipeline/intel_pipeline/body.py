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

#: The heading half of that class on its own (#1925). `clean_body` discounted a heading
#: line before testing a body against `BODY_FLOOR` and then returned the text still
#: holding it, so a filled PR opening `# Description` cleared the floor and published the
#: heading one or two levels inside the digest's own `## 2026-MM-DD` section — flattening
#: the dated structure every corpus-shape count reads, by the upstream author's headings
#: and not by Lloyd's.
#:
#: #1925 stripped only this half from the returned value and kept checkbox lines, for
#: the sake of a filled release body that writes real content on them
#: (`- [ ] Added Newton visualizer support`). #2012 reversed that: a line the floor
#: does not count is not published either, so the returned body is now built from the
#: same lines the floor measured, and this pattern is kept for callers and tests that
#: ask about headings alone. The price, owned here: content written on a checkbox line
#: is no longer published.
#:
#: The one cost of a line-based strip, stated here rather than discovered at review: a
#: `# comment` shell line inside a fenced code block is a heading marker to this pattern
#: and goes with them. It is the same line the floor measurement has always ignored, so
#: the measurement and the publication agree — but it is a real, small content loss.
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}(?:\s|$)")

#: The checkbox half of `_SCAFFOLD_LINE_RE` on its own: the line a wrapped task-list
#: item starts on.
_CHECKBOX_LINE_RE = re.compile(r"^\s*[-*+]\s*\[[ xX]?\]")

#: A line that opens a block of its own, so it cannot be the wrapped tail of the
#: checkbox line above it: another list item, a heading, a quote, a fence, a table row.
_BLOCK_OPENER_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#|>|```|~~~|\|)")

#: An issue-number trailer the author never filled in: a closing keyword, a `#`, and a
#: parenthesised placeholder holding no digit — `Fixes # (issue)`. Recognised by shape,
#: not by phrase (#2012): `TEMPLATE_PHRASES` is four sentences from one upstream and
#: cannot enumerate every template, while "a reference to an issue with no number in
#: it" is the same in all of them. `Fixes #8219` has its number and is content.
_UNFILLED_ISSUE_TRAILER_RE = re.compile(
    r"^\s*(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s*:?\s*#\s*\([^)\d]*\)\s*\.?\s*$",
    re.IGNORECASE)

#: A rule alone on its line: `______` or `======`. Anchored to the whole line on
#: purpose — markdown turns a `______` sitting under a heading into a setext H2, and
#: a run of underscores inside text (`snake__case`, a table row) is not a rule.
_RULE_LINE_RE = re.compile(r"^\s*[_=]{5,}\s*$")

#: A line that is NOTHING but a short label closed by a colon: `Full post:`, `Links:`,
#: `Podcast:`. No word may sit after the colon, so a real sentence cannot match it, and the
#: 40-character ceiling is what keeps the first line of a wrapped paragraph out. It is the
#: fourth shape a channel's link block is built from that `_URL_RE` and `_FOOTER_LABEL_RE`
#: do not already cover: `My Links 🔗` is a label, the destination is a URL, and only the
#: words pointing at them are neither (#1861).
_RUN_LABEL_RE = re.compile(r"^\W*?[^:\n]{1,40}:\s*$")

#: The arrows a pointer line starts with: `👉🏻 Nate's Library MCP: https://…`,
#: `➡️ Twitter: https://…`. Compared as a prefix, not a pattern, so the skin-tone modifier
#: on `👉🏻` and the variation selector on `➡️` both fall inside the match. Such a line
#: normally carries its URL too and is already a link line; it earns its own test at the
#: clip boundary, where `clip_body` has cut the URL off and left the pointer behind, which
#: is exactly how #1861's stored row ends.
_ARROW_PREFIX = ("👉", "➡", "→", "📌")

#: The channel-intro anchors (#1900). TheAIGRID opens EVERY description with the same
#: paragraph — `Welcome to TheAIGRID — the place to learn AI for free. … Subscribe to
#: start learning AI for free…` — and because it sits at offset 0 and is the WHOLE
#: description, the two footer anchors cannot reach it: both scan a TAIL. This is the
#: same closed-list discipline as `_FOOTER_LABEL_RE`, and the same reason it is a list:
#: a promo classifier's mis-fire eats the description's opening paragraph, which is
#: #856's failure mode. Three phrases, each one observed in the stored corpus
#: (1 row in 1078 youtube rows), and `_greeting_is_whole_body` requires the whole
#: description to be nothing else.
_GREETING_ANCHOR_RE = re.compile(
    r"^\W*?(?:welcome\s+to\b|this\s+channel\b|subscribe\s+to\b)\b", re.IGNORECASE)

#: The same three phrases anywhere inside a LINE (#1900), each a named kind. Two uses,
#: both of them abstentions-by-narrowing: every non-blank line of the description must
#: contain one of them (`_greeting_is_whole_body`), and at least two DIFFERENT kinds must
#: occur in the text. The second is what refuses the shape the first cannot: a line of
#: real news that merely opens with `Welcome to my deep dive today, which covers …` uses
#: one kind and is the video's subject, while a channel's standing intro repeats itself
#: across all three.
_GREETING_PHRASE_RE = re.compile(
    r"(?P<welcome>welcome\s+to\b)|(?P<channel>this\s+channel\b)"
    r"|(?P<subscribe>subscribe\s+to\b)", re.IGNORECASE)


def _greeting_kinds(text: str) -> set:
    """Which of the three intro phrases occur in `text`, as a set of kind names."""
    return {m.lastgroup for m in _GREETING_PHRASE_RE.finditer(text) if m.lastgroup}

#: The verbs a signup line opens with: `👉 Join the free GPT-6 Astra Crash Course here:
#: https://…`, `➡️ Sign up for the waitlist: https://…`. `_link_run_bounds` needs TWO
#: such lines to call them a run (`body.py`, the floor of two), and one arrow line is
#: how a channel signs off, so the run rule cannot remove it — that is the shape #1900
#: found published. A verb list, not a sentence model: `👉 My repo: https://…` and
#: `➡️ Twitter: https://…` are pointers to the video's subject and are NOT signup asks,
#: so they stay, and they are the two cases the tests below pin as abstentions.
_SIGNUP_CTA_RE = re.compile(
    r"^(?:join|sign\s?up|register|enroll|claim|apply)\b", re.IGNORECASE)

#: What to skip between an arrow and the word after it (`👉🏻 Join`, `➡️  Sign up`),
#: including the skin-tone modifier and the variation selector.
_LEADING_NON_WORDS_RE = re.compile(r"^[^\w]+")

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

#: A contact address on a line of its own channel's copy: `collabs@nouralabs.com`.
#: `_URL_RE` cannot see it — an address is not a `http://` or `www.` — and the
#: rule-free footer of #1819 is made of exactly these.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

#: `0:00 — Intro`, `0:41 — How Weco's AIDE²…`, `1:02:03`. A timestamp opening a line is
#: a chapter marker, and a chapter list is the second half of the #1819 footer.
_CHAPTER_LINE_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\b")

#: The second anchor class (#1819): the labels a channel writes over the block it puts
#: at the end of a description when it writes NO rule line above it. AI Revolution ends
#: with `📩 Brand Deals & Partnerships: …`, `✉️ General Inquiries: …` and a
#: `What You'll See:` chapter list under a blank line, and `_RULE_LINE_RE` has nothing
#: to find, so #1561's strip never fires and the whole description is published. Same
#: discipline as `_FOOTER_LABEL_RE`: a CLOSED list of labels, not a "is this promo"
#: classifier, because a classifier's mis-fire eats the description's opening paragraph
#: (#856) and clause 4 (#1819) pins footer-less channel copy passing through untouched.
#: `\W*?` skips the leading emoji, which is why `_FOOTER_LABEL_RE`'s `inquiries?\b` never
#: matched `✉️ General Inquiries:` — `General` sits between the skipped characters and
#: the label.
_RULELESS_FOOTER_ANCHOR_RE = re.compile(
    r"^\W*?(?:brand\s+deals?|general\s+inquiries?|business\s+inquiries?|collabs?\b"
    r"|what\s+you.?ll\s+(?:see|cover|learn)|chapters?\b|timestamps?\b)\b", re.IGNORECASE)

#: The anchor inside that block whose whole tail IS its list: a channel writing
#: `What You'll See:` is announcing a chapter list, and the list runs to the end of the
#: description. Below one of these the lines are not individually classified — the
#: 2026-09-27 row's list is `How a suspected Gemini 4 Pro checkpoint is hiding in
#: Arena`, prose to any line-matching rule — so this heading, and not a guessed shape,
#: is what licenses removing the text under it.
_CHAPTERS_HEADING_RE = re.compile(
    r"^\W*?(?:what\s+you.?ll\s+(?:see|cover|learn)|chapters?\b|timestamps?\b)\b",
    re.IGNORECASE)

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

    The `body` of the first form is built from exactly the lines `BODY_FLOOR` was
    measured on. Three kinds of line are counted by neither and returned by neither:
    a heading marker (#1925: returning it pasted the author's `# Description` into a
    section the floor had judged on its content), a task-list checkbox line together
    with its wrapped tail, and an issue trailer nobody filled in (`Fixes # (issue)`).
    The last two are #2012: an upstream PR checklist cleared the floor as prose and was
    published as knowledge. Everything else — prose, plain list items, URLs — is
    returned unchanged and in the order upstream wrote it. See `_HEADING_LINE_RE` for
    the fenced-code case this line-based strip costs.
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
    in_checkbox = False
    for line in body.splitlines():
        if _SCAFFOLD_LINE_RE.match(line):
            saw_scaffold = True
            in_checkbox = bool(_CHECKBOX_LINE_RE.match(line))
            continue
        if in_checkbox and line.strip() and not _BLOCK_OPENER_RE.match(line):
            # The wrapped tail of the checkbox line above: one list item, hard-wrapped
            # upstream. Counted, it cleared the floor on its own (51 characters of
            # `active release branch after it merges into …`).
            continue
        in_checkbox = False
        if _UNFILLED_ISSUE_TRAILER_RE.match(line):
            continue
        content.append(line)
    content_text = " ".join(" ".join(content).split())
    # The value that gets pasted is the lines the floor above was measured on, and no
    # others (#1925 for headings, #2012 for checkbox lines and the unfilled trailer):
    # a line that does not count towards a body being worth publishing is not part of
    # what is published.
    published = _collapse_blank_runs("\n".join(content))

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
    return published, None


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


def _is_footer_tail_line(line: str) -> bool:
    """A line of a rule-free footer: a promo label, a contact address, a URL, or a
    timestamped chapter line. Narrow on purpose — inside an already-anchored block these
    are the shapes a channel's closing block is made of, and anything else is prose the
    strip must leave alone."""
    return bool(_URL_RE.search(line) or _EMAIL_RE.search(line)
                or _CHAPTER_LINE_RE.match(line)
                or _RULELESS_FOOTER_ANCHOR_RE.match(line)
                or _FOOTER_LABEL_RE.match(line))


def _is_link_run_line(line: str) -> bool:
    """A line that is a link, a pointer to one, or a heading over some — and is not a
    sentence.

    Stricter than `_is_link_block_line`, which classifies lines INSIDE a block #1561 has
    already anchored: there, the anchor carries the "this is promo" judgement and the tail
    test only has to exclude prose. A run found without any anchor carries no such
    evidence, so every shape in it has to be a link shape on its own: a URL, a label from
    the closed `_FOOTER_LABEL_RE` list, an arrow-prefixed pointer, or a line that is nothing
    but a short colon-terminated label (`Full post:`). Emails, `0:00 —` chapter lines and
    plain prose are deliberately NOT link-run lines, which is what keeps a contact block or
    a chapter list from being cut out of the middle of a description.
    """
    return bool(_URL_RE.search(line)
                or _FOOTER_LABEL_RE.match(line)
                or line.lstrip().startswith(_ARROW_PREFIX)
                or _RUN_LABEL_RE.match(line))


def _greeting_is_whole_body(lines: list) -> bool:
    """True when the description is NOTHING but a channel's standing self-intro (#1900).

    Three conditions, each one narrowing, all three asked of the text's non-blank LINES
    (a description is unwrapped in practice, so a line is a paragraph and a paragraph that
    continues on the next line is a line that is not the intro):

    1. The first non-blank line OPENS with a phrase from `_GREETING_ANCHOR_RE`.
    2. EVERY non-blank line contains one of the three phrases somewhere. This is what
       refuses news that continues under the greeting, whether below a blank line or on
       the very next line with no blank between — a sentence about Manus 2.0 or a model
       release carries none of `welcome to` / `this channel` / `subscribe to`.
    3. At least TWO DIFFERENT phrases occur (`_greeting_kinds`). This is what refuses the
       shape condition 2 alone lets through, because it is a whole single line: `Welcome
       to my deep dive today, which covers how Manus 2.0 gave Cue agents their own phone`
       opens with the anchor, is the only line, and is the video's subject — one kind, so
       not an intro. A channel's standing boilerplate always repeats itself across the
       three, as TheAIGRID's does (`Welcome to TheAIGRID …`, `this channel gives you`,
       `Subscribe to start learning AI for free`).

    The residual over-reach is worth naming rather than hiding: a description whose lines
    ALL happen to mention one of three phrases, and two of them different, is treated as
    the intro even if some of it is news, and a description hard-wrapped so that a middle
    line carries no phrase is NOT stripped. Both follow from a closed list of three
    phrases, the same trade `_FOOTER_LABEL_RE` makes; the alternative #1819's triage ruled
    out is a classifier that eats opening paragraphs, which is #856's failure mode.
    Under-reach costs one more day of an advert in the note; over-reach deletes news.

    Why this needs a rule at all when the footer anchors exist: they scan backwards from
    a rule line or a footer label, so a block at offset 0 is outside their reach by
    construction, and the writer's footer-only ruling is never reached either — with
    nothing stripped, `stripped == summary` returns the ad verbatim (`vault_writer.py:426`).
    TheAIGRID's greeting was 100 % of the published body for exactly that reason.
    Returning "" is what the caller already understands as "the body WAS the footer":
    `ends_a_sentence("")` is False, and the entry carries the scorer's `why` instead.
    """
    rows = [ln for ln in lines if ln.strip()]
    if not rows or not _GREETING_ANCHOR_RE.match(rows[0]):
        return False
    if any(not _GREETING_PHRASE_RE.search(ln) for ln in rows):
        return False
    return len(_greeting_kinds("\n".join(rows))) >= 2


def _is_signup_cta_line(line: str) -> bool:
    """A line that is ONLY an arrow-prefixed signup ask carrying a URL (#1900).

    Three things on the one line: it starts with an `_ARROW_PREFIX` arrow, the first
    word after that arrow is a verb from the closed `_SIGNUP_CTA_RE` list, and the line
    carries a URL. Drop any of the three and the line is something else — a pointer to
    the video's own subject (`👉 My repo: https://…`), or an ask with no destination —
    and it is left in.
    """
    stripped = line.strip()
    if not stripped.startswith(_ARROW_PREFIX):
        return False
    return bool(_SIGNUP_CTA_RE.match(_LEADING_NON_WORDS_RE.sub("", stripped))
                and _URL_RE.search(stripped))


def _drop_signup_cta_lines(lines: list) -> list:
    """The lines with every set-off signup line taken out, or `lines` itself (#1900).

    Set-off (`_set_off`) is what makes a LINE droppable rather than a sentence: an ask
    wrapped into the middle of a paragraph has no blank above it, and cutting a line out
    of a paragraph would edit prose the channel wrote. The blank that set the line off
    goes with it, so the prose around the removal keeps the spacing it had —
    `prose\\n\\nCTA\\n\\nmore prose` comes back as `prose\\n\\nmore prose`, not with a
    doubled gap — and every other line is returned byte-for-byte.
    """
    if not any(_set_off(lines, i) and _is_signup_cta_line(ln)
               for i, ln in enumerate(lines)):
        return lines
    out: list = []
    for i, line in enumerate(lines):
        if _set_off(lines, i) and _is_signup_cta_line(line):
            # Take the blank above only when the line ENDED its paragraph, because then
            # that blank was the paragraph's own separator. A signup line that HEADS a
            # link run has a link line below it, and its blank is the run's separator —
            # eating it would weld the run onto the prose above and take the run out of
            # `_link_run_bounds`' reach (its `_set_off` check needs that blank). Keeping
            # it leaves the run set off, which is the shape #1861 already handles.
            ends_paragraph = i == len(lines) - 1 or not lines[i + 1].strip()
            if ends_paragraph and out and not out[-1].strip():
                out.pop()
            continue
        out.append(line)
    return out


def _link_run_bounds(lines: list) -> Optional[tuple]:
    """The `(start, end)` line indices of a set-off run of link lines, or None (#1861).

    `end` is the first line AFTER the run, so `lines[start:end]` is the block and
    `lines[:start] + lines[end:]` is everything the channel wrote around it.

    A run is two or more non-blank link lines with at least one blank between them and the
    line above (`_set_off`) and the line below. The floor of two is what makes it a RUN: a
    lone `My Links 🔗` over a paragraph is how a description heads its own sections, and one
    heading line is not evidence of a promo block.

    Unlike the two footer anchors this does not require the rest of the text to be links.
    #1561 and #1819 strip a FOOTER, so they must prove the whole tail is links — and that is
    precisely why neither of them can see a link block sitting in the MIDDLE of a
    description with the video's own prose below it, the shape that was published into the
    digest on 2026-09-29. Every line after the run is returned unchanged.

    The abstention is the run's own prose, not its surroundings: one sentence among the link
    lines and there is no run to cut. That is the same protection `_ruleless_footer_start`
    applies to a footer's tail, moved from "is this the end" to "is this a block".
    """
    for i, line in enumerate(lines):
        if not line.strip() or not _is_link_run_line(line) or not _set_off(lines, i):
            continue
        j, non_blank, last = i + 1, 1, i
        while j < len(lines) and (not lines[j].strip() or _is_link_run_line(lines[j])):
            if lines[j].strip():
                non_blank += 1
                last = j
            j += 1
        closed_below = last + 1 >= len(lines) or not lines[last + 1].strip()
        if non_blank >= 2 and closed_below:
            return (i, j)
    return None


def _ruleless_footer_start(lines: list) -> Optional[int]:
    """Where a promotional footer with NO separator rule above it starts, or None (#1819).

    The candidate is the FIRST line that is a label from the closed anchor list and is set
    off from the line above it — the same blank-line discriminator `_set_off` applies to a
    rule, for the same reason: a channel's closing block sits under a blank, and prose
    welded to the line above it is not a footer. First, not last, because the block starts
    with its contact lines and a strip anchored on the `What You'll See:` heading below
    them would publish the addresses.

    Everything from there to the end has to be footer for anything to be removed: contact
    and label lines, emails, URLs, `0:00 —` chapter lines (`_is_footer_tail_line`), or
    chapter content once a `_CHAPTERS_HEADING_RE` heading has been reached — below such a
    heading the list is the tail, so those lines are not classified one by one. The first
    line that is none of those abandons this candidate, and the scan moves on to any later
    anchor; a description that merely contains a contact line above more prose keeps all
    of it.
    """
    for idx, line in enumerate(lines):
        if not _RULELESS_FOOTER_ANCHOR_RE.match(line) or not _set_off(lines, idx):
            continue
        chapters = bool(_CHAPTERS_HEADING_RE.match(line))
        for below in lines[idx + 1:]:
            if not chapters and _CHAPTERS_HEADING_RE.match(below):
                chapters = True
            if chapters or not below.strip() or _is_footer_tail_line(below):
                continue
            break
        else:
            return idx
    return None


def strip_link_footer(text: str) -> str:
    """Remove a trailing promotional footer from channel-authored text (#1561, #1819).

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

    #1819 added the second anchor, tried only when the text contains no set-off rule at
    all: AI Revolution's footer is `📩 Brand Deals & Partnerships: …` over
    `✉️ General Inquiries: …` and a `What You'll See:` chapter list, under a blank line
    and with no rule anywhere, so an anchor that is only ever a rule cannot see the most
    common footer on the feed. `_ruleless_footer_start` anchors on a closed list of those
    labels instead, under the same set-off test, and removes the block only while its
    lines keep looking like footer. A rule with prose under it still abstains outright:
    the label scan runs after that `return`, never instead of it.

    What is kept is returned as it was: the strip removes the block and does not rewrap
    or re-case the prose it was attached to.

    #1900 added the two shapes that are not a tail, so no backwards anchor can see them:
    `_greeting_is_whole_body`, when a channel's standing self-intro is the ENTIRE
    description (TheAIGRID's greeting reached the digest as the whole body on
    2026-09-30), which returns "" so the writer renders `why`; and
    `_drop_signup_cta_lines`, which removes a single set-off `👉 Join … https://…` line
    that `_link_run_bounds` cannot call a run because its floor is two link lines. A
    greeting above real prose is returned byte-for-byte, as is any arrow line missing
    the signup verb or the URL — neither rule is a licence to prefer `why` over the
    channel's own copy, which is the caution #1819 clause 4 pins for the footer anchors.

    Returns "" for a body that WAS nothing but the footer; the caller decides what an
    empty body means (`ends_a_sentence` below is the writer's test for that).
    """
    if not text:
        return ""
    lines = text.splitlines()
    # The two shapes that are not a TAIL, so no backwards anchor can reach them (#1900):
    # a channel-intro greeting that is the whole description, and one set-off signup line.
    # Both are properties of this text as a whole, so they are asked before the scan; the
    # CTA drop rebinds `text` as well as `lines`, because the abstention below returns
    # `text.rstrip()` and it must return the text this function actually looked at.
    if _greeting_is_whole_body(lines):
        return ""
    kept = _drop_signup_cta_lines(lines)
    if kept is not lines:
        lines = kept
        text = "\n".join(lines)
    for idx in range(len(lines) - 1, -1, -1):
        if not _RULE_LINE_RE.match(lines[idx]) or not _set_off(lines, idx):
            continue
        tail = [ln for ln in lines[idx + 1:] if ln.strip()]
        if not tail or all(_is_link_block_line(ln) for ln in tail):
            return "\n".join(lines[:idx]).rstrip()
        # A rule with prose under it is section furniture, not a footer. Anything
        # above that rule is the description too, so there is nothing to take off —
        # and no second anchor is tried, which is what keeps #1561's abstention whole.
        return text.rstrip()
    # No set-off rule anywhere in the text: the shape #1561's anchor cannot reach (#1819).
    #
    # A footer is not the only block that can sit where an anchor cannot reach it. The same
    # link block in the MIDDLE of a description, with the video's own prose below it, is
    # what this function shipped into the digest on 2026-09-29 (#1861): both anchors above
    # require the block to be the tail, and this one was not. Take the run out FIRST and let
    # the footer anchor work over what is left, because the two cut different parts of one
    # description and the recorded row needs both: the run at lines 2-9, the footer at its
    # `Chapters:` heading. Any order that asks the footer question first returns
    # `lines[:start]`, which still holds the block it never looked at.
    run = _link_run_bounds(lines)
    if run is not None:
        lines = lines[:run[0]] + lines[run[1]:]
    start = _ruleless_footer_start(lines)
    if start is not None:
        return "\n".join(lines[:start]).rstrip()
    return "\n".join(lines).rstrip() if run is not None else text.rstrip()


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
