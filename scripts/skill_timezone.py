"""Skill templates must not state a clock in the wrong zone — stated once, and
checked where writes land. Two rules share this module and its two consumers: a
hand-typed season abbreviation beside a displayed-time token, and (since #2454) a
commit subject whose date interpolation reads the box's local day.

Why this is a module and not only a test
----------------------------------------
The same defect was filed five times — ``app/post_capture.py``'s auto-captured
heading asserted `PDT` year-round (#601), ``morning-brief-and-triage`` typed
`HH:MM PST` with no clock step (#1079), ``nightly-reflection-signals`` typed
`generated: … PST` (#1080), three capture skills typed `### Session HH:MM PDT`
(#1081) — and #1112 named the reason it kept coming back: the machine check for
exactly this class was gate-passed on branches ``9c09e9e``/``6620216`` and
never merged, so every fix was prose plus a hand-run grep and the next edit to
the same heredoc could silently restore the literal. #1189 is that umbrella.

This is the architecture ``reflection_archive`` documents, followed literally:
the definition lives here and gets two consumers.

* ``scripts/automod/vault_round.py::skill_timezone_errors`` calls
  :func:`template_clock_violations` on every ``skills/**/SKILL.md`` a vault
  round touches — **the writer**. A template that regains a hardcoded zone
  abbreviation is refused at ``automod_vault_land``, before the commit, and the
  round's paths revert.
* ``tests/test_skill_timezone_literals.py`` imports the same function: its
  mutations (unmarked, so every gate rung runs them) prove the rule can fail on
  the exact strings history typed, and its live-vault group asserts the real
  skills against it.

Stdlib only, and it does not import ``prompt_builder`` at module level, for the
reason ``reflection_archive`` states: ``vault_round`` runs its validators with
the round's interpreter warm and the loaders in a fresh one.

What counts, and what must not
------------------------------
A *zone abbreviation* — the three letters ``PST`` or ``PDT``, as a word — sitting
on a line that is **template content**, in either of two shapes a template
displays a time:

1. a heading/header carrying a displayed-time placeholder (``### Session HH:MM
   PDT — Title``, ``Brief + Triage — YYYY-MM-DD HH:MM PST``);
2. a report-header field (``generated:``/``updated:``/``created:``/``timestamp:``
   /``date:``) whose value ends in one (``generated: YYYY-MM-DD HH:MM PST``).

Two shapes, not three, because the third candidate was measured and rejected:
matching the *quoted* strftime token ``%H:%M`` beside a zone abbreviation fires
on ``Use yesterday's date (PST) to read
`~/obsidian/memory/YYYY-MM-DD.md``` — a path template in escape-name position —
and on any sentence naming the seasons next to a time format. Measured over all
194 active skills the two retained shapes have zero false positives and catch
every string #1189's triage measured; a third shape with a known false positive
would earn its own removal (the way a noisy alert gets silenced), so the fix
side must instead derive the abbreviation from a clock — a header whose value is
``date -u``'s ISO-8601 ``Z`` stamp, or a paste of
``TZ=America/Los_Angeles date '+%H:%M %Z'``, neither of which spells the
abbreviation as a word and so cannot be matched by a zone-word rule at all.

Prose *about* the zone is not template content and stays untouched, by two
carve-outs, each pinned by a mutation test:

* **inline code** (backticks): a line discussing a token is teaching, not
  displaying it — ``the brief injected `Brief + Triage — 2026-09-17 02:53 PDT````,
  ``a hand-typed `PST` anywhere is a defect``, and ``all times in PST/PDT
  (local) — the abbreviation comes from `%Z` in Step 0a, never typed`` all sit in
  live skills. Fencing is not exempted wholesale, only the span between
  backticks, so the fenced template lines themselves are still judged;
* **schedule and convention prose** — ``Runs at 6:00 AM PST daily``,
  ``Time (PST)`` — which has no displayed-time placeholder and no header field,
  and is why the shapes above require one.

That boundary is not a convenience: the dedupe that merged #1079 and #1112 into
unrelated items matched on exactly this shared timezone vocabulary (lexical
0.105 and 0.125). Vocabulary must never be what a check alarms on.

The second rule: a clock claim written into a commit subject (#2454)
--------------------------------------------------------------------
A commit subject built with ``$(date +%Y-%m-%d)`` states a date in the box's zone
while the commit's own author date is UTC, so on this UTC-7 host the subject
carries the *previous* calendar day for the seven hours either side of UTC midnight
— 8 h of disagreement once DST lands. ``#2227`` swept seven such subjects to
``$(date -u …)`` and shipped no rail, so the next edit deleting those two letters
re-enters through ``automod_vault_land`` unrefused; that is the regression
:func:`_subject_spans` now refuses, judged on the quoted message argument of
``git commit -m`` / ``vault-commit.sh`` only.

The scope difference from the zone rule is deliberate and pinned by mutations: a
``$(date …)`` in a filename or an env value on the same line makes no claim about the
commit, while a ``date -u`` *sibling* on that line cannot excuse a local subject —
which is exactly how #2227's own line-level ``grep -v 'date -u'`` under-reported
(printing 6 where the truth was 7), and why line-level filtering is not the check.
"""

from __future__ import annotations

import re

#: A season's zone abbreviation, as a word. These two letters are the whole
#: vocabulary of this class on this host; a second host's zone would add
#: members to this pattern, not change the rule.
ZONE_ABBR = re.compile(r"\bP[SD]T\b")

#: Inline code spans. An abbreviation between backticks is being *discussed*
#: ("a hand-typed `PST` anywhere is a defect", "`PDT` invented by that run"),
#: which is the skill teaching the rule this module enforces — flagging it
#: would make the fix for the class itself unlandable. Only the span is
#: removed; fenced template blocks are judged as written.
CODE_SPAN = re.compile(r"`[^`\n]*`")

#: Shape 1 — a displayed-time *placeholder*: the template forms in which a
#: time gets written for later substitution (`HH:MM`, `YYYY-MM-DD HH:MM`).
#: Deliberately the literal placeholder and not a measured time: `08-29 08:18
#: PDT` inside a sentence is a record of one past run, not a template future
#: runs copy — triage evidence itself cites such a line (#1079's
#: degraded-baseline note), and alarming on it would make the check lie about
#: history instead of about the future.
TIME_PLACEHOLDER = re.compile(r"\bHH:MM\b")

#: Shape 2 — a report-header field at the head of a line (bullet/quote prefixes
#: allowed). `generated:` is what the nightly reports use (#1080's sighting);
#: the sibling names are the same field under other skills' spellings, so a
#: renamed header cannot duck the rule.
#: A table cell is as much a template as a bullet is, so `|` joins the line
#: prefixes — while a table *column header* like `| generated | 2026-09-03
#: 22:35 PST |` stays clean because the field name must be followed by a colon.
HEADER_FIELD = re.compile(r"^\s*(?:[-*>|]\s*)?(?:generated|updated|created|timestamp|date)\s*:")


#: Shape 3 (#2454) — the second clock a template can get wrong: not a displayed
#: time a reader misparses, but a stamp that is silently written wrong. The two
#: commit invocations the skills tree writes messages with. ``git commit`` takes its
#: message from the argument a ``-m`` flag introduces and ``vault-commit.sh`` from
#: its first quoted positional, so a subject is only ever one of those spans — which
#: is what lets a ``$(date …)`` in a filename or an env value on the same line stay
#: out of scope.
COMMIT_CMD = re.compile(r"(?:\bgit\s+commit\b|\bvault-commit\.sh\b)")
_GIT_MESSAGE = re.compile(r"""(?:^|[ \t])(?:-m|--message)[ \t=]*(?P<s>"[^"\n]*"|'[^'\n]*')""")
_WRAPPER_MESSAGE = re.compile(r"""^[ \t]*(?P<s>"[^"\n]*"|'[^'\n]*')""")
#: A ``date`` call substitution inside a template. An invocation carrying an
#: explicit ``TZ=`` prefix is deliberately not matched: it states the zone it means,
#: which is the fix the first rule points at rather than this defect.
DATE_SUBST = re.compile(r"\$\(\s*date\b(?P<args>[^)]*)\)")
#: The flag that makes the invocation UTC. Token-anchored, so a format directive
#: that happens to end in ``u`` (``%Hu``, ``%-u``) is not read as the flag.
UTC_FLAG = re.compile(r"(?:^|\s)(?:-u|--utc)(?:\s|$)")


def _subject_spans(line: str) -> list[str]:
    """The quoted commit subjects on ``line``, in order, quotes included.

    Span-scoped rather than line-scoped, in both directions of the mistake:

    * A line carries other ``$(date …)`` calls that make no claim about the commit.
      The live corpus has ``| tee /tmp/preflight-$(date +%s).txt`` after a clean
      subject (``nightly-reflection-signals/SKILL.md:266``) and
      ``LLOYD_JOB_WRITES="…$(date -u +%F).md"`` before one
      (``nightly-reflection-config/SKILL.md:159``).
    * A line-level exemption is how #2227 under-reported: its own
      ``grep -v 'date -u'`` dropped a line whose only local clock read was inside
      the subject, because a ``date -u`` appeared somewhere else on it, and printed
      6 where the truth was 7.
    """
    spans: list[str] = []
    for cmd in COMMIT_CMD.finditer(line):
        tail = line[cmd.end():]
        if cmd.group(0).endswith("vault-commit.sh"):
            first = _WRAPPER_MESSAGE.match(tail)
            if first:
                spans.append(first.group("s"))
        else:
            spans.extend(m.group("s") for m in _GIT_MESSAGE.finditer(tail))
    return spans


def template_clock_violations(name: str, body: str) -> list[str]:
    """Why this skill template states a clock in the wrong zone, or [].

    Scoped to one body, like ``reflection_archive.skill_rule_violations``,
    because the vault writer is handed the one path a round touched — a
    pre-existing literal in an untouched skill must not block an unrelated
    round, and must not be able to hide either: the live-vault scan in
    ``tests/test_skill_timezone_literals.py`` judges the whole tree.

    The two rules read different versions of a line, on purpose:

    * The zone rule sees the line with inline code spans stripped, which is the
      documented exemption for code that legitimately needs local time
      (``$(TZ=… date …)``, ``local_timestamp = datetime.now()``) — and the residual
      it leaves, since exempted code can still name an abbreviation beside a
      displayed time, is pinned as a known gap rather than claimed refused.
    * The commit-subject rule sees the **raw** line, inline code included. For a
      command, backticks are markdown's way of saying "copy this", not "this word is
      under discussion" — the skills tree already documents subjects that way
      (``skills/github-pr-workflow/SKILL.md`` shows
      ``git add . && git commit -m "fix: ..." && git push``) — so stripping them
      would exempt the exact shape a regressed subject takes. Measured on the
      194-skill active corpus on 2026-10-09: 0 violations either way, so the wider
      scope costs nothing today.
    """
    out: list[str] = []
    for lineno, raw in enumerate(body.splitlines(), 1):
        line = CODE_SPAN.sub("", raw)
        if ZONE_ABBR.search(line):
            is_template = bool(TIME_PLACEHOLDER.search(line)) or bool(HEADER_FIELD.match(line))
            if is_template:
                out.append(
                    f"{name}: line {lineno}: hardcoded zone abbreviation beside a "
                    f"displayed-time token — interpolate it from a clock "
                    f"(TZ=America/Los_Angeles date '+%H:%M %Z') or emit an ISO-8601 Z "
                    f"stamp (date -u +%Y-%m-%dT%H:%M:%SZ) instead: {raw.strip()[:120]!r}"
                )
        for span in _subject_spans(raw):
            for local in DATE_SUBST.finditer(span):
                if UTC_FLAG.search(local.group("args") + " "):
                    continue
                out.append(
                    f"{name}: line {lineno}: interpolates `{local.group(0)}` into a "
                    f"commit subject, so the subject carries the box's local day "
                    f"while the commit's own author date is UTC — interpolate "
                    f"`$(date -u +…)` instead: {raw.strip()[:120]!r}"
                )
    return out
