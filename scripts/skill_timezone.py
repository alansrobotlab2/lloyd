"""Skill templates must not hand-type a season's timezone abbreviation beside a
displayed-time token — stated once, and checked where writes land.

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


def template_clock_violations(name: str, body: str) -> list[str]:
    """Why this skill template hand-typed a zone beside a displayed time, or [].

    Scoped to one body, like ``reflection_archive.skill_rule_violations``,
    because the vault writer is handed the one path a round touched — a
    pre-existing literal in an untouched skill must not block an unrelated
    round, and must not be able to hide either: the live-vault scan in
    ``tests/test_skill_timezone_literals.py`` judges the whole tree.
    """
    out: list[str] = []
    for lineno, raw in enumerate(body.splitlines(), 1):
        line = CODE_SPAN.sub("", raw)
        if not ZONE_ABBR.search(line):
            continue
        is_template = bool(TIME_PLACEHOLDER.search(line)) or bool(HEADER_FIELD.match(line))
        if not is_template:
            continue
        out.append(
            f"{name}: line {lineno}: hardcoded zone abbreviation beside a "
            f"displayed-time token — interpolate it from a clock "
            f"(TZ=America/Los_Angeles date '+%H:%M %Z') or emit an ISO-8601 Z "
            f"stamp (date -u +%Y-%m-%dT%H:%M:%SZ) instead: {raw.strip()[:120]!r}"
        )
    return out
