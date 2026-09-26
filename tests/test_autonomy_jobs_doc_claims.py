"""`architecture/autonomy-jobs.md`'s fleet-membership statements stay true.

The doc refuses to restate schedules — "a copy of them here would be wrong within
a week, which is exactly how [[autonomy]]'s own hand-maintained inventory of 19
tasks came to describe four retired jobs" (`:25-31`) — and that bargain only holds
if *membership* is complete, because membership is the one thing the doc does
hand-type. It went stale twice inside three days (#919 on the rung ladder, #1102
on the fleet), and both times the sentence that rotted was a count.

Task **#86, nightly-iv-metrics-series**, is the live proof: it has run nightly
since 2026-09-15, and at base the doc did not contain the token `86` at all
(`grep -cE '\\b86\\b|iv-metrics|iv_metrics' architecture/autonomy-jobs.md` → 0),
while `:38` still bounded the fleet at "24 through 85". Nothing caught it:
`workers/sources/arch_review.py` offers a group as a review unit by its *heading
name* only (`autonomy-jobs:Measure`, `config.yaml:1641`), so it never asks who is
*inside* a group, and `scripts/autonomy/validate_tasks.py` checks task-file
structure, not the doc.

So the checks here are deliberately **internal**: each derives its expectation
from another statement in the same doc — the group heading, the write-authority
table, the ids a paragraph actually names — rather than from the live fleet. An
internal check cannot go stale the way a hand-copied roster does, and it still
fails the moment two of the doc's own statements disagree, which is exactly how
this bug presented: the bound said 85, the newest job said 86.

The numbering note is the exception that proves the rule. It read "the fleet runs
24 through 86", a hand-typed ceiling, and a ceiling is false the moment the fleet
grows: #1102 set it to 86 and tasks #87–#90 were created on 2026-09-24/25, so the
literal rotted inside three days exactly as its own docstring predicted. It is now
a floor ("24 through 90 or beyond"), which no new task id can falsify, and it is
still the only `N through M` in `architecture/`. Every count the doc does type
carries the date it was true, and each is pinned to something re-derivable — the
pilot frozenset's line number against `autonomy.py` itself, the evidence verdicts
against the code path that writes them and against their own internal arithmetic,
the no-dependency count against the `depends_on` arrows the doc draws and §Distil's
own fleet size, and both membership tables against each other and against the four
newest jobs.

A second contract joined this file with item #1521 (2026-09-26): the
`_pipeline/` shorthand. The doc spells the prefix bare on twelve table and prose
lines, the root it names moved to `~/lloyd-data/` (`6426668b`, "Move all runtime
data out of the code tree into ~/lloyd-data", 2026-09-22), and at filing the only
sentence mapping shorthand to root was buried in §Distil — below the first table
row that uses the bare spelling, in a doc whose sibling architecture files all
spell the prefix the same bare way. The shorthand tests keep exactly one
definition, place it above every bare use, and make it point at [[data-home]],
which already owns the root-move rule (`architecture/data-home.md:33-34`).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "autonomy-jobs.md"
ARCH_DIR = ROOT / "architecture"

FUNCTIONS_HEADER = "| Function | Jobs | What it is for |"
TIERS_HEADER = "| Tier | Jobs | |"

#: The Measure-group jobs whose report has no scheduled consumer. Three
#: sentences in the doc count against this set — the "no consumer" heading, that
#: paragraph's closing "they are three of seven", and #78/#80's "same position as
#: three of the seven jobs in Measure" — so the set is stated once, here, and the
#: paragraph is checked to name exactly it. A job gaining or losing a consumer has
#: to move all three sentences together, which is the point.
NO_CONSUMER = {"#36", "#85", "#70"}

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "thirty": 30, "thirty-one": 31,
    "thirty-two": 32, "thirty-three": 33, "forty": 40,
}

#: The four jobs created 2026-09-24 and 2026-09-25 that neither membership table
#: listed: #87 #88 #89 (`research-agent`, one arXiv paper each → one knowledge
#: note) and #90 (`corpus-shape-trend`, the shape metrics over four nightly
#: corpora). Their absence was invisible to
#: `test_every_job_has_one_function_and_exactly_one_write_tier` because it
#: cross-checks the two tables against *each other*: symmetric omission passes.
NEW_JOBS = ("#87", "#88", "#89", "#90")
DIGESTS = ("#87", "#88", "#89")

#: The ten `depends_on` edges, as the wiring diagram's parent ─► child arrows.
#: Verified parent-by-parent on 2026-09-26: 74→48, 48→24, 39→42, 58→57, 42→38,
#: 40→39, 83→58, 47→40, 51→56, 57→56 (child→parent), i.e. these ten pairs
#: read the other way round.
TEN_EDGES = {
    ("#38", "#42"), ("#42", "#39"), ("#39", "#40"), ("#40", "#47"),
    ("#56", "#57"), ("#57", "#58"), ("#58", "#83"), ("#56", "#51"),
    ("#24", "#48"), ("#48", "#74"),
}


def _text() -> str:
    return DOC.read_text()


def _flat(text: str) -> str:
    """Prose with hard wraps collapsed, so a sentence can be matched whole.

    The doc wraps near 88 columns, so "three\nof the six jobs in Measure" is one
    sentence across three lines; matching a sentence against raw lines would
    match nothing and read as absence.
    """
    return re.sub(r"\s+", " ", text)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


def _ids(cell: str) -> list[str]:
    """The `#NN` job ids in a table cell or a stretch of prose, in the order written."""
    return [f"#{n}" for n in re.findall(r"#(\d+)", cell)]


def _measure_heading() -> str:
    m = re.search(r"^## (Measure: .+)$", _text(), re.M)
    assert m, "architecture/autonomy-jobs.md has no `## Measure:` group heading"
    return m.group(1)


def _measure_ids() -> list[str]:
    """The group's membership as the group's own heading states it.

    Every other membership statement in the doc points back at this line, so
    deriving the expectation from it is what keeps these tests from pinning a
    number that has itself already rotted.
    """
    return _ids(_measure_heading())


def _table(header: str) -> dict[str, tuple[list[str], str | None]]:
    """Rows under `header`: `{first cell: (job ids, trailing count cell or None)}`."""
    lines = _text().splitlines()
    i = lines.index(header)
    out: dict[str, tuple[list[str], str | None]] = {}
    for line in lines[i + 2:]:                        # header, then the |---|---| rule
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        count = cells[-1] if re.fullmatch(r"\d+", cells[-1]) else None
        out[cells[0]] = (_ids(cells[1]), count)
    return out


def _count(flat: str, pattern: str, what: str) -> tuple[int, int, str]:
    """The two number-words of a membership sentence, as `(numerator, denominator, matched text)`."""
    m = re.search(pattern, flat, re.I)
    assert m, f"no sentence matches /{pattern}/ — {what}"
    return _WORDS[m.group(1).lower()], _WORDS[m.group(2).lower()], m.group(0)


def _num(tok: str) -> int:
    """A count written either way the doc writes them: `36` or `thirty-six`."""
    return int(tok) if tok.isdigit() else _WORDS[tok.lower()]


def _functions_section() -> str:
    """§The functions and its three subsections, prose flattened.

    Stops at the first job-group heading so a claim checked "in §The functions" is
    not silently satisfied by the same words in a group section — §Distil, three
    sections on, states the pilot frozenset and the fleet size too, and the whole
    defect #1520 is about is those two sections disagreeing.
    """
    return _flat(_text().split("## The functions", 1)[1].split("\n## Ingest", 1)[0])


def _diagram_edges() -> set[tuple[str, str]]:
    """The arrows of the wiring diagram, as `{(parent, child)}`.

    The doc's answer to "how many jobs declare a dependency" is the sentence these
    arrows belong to, so reading the count off the diagram is reading it off the
    doc rather than off a constant — and a count that disagrees with its own diagram
    is a defect on either side. The lookahead keeps every hop of a chain: `#38 ─►
    #42 ─► #39` has to yield two edges, and a plain findall would eat `#42` in the
    first match and lose the second. The `──►` in the annotation column never
    matches, because no `#id` precedes it.
    """
    after = _text().split("Ten `depends_on` edges exist", 1)[1]
    block = after.split("```", 2)[1]
    assert block.startswith("\n#"), f"the wiring block is not a fenced diagram: {block[:40]!r}"
    return {(f"#{a}", f"#{b}") for a, b in re.findall(r"#(\d+)\s+─►\s+(?=#(\d+))", block)}


def _review_log() -> str:
    """The dated §Review log, which records what was true on the day it says."""
    log = _text().split("## Review log", 1)
    assert len(log) == 2, "architecture/autonomy-jobs.md has no §Review log to keep history in"
    return log[1]


def _live_prose() -> str:
    """Everything except the §Review log: the part a reader takes as current."""
    return _text().split("## Review log", 1)[0]


# ── clause 1 — #86 has an entry, and it states the four contract points ───────


def test_the_measure_group_lists_the_live_iv_metrics_series():
    """Clause 1, membership. #86 is `up_next`, `frequency: daily`, with eight
    run records since 2026-09-15, and appears nowhere in the document that says
    what each scheduled task is for.
    """
    assert "#86" in _measure_ids(), (
        "#86 nightly-iv-metrics-series is a live daily task with nightly runs "
        "since 2026-09-15 and no Measure-group entry, so the one document that "
        "says what a scheduled job is for is silent about a nightly primary-engine "
        "turn inside the 01:00-04:00 window the reflection chain also occupies")


def test_the_86_entry_states_its_frequency_its_subject_and_its_two_prohibitions():
    """Clause 1, content: its frequency, what it watches, that it appends one row
    per night to `_pipeline/reflection/iv-metrics.jsonl`, and that it must not
    write `usage.db` or hand-write a row.
    """
    section = _text().split("## Measure:", 1)[1].split("\n## ", 1)[0]
    row = next((ln for ln in section.splitlines() if re.match(r"\|\s*#86\s*\|", ln)), None)
    assert row, "#86 has no row in the Measure group's ID/Freq/Watches/Role table"
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 4, f"the #86 row needs the table's four cells, got {cells}"
    assert cells[1] == "daily", f"the task file declares `frequency: daily`; Freq cell reads {cells[1]!r}"
    assert "observer" in cells[2].lower(), f"the Watches cell must name its subject, reads {cells[2]!r}"

    assert "**#86" in section, "#86 has no prose entry in the Measure section"
    body = _flat(section.split("**#86", 1)[1].split("\n### ", 1)[0])
    for probe, why in (
        (r"one row per night", "one append per run is the invariant, not a style"),
        (r"_pipeline/reflection/iv-metrics\.jsonl", "where the series lands"),
        (r"nightly|daily", "how often it runs"),
        (r"dropped_rate", "the number it watches"),
        (r"usage\.db", "the store it must never write"),
        (r"hand-writ", "what a missing night must never be papered over with"),
    ):
        assert re.search(probe, body, re.I), (
            f"the #86 entry never states {why}; no /{probe}/ in {body[:300]!r}")


# ── clause 4 — the numbering note floors the range, and nothing else types it ─


def _numbering_note() -> str:
    m = re.search(r"^ordered: (.+)$", _text(), re.M)
    assert m, "the numbering note no longer starts a line with `ordered: `"
    return m.group(1)


def test_the_fleet_bound_floors_the_range_at_the_newest_task():
    """Clause 4, first half. This node was `test_the_fleet_bound_reads_24_through_86`
    and asserted that literal; #1520 retires the literal, so the name goes with it.

    The note is the last hand-typed fleet bound in the doc set. At base it read "24
    through 86" — a ceiling one job below the newest live task, which is the
    inference that made #86 invisible (#1102). #1102 repaired the string to 86, and
    #87, #88, #89 and #90 were created on 2026-09-24 and 2026-09-25, so the repaired
    literal was false within three days: a ceiling is re-falsified by every task
    someone files. What is asserted now is a floor — the range must reach at least
    the newest live id and must say it stops nowhere ("or beyond") — which is true
    for every fleet that will ever succeed this one.
    """
    note = _numbering_note()
    ranges = re.findall(r"\b(\d+) through (\d+)\b", note)
    assert len(ranges) == 1, f"expected exactly one id range in the numbering note: {note!r}"
    low, high = int(ranges[0][0]), int(ranges[0][1])
    assert low == 24, f"the note's range starts at {low}, not 24 — 24 is the oldest live task id"
    # The ceiling to beat is read off the doc's own membership tables, not typed
    # here: clause 5 requires the newest four jobs to be listed, so a job added to
    # the tables raises this floor without anyone remembering to raise a constant —
    # and a job omitted from the tables is clause 5's failure, not this one's silence.
    newest = max(int(i[1:]) for i in
                 [j for ids, _ in _table(FUNCTIONS_HEADER).values() for j in ids])
    assert high >= newest, (
        f"{note!r} tops out at {high} while the membership tables list job #{newest} "
        f"— the defect #1102 closed for 86 and #1520 for 90")
    assert "or beyond" in note, (
        f"{note!r} states a ceiling again: it is false the day after the next job is "
        f"filed, which is what let '24 through 86' rot twice")


def test_no_second_fleet_bound_appears_anywhere_in_architecture():
    """Clause 4, second half: repairing one bound must not seed others. Every
    `N through M` in `architecture/` is enumerated, so an integer copied into
    `index.md` — whose "each of the 32 scheduled jobs" count rotted inside three
    days — fails here instead of in next month's arch review.

    What is pinned is the *number* of typed ranges, their file, and that the one
    range travels with its date; not which line it lands on. The doc wraps near 88
    columns, so a rewrap is not a defect and an exact-string pin would fail on
    formatting — that is how a bound comes to look "not our problem" and stop being
    checked.
    """
    found = {md.name: [ln.strip() for ln in md.read_text().splitlines()
                       if re.search(r"\b\d+\s+through\s+\d+\b", ln)]
             for md in sorted(ARCH_DIR.glob("*.md"))}
    found = {k: v for k, v in found.items() if v}
    assert set(found) == {"autonomy-jobs.md"}, (
        f"a fleet id range is typed somewhere it was not before: {found}")
    lines = found["autonomy-jobs.md"]
    assert len(lines) == 1, (
        f"autonomy-jobs.md now types {len(lines)} id ranges: {lines} — one bound is "
        f"the agreed shape, a second one is a second thing to rot")
    assert "or beyond" in lines[0], (
        f"the one id range reads {lines[0]!r} without the floor wording, so it is a "
        f"ceiling again — or some other sentence has taken the numbering note's place")
    near = _flat(_text()).split("or beyond", 1)[1][:160]
    assert "tasks were retired" in near, (
        f"the sentence after 'or beyond' is no longer the numbering note: {near[:80]!r}")
    assert re.search(r"\(\d{4}-\d{2}-\d{2}\)", near), (
        f"the typed range carries no date near it: {near[:80]!r} — an id that reaches "
        f"90 is a measurement of one day, and the day has to travel with it")
    # The same integer can also arrive wearing a validator's clothes: this doc
    # used to close the #85 entry with "all 33 task files pass them", which read
    # as a test result and was the fleet size, correct only until task 87.
    typed = re.findall(r"\b\d+ task files\b", _text())
    assert not typed, (
        f"the doc types the number of task files ({typed}); it is the fleet size in "
        f"another form and rots the same way — say 'every task file'")


# ── clause 3 — every statement that counts the Measure group ─────────────────


def test_the_functions_table_row_and_every_anchor_follow_the_group_heading():
    """Clause 3, first half. Membership is restated as link text in the function
    table and as an anchor in four places (`:50`, `:475`, `:551`). Rewriting the
    heading and leaving the anchors silently breaks the jump while the row still
    reads fine, so the expected anchor is *derived* from the heading rather than
    asserted from a string.
    """
    heading = _measure_heading()
    ids = _measure_ids()
    funcs = _table(FUNCTIONS_HEADER)
    label = next((k for k in funcs if k.startswith("[Measure]")), None)
    assert label, f"no Measure row in the function table; parsed {list(funcs)}"
    assert funcs[label][0] == ids, (
        f"the function table lists {funcs[label][0]} for Measure, its group "
        f"heading lists {ids}")

    want = _slug(heading)
    anchors = re.findall(r"\]\(#(measure[^)]*)\)", _text())
    assert anchors, "nothing links to the Measure group at all"
    assert set(anchors) == {want}, (
        f"anchors point at {sorted(set(anchors) - {want})} while the heading's slug "
        f"is {want!r} — the row reads correctly and every jump into it goes nowhere")


def test_every_count_of_the_measure_group_matches_its_membership():
    """Clause 3, second half. Four sentences count this group: the "change
    nothing" line under the heading, the no-consumer heading and its closing
    clause, and #78/#80's cross-reference from Bound entropy. Each denominator
    must equal the group's own id list and each numerator must be re-derivable
    from the doc — a numerator copied from a previous membership is the bug.
    """
    ids = _measure_ids()
    flat = _flat(_text())

    n, d, said = _count(flat, r"(\w+) of the (\w+) change nothing",
                        "the Measure section no longer counts its non-writers")
    acting = _table(TIERS_HEADER)["Acts on the fleet itself"][0]
    assert d == len(ids), f"'{said}': denominator {d}, but the heading lists {len(ids)} — {ids}"
    assert n == len([i for i in ids if i not in acting]), (
        f"'{said}': {len(ids) - len(acting)} of {ids} sit outside the acting tier {acting}")

    n, d, said = _count(flat, r"(\w+) of the (\w+) jobs in Measure",
                        "#78/#80's entry no longer relates them to the Measure group")
    assert d == len(ids), f"'{said}': denominator {d}, group is {len(ids)} — {ids}"
    assert n == len(NO_CONSUMER), f"'{said}' counts {n}; the consumerless set is {sorted(NO_CONSUMER)}"

    n, d, said = _count(flat, r"(\w+) of the (\w+) have no consumer",
                        "the no-consumer claim is no longer a count of the group")
    assert d == len(ids), f"'{said}': denominator {d}, group is {len(ids)} — {ids}"
    assert n == len(NO_CONSUMER), f"'{said}' counts {n}; the consumerless set is {sorted(NO_CONSUMER)}"

    n, d, said = _count(flat, r"they are (\w+) of (\w+), which is a pattern",
                        "the no-consumer paragraph no longer restates its own count")
    assert d == len(ids), f"'{said}': denominator {d}, group is {len(ids)} — {ids}"
    assert n == len(NO_CONSUMER), f"'{said}' counts {n}; the consumerless set is {sorted(NO_CONSUMER)}"


def test_the_no_consumer_paragraph_names_exactly_the_jobs_its_counts_claim():
    """The set `NO_CONSUMER` is an assertion about prose, so the prose is checked
    against it: the paragraph must name those ids and no other member. This is
    what stops a fifth sentence quietly making #86 the fourth one.
    """
    blocks = [b for b in _text().split("\n\n") if "have no consumer" in b]
    assert len(blocks) == 1, f"expected one no-consumer paragraph, found {len(blocks)}"
    ids = _measure_ids()
    after = blocks[0].split("have no consumer.**", 1)[1]
    named = {i for i in _ids(after) if i in ids}
    assert named == NO_CONSUMER, (
        f"the paragraph names {sorted(named)} as the jobs with no consumer, but "
        f"the group's four counting sentences all say {sorted(NO_CONSUMER)}")


# ── clause 4 — write authority: one tier per job, counts that match their rows ─


def test_86_appears_in_exactly_one_write_authority_tier():
    """Clause 4, first half. #86 appends to its own metrics series and to nothing
    else, which is what #82 does when it "records the trend", so it belongs beside
    it under Reports only. What is under test is not the choice of tier but that
    a job appears in exactly one of them.
    """
    tiers = _table(TIERS_HEADER)
    hits = [t for t, (ids, _) in tiers.items() if "#86" in ids]
    assert hits == ["Reports only"], (
        f"#86 is in {hits or 'no tier'}; the tiers are the doc's answer to 'what "
        f"may this job do when nobody is watching', so it must sit in exactly one")


def test_68_is_not_counted_as_a_durable_writer():
    """#1505. #68's output is an inject into a live session (an entry with a
    TTL) plus its own 24 h dedup state; it writes no vault file. The tier
    table is the doc's answer to "what may write while nobody is watching", so
    counting an injector there hides where the durable exposure really is.
    """
    tiers = _table(TIERS_HEADER)
    hits = [t for t, (ids, _) in tiers.items() if "#68" in ids]
    assert hits == ["Injects expiring context unattended"], (
        f"#68 is in {hits or 'no tier'}; its job writes no durable state")


def test_every_tier_row_count_equals_the_ids_listed_beside_it():
    """Clause 4, second half. The `| N |` column is a hand-typed count of the cell
    beside it — exactly the shape this item exists to stop: a number true when it
    was written and untrue now, with nothing able to see the difference.
    """
    tiers = _table(TIERS_HEADER)
    assert len(tiers) == 5, f"expected the five documented tiers, parsed {list(tiers)}"
    bad = {t: (len(ids), n) for t, (ids, n) in tiers.items() if n is None or int(n) != len(ids)}
    assert not bad, (
        "tier rows whose count column disagrees with their own id list "
        f"(counted, written): {bad}")


def test_every_configured_group_still_resolves_to_a_heading_in_the_doc():
    """The seam this round actually crosses. `workers/sources/arch_review.py`
    turns each `autonomy-jobs:<Group>` entry in config.yaml into a review unit by
    locating its heading through `find_section`, and a name that matches no
    heading is *skipped and recorded as `section_missing`* — the group silently
    stops being reviewed. This round rewrote the Measure heading line, which is
    the one line in the diff that parser reads, so the pair is tested through the
    production lookup rather than a fixture tree (`test_arch_review_source.py`
    covers the matcher; nothing covered the real doc against it).
    """
    from workers.sources import arch_review as A
    from workers.sources import get_sources_config

    groups = A.parse_groups((get_sources_config().get(A.NAME, {}) or {}).get("groups"))
    mine = [name for slug, name in groups if slug == "autonomy-jobs"]
    assert mine, "config.yaml declares no autonomy-jobs groups to review"
    doc = _text()
    missing = [name for name in mine if A.find_section(doc, name) is None]
    assert not missing, (
        f"config.yaml offers {missing} as review units and the doc has no heading "
        f"they match — arch-review would skip them and file nothing")

    start, end = A.find_section(doc, "Measure")
    row = next(i for i, ln in enumerate(doc.splitlines(), start=1)
               if ln.startswith("| #86 |"))
    assert start <= row <= end, (
        f"the #86 row is at line {row}, outside the section arch-review would "
        f"hand a reviewer for Measure ({start}-{end}) — the entry exists but no "
        f"pass over that group can ever see it")


def test_every_job_has_one_function_and_exactly_one_write_tier():
    """The two tables partition one fleet from two axes. Neither can be checked
    against the live fleet from inside a unit test, so each is checked against the
    other: the same ids, none in two groups, none in two tiers. This is the rail
    #86's absence trips — a job in one table and not the other — and it cannot be
    asserted from a constant without rebuilding the very defect.
    """
    funcs, tiers = _table(FUNCTIONS_HEADER), _table(TIERS_HEADER)
    grouped = [i for ids, _ in funcs.values() for i in ids]
    tiered = [i for ids, _ in tiers.values() for i in ids]
    assert len(grouped) == len(set(grouped)), "a job is listed in two function groups"
    assert len(tiered) == len(set(tiered)), "a job is listed in two write-authority tiers"
    assert set(grouped) == set(tiered), (
        f"listed only as a function: {sorted(set(grouped) - set(tiered))}; "
        f"listed only as a tier: {sorted(set(tiered) - set(grouped))}")


# ── #1521 — the `_pipeline/` shorthand: one definition, above every use ───────

#: The shorthand itself: a backticked path that *is* just `_pipeline/`. Prose
#: carrying it alongside the rooted path is defining the mapping — no other
#: sentence shape puts the two spellings side by side. §Distil's PATH_ESCAPE
#: paragraph did exactly that at filing, which is what made it the duplicate.
_BARE_SHORTHAND = re.compile(r"`_pipeline/`")

#: Any use of the bare spelling, including artifact paths spelled from it
#: (`_pipeline/reflection/...`); this is what `grep '_pipeline/' | grep -v
#: lloyd-data` counts, line-wise, in the item's acceptance command.
_BARE_USE = re.compile(r"`_pipeline/")

#: The data root spelled in full (trailing artifact path allowed).
_ROOTED = re.compile(r"`~/lloyd-data/_pipeline/")


def _defining_blocks() -> list[str]:
    """Paragraphs that carry the shorthand and the rooted path together.

    Paragraph-wise, not line-wise: the doc wraps near 88 columns, so a
    definition can straddle lines, and a line-wise match would call a wrapped
    definition absent — the same false-absence that made the item's bare glob
    read as "no fresh handoff" when it was really "wrong root".
    """
    return [b for b in _text().split("\n\n")
            if _BARE_SHORTHAND.search(b) and _ROOTED.search(b)]


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def test_the_bare_pipeline_shorthand_is_defined_above_its_first_bare_use():
    """Item #1521 clause 1. Exactly one paragraph defines the shorthand, it sits
    above the first bare use, and it states the root and names the move commit.

    At base this fails on position: the only paragraph carrying both spellings
    was the §Distil PATH_ESCAPE one, ~60 lines *below* the first bare table row
    — so a reader who copies a bare path into a command run from `~/lloyd`
    meets it before meeting the definition.
    """
    text = _text()
    defs = _defining_blocks()
    assert len(defs) == 1, (
        f"{len(defs)} paragraphs carry both the bare shorthand and the rooted path; "
        f"the contract is exactly one definition — a second one is the half-"
        f"corrected doc reforming (at base this file had one, buried in §Distil)")
    block = defs[0]
    def_start = text.index(block)
    def_end = def_start + len(block)

    uses = [m.start() for m in _BARE_USE.finditer(text)
            if not def_start <= m.start() < def_end]
    assert uses, (
        "no paragraph outside the definition spells `_pipeline/` bare at all — "
        "this test's premise (a doc that *uses* the shorthand) has moved; revisit "
        "the #1521 contract rather than let 'above every use' pass vacuously")
    first_use = min(uses)
    assert first_use >= def_end, (
        f"the shorthand is defined at line {_line_of(text, def_start)} but first "
        f"used bare at line {_line_of(text, first_use)} — the definition belongs "
        f"above every use, which is the whole of #1521's placement requirement")

    flat = _flat(block)
    assert re.search(r"bare `_pipeline/`[^.]{0,40}means `~/lloyd-data/_pipeline/`", flat), (
        f"the defining paragraph no longer states the mapping clause 1 names "
        f"(bare prefix means the data root): {flat!r}")
    assert "6426668b" in block, (
        "the definition no longer names `6426668b`, the commit that moved runtime "
        "data out of the code tree — the reason a reader must not resolve the "
        "shorthand against `~/lloyd`")


def test_the_defining_paragraph_wikilinks_data_home_and_defines_it_alone():
    """Item #1521 clause 2. The definition points at [[data-home]] — the doc
    that already owns the rule (`architecture/data-home.md:33-34`: "every name
    is the one it had in the tree, so a path `~/lloyd/X` became
    `~/lloyd-data/X`") — and nothing else in the file competes as a definition.

    The sole-definition rail is re-derived through `_defining_blocks`, not
    asserted from a location: qualifying a table cell here, or restating the
    root in another section, would put a second paragraph in that set. This
    failed at base because `autonomy-jobs.md` contained no `data-home` string
    at all.
    """
    defs = _defining_blocks()
    assert len(defs) == 1, (
        f"{len(defs)} competing definitions; clause 2 allows one")
    assert "[[data-home]]" in defs[0], (
        "the defining paragraph does not wikilink [[data-home]], the owner of the "
        "root-move rule — a definition that stands alone here is the duplicate "
        "this item exists to prevent")
    assert (ARCH_DIR / "data-home.md").is_file(), (
        "[[data-home]] points at a file absent from architecture/ — the wikilink "
        "would dangle in the one doc that delegates the rule to it")


def test_the_distil_paragraph_no_longer_restates_the_root():
    """Item #1521 clause 3. §Distil's PATH_ESCAPE paragraph was folded into the
    definition instead of duplicating it, so its phrase occurs zero times — with
    a positive control, because a zero-count that cannot tell "folded away" from
    "paragraph deleted" is not a check.

    The paragraph's own rule (vault tools refuse `~/lloyd/` with `PATH_ESCAPE`,
    so chain artifacts go through `Write` at an absolute path) still stands;
    only the root-restatement left it.
    """
    text = _text()
    assert "the root this document spells bare" not in text, (
        "§Distil still restates the root the intro now defines — the duplicate "
        "clause 3 exists to keep out, and two statements of one mapping are how "
        "this doc drifted half-corrected in the first place")
    assert "PATH_ESCAPE" in text, (
        "positive control tripped: the PATH_ESCAPE paragraph is gone entirely. "
        "Folding the root clause must leave the paragraph's own rule standing")


def test_the_bare_spelling_still_appears_on_exactly_twelve_lines():
    """The acceptance count, as the item states it: `grep -n '_pipeline/'
    architecture/autonomy-jobs.md | grep -v lloyd-data` was 12 lines at triage
    (2026-09-26) and must stay 12. The fix is one definition, not per-cell
    qualification — the other eleven `architecture/*.md` files carry 27 more
    bare mentions, so qualifying this doc's cells would make it the only one
    that reads differently from the corpus it is part of.

    This node passes at base by design: it is the rail against the *other*
    failure mode, a "fix" that rewrites the twelve cells instead of defining
    the shorthand.
    """
    bare = [ln for ln in _text().splitlines()
            if "_pipeline/" in ln and "lloyd-data" not in ln]
    assert len(bare) == 12, (
        f"{len(bare)} lines spell `_pipeline/` without the root, not the 12 the "
        f"triage counted: {bare}")


# ── #1520 clause 1 — the pilot citation resolves to the code it cites ────────


def test_the_pilot_frozenset_citation_resolves_to_the_line_it_names():
    """#1520 clause 1. §The functions cited `autonomy.py:710`, which is
    `RUN_SUMMARY_CAP = 300` inside the comment about tail-slicing a run summary —
    nothing to do with evidence — while the frozenset it meant is at
    `autonomy.py:2104`, where §Distil has pointed since #1102. A reader who walked
    to 710 came back concluding the pilot was documented somewhere it is not.

    The expectation is `autonomy.py` itself, not a remembered number: the line the
    prose names must BE the definition, with the members the prose quotes, and the
    test file it calls a pin must assert that set. So the citation cannot rot the
    way its predecessor did — the code moving under the prose fails here first.
    """
    funcs = _functions_section()
    assert "`EVIDENCE_PILOT_TASK_IDS" in funcs, (
        "§The functions no longer names the pilot frozenset literally")
    window = funcs[funcs.index("`EVIDENCE_PILOT_TASK_IDS"):][:320]
    cited = re.search(r"`autonomy\.py:(\d+)`", window)
    assert cited, f"the pilot sentence cites no autonomy.py line: {window[:160]!r}"
    line = int(cited.group(1))
    src = (ROOT / "autonomy.py").read_text().splitlines()
    at = re.fullmatch(r"EVIDENCE_PILOT_TASK_IDS = frozenset\(\{([0-9, ]+)\}\)",
                      src[line - 1].strip())
    assert at, (
        f"the doc cites `autonomy.py:{line}` for the pilot frozenset, and that line "
        f"reads {src[line - 1].strip()!r} — follow the citation and you learn about "
        f"run-summary slicing instead of the pilot")
    quoted = re.search(r"frozenset\(\{([0-9, ]+)\}\)", window).group(1)
    assert (sorted(quoted.replace(" ", "").split(","))
            == sorted(at.group(1).replace(" ", "").split(","))), (
            f"the doc quotes {quoted} at the line it cites, which defines {at.group(1)}")
    assert "`tests/test_worker_evidence.py`" in window, (
        f"the pilot sentence no longer names the test that pins the set: {window[:160]!r}")
    pin = (ROOT / "tests" / "test_worker_evidence.py").read_text()
    assert re.search(r"EVIDENCE_PILOT_TASK_IDS == frozenset\(\{38, 42, 39, 40\}\)", pin), (
        "the doc says tests/test_worker_evidence.py pins the pilot set, and that file "
        "no longer asserts it — the citation has become a promise nothing keeps")
    assert "autonomy.py:710" not in _live_prose(), (
        "`autonomy.py:710` is still cited in current prose; only the dated 2026-09-12 "
        "§Review log entry may carry it, as the error that was corrected there")


# ── #1520 clause 2 — the evidence state, dated, and its mechanism still on disk ─


def test_the_evidence_state_in_the_functions_is_dated_and_self_consistent():
    """#1520 clause 2. §The functions said not one claim had been verified (#902).
    #945 then copied each run's `claims` key into the dict the pool records
    (`workers/sources/scheduled_task.py`), so verdicts have been written since: on
    2026-09-26 `GET /api/autonomy/health?days=7` reported 22 claims checked, 22
    verified, 0 refuted and 0 insufficient over 4 runs that carried a bundle — #38
    twice for 11 claims, #39 once for 6, #40 once for 5. `runs_with_bundle` counts
    runs, not tasks, so the breakdown has to reconcile with both totals, which is
    the arithmetic the first draft of this round's own sentence got wrong (it named
    three tasks under a total of four runs).

    Those counts are not fixture-able from a unit test — the route reads
    `workers.db`, and its seven-day window moves every night — so pinning their
    magnitude would only guarantee a red suite on a day when nothing was wrong.
    What is pinned is what makes a typed count trustworthy without being able to
    re-run it: the sentence carries the date it was true, its arithmetic closes
    (checked = verified + refuted + insufficient; the per-task runs sum to the
    declared runs and the per-task claims to the checked total; no task contributes
    more runs than the total has), the sentence's claim of never having rejected a
    claim is allowed to stand exactly while its own numbers say so — never asserted
    as a permanent property of the pilot, which would make its first correct
    rejection a failing test — and the code path that produces verdicts at all is
    still in the tree.
    """
    funcs = _functions_section()
    m = re.search(
        r"On (\d{4}-\d{2}-\d{2})(.+?from (\d+) runs that carried a bundle: )"
        r"([^.;]*\.)", funcs)
    assert m, (
        "§The functions no longer states the pilot's evidence state as a dated "
        "sentence running '... N runs that carried a bundle: #NN n runs n claims'. "
        "It must not go back to an undated claim about what the pilot has or has "
        "not verified")
    middle, runs, breakdown = m.group(2), int(m.group(3)), m.group(4)
    counts = re.search(
        r"(\d+) claims checked, (\d+) verified, (\d+) refuted and (\d+) insufficient",
        middle)
    assert counts, f"the dated evidence sentence names no four counts: {middle!r}"
    checked, verified, refuted, insufficient = (int(x) for x in counts.groups())
    assert checked == verified + refuted + insufficient, (
        f"{checked} checked but {verified} verified + {refuted} refuted + "
        f"{insufficient} insufficient = {verified + refuted + insufficient}")
    # Not `assert refuted == 0`: that would freeze today's production state as an
    # invariant and turn the pilot's first correct rejection into a red suite. What
    # is asserted is that the sentence's claim and its own numbers cannot part
    # company — so the rejection, when it lands, costs an edit to one sentence.
    never_rejected = bool(re.search(
        r"none refuted|never\s+rejected a claim", funcs[m.start():m.start() + 600]))
    assert not never_rejected or (refuted == 0 and insufficient == 0), (
        f"the sentence claims the pilot has never rejected a claim while its own "
        f"numbers read {refuted} refuted and {insufficient} insufficient — one "
        f"direction only: dropping the claim is always allowed, keeping it costs the "
        f"zeros, so the first real rejection asks for an edit to one sentence rather "
        f"than a failing suite")
    per_task = re.findall(r"#(\d+) (\d+) runs? (\d+) claims?", breakdown)
    assert per_task, f"the breakdown names no task with its runs and claims: {breakdown!r}"
    assert len(per_task) <= runs, (
        f"'{runs} runs that carried a bundle' broken out over {len(per_task)} tasks: "
        f"{breakdown!r} — a task cannot contribute a run that did not happen")
    assert sum(int(r) for _, r, _ in per_task) == runs, (
        f"the per-task runs in {breakdown!r} sum to "
        f"{sum(int(r) for _, r, _ in per_task)}, not the {runs} runs declared — the "
        f"route's `runs_with_bundle` counts runs, and #38 has shipped two bundled "
        f"runs to 39's and 40's one, so a breakdown that forgets the second quietly "
        f"describes a different fleet of runs than the total does")
    assert sum(int(c) for _, _, c in per_task) == checked, (
        f"the per-task claims {breakdown!r} do not sum to the {checked} checked")
    src = (ROOT / "workers" / "sources" / "scheduled_task.py").read_text()
    assert re.search(r'if "claims" in result:\s*\n\s*out\["claims"\] = result\["claims"\]',
                     src), (
        "the doc attributes the verdicts to the #945 passthrough in "
        "workers/sources/scheduled_task.py and that copy is gone: the counts would "
        "stop being written and this doc would keep quoting them")


def test_no_live_sentence_revives_the_refuted_zero_verdict_claim():
    """#1520 clause 2, negative half. The falsified claim — the pilot has verified
    nothing — is refuted by the route and #902 is done, so those words may survive
    only where they were recorded as a correction, never in prose a reader takes as
    current.
    """
    live = _live_prose()
    for pattern, why in (
        (r"not one claim has yet been verified", "the #902 state that #945 refuted"),
        (r"verified \*\*nothing\*\*", "the same claim in the shape the 2026-09-12 review left it"),
        (r"pilot has verified nothing", "the same claim in plain prose"),
    ):
        assert not re.search(pattern, live), (
            f"a current sentence asserts {why} again; only the dated §Review log may")


def test_the_dated_review_log_entry_keeps_what_was_true_that_day():
    """#1520 clause 2's out-of-scope half, as a rail rather than a comment. The
    2026-09-12 entry records the state at `28abecf073ee`, the wrong line number and
    the zero verdicts included, because that is what was true then. Sweeping it
    "consistent with the fix" would manufacture a September in which the pilot had
    verified claims, which never happened; the entry is dated precisely so a later
    reader can tell when each claim held.
    """
    log = _review_log()
    assert re.match(r"\s*- \*\*2026-09-12", log), (
        f"the §Review log's first entry is no longer the dated 2026-09-12 one: {log[:60]!r}")
    assert "autonomy.py:710" in log, (
        "the 2026-09-12 entry's record of the wrong citation has been 'corrected' — "
        "that rewrite is a false history, and clause 1 depends on it staying put")
    assert "verified **nothing**" in log, (
        "the 2026-09-12 entry's record of the zero verdicts has been 'corrected'; it "
        "was true at 28abecf073ee and must stay readable as what it was")


# ── #1520 clause 3 — one fleet size, and the ten edges that must not move ─────


def test_the_no_dependency_count_is_the_fleet_minus_the_edges_the_doc_draws():
    """#1520 clause 3. §The functions said "22 of the 32 tasks declare no
    dependency at all" while §Distil said "Ten of the fleet's 36 jobs" — two fleet
    sizes in one document on the same day, and the live route says 36 parsed tasks
    with 10 declaring `depends_on`.

    The count is derived, not remembered: it has to equal §Distil's denominator
    minus the `depends_on` arrows this very section draws three lines above it, and
    travel with the date it was taken. Three statements that must be edited together
    is the point — a fleet that grows makes all three fail at once instead of
    leaving one section quietly ahead.
    """
    funcs = _functions_section()
    m = re.search(r"\b(\w+) of the (\w+) tasks\b[^.]*\.", funcs)
    assert m, "§The functions no longer counts the jobs that declare no dependency"
    said, without, fleet = m.group(0), _num(m.group(1)), _num(m.group(2))
    distil = re.search(r"[Tt]en of the fleet's (\d+) jobs", _text())
    assert distil, (
        "§Distil no longer counts its share of the fleet, so §The functions has "
        "nothing left to be made to agree with")
    assert fleet == int(distil.group(1)), (
        f"'{said}' counts a fleet of {fleet} and §Distil counts {distil.group(1)} — "
        f"two sections of one document disagreeing on the fleet size is the defect "
        f"#1520 is about, whichever of them the live route has outgrown")
    edges = _diagram_edges()
    assert len(edges) == fleet - without, (
        f"'{said}': {fleet} − {without} = {fleet - without}, but this section's own "
        f"wiring diagram draws {len(edges)} `depends_on` arrows ({sorted(edges)}). "
        f"Neither number can be bumped alone: the denominator has to move with the "
        f"arrows, the arrows have to move with the diagram, and §Distil has to agree "
        f"on the denominator. `GET /api/autonomy/tasks` is the live view — measured "
        f"on 2026-09-26 at 36 parsed tasks and 10 declaring `depends_on` — so "
        f"re-measure it before editing any of the three.")
    assert re.search(r"\(\d{4}-\d{2}-\d{2}\)", said), f"'{said}' carries no date"


def test_the_wiring_sentence_and_its_diagram_still_declare_the_ten_verified_edges():
    """#1520 clause 3, the half that must NOT change. The ten edges were re-checked
    parent-by-parent on 2026-09-26 — 74→48, 48→24, 39→42, 58→57, 42→38, 40→39,
    83→58, 47→40, 51→56, 57→56 — so both the sentence and its arrows stay exactly as
    they are. A count that is right deserves the same protection as one that is
    wrong: this fails if the diagram loses an arrow, gains one, or the sentence's
    number stops matching the picture it introduces.
    """
    funcs = _functions_section()
    assert re.search(r"Ten `depends_on` edges exist and \*\*three of them now cross a "
                     r"function boundary\*\*", funcs), (
        "the sentence naming the ten edges and their three cross-function hops has "
        "been edited; #1520 says leave it, because ten is still correct")
    assert _diagram_edges() == TEN_EDGES, (
        f"the wiring diagram draws {sorted(_diagram_edges())}, not the ten verified "
        f"edges {sorted(TEN_EDGES)}")


# ── #1520 clause 5 — the four newest jobs are in both tables ─────────────────


def test_the_four_newest_jobs_are_listed_once_in_each_membership_table():
    """#1520 clause 5. #87, #88, #89 (`research-agent`, one arXiv paper each) and
    #90 (`corpus-shape-trend`) were created 2026-09-24 and 2026-09-25 and have run
    since — two runs each for the digests, one for #90, all green — yet neither
    table listed them, and the five tier rows still added up (20/1/3/8/1 = 33)
    against a 36-task fleet.

    `test_every_job_has_one_function_and_exactly_one_write_tier` could not see it:
    it compares the tables with each other, so a job missing from both is symmetric
    and passes. This names the four ids and requires exactly one row apiece in each
    table, because a count in a header cell is not a membership check.
    """
    funcs, tiers = _table(FUNCTIONS_HEADER), _table(TIERS_HEADER)
    grouped = [i for ids, _ in funcs.values() for i in ids]
    tiered = [i for ids, _ in tiers.values() for i in ids]
    for job in NEW_JOBS:
        assert grouped.count(job) == 1, (
            f"{job} appears {grouped.count(job)} times in the function table; the one "
            f"document that says what each scheduled job is for is silent, or double, "
            f"about a live task")
        assert tiered.count(job) == 1, (
            f"{job} appears {tiered.count(job)} times in the write-authority table, "
            f"which is the doc's answer to what that job may do unattended")


def test_the_four_newest_jobs_sit_in_the_group_and_tier_their_output_belongs_to():
    """#1520 clause 5's judgement, pinned the way #86's placement is. #87 #88 #89
    read one outside document — an arXiv paper — and write one knowledge note into
    `knowledge/`, which is what Ingest is for, and that note is durable state, so
    they sit in the unattended-writer tier beside #30 and #53. #90 measures the
    shape of four nightly-written corpora and appends one dated metrics row per run,
    which is what #82 and #86 do when they record a trend: Reports only, under the
    Bound entropy group whose #78/#80 pair is this doc's report-only half of bounding
    growth. The choice of group is arguable; that each job has exactly one group and
    one tier is not.
    """
    funcs, tiers = _table(FUNCTIONS_HEADER), _table(TIERS_HEADER)

    def tier_of(job: str) -> list[str]:
        return [t for t, (ids, _) in tiers.items() if job in ids]

    ingest = next(v for k, v in funcs.items() if k.startswith("[Ingest]"))
    assert all(job in ingest[0] for job in DIGESTS), (
        f"the digests {list(DIGESTS)} are not all in the Ingest row, which lists "
        f"{ingest[0]} — an outside document in, a vault note out, is that group")
    bound = next(v for k, v in funcs.items() if k.startswith("[Bound entropy]"))
    assert "#90" in bound[0], f"#90 is not in the Bound entropy row, which lists {bound[0]}"
    for job in DIGESTS:
        assert tier_of(job) == ["Writes durable state unattended"], (
            f"{job} is in {tier_of(job) or 'no tier'}; it writes a knowledge note into "
            f"the vault with nobody watching, which is what that tier means")
    assert tier_of("#90") == ["Reports only"], (
        f"#90 is in {tier_of('#90') or 'no tier'}; outside its own metrics row it "
        f"changes nothing, the same position #82 and #86 hold")


def test_every_groups_heading_row_link_and_id_list_agree():
    """One rule over all seven groups, so a job can be added without the doc
    quietly breaking around it. The Measure case is pinned for one group by
    `test_the_functions_table_row_and_every_anchor_follow_the_group_heading`;
    widening Ingest for #87–#89 and Bound entropy for #90 is the same edit in two
    more places, and the failure looks the same from the outside — the row reads
    fine, the heading still names the old members, and every link into the group
    lands nowhere. The expected anchor is derived from the heading each time, never
    typed, and `workers/sources/arch_review.py` keys a group off the heading's text
    before the first `:` — which is why decorating a heading with more ids is safe
    and renaming it is not.
    """
    text, funcs = _text(), _table(FUNCTIONS_HEADER)
    bad = []
    for label, (ids, _) in funcs.items():
        name = label.split("]")[0].lstrip("[").strip()
        m = re.search(rf"^## {re.escape(name)}: (.+)$", text, re.M)
        if not m:
            bad.append(f"{name}: no `## {name}: ...` group heading")
            continue
        if ids != _ids(m.group(1)):
            bad.append(f"{name}: row lists {ids}, heading lists {_ids(m.group(1))}")
        href = re.search(r"\]\(#([^)]*)\)", label)
        want = _slug(f"{name}: {m.group(1)}")
        if not href:
            bad.append(f"{name}: row links to no anchor")
        elif href.group(1) != want:
            bad.append(f"{name}: row links #{href.group(1)}, heading slugs to {want}")
    assert not bad, f"group membership out of step with its own heading: {bad}"
