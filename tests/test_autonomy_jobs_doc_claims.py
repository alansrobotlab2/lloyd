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
pilot frozenset's line number against `app/autonomy.py` itself, the evidence verdicts
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
#: paragraph's closing "they are two of six", and #78/#80's "same position as two
#: of the six jobs in Measure" — so the set is stated once, here, and the
#: paragraph is checked to name exactly it. A job gaining or losing a consumer has
#: to move all three sentences together, which is the point.
#:
#: #85 left on 2026-09-27 with the retirement Alan ruled on backlog **#1577**: its
#: slot was switched off on 2026-09-20, so it had no report to leave unconsumed.
#: That is why the membership lives here and not in the sentences — dropping the id
#: from this set is what forces the three counts in the doc to be edited in the
#: same commit, and why a doc that quietly kept "three of the seven" is red.
NO_CONSUMER = {"#36", "#70"}

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


def test_the_measure_id_table_lists_the_heading_membership_and_nothing_else():
    """The group's own ID table is a membership statement no other check reads.

    `test_the_functions_table_row_and_every_anchor_follow_the_group_heading` follows
    the heading into the function table and the anchors, and
    `test_every_job_has_one_function_and_exactly_one_write_tier` compares the two
    summary tables with each other — so a row naming a retired job, or a job that
    never existed, could sit in the §Measure table while every one of those stayed
    green. Deleting #85's row for the retirement (backlog **#1577**) is what makes
    the gap concrete: that row is the line a reader trusts once the heading has
    scrolled past, and it is the line nothing was reading.
    """
    doc = _text()
    start = doc.index(_measure_heading())
    section = doc[start:doc.find("\n## ", start + 1)]
    rows = [m.group(1) for m in re.finditer(r"^\| (#\d+) \|", section, re.M)]
    ids = _measure_ids()
    assert rows, "the §Measure section types no job rows, so its table is gone"
    assert rows == ids, (
        f"the §Measure table's rows are {rows} while its heading lists {ids} — the "
        "two membership statements of one group, disagreeing, with nothing between")


def test_the_retired_secondary_eval_is_recorded_only_in_the_dated_log():
    """#1577. #85's slot was switched off on 2026-09-20 and Alan ruled the job
    retired rather than re-scoped, so the id may appear in exactly one place: the
    §Review log entry that says when, and why djev is not its successor.

    The sentence it carried here — that #85 was the fleet's only `draft` task — was
    false the day it was written, because #68 has held that status throughout. A
    false claim is not repaired by moving it somewhere truer-looking, so both halves
    are pinned: the claim is gone from prose a reader takes as current, and the log
    keeps the retirement with its date and its item rather than the job simply
    vanishing.
    """
    live = _live_prose()
    assert "#85" not in live, (
        "a current sentence names #85 again; the job is retired and belongs in the "
        "dated §Review log only")
    assert "fleet's only `draft` task" not in live, (
        "the only-draft claim is back in current prose — #68 is draft too, which is "
        "what made it false")
    log = _review_log()
    assert "#85" in log and "#1577" in log, (
        "the §Review log no longer records the retirement, so the deletion carries no "
        "dated explanation of itself")


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
    """#1520 clause 1. §The functions cited `app/autonomy.py:710`, which is
    `RUN_SUMMARY_CAP = 300` inside the comment about tail-slicing a run summary —
    nothing to do with evidence — while the frozenset it meant is at
    `app/autonomy.py:2104`, where §Distil has pointed since #1102. A reader who walked
    to 710 came back concluding the pilot was documented somewhere it is not.

    The expectation is `app/autonomy.py` itself, not a remembered number: the line the
    prose names must BE the definition, with the members the prose quotes, and the
    test file it calls a pin must assert that set. So the citation cannot rot the
    way its predecessor did — the code moving under the prose fails here first.
    """
    funcs = _functions_section()
    assert "`EVIDENCE_PILOT_TASK_IDS" in funcs, (
        "§The functions no longer names the pilot frozenset literally")
    window = funcs[funcs.index("`EVIDENCE_PILOT_TASK_IDS"):][:320]
    cited = re.search(r"`app/autonomy\.py:(\d+)`", window)
    assert cited, f"the pilot sentence cites no app/autonomy.py line: {window[:160]!r}"
    line = int(cited.group(1))
    src = (ROOT / "app" / "autonomy.py").read_text().splitlines()
    at = re.fullmatch(r"EVIDENCE_PILOT_TASK_IDS = frozenset\(\{([0-9, ]+)\}\)",
                      src[line - 1].strip())
    assert at, (
        f"the doc cites `app/autonomy.py:{line}` for the pilot frozenset, and that line "
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
    assert "app/autonomy.py:710" not in _live_prose(), (
        "`app/autonomy.py:710` is still cited in current prose; only the dated 2026-09-12 "
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
    assert "app/autonomy.py:710" in log, (
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


# ── #1524 — Canonicalize: what it may change, and what #48 actually writes ────

CANON_HEADER = "Canonicalize"
BUILD_HEADER = "Build the graph"
PROPOSES_TIER = "Proposes; an operator applies"
DURABLE_TIER = "Writes durable state unattended"

#: The tiers a scheduled #48 run is told to `--apply`, exactly as step 3 of
#: `skills/entity-resolution-sweep/SKILL.md` orders them (Alan's ruling on #990,
#: 2026-09-16, reaffirmed 2026-09-24). Every other merge claims a meaning and so
#: stays a human act — which is the half of the boundary this file cannot derive
#: and has to read across.
MECHANICAL_TIERS = {"CASE", "PUNCT"}


def _group_section(name: str) -> str:
    """One job group's own section, found through the matcher arch-review uses.

    Per section because #1524's defect is a claim asserted twice — once in
    §Build the graph, once in §Canonicalize, two separate review units — and true
    in neither. Checking "somewhere in the doc" would let one section's correct
    wording excuse the other's false one, which is the shape of the bug.
    """
    from workers.sources import arch_review as A
    span = A.find_section(_text(), name)
    assert span, f"architecture/autonomy-jobs.md lost its `## {name}` group heading"
    return _flat("\n".join(_text().splitlines()[span[0] - 1:span[1]]))


def test_canonicalize_says_48_applies_the_mechanical_tiers_unattended():
    """Clause 1. §Canonicalize asserted "All three run in plan mode and write
    nothing to the fact tree" and, of #48, that "it never passes `--apply`, so its
    scheduled form reports a plan and stops". Neither has been true since #990:
    step 3 of the job's own skill prompt ends the scheduled run with `--apply
    --tiers CASE,PUNCT`, and the run records through 2026-09-25 each end in an
    apply — 30 merges and 705 fact files moved, then 111, then 6, then 3 merges
    with 4 edges rewritten and 3 aliases written.

    #67 and #84 are left alone because their half is true: #67's own `--apply` was
    retired 2026-09-04 and its four 2026-09-23 proposals were all guard-downgraded
    to alias-only with 0 merges; #84 passed no `--apply` on 2026-09-25. The test
    therefore pins that they still read as proposing, so the fix cannot be
    "Canonicalize writes" either.
    """
    canon = _group_section(CANON_HEADER)
    for dead in ("never passes `--apply`",
                 "run in plan mode and write nothing",
                 "Nothing here moves fact files unattended"):
        assert dead not in canon, (
            f"§Canonicalize still carries {dead!r}. It is not a nuance but the "
            f"sentence an operator reads before deciding whether a nightly write into "
            f"kg.sqlite needs watching")
    m = re.search(
        r"#48[^.]*?\bappl(?:y|ies|ied)\b[^.]*?CASE[^.]*?"
        r"\b(unattended|nightly|every night)\b", canon, re.I)
    assert m, (
        "§Canonicalize never says, in one sentence, that #48 *applies* CASE/PUNCT "
        "*unattended*. What runs on schedule is `--apply --tiers CASE,PUNCT` with "
        "nobody watching, and both halves are the point: 'merges' without "
        "'unattended' describes an operator act")
    assert re.search(r"unattended|every night|nightly", canon, re.I) and \
        re.search(r"never\s+\w+[^.]{0,40}SUFFIX|SUFFIX[^.]{0,40}\b(?:human|person)\b"
                  r"|(?:human|person)[^.]{0,40}SUFFIX", canon, re.I), (
        "the section never marks the suffix tier a human act, so it states the apply "
        "without its bound and reads as 'Canonicalize merges whatever the judge likes'")
    # The true half, pinned so the fix cannot overshoot. The group's role row for
    # #67 also says "Proposes", so a loose `#67 … proposes` search would pass on
    # the table while the prose was rewritten; these two bind the dated facts in
    # the paragraph instead.
    assert re.search(r"#67[^.]*\bretired\b", canon, re.I), (
        "#67's plan-only status is a dated fact — its own `--apply` was retired "
        "2026-09-04 — and the paragraph that just gained #48's apply is where a "
        "reader learns the other two did not follow it")
    assert re.search(r"#84[^.]*operator act", canon, re.I), (
        "#84 writes nothing and applying is an operator act; if that has been "
        "rewritten to match #48, the fix has overshot a falsehood into a true "
        "sentence")


def test_canonicalize_names_the_authority_for_the_apply():
    """Clause 2. An unattended nightly write against the edge store has to be
    attributable on the page, not inferable: #1524's own open question is whether
    Alan retires the #990 ruling and sends #48 back to plan-only, and a reader
    cannot ask that question of a doc that never names the ruling.
    """
    canon = _group_section(CANON_HEADER)
    assert "#990" in canon, (
        "the apply is described with no authority behind it. A reader who finds "
        "an unattended `--apply` against kg.sqlite needs to know it is a decision "
        "someone made, and this item's needs-human half is precisely whether that "
        "decision still stands")
    assert re.search(r"Alan[^.]{0,60}#990", canon), (
        "`#990` is in the section but not attributed to the person who ruled: the "
        "item says the choice is Alan's, not a doc edit's, and provenance is the "
        "part that lets a later pass reopen it")
    assert re.search(r"#990[^.]{0,40}2026-09-16", canon), (
        "the ruling is undated. It is a 2026-09-16 ruling reaffirmed 2026-09-24, and "
        "an undated permission in a document whose counts all carry dates is the "
        "sentence that quietly stops being true")


def test_the_functions_paragraph_makes_the_guard_rails_the_boundary():
    """Clause 3. §The functions explained Canonicalize's strict gates by saying
    "all three of its jobs run in plan mode", and called the group "forbidden from
    writing". Both framing moves are now false, and they were the load-bearing
    half of the doc's argument for the Build-the-graph / Canonicalize split.
    """
    funcs = _functions_section()
    assert "run in plan mode" not in funcs and "forbidden from writing" not in funcs, (
        "§The functions still explains the group's gating with plan mode. The group "
        "applies nightly; the gates are what bounds it, and saying otherwise is how "
        "the split keeps getting re-justified by a reason that no longer holds")
    assert "Canonicalize carries" in funcs, "the Canonicalize bullet has been rewritten away"
    bullet = funcs.split("Canonicalize carries", 1)[1].split("Distil writes", 1)[0]
    assert "guard rails" in bullet, (
        "the bullet no longer names the rails, so the paragraph has lost the thing "
        "clause 3 asks it to credit as the boundary")
    assert re.search(r"#48.{0,120}?\bappl", bullet, re.I) and "unattended" in bullet, (
        "the paragraph does not say the scheduled run applies unattended, which is "
        "the fact that made 'plan mode' false in the first place")
    assert "#990" in bullet, "the paragraph attributes the apply to nothing"
    assert re.search(r"2026-08-22 wipe|2026-09-03 151-merge", bullet) and \
        re.search(r"(why the rails|rails are strict|boundary)", bullet, re.I), (
        "the two incidents are the reason the rails exist; the paragraph has to keep "
        "citing them as the reason the rails are strict, or it has swapped one false "
        "explanation for silence")


def test_build_the_graph_groups_by_what_each_job_writes():
    """Clause 4. §Build the graph opened "The only three jobs that write to the
    edge store", which is what made the invariant look like a property of the
    store rather than of these three jobs — and #48 writes the same store nightly.
    The grouping is kept, but on the axis that is actually true: these three
    *create* entities and edges, #48 re-points existing ones.
    """
    build = _group_section(BUILD_HEADER)
    assert "The only three jobs that write to the edge store" not in build, (
        "the sentence is back, and it is the one that let §Canonicalize's false "
        "claim read as corroborated by a second section")
    assert re.search(r"only jobs that \*{0,2}creat", build, re.I), (
        "the section no longer states what its three jobs uniquely do, so the "
        "grouping has lost its criterion instead of gaining a true one")
    assert re.search(r"#48[^.]*unattended", build, re.I), (
        "#48 is not named as an unattended writer of the store here, so a reader "
        "who comes to this section for the store's writers still misses it")
    assert re.search(r"alias", build, re.I) and re.search(r"no entity|creates no", build, re.I), (
        "the section must name both halves: that #48 writes aliases and edge "
        "rewrites, and that it creates no entity or relation — otherwise the "
        "distinction it rests on is only implied")


def test_48_is_tiered_apart_from_the_two_jobs_that_only_propose():
    """Clause 5. The tier table is the doc's answer to "what may write while
    nobody is watching", and it tiered #48 with #67 and #84 under "Proposes; an
    operator applies" — internally consistent, since every row's count equalled
    its own id list, and false all the same. That is the blind spot this file
    documents elsewhere: its checks are deliberately internal, so nothing here
    could see a row that no longer describes the fleet.
    """
    tiers = _table(TIERS_HEADER)
    assert len(tiers) == 5, (
        f"the table now has {len(tiers)} tiers; the five are the doc's own claim "
        f"(\"Five tiers, covering every job the table above names\"), and adding a "
        f"row is a decision about the taxonomy, not where #48 belongs")

    def tier_of(job: str) -> list[str]:
        return [t for t, (ids, _) in tiers.items() if job in ids]

    assert tier_of("#48") == [DURABLE_TIER], (
        f"#48 is tiered in {tier_of('#48') or 'no tier'}; it applies CASE/PUNCT "
        f"nightly, so it writes durable state unattended, and it belongs in exactly "
        f"one tier")
    for job in ("#67", "#84"):
        assert tier_of(job) == [PROPOSES_TIER], (
            f"{job} is tiered in {tier_of(job)}; its plan-only status is accurate "
            f"(#67's `--apply` was retired 2026-09-04, #84 passed none) and moving it "
            f"is the opposite error to the one #1524 fixes")
    for tier in (DURABLE_TIER, PROPOSES_TIER):
        ids, n = tiers[tier]
        assert n is not None and int(n) == len(ids), (
            f"'| {tier} |' counts {n} for {len(ids)} ids — the row this change "
            f"edited is the row whose count must have moved with it")
    grouped = {i for ids, _ in _table(FUNCTIONS_HEADER).values() for i in ids}
    tiered = [i for ids, _ in tiers.values() for i in ids]
    assert len(tiered) == len(set(tiered)) and set(tiered) == grouped, (
        "moving #48 between tiers must not change which jobs the table covers: "
        f"{sorted(set(grouped) ^ set(tiered))}")


def test_the_sweep_skill_and_the_doc_agree_on_which_tiers_apply():
    """The process boundary the defect sits on. §Canonicalize's claim is about
    what the job is *instructed* to do, and that instruction lives in the vault:
    `skills/entity-resolution-sweep/SKILL.md` is loaded as the prompt, and this
    document itself (§The skill is the job) says a vault edit takes effect on the
    next run with no deploy and no gate. So the doc can be made true here and
    falsified from the vault tomorrow, by a job that cannot see this file.

    Read across the boundary rather than around it, for the reason
    `board_presence.py` gives for the board: a skip when the vault is absent is a
    green that certifies nothing, and the vault is on every box that runs this
    suite.
    """
    import board_presence

    skill = board_presence.vault_root() / "skills" / "entity-resolution-sweep" / "SKILL.md"
    assert skill.is_file(), (
        f"{skill} is unreadable, and the claim pinned here — that §Canonicalize "
        f"describes the tiers #48 is told to apply — is a claim about that file")
    raw = skill.read_text()
    # Line by line, deliberately: `--apply` and `--tiers` sit on the same line
    # whenever the instruction is an instruction, and a match allowed to cross a
    # line break would pair a bare `--apply` with the next command's `--tiers` and
    # read an order that the skill never gave.
    ordered = {t for ln in raw.splitlines() if "--apply" in ln and "--tiers" in ln
               for m in [re.search(r"--tiers\s+([A-Z_,]+)", ln)] if m
               for t in m.group(1).split(",")}
    assert ordered == MECHANICAL_TIERS, (
        f"the skill orders `--apply --tiers {sorted(ordered)}` and this file records "
        f"{sorted(MECHANICAL_TIERS)}. If the ruling changed, the doc and this constant "
        f"change with it; if it did not, the skill did and that is the bigger question")
    text = _flat(raw)
    # Tight on purpose: the skill's own prose elsewhere says suffix clusters get
    # "surfaced for human" review, and a wide `SUFFIX … human` window matches that
    # and reports a human-only *apply* that is not stated there.
    assert re.search(r"never\s+(?:pass|apply)[^.]{0,40}SUFFIX"
                     r"|SUFFIX[^.]{0,40}\b(?:human|operator)\b", text, re.I), (
        "the skill no longer says a suffix apply is a human act or never happens on "
        "schedule. That exclusion is the half of §Canonicalize's sentence this test "
        "can re-derive and the doc cannot, and it is the bound the #990 ruling stops "
        "at")
    canon = _group_section(CANON_HEADER)
    for tier in sorted(MECHANICAL_TIERS):
        assert tier in canon, (
            f"the skill applies {tier} unattended and §Canonicalize never names it: "
            f"a section describing a write by its category is describing a write "
            f"nobody can bound")
    assert re.search(r"SUFFIX", canon), (
        "§Canonicalize never names the tier that stays a human act, so the reader "
        "cannot tell an approved-and-waiting merge from one that will never happen")
    assert "#48" not in _table(TIERS_HEADER)[PROPOSES_TIER][0], (
        "the tier table has #48 back under 'Proposes; an operator applies' while the "
        "skill still orders a scheduled apply — the pairing that made the doc false, "
        "and now red on the skill's side of the boundary too")


def test_the_sweep_apply_row_bounds_the_transaction_and_names_the_resumable_half():
    """#1558: §Canonicalize told the reader #48's apply was "one transaction per
    run", which reads as though the whole apply were bounded when only the store
    half is — the fact files move after the commit, and until #1558 nothing on
    disk said which of them had moved, so a kill there left a half-merge that
    neither a re-run nor `revert-suffix-merges.py` could finish.

    Both halves of the real bound are graded, on both sides of the vault boundary:
    one store transaction, then a file-move half that is resumable, naming
    `--resume`. Reading the skill as well as the row is not decoration — the skill
    is the prompt the job runs from, so a row fixed here and a skill left silent is
    a claim the next run can make false without this repo changing.
    """
    import board_presence

    row = next((ln for ln in _text().splitlines() if ln.lstrip().startswith("| #48 |")), None)
    assert row is not None, "there is no #48 row in the Canonicalize table any more"
    assert "one transaction per run" not in row.lower(), (
        "the row still claims the whole apply is one transaction. It is not: only the "
        "aliases and the edge rewrites are in the store, and the fact files move after "
        "that commit")
    flat = _flat(row)
    assert re.search(r"one store transaction", flat, re.I), (
        f"the row never names what IS bounded: {row}")
    assert "--resume" in flat and re.search(r"resum", flat, re.I), (
        "the row states no recovery for the half it now admits is outside the "
        "transaction, so a reader with a half-finished merge has nowhere to go")
    assert re.search(r"dir_operations?", flat), (
        "the row does not say WHAT the resume replays, which is the only reason a "
        "resume cannot simply be another --apply")

    skill = board_presence.vault_root() / "skills" / "entity-resolution-sweep" / "SKILL.md"
    assert skill.is_file(), f"{skill} is unreadable, and the bound pinned here is about that file"
    skill_flat = _flat(skill.read_text())
    assert re.search(r"ONE store transaction|one store transaction", skill_flat, re.I) \
        and re.search(r"transaction[^.]{0,80}(outside|resum)", skill_flat, re.I), (
        "the skill still describes the apply as one transaction with no word about the "
        "half that runs after it")
    assert "--resume" in skill_flat, (
        "the skill — the prompt the job actually runs from — names no way to finish a "
        "killed apply, so the operator improvises a re-run and loses the edge trail")


#: ── #1566: what §Queue the work credits its three jobs with ──────────────────
#
#: The doc's own §Queue the work opened by crediting the group with archiving —
#: "what gets researched, what reaches `up_next` on the kanban at
#: `~/obsidian/backlog/`, **what gets archived**" — and its #77 row spelled that
#: out as the job's role: "Archive stale and done tasks, clear draft clutter,
#: reprioritize what remains". The skill #77 binds forbids exactly that ("never
#: archives, edits, or deletes tasks"; "Do not touch any files. Report only."), so
#: the function named a doer and had none. The consequence was measured on the
#: board, not on the page: two weekly runs (2026-09-08, 2026-09-15) each used
#: their report to rediscover that nobody retires the closed items, and a
#: report-only job cannot file the finding it has found. #1566 is now the number
#: that holds the decision, in the doc and in the skill both.
#:
#: These four nodes read across the same boundary the sweep tests above do — the
#: doc says what a job is, the vault skill is what makes it so — and they keep the
#: agreement in both directions: the row may not exceed the skill, and the skill's
#: owner line may not go missing from the side the doc points at.
QUEUE_HEADER = "Queue the work"
QUEUE_TABLE = "| ID | Freq | Role |"
HYGIENE_SKILL = "skills/weekly-backlog-hygiene/SKILL.md"

#: A sentence about archiving is only allowed into this section if it says the
#: archiving does *not* happen. Four words, each visibly a negation, so what is
#: being permitted stays readable rather than becoming a matcher nobody can
#: re-derive.
NEGATORS = ("no ", "not ", "nothing", "never")

#: The reports the skill's bash script actually emits, read off its `echo` lines
#: (`=== Stale drafts (not updated in 30+ days) ===` &c). The role row is allowed
#: to name these and nothing else that implies a write.
HYGIENE_REPORTS = ("status", "stale draft", "duplicate id", "duplicate title")


def _sentences(text: str) -> list[str]:
    """Prose cut into sentences, with the doc's 88-column wraps rejoined."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", _flat(text)) if s.strip()]


def _role_cell(header: str, job: str) -> str:
    """The Role cell of `job`'s row in the table that starts with `header`."""
    lines = _text().splitlines()
    # Every copy of the header, not `lines.index`'s first one: the doc spells
    # `| ID | Freq | Role |` identically in three sections, so a first-match lookup
    # reads a different group's table and reports `no row for #77` while the real
    # row sits unexamined two hundred lines below.
    starts = [i for i, ln in enumerate(lines) if ln.strip() == header]
    assert starts, f"architecture/autonomy-jobs.md has no `{header}` table at all"
    found = []
    for i in starts:
        for line in lines[i + 2:]:                          # header, then |---| rule
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] == job:
                assert len(cells) == 3, (
                    f"{job}'s row under a {header!r} table has {len(cells)} cells, not "
                    f"ID/Freq/Role — the reader can no longer tell which column is the role")
                found.append(cells[2])
    assert len(found) == 1, (
        f"{job} has {len(found)} role rows in the doc's {len(starts)} ID/Freq/Role "
        f"tables. One job described twice is the double-assertion defect #1524 "
        f"pinpointed, and the test cannot tell which copy a reader believes")
    return found[0]


def _hygiene_skill() -> str:
    """#77's prompt, read from the live vault — the surface that makes the doc true.

    Read rather than skipped when absent, for the reason `board_presence.py` gives
    for the board: a green that certifies nothing is worse than a red, and the
    vault is on every box that runs this suite.
    """
    import board_presence

    path = board_presence.vault_root() / Path(HYGIENE_SKILL)
    assert path.is_file(), (
        f"{path} is unreadable, and clauses 2 and 4 pinned here are claims about "
        f"that file, not about this one")
    return path.read_text()


def test_queue_the_work_credits_no_job_with_archiving_backlog_items():
    """Clause 1. §Queue the work said the group decides "what gets archived" and
    §Measure repeated it ("#65, #35 and #77 all direct future effort — what gets
    researched, what reaches `up_next`, what gets archived"). No job in the fleet
    retires a closed backlog item: `sweep_autonomy_runs`
    (`scripts/groundskeeper/retention-sweep.py:368`) prunes logs, sessions and run
    records; `app/backlog_move.py` records a status move without touching a path;
    and the only writer that removes a backlog file is `backlog_task_delete`
    (`app/routers/backlog.py:754`), which deletes rather than archives and is
    wired to no job. So the phrase is not a shorthand for something true.
    """
    section = _group_section(QUEUE_HEADER)
    flat = _flat(section)
    assert "what gets archived" not in flat, (
        "§Queue the work is back to crediting the group with archiving. The clause "
        "this replaces is #1566's whole premise, and the phrase appears in two "
        "sections, so fix §Measure too")
    assert "what gets archived" not in _text(), (
        "the false group claim survives somewhere else in the doc; it is one claim, "
        "stated twice, and #1566 removes both")
    for sentence in _sentences(section):
        if "archiv" not in sentence.lower():
            continue
        assert any(n in sentence.lower() for n in NEGATORS), (
            f"an unnegated archiving sentence is back in §Queue the work: "
            f"{sentence!r}. Every such sentence here has to say the archiving does "
            f"not happen, because no job performs it")
    # The positive half of the clause: the row now states the job's real work.
    row = _role_cell(QUEUE_TABLE, "#77")
    assert "report-only" in row.lower(), (
        "the #77 role row no longer says what the job is: a report-only census")
    for report in HYGIENE_REPORTS:
        assert report in row.lower(), (
            f"the #77 row no longer names {report!r}, one of the four reports the "
            f"job's own script emits — the row is describing a job nobody runs")


def test_the_77_role_row_agrees_with_the_skill_it_binds():
    """Clause 2, across the process boundary the defect sits on. §Queue the work's
    #77 row promised "Archive stale and done tasks, clear draft clutter,
    reprioritize what remains" while `skills/weekly-backlog-hygiene/SKILL.md` —
    loaded whole as the run's prompt — forbids archiving, editing and deleting, and
    closes with "Do not touch any files. Report only." A doc row and a prompt may
    not disagree about what a job is: the row is what a reader believes, the prompt
    is what the run does, and the gap between them is where two weekly runs lost
    their reports.
    """
    skill_flat = _flat(_hygiene_skill())
    # What the skill forbids, taken from the skill rather than typed here, so the
    # row is checked against the prohibition rather than against this test's memory.
    m = re.search(r"never\s+([a-z, ]+?)\s+tasks", skill_flat, re.I)
    assert m, (
        f"{HYGIENE_SKILL} no longer states its forbidden verbs as "
        f"'never <verbs> tasks'; this test reads the list from that sentence and "
        f"refuses to guess it")
    forbidden = [v.strip(" .") for v in re.split(r",|\bor\b", m.group(1)) if v.strip(" .")]
    assert forbidden, (
        f"no verbs parsed out of {HYGIENE_SKILL}'s own 'never … tasks' clause, so "
        f"there is nothing to check the #77 row against")
    row = _role_cell(QUEUE_TABLE, "#77").lower()
    for verb in forbidden:
        stem = verb.rstrip("s").replace("delete", "delet").replace("modif", "modif")
        assert re.search(rf"{stem}\w* nothing", row) or "touches no file" in row, (
            f"the skill forbids {verb!r} and the #77 row neither negates it nor "
            f"carries the blanket 'Touches no file'. The row is one edit away from "
            f"the claim that made this item")
    assert "files nothing" in row, (
        "the row dropped the filing prohibition. #77 does not file the finding it "
        "finds either — that is the half #1566 is about, and the run of 2026-09-15 "
        "declined to file precisely because of it")
    assert re.search(r"[Dd]o not touch any files", skill_flat), (
        f"{HYGIENE_SKILL} no longer says a run may touch no file, which is the "
        f"sentence 'Touches no file' in the row is answering")
    for report in HYGIENE_REPORTS:
        assert report in skill_flat.lower(), (
            f"the row still promises {report!r}, which the script stopped emitting")


def test_queue_the_work_names_an_owner_for_the_closed_pile():
    """Clause 3. The finding outlives the false claim if nothing replaces it: with
    the archiving sentence deleted and no owner named, §Queue the work leaves a
    reader holding an unowned TODO, which is the state that produced #1566 in the
    first place. The section now states the absence plainly and names the item that
    holds the decision, and this node pins both halves plus the citations the
    paragraph leans on, so a paragraph cannot stay green while its own references
    rot.
    """
    flat = _flat(_group_section(QUEUE_HEADER))
    assert re.search(r"\bno job retires a closed item off the board\b", flat, re.I), (
        "§Queue the work no longer says plainly that retiring a closed item is "
        "performed by no job. Deleting the false archiving sentence without stating "
        "this is what leaves the TODO unowned")
    assert re.search(r"backlog #1566", flat), (
        "the closed pile has no named owner in the section a reader finds it in. "
        "#1566 is where the archive-or-accept decision lives; name it")
    assert re.search(r"#1566[^.]{0,200}\bdecision\b", flat, re.I), (
        "#1566 is named but not as the decision's home, which is the difference "
        "between an owner and a cross-reference")
    # The paragraph's own evidence, re-checked rather than trusted.
    root = ROOT
    router = (root / "app" / "routers" / "backlog.py").read_text()
    assert "async def backlog_task_delete" in router, (
        "app/routers/backlog.py no longer defines backlog_task_delete, the one "
        "writer that removes a backlog file — the paragraph cites it as deleting "
        "rather than archiving")
    assert "archiv" not in (root / "app" / "backlog_move.py").read_text().lower(), (
        "app/backlog_move.py gained an archiving route, and §Queue the work's "
        "'no job retires a closed item' needs to be re-read against it")
    page = (root / "web" / "src" / "components" / "pages" / "BacklogPage.tsx").read_text()
    m_win = re.search(r"const DONE_WINDOW_DAYS = (\d+)", page)
    assert m_win, (
        "BacklogPage.tsx no longer defines DONE_WINDOW_DAYS, the 7-day window the "
        "paragraph cites as the reason nothing downstream feels the pile")
    assert m_win.group(1) == "7", (
        f"the board's done-window is {m_win.group(1)} days, not the 7 the paragraph "
        f"cites — it still reads as a hiding window, so update the number")


def test_the_hygiene_skill_points_the_closed_pile_at_its_owner():
    """Clause 4, on the other side of the boundary. #77 is report-only three times
    over, so it cannot file the closed pile or land it; the only durable instruction
    it carries is this skill, and until it named an owner every run re-reported the
    pile as a fresh finding (2026-09-08, 2026-09-15). The doc's pointer and the
    skill's line are one fix in two surfaces, and this node is what stops the doc
    claiming an owner the prompt never tells the run to cite.
    """
    skill = _hygiene_skill()
    flat = _flat(skill)
    assert "#1566" in flat, (
        f"{HYGIENE_SKILL} names no owner for the closed pile, so a run reports it as "
        f"a new unowned finding again — the loop #1566 exists to close")
    owner = [s for s in _sentences(flat) if "#1566" in s and "owner" in s.lower()]
    assert owner, (
        "#1566 is mentioned in the skill but never as the owner of the finding; a "
        "bare cross-reference does not tell a run what to write in its report")
    assert re.search(r"cite\s+\*{0,2}owner #1566\*{0,2}[^.]{0,160}instead of reporting",
                     flat, re.I), (
        "the skill no longer instructs the run to cite owner #1566 in place of "
        "re-listing the pile. Naming the item is decoration; the instruction is the "
        "fix")
    assert "what gets archived" not in flat and "Archive stale" not in flat, (
        "the skill has taken on the archiving role the doc just gave up. It is "
        "report-only; the owner it names is #1566, not itself")


# ── #1568: what #35's eight-to-twelve actually is ───────────────────────────
#
# One status, two owners, and one number that described neither of them. `up_next`
# is the target of #35's promotions and, via `IMPLEMENT_POOL_STATUS` in
# `scripts/automod/backlog.py`, the self-modification loop's take-list. The job
# table handed a reader "target queue size 8–12" for both, while the loop bounds the
# same status at `max(IMPLEMENT_POOL_FLOOR, items landed in the trailing 7 days)` —
# `bound` 268 against `floor` 20, `ready` 0, measured 2026-09-27 — and counts a
# different set entirely: `ready_confirmed`, which additionally requires the item's
# latest ledger `confirmed` verdict to carry acceptance. A triage promotion writes a
# front-matter status and an `activity` line and no verdict, so it is never in that
# count at all, and neither number can validate the other.
#
# The boundary the defect actually sat on is not this file. The prompt a #35 run is
# served is `skills/backlog-triage/SKILL.md` plus that task's front-matter
# `description:` and nothing else (`app/autonomy.py::_build_task_prompt`), so a
# relabel that stopped in the repo doc would have changed what nobody executes and
# left the phrasing in front of the worker. Read across it, for the reason
# `board_presence.py` gives: the vault is on every box that runs this suite, so a
# missing vault is an assertion failure here, never a skip.

TRIAGE_CARRIERS = ("queue size", "8–12", "8-12")   # en dash and hyphen both


def _triage_skill() -> str:
    """The live triage skill, which is the job's prompt."""
    import board_presence

    path = board_presence.vault_root() / "skills" / "backlog-triage" / "SKILL.md"
    assert path.is_file(), (
        f"{path} is unreadable, and #35's instructions ARE that file — §Queue the "
        f"work's row for it is prose about it, not the job")
    return path.read_text()


def _task_35_front_matter() -> dict:
    """Task #35's parsed front matter, from the live vault."""
    import board_presence
    import yaml

    root = board_presence.vault_root() / "autonomy"
    files = sorted(root.glob("35-*.md"))
    assert len(files) == 1, (
        f"expected exactly one `autonomy/35-*.md` under {root}, found "
        f"{[f.name for f in files]} — with two, which one the engine loads is "
        f"undefined and this file cannot say which is being pinned")
    parsed = yaml.safe_load(files[0].read_text().split("---", 2)[1])
    assert isinstance(parsed, dict) and parsed.get("id") == 35, (
        f"{files[0]} does not parse to the task with id 35")
    return parsed


def test_the_triage_skill_labels_its_budget_as_attention_not_queue_size():
    """Clause 1. The number survives; what it is called does not.

    "Ideal `up_next` queue size: 8–12 items" made a review budget read as a fleet
    queue bound, in the one file that both the worker and §Queue the work describe.
    Step 2 now holds the pass to a human attention budget of eight to twelve
    promoted items and says so in those words, and the phrase "queue size" is gone
    from the file. The guard the relabel could have been used to gut — "Never
    promote more than 5 items per run" — is asserted as the floor it stays at: an
    attention budget of twelve was never a licence to promote twelve in one run.
    """
    skill = _triage_skill()
    stale = [c for c in TRIAGE_CARRIERS if c in skill]
    assert stale == [], (
        f"{stale} is back in skills/backlog-triage/SKILL.md. The cap this pass "
        f"applies is a human attention budget for untriaged promotions; the loop's "
        f"own limit is implement_pool.bound, and the skill's job is to point at it")
    assert "attention budget" in skill, (
        "the surviving cap is unlabelled again, which is the whole of #1568")
    assert "Never promote more than 5 items per run (keep queue manageable)" in skill, (
        "the per-run promotion guard went missing in the relabel. Twelve is what a "
        "person can review afterwards; five is what one run may write")


def test_the_triage_skill_points_the_run_at_the_bound_the_loop_derives():
    """Clause 2. The limit it must not chase, with the command that reads it.

    Step 2 names `board_health`'s `implement_pool.bound` / `.ready` and the bound's
    derivation, because "do not chase it" is only executable by a run that can look
    at it. So the command is not decoration: this node checks the names the skill
    tells the run to read are real names in the dict that call returns, and pins the
    constant the skill quotes (`IMPLEMENT_POOL_FLOOR = 20`) against the module that
    defines it. The bound is landing-rate driven, so it is not asserted as a value —
    on 2026-09-27 it read 268 with `ready` at 0, which is the point: with a floor of
    20 and 267 items landed in the trailing week, nothing about the derived side
    binds today, and a promotion pass reading it as a target would be promoting
    toward a number it cannot influence.
    """
    from scripts.automod import backlog as B
    from scripts.automod import state as S

    skill = _flat(_triage_skill())
    for token in ("board_health", "implement_pool", ".bound", ".ready"):
        assert token in skill, (
            f"Step 2 no longer names `{token}`, so the run has no way to read the "
            f"bound this file tells it not to chase")
    assert "IMPLEMENT_POOL_FLOOR" in skill and "trailing 7 days" in skill, (
        "the skill stopped stating how the bound is derived, and a reader is left "
        "with a large number sitting unexplained next to eight to twelve")
    assert B.IMPLEMENT_POOL_FLOOR == 20, (
        f"IMPLEMENT_POOL_FLOOR is {B.IMPLEMENT_POOL_FLOOR} in the code while the "
        f"skill still quotes 20 — the constant moved, and prose quoting a constant "
        f"has to move with it")
    assert "would be wrong" in skill, (
        "the skill lost the sentence saying promoting toward `bound` is wrong. "
        "Naming a large derived ceiling beside a small attention budget without that "
        "caveat is an invitation, which is how #1568's option A reads")

    pool = B.board_health(S.LEDGER_PATH).get("implement_pool")
    assert isinstance(pool, dict) and {"bound", "ready", "floor"} <= set(pool), (
        f"board_health()['implement_pool'] is {pool!r}, and every #35 run is told to "
        f"read `bound` and `ready` out of exactly that")
    assert pool["bound"] >= pool["floor"] == B.IMPLEMENT_POOL_FLOOR, (
        f"implement_pool reads {pool!r}: the bound no longer sits at or above the "
        f"floor the skill quotes, so implement_pool_bound's max() is not what the "
        f"prose says it is")


def test_the_triage_skill_names_the_denominator_the_loop_actually_counts():
    """Clause 3. Same status, different quantity, said out loud.

    `implement_pool_full` reports `ready` as `len(ready_confirmed(...))` — items
    that are `status == up_next` AND carry a ledger `confirmed` verdict with
    acceptance, minus grouped members, human-only items, spent attempts and rounds
    the loop is already gating or landing. #35's promotions write a status and an
    `activity` line and no verdict, so they never enter that set. Which means
    `ready: 0` is not a licence to promote toward `bound`, and a skill that does not
    say so out loud will be read as if it were.
    """
    from scripts.automod import backlog as B
    from scripts.automod import state as S

    skill = _flat(_triage_skill())
    assert "ready_confirmed" in skill, (
        "the skill stopped naming the set the loop's gate counts, which is the only "
        "reason the two numbers can never be reconciled by size alone")
    line = next((ln for ln in _triage_skill().splitlines() if "ready_confirmed" in ln), "")
    assert "verdict" in line and "acceptance" in line, (
        f"the `ready_confirmed` line is now {line!r} — naming the function without "
        f"the verdict-and-acceptance requirement is exactly how a reader concludes "
        f"raw `up_next` is what fills the pool")
    assert "invisible" in skill, (
        "the skill lost the statement that a triage promotion is invisible to the "
        "loop's gate. The asymmetry is the finding, not a footnote to it")

    gate = B.implement_pool_full(S.LEDGER_PATH)
    assert {"full", "ready", "bound", "floor"} <= set(gate), (
        f"implement_pool_full returns {sorted(gate)}; the skill's sentence describes "
        f"a gate whose report no longer has these keys")
    assert gate["ready"] == len(B.ready_confirmed(S.LEDGER_PATH)), (
        "`implement_pool_full.ready` is no longer `len(ready_confirmed(...))`, so "
        "what the skill tells the run about the denominator and what the gate counts "
        "have come apart — which is the defect #1568 was filed on")
    raw = B.board_health(S.LEDGER_PATH)["up_next"]["total"]
    assert gate["ready"] <= raw, (
        f"ready_confirmed reports {gate['ready']} while raw `up_next` holds {raw}. "
        f"The confirmed set is a subset of the status set by definition; if it ever "
        f"exceeds it, the two are no longer the same status and the skill's "
        f"explanation of the difference is false")


def test_the_prompt_task_35_is_served_names_no_queue_size_target():
    """Clause 4, across the seam that carries the number to the model.

    `_build_task_prompt` renders the skill plus the task's front-matter
    `description:` and nothing else, and that description used to read "Target queue
    size: 8-12 items" — a second carrier that reached every run independently of the
    skill, which is why editing the skill alone would not have closed this. So
    assert against the rendered prompt, not the file: the thing that has to be free
    of the phrasing is what the worker actually reads.
    """
    from app.autonomy import _build_task_prompt

    task = _task_35_front_matter()
    desc = str(task.get("description", "")).strip()
    assert desc, "task #35's `description:` is empty"
    stale = [c for c in TRIAGE_CARRIERS if c in desc]
    assert stale == [], (
        f"{stale} is back in #35's `description:`, which is rendered to the worker "
        f"beside the skill whether or not the skill still says it")
    assert "attention budget" in desc, (
        "the description no longer says what the budget it defers to is")

    prompt = _build_task_prompt(task, _triage_skill())
    assert "Task description:" in prompt, (
        "_build_task_prompt stopped rendering the description, so this node and the "
        "clause it pins are measuring a seam that has moved")
    leaked = [c for c in TRIAGE_CARRIERS if c in prompt]
    assert leaked == [], (
        f"the prompt a #35 run is served still carries {leaked}, from a carrier "
        f"outside the skill. Find it and relabel it — the skill is not the only door")
    assert "attention budget" in prompt, (
        "neither carrier that reaches the run names the cap an attention budget, so "
        "the run cannot tell it apart from a queue bound")


def test_the_job_row_labels_the_budget_and_cross_references_the_derived_bound():
    """Clause 5, in the file a reader who never opens the vault actually reads.

    §Queue the work's row for #35 was the only place in the repo stating 8–12, and
    it stated it as the queue target — the fleet's queue policy, to anyone reading
    the job doc. It now labels the cap a human attention budget and points at
    `implement_pool_bound` for the real limit, and the cross-reference is checked to
    resolve rather than merely present: a pointer into `[[automod]]` that lands on no
    such name is the same dead end as no pointer.

    One word changed in passing on the line being rewritten: the row promoted "inbox
    items", and the vault retired `inbox` in favour of `draft` (#786 — the same
    retirement the node above §Queue cites, whose commit lives on the vault's main).
    """
    role = _role_cell("| ID | Freq | Role |", "#35")
    stale = [c for c in TRIAGE_CARRIERS if c in role]
    assert stale == [], (
        f"{stale} is back in #35's role cell. Stating the attention budget there is "
        f"fine; stating it as the queue size is the conflation #1568 exists to kill")
    assert "attention budget" in role, "the row's cap is unlabelled again"
    assert "implement_pool_bound" in role, (
        "the row stopped pointing at the loop's real depth limit")
    assert "[[automod]]" in role, (
        "the row lost the link to the document that derives the bound it cites")
    automod = (ROOT / "architecture" / "automod.md").read_text()
    assert "implement_pool_bound" in automod, (
        "the row cross-references [[automod]] for `implement_pool_bound` and that "
        "name is not in architecture/automod.md — a cross-reference to a place that "
        "does not hold the thing is how a doc starts lying again")
    assert "`draft`" in role and "inbox" not in role, (
        f"#35's row now reads {role[:70]!r}… — the vault retired `inbox` in favour of "
        f"`draft` (#786), and a job row naming a status the board cannot hold is the "
        f"doc contradicting the state machine")
