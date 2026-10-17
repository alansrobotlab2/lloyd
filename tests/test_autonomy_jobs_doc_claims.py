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
name* only (one `autonomy-jobs:Measure` entry in
`workers.sources.arch-review.groups` in `config.yaml`), so it never asks who is
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

A third contract joined with #2805 (2026-10-16): a job that has run must not be
described in live prose as one that never ran. §Watch the machinery stated that
#97 had never run and that its journal did not exist, two days after its first
nightly run had appended a row — a paragraph written from the skill and the
module's docstring, which is what it said itself, instead of from the artefact the
job had by then produced. The rule the four nodes at the foot of this file hold is
therefore about *form*: an undated never-run claim is unfalsifiable until it rots,
so every one in live prose has to carry the date it was measured and the command
that re-derives it, and a superseded claim survives only inside a dated §Review log
entry. No node here asserts a run count: #97 and #98 fire daily, so any number
typed into a test is false the next morning — which is the same reason the doc
dates its counts, and why the owed-check re-reads the count instead of a test
freezing it.

A fourth contract joined with #2806 (2026-10-16): what `config.yaml` offers as a
review unit has to be the doc's own list of groups, in both directions. The pair
was read one way only — a configured name with no heading is reported, a heading
with no configured entry is invisible — and §Watch the machinery went in as the
doc's eighth group with no entry beside the seven, so its seven jobs were in no
unit a reviewer could be sent to. The reconciliation reads the live config through
`workers.sources.get_sources_config` and the live doc through `find_section`, so it
binds the two files the cadence actually reads rather than two copies of them, and
its failure is demonstrated on a seeded heading rather than described.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

#: Reads the live `~/obsidian`, which no round under test controls — the gate
#: deselects it (`pytest.ini:10-13`). Same convention as
#: tests/test_autonomy_task_prompt_claims.py:83.
live_vault = pytest.mark.live_vault

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

#: The jobs created since the last membership pass that both tables must list:
#: #90 (`corpus-shape-trend`, the shape metrics over four nightly corpora, created
#: 2026-09-25) and #91–#97, created between 2026-09-29 and 2026-10-11, which
#: reached 2026-10-12 with no entry anywhere in the document. The omission was
#: invisible to `test_every_job_has_one_function_and_exactly_one_write_tier`
#: because it cross-checks the two tables against *each other*: symmetric omission
#: passes, and no test reads `~/obsidian/autonomy/` (#2631 owns that check).
#: #91–#97 joined the tables through #2630.
NEW_JOBS = ("#90", "#91", "#92", "#93", "#94", "#95", "#96", "#97")

#: The three paper digests #2630 took out of the membership tables. Each was a
#: `research-agent` job that read one named arXiv paper into exactly one knowledge
#: note and was dispatched interactively on 2026-09-24; as of 2026-10-12 no
#: `87-*.md`, `88-*.md` or `89-*.md` exists in `~/obsidian/autonomy/` or in its
#: `_archived/`. They are the mirror image of the seven omissions above — the
#: tables named members with no task file behind them — and both directions were
#: unnoticed for the same reason: nothing reconciles the tables with the directory.
#: #90, created the same fortnight and still a live task file, is the positive
#: control in `test_the_departed_paper_digests_hold_no_membership_row`.
DEPARTED = ("#87", "#88", "#89")

#: The jobs §Watch the machinery holds, each with the schedule its own task file's
#: `frequency:` declares and three phrases its entry has to contain: what it reads,
#: the artefact it writes, and one thing it must not do. Read off
#: `~/obsidian/autonomy/9[1-7]-*.md` on 2026-10-12, off `99-*.md` on 2026-10-16
#: (#2813) and off `100-*.md`/`101-*.md` on 2026-10-17 (#2881), and typed here rather
#: than parsed from the vault on purpose: a check that opened the task directory is a
#: `live_vault` read, which the gate deselects (`pytest.ini:10-13`), and every
#: clause #2630 was accepted on has to be gradable by the rung that really runs.
#: The two 2026-10-16 arrivals are `status: draft`, which is no bar: #98 and #99 are
#: draft too and both are listed, because the table records what a filed task file
#: names, not what the scheduler has armed.
WATCH_JOBS = {
    91: ("daily", "~/lloyd-data/sessions", "thinking-fidelity-prevalence.jsonl",
         "preserve_thinking_iterations"),
    92: ("daily", "agent-llm-primary", "vllm_prefix_miss_", "stage or commit"),
    93: ("daily", "~/lloyd-data/autonomy-runs", "backlog draft", "learn --out"),
    94: ("daily", "mental-models.md", "referential-integrity/", "expect-dangling"),
    95: ("daily", "api/workers/status", "mitigation_drill.json",
         "scripts.mitigation_drill"),
    96: ("weekly", "name_prior_corpus.jsonl", "eval/djev/name_prior_", "--samples"),
    97: ("daily", "automod ledger", "landed_outcomes.jsonl", "--since"),
    # #2813: the four probes read off ~/obsidian/autonomy/99-supply-chain-inventory-regen.md
    99: ("weekly", "eval/supply-chain/baseline.yaml", "~/lloyd-data/supply-chain/",
         "`Write` or `Edit`"),
    # #2881: the four probes read off ~/obsidian/autonomy/100-eval-instrument-index-daily.md
    # and 101-gate-to-land-lag-daily-record.md on 2026-10-17. Both declare
    # `frequency: daily`, both write exactly one dated row to one JSONL outside this
    # repo, and neither may re-sort or compact the history it just appended. #100's
    # read is spelled `wc -l` — the count step its own task file names — rather than
    # the publisher's module name, because that module's name appearing in any module
    # is what `test_no_gate_promotion_or_rollback_path_reads_the_index` counts as a
    # reader of the published index outside the publisher itself.
    100: ("daily", "wc -l", "instrument-index.jsonl", "re-sort"),
    101: ("daily", "gate_to_land_lag.py", "gate_to_land_lag.jsonl", "re-sort"),
}

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


def _group_names() -> list[str]:
    """The function groups' heading prefixes, read from §The functions' first column.

    Derived rather than typed: a second hand-maintained list of the eight groups is
    exactly the thing this file keeps having to correct, and the table's own link text
    *is* the prefix of the heading the anchors must resolve to. So the node below
    follows the table instead of policing a copy of it.
    """
    body = _text().split("## The functions", 1)[1].split("\n## ", 1)[0]
    names = re.findall(r"^\| \[([^\]]+)\]\(#[^)]*\) \|", body, re.M)
    assert len(names) == 8, (
        f"§The functions' first column yields {len(names)} group links ({names}), not "
        f"the eight the doc's prose claims — the anchor node cannot tell a stale slug "
        f"from a table it cannot read")
    return names

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


def _parked_job_section() -> str:
    """§#68 — the parked-job subsection, prose flattened.

    Scoped to the subsection rather than the whole document so a citation checked
    "in the parked job's hazard" cannot be satisfied by the same id named in some
    other section, the way #1520 is about two sections disagreeing.
    """
    after = _text().split("### #68 — the name outlived the job", 1)
    assert len(after) == 2, "architecture/autonomy-jobs.md has no §#68 subsection to scope to"
    return _flat(re.split(r"\n#{2,3} ", after[1], 1)[0])


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
    # Live prose only, since #2813: the §Review log entry of that day quotes the
    # bound it found ("the hand-typed floor moved to `24 through 99 or beyond`"),
    # and that quotation is the dated record of what the note said before it moved.
    # A scan of raw lines would read the quote as a second hand-maintained ceiling.
    found = {md.name: [ln.strip() for ln in _live_prose().splitlines()
                       if re.search(r"\b\d+\s+through\s+\d+\b", ln)]
             if md.name == "autonomy-jobs.md" else
             [ln.strip() for ln in md.read_text().splitlines()
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


# ── #2892 — §#68's `model:` census is a dated typed MINIMUM, not a count ──────


def test_the_model_census_states_a_dated_minimum_and_says_so_in_words():
    """#2892 clause 4, and the record of which branch this doc took.

    The item offered two ways to stop §#68's `model:` census lagging the fleet (it was
    retyped three days running: #2747 39→40, #2814 40→41, #2882 41→43). The first was
    to read the floor off the §functions membership tables, the way
    `test_the_fleet_bound_floors_the_range_at_the_newest_task` reads `newest`. That
    branch is unsatisfiable as worded, and this node is where the reason is recorded
    where the next reader will meet it rather than in a round's report: both tables
    parse to 44 ids and one of them, #36, is retired with no task file
    (`_retired_named_ids(_text())` → `['#36']`), while the model-declaring fleet is 43
    files — `grep -h '^model:' ~/obsidian/autonomy/*.md | wc -l` → 43. A floor read
    off the tables would state 44 and the real-board node would fail its own `>=` the
    day it landed. Subtracting retired ids is #2895's writer-side check, not a
    sentence in this doc.

    So the number stays hand-typed, and what is pinned here is the shape that lets a
    typed number survive a fleet that grows: floor wording, the word "minimum" beside
    the figure so nobody retypes it as a count, and the measurement date in front of
    it. `tests/test_validate_tasks_linter.py` owns the parser and the join to the
    directory; this file owns what the sentence claims.
    """
    section = _parked_job_section()
    m = re.search(
        r"`grep -h '\^model:' ~/obsidian/autonomy/\*\.md` answers `(?P<value>\w+)` "
        r"at least (?P<count>\d+) times? or beyond", section)
    assert m, (
        "§#68 no longer states its `model:` census as a floor in the shape `answers "
        "`primary` at least N times or beyond`, immediately after the command that "
        "measures it — that spelling is what let the count be retyped on 2026-10-15, "
        "2026-10-16 and 2026-10-17 instead of simply being reached")
    assert m.group("value") == "primary", (
        f"the floor is stated for {m.group('value')!r}, but every task file declaring "
        f"a `model:` reads `primary`")
    assert re.search(r"20\d\d-\d\d-\d\d", section[max(0, m.start() - 200):m.end()]), (
        "the floor carries no ISO measurement date within 200 characters before the "
        f"number: …{section[max(0, m.start() - 200):m.end()][-160:]!r} — a minimum "
        "without its date is a claim no later reader can date-stamp as stale")
    tail = section[m.end():m.end() + 160]
    assert "minimum" in tail, (
        f"the sentence does not say in words that {m.group('count')} is a minimum "
        f"(nothing within 160 characters after the floor says so: {tail!r}) — a bare "
        "'at least N' reads as a count to the next editor, and the count is what got "
        "retyped three times")
    assert "ceiling" in tail, (
        f"the sentence no longer says what the floor is NOT ({tail!r}); #1520's defect "
        "was a typed ceiling on a fleet that grows, and the reader has to be told that "
        "is the thing this sentence refuses to be")
    # Flattened: the doc wraps near 88 columns, and a wrap inside the sentence is not a
    # second census — see `test_no_second_fleet_bound_appears_anywhere_in_architecture`.
    live = _flat(_live_prose())
    floors = re.findall(r"answers `\w+` at least \d+ times? or beyond", live)
    assert len(floors) == 1, (
        f"live prose now holds {len(floors)} `model:` census floors {floors}: one is "
        f"§#68's, a second is a second sentence for whoever notices it rotting")
    # Requires digits immediately after the value, so §#68's own floor cannot match it.
    exact = re.findall(r"answers `\w+` \d+ times?", live)
    assert not exact, (
        f"an exact `model:` census count is back in live prose ({exact}); the exact "
        f"shape is retired — a floor or nothing")


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


def test_the_parked_job_s_slot_confusion_is_cited_by_its_backlog_item():
    """#2634. The 2026-10-12 arch-review edit (`1d41b432`) that added #68's
    stale-slot hazard named it "the #85 confusion of #1555", and that appositive put
    a retired job's id back into current prose — the exact shape
    `test_the_retired_secondary_eval_is_recorded_only_in_the_dated_log` forbids, which
    is how the doc went red with no round's diff in front of it.

    The hazard stays and the citation moves, and both halves are pinned so the fix
    cannot be repeated by deleting the pointer along with the id: the subsection
    still says in its own words what arming the parked job would do, and still cites
    backlog **#1555**, the incident that confusion was recorded in.
    """
    section = _parked_job_section()
    assert "resolve_model_alias" in section and "retired slot" in section, (
        "§#68 no longer states the stale-slot hazard in its own words, so there is "
        "nothing left for a citation to belong to")
    assert "#1555" in section, (
        "§#68's stale-slot hazard is no longer cited to the backlog item that "
        "recorded it, so the sentence keeps its claim but loses its evidence")


def test_the_retired_eval_check_reaches_green_by_running_not_by_bookkeeping(request):
    """#2634 clause 2. The acceptance is that the node above passes because the
    document was corrected — not because a skip, an xfail, a `live_vault` re-mark or a
    deletion moved it off the gate's selection (`-m "not live_vault and not
    fault_injection"`).

    Each of those shapes is pinned semantically, never by a token scan this file's
    own prose could defeat: the node is still *defined* here, it is still in
    `session.items` **of the run that is executing** (deselection removes an item
    from that list, so a re-mark or an excluded expression fails this), and no test
    function in this module carries a skip or xfail mark. Run it under `-k "not
    test_the_retired_secondary_eval"` and this test fails — that is the proof it can
    fail.
    """
    name = "test_the_retired_secondary_eval_is_recorded_only_in_the_dated_log"
    assert name in globals(), (
        "the node the acceptance names is no longer defined in this module; deleting "
        "a check is not the same as passing it")
    assert not getattr(globals()[name], "pytestmark", []), (
        "the node now carries a mark of its own, which is how a check gets excluded "
        "from the gate's selection instead of being run by it")
    ran = {it.name for it in request.session.items}
    assert name in ran, (
        f"the node the acceptance names is not in this run's items, so it was "
        f"deselected rather than made true — {len(ran)} other nodes are running")
    suppressed = sorted(
        n for n, o in globals().items()
        if n.startswith("test_") and callable(o)
        and any(m.name in {"skip", "skipif", "xfail"} for m in getattr(o, "pytestmark", [])))
    assert not suppressed, (
        f"{suppressed} carry a skip or xfail mark; the acceptance forbids reaching "
        "green by suppressing a run")


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


#: A section the doc itself declares to be a functional group: a `## <Name>: <ids>`
#: heading, and under it at least one `| #N |` job row. The shape is the whole
#: discriminator — §The functions, §What the fleet actually costs, §Related and
#: §Review log are `##` headings too (and §The functions is a per-job table of the
#: groups), but none of them reads `## <Name>: <ids>`, so they are excluded by what
#: they are rather than by a list of names that would stop excluding them the day a
#: fifth prose section went in.
_GROUP_HEADING = re.compile(r"^## (?!#)(?P<name>[^:]+):(?P<ids>[^\n]*#\d+[^\n]*)$")
_GROUP_JOB_ROW = re.compile(r"(?m)^\| #\d+ \|")


def _arch_review_groups() -> list[str]:
    """The raw `workers.sources.arch-review.groups` list, through the reader production uses.

    `get_sources_config` reads `app.config.CONFIG`, and `app.paths` resolves that from
    the checkout the tests run in — so in a round's worktree or a throwaway checkout
    this is that tree's list, which is the copy the change under test is in. A literal
    copy of the list here would be a second source of the membership, which is the
    thing #2806 is about.
    """
    from workers.sources import arch_review as A
    from workers.sources import get_sources_config

    return list((get_sources_config().get(A.NAME, {}) or {}).get("groups") or [])


def _configured_group_names(groups: list[str]) -> list[str]:
    """The `autonomy-jobs` names a raw `groups:` list holds, as `parse_groups` reads them."""
    from workers.sources import arch_review as A

    return [name for slug, name in A.parse_groups(groups) if slug == "autonomy-jobs"]


def _job_group_names(text: str) -> list[str]:
    """The groups `text` declares, in the order their headings appear.

    The span comes from `find_section` — the matcher `workers/sources/arch_review.py`
    uses to turn a configured name into the lines it hands a reviewer — so a name
    counted here is a name that source could actually have reviewed.
    """
    from workers.sources import arch_review as A

    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        m = _GROUP_HEADING.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        span = A.find_section(text, name)
        if span is None:
            continue                                    # not this shape after all
        if _GROUP_JOB_ROW.search("\n".join(lines[span[0] - 1:span[1]])) and name not in out:
            out.append(name)
    return out


def _group_headings_the_config_never_names(text: str, groups: list[str]) -> list[str]:
    """Every group `text` declares that `groups` has no entry for.

    Both arguments are inputs, not reads of the live tree, which is what lets the
    failure be demonstrated rather than asserted: the same expression over the live
    doc and the live `groups:` list is the green case, and over a doc carrying one
    extra seeded group, or a list with one entry removed, it names what drifted.
    Comparison goes through `heading_key` because that is how the source compares a
    configured name to a heading — case-insensitively, and only up to the first `:`.
    """
    from workers.sources import arch_review as A

    named = {A.heading_key(name) for slug, name in A.parse_groups(groups)
             if slug == "autonomy-jobs"}
    return [g for g in _job_group_names(text) if A.heading_key(g) not in named]


def test_every_group_heading_in_the_doc_is_configured_as_a_review_unit():
    """#2806. The other direction, which nothing walked before.

    `test_every_configured_group_still_resolves_to_a_heading_in_the_doc` asks whether
    every configured name has a heading, and a group added to the doc answers it by
    not being in the list asked about. `workers/sources/arch_review.py` reports
    `section_missing` on that same single direction, so the only teller of a group
    with rows and no config entry was a model told to notice it, in a whole-doc unit
    that rests `review_interval_days` at a time. §Watch the machinery went in at
    `b221481c` (2026-10-12) as the doc's eighth group with no entry beside the seven,
    and nothing downstream changed: the seven `| #91 |` rows simply had no unit, and
    `~/.local/state/lloyd-automod/arch_review.json` carries a `group:autonomy-jobs:`
    key for each of the seven and none for it.
    """
    doc, groups = _text(), _arch_review_groups()
    declared, named = _job_group_names(doc), _configured_group_names(groups)
    # Three statements of one membership, so an empty answer below is the pair
    # agreeing and not a heading pattern that matched nothing.
    assert declared and named, (
        f"the doc declares {len(declared)} groups and config.yaml names {len(named)}; "
        f"one of the two reads empty, which is a broken pattern, not a clean tree")
    funcs = _table(FUNCTIONS_HEADER)
    assert len(named) == len(declared) == len(funcs), (
        f"config.yaml names {len(named)} autonomy-jobs groups ({sorted(named)}), the "
        f"doc's headings with job rows are {len(declared)} ({sorted(declared)}) and "
        f"§The functions lists {len(funcs)} — these are one membership in three "
        f"places and they have come apart")

    unconfigured = _group_headings_the_config_never_names(doc, groups)
    assert not unconfigured, (
        f"{unconfigured} hold `| #N |` job rows under a `## <Name>: <ids>` heading in "
        f"architecture/autonomy-jobs.md, but config.yaml's "
        f"`workers.sources.arch-review.groups` names no such unit, so no reviewer is "
        f"ever offered them and no check notices — add the `autonomy-jobs:<Name>` "
        f"entry beside the others")


def test_a_dropped_or_seeded_group_fails_the_review_unit_check():
    """The reverse check has to be able to fail, and the live pair is the control
    that shows it is not failing for the wrong reason.

    Both ways round: a heading the list never named (the §Watch the machinery case,
    seeded so the check is shown catching it rather than describing it) and a list
    that lost an entry its doc still has. Either way the drifted group is the thing
    the result names, because a check that reported "N groups, M entries" without
    saying which would be the same unreadable verdict #2630 was filed about.
    """
    groups = _arch_review_groups()
    doc = _text()
    assert _group_headings_the_config_never_names(doc, groups) == [], (
        "the live doc and the live config already disagree, so nothing below can "
        "show a seeded drift — fix #2806 first")

    seeded, n = re.subn(
        r"(?m)^## What the fleet actually costs$",
        "## Seeded group: #999\n\n"
        "| # | Job | What it does |\n"
        "|---|---|---|\n"
        "| #999 | seeded | a row that exists only to prove the check sees it |\n\n"
        "## What the fleet actually costs", doc, count=1)
    assert n == 1, "the seeded heading found no host section, so it proved nothing"
    assert _group_headings_the_config_never_names(seeded, groups) == ["Seeded group"], (
        "a group heading with a `| #999 |` row and no config entry came back clean, "
        "which is exactly how §Watch the machinery read for four days")

    dropped = [g for g in groups if g != "autonomy-jobs:Watch the machinery"]
    assert len(dropped) == len(groups) - 1, "the copied list dropped nothing"
    assert _group_headings_the_config_never_names(doc, dropped) == ["Watch the machinery"]


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


#: The names that take a node out of the gate's run: `skip`/`skipif`/`xfail` keep it
#: collected but not run, `live_vault` and `fault_injection` are what the gate's own
#: `-m "not live_vault and not fault_injection"` deselects on. `parametrize` is not
#: here because it multiplies a node rather than muting it.
_QUIETING_MARKS = frozenset({"skip", "skipif", "xfail", "live_vault", "fault_injection"})


def _decorator_names(node) -> set[str]:
    """The trailing attribute names of one decorator, for `pytest.mark.X` and `X(...)`.

    `ast.unparse` would do this in fewer lines and would match prose I have already
    written about skipping into the bargain; the point is to read what the function is
    actually decorated with, not what its docstring discusses.
    """
    out: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        while isinstance(target, ast.Attribute):
            out.add(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            out.add(target.id)
    return out


def test_neither_node_2833_closed_is_quieted_out_of_the_run():
    """#2833 clause 2: the fix is those two nodes PASSING, so this file checks itself.

    Clause 2 forbids four ways of turning a red node green without fixing anything —
    skip it, xfail it, delete it, or mark it `live_vault` so the gate's selection drops
    it — and all four are edits to THIS file, which makes them invisible to a check that
    only runs the tests. The two nodes are therefore located by name in the module's own
    syntax tree and refused both a quieting decorator and a `pytest.skip`/`pytest.xfail`
    call in their bodies. A node is here iff it exists at all, so deletion fails the
    first assertion, not the diff.

    The file's six `live_vault` nodes are conftest's, not this file's, and were
    deselected at `cffd22e6` exactly as they are now; this node says nothing about them,
    because the two it inspects are the two the item names.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    wanted = {
        "test_every_live_prose_citation_resolves_to_the_symbol_beside_it",
        "test_no_live_sentence_reports_97_as_never_run_and_the_record_names_its_command",
    }
    found: dict[str, tuple[set[str], set[str]]] = {}
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in wanted:
            calls: set[str] = set()
            for inner in ast.walk(fn):
                if isinstance(inner, ast.Call):
                    f = inner.func
                    if isinstance(f, ast.Attribute) and f.attr in {"skip", "xfail"}:
                        calls.add(f.attr)
            found[fn.name] = (_decorator_names(fn), calls)

    assert wanted <= set(found), (
        f"{sorted(wanted - set(found))} is gone from this file — #2833's clause 2 names "
        f"deletion as one of the four ways not to fix a red node, and a node nobody "
        f"collected is the quietest of them")
    for name, (decorators, calls) in sorted(found.items()):
        quieted = _QUIETING_MARKS & decorators
        assert not quieted, (
            f"{name} now carries {sorted(quieted)}: a node marked out of the gate's "
            f"selection or marked to be skipped is a green run that tested nothing")
        assert not calls, (
            f"{name} calls pytest.{sorted(calls)[0]} in its body, which is the same "
            f"refusal written as a statement instead of a decorator")


def test_the_pilot_frozenset_citation_resolves_to_the_line_it_names():
    """#1520 clause 1. §The functions cited `app/autonomy.py:710`, which is
    `RUN_SUMMARY_CAP = 300` inside the comment about tail-slicing a run summary —
    nothing to do with evidence — while the frozenset it meant was at
    `app/autonomy.py:2104` then, where §Distil had pointed since #1102. A reader who
    walked to 710 came back concluding the pilot was documented somewhere it is not.

    The expectation is `app/autonomy.py` itself, not a remembered number: the line the
    prose names must BE the definition, with the members the prose quotes, and the
    test file it calls a pin must assert that set. So the citation cannot rot the
    way its predecessor did — the code moving under the prose fails here first. That is
    not hypothetical: #2675 widened the roster to seven and wrote its per-id reasons
    above the definition, which moved it from 2818 to 2830, and this node followed the
    code instead of failing that change. #2704 moved it again, to 2898, by adding
    `_last_tool_result_section` above it — the same follow-the-code move, made again.
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
    # The roster the doc quotes, in the shape the pin writes it — the members moved
    # onto their own line when #2675 widened the pilot to seven, so the match allows
    # the wrap and still refuses any other membership.
    assert re.search(
        r"EVIDENCE_PILOT_TASK_IDS\s*==\s*frozenset\(\s*\{24, 38, 39, 40, 42, 53, 56\}\s*\)",
        pin), (
        "the doc says tests/test_worker_evidence.py pins the pilot set, and that file "
        "no longer asserts the seven-member set the doc quotes — the citation has "
        "become a promise nothing keeps")
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


def test_the_newest_jobs_are_listed_once_in_each_membership_table():
    """#1520 clause 5 (for #87–#90) and #2630 clause 1 (for #91–#97): the same
    defect twice. #87, #88, #89 (`research-agent`, one arXiv paper each) and #90
    (`corpus-shape-trend`) were created 2026-09-24 and 2026-09-25 and have run
    since — two runs each for the digests, one for #90, all green — yet neither
    table listed them, and the five tier rows still added up (20/1/3/8/1 = 33)
    against a 36-task fleet. Seven more task files were created between 2026-09-29
    and 2026-10-11 (#91–#97, `status: up_next`, 2–14 runs apiece on the live route)
    and reached 2026-10-12 with the same silence, which is what #2630 was filed for.

    `test_every_job_has_one_function_and_exactly_one_write_tier` could not see
    either episode: it compares the tables with each other, so a job missing from
    both is symmetric and passes. This names the ids and requires exactly one row
    apiece in each table, because a count in a header cell is not a membership
    check.
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


#: The tier #2630 put each of #91–#97 in, and the reason, which is the tier
#: table's own question — *may it write while nobody is watching* — not how much.
#: #91 #95 #96 #97 leave one runtime artefact each (a metrics row, a probe report,
#: an appended journal) and #95 leaves nothing at all: the drill's state file has a
#: different writer. None of the four touches the vault or the queue. #92 files a
#: backlog item when a §10 trigger moves, #93 files a backlog draft when a run wrote
#: no artefact, and #94 overwrites a dated ledger pair under `~/obsidian/autonomy/`;
#: those three act on durable state outside their own report, so they sit in the
#: unattended-writer tier beside #30 and #53 rather than with the reporters. The
#: triage of #2630 read all seven as Reports-only; three of them are not, and the
#: divergence is stated in the doc's §Watch the machinery rather than buried.
WATCH_TIER = {
    "#91": "Reports only",
    "#92": "Writes durable state unattended",
    "#93": "Writes durable state unattended",
    "#94": "Writes durable state unattended",
    "#95": "Reports only",
    "#96": "Reports only",
    "#97": "Reports only",
    # #2813, graded by the same question the tier table asks — may it write while
    # nobody is watching. Yes: its task file's `acceptance.objective_checks` prohibits
    # `Write`, `Edit`, `git` and `rm` precisely because its authority is one dated
    # report under the data root plus a backlog item when a headline count moves. The
    # report is what it prints; the filing is the write. Same reason #94 is here.
    "#99": "Writes durable state unattended",
    # #2881, the same question asked of the two 2026-10-16 arrivals. Neither may file
    # anything: each task file's `acceptance.objective_checks` names `Write` and `Edit`
    # in its `tool_not_called` list and its prohibition sentence forbids filing a
    # backlog item or opening a round, so the only durable thing either leaves is one
    # dated row in its own journal — `~/lloyd-data/eval/instrument-index/` for #100,
    # `~/.local/state/lloyd-automod/` for #101, both outside the repo, the vault and
    # the queue. That is #97's position exactly, which is why #2881's contract places
    # them here rather than beside #92–#94.
    "#100": "Reports only",
    "#101": "Reports only",
}


def test_the_newest_jobs_sit_in_the_group_and_tier_their_output_belongs_to():
    """#1520 clause 5's judgement (#87–#90), #2630 clauses 1 and 2's (#91–#97),
    pinned the way #86's placement is.

    #87 #88 #89 read one outside document — an arXiv paper — and wrote one
    knowledge note into `knowledge/`, which is what Ingest is for, and that note is
    durable state, so they sat in the unattended-writer tier beside #30 and #53
    until their task files disappeared (#2630, `DEPARTED`). #90 measures the shape
    of four nightly-written corpora and appends one dated metrics row per run, which
    is what #82 and #86 do when they record a trend: Reports only, under the Bound
    entropy group whose #78/#80 pair is this doc's report-only half of bounding
    growth.

    #91–#97 all read the machinery the other jobs run on — session traces, the
    serving engine's log window, other jobs' run records, the memory surface a
    prompt is built from, the mitigations that stop a run, the decision engine, the
    self-modification loop's own landings — which is not Measure's job of saying
    whether the fleet is working, so they form their own group. Their tiers split
    on what the run is permitted to leave behind, not on what it measures: see
    `WATCH_TIER`. The choice of group is arguable; that each job has exactly one
    group and one tier is not.
    """
    funcs, tiers = _table(FUNCTIONS_HEADER), _table(TIERS_HEADER)

    def tier_of(job: str) -> list[str]:
        return [t for t, (ids, _) in tiers.items() if job in ids]

    bound = next(v for k, v in funcs.items() if k.startswith("[Bound entropy]"))
    assert "#90" in bound[0], f"#90 is not in the Bound entropy row, which lists {bound[0]}"
    assert tier_of("#90") == ["Reports only"], (
        f"#90 is in {tier_of('#90') or 'no tier'}; outside its own metrics row it "
        f"changes nothing, the same position #82 and #86 hold")
    watch = next(v for k, v in funcs.items() if k.startswith("[Watch the machinery]"))
    for job, tier in WATCH_TIER.items():
        assert job in watch[0], (
            f"{job} is not in the Watch the machinery row, which lists {watch[0]} — "
            f"every job in `WATCH_TIER` reads the machinery another job runs on, "
            f"which is what makes it one group rather than Measure's")
        assert tier_of(job) == [tier], (
            f"{job} is in {tier_of(job) or 'no tier'}, not {tier!r} as #2630 ruled: "
            f"the tier answers what the run may write with nobody watching")


def test_every_groups_heading_row_link_and_id_list_agree():
    """One rule over every group, so a job can be added without the doc
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


# ── #1578: #76's poison step is read-only, in every channel that reaches the run ──

#: #76's prompt. The row and paragraph pinned below describe this file's steps,
#: and — as §The skill is the job says of every job — a vault edit makes them
#: false on the next run with no deploy and no gate, so the doc is read across the
#: boundary rather than around it, for the reason `board_presence.py` gives.
QUEUE_SKILL_REL = "skills/queue-health-check/SKILL.md"

#: A SQL statement that changes a `queue` row, in the shape the retired reset took
#: and the shape a reader who finds a pile of them would re-add. This is #1578's
#: own acceptance grep — `grep -nE "UPDATE queue|DELETE FROM queue"` — as a node,
#: so the grep and the test can never disagree about what counts as a write.
QUEUE_WRITE_RE = re.compile(r"\b(?:UPDATE|DELETE FROM)\s+queue\b", re.I)

#: The superseded claim as all three carriers worded it ("clears poisoned queue
#: items" in the skill's front matter, `autonomy/76-queue-health-check.md:16` and
#: `architecture/autonomy-jobs.md:797`). Narrow on purpose: it matches a statement
#: that the job *does* it, so a step is still free to say what it must not do.
POISON_CLAIM_RE = re.compile(
    r"\b(?:clears|cleared|clearing|resets?|resetting)\b[^.]{0,25}\bpoisoned\b", re.I)

#: The module that owns the transition now. Named by the skill, by the doc's
#: paragraph and by this file, so the three cannot drift into describing
#: different owners of the same write.
SWEEP_MODULE = "workers/maintenance.py"


def _queue_skill() -> str:
    """#76's prompt, read from the live vault.

    Read, never skipped when absent: `vault_root()` is env-overridable, so a skip
    on a missing vault is one `monkeypatch.setenv` away from being a green that
    certifies nothing, and the vault is on every box that runs this suite.
    """
    import board_presence

    path = board_presence.vault_root() / Path(QUEUE_SKILL_REL)
    assert path.is_file(), (
        f"{path} is unreadable, and the claims pinned here — that #76's poison step "
        f"writes nothing and names its real owner — are claims about that file")
    return path.read_text()


def _skill_step(raw: str, step: str) -> str:
    """One `## Step …` section of the skill, up to the next heading.

    Per step, not whole file, because the other steps legitimately act: Step 4
    pauses a chronically failing task through `autonomy_write_task`, which is the
    job's one remaining write and is not the queue's. A check allowed to range
    over the file would let Step 4 excuse Step 3.
    """
    m = re.search(rf"^## {re.escape(step)}[^\n]*$", raw, re.M)
    assert m, f"{QUEUE_SKILL_REL} has no `## {step}` section"
    rest = raw[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


#: §Measure's membership table, which carries a `Watches` column the other two
#: `| ID | Freq | Role |` tables in the doc do not — so #76's row is not reachable
#: with `_role_cell`, and reading the wrong table would report "no row for #76"
#: while the row that needs editing sits unexamined.
MEASURE_IDS_HEADER = "| ID | Freq | Watches | Role |"


def _measure_role_cell(job: str) -> str:
    """The Role cell of `job`'s row in §Measure's four-column table."""
    lines = _text().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == MEASURE_IDS_HEADER]
    assert starts, f"architecture/autonomy-jobs.md has no `{MEASURE_IDS_HEADER}` table"
    found = []
    for i in starts:
        for line in lines[i + 2:]:                              # header, then |---| rule
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] == job:
                assert len(cells) == 4, (
                    f"{job}'s row under a {MEASURE_IDS_HEADER!r} table has {len(cells)} "
                    f"cells, not ID/Freq/Watches/Role — the reader can no longer tell "
                    f"which column is the role")
                found.append(cells[3])
    assert len(found) == 1, (
        f"{job} has {len(found)} rows in the doc's {len(starts)} "
        f"{MEASURE_IDS_HEADER!r} tables — described twice is the double-assertion "
        f"defect #1524 pinpointed, and 0 means the row this clause edits is gone")
    return found[0]


def _task_76_front_matter() -> dict:
    """Task #76's parsed front matter, from the live vault."""
    import board_presence
    import yaml

    root = board_presence.vault_root() / "autonomy"
    files = sorted(root.glob("76-*.md"))
    assert len(files) == 1, (
        f"expected exactly one `autonomy/76-*.md` under {root}, found "
        f"{[f.name for f in files]} — with two, which one the engine loads is "
        f"undefined and this file cannot say which is being pinned")
    parsed = yaml.safe_load(files[0].read_text().split("---", 2)[1])
    assert isinstance(parsed, dict) and parsed.get("id") == 76, (
        f"{files[0]} does not parse to the task with id 76")
    return parsed


def test_the_poison_step_of_the_skill_issues_no_write_against_the_queue():
    """Clause 1. #76 stops being the one monitor with write authority over `queue`.

    What it ran was an unfiltered, unclassified write. The pool's sweep does the
    triage in code and applies two overrides no SQL statement can express: the
    item's own revive budget (`DEFAULTS["max_revives"] = 1`,
    `workers/maintenance.py:268`) and "an equivalent item is already open" (`:270`, because
    a poisoned row has lost its `dedup_key`). And `attempts=0` did not refund what
    the item claimed it refunded — the revive counter lives in `triage_json`, which
    the write never touched — it refunded the pool's *claim* budget instead, which
    the sweep deliberately leaves nearly spent (`:298`, `attempts=max(0,
    max_attempts - 1)`, one more claim rather than a fresh budget). So the write
    handed a structural payload a full retry run and hid the fact that it had
    already failed.

    Ranged over the whole file, not just the step, because that is the surface the
    item's acceptance grep is written against: a second write in some other step
    would be the same defect with a different heading.
    """
    raw = _queue_skill()
    step3 = _skill_step(raw, "Step 3")

    assert QUEUE_WRITE_RE.findall(raw) == [], (
        f"{QUEUE_WRITE_RE.findall(raw)} is back in {QUEUE_SKILL_REL}. The sweep owns "
        f"the poisoned→queued transition; a second writer refunds a budget it did "
        f"not spend and erases the sweep's evidence")
    # Positive control: the step is still a step that reads the pile, so the
    # absence above is a rewrite and not a deletion that vacuously passes.
    assert "poisoned" in step3, "Step 3 no longer looks at the poisoned pile at all"
    assert re.search(r"\bSELECT\b", step3, re.I), (
        "Step 3 no longer queries the queue, so it reports nothing and the "
        "write-free verdict above is measuring an empty section")


def test_the_poison_step_reports_the_pile_broken_out_by_source():
    """Clause 2: the report answers "whose rows are these", which the write never did.

    The retired write carried no `source` filter, and the stopped pile is not one
    fleet's: on the live queue today (2026-09-27) `SELECT state, source, count(*)
    FROM queue WHERE state IN ('poisoned','quarantined') GROUP BY state, source`
    returns 9 rows, 8 of them `autoresearch` and 1 `scheduled-task`. A monitor of
    the autonomy fleet that reset the table was therefore stopping eight rows owned
    by something else. The read breaks both states out by source, and — the half
    that a `WHERE source=` would defeat — it is not narrowed to one source, because
    a pile already filtered to your own fleet cannot show you whose rows you were
    about to touch.
    """
    step3 = _skill_step(_queue_skill(), "Step 3")
    query = " ".join(step3.split())          # the SQL is wrapped in the skill

    for state in ("'poisoned'", "'quarantined'"):
        assert state in query, (
            f"Step 3 no longer reports {state} rows. Quarantined is the sweep's "
            f"verdict on a structural cause, and a pile report that omits it shows "
            f"only the work that has not been triaged yet")
    assert re.search(r"GROUP\s+BY[^;]*source", query, re.I), (
        "the pile is counted without breaking it out by source — the count that "
        "made the old cross-fleet write visible is the number this step exists to "
        "print")
    assert "scheduled-task" not in step3, (
        "Step 3 filters the pile to one source. Breaking out a pile you have "
        "already narrowed to your own fleet is not a cross-fleet report")


def test_the_poison_step_names_the_sweep_and_where_its_decision_is_recorded():
    """Clause 3: name the owner and its paper trail, or the next reader re-adds it.

    The skill had zero mentions of the sweep, quarantine or `maintenance` before
    #1578 (`grep -in "sweep\\|quarantin\\|maintenance"` over it exited 1), which is
    why the write survived every prior edit of this file: a reader who finds a
    pile and no owner writes it away. So the step must name `workers/maintenance.py`
    as the
    triager, must say where the decision is written — a `sweep_*.md` under
    `~/lloyd-data/autonomy-runs/queue-maintenance/`, whose escalation lines carry
    one `(source, signature)` pair per recurring cause once it hits
    `DEFAULTS["repeat_threshold"]` (3) — and must say the pile is read-only here.

    The tally is the reason this is not cosmetic: `sweep()` selects only
    `state='poisoned'`, so a row moved to `queued` by hand was never scanned again
    and its `(source, signature)` count never incremented — the one alert that
    surfaces a cause still firing after a revive was erased by the write.
    """
    step3 = _flat(_skill_step(_queue_skill(), "Step 3")).lower()

    assert SWEEP_MODULE in step3, (
        f"Step 3 no longer names {SWEEP_MODULE} as the thing that triages the pile")
    assert "sweep" in step3, "Step 3 no longer names the sweep"
    assert "autonomy-runs/queue-maintenance" in step3, (
        "Step 3 lost the path where the sweep's decision is recorded, so a reader "
        "who disagrees with a quarantine has nowhere to check it")
    assert "escalation" in step3 and "signature" in step3 and "repeat_threshold" in step3, (
        "Step 3 stopped pointing at the per-(source, signature) escalation lines "
        "that fire at repeat_threshold — the recurring-cause alert a hand write "
        "silences")
    assert "quarantin" in step3, (
        "Step 3 no longer says a structural cause is quarantined, which is the "
        "decision it must not overwrite")
    assert re.search(r"read[- ]only|do not write|never write", step3), (
        "Step 3 no longer states that it writes nothing here — the prohibition is "
        "the part a reader needs, not the explanation")


def test_the_prompt_task_76_is_served_orders_no_reset_of_the_pile():
    """Clause 4, across the seam that carries an instruction to the model.

    `_build_task_prompt` renders the skill plus the task's front-matter
    `description:` and nothing else — the file's own body says so above its
    Activity Log — and that description read "…clears poisoned queue items after
    recording why…". So the same order reached every run twice, once from a step
    and once from a field no step edit touches, and grading the skill alone would
    leave the write-commanding sentence in the one channel guaranteed to arrive.
    Hence the rendered prompt, not the file.
    """
    from app.autonomy import _build_task_prompt

    task = _task_76_front_matter()
    desc = str(task.get("description", "")).strip()
    assert desc, "task #76's `description:` is empty"
    assert POISON_CLAIM_RE.search(desc) is None, (
        f"#76's `description:` claims the pile again: {desc[:90]!r}…")
    assert "sweep" in desc.lower(), (
        "the description dropped the order without saying what owns the pile now, "
        "which leaves a run looking for the step that does it")

    prompt = _build_task_prompt(task, _queue_skill())
    assert "Task description:" in prompt, (
        "_build_task_prompt stopped rendering the description, so this node and the "
        "clause it pins are measuring a seam that has moved")
    assert QUEUE_WRITE_RE.findall(prompt) == [], (
        f"{QUEUE_WRITE_RE.findall(prompt)} survives in the prompt a #76 run is "
        f"served, from a carrier outside Step 3. Find it and relabel it — Step 3 is "
        f"not the only door")
    assert POISON_CLAIM_RE.search(prompt) is None, (
        "the prompt a #76 run is served still states that the job clears the pile")


def test_the_measure_row_and_the_measure_paragraph_both_give_the_pile_to_the_sweep():
    """Clause 5, in the file a reader who never opens the vault actually reads.

    §Measure asserted the superseded behaviour twice — the ID/Freq/Role row at
    `architecture/autonomy-jobs.md:762` ("clears poisoned items and pauses failing
    tasks") and the paragraph at `:797` ("It clears poisoned queue items after
    recording why") — which is the #1524 shape exactly: one claim, two copies, and
    the item named only "the doc's #76 paragraph", singular. Both move together
    here, and the paragraph must name the module the skill names, so the row, the
    prose and the prompt cannot describe three different owners of one write.

    The row keeps the pause, because that half is still true: pausing a chronically
    failing task is the job's own act, and the tier table still lists #76 as the one
    job that acts on the fleet itself.
    """
    role = _flat(_measure_role_cell("#76"))
    measure = _group_section("Measure")

    for where, text in (("the Measure table row", role),
                        ("the §Measure paragraph", measure)):
        assert POISON_CLAIM_RE.search(text) is None, (
            f"{where} says #76 clears the poisoned pile again — that behaviour moved "
            f"into {SWEEP_MODULE}")
        assert "sweep" in text.lower(), (
            f"{where} no longer says the deterministic queue sweep triages the pile, "
            f"so the doc has an owner-less write again")
    assert SWEEP_MODULE in measure, (
        f"the §Measure paragraph never names {SWEEP_MODULE}, the module the skill "
        f"names, and the two carriers of this claim can drift apart")
    assert "pauses" in role, (
        "the row lost the pause with the write, and pausing a chronically failing "
        "task is still what #76 does — it is in the write tier for it")


#: What the two prose carriers of the corpus-shape row-key claim said while the
#: writer was keyed to the second, and had to stop saying when #1576 keyed it to the
#: UTC date. Matched case-insensitively against flattened text, because the module
#: docstring states its headline in capitals.
CORPUS_SHAPE_ROW_PER_RUN_CLAIMS = (
    "row per run",
    "appended to",
    "diffed against the previous run",
    "diffs it against the last",
    "append-only",
    "each run opens a new",
)

#: The two properties that must be present wherever the claim was. The second is the
#: one that has to SURVIVE the rewrite: it is why the writer was append-only in the
#: first place (#543's `graph-baseline.json` — a baseline the run it judges rewrites).
CORPUS_SHAPE_ROW_KEY_CLAIM = re.compile(r"one (?:dated )?row per UTC (?:date|day)", re.I)
CORPUS_SHAPE_EARLIER_BASE_CLAIM = re.compile(r"newest row from an earlier (?:day|date)", re.I)
CORPUS_SHAPE_READ_BEFORE_WRITE_CLAIM = re.compile(r"read before .*?wrote itself", re.I)

SCRIPT_DIR = ROOT / "scripts" / "maintenance"
TASK_90_FILE = Path.home() / "obsidian" / "autonomy" / "90-corpus-shape-trend.md"


def _module_docstring(path: Path) -> str:
    """A script's module docstring, wrapped sentences collapsed like §sections."""
    import ast

    return _flat(ast.get_docstring(ast.parse(path.read_text())) or "")


def test_the_corpus_shape_row_key_is_a_utc_date_in_both_prose_carriers():
    """#1576 clause 4. §Bound entropy and `corpus_shape.py`'s own docstring both
    said a row is written per run, and both had to move.

    The doc said #90 "writes one dated row per run ..., diffs it against the last"
    and its table row said the series is "appended to ... and diffed against the
    previous run"; the docstring's headline was "THE TREND FILE IS APPEND-ONLY ...
    Each run opens a NEW `corpus-shape-<UTC stamp>.json`". All three were true of the
    writer keyed to the second — three rows landed on 2026-09-24 within 39 s of each
    other — and are false of the writer keyed to the UTC date of the resolved
    `--now`, which writes nothing on a same-day retry. The paragraph carries the
    caveat the guard does NOT buy: one row per day bounds duplicates, not gaps.

    Checked in both carriers, not "somewhere in the tree": a review reads the doc and
    its §Bound entropy matcher is what refuted the false copies before (#1524), while
    the script's docstring is what an implementer reads before touching the writer.
    """
    bound = _group_section("Bound entropy")
    docstring = _module_docstring(SCRIPT_DIR / "corpus_shape.py")

    for where, text in (("§Bound entropy", bound),
                        ("corpus_shape.py's module docstring", docstring)):
        for stale in CORPUS_SHAPE_ROW_PER_RUN_CLAIMS:
            assert stale not in text.lower(), (
                f"{where} still says {stale!r}: a corpus-shape row is keyed to the "
                f"UTC date of the run's resolved --now, so a same-day retry writes "
                f"nothing (#1576)")
        for claim, what in ((CORPUS_SHAPE_ROW_KEY_CLAIM, "one row per UTC date"),
                            (CORPUS_SHAPE_EARLIER_BASE_CLAIM,
                             "the diff base is the newest row from an earlier day"),
                            (CORPUS_SHAPE_READ_BEFORE_WRITE_CLAIM,
                             "the base row is read before the run's own row exists, "
                             "so no run gates on a file it wrote itself")):
            assert claim.search(text), (
                f"{where} no longer says {what}; that is the property #1576 was for, "
                f"and it is the one the old prose was written to protect")


#: The sentence #761 shipped with its thresholds and #2200 retired: it explained the
#: bounds as a guess awaiting a week of series, and stayed true-looking in three
#: carriers long after the series existed. Checked case-insensitively against
#: flattened text, so the docstring's capitalised headline is covered too.
BOUND_PROVISIONAL_CLAIM = "provisional"

#: What each carrier has to say in its place: the bounds come off day-to-day moves,
#: and over a sample a reader can go and re-measure.
BOUND_DERIVATION_CLAIM = re.compile(r"day-over-day", re.I)
BOUND_SAMPLE_CLAIM = re.compile(r"8 clean UTC dates", re.I)
BOUND_SAMPLE_DATES = re.compile(r"2026-09-27.{0,80}2026-10-04", re.S)


def test_the_bound_provenance_replaces_provisional_in_both_prose_carriers():
    """#2200 clause 5: §Bound entropy and `corpus_shape.py`'s docstring both stop
    calling the corpus-shape threshold provisional, and both say what it became.

    The thresholds sat at abs 0.02 on `duplicate_rate` while the daily corpus moved
    0.1117 in one night — 6 of the 8 nights of the first clean series printed a MOVED
    line about numbers nobody would call a finding — and the sentence excusing them
    ("provisional … real ones need a week of series", a human clause on #761) outlived
    the week by a month because no test read it. So the check is on both halves: the
    excuse is gone, and the measurement it was standing in for is stated — day-over-day
    deltas over the 8 clean UTC dates 2026-09-27 to 2026-10-04.

    In both carriers, not "somewhere in the tree": a review reads §Bound entropy, and
    the docstring is what whoever edits the constant reads first. The third carrier of
    the same claim is the task file `~/obsidian/autonomy/90-corpus-shape-trend.md`,
    whose front-matter description is what `_build_task_prompt` hands the worker; the
    vault is not a tree this round controls, so that half moves by the vault route as
    its own sha and is pinned by a `@live_vault` node instead — the split
    `test_the_corpus_shape_task_description_says_at_most_one_row_per_day` already makes.
    """
    carriers = (("§Bound entropy", _group_section("Bound entropy")),
                ("corpus_shape.py's module docstring",
                 _module_docstring(SCRIPT_DIR / "corpus_shape.py")))
    for where, text in carriers:
        assert BOUND_PROVISIONAL_CLAIM not in text.lower(), (
            f"{where} still calls the threshold provisional, the sentence that said "
            f"nobody had measured the bounds; #2200 measured them")
        assert BOUND_DERIVATION_CLAIM.search(text), (
            f"{where} no longer says the bounds come from day-over-day deltas, so a "
            f"reader of the constant cannot tell what it was derived from")
        assert BOUND_SAMPLE_CLAIM.search(text), (
            f"{where} no longer says how many dates the bounds were derived from")
        assert BOUND_SAMPLE_DATES.search(text), (
            f"{where} does not name the sampled dates 2026-09-27..2026-10-04, which is "
            f"the sample a later reader re-measures")


@live_vault
def test_the_corpus_shape_task_description_says_at_most_one_row_per_day():
    """#1576 clause 5, against the live task file. `@live_vault` because the vault is
    not a tree this round controls (`pytest.ini:10-13`); run it from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_autonomy_jobs_doc_claims.py::test_the_corpus_shape_task_description_says_at_most_one_row_per_day -q

    — the vault half of #1576 landed as its own sha before this diff, so the node is
    green against `~/obsidian` and the gate never executes it.

    The description is what a runner is handed, so this parses it through the real
    loader (`app.autonomy._parse_task_file`, the call the scheduler makes) rather than
    reading raw front matter: a description the parser cannot reach is unpinnable.
    """
    from app import autonomy

    task = autonomy._parse_task_file(TASK_90_FILE)
    assert task, f"{TASK_90_FILE} does not parse as an autonomy task"
    desc = _flat(task.get("description") or "")
    body = _flat(TASK_90_FILE.read_text())
    assert desc, f"{TASK_90_FILE.name} has no description to check"

    for stale in ("appends one new dated", "appends a new", "diffs it against the previous one"):
        assert stale not in desc.lower(), (
            f"the task description still instructs the runner that every run {stale!r}")
    assert "at most one row exists per utc day" in desc.lower(), (
        "the description never states the one-row-per-day key, so a runner reading "
        "only the description would still expect a fresh row each run")
    assert CORPUS_SHAPE_EARLIER_BASE_CLAIM.search(desc), (
        "the description does not say a same-day run diffs against an earlier day's "
        "row, which is the half that stops a retry becoming tomorrow's base")
    assert "writes no row" in body.lower(), (
        "the task body no longer states that a retry inside a day writes nothing")
    assert "not gap handling" in body.lower(), (
        "the task body reads as 'one row per day therefore no gaps', which the guard "
        "does not deliver")


@live_vault
def test_the_corpus_shape_task_file_says_the_bounds_are_measured_not_provisional():
    """#2373 clause 3, against the live task file — the third carrier that
    `test_the_bound_provenance_replaces_provisional_in_both_prose_carriers` names in
    its own docstring but cannot read, because `~/obsidian/autonomy/90-corpus-shape-trend.md`
    is not in the tree that node runs against.

    Same claim, same retirement, different tree. The front-matter `description` is
    what `_build_task_prompt` (`app/autonomy.py:2487`, appended verbatim at
    `:2504-2506`) hands every #90 run, so the worker was told its bounds were
    "provisional" three days after #2200 measured them, while the body excused them
    "until a week of series exists (a human clause on #761)". The vault is not a tree
    this round controls (`pytest.ini:14-19`), so that half landed as its own sha
    (`b25fef3f`, 2026-10-07) and this node is the reporting copy the gate never
    executes. Run it by hand from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_autonomy_jobs_doc_claims.py::test_the_corpus_shape_task_file_says_the_bounds_are_measured_not_provisional -q

    Both halves are graded, as in the two repo carriers: the excuse gone AND the
    measurement it stood in for named — the largest day-over-day delta per metric over
    the 8 clean UTC dates 2026-09-27 to 2026-10-04 — since a file that only drops the
    word leaves the next reader no way to re-derive a bound. Checked on the PARSED
    description through `app.autonomy._parse_task_file`, the call the scheduler makes:
    a claim the loader cannot reach never reaches the worker, and the raw body is
    checked beside it because that is where the #761 clause was standing.
    """
    from app import autonomy

    task = autonomy._parse_task_file(TASK_90_FILE)
    assert task, f"{TASK_90_FILE} does not parse as an autonomy task"
    desc = _flat(task.get("description") or "")
    body = _flat(TASK_90_FILE.read_text())
    assert desc, f"{TASK_90_FILE.name} has no description to check"

    for where, text in (("the parsed description", desc),
                        ("the raw task file", body)):
        assert BOUND_PROVISIONAL_CLAIM not in text.lower(), (
            f"{where} still calls the corpus-shape bounds provisional; #2200 measured "
            f"them, and the description is the text every #90 run is handed")
        assert BOUND_DERIVATION_CLAIM.search(text), (
            f"{where} no longer says the bounds come from day-over-day deltas, so a "
            f"runner cannot tell what the bound it reports was derived from")
        assert BOUND_SAMPLE_CLAIM.search(text), (
            f"{where} no longer says how many dates the bounds were derived from")
        assert BOUND_SAMPLE_DATES.search(text), (
            f"{where} does not name the sampled dates 2026-09-27..2026-10-04, the "
            f"sample a later reader re-measures")


#: The four docs this one absorbed on 2026-09-11, and the commit that last left a
#: tracked copy of each in `architecture/.archive/` before untracking it. Both are
#: fixed history, so the pair is typed here rather than derived: a node that read
#: the roster out of the doc it is grading would pass on a doc that named none.
FOLDED_DOCS = ("nightly-reflection", "nightly-skills-management",
               "morning-briefing", "groundskeeper")
ARCHIVE_RECOVERY_SHA = "f80c9d00"


def test_the_folded_away_docs_are_recorded_as_gone_from_the_tree():
    """#2329 clause 1. The paragraph under `## Related` used to send a reader to
    `architecture/.archive/` for the four absorbed docs, and that directory is not
    in the tree and not in git (`git ls-files architecture/.archive/` is empty;
    `f80c9d00` untracked the five copies it ever held), so following the pointer
    found nothing at all. It now says the text is gone from the tree, recoverable
    only from history, and names the command that gets it back.

    The named command is resolved, not admired: each of the four paths must exist
    at `f80c9d00^`, because a doc that trades one dead pointer for another has not
    been fixed, and the commit whose parent it names is the half a reader cannot
    check by eye while reading.
    """
    text = _text()
    assert "retired to `architecture/.archive/`" not in text, (
        "the paragraph points the reader at architecture/.archive/ again, which "
        "is neither on disk nor tracked in git")
    for slug in FOLDED_DOCS:
        assert f"`{slug}`" in text, f"{slug} is no longer named as a folded doc"
    flat = _flat(text)
    assert "taken out of the tree" in flat, (
        "the paragraph no longer says plainly that the four are gone from the tree")
    assert "recoverable only from history" in flat, (
        "the paragraph does not say the only copy left is in git history")
    recovery = f"git show {ARCHIVE_RECOVERY_SHA}^:architecture/.archive/<slug>.md"
    assert recovery in flat, (
        f"the paragraph names no recovery command of the shape "
        f"{recovery!r} — an absent directory with no way back is not a fix")
    for slug in FOLDED_DOCS:
        probe = subprocess.run(
            ["git", "cat-file", "-e",
             f"{ARCHIVE_RECOVERY_SHA}^:architecture/.archive/{slug}.md"],
            cwd=ROOT, capture_output=True)
        assert probe.returncode == 0, (
            f"the command the doc names does not resolve for {slug}: "
            f"git cat-file -e {ARCHIVE_RECOVERY_SHA}^:architecture/.archive/"
            f"{slug}.md exits {probe.returncode}")


def test_the_numbering_note_survives_the_archive_pointer_being_fixed():
    """#2329 clause 1's second half, pinned as its own node because the two edits
    sit in one file and the failure mode is collision, not correctness: rewriting
    the retirement paragraph must not disturb the id-range sentence, which is what
    `test_no_second_fleet_bound_appears_anywhere_in_architecture` depends on for its
    floor wording, its numbering-note anchor and its dated-range requirement.

    The bound itself is no longer pinned as a literal. #2329 wrote the probe around
    `24 through 90`, but the doc's own rule is that the named id is a floor that
    advances and carries the date it was newest — which is what #1102 was filed for
    when the bound stopped at 85 with 86 already running nightly, and what #2630
    repeats for #91–#97. A literal `90` therefore punished the correction the doc
    asks for, and the collision this node exists to catch could not be told from a
    bound being raised. What is pinned now is the shape of the whole sentence and a
    bound that cannot sit below the highest id the document itself lists as a
    member: `24 through <N> or beyond, with gaps where tasks were retired
    (<a date>)`, with N at least the largest member id. Below that, the note names a
    ceiling while the tables two rows down name jobs above it — the #1102 state, in
    the one sentence whose job is not to be that.
    """
    flat = _flat(_text())
    m = re.search(
        r"ids run 24 through (\d+) or beyond, with gaps where tasks were retired "
        r"\((\d{4}-\d{2}-\d{2})\)", flat)
    assert m, (
        "the numbering note moved or was reworded out of the shape "
        "`ids run 24 through <N> or beyond, with gaps where tasks were retired "
        "(<date>)`, and the node that reads the range as a floor rather than a "
        "ceiling loses its anchor")
    named = int(m.group(1))
    listed = max(int(i.lstrip("#")) for ids, _ in _table(FUNCTIONS_HEADER).values()
                 for i in ids)
    assert named >= listed, (
        f"the note bounds the fleet at {named} while the functions table lists "
        f"#{listed} as a member — a bound below the fleet's own newest member is the "
        f"#1102 defect, and the date beside it ({m.group(2)}) is what makes it "
        f"checkable rather than merely stale")


# ── #2482 — the digester's filing rule lives in its skill, graded here ───────

#: Task #53's only instruction surface. `_build_task_prompt` renders the
#: `skill_name` SKILL.md plus the task file's front-matter `description` and nothing
#: else, and `~/obsidian/autonomy/53-documentation-digester.md` says in its own header
#: that its body is documentation, "not an instruction channel". So a rule meant to
#: reach that job's reasoning has exactly one place it can live, and a rule with no
#: node against it is a sentence the next model can drop without anything noticing.
#: `live_vault` for the usual reason: a nightly skills pass rewrites this file, and
#: the gate must not fail this round for that writer's edit.
DIGESTER_SKILL = Path.home() / "obsidian" / "skills" / "documentation-digester" / "SKILL.md"
PROBES_HEADER = "## Probes That Read As Verdicts But Are Not"


def _probes_section() -> str:
    """§Probes That Read As Verdicts But Are Not, raw, to the next heading.

    Raw and not `_flat`, because the bullet is the unit of clause 1 — a flattened
    section has no `\\n- ` left to split on, and a test that split it anyway would
    find one "bullet" containing everything and pass on any wording. Callers flatten
    each bullet themselves before matching a wrapped sentence inside it.
    """
    text = DIGESTER_SKILL.read_text(encoding="utf-8")
    assert PROBES_HEADER in text, (
        f"{DIGESTER_SKILL} no longer carries {PROBES_HEADER!r}, the section every "
        "probe-lie rule lives in — #2482's filing rule has no home")
    return text.split(PROBES_HEADER, 1)[1].split("\n## ", 1)[0]


@live_vault
def test_the_digester_skill_demands_a_read_back_artifact_before_it_files():
    """#2482 clause 1, on the live skill.

    The run that motivated it (`autonomy-runs/53/run_53_20261009_113528.md`) filed
    backlog #2478 at `priority: high` quoting an audit database, its `device_events`
    table and an `is_true_positive` column, and the reading "1804 rows, 27 positives".
    `ls -la ~/lloyd-data/wakeword-audit/` returns `No such file or directory` and
    `git grep` over the checkout returns 0 files for each of those names, so the item
    asserted artifacts that have never existed here. The rule that was missing is the
    one this node reads back: name the artifact, or carry the command that produced
    it, or do not file.
    """
    section = _probes_section()
    assert "Five ways this pass's own probes lie" in _flat(section), (
        "the intro no longer counts five probes, so the new rule is not the fifth one")
    bullets = [_flat(b) for b in section.split("\n- ")[1:] if b.strip()]
    assert len(bullets) >= 5, (
        f"the section lists {len(bullets)} probe-lies after its intro; clause 1 asks "
        "for a fifth, and four is the pre-#2482 state")
    assert "command line that produced it" in bullets[4], (
        "the fifth bullet is not the artifact rule: a finding naming a path, DB, "
        "table/column or task id must carry the command line and output that produced "
        f"it. Fifth bullet read: {bullets[4][:120]!r}")
    artifact_rule = bullets[4]
    for token in ("`ls -la <dir>`", "`sqlite3 <db> .tables`", "autonomy_get_task",
                  "observation, not a filing", "`backlog_write_task`", "`date -u -d @"):
        assert token in artifact_rule, (
            f"the artifact rule dropped {token!r}: a cited artifact must show the "
            "reading that produced it, an unbacked claim is logged as an observation "
            "rather than filed, and a derived literal prints its own command")


@live_vault
def test_a_task_id_cited_by_the_digester_must_resolve_before_it_is_called_a_gate():
    """#2482 clause 2, on the live skill.

    #2478 cross-checked itself against "task #71" as a live wake-word gate. `#71` is
    `~/obsidian/autonomy/_archived/71-skill-authoring-error-pattern-mining-batch.md`
    — `archived_at: '2026-09-03'`, subject error-pattern skill authoring — and
    `autonomy_get_task(id=71)` answers `Task #71 not found`. Resolving an id is one
    tool call, and that call refuted the finding.
    """
    section = _flat(_probes_section())
    assert "Resolve a cited task id before calling it a live gate" in section, (
        "the task-id resolution rule is gone, so a retired job can be cited as a "
        "live gate again")
    rule = section.split("Resolve a cited task id before calling it a live gate", 1)[1]
    for token in ("`autonomy_get_task(id=N)`",
                  "`ls ~/obsidian/autonomy/ ~/obsidian/autonomy/_archived/`",
                  "`archived_at:`", "retires the claim"):
        assert token in rule, f"the task-id rule lost {token!r}"
    assert "0 files" in rule, (
        "the standing control — no wake-word autonomy task exists, live or archived — "
        "is what makes this rule checkable rather than advice")


# ── #2630 — the seven watch-the-machinery jobs get entries; the digests lose rows ─
#
# The bargain this file's own docstring states is that schedules live in the task
# files and membership is complete *here*, and #2630 is what happens when the second
# half is false in both directions at once: seven live task files (#91–#97,
# `status: up_next`, every one running on 2026-10-12) named nowhere, and three
# departed ones (#87–#89) still named as unattended writers of `knowledge/` with no
# task file behind them. The four tests below pin the four states #2630 had to
# produce: an entry per job in its own group, the §intro claim that the seven are
# undocumented, and the trio out of the membership tables. None of them reads
# `~/obsidian/autonomy/` — that reconciler is #2631, and a check needing it would
# carry `live_vault` and be deselected at the gate (`pytest.ini:10-13`), so these
# ids are typed as an external fact, exactly as `NEW_JOBS` types #90's.

WATCH_HEADING = "## Watch the machinery:"


def _watch_slice() -> str:
    """§Watch the machinery as written: the heading text through the next `## `.

    Raw, not flattened, because its per-job table is read row by row; apply `_flat`
    to the result for the prose. The assertion is the usual one in this file — a
    section that stopped existing must fail loudly, not read as a doc with nothing
    left to check.
    """
    text = _text()
    assert WATCH_HEADING in text, (
        "architecture/autonomy-jobs.md has no §Watch the machinery to hold the "
        "watchers, so their live task files are back to having no entry — the "
        "#2630 defect, #2813's and #2881's all at once")
    return text.split(WATCH_HEADING, 1)[1].split("\n## ", 1)[0]


def _watch_row(job: str) -> str:
    """One job's row in §Watch the machinery's table, or a failure naming the ids the
    table actually has. A missing row must never read as a row with nothing in it."""
    slice_ = _watch_slice()
    have = re.findall(r"^\| #(\d+) \|", slice_, re.M)
    row = next((ln for ln in slice_.splitlines()
                if ln.strip().startswith(f"| {job} |")), "")
    assert row, (
        f"{job} has no row in §Watch the machinery's table, which has rows for "
        f"{sorted(int(h) for h in have)} — #2630 clause 3 is one entry per job, in "
        f"the shape of the #86 entry in §Measure")
    return row


def test_the_watch_group_table_holds_one_row_per_live_job_and_no_others():
    """#2630 clause 3's membership half, over the group as it now stands: every job
    in `WATCH_JOBS` has a row, and the group holds nothing that is not one of them.

    `test_each_watch_entry_states_its_schedule_its_reads_its_writes_and_a_prohibition`
    checks the rows' contents; this checks the row *set*, because a doc that gained
    nine rows or kept the departed trio in here would satisfy the content test
    row-by-row and still mis-state the group.
    """
    ids = [int(n) for n in re.findall(r"^\| #(\d+) \|", _watch_slice(), re.M)]
    assert sorted(ids) == sorted(WATCH_JOBS), (
        f"§Watch the machinery has per-job rows for {sorted(ids)}, not exactly the "
        f"{len(WATCH_JOBS)} live task files the entry-content node below is "
        f"parametrised over — {sorted(WATCH_JOBS)}. One list for both, so a job "
        f"admitted to the group moves the rows and the probes together (#2813 "
        f"admitted #99 that way) instead of leaving one of them behind")
    assert len(ids) == len(set(ids)), f"a job has two entries: {sorted(ids)}"


@pytest.mark.parametrize("job", sorted(WATCH_JOBS))
def test_each_watch_entry_states_its_schedule_its_reads_its_writes_and_a_prohibition(
        job: int):
    """#2630 clause 3, in the shape #1102 left behind for #86
    (`test_the_86_entry_states_its_frequency_its_subject_and_its_two_prohibitions`):
    schedule in its own cell, then subject, then what it writes, then what it must
    not do — four probes, each matching only in the entry's own row.

    The schedule is checked against the `frequency:` the task file declares rather
    than against the run record's cadence for the reason #1102 states: #86 said
    "every night" while a later hand typed `frequency: weekly`, and the cell is the
    only place the two can be caught disagreeing. The other three probes are the
    artefact and the bound the task file itself names, so a row that reads as
    self-evidently read-only but actually files a backlog draft fails here.
    """
    freq, reads, writes, prohibition = WATCH_JOBS[job]
    row = _watch_row(f"#{job}")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[1] == freq, (
        f"#{job}'s row reads {cells[1]!r} where its task file's `frequency:` says "
        f"{freq!r} — the same drift #1102's clause 2 was written for")
    for want, why in ((reads, "what it reads"), (writes, "what it writes"),
                      (prohibition, "what it must not do")):
        assert want in row, (
            f"#{job}'s entry never states {why} in terms (`{want}` is what "
            f"`~/obsidian/autonomy/{job}-*.md` names) — a schedule with a subject and "
            f"no reads or writes is what #2630 was filed for")


def test_the_doc_no_longer_states_that_the_seven_have_no_entry():
    """#2630 clause 4. The §intro said, in 2026-10-12's own words, that the seven
    task files exist "with no entry anywhere in this document". Once the entries
    exist that sentence is false, and a false disclaimer is worse than none: a
    reader who met #93 in §Watch the machinery and then read the intro would conclude
    the section was describing a job that does not exist.

    The probe is the disclaimer's own phrase, and the control beside it is the group
    name — proving the grep resolved against a document that does name the seven, so
    the absence below cannot be the shape of a failed search.
    """
    live = _flat(_live_prose())
    for stale in ("no entry anywhere in this document",
                  "have no entry here at all",
                  "Seven task files exist"):
        assert stale not in live, (
            f"live prose still says the seven are undocumented ({stale!r}); the "
            f"§Review log may keep the claim, being dated, and does")
    assert "Watch the machinery" in live, (
        "positive control failed: no live sentence names the group holding #91–#97, "
        "so the three absences above would prove nothing")


def test_the_departed_paper_digests_hold_no_membership_row():
    """#2630 clause 5, the defect that is the seven's mirror image. #87 #88 #89 are
    named in the functions row, the write-authority row and the §Ingest heading as
    live members that "write into `knowledge/` unattended", while no `87-*.md`,
    `88-*.md` or `89-*.md` exists in `~/obsidian/autonomy/` or its `_archived/`
    (verified 2026-10-12) — so the tables named jobs that cannot run, in the one
    document whose bargain is membership.

    #90 is the control: created in the same fortnight, still a live task file, still
    listed. If the control fails, the tables have stopped parsing and every absence
    asserted here is vacuous — which is the failure mode this file's own docstring
    calls out for a 0-hit grep.
    """
    text = _text()
    funcs, tiers = _table(FUNCTIONS_HEADER), _table(TIERS_HEADER)
    grouped = [i for ids, _ in funcs.values() for i in ids]
    tiered = [i for ids, _ in tiers.values() for i in ids]
    assert "#90" in grouped and "#90" in tiered, (
        "positive control: #90 is in neither table, which means both stopped parsing "
        "— the trio's absence below would then prove nothing")
    for job in DEPARTED:
        assert job not in grouped, (
            f"{job} is still a member of a function row; its task file is gone from "
            f"`~/obsidian/autonomy/` and from `_archived/`")
        assert job not in tiered, (
            f"{job} is still granted write authority in a doc that claims its "
            f"membership is complete")
        assert not re.search(rf"^\| {job} \|", text, re.M), (
            f"{job} still has a per-job row claiming a cadence for a job that has no "
            f"task file to carry it")
    ingest = next(v for k, v in funcs.items() if k.startswith("[Ingest]"))
    assert ingest[0] == ["#68", "#30", "#53"], (
        f"the Ingest row reads {ingest[0]}; the trio's ids have to leave the row, not "
        f"just its count")
    heading = re.search(r"^## Ingest: (.+)$", text, re.M)
    assert heading and _ids(heading.group(1)) == ["#68", "#30", "#53"], (
        f"the §Ingest heading still decorates itself with {_ids(heading.group(1)) if heading else 'no heading'}; "
        f"`test_every_groups_heading_row_link_and_id_list_agree` would catch the "
        f"mismatch too, but only once the row is wrong as well")


def test_the_seven_add_no_dependency_edge_and_the_fleet_arithmetic_moves_once():
    """#2630's bound on itself: listing seven jobs must not invent wiring.

    None of the seven declares `depends_on` — `grep -c depends_on
    ~/obsidian/autonomy/9[1-7]-*.md` reads 0 for all seven, and
    `GET /api/autonomy/tasks` on 2026-10-12 parsed 39 tasks with exactly ten
    declaring a parent (39, 40, 42, 47, 48, 51, 57, 58, 74, 83) — so the ten arrows
    stay, and the one arithmetic they sit inside moves: §The wiring's
    "N of the M tasks declare no dependency" and §Distil's "Ten of the fleet's M
    jobs" both take the route's 39, and 39 − 29 = 10 is still the diagram's arrow
    count.
    """
    edges = _diagram_edges()
    assert edges == TEN_EDGES, (
        f"the wiring diagram reads {sorted(edges)}, not the ten verified edges "
        f"{sorted(TEN_EDGES)} — none of #91–#97 declares a parent, so none of them "
        f"belongs in this picture")
    grouped = {f"#{job}" for job in WATCH_JOBS}
    assert not (grouped & {e for pair in edges for e in pair}), (
        f"the diagram drew an edge for one of {sorted(grouped)}; every file in "
        f"`WATCH_JOBS` reads `depends_on` 0 — #100 and #101 verified on 2026-10-17 — "
        f"so none of them belongs in this picture")
    funcs = _functions_section()
    m = re.search(r"\b(\w+) of the (\w+) tasks\b", _flat(funcs))
    assert m, "§The wiring no longer counts the jobs that declare no dependency"
    without, fleet = _num(m.group(1)), _num(m.group(2))
    assert fleet == 39, (
        f"the doc counts a fleet of {fleet} while `GET /api/autonomy/tasks` parsed 39 "
        f"tasks on 2026-10-12, #91–#97 included — the denominator is the thing "
        f"#1520 clause 3 pinned two sections together over")
    assert fleet - without == len(edges), (
        f"{fleet} − {without} = {fleet - without} but the diagram draws {len(edges)} "
        f"edges; a job joining §Watch the machinery adds no parent, so the two "
        f"numbers move together or the sentence and the picture part company")


#: A claim that some id is the fleet's newest, as the doc writes it in §The
#: functions' numbering note and in §The wiring's parenthetical. `is` and not `was`:
#: the §Review log and the wiring note both keep superseded newest-ids in the past
#: tense on purpose, and those are records rather than current claims.
_NEWEST_CLAIM = re.compile(r"#(\d+)[^.]*\bis the newest\b")


def test_the_2881_pair_is_listed_once_in_each_table_and_the_watch_row_agrees():
    """#2881 clause 1, gradable without the vault.

    `test_every_live_task_file_is_named_once_in_each_membership_table` is the real
    reconciler and carries `@live_vault`, so the gate never runs it
    (`pytest.ini:10-13`) and #100/#101 were invisible to every node here: symmetric
    omission passes `test_every_job_has_one_function_and_exactly_one_write_tier`,
    which is how #2630, #2813 and #2881 all happened. This pins the two ids the same
    file can check — exactly one row apiece in each table, and the §Watch the
    machinery row agreeing with `WATCH_JOBS`, so the group's membership, the two
    tables and the entry rows cannot drift apart silently. #36 is asserted present
    rather than absent: the repair adds rows, and a repair that deleted a row to make
    a count agree would fail here as loudly as one that duplicated the pair.
    """
    funcs = _table(FUNCTIONS_HEADER)
    # The same two parses `test_every_live_task_file_is_named_once_in_each_membership_table`
    # feeds its reconciler, so this node's id sets are the ones the vault-side check
    # compares against the directory and not a second reading of the tables.
    grouped, tiered = (ids for _, ids in _membership_tables())   # functions, tiers
    assert sorted(grouped) == sorted(tiered), (
        "the two membership tables name different id sets; symmetric omission is the "
        "one drift `test_every_job_has_one_function_and_exactly_one_write_tier` "
        "cannot see, and it is how all three of #2630, #2813 and #2881 arrived")
    for job in ("#100", "#101"):
        assert grouped.count(job) == 1, (
            f"{job} appears {grouped.count(job)} times in the function table; #2881 "
            f"is one row per live task file in each table, counting #98's and #99's "
            f"`status: draft` as membership and not as arming")
        assert tiered.count(job) == 1, (
            f"{job} appears {tiered.count(job)} times in the write-authority table, "
            f"which is the doc's answer to what that job may do unattended")
    assert "#36" in grouped and "#36" in tiered, (
        "#36 lost its row: the ghost exemption in the reconciler is the reason it "
        "keeps one row in each table, so a repair that dropped it to fix a count "
        "would not also be a repair that kept the tables symmetric")
    watch = next(v for k, v in funcs.items() if k.startswith("[Watch the machinery]"))
    assert sorted(watch[0]) == sorted(f"#{j}" for j in WATCH_JOBS), (
        f"§Watch the machinery's row lists {sorted(watch[0])} while `WATCH_JOBS` "
        f"holds rows for {sorted(f'#{j}' for j in WATCH_JOBS)} — the heading node "
        f"checks the row against the heading, and only this one ties the group to "
        f"the entries and tiers typed above")


def test_no_live_sentence_calls_a_superseded_id_the_newest():
    """#2881 clause 4's prose half. The numbering note said "#99 … is the newest",
    in the present tense, while the directory already held #100 and #101 — the
    second time this file has had to retire a sentence of that shape (the wiring
    note's "#99 is the newest file" is its first), and the second time the floor
    beside it was the thing that outlived the id it named.

    The expectation is derived, not typed: the highest id either membership table
    names is what a present-tense newest-claim has to name. Past tense stays legal —
    §The wiring and §Review log keep the id each was written under, and rewriting
    dated prose would destroy the record #2813's entry is for.
    """
    listed = max(int(i[1:]) for ids, _ in _table(FUNCTIONS_HEADER).values() for i in ids)
    claims = [(int(m.group(1)), s) for s in _sentences(_live_prose())
              if (m := _NEWEST_CLAIM.search(s))]
    assert claims, (
        "no live sentence names the newest job, so the bound below would prove "
        "nothing — positive control failed")
    for named, sentence in claims:
        assert named == listed, (
            f"{sentence!r} calls #{named} the newest while the membership tables list "
            f"#{listed}; the numbering note's own rule is that the id carries the "
            f"date it was the newest, which is a past-tense sentence")


# ── #2631 — the two membership tables reconcile against the task-file directory ─
#
# Everything above this line derives its expectation from another statement in the
# same document — the design this file's own header states, and the reason no count
# in here rots the way a hand-copied roster does. It is also why the whole checking
# surface was blind to the one thing the doc hand-types and no other artifact owns:
# `test_every_job_has_one_function_and_exactly_one_write_tier` compares the two
# tables with *each other*, so a job missing from both is symmetric and passes, and
# `scripts/autonomy/validate_tasks.py` never opens the doc at all
# (`git grep -c "autonomy-jobs" scripts/autonomy/validate_tasks.py` → 0, 2026-10-13).
#
# The drift that proves both directions: on 2026-10-12 the tables named #87 #88 #89,
# whose task files had been removed (their run records still sit in
# `~/lloyd-data/autonomy-runs/8[789]/`), and omitted #91–#97, whose task files were
# live and running. #2630 repaired both halves by hand. The reconciler below is the
# check that would have seen them, and its source of truth is the directory, because
# the directory is what dispatches: the scheduler lists `~/obsidian/autonomy/*.md`
# and nothing else, so a table row with no file behind it cannot run and a file with
# no table row runs unseen.
#
# The one case the ghost rule must not fire on is #36, whose row at
# `architecture/autonomy-jobs.md:823` says in prose that it is retired and whose file
# is in `_archived/`. So the exemption is read off the row rather than typed into a
# list: an allowlist here would be the same hand-maintained roster this section
# exists to replace, and a ghost could buy its way back by editing this file instead
# of writing a task file.
#
# Two of these nodes read the live vault and carry `@live_vault`, so the automod
# gate deselects them (`pytest.ini:10-13`) — a nightly job files a task file, and
# the next author's unrelated round would go red for it. They run under a bare
# `pytest tests/`, which is what a suite pass and the owed-check job execute. The
# three nodes above them are fixture-backed and run at the gate, which is where the
# rule itself is proved: a check that can only be exercised against the live vault
# is a check no rung can see fail.

TASK_FILE_NAME = re.compile(r"(\d+)-.*\.md$")

#: A membership row that discharges its own id: `| #36 | … Retired 2026-09-23 …`.
#: Case-sensitive on purpose — the words *retired* and *archived* recur in prose
#: narrating other jobs' histories, and a row that merely mentions a retirement is
#: not that row's own retirement. Only #36's row matches in today's doc.
RETIRED_MEMBERSHIP_ROW = re.compile(r"^\| #(\d+) \|.*\b(?:Retired|Archived)\b", re.M)


def _task_dir() -> Path:
    """`<vault>/autonomy`, resolved the way `tests/board_presence.py:vault_root()`
    resolves the vault (`LLOYD_OBSIDIAN_VAULT`, else `Path.home()/"obsidian"`), so a
    caller that redirects the vault is honoured instead of a second hard-coded path.
    """
    from tests.board_presence import vault_root

    d = vault_root() / "autonomy"
    assert d.is_dir(), f"{d} is not a directory, so the fleet's source of truth is unreadable"
    return d


def _task_file_ids(tasks_dir: Path) -> set[str]:
    """The live fleet as ids, read off `tasks_dir`'s filenames: `{#24, …, #97}`.

    Not `_archived/`: `glob` does not recurse, and an archived task is not a fleet
    member — which is what #36's own row says, and why that row rather than this
    directory is what discharges it.

    The non-empty assertion is this file's standing positive control. A helper
    handed a path it cannot read yields the empty set, and the empty set is the one
    input that makes every id the doc names an offence *and* every real omission
    invisible, in the same run.
    """
    ids = {f"#{m.group(1)}"
           for m in (TASK_FILE_NAME.match(p.name) for p in tasks_dir.glob("*.md"))
           if m}
    assert ids, (
        f"no `NN-*.md` task file in {tasks_dir}, so there is nothing to reconcile "
        f"against — a fleet of zero tasks is not a fleet")
    return ids


def _retired_named_ids(doc_text: str) -> set[str]:
    """Ids the doc discharges by its own row: `{#36}` in today's document.

    Read off the row, never typed, for the reason given on `RETIRED_MEMBERSHIP_ROW`.
    """
    return {f"#{n}" for n in RETIRED_MEMBERSHIP_ROW.findall(doc_text)}


def _job_num(job: str) -> int:
    return int(job[1:])


def _membership_drift(named: list[str], live: set[str],
                      retired: set[str]) -> tuple[list[str], list[str], list[str]]:
    """`(ghosts, unlisted, duplicated)` for one table's ids against `live`.

    Three ways a hand-typed roster lies about the fleet, each a separate failure so
    the message says which one happened:

    - **ghost** — the table names an id with no live task file behind it, and that
      id's own membership row does not say it is retired. (#87 #88 #89, 2026-10-12.)
    - **unlisted** — a live task file whose id the table never names. (#91–#97.)
    - **duplicate** — the id appears twice in the table, so the roster claims the
      job holds the group twice.
    """
    named_set = set(named)
    ghosts = sorted(named_set - live - retired, key=_job_num)
    unlisted = sorted(live - named_set, key=_job_num)
    duplicated = sorted({j for j in named if named.count(j) > 1}, key=_job_num)
    return ghosts, unlisted, duplicated


def _assert_no_ghosts(table: str, named: list[str], live: set[str],
                      retired: set[str], tasks_dir: Path) -> None:
    """Every id `table` names is dispatchable, or its own row retires it."""
    ghosts, _, _ = _membership_drift(named, live, retired)
    assert not ghosts, (
        f"{table} names {', '.join(ghosts)} as a fleet member, but {tasks_dir} holds "
        f"no {' or '.join(f'`{j[1:]}-*.md`' for j in ghosts)} — "
        f"`test_the_departed_paper_digests_hold_no_membership_row` called exactly this "
        f"defect for #87 #88 #89. Take the id out of both membership tables, or give "
        f"it a task file, or mark its own group-table row "
        f"`| #N | … Retired <date> …` as #36's does")


def _assert_no_unlisted(table: str, named: list[str], live: set[str],
                        tasks_dir: Path) -> None:
    """Every live task file has a row in `table`."""
    _, unlisted, _ = _membership_drift(named, live, set())
    assert not unlisted, (
        f"{tasks_dir} holds a live task file for {', '.join(unlisted)} that {table} "
        f"never names — the omission #2630 closed for #91–#97, which ran for two "
        f"weeks while this document claimed its membership was complete. The doc's "
        f"bargain is that the schedule lives in the task file and the membership "
        f"lives here")


def _assert_listed_once(table: str, named: list[str]) -> None:
    """No id appears in `table` more than once."""
    _, _, duplicated = _membership_drift(named, set(named), set())
    assert not duplicated, (
        f"{table} lists {', '.join(duplicated)} twice, so the roster says the job "
        f"sits in the group more than once — one row per job per table is what "
        f"`test_every_job_has_one_function_and_exactly_one_write_tier` counts on")


def _membership_tables() -> list[tuple[str, list[str]]]:
    """`(table name, the ids it names)` for the two hand-typed membership tables."""
    out: list[tuple[str, list[str]]] = []
    for header, name in ((FUNCTIONS_HEADER, "§The functions membership table"),
                         (TIERS_HEADER, "the write-authority tier table")):
        named = [j for ids, _ in _table(header).values() for j in ids]
        assert named, (
            f"{name} parsed to no job ids at all, so any reconciliation below would "
            f"be vacuous — the header the parser looks for is {header!r}")
        out.append((name, named))
    return out


def test_the_reconciler_fails_a_table_naming_an_id_with_no_task_file(tmp_path: Path):
    """#2631 clause 3, ghost direction, against a seeded directory.

    `tmp_path` is a real directory read by the real helper, not an id set typed
    beside the expectation: the point of the clause is that the check consults the
    filesystem, so the fixture is files on disk and the offender is a filename that
    is not there.
    """
    for name in ("24-data-pipeline.md", "30-intelligence-pipeline-scan-score.md"):
        (tmp_path / name).write_text("---\ntitle: seeded\n---\n")
    live = _task_file_ids(tmp_path)
    assert live == {"#24", "#30"}, f"the seeded directory yielded {sorted(live)}"

    named = ["#24", "#30", "#87"]
    with pytest.raises(AssertionError) as err:
        _assert_no_ghosts("the seeded membership table", named, live, set(), tmp_path)
    assert "#87" in str(err.value), (
        f"a ghost went un-named in its own failure: {err.value}")
    assert "87-*.md" in str(err.value), (
        f"the failure does not name the task file it looked for: {err.value}")
    _assert_no_ghosts("the seeded membership table", ["#24", "#30"], live,
                      set(), tmp_path)


def test_the_reconciler_fails_a_live_task_file_the_tables_omit(tmp_path: Path):
    """#2631 clause 3, unlisted direction, against a seeded directory.

    #98 is the shape of #91–#97: a task file someone creates, which fails nothing in
    the old surface because the tables are only ever compared with each other.
    """
    for name in ("24-data-pipeline.md", "98-seeded-watch-job.md"):
        (tmp_path / name).write_text("---\ntitle: seeded\n---\n")
    live = _task_file_ids(tmp_path)
    assert live == {"#24", "#98"}, f"the seeded directory yielded {sorted(live)}"

    with pytest.raises(AssertionError) as err:
        _assert_no_unlisted("the seeded membership table", ["#24"], live, tmp_path)
    assert "#98" in str(err.value), (
        f"an omitted live task went un-named in its own failure: {err.value}")
    _assert_no_unlisted("the seeded membership table", ["#24", "#98"], live, tmp_path)


def test_the_reconciler_lets_a_retired_row_keep_its_id(tmp_path: Path):
    """#2631 clause 1's exception, seeded: #36 stays named, and must not be flagged.

    Without this the rule is unsatisfiable on today's tree — #36 is named in both
    tables and has no live task file — and a check that cannot pass is not a check.
    The exemption is exercised through the real row parser, because a rule with an
    exception nobody feeds it through is an exception nobody can rely on.
    """
    (tmp_path / "24-data-pipeline.md").write_text("---\ntitle: seeded\n---\n")
    live = _task_file_ids(tmp_path)
    row = ("| #36 | daily | the vault | **Retired 2026-09-23 (#1012)** — the row and "
           "the id stay because the group tables pin each other; its task file is in "
           "`_archived/` and nothing dispatches it |")
    retired = _retired_named_ids(row)
    assert retired == {"#36"}, f"the row that retires itself did not parse: {sorted(retired)}"

    _assert_no_ghosts("the seeded membership table", ["#24", "#36"], live, retired,
                      tmp_path)
    with pytest.raises(AssertionError) as err:
        _assert_no_ghosts("the seeded membership table", ["#24", "#36"], live, set(),
                          tmp_path)
    assert "#36" in str(err.value), (
        f"with the exemption withheld, the failure did not name #36: {err.value}")


def test_the_reconciler_fails_a_job_listed_twice_in_one_table():
    """#2631 clause 2's `exactly once`, as a failure rather than as a word.

    `test_every_job_has_one_function_and_exactly_one_write_tier` counts tier rows,
    but a second row for the same id in the *functions* table moves the group's
    count without moving that node's expectation.
    """
    named = ["#24", "#30", "#30"]
    with pytest.raises(AssertionError) as err:
        _assert_listed_once("the seeded membership table", named)
    assert "#30" in str(err.value), (
        f"a duplicated id went un-named in its own failure: {err.value}")
    _assert_listed_once("the seeded membership table", ["#24", "#30"])


@live_vault
def test_every_id_the_membership_tables_name_has_a_task_file_or_a_retired_row():
    """#2631 clause 1, against `~/obsidian/autonomy/` as it is on disk.

    The source of truth is the directory, not the doc — which is what makes this
    node the counterpart of everything above it in this file. `@live_vault` for the
    standing reason (`pytest.ini:10-13`): a nightly job can file or retire a task
    file, and the automod gate must not make the next author's round red for it. Run
    it from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            "tests/test_autonomy_jobs_doc_claims.py::test_every_id_the_membership_tables_name_has_a_task_file_or_a_retired_row" -q

    #36 is the positive control: it is named in both tables and has no live task
    file, so if its own row stopped parsing, the exemption would be silently gone and
    this node would fail for the wrong reason — the assertion below says which.
    """
    tasks_dir = _task_dir()
    live = _task_file_ids(tasks_dir)
    retired = _retired_named_ids(_text())
    assert "#36" in retired, (
        "no membership row carries a Retired/Archived marker, so #36 — named in both "
        "tables, its file in `_archived/` — would be reported as a ghost. Its row at "
        "architecture/autonomy-jobs.md is the one case the rule exempts")
    for name, named in _membership_tables():
        _assert_no_ghosts(name, named, live, retired, tasks_dir)


@live_vault
def test_every_live_task_file_is_named_once_in_each_membership_table():
    """#2631 clause 2, against `~/obsidian/autonomy/` as it is on disk.

    39 live task files on 2026-10-13, every one of them named in both tables — the
    state #2630 produced by hand and this node is what keeps. `@live_vault`, same
    reason and same command as
    `test_every_id_the_membership_tables_name_has_a_task_file_or_a_retired_row`.
    """
    tasks_dir = _task_dir()
    live = _task_file_ids(tasks_dir)
    assert len(live) >= 39, (
        f"{tasks_dir} holds {len(live)} live task files; the last measurement "
        f"(2026-10-13) counted 39, so the directory itself changed shape and this "
        f"node's own reading needs re-checking before its verdict means anything")
    for name, named in _membership_tables():
        _assert_no_unlisted(name, named, live, tasks_dir)
        _assert_listed_once(name, named)


# ── #2632 — no current sentence may claim a task file names the retired slot ──
#
# `autonomy/68-morning-brief-triage.md` carried `model: secondary` from 2026-04-22
# while `secondary_enabled: false` has been in force since `551e9044` (2026-09-20).
# This document described that line in the present tense in two places — §Ingest's
# "the only one whose task file still names the **secondary** engine" and §#68's
# "its `model: secondary` now names the slot" — and no node here pinned either one:
# the checks above read the doc against other sentences of the doc, and the one fact
# these two sentences depend on lives in another tree entirely (`~/obsidian/autonomy/`
# plus a config flag), so retyping the front matter would have left both sentences
# asserting an engine the box does not boot. Nothing in the gate would have noticed.
#
# The scan is therefore on the shapes of the two stale sentences, over `_live_prose()`
# only. §Review log keeps its dated "2026-10-12 — `stale`" entry saying what the file
# held that day, which is correct as written and is what the log exists for; a reader
# who takes that entry as current is reading a date. Past tense with its date and its
# item is also allowed in live prose — the hazard #2634 pinned stays stated, with
# `resolve_model_alias` and **#1555** in it.

#: Present-tense claims that a task file's `model:` line names the secondary engine.
#: `*{0,2}` because the doc bolded the word once and did not bold it the other time.
STALE_MODEL_CLAIMS = (
    re.compile(r"still names the \*{0,2}secondary\*{0,2}"),
    re.compile(r"`model: secondary`[^.]{0,160}\b(?:now|still)\b"),
)


def test_no_current_sentence_claims_a_task_file_names_the_retired_secondary_engine():
    """#2632 clause 5: the doc's two §#68 sentences move out of the present tense.

    Reverting either sentence — or writing a new one like it for some future retired
    slot — fails this. The second assertion is the positive half: §#68 has to say the
    value the file carries now, because a paragraph that only stops being wrong has
    replaced a claim with a silence, and the next reader learns nothing about where
    the job's 15-minute brief actually runs.
    """
    live = _flat(_live_prose())
    offenders = [p.pattern for p in STALE_MODEL_CLAIMS if p.search(live)]
    assert not offenders, (
        f"current prose claims a task file names the secondary engine again "
        f"(matched {offenders}); #2632 set `68-morning-brief-triage.md` to "
        f"`model: primary` on 2026-10-13, so the present tense is false — say what "
        f"it reads now, and keep any history dated and inside §Review log")

    section = _parked_job_section()
    assert "set it to `primary`" in section, (
        "§#68 no longer states the value #68's `model:` line carries, so the "
        "hazard's history is left with nothing to be the history of")
    assert "validate_tasks" in live, (
        "§#68 lost the sentence saying the linter now reports a `model:` naming a "
        "switched-off slot — the check is the reason the stale line could not "
        "survive a second time, and the doc is where a reader looks for that")


# ── #2633 — a `file:line` citation is a claim, so it gets checked like one ────
#
# Four citations in this document had already drifted by 2026-10-12 and nothing
# noticed: `1d41b432` ("arch-review: doc:autonomy-jobs — stale") corrected exactly
# four numbers in place — `tests/test_skill_tool_names.py` 137→150,
# `scripts/skill_lint.py` 276→281, `app/routers/backlog.py` 754→804,
# `BacklogPage.tsx` 571→572 — and the suite stayed green throughout, because the only
# citation any node resolved was `EVIDENCE_PILOT_TASK_IDS` at
# `test_the_pilot_frozenset_citation_resolves_to_the_line_it_names`. A number in prose
# is load-bearing here in a specific way: several sentences cite a line to prove the
# thing is *not* where a reader might assume ("not inside the verifier's own file"), so
# when the definition moves the sentence stops being evidence and still reads like it
# is. This is the class rule `growing-a-docstring-moves-every-line`: any docstring
# growth below a citation silently voids it.
#
# The rail is a resolver, not a list of pins: it reads every `path.ext:N` citation out
# of the live prose, works out which backticked symbol the doc quotes *beside* it, and
# asks the named file whether that line still is that symbol. A list of four pinned
# numbers would have had the same blind spot as the four it would have fixed — the
# next citation someone writes is not on the list.
#
# Scope is the §Review log's exclusion, and it is not a convenience: §Review log's
# 2026-09-12 entry cites `app/autonomy.py:710` for the worker-evidence gate, a number
# that was correct that day and is deliberately not corrected, because the entry
# records what was true on the day it says.
# `test_the_pilot_frozenset_citation_resolves_to_the_line_it_names` asserts that string
# is absent from live prose, and
# `test_the_dated_review_log_entry_keeps_what_was_true_that_day` asserts it IS present
# in the log — so a resolver that scanned the whole file would fight both of them
# instead of checking the claims. Neither is cited by a line number here, because this
# is the item about what happens to a number in prose.

#: A citation in the `path.ext:N` shape the doc uses for "go look at this line".
CITATION = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|tsx|ts|md|sh|json)):(\d+)`")

#: A backticked span. The doc wraps near 88 columns and a symbol's span can straddle
#: a wrap, so spans are matched only after `_flat` has collapsed the newlines.
QUOTED_SPAN = re.compile(r"`([^`\n]+)`")

#: One candidate symbol. A hyphen stays INSIDE the token: the doc quotes skill names
#: (`nightly-skills-management`) as often as it quotes identifiers, and splitting a
#: slug into `nightly` + `skills` + `management` invents three symbols the sentence
#: never said, then reports line numbers no reader asked for — the seeded perturbation
#: in `test_a_drifted_citation_names_the_line_and_where_the_symbol_actually_is` produced
#: four such locations for one citation before the hyphen was kept in the token. A slug
#: never matches a Python or TSX source line, so it can only ever be a candidate that
#: declines to resolve.
IDENT_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")

#: Words that sit in backticks for a reason other than being a symbol to locate.
NOT_A_SYMBOL = frozenset({"py", "tsx", "ts", "md", "sh", "json", "true", "false",
                          "None", "True", "False", "id", "line"})

#: How far either side of a citation to look for the symbol the doc quotes beside it.
#: 320 characters is one or two sentences of this doc's prose: wide enough that
#: "`ALLOWED_TO_MENTION`) and `scripts/skill_lint.py:281` (`PHANTOM_EXEMPT`)" resolves
#: whichever side the name falls on, narrow enough that a citation is not credited to a
#: symbol named two sections away.
CITATION_WINDOW = 320

#: The `path:line` pairs live prose MUST cite. This is the vacuity guard, and it is a
#: SET rather than a count because of #2833: the guard used to compare against the
#: exact `(instances, distinct)` that was true the day #2633 wrote it, and `cffd22e6`
#: added one correct citation — `eval/djev_name_prior_probe.py:647`, in #96's Reads
#: cell, naming the line whose `now().astimezone()` makes the probe's stamp a local
#: date. Measured on the two commits either side of it, live prose went from 9
#: instances over 8 pairs at `a76859f9` to 10 over 9 at `cffd22e6`, one pair added and
#: none removed, and main went red over a sentence that was true. A count can only go
#: up when prose gains a citation and down when it loses one, so naming the pairs keeps
#: both things the count guard was for: a
#: resolver whose pattern stopped matching this shape finds none of them, and a
#: deletion of any citation this document depends on is named in the failure. What it
#: no longer does is punish an author for adding a true citation. Each annotation is
#: the symbol that line defines, which is what the drift resolver checks it against.
#:
#: Eight pairs, not nine: #2820 removes `eval/djev_name_prior_probe.py:647` from #96's
#: row. That citation was true the day `cffd22e6` wrote it and became false by an edit
#: to the function it pointed into — #2820 inserts a helper above that line, which is
#: exactly the way a `path:line` citation goes stale while the prose around it stays
#: right. The row now names `build_report` and `_now()` by symbol instead, and
#: `test_the_name_prior_report_row_states_a_utc_date_key_and_keeps_its_four_tokens`
#: forbids the numeric form coming back.
REQUIRED_CITATIONS = frozenset({
    "app/autonomy.py:2898",                        # EVIDENCE_PILOT_TASK_IDS
    "tests/test_worker_evidence.py:686",           # the assert pinning that frozenset
    "tests/test_skill_tool_names.py:150",          # ALLOWED_TO_MENTION
    "scripts/skill_lint.py:281",                   # PHANTOM_EXEMPT
    "agent_mcp/skills.py:87",                      # _QUARANTINE_STATUSES
    "app/routers/backlog.py:812",                  # backlog_task_delete
    "web/src/components/pages/BacklogPage.tsx:572",   # DONE_WINDOW_DAYS
    "app/paths.py:278",                            # VAULT_FACTS_ROOT, added by #2795
})


def _pair(rel: str, line_s: str) -> str:
    """One citation as the guards compare on it: `path:line`, exactly as written."""
    return f"{rel}:{line_s}"


def _cited_pairs(text: str) -> set[str]:
    """The distinct `path:line` pairs `text` cites, through `CITATION` itself.

    A separate function rather than a re-run of the regex in the node, so the count
    inside `_citation_drift` and the required-set check beside it are the same match
    on the same slice — a second pattern of the author's own is how a guard starts
    agreeing with itself instead of agreeing with the document.
    """
    return {_pair(rel, line_s) for rel, line_s in CITATION.findall(text)}


def _cited_symbols(text: str, start: int, end: int) -> list[str]:
    """Identifiers the doc backticks near one citation, in the order it says them.

    A span is skipped when it contains `/` or `:`: those are paths and other
    citations, never the symbol being located. Everything else in the window is
    offered, which makes the check permissive in one direction only — a stray quoted
    word can make a drifted citation pass, it can never make a true one fail. That is
    the right bias for a rail whose failure costs an author a red node; the seeded
    nodes below are what prove it can still see the drift.
    """
    window = _flat(text[max(0, start - CITATION_WINDOW):min(len(text), end + CITATION_WINDOW)])
    found: list[str] = []
    for span in QUOTED_SPAN.findall(window):
        if "/" in span or ":" in span:
            continue
        found += [t for t in IDENT_TOKEN.findall(span) if t not in NOT_A_SYMBOL]
    return list(dict.fromkeys(found))


def _symbol_definition(rel: str, src_text: str, symbol: str) -> int | None:
    """The line where `symbol` is DEFINED in the file at `rel`, or None.

    Clause 2 asks the failure text to say where the symbol actually lives, and a
    first-occurrence search answers that question wrongly in the one file this item was
    filed over: `ALLOWED_TO_MENTION` is mentioned in a prose comment at
    `tests/test_skill_tool_names.py:41` and defined at :150, so a resolver reporting
    ":41" would send an author to retype a citation to a number that is not the
    definition either. Python goes through `ast`, which is the only way to be sure a
    name is being bound rather than used; everything else (this doc cites one `.tsx`)
    is matched against declaration keywords, with an `export`-prefixed line accepted
    because `web/src/…/BacklogPage.tsx:572` is the file where that matters.

    None is an answer, not a failure: it means the resolver will report where the name
    is *mentioned* instead, and say plainly that nothing there is a definition.
    """
    name = re.escape(symbol)
    if rel.endswith(".py"):
        try:
            tree = ast.parse(src_text)
        except SyntaxError:
            return None
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound = node.name
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                bound = getattr(target, "id", None)
            else:
                continue
            if bound == symbol:
                lines.append(node.lineno)
        return min(lines) if lines else None
    decl = re.compile(rf"^(?:export\s+)?(?:const|let|var|function|class|def)\s+{name}\b")
    for i, line in enumerate(src_text.splitlines()):
        if decl.match(line.strip()):
            return i + 1
    return None


def _citation_drift(text: str) -> tuple[int, int, list[str]]:
    """`(instances, distinct, drift lines)` for one slice of the document.

    Two counts because the doc cites one number twice — `app/autonomy.py:2898` appears
    in both §The functions and §Evidence pilot, so `instances > distinct` is the shape
    of today's prose and a node below asserts it. The pair count is no longer asserted
    as a number: see `REQUIRED_CITATIONS` for the #2833 failure that produced, and a
    citation carrying no symbol beside it is what `drift` reports, so nothing here
    needs a fixed total to notice either of those two things.

    Returned rather than asserted so a caller can feed it a seeded sentence; the nodes
    below are thin wrappers over the same measurement.
    """
    drift: list[str] = []
    seen: set[str] = set()
    instances = 0
    for m in CITATION.finditer(text):
        rel, line_s = m.group(1), m.group(2)
        line = int(line_s)
        instances += 1
        seen.add(_pair(rel, line_s))
        citation = f"`{rel}:{line_s}`"
        source = ROOT / rel
        if not source.is_file():
            drift.append(f"{citation}: no such file in this checkout")
            continue
        src_text = source.read_text(errors="replace")
        src_lines = src_text.splitlines()
        quoted = _cited_symbols(text, m.start(), m.end())
        if not quoted:
            drift.append(
                f"{citation}: no backticked symbol is quoted beside it, so nothing "
                f"here can confirm the number — quote the symbol the line defines")
            continue
        cited = src_lines[line - 1] if 0 < line <= len(src_lines) else ""
        if cited and any(re.search(rf"\b{re.escape(s)}\b", cited) for s in quoted):
            continue
        # Clause 2: say where the symbol IS. The first line that merely *mentions* it
        # is not the answer — `ALLOWED_TO_MENTION` appears in a docstring comment at
        # tests/test_skill_tool_names.py:41 and 109 lines below that is the definition
        # the citation means — so the location is found as a definition, and only falls
        # back to an occurrence when nothing in the file defines the name.
        lives = []
        for s in quoted:
            at = _symbol_definition(rel, src_text, s)
            if at:
                lives.append(f"`{s}` is defined at {rel}:{at}")
        if not lives:
            occurrences = [i + 1 for i, l in enumerate(src_lines)
                           if any(re.search(rf"\b{re.escape(s)}\b", l) for s in quoted)]
            if occurrences:
                shown = ", ".join(str(o) for o in occurrences[:3])
                lives = [f"{rel} mentions {'/'.join(quoted)} at line {shown}"
                         f"{'…' if len(occurrences) > 3 else ''} but defines none "
                         f"of them"]
            else:
                lives = [f"not one of {quoted} appears anywhere in {rel}"]
        reads = (cited.strip()
                 if cited else f"<no such line: {rel} ends at {len(src_lines)} lines>")
        drift.append(
            f"{citation} is not where this document says. {rel}:{line} reads "
            f"{reads!r}, and " + "; ".join(lives)
            + " — retype the number in the sentence, or the symbol it names has been "
              "renamed away and the sentence needs the new one")
    return instances, len(seen), drift


def test_every_live_prose_citation_resolves_to_the_symbol_beside_it():
    """#2633 clauses 1 and 3, and #2833 clause 1: every citation in the current prose
    points at its symbol, and a doc that GAINS a true citation stays green.

    Ten citation instances over nine distinct `path:line` pairs in today's live prose
    (`app/autonomy.py:2898` twice — §The functions and §Evidence pilot — then
    `tests/test_worker_evidence.py:686`, `tests/test_skill_tool_names.py:150`,
    `scripts/skill_lint.py:281`, `agent_mcp/skills.py:87`,
    `app/routers/backlog.py:812`,
    `web/src/components/pages/BacklogPage.tsx:572`, `app/paths.py:278` from #2795, and
    `eval/djev_name_prior_probe.py:647` from cffd22e6), the nine of them named in
    `REQUIRED_CITATIONS`. The tenth distinct pair in the file — `app/autonomy.py:710`,
    the worker-evidence gate as numbered on 2026-09-12 — lives only in the dated
    §Review log and is out of scope by design, which is clause 3;
    `test_the_pilot_frozenset_citation_resolves_to_the_line_it_names` already forbids
    it in live prose.

    The vacuity guard used to be `assert (instances, distinct) == (9, 8)`, which is a
    date disguised as a check: it held on the day #2633 wrote it and `cffd22e6` broke
    main by adding one correct citation to #96's Reads cell (#2833, measured on the two
    commits either side of it as 9/8 → 10/9, one pair added and none removed). Naming
    the pairs that must be present is the same strength where strength was wanted — a
    resolver whose pattern stopped matching finds none of them, and deleting any of
    these nine is named in the failure — and it stops charging an author a red node for
    a true sentence. `instances > distinct` is the one shape claim kept, because the
    double citation is the fact the two counts were introduced to hold.
    """
    instances, distinct, drift = _citation_drift(_live_prose())

    assert not drift, "; ".join(drift)

    missing = sorted(REQUIRED_CITATIONS - _cited_pairs(_live_prose()))
    assert not missing, (
        f"{len(missing)} citation(s) this document is pinned on are gone from live "
        f"prose: {missing} — either the sentence that made them was deleted, or the "
        f"pattern in CITATION stopped matching the shape they are written in, and a "
        f"resolver that matches nothing reports no drift, which is not a clean result")
    assert instances > distinct, (
        f"{instances} instances over {distinct} pairs: live prose is supposed to cite "
        f"one `path:line` twice (`app/autonomy.py:2898`, in §The functions and "
        f"§Evidence pilot), and an instances count equal to the distinct count means "
        f"the second of them is gone")

    # The guard's own failure mode, seeded: a slice holding no citation at all must be
    # refused, not read as a clean document. #2833 clause 2 — nothing here is skipped,
    # xfailed or deleted, and this is the read that proves the rewritten guard can
    # still go red.
    blind_instances, blind_distinct, blind_drift = _citation_drift(
        "A slice with nothing in `path.ext:N` shape at all.")
    assert (blind_instances, blind_distinct, blind_drift) == (0, 0, []), (
        f"the seeded no-citation slice counted {blind_instances}/{blind_distinct} with "
        f"drift {blind_drift} — CITATION matches something it should not, so the "
        f"control below is weighing the wrong thing")
    blind_pairs = _cited_pairs("A slice with nothing in `path.ext:N` shape at all.")
    assert blind_pairs == set(), (
        f"the seeded slice resolved to {sorted(blind_pairs)}, which is not the empty "
        f"set the control needs")
    assert REQUIRED_CITATIONS - blind_pairs == REQUIRED_CITATIONS, (
        "the required-set guard found citations in a slice that holds none, so a "
        "resolver that stopped matching this doc's shape could report a clean "
        "document — the vacuity case #2633 clause 3 was written for")


def test_a_drifted_citation_names_the_line_and_where_the_symbol_actually_is():
    """#2633 clause 2, seeded: the failure text is the repair.

    The perturbation is the exact drift #2633 was filed for: `ALLOWED_TO_MENTION`'s
    definition sat at `tests/test_skill_tool_names.py:137` and moved to :150 when that
    file's docstring grew. Seeding :137 and reading the message must be enough to fix
    the sentence — no hunt, because the hunt is the thing that was never done.
    """
    seeded = ("Two linters enforce rule 4. `tests/test_skill_tool_names.py:137` "
              "(`ALLOWED_TO_MENTION`) and `scripts/skill_lint.py:281` "
              "(`PHANTOM_EXEMPT`) let `nightly-skills-management` write the phantom "
              "names down.")

    instances, distinct, drift = _citation_drift(seeded)

    assert (instances, distinct) == (2, 2), (
        f"the seeded sentence holds 2 citations, resolver saw {instances} "
        f"({distinct} distinct)")
    assert len(drift) == 1, f"expected exactly the seeded citation to drift: {drift}"
    msg = drift[0]
    assert "`tests/test_skill_tool_names.py:137`" in msg, msg
    assert "tests/test_skill_tool_names.py:150" in msg, (
        f"the message does not say where the symbol really is: {msg}")
    assert "ALLOWED_TO_MENTION" in msg, msg


def test_the_resolver_ignores_a_citation_that_lives_only_in_the_dated_review_log():
    """#2633 clause 3: §Review log's stale number stays a record, not a failure.

    Three facts, because the exclusion is only safe if all three hold. (a) The
    resolver run over this document's live prose reports nothing about
    `app/autonomy.py:710`, and the string is still in §Review log — clause 3's
    requirement. (b) The SAME citation and symbol, fed as live prose, DOES drift, so
    "not reported" is the scope doing the work rather than a blind pattern: the
    2026-09-12 entry cites `app/autonomy.py:710` for `EVIDENCE_PILOT_TASK_IDS`, which
    is defined at `app/autonomy.py:2898` today (#2675 moved it there from 2818 by
    writing the roster's per-id reasons above it) — seed that sentence as live prose
    and it drifts, naming 2898. (c) The two existing nodes that keep the log's
    history honest (`test_the_pilot_frozenset_citation_resolves_to_the_line_it_names`
    forbidding the number in live prose, `test_the_dated_review_log_entry_keeps_what_was_true_that_day`
    requiring it in the log) still assert their own pair of claims — this node neither
    replaces them nor needs either to change.
    """
    instances, distinct, drift = _citation_drift(_live_prose())

    assert "app/autonomy.py:710" in _review_log(), (
        "the 2026-09-12 review-log entry no longer cites `app/autonomy.py:710`; if the "
        "history was rewritten rather than corrected, clause 3 has nothing to exclude")
    assert not [d for d in drift if "app/autonomy.py:710" in d], (
        f"the resolver reported the log's historical citation as drift: {drift}")
    assert "app/autonomy.py:710" not in _live_prose(), (
        "the historical number has crept back into current prose, which "
        "test_the_pilot_frozenset_citation_resolves_to_the_line_it_names forbids; this "
        "node must keep failing for THAT reason and not look like a resolver bug")

    seeded = ("the pilot frozenset `EVIDENCE_PILOT_TASK_IDS` is in "
              "`app/autonomy.py:710` and not in `workers/evidence.py`, which is what "
              "the review found")
    seeded_instances, seeded_distinct, seeded_drift = _citation_drift(seeded)
    assert (seeded_instances, seeded_distinct) == (1, 1), seeded_instances
    assert len(seeded_drift) == 1, (
        f"a live-prose sentence citing :710 for `EVIDENCE_PILOT_TASK_IDS` must drift, "
        f"or the exclusion above is a blind spot rather than a scope: {seeded_drift}")
    # The number the drift must name is READ from the source, not remembered: the
    # pilot frozenset moved from 2818 to 2830 when #2675 wrote its roster reasons
    # above the definition, and a hardcoded expectation here would have reported
    # that legitimate move as a resolver failure.
    where = next(n for n, text in
                 enumerate((ROOT / "app" / "autonomy.py").read_text().splitlines(), 1)
                 if text.startswith("EVIDENCE_PILOT_TASK_IDS = frozenset("))
    assert f"app/autonomy.py:{where}" in seeded_drift[0], seeded_drift[0]


#: The two exemption lists the §linters paragraph quantifies, in the files it names.
EXEMPTION_LISTS = (("tests/test_skill_tool_names.py", "ALLOWED_TO_MENTION"),
                   ("scripts/skill_lint.py", "PHANTOM_EXEMPT"))


def _exemption_names(source: str, name: str) -> set[str]:
    """The string entries of one exemption list, read with `ast`.

    Parsed, not imported: `tests/test_skill_tool_names.py` is a test module and
    `scripts/skill_lint.py` reaches for the vault at import, and the claim being
    checked is about the *literals in the file* — which is also the only way to see
    `frozenset({...})` and a bare `{...}` as the same shape, since the two lists are
    written those two different ways.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if getattr(target, "id", None) != name:
            continue
        return {c.value for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    raise AssertionError(f"{name} is no longer an assignment in its source file")


#: Spelled-out numbers, because this doc writes "seven-name", not "7-name".
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _linters_paragraph() -> tuple[int, list[str]]:
    """The size and the chain names the §linters paragraph states, read off the doc.

    Derived rather than typed because the paragraph is the thing under test. It says
    "each carries a seven-name exemption list of which three entries are this chain"
    and then names those three, so both numbers come from its own words: retype the
    sentence to "eight-name" and the rail follows the sentence, while a name added to
    one file without the sentence being told reddens it. That is the drift #2633 is
    about, and why a symbol-line pin would have missed it — appending an entry to a
    set literal moves no definition line.

    The names are extracted from between "let " and " write the phantom names down"
    rather than from a comma pattern, because the doc closes its list with "and", and
    an `A`, `B` and `C` shape is exactly what a comma-only regex drops silently — a
    dropped name is a membership claim this node would then stop checking.
    """
    para = _flat(_live_prose())
    m = re.search(r"a (\w+)-name exemption list of which (\w+) entries are this chain",
                  para)
    assert m, (
        "§linters no longer states the exemption lists' size and chain count in one "
        "sentence; this node reads both out of that sentence and cannot guess")
    size, count = WORDS.get(m.group(1)), WORDS.get(m.group(2))
    assert size and count, f"unspelled number in the exemption sentence: {m.group(0)}"
    # Windowed from the size sentence rather than searched whole-document: the first
    # `let ` in this doc sits 36,107 characters before that sentence, and a lazy `.+?`
    # across the whole file reaches past it and collects every backticked word in
    # between — 344 comma-separated spans over 36,375 characters on today's text, which
    # would silently widen "the chain's three entries" into "everything". 700 characters
    # is a floor on the same property the stop below makes a hard bound.
    window = para[m.start():m.start() + 700]
    named = re.search(r"let (.+?) write the phantom \w+ down", window)
    assert named, (
        "§linters no longer names the chain skills it says are exempt, so there is "
        "nothing here to check membership against")
    # The bound the window exists for, checked rather than assumed: `para` is flat, so
    # a heading inside the window would mean the window crossed out of §linters and the
    # name list below would be collecting the NEXT section's quoted words as claimed
    # chain entries. Today the next heading is 3,453 characters away, so 700 is inside
    # the paragraph with room; if the paragraph is ever edited down to under 700
    # characters this fires and the window is what needs retuning.
    nxt = para.find(" ## ", m.start())
    assert nxt == -1 or nxt > m.start() + 700, (
        f"the 700-character window runs past the §linters paragraph into the next "
        f"section (heading at +{nxt - m.start()}), so the chain-name list below is no "
        f"longer the sentence's own")
    chain = re.findall(r"`([\w-]+)`", named.group(1))
    assert chain, f"no backticked skill names between 'let' and 'write down': {para[:200]}"
    assert len(chain) == count, (
        f"§linters says {count} entries are this chain and then names {len(chain)} "
        f"({chain}) — the sentence disagrees with itself, so neither number can be "
        f"trusted as the claim")
    return size, chain


def _exemption_verdict(sources: dict[str, str], size: int,
                       chain: list[str]) -> list[str]:
    """Complain about these two exemption-list sources against the doc's sentence.

    An empty list means the sentence and the files agree. One function because the whole
    point is that seeded bytes get the SAME judgement the real ones get — an assertion
    hand-copied into a perturbation node is a second rule, and the thing being tested
    here is that one rule fires.

    A name's absence is reported even when the size happens to match: swapping one
    entry for another keeps the count and voids the sentence all the same.
    """
    problems: list[str] = []
    names = [n for _, n in EXEMPTION_LISTS]
    measured = {n: _exemption_names(sources[n], n) for n in names}

    for rel, name in EXEMPTION_LISTS:
        other = [n for n in names if n != name][0]
        if len(measured[name]) != size:
            problems.append(
                f"{name} in {rel} holds {len(measured[name])} names, while §linters "
                f"says each list carries {size} — measured now: "
                f"{name}={sorted(measured[name])}, {other}={sorted(measured[other])}")
        missing = [c for c in chain if c not in measured[name]]
        if missing:
            problems.append(
                f"{name} in {rel} no longer exempts {missing}, which §linters says "
                f"write the phantom names down as their job — measured: "
                f"{sorted(measured[name])}")
    return problems


def _real_exemption_sources() -> dict[str, str]:
    """The two files' text, keyed by the list name each one carries."""
    return {name: (ROOT / rel).read_text(errors="replace")
            for rel, name in EXEMPTION_LISTS}


def test_the_exemption_lists_are_the_size_and_membership_the_paragraph_claims():
    """#2633 clause 4: `ALLOWED_TO_MENTION` and `PHANTOM_EXEMPT`, measured.

    §linters says "each carries a seven-name exemption list of which three entries are
    this chain", and names the three. Both halves re-measured from the two files the
    paragraph cites (`ALLOWED_TO_MENTION` in `tests/test_skill_tool_names.py`,
    `PHANTOM_EXEMPT` in `scripts/skill_lint.py` — named without numbers, because a
    number in prose is the thing this item is about): 7 names in each today, and all
    three of `nightly-skills-management`, `trajectory-skill-mining` and
    `nightly-skill-consolidation` present in both.

    The size and the three names are read out of the doc's own sentence rather than
    typed here, which is what makes adding a name to one file without retyping the
    sentence a failure: the files move, the claim does not. And
    `test_growing_one_exemption_list_reddens_the_rail` feeds seeded bytes through the
    same `_exemption_verdict`, so the shape of this node's failure is known before an
    author has to read it.

    What is deliberately NOT asserted is equality between the two lists. The
    paragraph's own words are that they are "hand-maintained in two different files,
    with nothing pinning them equal", and that is a statement about how they are kept,
    not a contract about their contents: pinning equality would forbid exempting a
    skill in one linter and not the other, which two different checks may each decide
    for themselves.
    """
    size, chain = _linters_paragraph()

    assert len(chain) == 3, (
        f"§linters names {chain} as this chain's entries; its sentence says three, so "
        f"either the list of names or the words 'which three entries' moved")

    problems = _exemption_verdict(_real_exemption_sources(), size, chain)
    assert not problems, "\n".join(problems)


def test_growing_one_exemption_list_reddens_the_rail():
    """Clause 4's failure leg, seeded: the drift a line-number pin cannot see.

    `PHANTOM_EXEMPT` gains one entry and the sentence still says seven. Appending to a
    set literal moves no definition line, so every `scripts/skill_lint.py:281`-style
    check — including the citation resolver above in this same file — stays green while
    the sentence goes stale. That is the drift #2633 was filed for, so its failure is
    exercised rather than assumed:

      * exactly one complaint, naming the grown list, the sentence's number and BOTH
        measured sets, so a reader can see which file is the odd one out;
      * the untouched pair judged at the same `size + 1` complains about the OTHER list
        instead, which is the sentence's word `each` doing its work: growing one file
        and retyping the number strands the file that did not grow;
      * both sources gain the entry and the sentence is retyped to eight: passes. The
        rail follows a corrected claim rather than fossilising today's count, so a real
        eighth exemption can be landed by the same edit in both places.
    """
    size, chain = _linters_paragraph()
    grown_name = EXEMPTION_LISTS[1][1]
    real = _real_exemption_sources()
    grown_text = real[grown_name].replace(
        f"{grown_name} = frozenset({{",
        f'{grown_name} = frozenset({{\n    "some-new-skill",', 1)
    assert grown_text != real[grown_name], (
        f"the seeded edit matched nothing in {EXEMPTION_LISTS[1][0]}; the list literal "
        f"there changed shape and this node now proves nothing")

    grown = _exemption_names(grown_text, grown_name)
    assert len(grown) == size + 1, sorted(grown)
    assert "some-new-skill" in grown and "some-new-skill" not in real[grown_name]

    problems = _exemption_verdict(
        {**real, grown_name: grown_text}, size, chain)
    assert len(problems) == 1, f"expected exactly one complaint, got: {problems}"
    msg = problems[0]
    assert grown_name in msg, msg
    assert f"{size + 1} names" in msg, msg
    for _, other_name in EXEMPTION_LISTS:
        assert f"{other_name}=" in msg, (
            f"the message omits {other_name}'s measured set, so the reader cannot tell "
            f"which of the two files is the odd one out: {msg}")
    assert "some-new-skill" in msg, msg

    other_name = EXEMPTION_LISTS[0][1]
    at_eight = _exemption_verdict({**real, grown_name: grown_text}, size + 1, chain)
    assert len(at_eight) == 1 and other_name in at_eight[0], (
        f"with one list grown and the sentence retyped to {size + 1}, the verdict "
        f"should now complain about the list that did NOT grow: {at_eight}")

    both_text = real[other_name].replace(
        f"{other_name} = {{", f'{other_name} = {{\n    "some-new-skill",', 1)
    assert both_text != real[other_name], (
        f"the seeded edit matched nothing in {EXEMPTION_LISTS[0][0]}")
    assert not _exemption_verdict(
        {other_name: both_text, grown_name: grown_text}, size + 1, chain), (
        "the verdict still complains when both lists carry the entry and the sentence "
        "says the number they carry, so a legitimate eighth exemption could never be "
        "landed")


# ── #2722: the runner job's own schedule and prune, checked from the repo ────
#
# The task file and its skill are vault paths and land through
# `automod_vault_land`, so nothing in the vault can carry a test. What the repo
# *can* do is read those two published definitions and refuse them when they
# stop meaning what the item asked for — which is what the two nodes below do.

_RETAG_JOB = "98-retag-untagged-facts.md"
_ALARM_JOB = "60-knowledge-health-report.md"
_RETAG_SKILL = Path("retag-untagged-facts") / "SKILL.md"


def _front_matter(name: str) -> dict:
    """The job's own fields, as the scheduler's reader would take them."""
    import yaml

    text = (Path.home() / "obsidian" / "autonomy" / name).read_text(encoding="utf-8")
    fm = text.split("---", 2)
    assert len(fm) == 3, f"{name} has no front matter to read a schedule from"
    return yaml.safe_load(fm[1])


def test_the_retag_runner_declares_its_own_rate_and_an_earlier_hour_than_the_alarm():
    """Clause 2: the repair is ordered before the report that reads its tree.

    Both halves are needed and neither is decoration. `frequency:` has to be on
    the file itself: #24 — the job #98 names as its parent — runs `6x-daily`, so a
    child that inherited a rate would be scheduled six times a day against a
    nightly repair, and `validate_tasks.py` treats a contradicting
    `runs_per_day:` as exactly that error. And the hour is the only ordering the
    scheduler enforces here, because `depends_on` cannot bound a child of a
    6x-daily parent.

    The hour is machine-local, not UTC: `_is_preferred_hour` compares
    `_local_hour()` to the list (`app/autonomy.py:2150`), and #60's own pair —
    hour 4, `next_run` 11:00Z — proves the frame on a UTC-7 box. So the test
    compares the declared hours to each other, which is frame-free, and refuses
    any file that quietly redefines what its number means.
    """
    import yaml

    runner = _front_matter(_RETAG_JOB)
    alarm = _front_matter(_ALARM_JOB)

    assert runner.get("frequency"), (
        f"{_RETAG_JOB} declares no frequency of its own, so its rate would come from "
        f"a default rather than from a decision — #24, the parent it names, is 6x-daily")
    assert runner["frequency"] == alarm["frequency"] == "daily", (
        f"runner {runner['frequency']!r} vs alarm {alarm['frequency']!r}: the report "
        f"that reads the repaired tree has to be the same night's")
    assert not runner.get("runs_per_day") or runner["runs_per_day"] == 1, (
        f"runs_per_day: {runner.get('runs_per_day')!r} contradicts a daily rate, which "
        f"is the shape validate_tasks.py calls out")

    runner_hours, alarm_hours = runner.get("preferred_hours"), alarm.get("preferred_hours")
    assert runner_hours and alarm_hours, (
        "ordering this job against the alarm is the hour, not depends_on: #24 has no "
        "window, so the dependency cannot bound when the child becomes eligible")
    assert min(alarm_hours) == 4, (
        f"the alarm job's window moved to hour {min(alarm_hours)}; the clause's 'before "
        f"the Knowledge Health Report' is stated against hour 4 and has to move with it")
    assert min(runner_hours) < min(alarm_hours), (
        f"runner hour {min(runner_hours)} is not earlier than the alarm's "
        f"{min(alarm_hours)}: the report would read the un-repaired tree and the alarm "
        f"buys one more red night, which is the whole point of the ordering")

    body = (Path.home() / "obsidian" / "autonomy" / _RETAG_JOB).read_text(encoding="utf-8")
    assert "_local_hour" in body, (
        f"{_RETAG_JOB} states an hour without naming the frame it is measured in; the "
        f"gate compares machine-local hours, so a UTC reading of the same number is a "
        f"different night")
    assert yaml.safe_load(
        body.split("---", 2)[1]
    )["next_run"].endswith("+00:00"), "next_run has to be the UTC form"


def test_the_runner_keeps_seven_snapshots_and_cannot_reach_the_store_or_a_quarantine(tmp_path):
    """Clause 3: the prune is a number, and it is pointed only at snapshots.

    Step 4 of `skills/retag-untagged-facts/SKILL.md` is what bounds the store this
    job opens: the facts tree is 40,411 `.md` files, and
    `scripts/groundskeeper/retention-sweep.py` has no pattern for `facts-snapshot-*`,
    so an unpruned nightly `cp -al` grows forever. Reading the command out of the
    skill rather than restating it is deliberate — the run executes the skill's
    line, so that is the line worth breaking here.

    Two failure modes are the expensive ones, and both are checked: a keep-count
    that drifts (the store grows, silently), and a glob that also matches its
    neighbours — `facts/` is the live store and `facts-quarantine-*` is the
    2026-09-23 incident's evidence, and an `rm -rf` that reached either would be a
    second incident on the night of a repair.
    """
    skill = (Path.home() / "obsidian" / "skills" / _RETAG_SKILL).read_text(encoding="utf-8")
    kept = re.search(r"[Kk]eep[s]? (?:exactly )?(\d+) ", skill)
    assert kept, "the prune step states no keep-N as a number, so nothing bounds the store"
    n = int(kept.group(1))
    assert n == 7, f"the skill keeps {n} snapshots; clause 3 is stated as the newest 7"

    command = re.search(r"ls -1d facts-snapshot-\*/[^\n]*?xargs -r rm -rf --", skill)
    assert command, (
        "the prune step no longer prints the command it runs, so this test would be "
        "checking a shape the runner does not execute")

    root = tmp_path / "vault-derived"
    (root / "facts" / "Entity").mkdir(parents=True)
    (root / "facts" / "Entity" / "Entity-state.md").write_text("the live store\n")
    quarantine = root / "facts-quarantine-20260923T070837Z"
    quarantine.mkdir()
    (quarantine / "held.md").write_text("the 2026-09-23 incident's evidence\n")
    snapshots = [f"facts-snapshot-20261013T{i:02d}0000Z" for i in range(10)]
    for name in snapshots:
        (root / name / "Entity").mkdir(parents=True)
        (root / name / "Entity" / "Entity-state.md").write_text("linked bytes\n")

    subprocess.run(command.group(0), shell=True, cwd=root, check=True)

    left = sorted(p.name for p in root.glob("facts-snapshot-*"))
    assert left == sorted(snapshots[-n:]), (
        f"kept {left}; the prune must drop the oldest {10 - n} and keep the newest "
        f"{n} by name order, which is the timestamp order")
    assert (root / "facts" / "Entity" / "Entity-state.md").read_text() == "the live store\n", (
        "the prune touched the live facts tree")
    assert (quarantine / "held.md").exists(), (
        "the prune reached facts-quarantine-*: that glob has to match snapshots only")


def test_the_runner_definition_orders_snapshot_then_apply_then_the_scan():
    """Clause 1's ORDER, pinned in the repo rather than read off a vault path.

    The runner is three commands plus a prune, and the order is the whole of its
    safety: an apply that runs before `cp -al` rewrites up to 831 files with no
    witness of what they said beforehand, and a prune that ran before the scan
    would have deleted the snapshot the scan's reading is compared against. The
    two nodes above check the job's rate, its hour and its prune command — none
    of them can see an order, so a definition that quietly reordered the steps
    would still pass both.

    Both surfaces are checked because both reach the worker: `_build_task_prompt`
    (`app/autonomy.py`) renders the skill body together with the front-matter
    `description`, so a body that says snapshot-first beside a description that
    lists the apply first is a live contradiction, not a documentation slip. The
    deletion marker differs per surface for a reason — the skill states the prune
    as the command it runs (`rm -rf`), the description states it as prose
    ("keeping the newest 7") — and the skill's own "snapshot, then apply, then
    log, then prune" line is why the word `prune` cannot serve as a marker there.
    """
    runner = _front_matter(_RETAG_JOB)
    assert runner["skill_name"] == "retag-untagged-facts", (
        f"the task names skill {runner['skill_name']!r}, so the definition this test "
        f"reads is not the one the scheduler renders into the prompt")

    core = ("cp -al", "retag_untagged_facts.py --apply", "kg_hygiene.scan")

    skill = (Path.home() / "obsidian" / "skills" / _RETAG_SKILL).read_text(encoding="utf-8")
    body = skill.split("---", 2)[2]
    steps = [body.index(m) for m in core] + [body.index("rm -rf")]
    assert steps == sorted(steps), (
        f"the skill's steps sit at {steps}, not in the order snapshot → apply → scan → "
        f"prune: an apply ahead of the snapshot leaves an 831-file rewrite unwitnessed, "
        f"and a prune ahead of the scan deletes the witness")

    desc = runner["description"]
    said = [desc.index(m) for m in core] + [desc.index("keeping the newest 7")]
    assert said == sorted(said), (
        f"the task's own `description` orders the steps as {said}, and that field is "
        f"rendered into the prompt beside the skill, so the two definitions have to "
        f"agree on the order rather than merely both be present")


# ── #2805 — a job that has run is not a job that never ran ────────────────────

#: A never-run claim, in either tense this doc writes it in. The claim is not
#: forbidden — #98's was true on 2026-10-16 — only the *undated* one is: both jobs
#: fire daily, so the sentence is false the morning after it is written and nothing
#: on the page says when it was worth reading. "never been bound to disk" (:158,
#: :462) is deliberately outside this: it describes a count, not a job.
_NEVER_RUN = r"\bhas never run\b|\bnever ran\b|\bnever runs\b|\bhas never been run\b"

#: The two halves of the 2026-10-12 statement #2805 retires — the module's creation
#: timestamp and its absence from disk — spelled to catch either tense, because the
#: §Review log keeps it in the past and live prose may hold neither.
_SUPERSEDED_97_CLAIM = (
    r"created at 19:52Z",
    r"\b(?:does|did) not exist on disk\b",
)

#: A run count written so a later reader can tell how stale it is: a number, the word
#: "run", and the date it was measured on. The optional `T…` tail is what #2833 adds.
#: The §Watch the machinery record was re-dated by `cffd22e6` as "4 runs as of
#: 2026-10-16T10:00Z" — a count MORE precise than the "3 runs as of 2026-10-16" it
#: replaced — and the old shape's trailing `\b` sat between the day and the `T`, two
#: word characters with no boundary between them, so the shape matched nothing and the
#: node reported a timestamped count as an undated one. The rail is on a count
#: carrying its date at all; how many digits follow the date is not the defect.
DATED_RUN_COUNT = re.compile(
    r"\b\d+ runs? as of 20\d\d-\d\d-\d\d(?:T\d\d:\d\d(?::\d\d)?Z?)?\b")

#: A period ends a sentence only where whitespace follows it, so the three
#: spellings here whose periods sit inside a path — `retag_untagged_facts.py`,
#: `~/lloyd-data/workers.db`, `~/.local/state/` — do not end one. The two extra
#: lookbehinds are the doc's other habit: it bolds whole lead-in sentences, and
#: `not this page's.** The defect` is a sentence boundary that a plain `[.!?]`
#: split reads as one sentence, quietly lending a claim a date it does not carry.
_SENT_SPLIT = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][*_`])|(?<=[.!?][*_`]{2}))\s+"
)


def _live_sentences() -> list[str]:
    """Live prose as sentences. Table rows are dropped: a cell is not a sentence,
    and both membership tables are pinned by their own nodes above."""
    body = "\n".join(ln for ln in _live_prose().splitlines()
                     if not ln.lstrip().startswith("|"))
    return [s.strip() for s in _SENT_SPLIT.split(_flat(body)) if s.strip()]


def _undated_never_run(sentence: str) -> bool:
    """True for a sentence that says some `#N` never ran without saying when, or
    how to re-check it.

    The re-derivation half is `select count(*) from runs where task_id=N` naming one
    of the ids in the sentence — the command the doc already uses for exactly this
    at §Watch the machinery — because a date alone only marks the claim as rotting,
    while the command is what lets the next pass settle it in one query.
    """
    ids = re.findall(r"#(\d+)", sentence)
    if not ids or not re.search(_NEVER_RUN, sentence, re.I):
        return False
    dated = re.search(r"\b20\d\d-\d\d-\d\d\b", sentence)
    commanded = any(f"select count(*) from runs where task_id={i}" in sentence
                    for i in ids)
    return not (dated and commanded)


def _watch_section() -> str:
    """§Watch the machinery, up to the next `##` heading, as written (not flat)."""
    after = _text().split("## Watch the machinery", 1)
    assert len(after) == 2, (
        "architecture/autonomy-jobs.md has no §Watch the machinery section — clause 1 "
        "names that section as where #97's record belongs")
    return re.split(r"\n## ", after[1], 1)[0]


def _watch_paragraph(marker: str) -> str:
    """The §Watch the machinery paragraph carrying `marker`, prose flattened."""
    for para in _watch_section().split("\n\n"):
        if marker in _flat(para):
            return _flat(para)
    raise AssertionError(
        f"no §Watch the machinery paragraph contains {marker!r}; the confirmed record "
        "has to sit in the section whose table row it confirms")


def _review_log_entries() -> list[tuple[str, str]]:
    """§Review log as `(first line, flattened body)` per bullet.

    The first line stays a line rather than joining the body because it is what
    carries the entry's date, and clause 2 is about an entry whose *own line* begins
    with that date — flattening it away is what let the 2026-10-12 count survive as
    "the day this was written".
    """
    entries: list[str] = []
    for line in _review_log().splitlines():
        if re.match(r"^\s*[-*] ", line):
            entries.append(line)
        elif entries:
            entries[-1] += "\n" + line
    return [(e.splitlines()[0], _flat(e)) for e in entries]


def test_no_live_sentence_reports_97_as_never_run_and_the_record_names_its_command():
    """Clause 1. §Watch the machinery said #97 had never run and that its journal
    did not exist on disk, and named the skill and the module's `record()` docstring
    as the sources of its reads and writes — an admission, it turned out, that the
    row had been written before the job had anything to be read against.

    Both claims are now retired from live prose, and the replacement paragraph has to
    be the thing that can be re-checked: the command, the store it runs over, the
    journal, the first run's UTC timestamp, and a count written as a dated one. The
    count itself is asserted as a *shape* and never as a number — #97 fires daily at
    ~08:0xZ, so the figure measured on 2026-10-16 is one run stale the next morning,
    and freezing it would rebuild the exact rot this item exists to close.

    #2833 is that rot arriving inside the guard rather than the prose: `cffd22e6`
    redated the count to `2026-10-16T10:00Z` and the shape, whose `\b` landed between
    the day and the `T`, saw no date at all. The shape now takes the optional
    `THH:MMZ` tail — a count dated to the hour is dated — and keeps refusing a count
    with no date, which the two probes below are what pin.
    """
    offending = [s for s in _live_sentences()
                 if re.search(r"#97\b", s) and re.search(_NEVER_RUN, s, re.I)]
    assert not offending, (
        f"live prose reports #97 as never having run again: {offending} — its first run "
        f"was 2026-10-13T08:06:10Z, so the claim is false today and the only place it "
        f"may survive is a dated §Review log entry")

    missing = [s for s in _live_sentences()
               if "landed_outcomes.jsonl" in s
               and re.search(r"\b(?:does|did) not exist\b|\bis absent\b", s, re.I)]
    assert not missing, (
        f"live prose still reports the landed-outcomes journal as absent from disk: "
        f"{missing} — #97's `--record` run appends a row to it every night")

    para = _watch_paragraph("select count(*) from runs where task_id=97")
    assert "~/lloyd-data/workers.db" in para, (
        "the confirmed record does not name the store its own command reads, so a "
        "reader who finds the count stale has nowhere to re-derive it from")
    assert "~/.local/state/lloyd-automod/landed_outcomes.jsonl" in para, (
        "the confirmed record does not name the journal it says holds a row per run")
    # The probe pair first, which is this file's convention for a shape rail: a shape
    # that cannot fail is decorative, and #2833 is the same defect pointing the other
    # way — the old shape could not see the timestamped spelling at all, so on the
    # sentence the doc actually holds it could only ever fail.
    assert DATED_RUN_COUNT.search("returned 4 runs as of 2026-10-16T10:00Z, all"), (
        "the shape cannot see a count dated down to the hour, which is the spelling "
        "the #97 record has used since cffd22e6 redated it — a rail that rejects a "
        "MORE precise date than the one it was written for is a false alarm, and "
        "false alarms are what made main red")
    assert not DATED_RUN_COUNT.search("returned 4 runs, all status=success"), (
        "the shape accepts a run count with no date beside it, which is the one "
        "thing this assertion exists to refuse")
    assert DATED_RUN_COUNT.search(para), (
        "the run count is not written as a dated count ('N runs as of YYYY-MM-DD', "
        "or with its 'THH:MMZ' tail); a bare number is false the morning after the "
        "job next fires")
    assert re.search(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d", para), (
        "the record carries no first run's UTC timestamp, so the dated count has no "
        "starting point and no way to tell a fresh series from a resumed one")


def test_the_superseded_2026_10_12_97_claim_survives_only_in_a_dated_review_log_entry():
    """Clause 2, both directions. The retired statement was a true measurement of
    2026-10-12, so clause 2 moves it into §Review log rather than deleting it — and
    the 2026-10-12 entry that carried the nearest thing to it said only "was 0 the
    day this was written", a relative date, which is how a superseded claim comes to
    read as a current one.

    So the claim must be gone from live prose, present in the log, and held by an
    entry that opens with its own absolute date.
    """
    live_hits = [m.group(0) for m in
                 (re.search(p, _flat(_live_prose()), re.I) for p in _SUPERSEDED_97_CLAIM)
                 if m]
    assert not live_hits, (
        f"live prose carries the retired 2026-10-12 statement again ({live_hits}); the "
        "artefact contradicts it, and only a dated §Review log entry may hold it")

    entries = _review_log_entries()
    holding = [(first, body) for first, body in entries
               if any(re.search(p, body, re.I) for p in _SUPERSEDED_97_CLAIM)]
    assert holding, (
        "the retired 2026-10-12 statement is nowhere in §Review log: clause 2 says it "
        "moves there, not that it is deleted — the 2026-10-12 pass recorded a real "
        "measurement, and a reader of that pass needs to see what held on its day")
    for first, _body in holding:
        assert re.match(r"^\s*[-*]\s*\*\*\s*20\d\d-\d\d-\d\d", first), (
            f"an entry holding the retired claim does not open with its own date: "
            f"{first!r} — 'the day this was written' is not a date, which is the "
            f"convention this item is enforcing")


def test_the_owed_sentence_kept_the_three_open_items_and_dropped_97():
    """Clause 3. The §Review log's "What is still owed on this document" sentence
    closed with #97's row "above that cannot be confirmed until its job has run".
    The journal is the confirmation, so that half comes out — while #2631's
    reconciler, #2632's stale `model:` and #2633's citation form all stay named,
    because none of the three has landed and an owed sentence that loses an item on
    the way to editing itself is how an open defect becomes invisible.
    """
    m = re.search(r"What is still owed on this document:(.*?)\.",
                  _flat(_review_log()), re.S)
    assert m, (
        "§Review log no longer carries a 'What is still owed on this document' "
        "sentence, so there is nothing left to check the owed list against")
    owed = m.group(1)
    assert "cannot be confirmed" not in owed, (
        f"the owed sentence still holds the 'cannot be confirmed until its job has "
        f"run' half: {owed.strip()!r}")
    assert "#97" not in owed, (
        f"the owed sentence still names #97: {owed.strip()!r} — its reads and writes "
        f"are confirmed against the journal now, which is the whole of clause 3")
    for still_open in ("#2631", "#2632", "#2633"):
        assert still_open in owed, (
            f"the owed sentence dropped {still_open}, which no round has landed — "
            f"clause 3 keeps the three open items named: {owed.strip()!r}")


def test_a_live_never_run_claim_about_a_job_id_carries_its_date_and_its_command():
    """Clause 4, the general rail. §Canonicalize says of #98 that the repair script
    "has been able to repair them since 2026-10-13 and has never run" — true, with
    `count(*) where task_id=98` = 0 measured on 2026-10-16, and true-but-undated for
    the never-run half: the 2026-10-13 date belongs to the capability, not to the
    count, and no command was named to re-derive it.

    That is the shape this node forbids everywhere in live prose, which is why
    #98's sentence had to acquire both. #99 also has 0 runs and states no never-run
    claim, so it is untouched — the rail is on the sentence form, not on any
    particular job, and it is checked against a probe as well as against the doc so
    that a predicate which stopped matching anything cannot pass as clean prose.
    """
    assert _undated_never_run("#42 has never run, because the box was rebuilt."), (
        "the predicate no longer flags an undated never-run claim, so this node "
        "cannot fail and the rail is decorative")
    assert not _undated_never_run(
        "#42 has never run as of 2026-10-16 "
        "(`select count(*) from runs where task_id=42` over `~/lloyd-data/workers.db` "
        "was 0 that day)."), (
        "the predicate rejects a never-run claim that does carry its date and its "
        "count command, which is the form clause 4 asks for")

    sentences = _live_sentences()
    assert any(re.search(_NEVER_RUN, s, re.I) for s in sentences), (
        "no live sentence makes a never-run claim at all, so the scan below is "
        "weighing an empty set — either §Canonicalize lost #98's sentence, in which "
        "case re-read it, or §Watch the machinery has been emptied")

    offenders = [s for s in sentences if _undated_never_run(s)]
    assert not offenders, (
        f"{len(offenders)} live sentence(s) say a job never ran without both a "
        f"measurement date and the command that re-derives the count: {offenders}. "
        f"Write it as '#N has never run as of YYYY-MM-DD (`select count(*) from runs "
        f"where task_id=N` over `~/lloyd-data/workers.db` was 0 that day)' — these "
        f"jobs fire daily, so an undated count is stale before the next reader.")


def test_every_live_link_into_a_job_group_points_at_that_groups_current_slug():
    """#2813 clause 5: a group heading carries its id list, so admitting one job renames
    every anchor that targets it — `watch-the-machinery-…-96-97` became
    `…-96-97-99` — and only §Measure's own row was checked for that.

    `test_the_functions_table_row_and_every_anchor_follow_the_group_heading` is the only
    anchor rail the file had, and it searches `](#measure…)`: Measure's row, Measure's
    bullet, Measure's slug. Every other group's anchors were unchecked, so admitting #99
    to §Watch the machinery — which renames that heading's slug — could strand the
    §Membership sentence pointing at the pre-#99 anchor with nothing on it. Two shapes
    to fail on: a slug no heading has at all, which is exactly what a rename leaves
    behind once the old slug is retired, and a link whose label names one group while
    its anchor resolves to another group's heading — both headings exist there, so no
    heading-by-heading comparison sees it either. Live prose only: §Review log keeps the
    anchor as it stood
    the day its entry was written, and rewriting the 2026-10-12 entry to a nine-day-
    newer slug would make that entry claim the document said something it did not say
    that day.
    """
    group_names = _group_names()
    # Every heading level, because a live link can target a `###` section that is not a
    # job group at all — §The wiring is one — and calling that a dead click would make
    # the node red on a link that resolves fine.
    headings = re.findall(r"^#{2,4}\s+(.+?)\s*$", _text(), re.M)
    slug_of = {h: _slug(h) for h in headings}
    by_slug = {slug_of[h]: h for h in headings
               if any(h.startswith(g + ":") for g in group_names)}
    assert len(by_slug) == len(group_names), (
        f"{len(group_names)} groups in §The functions' table, {len(by_slug)} headings "
        f"matching them — a group whose heading cannot be found would leave the checks "
        f"below weighing an empty set")
    group_slugs = set(by_slug)

    live = "\n".join(ln for ln in _live_prose().splitlines()
                      if not ln.lstrip().startswith("|"))
    anchors = set(re.findall(r"\]\(#([a-z0-9][a-z0-9-]*)\)", live))

    # The link's own text is read with its anchor, because that is the only way to state
    # this drift at all: `by_slug` is keyed by `slug_of[h]` for exactly the headings it
    # stores, so an anchor that resolves to a group heading is by construction that
    # heading's current slug, and comparing the two could never fire. What can fire is a
    # sentence naming one group and pointing at another's heading: both headings exist,
    # so no heading-by-heading check can see it.
    owner_of = {slug: next(g for g in group_names if h.startswith(g + ":"))
                for slug, h in by_slug.items()}
    mispointed = [{"label": label, "anchor": anchor, "resolves_to": owner_of[anchor]}
                  for label, anchor in re.findall(
                      r"\[([^\]]+)\]\(#([a-z0-9][a-z0-9-]*)\)", live)
                  if label.strip() in group_names and anchor in owner_of
                  and owner_of[anchor] != label.strip()]
    assert not mispointed, (
        f"live prose labels a link after one job group and points it at another group's "
        f"heading: {mispointed}")

    # The coupling clause 5 exists for shows up below, as an orphan: a group heading
    # carries its id list in its slug, so admitting one job retires the old slug and a
    # prose link still carrying it targets nothing.

    orphans = sorted(a for a in anchors - group_slugs if a not in set(slug_of.values()))
    assert not orphans, (
        f"live prose targets anchors no heading in the document has: {orphans}. A group "
        f"slug that outlives its own rename is a dead click, and it survives precisely "
        f"because every heading-by-heading check finds its heading intact")


def test_99_sits_in_the_tier_its_output_belongs_to():
    """#2813 clause 4. #99 writes nothing into the vault and touches no lockfile: the
    scan is documented `READ-ONLY / NO NETWORK`, runs offline, and its own task file
    forbids it from calling `Write` or `Edit` at all. The case for the writer tier is
    its second output — it files a backlog item when either headline count moves —
    which is exactly the reason #92 and #93 sit there rather than with the reporters.

    Measured, not asserted from the prose: the row it landed in, its count cell, and
    the `WATCH_TIER` ruling this file carries all have to say the same thing.
    """
    writes = "Writes durable state unattended"
    assert WATCH_TIER["#99"] == writes, (
        "a job that files a backlog item is not in the writer tier, which is the one "
        "question this table asks")
    tiers = _table(TIERS_HEADER)
    rows = [row for row, (ids, _c) in tiers.items() if "#99" in ids]
    assert rows == [writes], (
        f"#99's tier row is {rows}, not the single row `WATCH_TIER` rules for it")
    ids, count = tiers[writes]
    assert count is not None and int(count) == len(ids), (
        f"the writer tier counts {count} against {len(ids)} ids: its cell did not "
        f"move with #99, or #99 is listed twice")


#: A clause of job 96's row that ties the report's calendar to the local zone: a
#: word naming the key/stamp/filename, a copula or naming verb within 60 characters,
#: and `local` within another 60, all inside one sentence-bounded clause. Matched
#: with markdown emphasis stripped, so a regression cannot return by rewording or by
#: hiding inside `*local*`, and a correct sentence cannot be missed because an author
#: italicised a word in it. The review of SM_20261016_104300 sent back a ban written
#: as three verbatim sentences, which bans typography rather than a claim — this is
#: the shape those sentences were examples of.
NAME_PRIOR_LOCAL_STAMP_CLAIM = re.compile(
    r"\b(date key|date|stamp|filename|file)\b[^.;]{0,60}"
    r"\b(is|are|was|were|keeps?|keys?|names?|derives?|taken|built)\b[^.;]{0,60}\blocal\b")

#: … and the words that mark such a clause as HISTORY of the bug rather than a claim
#: about the present — what lets the row keep explaining why the artefact on disk
#: carries the wrong name after the code stopped producing it.
NAME_PRIOR_LOCAL_STAMP_HISTORY = re.compile(
    r"\bused to\b|\bprevious(?:ly)?\b|before #2820|until #2820")


def test_the_name_prior_report_row_states_a_utc_date_key_and_keeps_its_four_tokens():
    """#2820 clause 5: §Watch the machinery stops calling the date key local.

    Two halves in one node, because the row is both the fleet's only prose about where
    this job's artefact goes and the thing `WATCH_JOBS[96]` grades, and a rewrite could
    plausibly satisfy either half while breaking the other.

    Measured before this node existed, at the base it runs on: the row read "the stamp
    is the probe's *local* date (…`when.date()`), so on this box the file is named one
    UTC day behind its own run row and the weekly series does not join to `runs` on
    date". After #2820 that is a description of a bug the code no longer has — and the
    example is still on disk, because `name_prior_2026-10-11.json` stays mis-named (a
    round must not mutate `~/lloyd-data`), so the row now states the UTC key and keeps
    the superseded local key as history beside the filename that history produced.

    The four tokens are what `WATCH_JOBS[96]` pins for every watch job and
    `test_each_watch_entry_states_its_schedule_its_reads_its_writes_and_a_prohibition`
    grades. Restating them here is deliberate: that grader is parametrised over all
    eight watch jobs, so a rewrite of THIS row that quietly dropped a token would fail
    there under a node named for no item, and this node is where a reader of #2820
    would look.
    """
    row = _watch_row("#96")

    # Half one, the ban: no clause of this row may claim the calendar is local…
    for clause in re.split(r"[.;]", row):
        if NAME_PRIOR_LOCAL_STAMP_CLAIM.search(clause.replace("*", "")):
            assert NAME_PRIOR_LOCAL_STAMP_HISTORY.search(clause), (
                "the #96 row says the report's date key is on the local calendar, the "
                f"claim #2820 removed: {clause.strip()[:160]!r}")
    # …and, positively, that the ban cannot be satisfied by a row that simply stopped
    # talking about the calendar.
    assert re.search(r"date key[^.;]{0,120}\bUTC\b", row), (
        "the #96 row never says which calendar the date key — the report's filename "
        "— is the run's day in; refusing the local claim is one half of clause 5, "
        "naming the UTC one is the other")
    assert "joins to `runs` on date" in row, (
        "the row does not state the property the UTC key exists for. The pre-#2820 "
        "wording was the negative ('does not join to'), which is exactly the claim "
        "this node is here to fail on, so the positive is what has to be present")
    assert "used to" in row, (
        "the row dropped the superseded local key without keeping it as history, and "
        "then `name_prior_2026-10-11.json` — mis-named, and staying so — is an "
        "artefact the prose no longer explains")
    assert "artefact can read as missing" in row, (
        "the row lost the consequence that makes the off-by-one worth reading about: a "
        "run whose report is named for the day before looks like a run that wrote "
        "nothing, which is how the 2026-10-13 owed-check pass came to summarise a "
        "directory of two runs as one report")
    assert "eval/djev_name_prior_probe.py:" not in row, (
        "the row cites the probe by `path:line` again. #2820 inserts a helper into the "
        "file above the line `cffd22e6` cited, which is how that citation became a "
        "false claim about the code while its sentence stayed true; name `build_report` "
        "and `_now()` by symbol instead")

    # Half two: the schedule cell and the three tokens, exactly as the shared grader
    # reads them — the Freq cell equal, the rest substrings of the row.
    freq, reads, writes, prohibition = WATCH_JOBS[96]
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[1] == freq, (
        f"the #96 row's Freq cell reads {cells[1]!r}, not {freq!r} — the rewrite that "
        f"fixed the calendar claim also moved the schedule the task file declares")
    for want in (reads, writes, prohibition):
        assert want in row, (
            f"the #96 row no longer contains {want!r}, one of the four tokens "
            "`WATCH_JOBS[96]` pins and the per-job doc-claim grader requires")
