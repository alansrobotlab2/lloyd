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

The one external expectation pinned here is the item's own: the bound reads "24
through 86", and it is the only such bound in `architecture/`. That literal moves
when the fleet does — deliberately, because a doc saying "24 through 86" while a
task 87 exists is the defect this file exists to stop.

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


# ── clause 2 — the bound, and nothing else typing the fleet's size ────────────


def test_the_fleet_bound_reads_24_through_86():
    """Clause 2, first half. `:38` is the last hand-typed fleet bound in the doc
    set and it named a ceiling one job below the newest live task.
    """
    assert "the fleet runs 24 through 86" in _text(), (
        "the numbering note still bounds the fleet short of its newest task — the "
        "inference that made #86 invisible to every reader of this doc")


def test_no_second_fleet_bound_appears_anywhere_in_architecture():
    """Clause 2, second half: repairing one bound must not seed others. Every
    `N through M` in `architecture/` is enumerated, so an integer copied into
    `index.md` — whose "each of the 32 scheduled jobs" count rotted inside three
    days — fails here instead of in next month's arch review.
    """
    found = {md.name: [ln.strip() for ln in md.read_text().splitlines()
                       if re.search(r"\b\d+\s+through\s+\d+\b", ln)]
             for md in sorted(ARCH_DIR.glob("*.md"))}
    found = {k: v for k, v in found.items() if v}
    assert found == {"autonomy-jobs.md": [
        "ordered: the fleet runs 24 through 86 with gaps where tasks were retired."]}, (
        f"a fleet bound is typed somewhere it was not before, or reads differently: {found}")
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
