"""#2563 pins `architecture/threat-model.md` to its own shape and to its own arithmetic.

The doc is a threat enumeration plus a scored retrospective. Both halves are
tables, and a table is the one artifact form that rots silently: a row deleted
from the threat list leaves a page that still *reads* like a complete model, and
a retrospective whose headline count drifted from its rows keeps reporting a
coverage number nothing produces. So every assertion here extracts from the doc
and compares the extraction to a property of the extraction — never to a reworded
sentence — and each extractor is shown able to fail on text this file writes, in
`test_the_four_extractors_are_shown_able_to_fail`, because a regex that matched
nothing looks exactly like a doc with nothing wrong.

What is deliberately NOT pinned here, and why:

  * **Whether a cited control resolves on disk.** `architecture/threat-model.md`
    joins `NEW_DOCS` in `tests/test_architecture_coverage_doc_claims.py`, whose
    parameterized node reads every code span in the doc against the working tree,
    line anchors included. A second resolution check in this file would be a
    second definition of the same rule, and the two would disagree inside a
    week — the failure `architecture/authority-surfaces.md` §"Where one rule is
    spelled more than once" exists to record. This file asks only whether a
    control CELL has the shape of a citation; the sibling file asks whether that
    citation is true.
  * **Whether a `#NNNN` control's backlog item is still open.** Liveness is
    vault state a person changes by closing things. A suite node that goes red
    because a human closed an item is not pinning the doc, it is holding the
    board still. The doc says so in its own boundary section.
  * **The size of the incident set or the coverage counts.** Both come out of
    the rows. Only the 25-threat floor is a constant, because it is the
    acceptance clause, and the second constant is the two-word miss vocabulary
    the same clause fixes.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "threat-model.md"

#: Acceptance clause 1: a threat table of at least this many rows. Counted over
#: the modelled threats and the open set together, because a threat with no
#: control is still a threat — dropping it from the count would reward a doc that
#: filed only the comfortable ones.
MIN_THREATS = 25

#: The section headings the extraction is scoped to. `_section` refuses to
#: return an empty body, so a renamed heading is a red node naming the heading,
#: not a green node over a table nobody read.
BOUNDARIES = "Boundaries"
THREATS = "Threats"
OPEN_HOLES = "Open holes"
RETRO = "Retrospective: scoring the model against the incident corpus"

#: The only two labels a miss may carry (clause 4). The distinction is the
#: finding: an absent boundary is a wrong map, an absent threat is a thin
#: enumeration over a map that was right.
MISS_KINDS = ("boundary absent", "boundary present, threat not enumerated")

#: A table row with a leading id cell. `T-` modelled threat, `O-` open-set
#: threat, `I-` incident, `B` boundary.
_ROW = re.compile(r"^\|\s*(?P<id>[TOI]-\d{1,3}|B\d{1,2})\s*(?P<rest>\|.*)$",
                  re.MULTILINE)

#: The headline arithmetic, in the words the doc states it in.
_HEADLINE = re.compile(
    r"(\d+)\s+incidents assessed,\s*(\d+)\s+predicted,\s*(\d+)\s+missed")

#: A control cell counts as cited if it carries a code-span path or a backlog id.
#: Deliberately looser than `cited_paths` in the sibling file: shape here, truth
#: there. A bare `safety.py` is not this shape and must not be — a control that
#: does not say which `safety.py` is a control nobody can go read.
_CITED = re.compile(
    r"`[^`]*[A-Za-z0-9_.\-]+/[^`]*`"        # a path with a directory in it
    r"|`[\w.\-]+\.py:\d+`"                  # or a root-level module, anchored
    r"|#\d{3,}")                            # or a board item

#: The marker an open-set row puts in its control cell instead of a citation.
NO_CONTROL = "**none**"


def _text() -> str:
    assert DOC.exists(), f"{DOC} is not on disk"
    return DOC.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Body of one `## <heading>` section, up to the next `## `."""
    m = re.search(rf"^## {re.escape(heading)}.*?$(.*?)(?=^## |\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading}' section — the extraction has nowhere to look"
    return m.group(1)


def _rows(section_body: str, prefix: str) -> list[tuple[str, list[str]]]:
    """`(id, cells)` for every table row in `section_body` whose id starts with
    `prefix`. `cells` excludes the id cell and the trailing empty cell."""
    out = []
    for m in _ROW.finditer(section_body):
        if not m.group("id").startswith(prefix):
            continue
        cells = [c.strip() for c in m.group("rest").split("|")]
        if cells and cells[-1] == "":
            cells.pop()
        while cells and cells[0] == "":
            cells.pop(0)
        out.append((m.group("id"), cells))
    return out


def boundaries() -> set[str]:
    body = _section(_text(), BOUNDARIES)
    ids = {m.group("id") for m in _ROW.finditer(body)
           if m.group("id").startswith("B")}
    assert ids, "the Boundaries section has no `B<n>` rows, so the map is empty"
    return ids


def threat_rows() -> list[tuple[str, list[str]]]:
    body = _section(_text(), THREATS)
    rows = _rows(body, "T-")
    assert rows, "the Threats section parsed to zero rows"
    return rows


def open_rows() -> list[tuple[str, list[str]]]:
    return _rows(_section(_text(), OPEN_HOLES), "O-")


def incident_rows() -> list[tuple[str, list[str]]]:
    body = _section(_text(), RETRO)
    rows = _rows(body, "I-")
    assert rows, "the Retrospective section parsed to zero incident rows"
    return rows


# --------------------------------------------------------------------------- #
# clause 1 — the threat table
# --------------------------------------------------------------------------- #

def test_the_table_enumerates_at_least_twenty_five_threats_over_named_boundaries():
    """Clause 1: >=25 rows, each stating a capability over a NAMED boundary.

    "Named" is the load-bearing word: a threat whose boundary cell is prose is a
    threat nobody can locate in the map, which is how an enumeration quietly
    becomes a list of worries. Every boundary id used by a threat or an open-set
    row therefore has to be a row of the Boundaries table itself.
    """
    named = boundaries()
    modelled, open_set = threat_rows(), open_rows()
    total = modelled + open_set
    assert len(total) >= MIN_THREATS, (
        f"{len(total)} threat rows ({len(modelled)} modelled, {len(open_set)} "
        f"open) against a floor of {MIN_THREATS}")

    for rid, cells in total:
        assert len(cells) >= 4, (
            f"{rid} has {len(cells)} cells; the shape is id, boundary, class, "
            "capability, control, control-cited-ness, and an open row's fifth")
        boundary = cells[0]
        assert boundary in named, (
            f"{rid} states its threat over boundary {boundary!r}, which is not a "
            f"row of the Boundaries table — a threat must sit on a mapped seam")
        capability = cells[2]
        assert len(capability) > 40, (
            f"{rid}'s capability is {len(capability)} chars: a threat has to state "
            "a capability, not name a worry")


@pytest.mark.parametrize(
    "rows", [threat_rows, open_rows], ids=["threats", "open-holes"])
def test_a_capability_row_states_a_capability_and_not_an_occurrence(rows):
    """Clause 1's "stated as capability, not occurrence".

    The distinction the item insists on, and it is mechanically checkable in one
    direction: a row that narrates something having happened is a post-mortem
    wearing a threat id, and a model made of post-mortems scores 100% against its
    own incidents by construction. So no threat row may carry a past-tense
    incident date or the words that introduce one.
    """
    for rid, cells in rows():
        capability = cells[2]
        lowered = capability.lower()
        assert not re.search(r"\b(2026|2025)-\d\d-\d\d\b", capability), (
            f"{rid} carries a date: that is an incident record, not a capability")
        for phrase in ("happened", "once did", "was exploited", "was attacked"):
            assert phrase not in lowered, (
                f"{rid} narrates an occurrence ({phrase!r}) instead of a capability")


def test_every_modelled_threat_carries_a_control_citation_or_a_backlog_id():
    """Clause 1's control half: outside the open set, nothing goes uncited.

    Checked cell by cell rather than row-wide, so a row cannot satisfy it by
    mentioning a path in its capability prose while its control column says
    nothing. The open set is checked from the other side below.
    """
    for rid, cells in threat_rows():
        control = cells[-1]
        assert control != NO_CONTROL, (
            f"{rid} puts the open-set marker in the Threats table: an uncited "
            "threat belongs under the Open holes heading, where it stays visible "
            "as a hole")
        assert _CITED.search(control), (
            f"{rid} has no control: control cell is {control!r}, which is neither "
            "a code-span path nor a #NNNN backlog id")


def test_the_open_set_is_where_the_uncited_threats_live_and_is_not_empty():
    """Clause 1's parenthesis, from both sides.

    An open set that is empty is the suspicious result, not the good one: the
    item's own premise is that the ~40-item security backlog is a set of
    reactions with no model over it, so a model that closes every row it can
    name has modelled the reactions and not the system. And an open-set row that
    DID find a control is a row that should have been in the Threats table, which
    is how a hole stops being visible.
    """
    rows = open_rows()
    assert rows, (
        "the Open holes section has no rows: either every threat on this box has "
        "a named control, which the model's own incidents contradict, or the "
        "section was trimmed and the open set is no longer published")
    for rid, cells in rows:
        assert cells[-1] == NO_CONTROL, (
            f"{rid} is under Open holes but its control cell cites something "
            f"({cells[-1]!r}) — it belongs in the Threats table")
        assert not _CITED.search(cells[-1]), (
            f"{rid} carries a citation in the control cell of the open set")


# --------------------------------------------------------------------------- #
# clause 4 — the retrospective's arithmetic
# --------------------------------------------------------------------------- #

def test_the_retrospective_headline_agrees_with_its_own_rows():
    """Clause 4: predicted + missed == incidents assessed, against the rows.

    Every count is derived from the table; the headline is compared to the
    derivation rather than trusted. This is the whole reason the section is a
    table: prose coverage claims drift from their evidence inside one edit, and
    a number that cannot be recomputed is not a measurement.
    """
    rows = incident_rows()
    assessed = {rid for rid, _ in rows}
    assert len(assessed) == len(rows), "an incident id appears twice"

    verdicts = [cells[-3].lower() for _, cells in rows]
    predicted = sum(1 for v in verdicts if v == "predicted")
    missed = sum(1 for v in verdicts if v == "missed")
    assert predicted + missed == len(rows), (
        f"{predicted} predicted + {missed} missed != {len(rows)} rows — every "
        "incident row's verdict cell must be exactly one of those two words")

    m = _HEADLINE.search(_text())
    assert m, ("no '<N> incidents assessed, <M> predicted, <K> missed' line in the "
               "doc: the headline must be re-derivable, not just stated")
    n, p, k = (int(g) for g in m.groups())
    assert (n, p, k) == (len(rows), predicted, missed), (
        f"headline says {n}/{p}/{k}, the rows give "
        f"{len(rows)}/{predicted}/{missed}")
    assert p + k == n


def test_every_miss_is_classified_as_absent_boundary_or_absent_threat():
    """Clause 4's second half: the miss split is the finding, so it is closed.

    Exactly two labels are legal, and only the missed rows carry one: an open
    vocabulary here is how every awkward miss eventually becomes "other".
    """
    rows = incident_rows()
    seen: collections.Counter = collections.Counter()
    for rid, cells in rows:
        verdict, kind = cells[-3].lower(), cells[-2]
        if verdict == "predicted":
            assert kind in {"—", "-", "n/a"}, (
                f"{rid} is predicted but carries a miss class ({kind!r})")
            continue
        assert kind in MISS_KINDS, (
            f"{rid} is missed with class {kind!r}; the legal labels are exactly "
            f"{MISS_KINDS}")
        seen[kind] += 1
    assert sum(seen.values()) > 0, (
        "zero misses across the whole incident set: the model predicted "
        "everything, which is what a model scored against the incidents that "
        "taught it would look like — the doc's own caveat section says why that "
        "is not a believable result")


def test_each_incident_row_names_a_source_a_reader_can_go_and_read():
    """The incident set has to be an explicit set, or "N incidents" is a claim.

    Shape only: each row carries a code span naming where the writeup is.
    `tests/test_architecture_coverage_doc_claims.py` is what proves the span
    resolves, for this doc and for every other one in the corpus.
    """
    for rid, cells in incident_rows():
        source = cells[-1]
        assert "`" in source and "/" in source, (
            f"{rid} names no source path: the incident it scores is unlocatable")


# --------------------------------------------------------------------------- #
# vacuity controls
# --------------------------------------------------------------------------- #

def test_the_four_extractors_are_shown_able_to_fail():
    """Each extractor is run against text this file writes, and must object.

    Four failures, one per extraction that would otherwise pass quietly: a
    threat row with an uncited control, an uncited row outside the open-set
    section, a miss with an off-vocabulary class, and a headline that no longer
    matches its rows.
    """
    good_threat = "| T-99 | B4 | E | a turn can do a thing that takes a while to say | `app/harness/safety.py` |\n"
    assert _CITED.search(_rows(good_threat, "T-")[0][1][-1]), (
        "the citation shape no longer accepts a real control, so "
        "test_every_modelled_threat_carries_a_control_citation_or_a_backlog_id "
        "passes on any text at all")

    uncited = "| T-98 | B4 | E | a turn can do a thing that takes a while to say | nothing |\n"
    assert not _CITED.search(_rows(uncited, "T-")[0][1][-1]), (
        "an uncited control cell now reads as cited")

    bad_class = "| I-99 | x | B4 | missed | guard was sad | added after |\n"
    rid, cells = _rows(bad_class, "I-")[0]
    assert cells[-2] not in MISS_KINDS, (
        "the miss vocabulary check would accept an invented class")

    rows = "| I-98 | x | B4 | predicted | — | added after |\n" \
           "| I-99 | x | B4 | missed | boundary absent | added after |\n"
    verdicts = [c[-3].lower() for _, c in _rows(rows, "I-")]
    derived = (len(verdicts), verdicts.count("predicted"), verdicts.count("missed"))
    assert derived == (2, 1, 1)
    m = _HEADLINE.search("**9 incidents assessed, 8 predicted, 1 missed.**")
    assert m and (int(m.group(1)), int(m.group(2)), int(m.group(3))) != derived, (
        "the headline extractor or the comparison is broken: a mismatched "
        "headline would have passed")

    # And the boundary map: a threat naming an unmapped seam has to be visible as
    # one, which is the whole check in the first node.
    assert "B99" not in boundaries()


def test_the_doc_declares_the_two_things_it_cannot_promise(tmp_path):
    """The honesty clause, pinned as a section rather than hoped for.

    A threat model scored against the incidents that informed it reports a
    coverage number inflated by its own authorship — a control that exists
    *because* an incident produced it predicts that incident by construction.
    The doc therefore has to carry, in its own words, both the per-incident
    statement of whether the control predates the damage and the statement that
    the marginal-value arm is not measured here. Asserted as headings, not as
    sentiment, because a paragraph of hedging is not a claim anyone can audit.
    """
    text = _text().lower()
    for needle in ("control predates the damage?",
                   "not measured in this round"):
        assert needle in text, (
            f"the retrospective no longer carries {needle!r}: without it the "
            "coverage number is a number with no stated inflation")
