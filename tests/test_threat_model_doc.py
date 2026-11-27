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

Since #4046 the extraction reaches a second doc. `architecture/threat-model.md`
claims in its own not-cover bullet that "Every boundary here overlaps a row of
[[authority-surfaces]]", and that where the two disagree the ladder "wins about
the machine" — a one-directional claim, and the direction that broke was the
unpinned one: four ladder rows named a module that no row of this map's
Boundaries or Threats tables cited at all (`agent_mcp/_path_sandbox.py`,
`app/harness/capabilities.py`, `agent_mcp/_tool_effects.py`,
`workers/sources/arch_review.py`), and a fifth (`agent_mcp/browser.py`) was
credited only by a bare ``browser.py`` span in a control cell — a spelling the
sibling path node cannot grade, because a citation has to carry a directory to be
resolvable. So the ladder's own table is now an input:
`test_every_ladder_row_that_names_a_module_is_a_seam_this_map_cites` derives the
row set from `architecture/authority-surfaces.md` instead of a hand-kept list,
because a second definition of a set is wrong within a week.

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
    the same clause fixes. Since #4046 a third exists and it is a floor on the
    instrument, not on the doc: how many ladder rows must name a module before
    the overlap node counts as having read the ladder at all. The ladder's row
    count itself is never a constant here — the node derives it from
    `architecture/authority-surfaces.md`, which is the point of #4046.
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
# #4046 — the ladder, and the two docs' overlap
# --------------------------------------------------------------------------- #

#: The second doc, and the set this map claims to rank. Read from disk on every
#: call like `_text()`, so a gate running in a worktree grades the worktree's
#: copy of both and never the live checkout's.
LADDER_DOC = ROOT / "architecture" / "authority-surfaces.md"

#: The ladder's own heading, matched as a prefix the way `_section` allows: the
#: real heading is "The ladder, in the order an action meets it", and the comma
#: clause is prose that a rewriter may edit without moving the table.
LADDER_HEADING = "The ladder"

#: How many ladder rows have to name a `.py` module for the overlap node to count
#: as having read the ladder. Fifteen of fifteen do since #4286; the floor is here so
#: the node cannot go green over an empty intersection if the Module column ever
#: turns to prose, which is the failure mode this file exists to refuse.
MIN_LADDER_ROWS_WITH_A_MODULE = 10

#: The four surfaces #4046 put on the map. Each has to be answered by a threat
#: row naming its decider, not merely mentioned in a cell: a seam with nothing
#: over it is a seam nothing prioritises, which is the intro's whole argument for
#: drawing the map before ranking the ladder.
SURFACES_WITH_A_DECIDER_TO_MODEL = (
    "agent_mcp/_path_sandbox.py",      # ladder row 5 — the kernel's bind
    "app/harness/capabilities.py",     # ladder row 7 — the capability ceiling
    "agent_mcp/_tool_effects.py",      # ladder row 9 — the effect ledger
    "workers/sources/arch_review.py",  # ladder row 14 — a job-scoped deny
)

#: A ladder row is numbered, not id-prefixed, so `_ROW` cannot see it:
#: `| 7 | Surface | Module | Question it answers |`.
_LADDER_ROW = re.compile(r"^\|\s*(?P<n>\d{1,2})\s*\|\s*(?P<rest>.*?)\s*\|\s*$",
                         re.MULTILINE)

#: A module named inside a Module cell. The directory is required, so row 1's
#: vault path (`~/obsidian/lloyd/SOUL.md`) and the prose pointer row 15 keeps beside
#: its modules (`architecture/vault-protection.md` §2) do not match, and neither does a bare
#: `safety.py` in prose — the rule is about the module a row *names*.
_LADDER_MODULE = re.compile(
    r"[A-Za-z0-9_.\-]*(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.py")

_NUMBER = (r"\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
           r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
           r"|nineteen|twenty")

#: A rung count in prose, in either word order: "fifteen rungs", "rungs: fifteen".
_RUNG_COUNT = re.compile(
    rf"\b(?P<n>{_NUMBER})\s+(?P<unit>rungs?)\b"
    rf"|\b(?P<unit2>rungs?)[\s:—–-]{{1,4}}(?P<n2>{_NUMBER})\b",
    re.IGNORECASE)

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
          "twenty": 20}


def _ladder_text() -> str:
    assert LADDER_DOC.exists(), (
        f"{LADDER_DOC} is not on disk — the map has nothing left to rank")
    return LADDER_DOC.read_text(encoding="utf-8")


def ladder_rows(text: str | None = None) -> list[tuple[int, list[str]]]:
    """`(row number, cells)` for every row of the ladder table. `cells` is
    `[Surface, Module, Question]`. The row numbers are asserted contiguous
    because a table that stopped parsing mid-way would otherwise look like a
    shorter ladder, and a shorter ladder silently excuses rows from the overlap
    check."""
    body = _section(_ladder_text() if text is None else text, LADDER_HEADING)
    out = []
    for m in _LADDER_ROW.finditer(body):
        cells = [c.strip() for c in m.group("rest").split("|")]
        while cells and cells[-1] == "":
            cells.pop()
        assert len(cells) >= 3, (
            f"ladder row {m.group('n')} parsed to {len(cells)} cells; the table is "
            "id, Surface, Module, Question")
        out.append((int(m.group("n")), cells))
    assert out, "the ladder table parsed to zero rows"
    nums = [n for n, _ in out]
    assert nums == list(range(1, len(nums) + 1)), (
        f"ladder rows are {nums}, not a contiguous 1..{len(nums)} — a row went "
        "unread, and an unread row is an unchecked seam")
    return out


def ladder_module_rows(text: str | None = None) -> list[tuple[int, list[str]]]:
    """`(row number, [modules])` for the rows that name at least one `.py` module.

    Rows naming none are skipped rather than failed: a Module cell that points only
    at a prose section (row 15's did, at `architecture/vault-protection.md` §2, until
    #4286 named its deciders) has no module the map could cite, and whether such a
    row owes a seam is the map's seam criterion, published in the doc's not-cover
    bullet, not a drift this extractor can see. No real row exercises the skip
    today; the synthetic ladder in
    `test_the_ladder_overlap_extractor_is_shown_able_to_fail` does. Whether a multi-module row is satisfied
    by one citation is the other edge, and it is answered the same way the map
    is drawn — B16 cites `app/desktop_lease.py` and not `app/routers/desktop.py`,
    so one hit names the row's seam and the row passes.
    """
    rows = [(n, sorted(set(_LADDER_MODULE.findall(cells[1]))))
            for n, cells in ladder_rows(text) if len(cells) > 1]
    named = [(n, mods) for n, mods in rows if mods]
    assert named, (
        "no ladder row names a `.py` module, so the overlap node is reading an "
        "empty set — which is what this assert exists to make loud")
    return named


def ladder_row_count() -> int:
    return len(ladder_rows())


def map_citation_text() -> str:
    """The Boundaries and Threats tables' rows, joined. Section prose is excluded
    on purpose: a module named in a paragraph is not a seam being named, and the
    claim under test is about the map."""
    lines = []
    for heading, prefix in ((BOUNDARIES, "B"), (THREATS, "T-")):
        for m in _ROW.finditer(_section(_text(), heading)):
            if m.group("id").startswith(prefix):
                lines.append(m.group(0))
    assert len(lines) >= MIN_THREATS, (
        f"the two tables parsed to {len(lines)} rows, below the {MIN_THREATS} "
        f"threat floor this file already enforces — the extraction broke, the doc "
        "did not shrink")
    return "\n".join(lines)


def uncited_ladder_rows(ladder_text: str | None = None,
                        map_text: str | None = None) -> list[tuple[int, list[str]]]:
    """Ladder rows whose every named module is absent from the map's tables.
    Takes text so the vacuity node can hand it a ladder of its own."""
    cited = map_citation_text() if map_text is None else map_text
    return [(n, mods) for n, mods in ladder_module_rows(ladder_text)
            if not any(mod in cited for mod in mods)]



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


def test_the_gate_self_edit_row_states_the_control_that_actually_exists():
    """#4045 clause 5: T-24's control column has to describe the check that runs.

    The row cited one gate.py line number and nothing else, which reads as "the gate
    refuses a diff that touches the gate". Nothing on this tree does that:
    `check_scope` puts `scripts/automod/**` in `PROTECTED_GLOBS`, and protected paths
    are permitted by the scope check — what they do is turn on the rollback `drill`
    rung — while `git grep` over `scripts/automod/` and `app/` finds no refusal of a
    self-editing diff at all. So the honest cell names what is really between a round
    and its own gate: the protected-scope classification, the drill rung it switches
    on, the second reader, and a person. A row that claims a refusal nobody wrote is
    the failure mode this whole doc exists to catch — a threat scored as covered
    because the model described a control that does not exist.
    """
    cells = dict(threat_rows())["T-24"]
    control = cells[-1]
    assert cells[0] == "B11", (
        f"T-24 now sits on boundary {cells[0]!r}; the round→`main` seam is B11 and "
        "the control below is that seam's")
    assert "nothing refuses" in control, (
        f"T-24's control is {control!r}: it has to say plainly that no scope check "
        "refuses a diff touching the gate, which is the false implication the row "
        "used to carry")
    for part in ("protected", "drill", "rung_review", "person"):
        assert part in control, (
            f"T-24's control omits {part!r} — the real chain is protected scope -> "
            "the drill rung -> the review rung -> a person, and dropping a link is "
            f"how the row goes back to reading like a refusal: {control!r}")
    assert _CITED.search(control), (
        f"T-24 lost its control citation, which the row-shape rule above requires: "
        f"{control!r}")


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
# #4046 — the overlap, in the direction that broke
# --------------------------------------------------------------------------- #

def test_every_ladder_row_that_names_a_module_is_a_seam_this_map_cites():
    """Clause 1: a rung of the ladder with no decider in this map is a red node.

    The map's not-cover bullet says every boundary here overlaps a row of
    [[authority-surfaces]] and that the ladder wins "about the machine"; four rows
    said otherwise, and nothing read the two docs against each other to notice.
    The row set is derived from the ladder's own table, so the node goes red when
    a ladder row is filed and the map is not revisited — which is the standing
    obligation #4046 is taking on, and the reason the message names the fix rather
    than a count.

    Both edges of the rule are asserted as properties of the table rather than as
    row numbers, because a row inserted tomorrow moves every number after it: at
    least one row names more than one module (row 13 today — B16 cites
    `app/desktop_lease.py` and not `app/routers/desktop.py`, so one hit names the
    row's seam and the row passes). The other edge — a row naming no module is
    skipped rather than failed — is no longer exercised by the real table: #4286
    gave row 15 its modules, so fifteen of fifteen name one, and an assert that
    some row names none would redden on exactly the table the seam criterion asks
    for. That edge is pinned on the synthetic ladder in
    `test_the_ladder_overlap_extractor_is_shown_able_to_fail` instead, and what is
    asserted here is the property that survives either table: a row the extractor
    skipped is one whose Module cell really holds no `.py` module.
    """
    named = ladder_module_rows()
    assert len(named) >= MIN_LADDER_ROWS_WITH_A_MODULE, (
        f"only {len(named)} ladder rows name a module, below the "
        f"{MIN_LADDER_ROWS_WITH_A_MODULE} this node will read — a Module column "
        "that turned to prose would otherwise make the overlap check green over "
        "an empty set")
    assert any(len(mods) > 1 for _, mods in named), (
        "no ladder row names more than one `.py` module, so the one-hit rule the "
        "map depends on is no longer exercised by anything and would silently "
        "start demanding every module a row mentions")
    named_numbers = {n for n, _ in named}
    for n, cells in ladder_rows():
        if n not in named_numbers:
            assert not _LADDER_MODULE.search(cells[1]), (
                f"ladder row {n} names a `.py` module in its Module cell "
                f"({cells[1]!r}) but the extractor skipped it, so the overlap check "
                "below never asks the map about that row")

    uncited = uncited_ladder_rows()
    assert not uncited, (
        f"{len(uncited)} ladder row(s) name a module that no row of this map's "
        f"Boundaries or Threats tables cites: {uncited}. Either name the decider "
        "on the boundary row that owns the crossing (that is what B6, B7, B8 and "
        "B9 do now) or add a boundary row — and if you add one, "
        "`test_no_mapped_seam_is_bare_every_boundary_carries_a_threat_row` wants a "
        "threat over it, because a seam nobody threatens is a seam nobody ranks")


def test_no_mapped_seam_is_bare_every_boundary_carries_a_threat_row():
    """Clause 2: no row of the map may be a caption.

    Already true of B1-B19 at the head #4046 opened on, which is what makes it
    free to pin — and it is the node that stops a next round from adding a seam to
    look complete and nothing over it to make it rankable.
    """
    named = boundaries()
    covered = {cells[0] for _, cells in (threat_rows() + open_rows()) if cells}
    bare = sorted(named - covered, key=lambda b: int(b[1:]))
    assert not bare, (
        f"{bare} are rows of the Boundaries table that no threat and no open hole "
        "is stated over. The Boundaries preamble makes a seam a boundary because "
        "code authorizes a crossing there; if nothing crosses it in the "
        "enumeration, the row is describing the tree without modelling it")

    probe = _rows("| T-96 | B99 | T | a capability comfortably over the length "
                  "this file's shape rule demands for a threat | "
                  "`app/harness/safety.py` |\n", "T-")
    assert probe and probe[0][1][0] == "B99" and "B99" not in named, (
        "the boundary-cell extraction or the boundary set no longer distinguishes "
        "a covered seam from an invented one, so the assert above passes on "
        "anything")


def test_each_surface_4046_added_to_the_map_is_answered_by_a_cited_control():
    """Clause 3: each of the four surfaces is a control somebody credits.

    Naming a module in an authority cell puts it on the map; the overlap node
    checks that much for every row of the ladder. This node is the narrower claim
    #4046 is actually about — that a seam is answered by a decider, which in this
    doc's shape means the module appears in a Threats row's Control cell, where
    the file's `_CITED` rule (and the sibling file's resolution node) can be
    pointed at it. A seam named only in prose is a seam whose threat is still
    scored as covered by whatever the row used to credit.
    """
    rows = threat_rows()
    by_id = dict(rows)
    for mod in SURFACES_WITH_A_DECIDER_TO_MODEL:
        carriers = [rid for rid, cells in rows if cells and mod in cells[-1]]
        assert carriers, (
            f"{mod} is on the map as a decider but no Threats row credits it as a "
            "control, so the threat over that seam is still being answered by the "
            "control that was there before it existed — which is the exact error "
            "#4046 is about: B6 credited 'the deny table, twice' while a second "
            "decision ran on the same call, and B7 credited a Python path check "
            "while the shell's kernel bind was the one that could be off")
        for rid in carriers:
            control = by_id[rid][-1]
            assert control != NO_CONTROL, (
                f"{rid} credits {mod} and also marks itself uncited: an open hole "
                "belongs under its own heading")
            assert _CITED.search(control), (
                f"{rid}'s control cell cites {mod} in passing but carries no "
                f"citation of the shape the rule requires: {control!r}")


def _intro_text() -> str:
    """Everything before the first `## ` — front matter, title, intro.

    Scoped this way on purpose: the Review log is entitled to say "fifteen rows,
    not twelve" about the tree it measured on its own date, and a rule that
    silenced that would be a rule rewriting dated records to match today.
    """
    text = _text()
    m = re.search(r"^## ", text, re.MULTILINE)
    assert m, "the doc has no `## ` heading, so an intro-scoped rule has no edge"
    return text[:m.start()]


def test_the_intro_states_no_rung_count_the_ladder_does_not_agree_with():
    """Clause 4: a stated count in an intro is a number nobody re-runs.

    The pointer half of the fix shipped in `1b704b04`, which replaced "eleven
    rungs — now twelve" with a correct "fifteen" and a link to the doc that counts
    them. A correct number is still a second definition of a set, and the ladder
    grows by a row every time a guard is filed; so the intro states no count and
    the derivation lives here, where it is recomputed on every suite run.
    """
    derived = ladder_row_count()
    intro = _intro_text()
    for hit in _RUNG_COUNT.finditer(intro):
        token = (hit.group("n") or hit.group("n2")).lower()
        stated = int(token) if token.isdigit() else _WORDS[token]
        assert stated == derived, (
            f"the intro states {stated} rungs; `architecture/authority-surfaces.md`"
            f" has {derived} rows. Name no count, or name the one the ladder "
            "derives — the sentence is not the measurement, this node is")

    # Vacuity, in both directions the node could go quietly green: an extractor
    # blind to the shape, and a comparison that can no longer disagree.
    seen = _RUNG_COUNT.search("the ladder is **fifteen rungs** wide")
    assert seen and (seen.group("n") or "").lower() == "fifteen", (
        "the rung-count extractor no longer sees 'fifteen rungs', the exact "
        "wording this round removed, so the loop above is grading an empty set")
    reversed_ = _RUNG_COUNT.search("the ladder is rungs: sixteen wide")
    assert reversed_ and (reversed_.group("n2") or "").lower() == "sixteen", (
        "the extractor misses the reverse order ('rungs: sixteen'), which is how "
        "a reworded intro would hide a count from this node")
    assert _WORDS["sixteen"] != derived, (
        f"the probe count has collided with the derived ladder length {derived}, "
        "so the comparison above can no longer be shown to refuse anything")


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


def test_the_ladder_overlap_extractor_is_shown_able_to_fail():
    """Clause 5: the overlap node can object, on a ladder this file writes.

    Four rows, one per way the extraction could go quietly green: a module the map
    cites, a module it does not, a row naming no module at all (which must be
    skipped, not failed — a dead skip and a dead check look identical from a green
    node), and a row naming two modules where only one is cited (which must pass,
    or the node starts demanding citations the map has never owed).
    """
    fake = """## The ladder, in the order an action meets it

| # | Surface | Module | Question it answers |
|---|---|---|---|
| 1 | A seam | `app/tool_bans.py` | what may a turn call at all |
| 2 | Another seam | `app/harness/no_such_seam_zzz.py` | what else may it call |
| 3 | Prose only | `architecture/vault-protection.md` §2 | what rewrites the vault registration |
| 4 | Two modules | `app/desktop_lease.py`, `app/routers/desktop.py` | may it act on the desktop |
"""
    named = dict(ladder_module_rows(fake))
    assert sorted(named) == [1, 2, 4], (
        f"the ladder extractor returned rows {sorted(named)} for a table holding "
        "three module-naming rows and one prose row: the skip, the numbering or "
        "the module pattern has drifted, and a row the extractor cannot see is a "
        "row it can never fail")
    assert named[4] == ["app/desktop_lease.py", "app/routers/desktop.py"], (
        f"row 4's modules extracted as {named[4]}, so the multi-module case the "
        "one-hit rule exists for is not being exercised")

    cited = ("| B9 | an unattended session → the tool surface | calls | tuples | "
             "`app/tool_bans.py:23` |\n"
             "| B16 | an act → the desktop | a click | a lease | "
             "`app/desktop_lease.py:123` |\n")
    uncited = uncited_ladder_rows(fake, map_text=cited)
    assert [n for n, _ in uncited] == [2], (
        f"against a map citing only B9 and B16 the uncited rows should be exactly "
        f"[2], got {[n for n, _ in uncited]} — either the prose row was demanded a "
        "seam (the skip is dead) or an uncited module read as cited (the check is "
        "dead)")

    real = map_citation_text()
    assert "B19 is the seam every other row assumes away" not in real, (
        "the citation surface grew past the two tables into the Boundaries "
        "section's own prose, where any module named in a paragraph would count "
        "as a seam being named")


# --------------------------------------------------------------------------- #
# 4. the credential-escape rows name the control that acts on the bytes (#4047)
# --------------------------------------------------------------------------- #

#: Both rows are matched on their capability text, not their id or line number:
#: #4047 was filed because line numbers in this doc move under it (`1b704b04`
#: shifted every anchor in it by 6 the day it landed), and an id is only stable
#: until somebody retires a row.
_CREDENTIAL_BODY = "A credential rides out inside a request body the model composed"
_CREDENTIAL_VAULT = "A vault write copies a credential into a file the sync client pushes"

#: Cells of a Threats row, by the column order the table header declares: the row
#: id is returned separately, so `cells` is [B, Class, Capability, Control].
_CAPABILITY, _CONTROL = 2, -1

#: The line count of `architecture/threat-model.md` as of the last edit that moved it.
#: No node in this file quotes a line of the doc any more (the `:166-173` this comment
#: used to name was cited by nothing); the doc is quoted by line number from outside —
#: `scripts/automod/chrome_dist.py`'s docstring cites B11 at `threat-model.md:106` and
#: T-24/T-25 at `:155` and `:156` — so a change to this count means re-checking those.
#: The #4047 fix left the file at 402 lines; #4286 added B20, T-45, a T-45 note and
#: the not-cover bullet's two rulings, for 417.
_DOC_LINES = 417


def _credential_rows(needles, want):
    """The one Threats row whose capability contains `needle`, so a cell can be
    graded without naming a line number that the next arch-review will move."""
    hits = [(rid, cells) for rid, cells in threat_rows()
            if any(n in cells[_CAPABILITY] for n in needles)]
    assert len(hits) == want, (
        f"expected {want} row matching {needles!r}, found {[r for r, _ in hits]}")
    return hits[0]


def test_the_credential_escape_rows_name_the_substitution_and_its_limit():
    """Clauses 1-3: the two rows that model credential escape have to name the
    control that decides which credential bytes reach the wire, and T-20 has to
    state that control's limit in words a reader can act on.

    Both cells cited `app/harness/outbound_content.py:526` — `redact`, whose one
    caller in the tree is `outbound_content.py:604`, the findings scanner — so the
    map pointed a credential-escape reader at a text redactor for ledger excerpts
    that never sees a request, while the control #3846 (`a092f430`, 2026-11-06) put
    *inside* `egress.guard` was named nowhere in the file except the review note
    that deferred this fix. That is the failure mode this file exists for: prose
    carrying a fact about the code that nothing re-derives.

    The limit is the other half of the point, and the reason clause 2 is a test
    rather than a nicer sentence. `substitute_headers`
    (`app/harness/credential_store.py:376`) is that module's only substitution
    function, it is reached from `agent_mcp/http_tools.py:549` alone, and
    `agent_mcp/http_tools.py:576` sends `body.encode()` past it — so this row's
    capability is half covered. A cell that cites the substitution WITHOUT the
    limit reads as closed and nobody goes to check, which is worse than the wrong
    citation it replaces.
    """
    _, t20_cells = _credential_rows([_CREDENTIAL_BODY], 1)
    t20 = t20_cells[_CONTROL]
    assert "agent_mcp/egress.py" in t20, (
        "T-20 names no egress control at all; the substitution #3846 added inside "
        "`guard` is what acts on a credential's bytes")
    assert "placeholder" in t20 and "bound" in t20, (
        f"T-20 names egress without the substitution's mechanism: {t20}")
    assert "enforce" in t20, (
        "T-20 does not say the substitution is armed whether or not "
        "`harness.egress_policy.enforce` is true — the distinction T-19 rests on")
    assert "header" in t20.lower() and "only" in t20.lower(), (
        f"T-20 states the control but not its scope: {t20}")
    assert "unsubstituted" in t20.lower(), (
        "T-20 no longer says a model-composed BODY leaves unsubstituted, which is "
        "the half of this capability that no control closes")
    assert "outbound_content.py:859" in t20, (
        "T-20 stopped citing the module's outbound-payload control")
    assert "**Partial" in t20, (
        "T-20 is no longer marked a partial cell, so a reader takes the citation as "
        "covering the capability the row names")

    _, t08_cells = _credential_rows([_CREDENTIAL_VAULT], 1)
    t08 = t08_cells[_CONTROL]
    assert "agent_mcp/egress.py" in t08, (
        "T-08 credits no egress control, so a marked credential that does leave over "
        "a guarded lane is still credited to the scanner that only recorded it")
    assert "outbound_content.py:859" in t08, (
        "T-08 cites `redact` again, or lost the payload control beside the substitution")

    for rid, cells in ((t20_cells[_CAPABILITY], t20_cells),
                       (t08_cells[_CAPABILITY], t08_cells)):
        assert "outbound_content.py:526" not in cells[_CONTROL], (
            f"{rid[:40]}!r cites the findings-excerpt redactor as a control span again "
            f"rather than naming it in prose: {cells[_CONTROL]}")


def test_the_credential_edits_stayed_inside_their_cells():
    """Clause 5's half that no resolution suite can see: proving a citation POINTS
    at code says nothing about whether the edit that added it stayed inside its
    cell.

    The mechanism is recorded twice in `agent_mcp/http_tools.py`'s own comment
    (`:536` on 2026-10-31, `:549` on 2026-11-06): a witness anchor is graded by
    re-running a `git grep` and comparing the PRINTED line numbers, so one added
    line silently invalidates every citation that quotes a number — the doc's
    lines are quoted from `scripts/automod/chrome_dist.py` (see `_DOC_LINES`). The
    #4047 fix rewrote three cells and left the file at 402 lines; #4286's B20 row
    and rulings moved it to 417.

    The count is a proxy for the one thing the rest of this file cannot see: the
    row count, which the nodes above check against the 25/3 id sets, cannot move
    when a cell grows. So a doc edit that spills a cell over extra lines passes
    every other node here and is caught here or not at all.
    """
    assert len(DOC.read_text(encoding="utf-8").splitlines()) == _DOC_LINES, (
        "the doc's line count moved, so any citation quoting a line number inside it "
        "— `scripts/automod/chrome_dist.py`'s `threat-model.md:106/:155/:156` — now "
        "points somewhere else; re-point those, then update `_DOC_LINES`")
    for rid in ("T-08", "T-20"):
        cells = _credential_rows(
            [_CREDENTIAL_BODY] if rid == "T-20" else [_CREDENTIAL_VAULT], 1)[1]
        assert cells[_CAPABILITY].startswith("A ") and cells[_CONTROL], (
            f"{rid} no longer parses as a four-column row, so the edit spilled out "
            "of its cell")
