"""The vault witnesses in `backlog/data/`, checked for real.

Two witnesses and one retired path. #2079 clause 6 quotes figures whose bytes are
tracked here as 2 dated rows; #2175 clause 5 asks that the twelve `vault_land` rows
behind its count stay re-derivable from committed bytes rather than out of a ledger
another job appends to. Both are checked here against the file on disk, not a fixture
built here. The retired path is the promotions-ledger mirror the retention sweep took
out of the vault — #2054's para, #2064's `ffc04ce5` — and no witness may stand on it;
#2178 was filed for the moment a second witness did exactly that and left main red.

A separate file, and deliberately NOT inside `tests/test_failure_ledger.py`: that
module has no collection-time skip, and neither does this one, so every node here
runs in the default `pytest tests/` the gate's `tests` rung executes. A node behind
a module-level skip would be a claim with no witness — which is the exact thing
clause 6 is about.

It ships six nodes: five over the vault bytes —
`test_the_witness_is_a_dated_file_not_the_retired_mirror_path`,
`test_the_witness_holds_the_two_live_rows_the_clause_quotes`,
`test_every_witness_row_is_an_error_alert_of_the_quoted_family`,
`test_the_sibling_note_names_the_mirror_it_is_not`,
`test_the_2175_vault_land_extract_is_a_dated_file_too` — plus
`test_the_docstring_names_every_node_this_file_ships`, the rail that re-checks this
paragraph against the module's own bytes on every run.

That rail exists because the review rung's advisory on the previous round was
exactly this paragraph: it said "all three nodes here run" while `pytest -v`
collected four. Retyping the number fixes one drift and leaves the next in place; a
number this file checks against itself is the only version that stays true.

These nodes read `~/obsidian`, so on a machine with no vault they fail loudly
rather than silently passing: the witness is supposed to be there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import failure_ledger as fl
from app import paths

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "failure_ledger"
WITNESS_NAME = "2026-10-02.2079-promotions-witness.jsonl"
NOTE_NAME = "2026-10-02.2079-promotions-witness.md"
#: #2175 clause 5's extract, re-homed off the retired mirror path by #2178. The bytes
#: are the ones that landed at vault `87a7a104`; only the name moved.
WITNESS_2175_NAME = "2026-10-04.2175-vault-land-witness.jsonl"


def witness_rows():
    return list(fl.iter_jsonl(paths.VAULT_ROOT / "backlog" / "data" / WITNESS_NAME))


def test_the_witness_is_a_dated_file_not_the_retired_mirror_path():
    """The 2 live rows live in a dated witness beside every other witness in that
    directory, NOT on the promotions-ledger path the retention sweep retired.

    This node exists because the first draft of this round wrote them to the
    retired path and this repo's own rail refused it —
    `tests/test_automod_vault_round.py::test_no_reader_under_tests_or_scripts_opens_the_retired_mirror`
    — which is that rail doing its job: a 2-row file on a path known as a
    33,113,707-byte wholesale copy reads as the copy coming back. The witness is
    now named for its date, and the retired path stays unopened.
    """
    data = paths.VAULT_ROOT / "backlog" / "data"
    assert (data / WITNESS_NAME).is_file(), f"{WITNESS_NAME} must be committed"
    assert not (data / "promotions.jsonl").exists(), \
        "the retired promotions-mirror path must stay absent"


def test_the_witness_holds_the_two_live_rows_the_clause_quotes():
    """The clause's figure is "reading the live file alone yields 2", and the bytes
    behind it are tracked in the vault.

    Two rows, and they are the same records as the repo's
    `promotions_live_rows.jsonl` fixture field for field: the figure quoted in the
    item and the figure the ingestion tests exercise are then the same two rows,
    which is what stops the prose and the suite drifting apart.
    """
    rows = witness_rows()
    assert len(rows) == 2, "the clause quotes 2; the witness is exactly those rows"
    fixture = list(fl.iter_jsonl(FIXTURES / "promotions_live_rows.jsonl"))
    assert [json.dumps(r, sort_keys=True) for r in rows] == \
        [json.dumps(r, sort_keys=True) for r in fixture]


def test_every_witness_row_is_an_error_alert_of_the_quoted_family():
    """The witness is 2 of one title, not the ledger.

    The source it was extracted from holds 57 `alert` rows at this base across 10
    titles, so a reader who takes these 2 as the whole picture understates the
    guardian's output by 28x. Pinned here because the file on its own does not say
    what it excluded.
    """
    for row in witness_rows():
        assert row["event"] == "alert"
        assert row["level"] in fl.ALERT_LEVELS
        assert row["title"] == "Service down, but no promotion to revert"


def test_the_sibling_note_names_the_mirror_it_is_not():
    """`backlog/data/` already held a promotions-ledger copy once, and #2064 deleted
    33,113,707 bytes of it at vault `ffc04ce5`. The note beside this witness has to
    say in its own text that this is 2 dated rows, not that file reborn, or the
    next reader of the directory cannot tell.

    Asserted against the note's bytes, including the node names it cites: this
    repo has already refused a round for a note that named a test node which did
    not exist.
    """
    note = paths.VAULT_ROOT / "backlog" / "data" / NOTE_NAME
    assert note.is_file(), (
        f"{NOTE_NAME} must sit beside the witness and name the retire it is not")
    text = note.read_text(encoding="utf-8")
    for token in ("ffc04ce5", "4bc93177", "33,113,707", "#2064",
                  '"title": "Service down, but no promotion to revert"',
                  "16 + 2", WITNESS_NAME,
                  "test_the_guardian_family_is_18_occurrences_from_2026_09_06",
                  "test_reading_the_live_ledger_alone_understates_the_family_as_two"):
        assert token in text, f"the note dropped the token {token!r}"
    for node in ("test_the_guardian_family_is_18_occurrences_from_2026_09_06",
                 "test_reading_the_live_ledger_alone_understates_the_family_as_two"):
        assert (ROOT / "tests" / "test_failure_ledger.py").read_text(
            encoding="utf-8").count(f"def {node}(") == 1, f"{node} is not defined once"


def test_the_2175_vault_land_extract_is_a_dated_file_too():
    """#2178: the first node proves the retired path is empty; it cannot prove a
    witness exists, and a red tree was built exactly in that gap.

    #2175 clause 5 told its round to put twelve `vault_land` rows at
    `backlog/data/promotions.jsonl` — a path #2054 had already retired and #2064 had
    emptied at vault `ffc04ce5` — and the round obeyed, so at `4e6bae21` the vault
    held a committed witness and the suite held a failing node at the same time. The
    two items each wrote a clause about the same directory and neither read the
    other's. The bytes now sit at a dated name; this node reads them THERE, which is
    what makes it more than a restatement of the node above: that one passes happily
    on a vault where the extract was deleted outright, and this one does not.

    The figures are re-derived from the committed bytes rather than quoted back at
    the item that landed them: 12 lines, 12 rows, every one a `vault_land` with
    `ok: true` and no `held_by` key — the absence #2175's report is about — and the
    span 2026-10-04T00:51:17Z to 2026-10-04T08:31:40Z, which is the count's own
    timestamp and the reason the extract is cut there instead of at the end of day.
    """
    data = paths.VAULT_ROOT / "backlog" / "data"
    witness = data / WITNESS_2175_NAME
    assert witness.is_file(), (
        f"{WITNESS_2175_NAME} must be committed: #2175 clause 5 asks that the twelve "
        "rows be re-derivable from bytes, and no dated witness was found")
    text = witness.read_text(encoding="utf-8")
    rows = list(fl.iter_jsonl(witness))
    assert len(text.splitlines()) == 12, "wc -l on the extract is the 12 the report quotes"
    assert len(rows) == 12, f"the extract holds {len(rows)} rows, not the 12 counted"
    assert all(r["event"] == "vault_land" for r in rows), \
        "rows that are not vault_land are not the extract #2175 asked for"
    assert all(r["ok"] is True for r in rows), \
        "a refusal is in the extract, so it no longer shows a day with none"
    assert not [r for r in rows if "held_by" in r], \
        "the extract postdates #2175 and cannot show the pre-fix absence"
    stamps = [str(r["created_at"]) for r in rows]
    assert all(s.startswith("2026-10-04") for s in stamps), stamps[0]
    assert min(stamps) == "2026-10-04T00:51:17Z" and \
        max(stamps) == "2026-10-04T08:31:40Z", \
        "the extract is not the twelve rows up to the count's own timestamp"


def test_the_docstring_names_every_node_this_file_ships():
    """A docstring that counts its own nodes has to be checked against the file.

    The previous round's version of the module docstring said "all three nodes",
    `pytest -v` collected four, and nothing in the suite noticed — the count was a
    claim nobody re-ran, which is the same failure class as a note naming a test
    node that does not exist (this repo has already refused a round for one). So the
    list is now read out of this file's own bytes and compared with the names the
    docstring advertises, in both directions: naming a node this file does not ship
    fails, and shipping one the docstring never names fails.

    The node names are matched only inside backticks, because the docstring also
    names the file `tests/test_failure_ledger.py` and a bare word-boundary scan
    reads a node id out of that filename — a failure that would look like drift and
    be nothing of the kind.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    docstring = src.split('"""', 2)[1]
    advertised = set(re.findall(r"`(test_[a-z0-9_]+)`", docstring))
    shipped = set(re.findall(r"^def (test_[a-z0-9_]+)\(", src, re.M))
    assert advertised == shipped, (
        f"the module docstring advertises {sorted(advertised - shipped)} that this "
        f"file does not ship; this file ships {sorted(shipped - advertised)} the "
        "docstring never names")
    assert len(shipped) == 6, (
        "five witness nodes over the vault bytes plus this rail. The count is spelled "
        "out so that adding a sixth node has to come back here and re-read the "
        "paragraph, instead of leaving it a sentence behind")
