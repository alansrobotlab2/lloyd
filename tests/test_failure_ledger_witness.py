"""The vault witnesses in `backlog/data/`, checked for real.

Five witnesses and one retired path. #2079 clause 6 quotes figures whose bytes are
tracked here as 2 dated rows; #2175 clause 5 asks that the twelve `vault_land` rows
behind its count stay re-derivable from committed bytes rather than out of a ledger
another job appends to; #2228 clause 5 asks that the gate report of the round that item
was resumed from be re-derivable the same way, and with `gate.json` — a path already
holding another item's witness, so not a free name — left byte-for-byte as it was; and
#2383 owed two, a 25-row extract behind its `3-in-25` baseline and the gate report of
the round it was resumed from, both of which its owed entry told it to write ONTO
retired or taken paths. Following that wording is what left main red and got #2387
filed, so the two #2383 witnesses are the ones this file exists to have warned about.
All five are checked here against the file on disk, not a fixture built here. The
retired path is the promotions-ledger mirror the retention sweep took out of the vault —
#2054's para, #2064's `ffc04ce5` — and no witness may stand on it: #2178 was filed when
#2175's extract did exactly that, and #2387 when #2383's did it again.

A separate file, and deliberately NOT inside `tests/test_failure_ledger.py`: that
module has no collection-time skip, and neither does this one, so every node here
runs in the default `pytest tests/` the gate's `tests` rung executes. A node behind
a module-level skip would be a claim with no witness — which is the exact thing
clause 6 is about.

It ships nine nodes: eight over the vault bytes —
`test_the_witness_is_a_dated_file_not_the_retired_mirror_path`,
`test_the_witness_holds_the_two_live_rows_the_clause_quotes`,
`test_every_witness_row_is_an_error_alert_of_the_quoted_family`,
`test_the_sibling_note_names_the_mirror_it_is_not`,
`test_the_2175_vault_land_extract_is_a_dated_file_too`,
`test_the_2228_gate_witness_is_a_dated_sibling_and_the_mirror_is_untouched`,
`test_the_2383_extract_is_a_dated_file_reproducing_the_unadjudicated_baseline`,
`test_the_2383_gate_witness_is_a_dated_sibling_because_gate_json_was_taken` — plus
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
#: #2228 clause 5's witness: the gate report of the round that item was resumed from,
#: landed at vault `dff182cb` on the dated-sibling route #2239 (`3cee398c`) amended into
#: the clause. Its source is not a git tree, so these bytes are its only history.
WITNESS_2228_NAME = "2026-10-05.2228-gate-witness.json"
#: #2383's baseline extract, re-homed off the retired mirror path by #2387 the same way
#: #2178 re-homed #2175's. #2383's owed entry asked for the extract "committed to
#: `backlog/data/promotions.jsonl`" — a path this file's own rail keeps absent — so the
#: bytes are the ones that were written there and the name is the dated sibling instead.
WITNESS_2383_NAME = "2026-10-08.2383-promotions-witness.jsonl"
#: #2383's second owed witness: the gate report of round `SM_20261008_002738`, the round
#: it was resumed from. The owed entry named `backlog/data/gate.json` for it, a path that
#: already holds #2228's witness — the exact collision #2239 settled by routing that one
#: onto a dated sibling — so this byte copy took the sibling route and `gate.json` was
#: left as it was found.
WITNESS_2383_GATE_NAME = "2026-10-08.2383-SM_20261008_002738.gate-witness.json"


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


def test_the_2228_gate_witness_is_a_dated_sibling_and_the_mirror_is_untouched():
    """#2228 clause 5: the report of the round that item was resumed from is committed
    at a dated name, every figure the item quotes is re-derivable from those bytes, and
    the one witness path it was told to write to still belongs to someone else.

    The clause originally said "copy the rung report to `backlog/data/gate.json`". That
    path is occupied — round `SM_20260930_063800`'s 34-line witness, #1878/#1883's
    rung-0 refusal, vault `cfe9a113` — so obeying it literally would have destroyed
    another item's graded evidence, which is what #2239 was filed and landed
    (`3cee398c`) to settle. #2239 amended the clause to the dated sibling and left the
    mirror alone; the amendment's own verification was a hand-run `wc -l`/`sha1sum`, and
    #2239's round recorded that no test covered it ("No test pins any of this, by
    construction of a `vault` item"). That is the gap this node closes: a clause can rest
    on vault bytes, but nothing stops those bytes drifting or the mirror being clobbered
    again until a node reads them.

    So both sides are read here rather than quoted back. The extract: 60 lines, which is
    the figure the clause quotes; round `SM_20261005_101132` at head `34c6b7ab` on base
    `506e250a`, `ok: false`; five rungs in ladder order with the first four ok and the
    `tests` rung not, its detail carrying both `44 tests skipped` and `limit 40`; and the
    ten `tests_rung_data` figures the item cites — 16109 passed, 10 failed, 0 errors, 1
    xfailed, 44 skipped, 16164 collected, 8 workers, with `pin_findings` naming
    `DASHBOARD_PINS_NOT_EXECUTED`. The mirror: `wc -l` still 34, and still round
    `SM_20260930_063800`'s — the count is taken the way the clause takes it, on newline
    characters, because that mirror's last byte is not a newline and `splitlines()`
    returns 35 for it.
    """
    data = paths.VAULT_ROOT / "backlog" / "data"
    witness = data / WITNESS_2228_NAME
    assert witness.is_file(), (
        f"{WITNESS_2228_NAME} must be committed: #2228 clause 5 asks that the report of "
        "the round it was resumed from be re-derivable from bytes, and the automod state "
        "dir is not a git tree")
    text = witness.read_text(encoding="utf-8")
    # `wc -l` counts newline CHARACTERS, not lines, and the two files here differ in
    # their last byte — the witness ends with one, the mirror does not. Measuring the
    # way the clause measures is the only way its figure means anything.
    assert text.count("\n") == 60, (
        "wc -l on the witness is the 60 the clause quotes, from the 132-line source "
        "report it was extracted from")
    doc = json.loads(text)
    assert doc["round_id"] == "SM_20261005_101132"
    assert doc["head"].startswith("34c6b7ab") and doc["base"].startswith("506e250a")
    assert doc["ok"] is False, (
        "the report being witnessed is a refusal; a passing copy would erase the "
        "evidence that #2233's ceiling, not this diff, stopped that round")
    rungs = {r["name"]: r for r in doc["rungs"]}
    assert [r["name"] for r in doc["rungs"]] == [
        "preflight", "vet", "static", "frontend", "tests"], (
        "the extract is not every rung of the ladder it claims to witness")
    for name in ("preflight", "vet", "static", "frontend"):
        assert rungs[name]["ok"] is True, name
    assert "2 file(s) in scope" in rungs["preflight"]["detail"]
    assert "260 changed line(s) of 12000" in rungs["vet"]["detail"]
    assert rungs["tests"]["ok"] is False
    assert "44 tests skipped" in rungs["tests"]["detail"] and \
        "limit 40" in rungs["tests"]["detail"], rungs["tests"]["detail"]
    d = doc["tests_rung_data"]
    assert (d["passed"], d["failed"], d["errors"], d["xfailed"],
            d["tests_skipped"], d["collected"], d["workers"]) == (
        16109, 10, 0, 1, 44, 16164, 8), (
        f"the ten figures the item quotes are no longer the committed ones: {d}")
    assert d["pin_findings"] and "DASHBOARD_PINS_NOT_EXECUTED" in d["pin_findings"][0], (
        "the 10 failures are the dashboard pins, and the extract has to say so")

    mirror = data / "gate.json"
    assert mirror.is_file(), "the mirror #2239 refused to clobber has gone missing"
    assert mirror.read_text(encoding="utf-8").count("\n") == 34, (
        "wc -l on backlog/data/gate.json is 34: it is #1883's witness, not this item's, "
        "and a second witness written over it is the hazard #2239 settled")
    assert json.loads(mirror.read_text(encoding="utf-8"))["round_id"] == \
        "SM_20260930_063800", "the mirror no longer holds the witness it is named for"


def test_the_2383_extract_is_a_dated_file_reproducing_the_unadjudicated_baseline():
    """#2387: the extract #2383 owed is committed bytes, reproduces its figure, and
    stands on a dated sibling — not on the retired mirror path #2387 was filed because
    it was written to.

    #2383's owed entry demanded "an extract of it — the subset that reproduces the
    figure the item quotes — committed to `backlog/data/promotions.jsonl`". That path is
    the promotions mirror the retention sweep took out of the vault (#2054's para,
    #2064's `ffc04ce5`), which this file's first node keeps absent and
    `tests/test_diagnosis_fanout_replay.py` keeps absent too, so the clause as written
    could not be obeyed: writing it there is exactly what made those two nodes fail at
    base `f3219070` and got this item filed. The resolution is the one #2178 and #2239
    set: the bytes stay, the name becomes the dated sibling, and a node reads them.

    What makes those bytes worth keeping is that they are a BASELINE, not a snapshot:
    #2383's post-landing owed check counts skips with a failed candidate against
    `3-in-25` measured 2026-10-06..10-08, and clause 3 is judged by whether the derived
    budget ever appeared on a row. Both counts are taken from the live ledger, which
    another job appends to, so without committed bytes the figure moves under the check
    that is supposed to compare against it. Hence the five figures the item quotes are
    re-derived here from the committed rows, and the two zeroes among them are the state
    of the row BEFORE #2383 landed — no row in this window carries a `budget` key or a
    re-ask draw's seconds, which is what its clause 4 and 5 exist to change.
    """
    data = paths.VAULT_ROOT / "backlog" / "data"
    witness = data / WITNESS_2383_NAME
    assert witness.is_file(), (
        f"{WITNESS_2383_NAME} must be committed: #2383's baseline is only re-derivable "
        "if its extract has history, and the automod state dir is not a git tree")
    assert not (data / "promotions.jsonl").exists(), \
        "the extract belongs on the dated sibling; the retired mirror stays absent"

    rows = list(fl.iter_jsonl(witness))
    assert len(rows) == 25, "the extract is 25 rows, one JSON object per line"
    assert witness.read_text(encoding="utf-8").count("\n") == 25, \
        "`wc -l` on the extract is 25, which is the figure #2383 re-derives with"
    # The projection is pinned by its key set rather than by an `event` field: the
    # extract kept only the fields the figures need, and every row in it IS a
    # `vault_land` row by how the window was taken, which the window bound below is the
    # part of that claim a committed byte can still support.
    assert {k for r in rows for k in r} == {"created_at", "item_id", "ok", "guards"}, \
        "the extract is the row's date, item, ok flag and guards projection, nothing else"
    assert {k for g in (r["guards"] for r in rows) for k in g} == {
        "state", "seconds", "candidate_failed", "candidate_seconds",
        "parallel_retry_seconds", "parallel_retry_2_seconds", "budget",
        "reason_prefix"}, "the guards projection the four figures are counted over"
    assert min(r["created_at"] for r in rows) >= "2026-10-06T17:00", \
        "the window #2383 measures starts after #2283's first row with proposed_runs"

    guards = [r["guards"] for r in rows]
    assert sum(1 for g in guards if g["state"] == "skipped") == 8, \
        "8 of the 25 rows are skips: 3 budget-short and 5 that never got a run"
    seen = [r for r, g in zip(rows, guards)
            if g["state"] == "skipped" and (g["candidate_failed"] or 0) > 0]
    assert [(r["created_at"], r["item_id"]) for r in seen] == [
        ("2026-10-06T17:57:25Z", 2304), ("2026-10-07T03:54:53Z", None),
        ("2026-10-07T17:15:16Z", 2367)], \
        "these three are the 3-in-25 the item quotes: a land that went in over a " \
        "failure the probe had already watched"
    assert all(r["ok"] for r in seen), \
        "all three committed anyway — whether a budget-short probe may land is the " \
        "ruling #2383 leaves owed, not a claim this extract makes"

    # The two zeroes are the pre-#2383 shape, and they are why this extract is a
    # baseline rather than a note: read the same keys after #2383 lands and a non-zero
    # is the change working.
    assert sum(1 for g in guards if g["budget"] is not None) == 0, \
        "no row in the window carries the budget the probe was allowed — #2383 clause 4"
    assert sum(1 for g in guards
               if g["parallel_retry_seconds"] is not None
               or g["parallel_retry_2_seconds"] is not None) == 0, \
        "no row carries a re-ask draw's cost either, so no budget could be derived " \
        "from measured cost before #2383 — the input its clause 3 names"

    # The margin that makes the budget the point of the item, not an ornament.
    drawn = [g["candidate_seconds"] for g in guards if g["candidate_seconds"]]
    assert (len(drawn), sum(1 for s in drawn if s >= 235)) == (22, 13), \
        "13 of the 22 candidate-bearing draws measured >= 235s of a 300s probe"
    assert max(drawn) == 292.6, "the slowest draw in the window left 7.4s of the probe"




def test_the_2383_gate_witness_is_a_dated_sibling_because_gate_json_was_taken():
    """#2383 owed a copy of the gate report of the round it was resumed from, and named
    `backlog/data/gate.json` for it — the path #2239 established is not a free name.

    That mirror holds #1883's witness, which the node above this one keeps pinned to
    `SM_20260930_063800`, so writing #2383's report there would have clobbered a
    committed witness to satisfy an owed entry. The bytes went to a dated sibling
    instead, and this node is what makes them more than an unreferenced file in a
    directory: the ladder, the round, the head and the base are read back out of the
    committed bytes, including the failing assertion itself — the two nodes #2387 was
    filed over are diagnosed from this report, and #2387 is the item that made it legal
    for them to be red at all.

    Unlike the extract node, this one asserts nothing about the mirror: that is
    `test_the_2228_gate_witness_is_a_dated_sibling_and_the_mirror_is_untouched`'s job,
    and a second node claiming the same fact is a second place for it to drift.
    """
    gate = paths.VAULT_ROOT / "backlog" / "data" / WITNESS_2383_GATE_NAME
    assert gate.is_file(), (
        f"{WITNESS_2383_GATE_NAME} must be committed: #2383 owed the report of the round "
        "it was resumed from as bytes, and the automod state dir is not a git tree")
    doc = json.loads(gate.read_text(encoding="utf-8"))
    assert doc["round_id"] == "SM_20261008_002738" and doc["ok"] is False, \
        "the committed report is not the refusal #2383 was resumed from"
    assert doc["head"].startswith("9dbcbb8f") and doc["base"].startswith("f3219070"), \
        "the head that failed and the base #2387 names as where its two nodes fail"
    assert [r["name"] for r in doc["rungs"]] == [
        "preflight", "vet", "static", "frontend", "tests"], \
        "the ladder stopped at `tests`, which is where the two red nodes were reported"
    tests = doc["rungs"][4]
    assert tests["ok"] is False and "KeyError: 'baseline'" in tests["detail"], \
        "the report has to still name the assertion that was wrong, or the extract " \
        "documents a failure a later reader cannot check"


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
    assert len(shipped) == 9, (
        "eight witness nodes over the vault bytes plus this rail. The count is spelled "
        "out so that adding a node has to come back here and re-read the "
        "paragraph, instead of leaving it a sentence behind")
