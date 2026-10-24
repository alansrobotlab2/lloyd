"""The vault witnesses in `backlog/data/`, checked for real.

Six witnesses and one retired path. #2079 clause 6 quotes figures whose bytes are
tracked here as 2 dated rows; #2175 clause 5 asks that the twelve `vault_land` rows
behind its count stay re-derivable from committed bytes rather than out of a ledger
another job appends to; #2228 clause 5 asks that the gate report of the round that item
was resumed from be re-derivable the same way, and with `gate.json` — a path already
holding another item's witness, so not a free name — left byte-for-byte as it was;
#2383 owed two, a 25-row extract behind its `3-in-25` baseline and the gate report of
the round it was resumed from, both of which its owed entry told it to write ONTO
retired or taken paths; and #2523 owes the 221 rows behind #2385's red-tree figure — the
implement runs that finished with no round at all — whose owed entry named the SAME
retired mirror path #2387 was filed over, so those bytes are committed at a dated
sibling and re-derived from there, never from the store #1975 folds. Following that
wording is what left main red and got #2387 filed, so the two #2383 witnesses are the
ones this file exists to have warned about.
All six are checked here against the file on disk, not a fixture built here. The
retired path is the promotions-ledger mirror the retention sweep took out of the vault —
#2054's para, #2064's `ffc04ce5` — and no witness may stand on it: #2178 was filed when
#2175's extract did exactly that, and #2387 when #2383's did it again.

A separate file, and deliberately NOT inside `tests/test_failure_ledger.py`: that
module has no collection-time skip, and neither does this one, so every node here
runs in the default `pytest tests/` the gate's `tests` rung executes. A node behind
a module-level skip would be a claim with no witness — which is the exact thing
clause 6 is about.

It ships fourteen nodes: thirteen that read a vault tree through git — the live
vault's for twelve of them, and one throwaway vault this file builds for the #3272 pair —
`test_the_witness_is_a_dated_file_not_the_retired_mirror_path`,
`test_the_witness_holds_the_two_live_rows_the_clause_quotes`,
`test_every_witness_row_is_an_error_alert_of_the_quoted_family`,
`test_the_sibling_note_names_the_mirror_it_is_not`,
`test_the_2175_vault_land_extract_is_a_dated_file_too`,
`test_the_2228_gate_witness_is_a_dated_sibling_and_the_mirror_is_untouched`,
`test_the_2383_extract_is_a_dated_file_reproducing_the_unadjudicated_baseline`,
`test_the_2383_gate_witness_is_a_dated_sibling_because_gate_json_was_taken`,
`test_the_2523_red_tree_extract_reproduces_the_two_roundless_items`,
`test_the_2444_promotions_extract_reproduces_the_at_the_max_ratio`,
`test_every_dated_promotions_extract_is_committed_and_parses`,
`test_the_promotions_directory_check_passes_on_an_extract_that_is_not_committed_yet` and
`test_the_2759_promotions_extract_reproduces_its_published_lag_percentiles` — plus
`test_the_docstring_names_every_node_this_file_ships`, the rail that re-checks this
paragraph against the module's own bytes on every run.

That rail exists because the review rung's advisory on the previous round was
exactly this paragraph: it said "all three nodes here run" while `pytest -v`
collected four. Retyping the number fixes one drift and leaves the next in place; a
number this file checks against itself is the only version that stays true.

`test_the_2444_promotions_extract_reproduces_the_at_the_max_ratio` — the seventh
witness, and the one whose demand was the last to name the retired slot. #2444 owed 1
asked its round for an extract of the live promotions ledger committed to
`backlog/data/promotions.jsonl` and re-derived from there; #2560 published it as a dated
sibling in this directory instead — the name is discovered by glob there rather than
spelled, because the middle term of it is the state directory that
`test_the_2523_red_tree_extract_reproduces_the_two_roundless_items` rightly refuses to
let these bytes name — so its figures come from committed bytes and no node here
re-reads the store it was cut from: 107 of 110 stored `review_confirm_votes[].reason`
strings sit at `CONFIRM_REASON_MAX` = 300 characters — 97.27% — over 73 rows, plus the 9
`arch_review` rows, all with `detail: null`, that share the window; 82 rows and 870,849
bytes in all, under the 1,048,576-byte ceiling the demand is written under. `tests/
test_witness_artifact_clause.py` is where the emitter is pinned to stop naming that
retired slot at all.

These nodes read `~/obsidian`, so on a machine with no vault they fail loudly
rather than silently passing: the witness is supposed to be there.
"""

from __future__ import annotations

import json
import re
import subprocess
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
#: #2523's red-tree extract: the 221 ledger rows behind #2385's "2 of the red-tree items
#: attempted opened no round". #2385's owed entry named the retired mirror path for them,
#: so they are committed at a dated sibling and re-derived here — the route #2178 and
#: #2387 set for the two extracts above. The name is the one the item asked for; the
#: directory's `YYYY-MM-DD.<item>-<subject>-witness` convention would name it
#: `2026-10-08.2523-red-tree-witness.jsonl`, and the item's own name is kept so this
#: round is not graded on a rename nobody asked for.
WITNESS_RED_TREE_NAME = "promotions-red-tree-20261008.jsonl"
#: The window the projection was taken over, inclusive at both ends and compared as ISO
#: strings — the same field the store itself is aged on.
RED_TREE_WINDOW = ("2026-09-26T18:19:58Z", "2026-10-08T02:46:25Z")
#: The keys the projection was allowed to keep, and nothing else. Every row of the live
#: store carries a dozen more (`ts`, `session_id`, `outcome`, the reasoning-bank fields);
#: dropping them is what fits 221 rows into 62,109 bytes instead of the store's 31 MB.
RED_TREE_KEYS = frozenset({
    "created_at", "event", "phase", "item_id", "round_id", "num_turns", "closed_by",
    "base", "action", "node_ids", "nodes", "acceptance"})


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


def red_tree_attempt(rows):
    """The two figures #2385 quotes, counted over a list of rows and nothing else.

    Split out of the node below on purpose: this function takes rows and no path, so a
    figure it returns cannot have come from the store the extract was cut out of. That is
    the whole point of committing the extract — #1975 folds `promotions.jsonl`, and a
    figure whose only home is that file has an expiry date nobody announced.
    """
    finished = [r for r in rows if r.get("event") == "backlog_implement"
                and r.get("phase") == "finished"]
    return ({r["item_id"] for r in finished},
            {r["item_id"] for r in finished if r.get("round_id") is None})


def test_the_2523_red_tree_extract_reproduces_the_two_roundless_items():
    """#2523: the bytes behind #2385's "the only 2 that opened no round" are committed,
    and the figure comes out of those bytes alone.

    What the figure is. Across `2026-09-26T18:19:58Z..2026-10-08T02:46:25Z` the loop filed
    a red tree for 55 items, and 48 of them have a `backlog_implement` row that reached
    `phase: finished`. Exactly 2 of those finished rows carry a null `round_id` — item 1846
    on 2026-09-29 at 14 turns, item 2384 on 2026-10-08 at 22 — and those two are the loss
    #2385 is about: an implement turn spent with no round opened, so 36 turns bought
    nothing but the filing.

    Two numbers this docstring exists to carry, because a later reader cannot re-derive
    them from the extract and will re-derive something else. (1) #2385's report quotes
    "50 attempted". Under the definition the null-round count is actually taken on — items
    with a `phase: finished` `backlog_implement` row in the window — the population is 48,
    and 48 is all these bytes can yield; the only other census the extract supports is the
    55 items with a `red_tree_filed` row. "50" is not hiding in here, so a re-derivation
    that expects it will conclude the extract is broken when it is intact. (2) The
    committed extract is 221 rows — measured, 221 newline characters over 221 parsed
    objects, 62,109 bytes. The projection as filed in #2523 said 220; the projection
    yields 221, and a node written to assert 220 would have to fail on the very bytes it
    was written to protect.

    Why the bytes are committed at all: the only other home they have is the live
    promotions ledger, 31,995,042 bytes over 22,257 rows at triage and ~30x the
    `WITNESS_MAX_BYTES` ceiling on what an owed entry may order whole — and #1975 folds
    that store, so a figure that lives only there stops being re-derivable the day the
    fold rewrites it. Why they are NOT at the path the owed entry named: that path is the
    promotions mirror the retention sweep retired (#2054's para, #2064's `ffc04ce5`),
    which this file's first node and `tests/test_diagnosis_fanout_replay.py` both keep
    absent. Writing an extract there is what got #2387 filed over #2383's, and the fix
    #2387 applied there is the fix applied here: the bytes stay, the name becomes a
    sibling, and a node reads them.
    """
    from scripts.automod import backlog as B

    data = paths.VAULT_ROOT / "backlog" / "data"
    witness = data / WITNESS_RED_TREE_NAME
    assert witness.is_file(), (
        f"{WITNESS_RED_TREE_NAME} must be committed: #2385's two round-less items are "
        "only re-derivable from bytes with history, the automod state dir is not a git "
        "tree, and the live store is a folding target (#1975)")
    assert not (data / "promotions.jsonl").exists(), \
        "the extract belongs on the dated sibling; the retired mirror stays absent"

    # The seam this round crosses is a process one, not a function one: the figure has to
    # survive #1975's next fold, and that fold runs elsewhere and rewrites a 31,995,042-byte
    # file. Nothing in this module may reach that file — an import of the module that
    # resolves it, or a spelling of the environment variable that redirects it, is the only
    # way a later edit could. Each mark is built by concatenation because a scan that
    # forbids a token has to avoid spelling it: the first version of this node wrote the
    # state directory's name into its own assert and then tripped over its own assert text.
    # Four marks: the state directory, the variable that redirects it, and the two import
    # forms that reach the module resolving it — the review rung caught that the first
    # version named only the `import X`-with-`from` spelling, so a dotted-module import would
    # have walked past the scan straight into the live store. Every mark below is built by
    # concatenation because a scan that forbids a token cannot spell it: the first version
    # wrote the state directory's name into its own assert text and tripped over it.
    own = Path(__file__).read_text(encoding="utf-8")
    for mark in ("lloyd" "-automod", "LLOYD_AUTOMOD" "_STATE",
                 "automod import " "state", "scripts.automod" ".state"):
        assert mark not in own, (
            f"this file now names the live promotions-ledger store ({mark!r}), so the "
            "figures below are no longer derived from committed bytes alone — which is "
            "the only thing that keeps them alive across the fold")

    # Clause 1's word is COMMITTED, and `is_file()` alone would green-light a scratch file
    # dropped into that directory. Every other witness node in this file stops at
    # `is_file()` — this one goes one subprocess further, to the same `git ls-files` the
    # item's own check and #2064's absence rail use, because the property here is history:
    # a figure with no commit behind it is a figure #1975's fold has nothing to hold.
    tracked = subprocess.run(
        ["git", "-C", str(paths.VAULT_ROOT), "ls-files", "--error-unmatch",
         str(witness.relative_to(paths.VAULT_ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{WITNESS_RED_TREE_NAME} is on disk but not tracked: "
        f"`git ls-files --error-unmatch` said {tracked.stderr.strip() or 'nothing'}, and an "
        "uncommitted extract is a scratch file rather than bytes with history")

    # The docstring is part of what this node ships, and its two numbers are the two a later
    # reader cannot re-derive from these bytes and will re-derive wrong instead. Pinned as
    # prose — the shape tests/test_action_review.py uses for a preamble it needs to keep
    # saying a thing — because the rail at the end of this file compares only node NAMES, so
    # a docstring that quietly dropped "48" would have left the module green and the extract
    # documented by a figure of record that its own bytes cannot produce.
    doc = test_the_2523_red_tree_extract_reproduces_the_two_roundless_items.__doc__ or ""
    for phrase in ("\"50 attempted\"", "the population is 48", "extract is 221 rows",
                   "62,109 bytes", "said 220"):
        assert phrase in doc, (
            f"the docstring no longer states {phrase!r}, and #2385's figure is only safe "
            "while the numbers it cannot re-derive are stated next to the bytes")

    # The extract's own integrity, measured the way `wc` measures: newline CHARACTERS (the
    # file ends with one, so `wc -l` and the parsed-row count have to agree, and an
    # agreement is what proves no row was truncated mid-object) and `wc -c` on disk.
    raw = witness.read_text(encoding="utf-8")
    rows = list(fl.iter_jsonl(witness))
    assert raw.count("\n") == len(rows) == 221, (
        f"the extract has {raw.count(chr(10))} newline characters and parses to "
        f"{len(rows)} rows; 221 is the count measured when it was cut, and either figure "
        "moving means the committed bytes are no longer the projection they document")
    assert witness.stat().st_size == 62109 == len(raw.encode("utf-8")), (
        f"`wc -c` on the extract is {witness.stat().st_size}, not the 62,109 bytes the "
        "projection yields")
    assert witness.stat().st_size < B.WITNESS_MAX_BYTES, (
        f"the extract is {witness.stat().st_size} bytes against the "
        f"{B.WITNESS_MAX_BYTES:,}-byte ceiling an owed entry may ask for — the whole "
        "reason this is a projection and not a copy")

    # The projection's shape: the window it was cut over, inclusive, on the ISO string the
    # store itself is aged by, and only for items a red tree was actually filed for.
    lo, hi = RED_TREE_WINDOW
    stamps = [str(r["created_at"]) for r in rows]
    assert min(stamps) == lo and max(stamps) == hi, (
        f"the extract spans {min(stamps)}..{max(stamps)}, not the window "
        f"{lo}..{hi} #2385's figure was counted over")
    assert all(lo <= s <= hi for s in stamps), "a row sits outside the projection's window"
    for r in rows:
        assert set(r) <= RED_TREE_KEYS, f"row keeps keys the projection excluded: {r}"
    assert {r["event"] for r in rows} == {
        "backlog_implement", "red_tree_filed", "red_tree_closed"}, (
        "an extract of a third event is not the census of red-tree items it is named for")
    filed_items = {r["item_id"] for r in rows if r["event"] == "red_tree_filed"}
    assert {r["item_id"] for r in rows} == filed_items, (
        "the extract holds a row for an item with no `red_tree_filed` row in the window, "
        "so it is no longer a census of red-tree items and its counts mean something else")
    assert (len(filed_items), sum(1 for r in rows if r["event"] == "red_tree_filed"),
            sum(1 for r in rows if r["event"] == "backlog_implement"),
            sum(1 for r in rows if r["event"] == "red_tree_closed")) == (55, 83, 109, 29), (
        "the extract is no longer 83 filings + 109 implement rows + 29 closes over 55 "
        "items, which is what the 221 rows are made of")

    # The figure itself, out of `red_tree_attempt`, which sees rows and no path.
    attempted, roundless = red_tree_attempt(rows)
    assert len(attempted) == 48, (
        f"{len(attempted)} items finished an implement in the window, not the 48 #2385's "
        "figure is a fraction of — and not the 50 its report quotes, which is the number "
        "these bytes cannot produce")
    assert sorted(roundless) == [1846, 2384], (
        f"items with a finished implement and no round are {sorted(roundless)}; the two "
        "that opened no round are the whole motivating finding of #2385")
    assert len(attempted - roundless) == 46, \
        "46 of the 48 opened a round, which is the denominator of '2 of 48'"
    assert sorted((r["item_id"], r["created_at"], r["num_turns"]) for r in rows
                  if r.get("event") == "backlog_implement"
                  and r.get("phase") == "finished"
                  and r.get("round_id") is None) == [
        (1846, "2026-09-29T17:09:14Z", 14), (2384, "2026-10-08T01:20:02Z", 22)], (
        "the two null-round rows are no longer the rows #2385 names, so the extract "
        "documents a finding a later reader cannot check")
    # 54 finished rows over 48 items: the loop re-attempted items 1694, 1866 and 2218. The
    # row/item split is pinned because "2 of 54" and "2 of 48" are different claims and the
    # file alone does not say which one it supports.
    finished_rows = [r for r in rows if r.get("event") == "backlog_implement"
                     and r.get("phase") == "finished"]
    assert len(finished_rows) == 54, (
        f"{len(finished_rows)} finished implement rows over {len(attempted)} items — the "
        "figure is per item, and a reader who counts rows gets a different fraction")


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
    assert len(shipped) == 14, (
        "thirteen nodes that read a vault tree through git plus this rail. The count is "
        "spelled out so that adding a node has to come back here and re-read the "
        "paragraph, instead of leaving it a sentence behind")


def test_the_2444_promotions_extract_reproduces_the_at_the_max_ratio():
    """#2444 owed 1, published by #2560: the confirm-revote extract, on a dated sibling.

    The seventh witness, and the one whose demand was the last to aim at the retired
    slot: the owed note #2444 carried told its round to commit this extract to
    `backlog/data/promotions.jsonl` and re-derive it from there, which is the path #2064
    deleted (`ffc04ce5`) and three assertions above keep absent. It is committed instead
    as a dated sibling under this directory — the form `_witness_artifact` emits for a
    source whose basename is retired — and its name is DISCOVERED by glob below rather
    than retyped, because that name's middle term is the live ledger's own directory and
    the scan inside `test_the_2523_red_tree_extract_reproduces_the_two_roundless_items`
    rightly refuses this file to spell it. That the emitter names exactly this file is
    pinned in `tests/test_witness_artifact_clause.py`, which has no such wall.

    82 rows, 870,849 bytes, copied byte-for-byte out of the live ledger and re-derived
    from these committed bytes alone: `confirm_votes_at_300 107` of
    `confirm_votes_total 110` — the ratio owed-check #2444/1 re-sizes against — carried
    by `rows_with_votes 73`, plus `arch_review_rows 9`, every one with `detail: null`.
    Under `WITNESS_MAX_BYTES` (1,048,576) by 177,727 bytes.

    What is deliberately NOT asserted is that those 82 rows are still in the live store:
    `sweep_promotions_ledger` ages rows out, and a node that went red the first time a
    row aged would be an alarm about the passage of time, not about a broken witness.
    The `.md` sibling carries the route back for a re-cut (the dated archive beside the
    store, and vault `4bc93177` for the pre-window copy).
    """
    from scripts.automod import backlog as B

    data = paths.VAULT_ROOT / "backlog" / "data"
    hits = sorted(data.glob("*.2444-*promotions.jsonl"))
    assert len(hits) == 1, f"the extract is one committed file, not {len(hits)}: {hits}"
    witness = hits[0]
    assert re.match(r"^\d{4}-\d{2}-\d{2}\.2444-[\w.\-+]+-promotions\.jsonl$",
                    witness.name), (
        f"not the dated-sibling form the emitter owes this store: {witness.name}")
    assert witness.is_file() and not witness.is_symlink(), witness
    assert not (data / "promotions.jsonl").exists(), \
        "the extract belongs on the dated sibling; the retired mirror stays absent"

    # Clause 4's word is COMMITTED, so this goes to the same `git ls-files` the #2523
    # node uses: an extract on disk but untracked is a scratch file with no history for
    # the next fold to hold.
    tracked = subprocess.run(
        ["git", "-C", str(paths.VAULT_ROOT), "ls-files", "--error-unmatch",
         str(witness.relative_to(paths.VAULT_ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{witness.name} is on disk but not tracked: `git ls-files --error-unmatch` said "
        f"{tracked.stderr.strip() or 'nothing'}, and an uncommitted extract is a scratch "
        "file rather than bytes with history")

    raw = witness.read_bytes()
    assert len(raw) == witness.stat().st_size < B.WITNESS_MAX_BYTES, (
        f"the extract must fit under the ceiling it was cut to fit: {len(raw)}")
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]

    votes = at_max = with_votes = arch = arch_null = 0
    for r in rows:
        for e in (r.get("review_confirm_votes") or []):
            if isinstance(e, dict) and "reason" in e:
                votes += 1
                at_max += (len(str(e.get("reason") or "")) == 300)
        if r.get("review_confirm_votes"):
            with_votes += 1
        if r.get("event") == "arch_review":
            arch += 1
            arch_null += r.get("detail") is None
    assert len(rows) == 82 and with_votes == 73, (len(rows), with_votes)
    assert votes == 110 and at_max == 107, (votes, at_max)
    assert arch == 9 and arch_null == 9, (arch, arch_null)
    assert all(r.get("review_confirm_votes") or r.get("event") == "arch_review"
               for r in rows), "a row kept for no reason this extract publishes"
    assert len(raw) == 870849, len(raw)

    md = witness.with_suffix(".md").read_text(encoding="utf-8")
    for quoted in (f"rows {len(rows)} bytes {len(raw)}",
                   f"confirm_votes_total {votes}",
                   f"confirm_votes_at_300 {at_max}",
                   f"rows_with_votes {with_votes}",
                   f"arch_review_rows {arch}"):
        assert quoted in md, (
            f"the note quotes figures this extract does not print: {quoted!r}")


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation at `i = (n - 1) * q` — the method the extract was built with.

    Not `statistics.quantiles`, and not nearest-rank: the header publishes values only
    THIS method reproduces, and the whole value of re-deriving a published figure is
    that a mismatch means the bytes changed rather than that the test picked a different
    convention. Checked against the committed extract: all three percentiles match to
    the last digit.
    """
    i = (len(sorted_values) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (i - lo)



# ── #2878: every dated promotions extract, not just the one already pinned ─────
#
#: A FLOOR, not an equality. Extracts are only ever added to this directory (each
#: owed entry cuts a dated sibling and nothing in the loop deletes a witness), so a
#: floor cannot rot the way `== 7` would, and it is still the assertion that turns
#: the suite red the first time one goes missing — which is what the six unpinned
#: ones never did. 7 is the count of committed dated promotions extracts as at the
#: commit this node was written.
_MIN_PROMOTION_EXTRACTS = 7

#: The dated-sibling form `_witness_artifact` emits for a source whose basename is
#: retired. Group 1 is the item id, which is what the one-per-item check below
#: counts — and it is counted out of this pattern, never out of a list of ids, so a
#: node that named its subjects would stop being a property of the directory.
_PROMOTION_EXTRACT_NAME = re.compile(
    r"^\d{4}-\d{2}-\d{2}\.(\d{4})-[\w.\-+]+-promotions\.jsonl$")


#: The witness directory, as the vault writes it. One constant for both `git`
#: questions below and for the throwaway vault the #3272 node builds, so a fixture
#: cannot drift out of the directory the live node reads.
_PROMOTION_SUBDIR = "backlog/data"

#: One extract of each of the two states an extract can be in on the day a land is
#: asked to make it a witness: one committed, so it has a HEAD entry, and one left
#: untracked, so it has none. Both names are invented — the fixture vault they live in
#: is a throwaway, and the live node below may not spell an item id, so neither may this
#: pair be one the vault has.
_EXTRACT_IN_HEAD = "2026-09-30.4111-automod-promotions.jsonl"
_EXTRACT_UNTRACKED = "2026-10-19.4112-automod-promotions.jsonl"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """One git command in `repo`, with its return code left to the caller.

    The nodes in this file usually assert on a subprocess directly, because the
    subprocess is the claim. #3272 split one claim into two — is this path in the
    commit, is it in the index — and both are asked of one tree and answered as a set,
    so they share a call and the nodes assert on the sets.
    """
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _vault_commit_paths(vault: Path, subdir: str, rev: str = "HEAD") -> set[str]:
    """Paths under `subdir` that the commit `rev` holds, as one set.

    `ls-tree` reads a COMMIT. That is the question a pre-commit guard is allowed to
    ask, because the rev is named: it is answerable of a tree that does not hold the
    file yet, and it answers "no" as a fact about the candidate instead of failing.
    """
    r = _git(vault, "ls-tree", "-r", "--name-only", rev, "--", subdir)
    assert r.returncode == 0, (
        f"`git ls-tree -r --name-only {rev} -- {subdir}` in {vault} exited "
        f"{r.returncode}: {r.stderr.strip()[:200]}")
    return {ln for ln in r.stdout.splitlines() if ln.strip()}


def _vault_index_paths(vault: Path, subdir: str) -> set[str]:
    """Paths under `subdir` that the vault's INDEX holds, as one set.

    `ls-files` reads the index, which is the second reason #3272 stopped asking it of a
    promotions extract: a vault land runs the code-agreement probe BEFORE its own
    `git add` (`vault_round.py`'s `land()` probes at the `guards = VG.agreement(` call
    and first stages seventeen lines later, past the whole refusal branch), so during
    the probe a brand-new extract is in the index no more than it is in HEAD. A guard
    that fails on the state every candidate is born in does not check the candidate; it
    bans the class.
    """
    r = _git(vault, "ls-files", "--", subdir)
    assert r.returncode == 0, (
        f"`git ls-files -- {subdir}` in {vault} exited {r.returncode}: "
        f"{r.stderr.strip()[:200]}")
    return {ln for ln in r.stdout.splitlines() if ln.strip()}


def _fixture_promotions_vault(tmp_path: Path, *, committed: dict[str, str],
                              untracked: dict[str, str]) -> Path:
    """A throwaway vault: `committed` extracts at HEAD, `untracked` ones on disk only.

    A real `git init` and a real commit rather than two sets handed to the check,
    because the claim under test is that the directory check now reads a COMMIT and an
    INDEX. A fixture that supplied those two answers itself would be the node agreeing
    with itself, which is the hole `vault_round.py` names when it refuses to hand the
    probe a candidate tree (#2036).
    """
    vault = tmp_path / "vault"
    data = vault / _PROMOTION_SUBDIR
    data.mkdir(parents=True)
    assert _git(vault, "init", "-q", "-b", "main").returncode == 0, "git init failed"
    for name, body in committed.items():
        (data / name).write_text(body, encoding="utf-8")
    assert _git(vault, "add", "-A").returncode == 0, "staging the fixture failed"
    # `--allow-empty` because the fixture's whole subject is a tree with NOTHING
    # committed yet, and such a tree still has to have a HEAD for `ls-tree` to be asked
    # of it — which is the state a candidate vault land is in: the vault has commits, the
    # candidate is not one of them.
    commit = _git(vault, "-c", "user.email=fixture@lloyd.invalid",
                  "-c", "user.name=Lloyd Fixture", "commit", "-q", "--allow-empty",
                  "-m", "fixture history")
    assert commit.returncode == 0, commit.stderr.strip()[:200]
    for name, body in untracked.items():
        (data / name).write_text(body, encoding="utf-8")
    return vault


def _promotion_extract_findings(vault: Path
                                ) -> tuple[list[str], dict[str, list[str]], list[Path]]:
    """Every way the dated promotions extracts under `vault/backlog/data` are wrong.

    Returns `(findings, ids_by_item, hits)`; `findings` is empty exactly when the
    directory is in order. A function that takes a vault rather than a node that
    asserts on the live one, because #3272 has to show two things the live tree cannot
    show: that an extract with no commit yet is not a violation, and that the
    narrowing removed ONE check and not four — which needs bytes this file writes and
    can therefore also corrupt on purpose.

    Tracked-ness is the only property asked conditionally, and the condition is that a
    commit exists to be tracked against. `ls-tree` naming a path while `ls-files` does
    not means history left the tree, which is a real loss and stays red; a path neither
    knows is a candidate on its way to being committed, which is where every tracked
    extract in this directory began. The other four checks are unconditional, because
    a misnamed, doubled, oversized or truncated extract is wrong in the index, in HEAD
    and in a land's working tree alike.
    """
    from scripts.automod import backlog as B

    data = vault / _PROMOTION_SUBDIR
    hits = sorted(data.glob("*.????-*promotions.jsonl"))
    in_head = _vault_commit_paths(vault, _PROMOTION_SUBDIR)
    in_index = _vault_index_paths(vault, _PROMOTION_SUBDIR)
    findings: list[str] = []

    ids: dict[str, list[str]] = {}
    for p in hits:
        m = _PROMOTION_EXTRACT_NAME.match(p.name)
        if not m:
            findings.append(
                f"{p.name} matches the promotions-extract glob but not the dated-sibling "
                "form `_witness_artifact` emits, so it is either misnamed or a scratch "
                "file wearing a promotions suffix")
            continue
        ids.setdefault(m.group(1), []).append(p.name)
    dupes = {k: v for k, v in sorted(ids.items()) if len(v) > 1}
    if dupes:
        findings.append(
            f"{dupes} — an item with more than one dated promotions extract leaves a "
            "later reader to choose between two witnesses of the same figure")

    for p in hits:
        rel = str(p.relative_to(vault))
        raw = p.read_bytes()
        if rel in in_head and rel not in in_index:
            findings.append(
                f"{p.name} is in the vault's HEAD and not in its index: `git ls-files` "
                "cannot name it, so the history this directory's guard exists to keep "
                "has left the tree and the bytes on disk are a scratch file again")
        if not len(raw) == p.stat().st_size < B.WITNESS_MAX_BYTES:
            findings.append(
                f"{p.name} is {len(raw):,} bytes against the "
                f"{B.WITNESS_MAX_BYTES:,}-byte ceiling an owed entry may ask a witness "
                "to fit under")
        bad = [(i + 1, ln) for i, ln in enumerate(raw.decode("utf-8").splitlines())
               if ln.strip() and not _parses_as_json(ln)]
        if bad:
            findings.append(
                f"{p.name} line {bad[0][0]} is not JSON: {bad[0][1][:120]!r} — a "
                "truncated row is the failure mode of a hand-copied extract, and the "
                "only one nobody sees by eye in 1,742 lines")
    return findings, ids, hits


def _own_source_of(node_name: str) -> str:
    """The bytes of one node in this file, docstring included, cut from `def` to `def`.

    Needed because the property node below has to prove it does not name the items it
    enumerates, and the file as a whole necessarily does name some: this module's
    other nodes are each about one item and say so, in their globs and their
    docstrings. The claim that survives that is the one about the node that claims to
    enumerate, so that is the span the scan reads.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    start = src.index(f"def {node_name}(")
    rest = src[start + 1:]
    end = rest.index("\ndef ") if "\ndef " in rest else len(rest)
    return src[start:start + 1 + end]


def test_every_dated_promotions_extract_is_committed_and_parses():
    """Every dated promotions extract in the witness directory is tracked, sized and parseable.

    #2878's premise, measured the day it was filed: seven dated promotions extracts
    are committed under `backlog/data/`, and exactly ONE was pinned — the node above,
    which globs for its own item alone. The other six could have been deleted from the
    vault tomorrow and this file would have stayed green, which is the same failure the
    module docstring rails against one level up: a figure with no check under it is a
    figure that quietly stops being true. This node is a property of the directory, so
    the eighth extract is covered the day it lands and nothing here has to be edited
    again to keep the guard whole.

    What is asserted per hit, and what is deliberately NOT:

    - The name is the dated-sibling form `_witness_artifact` emits, and AT MOST ONE
      file exists per item id — two extracts for one item is two candidate witnesses
      for one figure, which is the drift this directory's naming scheme exists to avoid.
    - A file the vault's HEAD holds is also held by its index. Same reason #2523's node
      goes one subprocess further than `is_file()`: an extract whose history has gone is
      a scratch file, and history is the only thing that outlives the fold of the live
      store the extract was cut from. #3272 narrowed WHICH files this is asked of, and
      the paragraph below is why the narrowing is the whole of the fix.
    - It fits under `WITNESS_MAX_BYTES` — the ceiling an owed entry may ask for. The
      largest extract today is 870,849 bytes of it; the smallest is 1,509.
    - Every non-empty line parses as JSON.

    Two assertions the item asked for that this node does NOT make, because they are
    false on today's bytes and a guard that is red on arrival guards nothing: only one
    of the seven begins with a header row carrying `measured_at`/`built_at` (the other
    six begin with a raw ledger row), and two of the seven have no `.md` sibling. So
    the parseable-line property is the one that holds for all seven, stated as the
    item's own fallback: "assert instead that every non-empty line parses as JSON".

    The item ids are discovered by the glob and by nothing else. This node's own bytes
    are scanned for them below, and no per-id allowlist appears: a node that listed its
    subjects would pass today and go silently stale the day a witness is added, which
    is precisely how the six unpinned extracts stayed unpinned.
    """
    data = paths.VAULT_ROOT / _PROMOTION_SUBDIR
    # The store this file may never read is a sibling of the vault, and this is the
    # node that walks a directory named after promotions. Say which root it is in.
    assert data.resolve().is_relative_to(paths.VAULT_ROOT.resolve()), (
        f"{data} is not inside the vault, so the `git ls-files` below is grading some "
        "other tree")

    # The five checks live in `_promotion_extract_findings`, which takes a vault and
    # returns what is wrong with it, so the #3272 node can drive the SAME function over
    # a throwaway tree. Left here, they would have been prose about a directory only the
    # live vault can show, and the fix would have had no witness of its own.
    findings, ids, hits = _promotion_extract_findings(paths.VAULT_ROOT)
    assert len(hits) >= _MIN_PROMOTION_EXTRACTS, (
        f"{len(hits)} dated promotions extracts are on disk, fewer than the "
        f"{_MIN_PROMOTION_EXTRACTS} committed as at this node's writing: a witness "
        "someone owes has left the tree, and only one of the six unpinned ones was "
        "ever checked for being there")

    # The enumeration is a property, not a roster. Every id discovered above must be
    # ABSENT from this node's own bytes; the day someone writes one in, the node stops
    # enumerating the directory and starts asserting a list of it.
    own = _own_source_of("test_every_dated_promotions_extract_is_committed_and_parses")
    spelled = sorted(i for i in ids if i in own)
    assert not spelled, (
        f"this node now spells {spelled}, so it is checking a list of items rather "
        "than a property of the directory — which is how the six unpinned extracts "
        "stayed unpinned: the one node that looked named only its own subject")

    assert not findings, (
        f"{len(findings)} problem(s) among the {len(hits)} dated promotions extracts of "
        f"the live vault: {findings[0]}")


def _parses_as_json(line: str) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def test_the_promotions_directory_check_passes_on_an_extract_that_is_not_committed_yet(tmp_path):
    """The directory check on a throwaway vault: one extract with history, one without.

    #3272, and the state is the one every promotions extract in `backlog/data/` is born
    into. The node above asked `git ls-files --error-unmatch` of every glob hit, and that
    node runs inside the gate's own selection and inside a vault land's code-agreement
    probe, which a `land()` call reaches SEVENTEEN LINES BEFORE its first `git add`
    (`vault_round.py`: the `guards = VG.agreement(` call, then the whole refusal branch,
    then `_ensure_main`, `stage = _stageable(norm)` and `add = _git("add", ...)`). So the
    only tree where that guard was green was the tree the refused commit was going to
    create: #3264's extract has sat untracked in the witness directory since 2026-10-19
    because its own land was refused by a check no commit could satisfy from the outside,
    and every round's `tests` rung carried the same node red in the meantime.

    The first assertion is the narrowing: two extracts, one at HEAD and one on disk only,
    and the directory is in order. What has to come with it, or the change is simply a
    check deleted, is the rest of the node — each of the four surviving properties fires
    once against an extract that HAS a commit and once against one that has NONE, and
    tracked-ness still fires where it names an actual loss: a file HEAD holds and the
    index no longer does.

    A fixture and not the live vault because the live vault cannot hold "not committed
    yet" for the length of a run — the moment the land that fixes #3264 succeeds the
    state stops existing — and because demonstrating that four checks survived needs
    bytes this file is free to corrupt.
    """
    from scripts.automod import backlog as B

    row = json.dumps({"kind": "vault_land", "item_id": 4111, "ok": True}) + "\n"
    vault = _fixture_promotions_vault(tmp_path / "pair",
                                      committed={_EXTRACT_IN_HEAD: row},
                                      untracked={_EXTRACT_UNTRACKED: row})
    # The two states the pass below is about have to be real and not assumed: one name in
    # the commit, the other in neither the commit nor the index.
    head = _vault_commit_paths(vault, _PROMOTION_SUBDIR)
    index = _vault_index_paths(vault, _PROMOTION_SUBDIR)
    assert f"{_PROMOTION_SUBDIR}/{_EXTRACT_IN_HEAD}" in head
    assert f"{_PROMOTION_SUBDIR}/{_EXTRACT_UNTRACKED}" not in head | index, (
        "the fixture's uncommitted extract is already somewhere git can see it, so the "
        "assertion below is grading a file nobody asked about")
    findings, ids, hits = _promotion_extract_findings(vault)
    assert len(hits) == 2 and len(ids) == 2, (
        f"the fixture is two extracts of two different items, not {len(hits)} extract(s) "
        f"reading as {sorted(ids)}: a name the pattern cannot read is skipped by the "
        "one-per-item check, and a clean result would then be about one file")
    assert not findings, (
        "an extract with no commit yet is a violation again, which is the circle #3264 "
        f"died in: {findings[0]}")

    bad = (
        ("misnamed", "2026-9-30.4113-automod-promotions.jsonl", row,
         "dated-sibling form"),
        ("oversized", "2026-10-01.4114-automod-promotions.jsonl",
         row * (B.WITNESS_MAX_BYTES // len(row) + 1), "byte ceiling"),
        ("truncated", "2026-10-02.4115-automod-promotions.jsonl",
         '{"kind": "vault_land", "it\n', "is not JSON"),
    )
    for label, name, body, needle in bad:
        for in_head in (True, False):
            got = _promotion_extract_findings(_fixture_promotions_vault(
                tmp_path / f"{label}-{in_head}",
                committed={name: body} if in_head else {},
                untracked={} if in_head else {name: body}))[0]
            assert any(needle in f and name in f for f in got), (
                f"the {label} check passed a bad extract "
                f"{'with' if in_head else 'without'} a commit behind it: {got[:2]} — "
                "narrowing tracked-ness was meant to take one check out of five, not four")

    # A second extract of one item id: the item the fixture already has a committed
    # extract for, so the HEAD-present half is the pair itself and the HEAD-absent half is
    # the twin a land is on its way to committing.
    twin = "2026-10-03.4111-automod-promotions.jsonl"
    for in_head in (True, False):
        got = _promotion_extract_findings(_fixture_promotions_vault(
            tmp_path / f"duplicate-{in_head}",
            committed=({_EXTRACT_IN_HEAD: row, twin: row} if in_head
                       else {_EXTRACT_IN_HEAD: row}),
            untracked=({} if in_head else {twin: row})))[0]
        assert any("more than one" in f and twin in f for f in got), (
            f"two extracts of one item id read as clean when the second is "
            f"{'committed' if in_head else 'uncommitted'}: {got[:2]}")

    # And the tracked-ness check itself, in the one shape that is a loss rather than a
    # birth state: HEAD holds the file, the index no longer does.
    dropped = _fixture_promotions_vault(tmp_path / "dropped",
                                        committed={_EXTRACT_IN_HEAD: row},
                                        untracked={})
    removed = _git(dropped, "rm", "-q", "--cached", f"{_PROMOTION_SUBDIR}/{_EXTRACT_IN_HEAD}")
    assert removed.returncode == 0, removed.stderr.strip()[:200]
    got = _promotion_extract_findings(dropped)[0]
    assert any(_EXTRACT_IN_HEAD in f and "not in its index" in f for f in got), (
        f"a file the commit holds and the index does not read as clean: {got[:2]} — "
        "history leaving the tree is the only thing tracked-ness was ever asked for, and "
        "#3272 keeps it")


def test_the_2759_promotions_extract_reproduces_its_published_lag_percentiles():
    """#2759 owed 2: its published gate→promote lag figures, recomputed from the extract's bytes.

    The extract's first line is a header that PUBLISHES a `published` block — n, the
    percentiles, the buckets, the ceiling — and until this node nothing in the repo had
    re-derived one of them. Those figures therefore had the status of the one this
    module exists to rail against (#2079 clause 6, #2385's red-tree count): quoted from
    the note, never measured against the bytes the note was cut from. A reader who
    wanted to check the p99 had to do the arithmetic by hand.

    Everything comes out of this file's own committed bytes, including both
    thresholds, so no literal appears that could disagree with the artifact:

    - population: the `kind: "round"` rows carrying both `gate_ts` and `promoted_ts`;
      lag = `promoted_ts - gate_ts`, the interval the item is named for. 1176 of the
      1676 round rows pair, the header's own `n_joined` says 1176, and one assertion
      ties the three together so a re-cut cannot move one of them alone.
    - percentiles by linear interpolation at `i = (n - 1) * q`. Stated because the
      method is load-bearing: nearest-rank returns different numbers for p90 and p99 on
      this population and would go red against a header that is correct.
    - `over_bucket` and `over_ceiling` count lags STRICTLY over `published.bucket_s`
      (600 s) and `published.ceiling_s` (4500 s), both read from the header. The
      ceiling is cross-checked against the header's `idle_hard_max_wait_s` too, because
      a second field carrying the same number is the only thing in these bytes that says
      the ceiling came from the wait policy rather than from whoever typed the row.
    - the WHOLE published block is re-derived and its key set pinned, so a figure a
      future re-cut adds cannot join the block without something re-computing it.
    - the file's shape: 1742 non-empty lines = 1676 round rows + 65 moved rows + 1
      header, and exactly 65 lines contain `live HEAD moved`, matching
      `moved_row_count`. That is the extract's own internal arithmetic; if it is ever
      re-cut and the header updated without the rows, this is the assertion that says so.

    271,722 bytes, read once for the header and once for the rows — the price of the
    figures coming from disk rather than from this paragraph. The exact float values are
    re-derived and deliberately NOT restated here, so this prose cannot rot; the rounded
    figures, for orientation only, are p50 19.3 s, p90 163.3 s, p99 2040.5 s, max
    3194.2 s, min −2.5 s (yes, negative: one promotion was stamped a moment before its
    own gate finished, and `negative: 1` is published for it and re-derived here).

    What this node does NOT do is read what the header points at. The header names a
    `snapshot_dir` holding the store's state at cut time, under the runtime data root:
    outside the commit, removable by a cleaner, and a node that asserted it exists would
    be a guard reading its own missing input — the same reason the #2523 node refuses
    this file to name the live store at all, a scan that re-reads this file's bytes on
    every run and so covers the two nodes added here as well. The path the header names
    is checked for the one property checkable from here: that this file never spells it.
    """
    from scripts.automod import backlog as B

    data = paths.VAULT_ROOT / "backlog" / "data"
    lines_of = {p: [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                for p in sorted(data.glob("*.????-*promotions.jsonl"))}
    assert len(lines_of) >= _MIN_PROMOTION_EXTRACTS, (
        f"{len(lines_of)} dated promotions extracts, fewer than "
        f"{_MIN_PROMOTION_EXTRACTS}: the sibling node names the failure")

    # Which of these is the #2759 one? Discovered, not spelled: it is the one whose
    # first line is a header carrying a published block, and today exactly one of them
    # is — the other six begin with a raw ledger row. So the item's number never enters
    # the selection, and a second extract cut the same way is a real failure here
    # rather than an arbitrary pick between two.
    def _published_header(lines: list[str]) -> dict:
        try:
            first = json.loads(lines[0])
        except ValueError:
            return {}
        if isinstance(first, dict) and isinstance(first.get("published"), dict):
            return first
        return {}

    with_published = [p for p, ls in lines_of.items() if _published_header(ls)]
    assert len(with_published) == 1, (
        f"{len(with_published)} of these extracts carry a published block, not one: "
        f"{[p.name for p in with_published]} — the figures below belong to a single "
        "artifact and this node can no longer say which")
    witness = with_published[0]
    assert witness.resolve().is_relative_to(paths.VAULT_ROOT.resolve()), (
        f"{witness} is not inside the vault, so its figures are not committed bytes")

    lines = lines_of[witness]
    rows = [json.loads(l) for l in lines]
    header = _published_header(lines)
    pub = header["published"]

    # Thresholds from the artifact, never a literal: both are inputs to the two counts.
    bucket, ceiling = pub["bucket_s"], pub["ceiling_s"]
    assert bucket > 0 and ceiling > bucket, (
        f"the published thresholds are bucket_s={bucket} ceiling_s={ceiling}; both "
        "counts below are 'strictly over', which needs them positive and ordered")
    assert ceiling == header["idle_hard_max_wait_s"], (
        f"published.ceiling_s {ceiling} is not this extract's own idle_hard_max_wait_s "
        f"{header['idle_hard_max_wait_s']}, so the ceiling the lags were bucketed "
        "against is no longer the ceiling the wait policy had")

    lag = sorted(r["promoted_ts"] - r["gate_ts"] for r in rows
                 if r.get("kind") == "round"
                 and r.get("promoted_ts") is not None
                 and r.get("gate_ts") is not None)
    assert len(lag) == pub["n"] == header["n_joined"], (
        f"{len(lag)} round rows pair here against published.n {pub['n']} and the "
        f"header's n_joined {header['n_joined']} — the population these percentiles are "
        "over is not the one the header says it published them over")
    assert pub["n"] > 0, "a published percentile over an empty population"

    derived = {
        "n": len(lag),
        "p50": _percentile(lag, 0.50),
        "p90": _percentile(lag, 0.90),
        "p99": _percentile(lag, 0.99),
        "max": lag[-1],
        "min": lag[0],
        "over_bucket": sum(1 for x in lag if x > bucket),
        "over_ceiling": sum(1 for x in lag if x > ceiling),
        "negative": sum(1 for x in lag if x < 0),
    }
    unaccounted = sorted(set(pub) - set(derived) - {"bucket_s", "ceiling_s"})
    assert not unaccounted, (
        f"the header publishes {unaccounted}, which nothing here recomputes — a figure "
        "inside the artifact that no test derives is a figure a re-cut can change "
        "without anything going red")
    mismatched = {k: f"published {pub[k]!r}, derived {v!r}"
                  for k, v in derived.items() if v != pub[k]}
    assert not mismatched, (
        "the header publishes figures these bytes do not produce: " + "; ".join(
            f"{k}: {m}" for k, m in sorted(mismatched.items())))

    assert len(lines) == header["distinct_round_ids"] + header["moved_row_count"] + 1, (
        f"{len(lines)} non-empty lines, but the header describes "
        f"{header['distinct_round_ids']} round rows + {header['moved_row_count']} moved "
        "rows + 1 header — the extract's own arithmetic no longer adds up, which is what "
        "a re-cut that updated one field and not the other looks like")
    moved = sum(1 for l in lines if "live HEAD moved" in l)
    assert moved == header["moved_row_count"], (
        f"{moved} lines say 'live HEAD moved' against a header counting "
        f"{header['moved_row_count']}: the rows carried over verbatim and the number "
        "published for them disagree")

    assert len(witness.read_bytes()) < B.WITNESS_MAX_BYTES, (
        f"{witness.name} is over the ceiling a witness may be cut to, and this node "
        "parses it twice")

    snap = str(header.get("snapshot_dir") or "")
    assert snap, (
        "the header no longer names the snapshot it was cut from, so the check below "
        "has nothing to keep away from and a reader cannot tell that these bytes came "
        "from somewhere else")
    assert snap not in Path(__file__).read_text(encoding="utf-8"), (
        f"this file now spells the path the extract names as its source ({snap!r}). "
        "That is runtime data outside the commit: a cleaner can remove it, and a node "
        "that reads it — or asserts it exists — is a guard reading its own missing "
        "input")
