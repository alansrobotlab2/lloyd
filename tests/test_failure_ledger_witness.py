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

It ships ten nodes: nine over the vault bytes —
`test_the_witness_is_a_dated_file_not_the_retired_mirror_path`,
`test_the_witness_holds_the_two_live_rows_the_clause_quotes`,
`test_every_witness_row_is_an_error_alert_of_the_quoted_family`,
`test_the_sibling_note_names_the_mirror_it_is_not`,
`test_the_2175_vault_land_extract_is_a_dated_file_too`,
`test_the_2228_gate_witness_is_a_dated_sibling_and_the_mirror_is_untouched`,
`test_the_2383_extract_is_a_dated_file_reproducing_the_unadjudicated_baseline`,
`test_the_2383_gate_witness_is_a_dated_sibling_because_gate_json_was_taken`,
`test_the_2523_red_tree_extract_reproduces_the_two_roundless_items` — plus
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
    assert len(shipped) == 11, (
        "ten witness nodes over the vault bytes plus this rail. The count is spelled "
        "out so that adding a node has to come back here and re-read the "
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
