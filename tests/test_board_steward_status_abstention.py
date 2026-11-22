"""#4412: one item, one answer — the table may not emit both directions of a round-trip.

`reconcile_statuses` moves every item whose status differs from `desired_statuses`, on
every autocode poll (900 s), and each move appends an `activity_log` line and re-stamps
`updated:`. That is why one pre-flight sweep commits 143-165 backlog files: measured on
the 2026-11-20 snapshots, `c102adc6a` carries 136 `up_next → draft: never triaged` lines
against 31 `draft → up_next: the board steward` ones over 165 files, and `e0bfed0d1` 286
against 65 over 160 — and of the 1,181 backlog-file changes across the last 30
backlog-touching vault commits, none is an `updated:` re-stamp. So the churn is a status
answer that flips, not a timestamp that refreshes.

Both strings come out of `desired_statuses`, and one call can only return one of them, so
a flip needs the table's INPUT to change between two passes. The fold is what changes it:
the retention sweep moves rows older than `LEDGER_ARCHIVE_AGE_DAYS = 14` out of the live
`promotions.jsonl` and into `promotions-archive-<YYYYMM>.jsonl.gz`, and while
`desired_statuses` read the confirmation over live PLUS rotations (#3612 — the comment
inside it names #3024, #3041, #3068, #3196 and #3284 as items a fold parked on
2026-11-03..07), `_steward_claims` read the live file alone. The claim therefore died
fourteen days after the steward wrote it while the verdict it outranks kept counting: the
table re-derived a status, `reconcile_statuses` moved the item, and #3467's abstention
expired on a schedule. Measured at this round's start, #3024, #3041, #3196, #3284 and
#3877 each hold their one `steward_move` row dated 2026-11-08T03:48Z — twelve days old,
two days from that fold.

The nodes below hold one event set still answering the same once its rows are filed
differently, and one further asymmetry: every arm of the table fires on a row it FOUND, so
a decode that returns nothing leaves only the two "no verdict" arms reachable and parks the
whole implement pool. `state._decode_ledger` swallows a live file caught empty mid-rewrite
and a truncated `.gz` alike (`EOFError` out of zlib, not `OSError`) and reports `[]` for
both, so that read is indistinguishable from an untriaged board from inside the loop.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B
from scripts.automod import state as S

#: The park line, verbatim from `desired_statuses`. Its presence on an item the steward
#: has claimed is the round-trip's first half; #4412's whole ask is that it never appear
#: on an item whose newest ledger row is a `steward_move`.
PARK = "never triaged; autotriage reads draft"

#: One rotation, named as `groundskeeper/retention-sweep.py` names it and matched by
#: `state.LEDGER_ARCHIVE_GLOB`, so the fold under test is the store shape the sweep
#: actually leaves — not a fixture's invention.
ARCHIVE = "promotions-archive-202609.jsonl.gz"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    # No cache reset here on purpose. `state.ledger_rows` invalidates on
    # (mtime_ns, size, inode) per PATH, so a fresh tmp_path can never read a sibling's
    # rows, and a call to `ledger_rows_reset()` in teardown would clear the read-count
    # instrument another module's fixture may be standing on.
    yield d


def write_item(d: Path, item_id, *, status="draft", days_old=100,
               name="A thing", priority="medium", board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": priority, "created": created,
          "board": board, "tags": ["backlog"]}
    path = d / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n"
                    f"Body of item {item_id}.\n", encoding="utf-8")
    return path


def _steward_move(item_id, *, frm, to, ts, note="parked: spent its re-offer",
                  tick="board-steward:tick#4412"):
    """One `steward_move` row exactly as `workers/sources/board_steward.py:apply_moves`
    writes it — through the real writer, so it carries the `created_at` stamp
    `append_event` supplies and can never drift from the production store (#4147)."""
    S.append_event({"event": "steward_move", "item_id": item_id, "from": frm, "to": to,
                    "note": note, "tick": tick, "ts": ts}, path=S.LEDGER_PATH)


def _fold_the_live_ledger(*, empty_payload=False):
    """Move every live row into a rotation, as the retention sweep does.

    Same rows, same `ts`, different file — the only way to hold this table to "the same
    events answer the same" while its store changes shape under it. `empty_payload`
    writes the rotation the sweep creates BEFORE it has written rows into it, so the
    archive decodes to nothing while the live file sits emptied by the same operation:
    the half-done fold, which is the one state in which a board full of verdicts can read
    as a board with none.
    """
    rows = S.ledger_rows(S.LEDGER_PATH)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (S.LEDGER_PATH.parent / ARCHIVE).write_bytes(
        gzip.compress(b"" if empty_payload else blob.encode()))
    S.LEDGER_PATH.write_text("", encoding="utf-8")
    # Both files are re-read on the next call by their own (mtime_ns, size, inode), and
    # the archive is a path no earlier test had, so nothing has to be forgotten here.


def _claimed_item(d: Path, item_id, *, to="up_next", ts=None, note="the note",
                  applied=False):
    """An item whose newest ledger row is a `steward_move`, with the table answering the
    OPPOSITE status without it: its only triage row is `unverifiable`, which parks an
    item in the implement pool as "not for the unattended loop".

    That opposition is the point. A confirmed item asked about here would answer `up_next`
    with the claim or without it, so the node would pass while the claim did nothing — the
    #3150 failure mode. `unverifiable` makes the claim load-bearing, so a claim the reader
    cannot see shows up as a move.
    """
    write_item(d, item_id, status="draft")
    S.append_event({"event": "backlog_triage", "item_id": item_id,
                    "verdict": "unverifiable", "check": "grep -n x",
                    "evidence": "unreadable store"}, path=S.LEDGER_PATH)
    _steward_move(item_id, frm="draft", to=to, ts=ts or (time.time() + 600), note=note)
    if applied:
        # The real store shape: `apply_moves` wrote the status through the one status
        # writer before it booked the row, so the file already carries the move.
        B.set_status(item_id, to, f"board steward: {note}")
    return B.item_by_id(item_id).path


# ── clause 1: the newest steward row answers for its item ─────────────────

def test_the_newest_steward_row_is_the_status_answer(isolated):
    """An open item whose newest ledger row is a `steward_move` naming an open status,
    with no `item_landed`, `promoted` or confirmed row at or after it, is answered by that
    row: `up_next`, the steward named, and never the park pair."""
    _claimed_item(isolated, 9100)
    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[9100][0] == "up_next", want[9100]
    assert "board steward" in want[9100][1], want[9100][1]
    assert PARK not in want[9100][1], "the park pair is the round-trip's first half"
    assert len(want[9100]) == 2, "a steward claim is not a person's decision: nothing tagged"


def test_the_answer_survives_the_fold_that_rotates_the_claim_out(isolated):
    """The same item, the same events, and the claim now filed in a rotation.

    The retention sweep moves an aged `steward_move` out of the live file exactly as #3612
    moved an aged `backlog_triage` out, and the confirmation this abstention defers to is
    already read over live plus rotations. Read live-only, the claim dies at day 14 and the
    table falls back to the item's `unverifiable` verdict: `draft`, and
    `reconcile_statuses` moves the item the steward put in `up_next`. One item, two
    directions, fourteen days apart."""
    _claimed_item(isolated, 9101)
    assert B.desired_statuses(S.LEDGER_PATH, None)[9101][0] == "up_next"
    _fold_the_live_ledger()
    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[9101][0] == "up_next", (
        f"the folded claim was not read; the table answered {want[9101]}")
    assert "board steward" in want[9101][1], want[9101][1]
    assert PARK not in want[9101][1]


# ── clause 2: two passes, one answer ─────────────────────────────────────

def test_two_passes_over_unchanged_input_answer_every_item_the_same(isolated):
    """`desired_statuses` twice over a byte-identical ledger and item set returns the same
    status for every item both times — and the same again after the fold files those exact
    rows in a rotation, because the events are the input and where they are filed is not.

    The board is deliberately mixed: a steward-claimed item the table would otherwise park,
    a steward-parked item the table would otherwise offer, an untriaged item in the pool
    (the park's own case), and a grouped item, whose rule sits ABOVE the abstention and so
    must still win when the claim is read from a rotation."""
    base = time.time() + 600
    _claimed_item(isolated, 9110)                                  # table: draft, claim: up_next
    _claimed_item(isolated, 9111, to="draft")                      # table: draft, claim: draft
    write_item(isolated, 9112, status="up_next")                   # never triaged: the park
    grouped = write_item(isolated, 9113, status="draft")           # grouped: above the claim
    B.update_frontmatter(grouped, {"tags": ["backlog", "grouped"], "group": 9})
    _steward_move(9113, frm="draft", to="up_next", ts=base)
    S.append_event({"event": "backlog_triage", "item_id": 9113, "verdict": "unverifiable",
                    "check": "grep", "evidence": "e"}, path=S.LEDGER_PATH)

    first = B.desired_statuses(S.LEDGER_PATH, None)
    second = B.desired_statuses(S.LEDGER_PATH, None)
    assert first == second, "a second call on unchanged bytes moved an item"
    assert first[9112][0] == "draft" and PARK in first[9112][1], (
        "an untriaged item in the pool is still parked: the guard is for an unread store")
    assert first[9110][0] == "up_next" and first[9111][0] == "draft"
    assert "umbrella" in first[9113][1] and "board steward" not in first[9113][1], first[9113]

    _fold_the_live_ledger()
    third = B.desired_statuses(S.LEDGER_PATH, None)
    flipped = {iid: (first.get(iid), third.get(iid))
               for iid in set(first) | set(third) if first.get(iid) != third.get(iid)}
    assert not flipped, f"the same events filed into a rotation changed these: {flipped}"


def test_a_folded_claim_leaves_the_park_pair_untouched_on_the_rest(isolated):
    """The fold may not spread: items with no claim of their own are answered exactly as
    before it, so the fix is a wider read for the claim and not a blanket abstention."""
    _claimed_item(isolated, 9120)
    write_item(isolated, 9121, status="up_next")                   # never triaged
    write_item(isolated, 9122, status="draft")                     # never triaged, already home
    before = B.desired_statuses(S.LEDGER_PATH, None)
    _fold_the_live_ledger()
    after = B.desired_statuses(S.LEDGER_PATH, None)
    assert before[9121] == after[9121] == ("draft", PARK)
    assert 9122 not in after, "already where triage looks, with no ledger opinion: untouched"
    assert after[9120][0] == "up_next"


# ── clause 3: a re-derived claim writes nothing ──────────────────────────

def test_applying_the_current_status_writes_no_activity_line(isolated):
    """`reconcile_statuses` on an item whose desired status equals its current one appends
    no `activity_log` entry and leaves the file byte-identical, before the fold and after
    it — which is what makes a claim older than the item's newest status move cost nothing.

    The `before` leg pins the no-op guard the file has had since `30be0518` (2026-09-11) —
    it passed before this change and is kept so a later "refresh the stamp" patch has
    something to fail against. The `after` leg is the one that fails without #4412: the
    fold re-derives `draft` for an item the steward left in `up_next`, the move happens,
    and the file gains an `activity_log` line for a decision nobody made."""
    item = _claimed_item(isolated, 9130, applied=True)     # status up_next, claim up_next
    assert B.item_by_id(9130).status == "up_next"
    before_bytes = item.read_text(encoding="utf-8")
    assert B.reconcile_statuses(S.LEDGER_PATH, None) == [], "a matching status was moved"
    assert item.read_text(encoding="utf-8") == before_bytes, "a no-op pass edited the file"

    _fold_the_live_ledger()
    moved = B.reconcile_statuses(S.LEDGER_PATH, None)
    assert moved == [], f"the fold moved an item nobody decided about: {moved}"
    after = item.read_text(encoding="utf-8")
    assert after == before_bytes, "re-deriving a claim wrote an activity_log entry"
    entries = (B._split_frontmatter(after)[0].get("activity_log") or [])
    assert entries == (B._split_frontmatter(before_bytes)[0].get("activity_log") or []), entries


# ── the asymmetry that parks a pool on an unread store ────────────────────

def test_an_empty_decode_parks_nothing(isolated):
    """A decode that returns no rows is not a board with no verdicts.

    The half-done fold is on the table: the rotation the sweep creates before it has
    written any rows into it, beside a live file the same operation has emptied. Both
    decode to nothing, and the store has demonstrably held verdicts — a rotation only has
    a name because rows went into one. Every arm of the table except the last two needs a
    row it found, so unguarded this is the pass that parks every item in the implement pool
    and hands the next healthy read a pile to pull back. `reconcile_statuses` must therefore
    move nothing and touch no file."""
    write_item(isolated, 9140, status="up_next")
    write_item(isolated, 9141, status="up_next")
    S.append_event({"event": "backlog_triage", "item_id": 9140, "verdict": "confirmed",
                    "check": "grep", "evidence": "e",
                    "acceptance": "the check no longer reproduces"}, path=S.LEDGER_PATH)
    assert B.desired_statuses(S.LEDGER_PATH, None)[9141][1] == PARK, "setup: healthy read parks it"
    _fold_the_live_ledger(empty_payload=True)

    assert B.desired_statuses(S.LEDGER_PATH, None) == {}, "the table answered on no rows"
    assert B.reconcile_statuses(S.LEDGER_PATH, None) == []
    for iid in (9140, 9141):
        assert B.item_by_id(iid).status == "up_next"


def test_a_store_with_no_ledger_at_all_still_answers(isolated):
    """The guard is for a store that has rows and decoded none. A fresh install has no live
    file and no rotations, and its pool must still be reconciled — otherwise this fix would
    silently stop the table answering anywhere the ledger has not been written yet."""
    write_item(isolated, 9150, status="up_next")
    assert not S.LEDGER_PATH.exists()
    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[9150] == ("draft", PARK)


# ===========================================================================
# A status pass may answer only from the store that belongs to the board it writes (#4471)
#
# #4412 stopped the ONE shape of empty decode it could see: a store whose rotation sits
# beside it and decodes nothing. This is the other shape, and #4412's guard cannot reach it.
# A gate or canary subprocess runs with `LLOYD_AUTOMOD_STATE` pointed at
# `<round>/gate-state/automod` (`gate.py::_child_env`, `gate.py::_pickup_env`,
# `canary_config.py::canary_env`) — a store with NO rotation beside it by construction, so
# the `if archives and …` test is false and the pass answers — while its board half is the
# account's own board: `worktree.py:65 HOME_LINK_SKIP` names only `lloyd` and `lloyd-data`,
# so a round's `~/obsidian` is a LINK ONTO the house vault, and `backlog.py:73`
# `BACKLOG_DIR = vault_root() / "backlog"` never looks at the redirected state root.
# `round.py::board_pass` is the reachable caller, and it is reachable on every gate.
#
# Measured over the live board at this round's start, both against the shipped code: with the
# live store the table answers 171 items and park-pairs 1; with a round-shaped scratch store
# (an empty `promotions.jsonl`, no `.gz` beside it) it answers 162 and park-pairs ALL 162 —
# #4409 among them, four times over on 2026-11-21, each park answered minutes later by a
# `draft → up_next` whose reason was `triage confirmed it with an acceptance check`. The
# window's 600 vault park lines over 158 items, every one of them carrying a `confirmed` row
# in the merged corpus, left exactly 1 `up_next → draft` row in the live ledger, because the
# park rows went into the round's store and died with its directory.
# ===========================================================================

def _house(tmp_path, monkeypatch):
    """Declare `tmp_path / "house"` to be the account's home, and hand back its board plus the
    store that belongs to that board.

    Those two paths are the whole of what `status_pass_store_mismatch` compares. Conftest's
    `_point_automod_at_scratch` already aims the pair at scratch for every test; this says so
    per-test over a home the node owns, so the guard's board arm is what is being exercised
    rather than an accident of isolation."""
    home = tmp_path / "house"
    board = home / "obsidian" / "backlog"
    board.mkdir(parents=True)
    store_dir = home / ".local" / "state" / "lloyd-automod"
    store_dir.mkdir(parents=True)
    monkeypatch.setattr(B, "BACKLOG_DIR", board)
    monkeypatch.setattr(B, "PRODUCTION_BOARD_DIR", board)
    monkeypatch.setattr(B, "PRODUCTION_STORE_DIR", store_dir)
    return board, store_dir / "promotions.jsonl"


def _confirm_in(store, item_id: int, status: str) -> None:
    """`_confirm` against an explicit store: these nodes reason about a named store, and the
    fixture-rebound `S.LEDGER_PATH` is a different file."""
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "verdict_source": "structured"}, path=store)
    B.set_status(item_id, status, "triage put it there")


def test_a_round_scratch_store_never_parks_the_board(tmp_path, monkeypatch):
    """#4471 clause 1: the gate/canary shape writes nothing, where the shipped code parks.

    The live figures are in the block comment above; reproduced here at a fraction of that
    volume — twelve open, untriaged, in-pool items where the live burst parked 154 in one pass —
    because the SHAPE is what breaks, not the count: a board that is the account's board, and
    the store a round hands its subprocesses.
    """
    board, paired = _house(tmp_path, monkeypatch)
    ids = list(range(9280, 9292))
    for iid in ids:
        write_item(board, iid, status="up_next")
    scratch = (tmp_path / "SM_20261121_145251" / "gate-state" / "automod"
               / "promotions.jsonl")
    scratch.parent.mkdir(parents=True)
    scratch.write_text("", encoding="utf-8")            # empty, and no .gz beside it
    before = {p.name: p.read_bytes() for p in board.glob("*.md")}
    assert len(before) == 12, "the board has to be standing for the abstention to mean anything"

    why = B.status_pass_store_mismatch(scratch)
    assert why and "4471" in why, f"the refusal must name the store it wanted: {why!r}"
    assert B.desired_statuses(scratch, None) == {}, (
        "a pass reading a store that is not this board's forms no opinion, and above all "
        "none of the `never triaged` park that shipped here")
    assert B.reconcile_statuses(scratch, None) == [], (
        "and writes nothing: the shipped pass moved all twelve of these items")
    assert {p.name: p.read_bytes() for p in board.glob("*.md")} == before, (
        "no status, no body line, no activity entry — an abstention leaves no trace on the "
        "board it abstained from")
    assert scratch.read_text(encoding="utf-8") == "", "nor does it book rows in its own store"

    # The same twelve items, the same board, read from the store that belongs to it: still all
    # twelve parked. That is the control — the guard above refused on identity, and this holds
    # with the guard deleted, which is what makes it a control and not the falsifier.
    assert B.desired_statuses(paired, None) == {iid: ("draft", PARK) for iid in ids}, (
        "a fresh install's own store answers even while empty")


def test_the_refusal_keys_on_the_store_and_not_on_its_rows(tmp_path, monkeypatch):
    """#4471 clause 2: the key is the store's identity, never its row count.

    Emptiness is #4412's key and cannot be this one's, because an empty store is also a fresh
    install — `test_a_store_with_no_ledger_at_all_still_answers` and
    `test_an_empty_decode_parks_nothing` exist to keep that difference alive. The two arms
    below fail in opposite directions: the paired-but-empty arm parks, so it stays green with
    the guard removed; the unpaired-but-populated arm abstains, and it is the falsifier."""
    board, paired = _house(tmp_path, monkeypatch)
    write_item(board, 9293, status="up_next")
    assert not paired.exists()
    assert B.desired_statuses(paired, None)[9293] == ("draft", PARK), (
        "no redirect, no ledger file at all: the board's own store still answers, so the "
        "refusal cannot be reading how many rows it holds")

    fat = tmp_path / "SM_20261122_000000" / "gate-state" / "automod" / "promotions.jsonl"
    fat.parent.mkdir(parents=True)
    fat.write_text("".join(
        json.dumps({"event": "backlog_triage", "item_id": 8000 + n,
                    "verdict": "confirmed", "verdict_source": "structured"}) + "\n"
        for n in range(2000)), encoding="utf-8")
    assert not B.status_pass_store_mismatch(paired), "the board's own store is never refused"
    assert B.status_pass_store_mismatch(fat), (
        "2,000 rows do not make a round's scratch store into the board's own")
    assert B.desired_statuses(fat, None) == {}, (
        "and a pass holding other items' whole triage history still will not park this "
        "board's items from it")


def test_a_pool_park_is_always_booked_in_the_store_the_pass_read(tmp_path, monkeypatch):
    """#4471 clause 4: 600 park lines to 1 ledger row was possible because a write and its
    `status_moved` row could be aimed at different stores, so the board's park count stopped
    being checkable against the ledger that grades it. Every in-pool move has to be paired
    with its row, in the file the pass read."""
    board, paired = _house(tmp_path, monkeypatch)
    for iid in (9296, 9297, 9298):                  # the store's healthy history
        write_item(board, iid)
        _confirm_in(paired, iid, "up_next")
    write_item(board, 9294, status="up_next")       # never triaged: the park
    write_item(board, 9295)
    _confirm_in(paired, 9295, "draft")              # confirmed, parked: the lift

    moved = B.reconcile_statuses(paired, None)
    assert sorted((m["item_id"], m["from"], m["to"]) for m in moved) == [
        (9294, "up_next", "draft"), (9295, "draft", "up_next")]

    rows = [e for e in S.ledger_rows(paired) if e.get("event") == "status_moved"]
    park_rows = sorted((e["item_id"], e["from"], e["to"]) for e in rows
                       if e["from"] == "up_next" and e["to"] == "draft")
    park_writes = sorted((m["item_id"], m["from"], m["to"]) for m in moved
                         if m["from"] == "up_next" and m["to"] == "draft")
    assert park_writes == park_rows == [(9294, "up_next", "draft")], (
        "one park written, exactly one park booked, in the store this pass read: a board "
        "whose parks outnumber its rows is the #4471 blind spot")
    assert {(e["item_id"], e["from"], e["to"]) for e in rows} == {
        (9294, "up_next", "draft"), (9295, "draft", "up_next")}, (
        "no move this pass makes goes unbooked, in either direction")
