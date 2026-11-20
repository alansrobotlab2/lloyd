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
