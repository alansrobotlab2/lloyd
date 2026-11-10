"""A confirmation the 14-day fold has archived is still a confirmation to the
status machine and to the dispatcher.

`triaged_ids`, `held_confirmations` and `implement_outcomes` have read the
`promotions-archive-*.jsonl.gz` rotations since #3612; `desired_statuses` and
`ready_confirmed` asked `confirmed_verdicts` for the live file alone. An item
confirmed, parked in `up_next` and not yet attempted therefore lost its
confirmation the day its row rotated while keeping its verdict: it matched the
`iid in verdicts` branch and was moved to `draft` as "triaged confirmed; not for the
unattended loop" — out of the implement pool, and out of the triage pool because it
is triaged. #3024, #3041, #3068, #3196 and #3284 were moved that way on
2026-11-03..07, each right after a fold.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

from scripts.automod import backlog as B
from scripts.automod import state as S

NOW = time.time()
OLD = NOW - 20 * 86400                      # past the sweep's 14-day window
MONTH = time.strftime("%Y%m", time.gmtime(OLD))
ITEM = 3024
ROW = {"event": "backlog_triage", "item_id": ITEM, "verdict": "confirmed",
       "verdict_source": "structured", "ts": OLD, "surface": "code",
       "check": "grep -n x", "evidence": "still there",
       "acceptance": "The check no longer reproduces."}


def _clear():
    S._rows_cache.clear()
    S._rows_reads.clear()
    if hasattr(B, "_clear_caches"):
        B._clear_caches()


@pytest.fixture()
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    (d / f"{ITEM}-a-confirmed-item.md").write_text(
        "---\ntype: backlog\nsegment: backlog\nstatus: up_next\npriority: low\n"
        "board: lloyd\nblocked: false\nassigned: false\ntags: [backlog]\n---\n\n"
        "# A confirmed item\n\nBody.\n", encoding="utf-8")
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(B, "implement_claimed_ids", lambda: set())
    return d


def _store(tmp_path: Path, name: str, *, live: bool, archived: bool) -> Path:
    store = tmp_path / name
    store.mkdir()
    line = json.dumps(ROW) + "\n"
    (store / "promotions.jsonl").write_text(line if live else "", encoding="utf-8")
    if archived:
        (store / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(
            gzip.compress(line.encode("utf-8")))
    _clear()
    return store / "promotions.jsonl"


def _answers(ledger: Path):
    desired = B.desired_statuses(ledger).get(ITEM)
    ready = [item.id for item, _ in B.ready_confirmed(ledger)]
    return (desired[0] if desired else None), ready


def test_the_item_reads_the_same_before_and_after_its_row_is_folded(board, tmp_path):
    before = _answers(_store(tmp_path, "live", live=True, archived=False))
    after = _answers(_store(tmp_path, "folded", live=False, archived=True))
    # Not vacuous: the live layout really is a pool item autocode would take.
    assert before == ("up_next", [ITEM]), before
    assert after == before, (
        "archiving the row moved the item; that is the up_next → draft strand")


def test_a_row_that_is_gone_rather_than_archived_does_move_it(board, tmp_path):
    """The negative control: the fixture distinguishes archived from absent."""
    status, ready = _answers(_store(tmp_path, "dropped", live=False, archived=False))
    assert status == "draft" and ready == []


def test_an_archived_retriage_mark_still_voids_an_archived_confirmation(board, tmp_path):
    store = tmp_path / "marked"
    store.mkdir()
    mark = {"event": "backlog_retriage", "item_id": ITEM, "ts": OLD + 3600}
    (store / "promotions.jsonl").write_text("", encoding="utf-8")
    (store / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(gzip.compress(
        (json.dumps(ROW) + "\n" + json.dumps(mark) + "\n").encode("utf-8")))
    _clear()
    ledger = store / "promotions.jsonl"
    assert ITEM not in B.confirmed_verdicts(ledger, with_archives=True)
    assert [i.id for i, _ in B.ready_confirmed(ledger)] == []
