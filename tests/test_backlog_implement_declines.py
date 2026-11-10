"""A row written without a `started` of its own never erases an attempt.

`implement_history` drops a `skipped` row together with the `started` it
follows — right for a turn a landing drain refused. `_decline_in_flight` (the
item is in another slot's hands) and the red-tree-healed pickup skip wrote
`skipped` rows with no `started` of their own, so they took the PREVIOUS
attempt's rows with them: a lower attempt count, a lost re-offer banner, and —
for a decline beside a live turn — the holder's own `started` row.
"""
from __future__ import annotations

import pytest

from scripts.automod import backlog as B, state as S


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", path)
    monkeypatch.setattr(B, "BACKLOG_DIR", tmp_path / "backlog")
    return path


def _row(ledger, phase, **kw):
    S.append_event({"event": "backlog_implement", "item_id": 7, "phase": phase, **kw},
                   path=ledger)


def _phases(ledger):
    return [r["phase"] for r in B.implement_history(ledger, with_archives=False).get(7, [])]


def test_a_decline_beside_a_live_turn_keeps_the_holders_started_row(ledger):
    _row(ledger, "started", slot="autocode:round")
    _row(ledger, "declined", slot="autocode:round:1", held_by="autocode:round")
    assert _phases(ledger) == ["started"]
    assert 7 in B.items_with_unfinished_rounds(ledger), "the holder's turn is in flight"


def test_a_decline_written_as_skipped_before_the_phase_existed_is_still_a_decline(ledger):
    _row(ledger, "started", slot="autocode:round")
    _row(ledger, "skipped", slot="autocode:round:1", held_by="autocode:round")
    assert _phases(ledger) == ["started"]


def test_a_healed_red_tree_skip_keeps_the_previous_attempt(ledger):
    _row(ledger, "started", slot="autocode:round")
    _row(ledger, "finished", slot="autocode:round", round_id="SM_1", stop_reason="max_turns")
    for phase in ("declined", "skipped"):     # today's spelling, and the #3709 row's
        _row(ledger, phase, slot="autocode:round", reason="red tree healed before the attempt")
        assert _phases(ledger) == ["started", "finished"], phase


def test_a_drain_refusal_still_takes_its_own_started_row(ledger):
    _row(ledger, "started", slot="autocode:round")
    _row(ledger, "skipped", reason="landing in progress: 503")
    assert _phases(ledger) == []
