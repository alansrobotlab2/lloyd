"""A single-item triage that keeps running out of clock is retired, not retried forever.

The `max_turns` branch capped incomplete triages at `MAX_INCOMPLETE_ATTEMPTS`;
the `TurnTimeout` branch wrote `incomplete` and never checked the cap, so a
deterministically top-ranked item that always timed out was re-triaged on
every run.
"""
from __future__ import annotations

import asyncio

import pytest

from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autotriage as M

from tests.test_autotriage import write_item


@pytest.fixture(autouse=True)
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    B._TRIAGE_CLAIMS.clear()
    yield d
    B._TRIAGE_CLAIMS.clear()


class _QItem:
    def __init__(self):
        self.payload = {"group_triage": False, "implement_pool_floor": 40}
        self.id = 1


def _triage_rows(item_id):
    return [e for e in S.read_events(path=S.LEDGER_PATH, limit=10**6)
            if e.get("event") == "backlog_triage" and e.get("item_id") == item_id]


def test_a_triage_that_always_times_out_is_retired_at_the_incomplete_cap(board, monkeypatch):
    write_item(board, 7, days_old=300)
    calls = []

    async def times_out(prompt, **kw):
        calls.append(1)
        raise C.TurnTimeout("exceeded 3540s", session_id=f"s{len(calls)}", cancelled=True)
    monkeypatch.setattr(C, "run_prompt_in_session", times_out)

    outs = [asyncio.run(M.execute(_QItem())) for _ in range(B.MAX_INCOMPLETE_ATTEMPTS + 1)]

    assert len(calls) == B.MAX_INCOMPLETE_ATTEMPTS, \
        f"the item was triaged {len(calls)} times; the cap is {B.MAX_INCOMPLETE_ATTEMPTS}"
    verdicts = [r["verdict"] for r in _triage_rows(7)]
    assert verdicts == [B.INCOMPLETE] * (B.MAX_INCOMPLETE_ATTEMPTS - 1) + ["unverifiable"], verdicts
    last = _triage_rows(7)[-1]
    assert last["verdict_source"] == "turn_timeout" and last["session_id"] == "s2", last
    assert outs[B.MAX_INCOMPLETE_ATTEMPTS - 1]["verdict"] == "unverifiable"
    assert outs[-1]["status"] == "skipped", "nothing left to triage after the cap"
