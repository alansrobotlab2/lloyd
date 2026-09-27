"""The loop keeps working while there is work (2026-09-27).

Over 2026-09-25..27 autocode was busy 37% of the wall clock and autotriage
13%, with 70 drafts on the board and nothing confirmed. Three causes, each
pinned here: autotriage ran one ~2-minute turn per 900 s interval whatever was
waiting; autocode, finding nothing confirmed, slept a full interval instead of
looking again when triage confirmed something; and a parked draft (ranked
below a round's worth by the sweep) was out of every pool until a person
removed the tag, so an idle loop never reached it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, state as S
from workers.sources import DECLINED
from workers.sources import autotriage as T


def write_item(d: Path, item_id, *, status="draft", tags=("backlog",)) -> Path:
    fm = {"status": status, "priority": "medium", "board": "lloyd", "tags": list(tags),
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    p = d / f"{item_id}-item-{item_id}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\nBody.\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def test_a_parked_draft_is_offered_only_when_nothing_unparked_waits(isolated):
    write_item(isolated, 1, tags=("backlog", B.PARKED_TAG))
    write_item(isolated, 2)
    fresh, _ = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in fresh] == [2], "better-ranked work first"
    B.set_status(2, "done", "gone")
    fresh, _ = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in fresh] == [1], "an idle loop reaches the parked item"


class _Q:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, **kw):
        self.enqueued.append(kw)
        return len(self.enqueued)


def test_triage_looks_again_soon_while_there_is_work_and_room(isolated, monkeypatch):
    write_item(isolated, 3)
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": False})
    q = _Q()
    assert asyncio.run(T.enqueue_if_due(q, {"max_inflight": 2})) == DECLINED
    assert len(q.enqueued) == 2, "both triage slots filled"


def test_triage_keeps_the_interval_on_an_empty_board(isolated, monkeypatch):
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": False})
    assert asyncio.run(T.enqueue_if_due(_Q(), {})) is None


def test_triage_keeps_the_interval_when_the_implement_pool_is_full(isolated, monkeypatch):
    """2026-09-16: back-to-back triage confirmed into a pool the loop could
    not drain (39 held). A full pool is not work autocode needs."""
    write_item(isolated, 4)
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": True})
    assert asyncio.run(T.enqueue_if_due(_Q(), {})) is None


def test_the_shipped_config_retries_triage_and_runs_two():
    from app.config import CONFIG
    cfg = CONFIG["workers"]["sources"]["autotriage"]
    assert cfg["retry_seconds"] == 60 and cfg["max_inflight"] == 2
    slots = CONFIG["workers"]["slots"]
    depth = CONFIG["workers"]["sources"]["autocode"]["max_inflight"]
    owed = CONFIG["workers"]["sources"]["owed-check"]["max_inflight"]
    assert slots >= depth + cfg["max_inflight"] + owed + 1
