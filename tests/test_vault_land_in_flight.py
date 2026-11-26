"""A vault land announces itself while it runs, and a second call is told so (#4583).

`automod_vault_land` is synchronous inside lloyd-mcp and can outlast the 660 s
tool-call ceiling: #4429's land measured `seconds: 1529` against `budget: 900`, with
`lock_wait_s: 629` queued behind a running gate's tests lock. Its caller saw a
transport timeout on a healthy land, and nothing it could read said "in flight".

Pinned here, against a scratch state dir:

* `vault_round.land` writes a marker before any work and removes it whatever the
  outcome (returned or raised);
* the marker names item, session, paths and start, and its reader derives the stage
  (`review` until a `vault_review` row for the item lands after the start, then
  `committing`) and the live holder of the gate's tests lock;
* a second call for the same item while the first is live is refused with that state
  and says not to call again — never run twice;
* a marker whose pid is gone (lloyd-mcp restarted under the land) is not in flight;
* the marker lives beside the ledger, so isolating `LEDGER_PATH` isolates it.

The finalizer's half is in `tests/test_automod_review.py` (#4583 section).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import pytest

from scripts.automod import state as S
from scripts.automod import vault_round as V


@pytest.fixture(autouse=True)
def _scratch_state(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VAULT_LAND_DIR", tmp_path / "vault_lands")
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "GATE_TESTS_LOCK_PATH", tmp_path / "gate-tests.lock")
    return tmp_path


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_land_holds_a_marker_while_it_runs_and_drops_it_after(monkeypatch):
    seen = {}

    def fake_land(paths, message, **kw):
        seen["rec"] = S.vault_land_in_progress(item_id=4429)
        return {"ok": True}

    monkeypatch.setattr(V, "_land", fake_land)
    assert V.land(["backlog/a.md"], "msg", item_id=4429, session_id="s1") == {"ok": True}
    rec = seen["rec"]
    assert rec is not None, "no marker was live while the land ran"
    assert rec["item_id"] == 4429 and rec["paths"] == ["backlog/a.md"]
    assert rec["session_id"] == "s1" and rec["pid"] == os.getpid()
    assert rec["stage"] == "review"
    assert S.read_vault_land_marker(item_id=4429) is None, "the marker outlived the land"


def test_a_land_that_raises_drops_its_marker_too(monkeypatch):
    def boom(paths, message, **kw):
        raise V.VaultRoundError("validation failed")

    monkeypatch.setattr(V, "_land", boom)
    with pytest.raises(V.VaultRoundError):
        V.land(["backlog/a.md"], "msg", item_id=7)
    assert S.read_vault_land_marker(item_id=7) is None


def test_a_second_call_while_the_first_is_live_is_refused_with_its_state(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_land(paths, message, **kw):
        calls.append(paths)
        started.set()
        release.wait(10)
        return {"ok": True}

    monkeypatch.setattr(V, "_land", slow_land)
    t = threading.Thread(target=V.land, args=(["backlog/a.md"], "m"),
                         kwargs={"item_id": 4429, "session_id": "s1"})
    t.start()
    try:
        assert started.wait(10)
        with pytest.raises(V.VaultRoundError) as exc:
            V.land(["backlog/a.md"], "m", item_id=4429, session_id="s2")
    finally:
        release.set()
        t.join(10)
    msg = str(exc.value)
    assert "already in flight" in msg and "item #4429" in msg, msg
    assert "Do not call it again" in msg and "backlog/a.md" in msg, msg
    assert calls == [["backlog/a.md"]], "the second call ran the land again"
    # Once the first returns, a call for the item proceeds normally.
    assert V.land(["backlog/a.md"], "m", item_id=4429) == {"ok": True}


def test_a_marker_whose_process_is_gone_is_not_in_flight(monkeypatch):
    S.write_vault_land_marker(item_id=11, paths=["a.md"], pid=_dead_pid())
    assert S.read_vault_land_marker(item_id=11) is not None
    assert S.vault_land_in_progress(item_id=11) is None
    calls = []
    monkeypatch.setattr(V, "_land", lambda paths, message, **kw: calls.append(1) or {})
    V.land(["a.md"], "m", item_id=11)
    assert calls == [1], "a dead marker blocked a fresh land"


def test_the_stage_moves_to_committing_once_the_review_row_lands():
    rec = S.write_vault_land_marker(item_id=4599, paths=["autonomy/97-x.md"])
    assert S.vault_land_in_progress(item_id=4599)["stage"] == "review"
    S.append_event({"event": "vault_review", "item_id": 4598, "kind": "pass"})
    assert S.vault_land_in_progress(item_id=4599)["stage"] == "review", \
        "another item's review moved this land's stage"
    S.append_event({"event": "vault_review", "item_id": 4599, "kind": "pass"})
    live = S.vault_land_in_progress(item_id=4599)
    assert live["stage"] == "committing (review pass)", live
    assert live["started_at"] == rec["started_at"]
    assert live["elapsed_s"] >= 0


def test_a_live_tests_lock_holder_is_named_and_a_stale_payload_is_not(_scratch_state):
    S.write_vault_land_marker(item_id=5, paths=["a.md"])
    lock = S.GATE_TESTS_LOCK_PATH
    lock.write_text(json.dumps({"pid": _dead_pid(), "owner": "gate-SM_old",
                                "since": time.time()}))
    assert "tests_lock_holder" not in S.vault_land_in_progress(item_id=5)
    # A live automod-shaped holder: this process, made to read as one.
    lock.write_text(json.dumps({"pid": os.getpid(), "owner": "gate-SM_live",
                                "since": time.time()}))
    import scripts.automod.state as ST
    orig = ST.marker_pid_is_live
    try:
        ST.marker_pid_is_live = lambda pid, since=None: pid == os.getpid()
        live = S.vault_land_in_progress(item_id=5)
    finally:
        ST.marker_pid_is_live = orig
    assert live["tests_lock_holder"]["owner"] == "gate-SM_live", live
    assert "gate-SM_live" in S.vault_land_in_flight_text(live)


def test_an_item_less_land_is_keyed_by_session_and_one_without_either_writes_nothing(
        monkeypatch):
    seen = {}

    def fake(paths, message, **kw):
        seen["by_session"] = S.vault_land_in_progress(session_id="s9")
        return {}

    monkeypatch.setattr(V, "_land", fake)
    V.land(["a.md"], "m", session_id="s9")
    assert seen["by_session"] is not None
    assert S.vault_land_marker_path() is None  # neither item nor session: no key
    S.VAULT_LAND_DIR.mkdir(parents=True, exist_ok=True)
    V.land(["a.md"], "m")
    assert not list((S.VAULT_LAND_DIR).glob("*.json"))


def test_the_marker_lives_beside_the_ledger_unless_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VAULT_LAND_DIR", None)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "elsewhere" / "promotions.jsonl")
    assert S.vault_land_marker_path(item_id=3) == tmp_path / "elsewhere" / "vault_lands" / "item-3.json"
