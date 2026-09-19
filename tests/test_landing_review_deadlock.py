"""A landing's drain must not starve the reviews of the rounds it is waiting for.

The night of 2026-09-18, four rounds at a time, by the ledger:

  * 17 of 51 review attempts ended `review could not run: HTTP 503 ... Lloyd is
    landing a code update` (16) or timed out (1);
  * the backend spent 189 of 840 minutes draining, in seven spans of 12-36
    minutes; every landing that did not collide drained for under two;
  * about eleven rounds were aborted with their change finished, committed and
    green on every rung that ran. #832 took five rounds, #800 four, and
    #1250, #789 and #874 three each.

The cycle: a landing arms the drain and waits for every turn in flight; a
sibling round's turn is waiting on its gate; that gate's review rung needs a
grader turn on the draining backend. Only the grader's 420 s give-up broke it.

Three fixes, pinned here:

  1. the drain admits a review grader while something else is still running;
  2. the reaper re-gates a round whose only failed rung was an unreachable
     grader, instead of aborting it, and the item is not offered meanwhile;
  3. `wait_for_rounds` does not take ONE unanswered probe for "cannot say",
     and what it saw goes on the ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.automod import backlog as B, promote as P, round as R, state as S, worktree as W
from workers.sources import _common as C
from workers.sources import autocode as I

HEAD = "255ed67aa0c2d1e4f5a6b7c8d9e0f1a2b3c4d5e6"
RID = "SM_20260919_041432"
ITEM = 874
DETAIL = ('review could not run: HTTP 503: {"detail":"Lloyd is landing a code update; retry in '
          '164s."} (backend still unavailable after 420s of 420s) — the grader, not the diff')


# ── 1. the drain admits a grader beside a running turn ──────────────────────

@pytest.fixture
def sessions(tmp_path, monkeypatch):
    import app.paths as paths
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path)

    def write(sid, **fields):
        (tmp_path / f"{sid}.json").write_text(json.dumps(
            {"session_id": sid, "platform": "worker", "source": "automod-review",
             "messages": [], **fields}), encoding="utf-8")
        return sid
    return write


def test_the_grader_the_review_rung_mints_is_the_one_admitted(sessions, tmp_path):
    """The real writer, so the two halves cannot drift apart by one string."""
    from app.routers import automod as A
    from scripts.automod import review as RV
    sid = RV.write_session(tmp_path, item_id=ITEM, round_id=RID, model="primary")
    assert A.drain_admits(sid, busy=True)


def test_a_grader_is_refused_on_a_quiet_backend(sessions):
    """The drain's own guarantee. The promoter restarts after N QUIET polls;
    a grader let onto a quiet backend could be the turn that restart kills."""
    from app.routers import automod as A
    assert not A.drain_admits(sessions("20260919_215626_review_a1b2"), busy=False)


@pytest.mark.parametrize("fields", [
    {"source": "mission-control"},
    {"source": None},
    {"platform": "mission-control"},
    {"platform": None},
])
def test_nothing_but_a_grader_is_admitted(sessions, fields):
    from app.routers import automod as A
    assert not A.drain_admits(sessions("20260919_215626_autocode_a1b2", **fields), busy=True)


def test_an_unknown_or_unreadable_session_is_refused(sessions, tmp_path):
    from app.routers import automod as A
    assert not A.drain_admits("", busy=True)
    assert not A.drain_admits("20260919_000000_review_dead", busy=True)
    (tmp_path / "20260919_000001_review_beef.json").write_text("{not json", encoding="utf-8")
    assert not A.drain_admits("20260919_000001_review_beef", busy=True)


def test_busy_is_read_from_the_backends_own_turn_count(sessions, monkeypatch):
    import app.sessions_io as sio
    from app.routers import automod as A
    sid = sessions("20260919_215626_review_a1b2")
    monkeypatch.setattr(sio, "active_turn_summary",
                        lambda: {"active": 0, "queued": 0, "harness_runs": 0})
    assert not A.drain_admits(sid)
    monkeypatch.setattr(sio, "active_turn_summary",
                        lambda: {"active": 0, "queued": 0, "harness_runs": 1})
    assert A.drain_admits(sid)

    def boom():
        raise RuntimeError("no")
    monkeypatch.setattr(sio, "active_turn_summary", boom)
    assert not A.drain_admits(sid), "a liveness read that raises is a refusal"


def test_the_chat_endpoint_consults_it():
    """The predicate is only a fix where the 503 is raised."""
    import inspect
    from app.routers import messages
    src = inspect.getsource(messages)
    assert "drain_active() and not drain_admits(session_id)" in src


# ── 2. the reaper re-gates an unreviewed round ──────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    import app.sessions_io as sio
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(C, "source_inner_voice", lambda source, default=True: False)
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [])
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    world = {"head": HEAD, "aborted": [], "spawned": [], "alive": {424242}}
    monkeypatch.setattr(W, "head", lambda path: world["head"])
    monkeypatch.setattr(R, "abort", lambda rid, reason="": world["aborted"].append(rid)
                        or {"aborted": rid})

    def spawn(argv, log, cwd=None):
        world["spawned"].append([str(a) for a in argv])
        return 424242
    monkeypatch.setattr(S, "spawn_detached", spawn)
    monkeypatch.setattr(S, "pid_alive", lambda pid: int(pid or 0) in world["alive"])
    fm = {"status": "in_progress", "priority": "medium", "board": "lloyd", "tags": [],
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    (d / f"{ITEM}-a-thing.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
        encoding="utf-8")
    return world


def _gate(*, rungs=None, head=HEAD, ok=False):
    (S.ROUNDS_DIR / RID).mkdir(parents=True, exist_ok=True)
    if rungs is None:
        rungs = [{"name": "tests", "ok": True, "detail": "6201 passed"},
                 {"name": "review", "ok": False, "detail": DETAIL,
                  "data": {"external_blocker": True, "external_reason": "grader unreachable"}}]
    (S.ROUNDS_DIR / RID / "gate.json").write_text(json.dumps(
        {"round_id": RID, "ok": ok, "head": head, "base": "768897f", "rungs": rungs}),
        encoding="utf-8")


MET = {"landed": False, "acceptance": "met", "deferred_to": [], "summary": "", "spawned": []}


def _finished(outcome=MET, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": ITEM, "phase": "finished",
                    "round_id": RID, "session_id": "s-874", "stop_reason": stop_reason,
                    "outcome": outcome}, path=S.LEDGER_PATH)


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def test_a_round_whose_grader_was_unreachable_is_gated_again(env):
    """#874 at 22:07 on 2026-09-18: `met`, commit 255ed67, "needs a re-gate from
    a fresh session once the backend is idle" — and the reaper aborted it in
    the same second."""
    _gate()
    _finished()
    out = I.reap_abandoned_rounds()
    assert env["aborted"] == [], "a finished change was thrown away over a sibling's landing"
    assert env["spawned"] and env["spawned"][0][-2:] == ["gate", RID]
    assert [(r["round_id"], r["verb"]) for r in out] == [(RID, "gating")]
    marker = S.gate_in_progress(RID)
    assert marker["by"] == "reaper" and marker["head"] == HEAD
    row = _events("gate_rescued")[0]
    assert row["item_id"] == ITEM and row["head"] == HEAD and "could not run" in row["reason"]
    assert not _events("round_abandoned")


def test_the_running_regate_is_left_alone_and_its_pass_is_landed(env):
    _gate()
    _finished()
    I.reap_abandoned_rounds()
    assert I.reap_abandoned_rounds() == [] and len(env["spawned"]) == 1
    # The gate finishes green: `run_gate` clears its marker and the report says ok.
    S.clear_gate_marker(RID)
    _gate(ok=True, rungs=[{"name": "review", "ok": True, "detail": "review: 12 met of 12"}])
    out = I.reap_abandoned_rounds()
    assert [(r["round_id"], r["verb"]) for r in out] == [(RID, "landing")]
    assert env["spawned"][-1][-2:] == ["land", RID] and env["aborted"] == []


def test_a_graded_refusal_after_the_regate_closes_as_it_always_has(env):
    _gate()
    _finished()
    I.reap_abandoned_rounds()
    S.clear_gate_marker(RID)
    _gate(rungs=[{"name": "review", "ok": False, "detail": "review sent it back (1/2)",
                  "data": {"blocking": True}}])
    I.reap_abandoned_rounds()
    assert env["aborted"] == [RID] and len(_events("gate_rescued")) == 1


def test_a_grader_that_stays_down_is_capped(env):
    _gate()
    _finished()
    for _ in range(I.REGATE_CAP):
        I.reap_abandoned_rounds()
        S.clear_gate_marker(RID)      # that gate ended the same way: gate.json unchanged
        assert env["aborted"] == []
    assert len(_events("gate_rescued")) == I.REGATE_CAP
    I.reap_abandoned_rounds()
    assert env["aborted"] == [RID], "re-gated forever against a grader that never came back"


@pytest.mark.parametrize("rungs", [
    # the tests rung is red too: not a grader problem
    [{"name": "tests", "ok": False, "detail": "3 failed"},
     {"name": "review", "ok": False, "detail": DETAIL, "data": {"external_blocker": True}}],
    # an external blocker on another rung is the re-offer's business, as before
    [{"name": "tests", "ok": False, "detail": "red at base", "data": {"external_blocker": True}}],
    # a review that RAN and refused
    [{"name": "review", "ok": False, "detail": "clause 5 has no evidence", "data": {}}],
])
def test_any_other_red_gate_closes_as_it_always_has(env, rungs):
    _gate(rungs=rungs)
    _finished()
    I.reap_abandoned_rounds()
    assert env["aborted"] == [RID] and env["spawned"] == []


def test_a_commit_made_after_that_gate_is_not_what_it_judged(env):
    _gate(head="0" * 40)
    _finished()
    I.reap_abandoned_rounds()
    assert env["aborted"] == [RID] and env["spawned"] == []


@pytest.mark.parametrize("verdict", ["unnecessary", "rejected"])
def test_a_turn_that_ended_on_an_item_verdict_is_not_regated(env, verdict):
    _gate()
    _finished({**MET, "acceptance": verdict})
    I.reap_abandoned_rounds()
    assert env["spawned"] == []


@pytest.mark.parametrize("blocked", ["is_halted", "is_broken"])
def test_nothing_is_gated_while_the_loop_is_stopped(env, monkeypatch, blocked):
    monkeypatch.setattr(S, blocked, lambda: True)
    _gate()
    _finished()
    I.reap_abandoned_rounds()
    assert env["spawned"] == [] and env["aborted"] == [RID]


def test_the_kill_switch(env, monkeypatch):
    monkeypatch.setattr(I, "_source_cfg", lambda name: {"regate_unreviewed": False})
    _gate()
    _finished()
    I.reap_abandoned_rounds()
    assert env["spawned"] == [] and env["aborted"] == [RID]


def test_the_item_is_not_offered_to_a_second_round_meanwhile(env):
    """Its outcome still reads `external` — a re-offer — until the gate ends."""
    S.write_gate_marker(RID, pid=424242, head=HEAD, by="reaper")
    _finished()
    assert B.items_being_gated_or_landed(S.LEDGER_PATH) == {ITEM}
    env["alive"].clear()
    assert B.items_being_gated_or_landed(S.LEDGER_PATH) == set(), \
        "a dead marker must not hold the item out of the pool for good"


def test_the_tool_and_the_reaper_share_one_spawn():
    import inspect
    from agent_mcp import automod as tool
    assert "R.gate_detached(" in inspect.getsource(tool._gate_detached)
    assert "spawn_detached" not in inspect.getsource(tool._gate_detached).split('"""')[-1]


# ── 3. the wait for the other rounds' turns ─────────────────────────────────

def _probes(monkeypatch, answers):
    seq = iter(answers)
    seen = []

    def probe():
        a = next(seq)
        seen.append(a)
        return a
    monkeypatch.setattr(P, "pool_in_flight", probe)
    monkeypatch.setattr(P, "ROUNDS_UNREADABLE_POLLS", 6)   # production's; conftest runs tests at 1
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    return seen


AUTOCODE = [{"source": "autocode"}]


def test_one_unanswered_probe_does_not_end_the_wait(monkeypatch):
    """21:46:08 on 2026-09-18: three sibling turns in flight, a post-session
    capture on the event loop, one 5 s probe unanswered — and the drain armed
    at 21:46:13 for the next 24 minutes."""
    seen = _probes(monkeypatch, [AUTOCODE, None, AUTOCODE, None, None, AUTOCODE, []])
    ok, why = P.wait_for_rounds(600)
    assert ok and why == "no autocode turn in flight" and len(seen) == 7


def test_a_backend_that_cannot_say_for_a_minute_still_goes_to_the_drain(monkeypatch):
    seen = _probes(monkeypatch, [AUTOCODE] + [None] * 6)
    ok, why = P.wait_for_rounds(600)
    assert ok and "unreadable for 6 consecutive polls" in why and len(seen) == 7


def test_other_sources_are_not_waited_for(monkeypatch):
    _probes(monkeypatch, [[{"source": "autotriage"}, {"source": "scheduled-task"}]])
    assert P.wait_for_rounds(600) == (True, "no autocode turn in flight")


def test_what_the_wait_saw_goes_on_the_ledger():
    import inspect
    src = inspect.getsource(R.land)
    assert '"event": "land_wait_rounds"' in src
    assert src.index("land_wait_rounds") < src.index("_land_lock("), \
        "recorded before the lock, where the wait runs"
