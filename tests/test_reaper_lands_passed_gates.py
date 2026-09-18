"""A round whose gate passed and whose turn is over is landed, not closed.

#278's rescue, made deterministic. That round died at its iteration cap with
the change written; two minutes later the Inner Voice observer sent "if the
gate passed, land it" into the session, and the second turn landed it. Cut 1
of senses-not-supervision (2026-09-12) switched the observer off for unattended
turns and gave its stall, budget and context nudges deterministic replacements
— but not that one. From then on a round that was gated and green when its
turn ended was aborted, and its item re-offered for a whole new round.

2026-09-18, by the ledger:

  round    item   gate started   turn ended   gate PASSED   reaper
  070314   #1129  minute 37      00:57:22     00:58:05      aborted 01:02
  083651   #1121  minute 45      02:20:53     02:25:57      aborted 02:41
  121744   #1236  minute 42      06:15:03     06:22:15      aborted 06:34
  122306   #1234  minute 40      06:14:31     06:13:44      aborted 06:14  (saw the pass, 146 s left)

All four landed on their next round — 250 of that day's 994 implement minutes
went to redoing changes that had already passed every rung.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.automod import backlog as B, round as R, state as S, worktree as W
from workers.sources import _common as C
from workers.sources import autocode as I

HEAD = "08a059b3c2d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7"
RID = "SM_20260918_122306"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """The reaper's world: one open round with a worktree, nothing running in
    its session, no observer, the loop enabled."""
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
    world = {"head": HEAD, "aborted": [], "spawned": []}
    monkeypatch.setattr(W, "head", lambda path: world["head"])
    monkeypatch.setattr(R, "abort", lambda rid, reason="": world["aborted"].append(rid)
                        or {"aborted": rid})

    def spawn(argv, log, cwd=None):
        world["spawned"].append([str(a) for a in argv])
        return 424242
    monkeypatch.setattr(S, "spawn_detached", spawn)
    monkeypatch.setattr(S, "pid_alive", lambda pid: int(pid or 0) == 424242)
    fm = {"status": "in_progress", "priority": "medium", "board": "lloyd", "tags": [],
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    (d / "1234-a-thing.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n", encoding="utf-8")
    return world


def _gate(ok=True, head=HEAD):
    (S.ROUNDS_DIR / RID).mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / RID / "gate.json").write_text(json.dumps(
        {"round_id": RID, "ok": ok, "head": head, "base": "9fab95d", "rungs": [
            {"name": "review", "ok": ok, "detail": "review: 4 met of 4 clause(s); fine"}]}),
        encoding="utf-8")


def _finished(outcome=None, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": 1234, "phase": "finished",
                    "round_id": RID, "session_id": "s-1234", "stop_reason": stop_reason,
                    "outcome": outcome}, path=S.LEDGER_PATH)


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


DEFERRED = {"landed": False, "acceptance": "deferred", "deferred_to": [1234], "summary": "",
            "spawned": [], "clause_outcomes": []}


# ── it lands ────────────────────────────────────────────────────────────────

def test_a_passed_gate_at_the_rounds_own_commit_is_landed(env):
    """#1234's first round, to the second: the pass at 06:13:44, the turn's
    end at 06:14:31 with `acceptance: deferred`, the abort in the same second."""
    _gate()
    _finished(DEFERRED)
    out = I.reap_abandoned_rounds()
    assert env["aborted"] == [], "a change that had passed every rung was thrown away"
    assert env["spawned"] and env["spawned"][0][-3:] == ["scripts.automod.round", "land", RID][-3:]
    assert env["spawned"][0][-2:] == ["land", RID]
    assert [(r["round_id"], r["verb"]) for r in out] == [(RID, "landing")]
    marker = S.land_in_progress(RID)
    assert marker["by"] == "reaper" and marker["pid"] == 424242
    row = _events("land_rescued")[0]
    assert row["item_id"] == 1234 and row["head"] == HEAD and "gate passed at 08a059b3" in row["reason"]
    assert not _events("round_abandoned") and not _events("round_aborted")
    fm = B._split_frontmatter(next(B.BACKLOG_DIR.glob("1234-*.md")).read_text())[0]
    assert "the loop started one" in fm["activity_log"][-1]
    assert fm["status"] == "in_progress", "not handed back to the pool under its own landing"


def test_a_turn_that_died_with_no_finalizer_is_landed_too(env):
    """`max_turns` and the wall clock leave no outcome at all. The item then
    closes on the review rung's grading (`backlog.code_review_outcome`)."""
    _gate()
    _finished(None, stop_reason="max_turns")
    I.reap_abandoned_rounds()
    assert env["spawned"] and env["aborted"] == []


def test_the_second_pass_leaves_the_running_landing_alone(env):
    _gate()
    _finished(DEFERRED)
    I.reap_abandoned_rounds()
    assert I.reap_abandoned_rounds() == []
    assert len(env["spawned"]) == 1 and env["aborted"] == []


# ── it does not ─────────────────────────────────────────────────────────────

def _expect_the_old_behaviour(env):
    before = len(_events("land_rescued"))
    out = I.reap_abandoned_rounds()
    assert env["spawned"] == [] and env["aborted"] == [RID]
    assert [(r["round_id"], r.get("verb")) for r in out] == [(RID, None)]
    assert len(_events("land_rescued")) == before, "a second rescue was recorded"


def test_a_commit_made_after_the_gate_was_never_judged(env):
    _gate()
    env["head"] = "f" * 40
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


def test_a_gate_that_failed_closes_as_it_always_has(env):
    _gate(ok=False)
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


def test_a_round_that_was_never_gated_closes_as_it_always_has(env):
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("verdict", ["rejected", "unnecessary"])
def test_a_turn_that_ended_on_an_item_verdict_decided_not_to_land(env, verdict):
    """Built it, gated it, measured no gain: the author's call, and a passed
    gate does not overrule it."""
    _gate()
    _finished({**DEFERRED, "acceptance": verdict, "summary": "p50 412 -> 409 ms; noise"})
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("prior", ["land_failed", "land_rescued"])
def test_one_rescue_per_round(env, prior):
    """A landing that failed is a ledger verdict the re-offer already reads;
    retrying it from here would loop on whatever refused it."""
    _gate()
    _finished(DEFERRED)
    S.append_event({"event": prior, "round_id": RID}, path=S.LEDGER_PATH)
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("blocked", ["is_halted", "is_broken", "disabled", "switch"])
def test_nothing_is_started_that_round_land_would_refuse(env, monkeypatch, blocked):
    _gate()
    _finished(DEFERRED)
    if blocked == "disabled":
        monkeypatch.setattr(S, "is_enabled", lambda repo=None: False)
    elif blocked == "switch":
        monkeypatch.setattr(I, "_source_cfg", lambda name: {"land_passed_gates": False})
    else:
        monkeypatch.setattr(S, blocked, lambda: True)
    _expect_the_old_behaviour(env)


def test_a_landing_that_cannot_start_falls_back_to_the_close(env, monkeypatch):
    _gate()
    _finished(DEFERRED)
    monkeypatch.setattr(R, "land_detached", lambda rid, by: {"error": "no worktree"})
    _expect_the_old_behaviour(env)


def test_a_gate_still_running_is_left_for_the_next_pass_and_then_landed(env):
    """The other three rounds: the turn ended first. The reaper must not touch
    a round mid-gate, and must come back for it."""
    _finished(DEFERRED)
    S.write_gate_marker(RID, pid=424242, head=HEAD, by="automod_gate")
    assert I.reap_abandoned_rounds() == [] and env["aborted"] == [] and env["spawned"] == []
    S.clear_gate_marker(RID)
    _gate()
    assert [r["verb"] for r in I.reap_abandoned_rounds()] == ["landing"]


# ── and it comes back within a retry, not an interval ───────────────────────

def test_a_held_loop_runs_the_reaper_at_the_retry_cadence(env, monkeypatch):
    """The reaper ran at turn end and then with housekeeping, up to 900 s
    later — and an open round with a passed gate is what holds the other
    slots (`_rounds_about_to_land`). #1236's gate passed at 06:22:15 and its
    round was closed at 06:34:14."""
    from workers.sources import DECLINED
    _gate()
    _finished(DEFERRED)
    monkeypatch.setattr(I, "_boot_settled", {"done": True})
    monkeypatch.setattr(I, "_housekeeping_due", lambda queue, cfg: False)
    monkeypatch.setattr(I, "_loop_is_free",
                        lambda depth=None: (False, f"{RID} passed its gate and is about to land"))
    out = asyncio.run(I.enqueue_if_due(object(), {"interval_seconds": 900}))
    assert out is DECLINED or out == DECLINED
    assert env["spawned"], "the round sat until housekeeping's next tick"
