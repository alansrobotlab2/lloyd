"""The land train: a merge is a landing, a restart is a flush.

Every restarting landing used to drain the loop to zero first: wait out the
sibling rounds' turns, pause the pool, wait for idle, restart, verify. 172 of
them in the week to 2026-09-24, 32 of the loop's 58 idle hours. With
`automod.landing.defer_restart` on (and only while the STAGED guardian can
judge a batch), a landing that need not restart anything now fast-forwards
`main`, writes `promoted`, and goes on `pending_restart.json`; a flush restarts
once for all of it and writes the ONE record the guardian judges, with
`commits` and the oldest landing's parent as the rollback target.

What these pin: the merge-only landing touches nothing that runs; eager
landings (a venv, `agent-services/`, the interlock off) keep today's path and
carry the batch; every flush variant (restart, already live, nothing to
restart, trim, failure before and after the restart); when a flush is due; the
interlock and the suspension after a batch rollback; `round restart` becoming a
flush; `bless` refusing while landings wait; and the batch drill's own judge
against the guardian's real revert.

The guardian's side — how a batch record is rolled back and settled — is in
`tests/test_guardian_rollback.py`; the implement source's side in
`tests/test_loop_depth.py`.
"""

from __future__ import annotations

import json
import subprocess as _sp
import sys
import time
from pathlib import Path

import pytest

from scripts.automod import promote as P, round as R, state as S

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN = ROOT / "agent-services" / "guardian"


def _git(repo, *args):
    return _sp.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    for name in ("STATE_DIR", "BROKEN_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(S, name, state)
    for name, fn in (("CURRENT_PATH", "current.json"), ("LEDGER_PATH", "promotions.jsonl"),
                     ("LKG_PATH", "last_known_good.json"), ("HALTED_PATH", "promotions-halted"),
                     ("BROKEN_PATH", "BROKEN"), ("DENIED_PATH", "denied.json"),
                     ("PAUSE_PATH", "pause"), ("LOCK_PATH", "lock"),
                     ("PENDING_PATH", "pending_restart.json"),
                     ("FLUSH_MARKER_PATH", "flush.running"),
                     ("ROLLBACK_REQUEST_PATH", "rollback_request.json")):
        monkeypatch.setattr(S, name, state / fn)
    return state


def _train(monkeypatch, on=True, aware=True, **cfg):
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {"defer_restart": on, **cfg})
    monkeypatch.setattr(S, "guardian_batch_aware", lambda bin_dir=None: aware)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A live repo, a stack that is a dict, and every wait counted."""
    live = tmp_path / "live"
    (live / "app").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(live))
    _git(live, "config", "user.email", "t@e.com")
    _git(live, "config", "user.name", "t")
    (live / "app" / "base.py").write_text("BASE = 1\n", encoding="utf-8")
    _git(live, "add", "-A")
    _git(live, "commit", "-q", "-m", "base")
    base = _git(live, "rev-parse", "HEAD").stdout.strip()

    w = {"live": live, "base": base, "tmp": tmp_path, "boot": 1, "running": base,
         "calls": {"wait_idle": 0, "wait_for_rounds": 0, "restart": [], "drain": [],
                   "pause": [], "announce": [], "settle": 0},
         "needs": lambda changed: (True, "the backend has loaded it"),
         "idle": (True, "idle for 3 consecutive polls"), "healthy": True}
    c = w["calls"]
    monkeypatch.setattr(P, "LIVE_ROOT", live)
    monkeypatch.setattr(P, "vault_commits_for", lambda rid: [])
    monkeypatch.setattr(P, "count_kg_rows", lambda: 10)
    monkeypatch.setattr(P, "count_vault_files", lambda: 20)
    monkeypatch.setattr(P, "announce", lambda head, body: c["announce"].append((head, body)))
    monkeypatch.setattr(P, "_announce_promoted", lambda *a, **k: c["announce"].append(a))
    monkeypatch.setattr(P, "_start_regression_runner", lambda: "stubbed")
    monkeypatch.setattr(P, "restart_needed", lambda changed, **k: w["needs"](changed))

    def get(url, timeout=5.0):
        return 200, {"boot_id": f"boot-{w['boot']}", "commit": w["running"],
                     "turns": {"active": 0, "queued": 0, "harness_runs": 0}}
    monkeypatch.setattr(P, "_get", get)

    def restart(program):
        c["restart"].append(program)
        w["boot"] += 1
        w["running"] = _git(live, "rev-parse", "HEAD").stdout.strip()
        return True, "ok"
    monkeypatch.setattr(P, "restart_process", restart)
    monkeypatch.setattr(P, "_wait_health", lambda url, budget: w["healthy"])
    monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: get(url)[1])

    def wait_idle(*a, **k):
        c["wait_idle"] += 1
        return w["idle"]
    monkeypatch.setattr(P, "wait_idle", wait_idle)

    def wait_for_rounds(*a, **k):
        c["wait_for_rounds"] += 1
        return True, "no autocode turn in flight"
    monkeypatch.setattr(P, "wait_for_rounds", wait_for_rounds)

    def wait_for_settle(**k):
        c["settle"] += 1
    monkeypatch.setattr(P, "wait_for_settle", wait_for_settle)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=0: c["drain"].append(on) or True)
    monkeypatch.setattr(P, "set_pool_paused", lambda p: True)
    monkeypatch.setattr(P, "pool_paused", lambda: False)
    monkeypatch.setattr(P.S, "set_pause", lambda lease: c["pause"].append(lease))
    return w


def _round(w, rid, rel="app/mod.py", text=None):
    """A gated round cut from live HEAD: one commit on `automod/<rid>`."""
    live = w["live"]
    base = _git(live, "rev-parse", "HEAD").stdout.strip()
    wt = w["tmp"] / f"wt_{rid}"
    _git(live, "worktree", "add", "-q", "-b", f"automod/{rid}", str(wt), base)
    (wt / rel).parent.mkdir(parents=True, exist_ok=True)
    (wt / rel).write_text(text or f"# {rid}\nX = {rid!r}\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"round {rid}")
    head = _git(wt, "rev-parse", "HEAD").stdout.strip()
    return {"rid": rid, "wt": wt, "base": base, "head": head,
            "report": {"head": head, "ok": True, "changed_paths": [rel]}}


def _land(rd, **k):
    return P.promote(rd["rid"], rd["wt"], rd["base"], gate_report=rd["report"], **k)


def _rows(event):
    return [e for e in S.read_events(path=S.LEDGER_PATH, limit=500) if e.get("event") == event]


def _head(w):
    return _git(w["live"], "rev-parse", "HEAD").stdout.strip()


# ── the merge-only landing ──────────────────────────────────────────────────

def test_a_deferrable_landing_only_merges(world, monkeypatch):
    """`main` moves, `promoted` is written, the entry goes on the train — and
    nothing that runs is touched: no wait, no drain, no lease, no restart, no
    window."""
    _train(monkeypatch)
    rd = _round(world, "SM_A")
    out = _land(rd)
    c = world["calls"]
    assert out["promoted"] is True and out["deferred"] is True and out["restart_pending"] is True
    assert _head(world) == out["commit"] == rd["head"]
    assert c["wait_idle"] == c["wait_for_rounds"] == c["settle"] == 0
    assert c["restart"] == [] and c["pause"] == [] and True not in c["drain"]
    assert not S.CURRENT_PATH.exists(), "no window opened for code that is not running"
    pending = S.read_pending()
    assert [e["commit"] for e in pending] == [rd["head"]]
    assert pending[0]["parent"] == rd["base"] and pending[0]["restart"] is True
    assert pending[0]["merged_ts"] and pending[0]["kg_rows"] == 10
    row = _rows("promoted")[-1]
    assert row["deferred"] is True and row["restart_pending"] is True
    assert row["errors_until"] is None and row["restarted"] is None


def test_a_merge_lands_over_an_observing_record(world, monkeypatch):
    """The window belongs to code already running; a merge changes none of it."""
    _train(monkeypatch)
    S.write_json(S.CURRENT_PATH, {"state": "observing", "commit": "f" * 40,
                                  "errors_until_ts": time.time() + 300})
    out = _land(_round(world, "SM_A"))
    assert out["deferred"] is True and S.read_current()["commit"] == "f" * 40


def test_a_merge_refuses_a_record_mid_landing(world, monkeypatch):
    _train(monkeypatch)
    S.write_json(S.CURRENT_PATH, {"state": "landing", "commit": "f" * 40})
    with pytest.raises(P.PromoteError, match="mid-landing"):
        _land(_round(world, "SM_A"))
    assert S.read_pending() == []


def test_a_merge_that_fails_to_fast_forward_leaves_nothing_pending(world, monkeypatch):
    _train(monkeypatch)
    rd = _round(world, "SM_A")
    (world["live"] / "app" / "other.py").write_text("Y = 1\n", encoding="utf-8")
    _git(world["live"], "add", "-A")
    _git(world["live"], "commit", "-q", "-m", "human")
    # A moved main is chased by a re-gate; stub it to hand back the same round.
    monkeypatch.setattr(P, "_regate_after_move",
                        lambda rid, wt, live, base, lh: (lh, rd["head"], rd["report"]))
    with pytest.raises(P.PromoteError, match="fast-forward failed"):
        _land(rd)
    assert S.read_pending() == []
    assert _rows("land_failed")[-1]["external_blocker"] is True


# ── eager landings ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("changed,report,why", [
    (["agent-services/bin/x.sh"], {}, "service definition"),
    (["app/x.py"], {"venv": "/wt/.venvs/lloyd"}, "candidate venv"),
])
def test_eager_landings_name_their_reason(monkeypatch, changed, report, why):
    _train(monkeypatch)
    assert why in " ".join(P.landing_is_eager(changed, report))
    assert P.landing_is_eager(["app/x.py", "tests/test_x.py"], {}) == []


@pytest.mark.parametrize("on,aware,want", [
    (False, True, "defer_restart is off"),
    (True, False, "staged guardian cannot roll back a batch"),
])
def test_the_train_off_or_the_interlock_off_makes_every_landing_eager(monkeypatch, on, aware, want):
    _train(monkeypatch, on=on, aware=aware)
    reasons = P.landing_is_eager(["app/x.py"], {})
    assert len(reasons) == 1 and want in reasons[0]


def test_with_the_interlock_off_a_landing_is_todays_landing(world, monkeypatch):
    _train(monkeypatch, aware=False)
    rd = _round(world, "SM_A")
    out = _land(rd)
    assert "deferred" not in out and out["restart"] is True
    assert world["calls"]["wait_idle"] == 1 and world["calls"]["restart"] == ["lloyd-mcp",
                                                                              "lloyd-backend"]
    cur = S.read_current()
    assert cur["state"] == "observing" and "commits" not in cur
    assert cur["rollback_target"] == rd["base"] and S.read_pending() == []


def test_an_eager_landing_restarts_inline_and_carries_the_batch(world, monkeypatch):
    """Two merges on the train, then a service-definition change: one
    restart, one record naming all three, the oldest one's parent to roll
    back to, the train emptied, and a `restart_flushed` row that says so."""
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    b = _land(_round(world, "SM_B", rel="app/b.py"))
    hot = _round(world, "SM_HOT", rel="agent-services/bin/tool.sh", text="echo hi\n")
    out = _land(hot)
    c = world["calls"]
    assert out["restart"] is True and out["flushed_pending"] == [a["commit"], b["commit"]]
    assert c["wait_idle"] == 1 and c["restart"] == ["lloyd-mcp", "lloyd-backend"]
    cur = S.read_current()
    assert cur["commits"] == [a["commit"], b["commit"], hot["head"]]
    assert cur["commit"] == hot["head"] and cur["schema"] == 2
    assert cur["rollback_target"] == world["base"]
    assert [e["commit"] for e in cur["entries"]] == cur["commits"]
    assert S.read_pending() == []
    row = _rows("restart_flushed")[-1]
    assert row["by"] == "land" and row["commits"] == cur["commits"] and row["batch"] == 3


# ── the flush ───────────────────────────────────────────────────────────────

def test_a_flush_restarts_once_for_the_batch_and_opens_one_window(world, monkeypatch):
    _train(monkeypatch, errors_window_s=300)
    a = _land(_round(world, "SM_A"))
    b = _land(_round(world, "SM_B", rel="app/b.py"))
    out = P.flush_pending("test", by="test")
    c = world["calls"]
    assert out["flushed"] is True and out["restarted"] is True
    assert c["wait_for_rounds"] == 1 and c["wait_idle"] == 1
    assert c["restart"] == ["lloyd-mcp", "lloyd-backend"], "one restart for two landings"
    cur = S.read_current()
    assert cur["state"] == "observing" and cur["commits"] == [a["commit"], b["commit"]]
    assert cur["commit"] == b["commit"] and cur["rollback_target"] == world["base"]
    assert cur["errors_until_ts"] - cur["landed_ts"] == pytest.approx(300, abs=1)
    assert cur["restart"] is True and cur["boot_id"] == f"boot-{world['boot']}"
    assert cur["kg_rows"] == 10, "counted before the FIRST merge"
    assert S.read_pending() == [] and S.read_flush_marker() is None
    row = _rows("restart_flushed")[-1]
    assert row["commits"] == [a["commit"], b["commit"]] and row["restarted"] is True
    assert "waited_s" in row and row["reason"] == "test"


def test_a_flush_of_what_is_already_live_restarts_nothing(world, monkeypatch):
    """The guardian's rollback restart, a human restart or a crash already
    booted the merged code: the window opens on what is running."""
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    world["running"], world["boot"] = a["commit"], 7
    out = P.flush_pending("test")
    assert out["already_live"] is True and out["restarted"] is False
    assert world["calls"]["restart"] == [] and world["calls"]["wait_idle"] == 0
    cur = S.read_current()
    assert cur["restart"] is True, "the code IS running: liveness and errors are judged"
    assert cur["flush"]["already_live"] is True


def test_a_flush_of_landings_that_need_no_restart_is_short_and_quiet(world, monkeypatch):
    _train(monkeypatch, errors_window_unrestarted_s=120)
    world["needs"] = lambda changed: (False, "nothing loaded")
    _land(_round(world, "SM_A", rel="tests/test_a.py"))
    out = P.flush_pending("test")
    c = world["calls"]
    assert out["restarted"] is False and c["wait_idle"] == c["wait_for_rounds"] == 0
    assert c["restart"] == [] and True not in c["drain"]
    cur = S.read_current()
    assert cur["restart"] is False
    assert cur["errors_until_ts"] - cur["landed_ts"] == pytest.approx(120, abs=1)


def test_a_flush_drops_entries_no_longer_in_main(world, monkeypatch):
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    S.append_pending({"round_id": "SM_GHOST", "commit": "e" * 40, "parent": world["base"],
                      "restart": True, "merged_ts": time.time()})
    P.flush_pending("test")
    assert S.read_current()["commits"] == [a["commit"]]
    assert _rows("pending_dropped")[-1]["commits"] == ["e" * 40]


def test_a_flush_that_never_got_idle_leaves_the_train_as_it_was(world, monkeypatch):
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    world["idle"] = (False, "backend never went idle within 900s")
    with pytest.raises(P.FlushNotStarted, match="never went idle"):
        P.flush_pending("test")
    assert [e["commit"] for e in S.read_pending()] == [a["commit"]]
    assert not S.CURRENT_PATH.exists() and world["calls"]["restart"] == []
    assert _rows("flush_failed")[-1]["restarted"] is False
    assert S.read_flush_marker() is None


def test_a_flush_that_failed_after_the_restart_undoes_the_batch(world, monkeypatch):
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    b = _land(_round(world, "SM_B", rel="app/b.py"))
    world["healthy"] = False
    undone: list = []

    def undo(live, commits, target):
        undone.append((list(commits), target))
        return {"route": "reset", "restored": target}
    monkeypatch.setattr(P, "_undo_batch", undo)
    with pytest.raises(P.PromoteError, match="never became healthy"):
        P.flush_pending("test")
    assert undone == [([a["commit"], b["commit"]], world["base"])]
    row = _rows("rollback_succeeded")[-1]
    assert row["commits"] == [a["commit"], b["commit"]] and row["trigger"] == "flush_failed"
    assert S.reverted_commits(S.read_events(path=S.LEDGER_PATH, limit=500)) >= {
        a["commit"], b["commit"]}
    assert S.read_pending() == [] and not S.CURRENT_PATH.exists()


def test_an_undo_the_promoter_cannot_do_is_one_request_to_the_guardian(world, monkeypatch):
    _train(monkeypatch)
    a = _land(_round(world, "SM_A"))
    b = _land(_round(world, "SM_B", rel="app/b.py"))
    monkeypatch.setattr(P, "_undo_batch", lambda *a_: (_ for _ in ()).throw(RuntimeError("conflict")))
    P._undo_batch_and_record(world["live"], {
        "commits": [a["commit"], b["commit"]], "commit": b["commit"],
        "rollback_target": world["base"]}, trigger="flush_failed")
    req = S.read_rollback_request()
    assert req["commits"] == [a["commit"], b["commit"]] and req["commit"] == b["commit"]
    assert req["target"] == world["base"]


def test_undo_batch_reverts_around_a_foreign_commit(world, monkeypatch):
    """Through the guardian's own `rollback.py` (this tree's; the fixture's
    live repo has none), with the services stubbed."""
    _train(monkeypatch)
    monkeypatch.setattr(P, "stop_process", lambda program, wait=True: (True, "ok"))
    real = P._guardian_rollback_module
    monkeypatch.setattr(P, "_guardian_rollback_module", lambda live: real(ROOT))
    a = _land(_round(world, "SM_A"))
    (world["live"] / "app" / "human.py").write_text("H = 1\n", encoding="utf-8")
    _git(world["live"], "add", "-A")
    _git(world["live"], "commit", "-q", "-m", "nightly")
    b = _land(_round(world, "SM_B", rel="app/b.py"))
    ev = P._undo_batch(world["live"], [a["commit"], b["commit"]], world["base"])
    assert ev["route"] == "revert"
    assert (world["live"] / "app" / "human.py").exists()
    assert not (world["live"] / "app" / "mod.py").exists()
    assert not (world["live"] / "app" / "b.py").exists()


def test_a_second_flush_refuses_while_one_runs(world, monkeypatch):
    _train(monkeypatch)
    _land(_round(world, "SM_A"))
    S.write_flush_marker(pid=1, by="other")   # pid 1 is always alive
    with pytest.raises(P.PromoteError, match="already running"):
        P.flush_pending("test")


def test_a_flush_will_not_write_a_batch_for_a_guardian_that_cannot_judge_one(world, monkeypatch):
    _train(monkeypatch)
    _land(_round(world, "SM_A"))
    _land(_round(world, "SM_B", rel="app/b.py"))
    monkeypatch.setattr(S, "guardian_batch_aware", lambda bin_dir=None: False)
    with pytest.raises(P.FlushNotStarted, match="cannot judge a batch"):
        P.flush_pending("test")
    assert len(S.read_pending()) == 2 and not S.CURRENT_PATH.exists()


# ── when a flush is due ─────────────────────────────────────────────────────

def _pend(n, *, restart=True, age=0.0):
    for i in range(n):
        S.append_pending({"round_id": f"SM_{i}", "commit": f"{i:x}" * 40, "restart": restart,
                          "merged_ts": time.time() - age})


@pytest.mark.parametrize("setup,rounds,due,why", [
    (lambda: None, 0, False, "nothing pending"),
    (lambda: _pend(1, restart=False), 3, True, "need no restart"),
    (lambda: _pend(1, age=2800), 3, True, "has waited"),
    (lambda: _pend(4), 3, True, "batch max 4"),
    (lambda: _pend(1), 0, True, "natural gap"),
    (lambda: _pend(1), 1, False, "wait for a restart"),
    (lambda: _pend(1), None, False, "wait for a restart"),
])
def test_flush_due(monkeypatch, setup, rounds, due, why):
    _train(monkeypatch)
    setup()
    got, reason = P.flush_due(rounds)
    assert got is due and why in reason


@pytest.mark.parametrize("block,why", [
    (lambda: S.write_json(S.CURRENT_PATH, {"state": "observing", "commit": "a" * 40}),
     "flushing after it settles"),
    (lambda: S.write_flush_marker(pid=1, by="x"), "already running"),
    (lambda: S.HALTED_PATH.write_text("x"), "halted"),
    (lambda: S.write_json(S.ROLLBACK_REQUEST_PATH, {"ts": 1}), "rollback request"),
])
def test_flush_due_waits_on_whatever_a_flush_would_wait_on(monkeypatch, block, why):
    _train(monkeypatch)
    _pend(4)
    block()
    got, reason = P.flush_due(0)
    assert got is False and why in reason


def test_flush_due_refuses_a_batch_the_staged_guardian_cannot_judge(monkeypatch):
    _train(monkeypatch, aware=False)
    _pend(2)
    got, reason = P.flush_due(0)
    assert got is False and "cannot judge a batch" in reason


# ── the interlock and the suspension ────────────────────────────────────────

def test_the_interlock_reads_the_staged_snapshot(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    assert S.guardian_batch_aware(bin_dir) is False, "no snapshot"
    (bin_dir / "gstate.py").write_text("HALT_SET_EVENT = 'x'\n")
    assert S.guardian_batch_aware(bin_dir) is False, "a pre-train snapshot"
    (bin_dir / "gstate.py").write_text("x = 1\nBATCH_SCHEMA = 1\n")
    assert S.guardian_batch_aware(bin_dir) is False
    (bin_dir / "gstate.py").write_text("x = 1\nBATCH_SCHEMA = 2\n")
    assert S.guardian_batch_aware(bin_dir) is True
    # The repo's own copy declares it, so staging this commit is what flips it.
    assert S.guardian_batch_aware(GUARDIAN) is True
    monkeypatch.setenv("LLOYD_GUARDIAN_STATE", str(tmp_path))
    assert S.guardian_bin_dir() == bin_dir and S.guardian_batch_aware() is True


def test_the_train_is_suspended_after_a_batch_rollback(monkeypatch):
    _train(monkeypatch, batch_suspend_after_rollback_s=21600)
    assert P.restart_deferred()[0] is True
    S.append_event({"event": "rollback_succeeded", "commit": "b" * 40})
    assert P.restart_deferred()[0] is True, "a single-commit rollback suspends nothing"
    S.append_event({"event": "rollback_succeeded", "commit": "b" * 40,
                    "commits": ["a" * 40, "b" * 40]})
    on, why = P.restart_deferred()
    assert on is False and "suspended" in why
    old = [{"event": "rollback_succeeded", "ts": time.time() - 30000,
            "commits": ["a" * 40, "b" * 40]}]
    assert S.batch_suspended_until(21600, old) < time.time()


# ── the human's restart, bless, status ──────────────────────────────────────

def test_round_restart_is_a_flush_while_landings_wait(monkeypatch):
    _pend(1)
    asked: list = []
    monkeypatch.setattr(P, "flush_pending",
                        lambda reason, **k: asked.append((reason, k)) or {"flushed": True})
    out = P.restart_stack(("lloyd-backend",), reason="picked up gate.py", skip_idle=True)
    assert out["flush"] == {"flushed": True}
    assert asked == [("human restart: picked up gate.py",
                      {"kill_turns": True, "by": "restart", "force_restart": True})]


def test_bless_refuses_while_landings_wait_for_a_restart(world, monkeypatch):
    monkeypatch.setattr(R, "LIVE_ROOT", world["live"])
    world["running"] = world["base"]
    S.append_pending({"round_id": "SM_A", "commit": "a" * 40, "restart": True})
    with pytest.raises(RuntimeError, match="merged but not yet running"):
        R.bless("test")
    S.write_pending([{"round_id": "SM_A", "commit": "a" * 40, "restart": False}])
    assert R.bless("test")["last_known_good"]["commit"] == world["base"]


def test_status_reports_the_train(monkeypatch):
    _train(monkeypatch)
    _pend(2)
    from workers.sources import autocode as AC
    monkeypatch.setattr(AC, "implement_turns_in_flight", lambda *a, **k: 2)
    out = R._status_pending()
    assert out["count"] == 2 and out["restart_needed"] == 2 and out["train"]["on"] is True
    assert out["flush_due"] is False and len(out["entries"]) == 2
    assert out["rounds_in_flight"] == 2 and "2 turn(s) in flight" in out["flush_why"]


def test_status_counts_turns_the_way_the_backend_trigger_does(monkeypatch):
    """`round status` runs in its own process and used to pass no count, so it
    printed "? turn(s) in flight" and could never show the natural gap."""
    _train(monkeypatch)
    _pend(1)
    from workers.sources import autocode as AC
    monkeypatch.setattr(AC, "implement_turns_in_flight", lambda *a, **k: 0)
    out = R._status_pending()
    assert out["flush_due"] is True and "natural gap" in out["flush_why"]

    def unreadable(*a, **k):
        raise RuntimeError("no queue")
    monkeypatch.setattr(AC, "implement_turns_in_flight", unreadable)
    out = R._status_pending()
    assert out["rounds_in_flight"] is None and out["flush_due"] is False, \
        "an unreadable count is never a gap"


# ── round.land on the train ─────────────────────────────────────────────────

def test_land_on_the_train_waits_for_no_round_and_no_settle(world, monkeypatch):
    _train(monkeypatch)
    rd = _round(world, "SM_A")
    (S.ROUNDS_DIR / "SM_A").mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / "SM_A" / "gate.json").write_text(json.dumps(
        {**rd["report"], "base": rd["base"]}))
    monkeypatch.setattr(R, "LIVE_ROOT", world["live"])
    monkeypatch.setattr(R.W, "worktree_path", lambda rid: rd["wt"])
    monkeypatch.setattr(R.W, "remove", lambda *a, **k: None)
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: True)
    S.write_json(S.CURRENT_PATH, {"state": "observing", "commit": "f" * 40,
                                  "errors_until_ts": time.time() + 300})
    flushes: list = []
    # Asked once the land marker is gone: `flush_due` waits while a landing
    # runs, and this landing must not be the one it waits for.
    monkeypatch.setattr(R, "maybe_flush", lambda **k: flushes.append(
        (k, S.land_in_progress("SM_A"))) or None)
    out = R.land("SM_A")
    assert out["deferred"] is True
    assert world["calls"]["wait_for_rounds"] == 0 and world["calls"]["settle"] == 0
    assert _rows("land_wait_rounds")[-1]["detail"] == \
        "not waited for: the restart is deferred to the flush"
    assert flushes == [({"by": "land SM_A"}, None)]


def test_flush_detached_writes_the_childs_pid_before_returning(monkeypatch):
    spawned: list = []
    monkeypatch.setattr(S, "spawn_detached", lambda argv, log, cwd=None: spawned.append(argv) or 1)
    out = R.flush_detached(by="autocode", now=True)
    assert out["pid"] == 1 and S.flush_in_progress()["pid"] == 1
    assert [str(a) for a in spawned[0][-4:]] == ["flush", "--by", "autocode", "--now"]
    assert "error" in R.flush_detached(by="again")


# ── the batch drill judges what the guardian really does ────────────────────

@pytest.mark.parametrize("kind", ["reset", "revert"])
def test_the_batch_drill_judges_the_guardians_own_routes(tmp_path, kind):
    from scripts.automod import rehearse as RH
    if str(GUARDIAN) not in sys.path:
        sys.path.insert(0, str(GUARDIAN))
    import rollback as rb
    src = tmp_path / "src"
    (src / "app").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "config", "user.email", "t@e.com")
    _git(src, "config", "user.name", "t")
    (src / "server.py").write_text("print('ok')\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "base")
    base = _git(src, "rev-parse", "HEAD").stdout.strip()
    scratch = tmp_path / "scratch"
    shas = RH._prepare_scratch(scratch, src, base, batch=kind)
    rec = RH._drill_record(shas, kind)
    assert rec["commits"] == [shas["broken"], shas["second"]] and rec["commit"] == shas["second"]
    order = _git(scratch, "rev-list", "--reverse", f"{base}..HEAD").stdout.split()
    want = ([shas["broken"], shas["second"]] if kind == "reset"
            else [shas["broken"], shas["later"], shas["second"]])
    assert order == want
    assert RH._judge_batch(scratch, shas, shas["second"], kind), "nothing done is a failure"
    head = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    exact = rb.range_is_exactly(str(scratch), base, head, rec["commits"])
    assert exact is (kind == "reset")
    if exact:
        rb.restore_tree(str(scratch), base, ("app",), ("app",))
    else:
        rb.revert_commits(str(scratch), rec["commits"])
    now = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    assert RH._judge_batch(scratch, shas, now, kind) == ""


# ── the scorecard reads the flush ───────────────────────────────────────────

def test_the_scorecard_reads_a_flush_as_the_restart_a_landing_used_to_be():
    from scripts.automod import scorecard as SC
    t0 = 1_000_000.0
    ev = [
        {"event": "backlog_implement", "item_id": 1, "phase": "started", "ts": t0},
        {"event": "backlog_implement", "item_id": 1, "phase": "finished", "ts": t0 + 600},
        {"event": "restart_flushed", "ts": t0 + 700, "waited_s": 240.0, "commits": ["a"]},
        {"event": "backlog_implement", "item_id": 2, "phase": "started", "ts": t0 + 1200},
        {"event": "backlog_implement", "item_id": 2, "phase": "finished", "ts": t0 + 1800},
    ]
    duty = SC._duty_cycle(ev, t0, t0 + 1800)
    # Classed with the promotion gaps (whatever that class is named), never `other`.
    assert sum(duty["gap_counts"].values()) == 1
    assert "other" not in duty["gap_counts"], "a flush gap is not an unexplained one"
    assert duty["waits"]["flush"] == {"n": 1, "waited": 1, "minutes": 4.0,
                                      "median_s": 240.0, "max_s": 240.0}
