"""What the promoter does when its own post-merge rollback goes wrong.

A landing that fails between the fast-forward and the verification is undone
inline. When that undo fails, the record the guardian rolls back by must stay,
the guardian must be asked, and the ledger must not say `rolled_back: True`.
Every supervisord call is stubbed: nothing here may reach the live socket.
"""

from __future__ import annotations

import json
import subprocess as _sp
from pathlib import Path

import pytest

from scripts.automod import promote as P, state as S

ROOT = Path(__file__).resolve().parent.parent


def _git(repo, *args):
    return _sp.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                   check=False).stdout.strip()


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
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {"defer_restart": False})
    # supervisord, never the live one.
    sup = {"stopped": [], "started": [], "restarted": []}
    monkeypatch.setattr(P, "stop_process",
                        lambda program, wait=True: sup["stopped"].append(program) or (True, "ok"))
    monkeypatch.setattr(P, "start_process",
                        lambda program, wait=False: sup["started"].append(program) or (True, "ok"))
    monkeypatch.setattr(P, "process_info", lambda program: {"statename": "RUNNING"})
    monkeypatch.setattr(P, "restart_process",
                        lambda program: sup["restarted"].append(program) or (True, "ok"))
    sup["alerts"] = []
    monkeypatch.setattr(P, "alert", lambda title, body, **k: sup["alerts"].append(title))
    # The guardian's rollback module from this tree; the scratch repo has none.
    real = P._guardian_rollback_module
    monkeypatch.setattr(P, "_guardian_rollback_module", lambda live: real(ROOT))
    return sup


@pytest.fixture()
def landing(tmp_path, monkeypatch):
    """A live repo, a round one commit ahead of it, and a backend that never
    picks the landing up — so verification fails after the merge."""
    from scripts.automod import backlog as B
    live = tmp_path / "live"
    live.mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", str(live))
    _git(live, "config", "user.email", "t@t")
    _git(live, "config", "user.name", "t")
    (live / "app").mkdir()
    (live / "app" / "f.py").write_text("A = 1\n", encoding="utf-8")
    (live / "notes.md").write_text("tracked\n", encoding="utf-8")
    _git(live, "add", ".")
    _git(live, "commit", "-qm", "base")
    base = _git(live, "rev-parse", "HEAD")
    wt = tmp_path / "wt"
    _git(live, "worktree", "add", "-q", "-b", "automod/SM_F", str(wt))
    (wt / "app" / "f.py").write_text("A = 2\n", encoding="utf-8")
    _git(wt, "add", ".")
    _git(wt, "commit", "-qm", "gated\n\nCo-Authored-By: Lloyd <lloyd@local>")
    head = _git(wt, "rev-parse", "HEAD")

    def get(url, timeout=5.0):
        return 200, {"turns": {"active": 0, "queued": 0, "harness_runs": 0},
                     "boot_id": "boot-1", "commit": base}
    monkeypatch.setattr(P, "LIVE_ROOT", live)
    monkeypatch.setattr(P, "count_kg_rows", lambda: 0)
    monkeypatch.setattr(P, "count_vault_files", lambda: 0)
    monkeypatch.setattr(P, "vault_commits_for", lambda round_id: [])
    monkeypatch.setattr(P, "restart_needed", lambda changed, **k: (True, "loaded"))
    monkeypatch.setattr(P, "wait_idle", lambda *a, **k: (True, "idle"))
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: True)
    monkeypatch.setattr(P, "pool_paused", lambda: False)
    monkeypatch.setattr(P, "set_pool_paused", lambda paused: True)
    monkeypatch.setattr(P, "_apply_service_changes", lambda changed: [])
    monkeypatch.setattr(P, "_wait_health", lambda url, budget: True)
    monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: get(url)[1])
    monkeypatch.setattr(P, "_announce_promoted", lambda *a, **k: None)
    monkeypatch.setattr(P, "_get", get)
    monkeypatch.setattr(B, "work_title_for_round", lambda ledger, round_id: "a title")
    return {"live": live, "wt": wt, "base": base, "head": head}


def _rows(event):
    return [e for e in S.read_events(path=S.LEDGER_PATH, limit=500) if e.get("event") == event]


def _land(t):
    return P.promote("SM_F", t["wt"], t["base"], gate_report={"head": t["head"]})


def test_a_failed_inline_rollback_keeps_the_record_and_asks_the_guardian(landing, monkeypatch):
    """`index.lock` held, `verify_tree` failing: the tree did not move. The
    record stays (the guardian rolls back by it), the guardian is asked, and
    the landing's row does not claim a rollback that never happened."""
    def blocked(*a, **k):
        raise RuntimeError("git reset --hard failed: index.lock exists")
    monkeypatch.setattr(P, "_rollback_inline", blocked)
    with pytest.raises(P.PromoteError, match="did not pick up"):
        _land(landing)
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["head"]
    cur = S.read_current()
    assert cur is not None and cur["commit"] == landing["head"], "the record was cleared"
    req = S.read_rollback_request()
    assert req["commit"] == landing["head"] and req["target"] == landing["base"]
    failed = _rows("land_failed")[-1]
    assert failed["rolled_back"] is False and failed["rollback"] == P.UNDO_HANDED_OFF
    assert "guardian was asked" in failed["detail"]
    assert _rows("rollback_succeeded") == []
    assert _rows("rollback_failed")[-1]["handed_to"] == "guardian"


def test_the_stranded_sweep_leaves_a_record_the_guardian_was_asked_about(landing, monkeypatch):
    monkeypatch.setattr(P, "_rollback_inline",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index.lock")))
    with pytest.raises(P.PromoteError):
        _land(landing)
    assert P.release_stranded_landing(by="test") is None
    assert S.read_current()["commit"] == landing["head"] and S.read_pending() == []


def test_a_rollback_whose_service_does_not_come_back_is_not_reported_done(landing, monkeypatch):
    """The tree goes back, then the backend never answers: not `rolled_back`,
    and not handed to the guardian either — a second rollback would act on a
    tree already restored. The record goes; it no longer describes `main`."""
    health = {"n": 0}

    def wait_health(url, budget):
        health["n"] += 1
        return health["n"] <= 2     # the landing's two legs; the rollback's fail
    monkeypatch.setattr(P, "_wait_health", wait_health)
    with pytest.raises(P.PromoteError, match="did not pick up"):
        _land(landing)
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["base"], "the tree went back"
    failed = _rows("land_failed")[-1]
    assert failed["rolled_back"] is False and failed["rollback"] == P.UNDO_TREE_ONLY
    assert _rows("rollback_succeeded") == []
    assert _rows("rollback_failed")[-1]["tree_restored"] is True
    assert S.read_rollback_request() is None and S.read_current() is None


def test_a_rollback_that_finished_every_leg_is_rolled_back(landing, isolated):
    with pytest.raises(P.PromoteError, match="did not pick up"):
        _land(landing)
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["base"]
    failed = _rows("land_failed")[-1]
    assert failed["rolled_back"] is True and failed["rollback"] == P.UNDO_ROLLED_BACK
    assert _rows("rollback_succeeded")[-1]["commit"] == landing["head"]
    assert S.read_current() is None and S.read_rollback_request() is None
    assert isolated["restarted"][-2:] == ["lloyd-mcp", "lloyd-backend"]


def test_a_rollback_restart_that_fails_raises(landing, monkeypatch):
    """`_rollback_inline` used to drop `restart_process`'s answer on the floor."""
    monkeypatch.setattr(P, "restart_process", lambda program: (False, "spawn error"))
    monkeypatch.setattr(P, "process_info", lambda program: {"statename": "FATAL"})
    monkeypatch.setattr(P, "start_process", lambda program, wait=False: (False, "spawn error"))
    _git(landing["live"], "merge", "-q", "--ff-only", "automod/SM_F")
    with pytest.raises(P.RollbackLegFailed, match="restart lloyd-mcp failed"):
        P._undo_batch(landing["live"], [landing["head"]], landing["base"])


# ── #3: the route back is the guardian's, and nothing is stashed ─────────────

def _merged(t):
    _git(t["live"], "merge", "-q", "--ff-only", "automod/SM_F")
    assert _git(t["live"], "rev-parse", "HEAD") == t["head"]


def test_a_commit_made_on_top_of_the_landing_survives_its_rollback(landing):
    """A human or nightly commit made during the restart legs used to be
    dropped by `reset --hard <parent>`; the landed commit is reverted instead."""
    _merged(landing)
    live = landing["live"]
    (live / "app" / "human.py").write_text("H = 1\n", encoding="utf-8")
    _git(live, "add", "-A")
    _git(live, "commit", "-qm", "human work")
    human = _git(live, "rev-parse", "HEAD")
    ev = P._undo_batch(live, [landing["head"]], landing["base"])
    assert ev["route"] == "revert"
    assert _sp.run(["git", "-C", str(live), "merge-base", "--is-ancestor", human, "HEAD"]
                   ).returncode == 0, "the human's commit left main"
    assert (live / "app" / "human.py").exists()
    assert (live / "app" / "f.py").read_text() == "A = 1\n", "the landing was not undone"


def test_an_exact_range_is_reset_and_uncommitted_edits_stay_in_the_tree(landing):
    """The live checkout's stash is one shared stack: an inline rollback
    writes the evidence patch and leaves a disjoint edit where it is."""
    _merged(landing)
    live = landing["live"]
    (live / "notes.md").write_text("wip by a human\n", encoding="utf-8")
    (live / "app" / "new_wip.py").write_text("W = 1\n", encoding="utf-8")
    stashes = _git(live, "stash", "list")
    ev = P._undo_batch(live, [landing["head"]], landing["base"])
    assert ev["route"] == "reset" and ev["restored"] == landing["base"]
    assert _git(live, "rev-parse", "HEAD") == landing["base"]
    assert (live / "notes.md").read_text() == "wip by a human\n", "the human's edit was moved"
    assert (live / "app" / "new_wip.py").exists()
    assert _git(live, "stash", "list") == stashes == "", "the rollback pushed a stash"
    assert ev["stash"] is None and Path(ev["patch"]).read_text().count("wip by a human") == 1


def test_a_landed_commit_no_longer_in_main_is_not_rolled_back_inline(landing, isolated):
    """Nothing to take off: the tree is not touched, no service is stopped, and
    the caller hands the question to the guardian."""
    with pytest.raises(P.PromoteError, match="no longer in main's history"):
        P._undo_batch(landing["live"], [landing["head"]], landing["base"])
    assert isolated["stopped"] == []
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["base"]


# ── #5: a signal after the merge does not roll a healthy landing back ────────

def test_a_sigterm_during_the_restart_legs_lets_the_landing_finish(landing, monkeypatch):
    """The handler `round.land` installs used to raise `LandingKilled` for the
    landing's whole life; after the merge that reached `promote`'s except
    path, which rolled back a healthy landing and wrote a `land_failed`
    saying it was killed "before it promoted anything"."""
    import signal
    from scripts.automod import round as R
    monkeypatch.setattr(P, "_start_regression_runner", lambda: "stubbed")
    running = {"commit": landing["base"], "boot": "boot-1"}

    def get(url, timeout=5.0):
        return 200, {"turns": {"active": 0, "queued": 0, "harness_runs": 0},
                     "boot_id": running["boot"], "commit": running["commit"]}

    def restart(program):
        if program == "lloyd-mcp":     # a host shutdown, mid-restart
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        running.update(commit=_git(landing["live"], "rev-parse", "HEAD"), boot="boot-2")
        return True, "ok"
    monkeypatch.setattr(P, "_get", get)
    monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: get(url)[1])
    monkeypatch.setattr(P, "restart_process", restart)
    restore = R._die_loudly_on_signal("SM_F")
    try:
        out = _land(landing)
    finally:
        restore()
        P.mark_merged(False)
    assert out["promoted"] is True
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["head"], "rolled back"
    assert _rows("land_failed") == [] and _rows("rollback_succeeded") == []
    assert _rows("land_signal_held")[-1]["signal"] == int(signal.SIGTERM)
    assert S.read_current()["state"] == "observing"


def test_a_sigterm_before_the_merge_still_stops_the_landing(landing, monkeypatch):
    import signal
    from scripts.automod import round as R

    def wait_idle(*a, **k):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(P, "wait_idle", wait_idle)
    restore = R._die_loudly_on_signal("SM_F")
    try:
        with pytest.raises(R.LandingKilled):
            _land(landing)
    finally:
        restore()
    assert _git(landing["live"], "rev-parse", "HEAD") == landing["base"]
    assert _rows("land_failed")[-1]["killed_by_signal"] == int(signal.SIGTERM)


# ── review: a job claimed during a re-gate is not restarted over ────────────

def test_a_job_that_started_during_the_re_gate_stops_the_landing(landing, isolated, monkeypatch):
    live = landing["live"]

    def wait_idle(*a, **k):   # a human commit lands while the landing waits
        (live / "docs.md").write_text("x\n", encoding="utf-8")
        _git(live, "add", "-A")
        _git(live, "commit", "-qm", "human")
        return True, "idle"
    monkeypatch.setattr(P, "wait_idle", wait_idle)
    moved = {}

    def regate(round_id, wt, live_, base, live_head):
        moved["to"] = live_head
        return live_head, landing["head"], {"head": landing["head"], "rungs": []}
    monkeypatch.setattr(P, "_regate_after_move", regate)
    monkeypatch.setattr(P, "pool_in_flight", lambda: [{"source": "autonomy"}])
    with pytest.raises(P.PromoteError, match="worker job started while the landing re-gated"):
        _land(landing)
    assert moved, "the re-gate path was not reached"
    row = _rows("land_failed")[-1]
    assert row["external_blocker"] is True and "autonomy" in row["detail"]
    assert isolated["restarted"] == [] and _rows("rollback_succeeded") == []


# ── review: a handed-off record is never handed back to the land train ──────

def test_a_handed_off_record_is_never_requeued_and_is_alerted_once(landing, isolated,
                                                                    monkeypatch):
    """The guardian discards a request older than 900 s, and may be down: the
    request was the only thing keeping the stranded sweep from requeueing the
    failed commit, which the next flush would restart into production."""
    monkeypatch.setattr(P, "_rollback_inline",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index.lock")))
    with pytest.raises(P.PromoteError):
        _land(landing)
    cur = S.read_current()
    assert cur["rollback_handed_off"] is True and cur["rollback_handed_off_ts"]
    S.clear_rollback_request()                     # discarded as stale
    assert P.release_stranded_landing(by="test") is None
    assert S.read_pending() == [] and S.read_current()["commit"] == landing["head"]
    assert isolated["alerts"] == [], "inside the grace the guardian may still be working"

    cur = S.read_current()
    cur["rollback_handed_off_ts"] -= P.HANDOFF_UNRESOLVED_AFTER_S + 1
    S.write_verified(S.CURRENT_PATH, cur)
    for _ in range(3):
        assert P.release_stranded_landing(by="test") is None
    assert S.read_pending() == [] and S.read_current()["commit"] == landing["head"]
    assert isolated["alerts"] == ["Failed landing was never rolled back"], "one alert, once"
    assert len(_rows("rollback_handoff_unresolved")) == 1


# ── review: the route is re-read after the slow stop ────────────────────────

def test_a_commit_made_during_the_stop_is_reverted_around_not_reset_over(landing, monkeypatch):
    """The route was chosen before `stop_process(wait=True)` on both services;
    a commit landing during that stop was then dropped by `reset --keep`."""
    _merged(landing)
    live = landing["live"]

    def stop(program, wait=True):
        if program == "lloyd-backend":
            (live / "app" / "late.py").write_text("L = 1\n", encoding="utf-8")
            _git(live, "add", "-A")
            _git(live, "commit", "-qm", "late human work")
        return True, "ok"
    monkeypatch.setattr(P, "stop_process", stop)
    ev = P._undo_batch(live, [landing["head"]], landing["base"])
    assert ev["route"] == "revert"
    assert (live / "app" / "late.py").exists(), "the commit made during the stop was dropped"
    assert (live / "app" / "f.py").read_text() == "A = 1\n"
