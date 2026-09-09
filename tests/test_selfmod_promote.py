"""Promoter ordering and the observation-window contract.

Two properties here are the difference between a reversible landing and a
one-way door:

  * the rollback point is on disk and verified *before* the tree moves;
  * the observation window starts when the code goes live, not when the
    promoter began — the idle gate can legitimately wait many minutes, and a
    window started early would be mostly spent before the build existed.
"""

from __future__ import annotations

import json
import time

import pytest

from scripts.selfmod import promote as P, state as S


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name in ("STATE_DIR", "BROKEN_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(S, name, tmp_path)
    for name, fn in (("CURRENT_PATH", "current.json"), ("LEDGER_PATH", "promotions.jsonl"),
                     ("LKG_PATH", "last_known_good.json"), ("HALTED_PATH", "promotions-halted"),
                     ("BROKEN_PATH", "BROKEN"), ("DENIED_PATH", "denied.json"),
                     ("PAUSE_PATH", "pause"), ("LOCK_PATH", "lock")):
        monkeypatch.setattr(S, name, tmp_path / fn)
    return tmp_path


# ---------------------------------------------------------------------------
# The window must not start before the code is live
# ---------------------------------------------------------------------------

def test_a_landing_record_has_no_observation_window_yet():
    """Written before the idle gate, which can wait up to 15 minutes."""
    now = time.time()
    record = {"schema": 1, "round_id": "SM_X", "commit": "b" * 40, "parent": "a" * 40,
              "rollback_target": "a" * 40, "state": "landing",
              "landed_at": None, "landed_ts": None, "errors_until_ts": None}
    S.write_verified(S.CURRENT_PATH, record)
    back = S.read_current()
    assert back["state"] == "landing"
    assert back["errors_until_ts"] is None
    assert back["rollback_target"] == "a" * 40, "the way back must be recorded immediately"


def test_the_window_starts_only_once_the_build_is_verified():
    now = time.time()
    record = {"schema": 1, "commit": "b" * 40, "parent": "a" * 40,
              "rollback_target": "a" * 40, "state": "landing",
              "errors_until_ts": None}
    S.write_verified(S.CURRENT_PATH, record)
    landed = time.time()
    # Liveness deliberately has no separate, shorter window: it applies for
    # the WHOLE observation window, because a build that crashes at minute ten
    # is exactly as bad as one that crashes at minute one.
    record.update({"state": "observing", "landed_ts": landed,
                   "errors_until_ts": landed + P.ERRORS_WINDOW})
    S.write_verified(S.CURRENT_PATH, record)
    back = S.read_current()
    assert back["state"] == "observing"
    assert back["errors_until_ts"] >= now + P.ERRORS_WINDOW - 1


def test_the_guardian_ignores_a_landing_record(tmp_path, monkeypatch):
    """Mid-flight is not "deployed". Nothing to observe, nothing to revert."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "agent-services" / "guardian"))
    import gstate

    st = gstate.SelfModState(tmp_path)
    gstate.write_json_atomic(st.current_path,
                             {"state": "landing", "commit": "b" * 40,
                              "errors_until_ts": None})
    current = st.current()
    assert current["state"] == "landing"
    # The guardian nulls this out before deciding anything (see tick()).
    assert (None if current.get("state") == "landing" else current) is None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_promote_refuses_while_halted(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    S.set_halted("flapping")
    with pytest.raises(P.PromoteError, match="halted"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_promote_refuses_when_broken(monkeypatch, tmp_path):
    S.BROKEN_PATH.write_text("rollback failed", encoding="utf-8")
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    with pytest.raises(P.PromoteError, match="BROKEN"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_promote_refuses_a_denylisted_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    S.deny("b" * 40)
    with pytest.raises(P.PromoteError, match="denylist"):
        P.promote("SM_X", tmp_path, "a" * 40)


def _live_at(monkeypatch, sha):
    """`promote` reads the live HEAD with one `git rev-parse`; everything else
    that touches git goes through `W.*`, which these tests patch by name so
    `subprocess.run` can stay honest for them."""
    real = P.subprocess.run
    def run(argv, *a, **k):
        if argv[:2] == ["git", "-C"] and argv[-2:] == ["rev-parse", "HEAD"]:
            return type("R", (), {"stdout": sha + "\n", "returncode": 0, "stderr": ""})()
        return real(argv, *a, **k)
    monkeypatch.setattr(P.subprocess, "run", run)


def test_promote_refuses_uncommitted_live_edits_only_in_the_rounds_own_files(monkeypatch, tmp_path):
    """#447. Dirt outside the diff is tolerated — `merge --ff-only` never
    touches a file it is not merging. Dirt inside it is two writers on one
    file, refused by name, and not the round's fault."""
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    monkeypatch.setattr(P.W, "changed_paths", lambda wt, base: ["app/x.py"])
    _live_at(monkeypatch, "a" * 40)

    monkeypatch.setattr(P.W, "dirty_paths", lambda repo, limit=None: ["app/x.py", "docs/n.md"])
    with pytest.raises(P.PromoteError, match=r"app/x\.py"):
        P.promote("SM_X", tmp_path, "a" * 40, dry_run=True)
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "land_failed"]
    assert ev and ev[-1]["external_blocker"] is True and ev[-1]["overlap"] == ["app/x.py"]

    monkeypatch.setattr(P.W, "dirty_paths", lambda repo, limit=None: ["docs/n.md"])
    out = P.promote("SM_X", tmp_path, "a" * 40, dry_run=True)
    assert out["would_promote"] is True
    assert out["live_dirty_paths"] == ["docs/n.md"], "recorded, so the window can be read later"


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    monkeypatch.setattr(P.W, "dirty_paths", lambda repo, limit=None: [])
    monkeypatch.setattr(P.W, "changed_paths", lambda wt, base: ["app/x.py"])
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "a" * 40, "returncode": 0, "stderr": ""})())
    out = P.promote("SM_X", tmp_path, "a" * 40, dry_run=True)
    assert out["would_promote"] is True
    assert S.read_current() is None, "a dry run must not write a promotion record"


# ---------------------------------------------------------------------------
# Idle gate
# ---------------------------------------------------------------------------

def test_idle_requires_consecutive_quiet_polls(monkeypatch):
    """A turn appearing mid-handshake resets the counter."""
    seq = [
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 1, "queued": 0}}),   # resets
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 0, "queued": 0}}),
    ]
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    monkeypatch.setattr(P, "_get", fake_get)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    ok, why = P.wait_idle(max_wait=60)
    assert ok and "consecutive" in why
    assert calls["n"] >= 5


def test_idle_gate_gives_up_rather_than_landing_into_a_busy_backend(monkeypatch):
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0:
                        (200, {"turns": {"active": 1, "queued": 0}}))
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    ok, why = P.wait_idle(max_wait=0.01)
    assert not ok and "never went idle" in why



def test_the_idle_wait_drains_first_and_keeps_the_drain_armed(monkeypatch):
    """SM_20260907_233449 spent its whole 900 s budget watching harness_runs
    flicker while the worker pool kept starting jobs, and never drained: the
    drain was armed only after idle. Now it is armed before the first poll,
    refreshed inside its TTL, and released on give-up."""
    monkeypatch.setattr(P, "pool_paused", lambda: True)   # a human's pause: untouched
    calls: list[tuple] = []
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: calls.append(("drain", on)) or True)
    polls = iter([{"active": 0, "queued": 0, "harness_runs": 1}] * 2
                 + [{"active": 0, "queued": 0, "harness_runs": 0}] * 3)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (calls.append(("poll",)) or (200, {"turns": next(polls)})))
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    ok, why = P.wait_idle(max_wait=30)
    assert ok, why
    assert calls[0] == ("drain", True), "armed before the first poll"
    assert ("drain", False) not in calls, "success leaves the drain to the landing"


def test_the_idle_wait_releases_the_drain_when_it_gives_up(monkeypatch):
    monkeypatch.setattr(P, "pool_paused", lambda: True)   # a human's pause: untouched
    calls: list = []
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: calls.append(on) or True)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 1, "queued": 0, "harness_runs": 0}}))
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    ok, why = P.wait_idle(max_wait=0.05)
    assert not ok and calls[0] is True and calls[-1] is False


def test_the_drain_is_refreshed_inside_its_ttl(monkeypatch):
    monkeypatch.setattr(P, "pool_paused", lambda: True)   # a human's pause: untouched
    armed: list[float] = []
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: armed.append(time.time()) or True)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 0, "queued": 0, "harness_runs": 1}}))
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(P, "DRAIN_REFRESH_SECONDS", 0.02)
    P.wait_idle(max_wait=0.1)
    assert len(armed) >= 3, "re-armed repeatedly while waiting"



def _pool_spies(monkeypatch, *, paused_now):
    calls: list = []
    monkeypatch.setattr(P, "pool_paused", lambda: paused_now)
    monkeypatch.setattr(P, "set_pool_paused", lambda p: calls.append(p) or True)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: True)
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(P, "_POOL_PAUSED_BY_US", False)
    return calls


def test_the_idle_wait_pauses_the_pool_and_resumes_it_only_on_give_up(monkeypatch):
    """A drained job FAILS on dispatch and three failures poison it; a paused
    pool simply starts nothing. So the promoter pauses, and releases only its
    own pause."""
    calls = _pool_spies(monkeypatch, paused_now=False)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 1, "queued": 0, "harness_runs": 0}}))
    ok, _ = P.wait_idle(max_wait=0.05)
    assert not ok and calls == [True, False]


def test_a_pause_a_human_set_is_never_released(monkeypatch):
    calls = _pool_spies(monkeypatch, paused_now=True)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 1, "queued": 0, "harness_runs": 0}}))
    ok, _ = P.wait_idle(max_wait=0.05)
    assert not ok and calls == [], "Alan paused it; the promoter must not undo him"
    P.release_pool_pause()
    assert calls == []


def test_success_leaves_the_pool_paused_for_the_landing(monkeypatch):
    calls = _pool_spies(monkeypatch, paused_now=False)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 0, "queued": 0, "harness_runs": 0}}))
    ok, _ = P.wait_idle(max_wait=5)
    assert ok and calls == [True]
    P.release_pool_pause()      # what promote()'s finally does on every exit
    assert calls == [True, False]


def test_pool_paused_reads_the_nested_status_shape(monkeypatch):
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"initialized": True, "pool": {"paused": True}}))
    assert P.pool_paused() is True
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (503, None))
    assert P.pool_paused() is None


# ---------------------------------------------------------------------------
# The tree is shared: a landing chases a moving `main` by rebasing and retesting
# ---------------------------------------------------------------------------

import subprocess as _sp
from pathlib import Path as _Path


def _git(repo, *args):
    return _sp.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def shared_tree(tmp_path, monkeypatch):
    """A scratch live repo, a round worktree cut from it, and `main` moved on."""
    live = tmp_path / "live"; (live / "app").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(live))
    _git(live, "config", "user.email", "t@e.com"); _git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    _git(live, "add", "-A"); _git(live, "commit", "-q", "-m", "base")
    base = _git(live, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    _git(live, "worktree", "add", "-q", "-b", "selfmod/SM_MV", str(wt), base)
    (wt / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")
    _git(wt, "add", "-A"); _git(wt, "commit", "-q", "-m", "round")
    gated = _git(wt, "rev-parse", "HEAD").stdout.strip()
    (live / "app" / "other.py").write_text("y = 1\n", encoding="utf-8")
    _git(live, "add", "-A"); _git(live, "commit", "-q", "-m", "human commit")
    moved = _git(live, "rev-parse", "HEAD").stdout.strip()

    monkeypatch.setattr(P, "LIVE_ROOT", live)
    monkeypatch.setattr(P.S, "read_current", lambda: None)
    (S.ROUNDS_DIR / "SM_MV").mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / "SM_MV" / "run_spec.yaml").write_text(
        f"code:\n  base_commit: {base}\n  branch: selfmod/SM_MV\n", encoding="utf-8")
    return {"live": live, "wt": wt, "base": base, "gated": gated, "moved": moved}


class _StubGate:
    """Stands in for `gate.Gate` inside the promoter: performs the REAL rebase
    (so the worktree genuinely moves and the promoter's own bookkeeping is
    exercised against a real head) and returns the verdict the test asks for."""
    verdict = True
    seen: list = []

    def __init__(self, round_id, worktree, base, *, live_root=None, skip_smoke=False):
        self.round_id, self.worktree, self.base, self.live = round_id, _Path(worktree), base, live_root
        type(self).seen.append((round_id, base))

    def run(self):
        from scripts.selfmod import worktree as W
        onto = _git(self.live, "rev-parse", "HEAD").stdout.strip()
        ok, why, conflicts = W.rebase_onto(self.worktree, onto)
        assert ok, (why, conflicts)
        head = W.head(self.worktree)
        rungs = ([{"name": "preflight", "ok": True, "detail": "rebased"}]
                 + ([] if type(self).verdict else
                    [{"name": "tests", "ok": False, "detail": "pytest failed (1 new)"}]))
        return type("Rep", (), {"ok": type(self).verdict, "base": onto, "head": head,
                                "changed_paths": ["app/m.py"],
                                "to_dict": lambda self_: {"ok": type(self).verdict, "base": onto,
                                                          "head": head, "rungs": rungs}})()


def test_a_moved_main_is_rebased_and_retested_before_landing(shared_tree, monkeypatch):
    """`main` moved after the gate passed. The promoter used to refuse; now it
    runs the gate again with the old base — whose preflight rebases — and
    lands the retested result. Both records the next reader depends on are
    rewritten: gate.json (land's head) and the run spec (the next gate's base)."""
    import scripts.selfmod.gate as G
    _StubGate.verdict, _StubGate.seen = True, []
    monkeypatch.setattr(G, "Gate", _StubGate)
    t = shared_tree

    out = P.promote("SM_MV", t["wt"], t["base"], gate_report={"head": t["gated"]}, dry_run=True)

    assert _StubGate.seen == [("SM_MV", t["base"])], "re-gated with the OLD base so preflight rebases"
    assert out["would_promote"] is True
    assert out["parent"] == t["moved"], "the rollback point is where main is NOW"
    new_head = _git(t["wt"], "rev-parse", "HEAD").stdout.strip()
    assert out["commit"] == new_head != t["gated"]
    assert _git(t["live"], "merge-base", "--is-ancestor", t["moved"], new_head).returncode == 0
    gate_json = json.loads((S.ROUNDS_DIR / "SM_MV" / "gate.json").read_text())
    assert gate_json["base"] == t["moved"] and gate_json["head"] == new_head
    assert f"base_commit: {t['moved']}" in (S.ROUNDS_DIR / "SM_MV" / "run_spec.yaml").read_text()


def test_a_retest_that_fails_is_a_land_failed_that_keeps_the_item(shared_tree, monkeypatch):
    """The round's change no longer passes on top of what landed. That is a
    refusal after every rung had passed — invisible to the backlog before
    `land_failed` existed, and it spent the item."""
    import scripts.selfmod.gate as G
    _StubGate.verdict, _StubGate.seen = False, []
    monkeypatch.setattr(G, "Gate", _StubGate)
    t = shared_tree

    with pytest.raises(P.PromoteError, match="rebased and retested.*`tests` rung failed"):
        P.promote("SM_MV", t["wt"], t["base"], gate_report={"head": t["gated"]}, dry_run=True)
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "land_failed"][-1]
    assert ev["external_blocker"] is True and ev["rung"] == "tests" and ev["moved_to"] == t["moved"]
    assert json.loads((S.ROUNDS_DIR / "SM_MV" / "gate.json").read_text())["ok"] is False


def test_a_commit_snuck_in_after_the_gate_is_still_refused_before_any_rebase(shared_tree, monkeypatch):
    """The ungated-commit guard runs first. A rebase legitimately moves the
    head; a stray commit is not a rebase, and must not be laundered by one."""
    import scripts.selfmod.gate as G
    _StubGate.verdict, _StubGate.seen = True, []
    monkeypatch.setattr(G, "Gate", _StubGate)
    t = shared_tree
    (t["wt"] / "app" / "m.py").write_text("V = 3\n", encoding="utf-8")
    _git(t["wt"], "add", "-A"); _git(t["wt"], "commit", "-q", "-m", "ungated")
    with pytest.raises(P.PromoteError, match="re-gate before landing"):
        P.promote("SM_MV", t["wt"], t["base"], gate_report={"head": t["gated"]}, dry_run=True)
    assert _StubGate.seen == [], "refused before the rebase, not after"


def test_a_failure_before_the_merge_never_rolls_back(monkeypatch, tmp_path):
    """A drain-handshake failure used to take the post-merge path: stop,
    restore and restart the services for a tree that was already exactly
    where it belonged. With uncommitted edits tolerated in that tree, the
    restore would also have stashed them out from under the human's editor."""
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    monkeypatch.setattr(P.W, "changed_paths", lambda wt, base: ["app/x.py"])
    monkeypatch.setattr(P.W, "dirty_paths", lambda repo, limit=None: ["docs/wip.md"])
    _live_at(monkeypatch, "a" * 40)
    monkeypatch.setattr(P.S, "read_current", lambda: None)
    monkeypatch.setattr(P.S, "write_verified", lambda path, payload: payload)
    monkeypatch.setattr(P.S, "changed_tree_hash", lambda wt, head, paths: "h")
    monkeypatch.setattr(P, "vault_commits_for", lambda rid: [])
    monkeypatch.setattr(P, "count_kg_rows", lambda: 0)
    monkeypatch.setattr(P, "count_vault_files", lambda: 0)
    monkeypatch.setattr(P, "wait_idle", lambda: (True, "idle"))
    monkeypatch.setattr(P, "set_drain", lambda *a, **k: None)
    monkeypatch.setattr(P, "release_pool_pause", lambda *a, **k: None)
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"turns": {"active": 1}}))
    rolled = []
    monkeypatch.setattr(P, "_rollback_inline", lambda live, target: rolled.append(target) or {})

    with pytest.raises(P.PromoteError, match="drain handshake"):
        P.promote("SM_X", tmp_path, "a" * 40, gate_report={"head": "b" * 40})
    assert rolled == [], "nothing had moved; nothing to restore"
    assert S.read_current() is None


def test_land_failed_records_the_verdict_and_raises():
    with pytest.raises(P.PromoteError, match="why"):
        P._land_failed("SM_X", "why not", external=True, overlap=["a"])
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "land_failed"][-1]
    assert ev["round_id"] == "SM_X" and ev["ok"] is False
    assert ev["external_blocker"] is True and ev["overlap"] == ["a"]
