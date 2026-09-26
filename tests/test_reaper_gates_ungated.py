"""A round whose turn ended with work nobody gated is gated, not destroyed.

The week to 2026-09-25, by the ledger: 139 of 489 rounds were aborted as
"implement turn ended with the round still open, no gate or landing" — 100 of
them on `stop`, a model writing its closing report with the change sitting in
the worktree — and 39 more as "the review rung sent the round back and the
turn ended without abort or re-gate". Each went back to the pool for a whole
new round, and the abort itself is `git worktree remove --force`: whatever the
turn had not committed was not kept on the branch, it was deleted.

`autocode._gate_if_ungated` commits the leftovers and gates the round with
`land_on_pass`, so a pass lands without waiting for the reaper's next look.
These run against a real git repository, because `commit_pending` and
`changed_paths` are the part that has to be right.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.automod import backlog as B, round as R, state as S, worktree as W
from workers.sources import _common as C
from workers.sources import autocode as I

RID = "SM_20260925_101500"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """One open round in a real repository: base commit, run spec, nothing
    running in its session, the loop enabled."""
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
    monkeypatch.setattr(C, "source_inner_voice", lambda source, default=True: True)
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [])

    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "mod.py").write_text("X = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(W, "worktree_path", lambda rid: repo)

    (S.ROUNDS_DIR / RID).mkdir(parents=True)
    (S.ROUNDS_DIR / RID / "run_spec.yaml").write_text(yaml.safe_dump(
        {"code": {"base_commit": base, "branch": f"automod/{RID}"}, "item": {"id": 1234}}))

    world = {"repo": repo, "base": base, "aborted": [], "spawned": []}
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
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
        encoding="utf-8")
    return world


def _commit(env, text="X = 2\n"):
    (env["repo"] / "mod.py").write_text(text)
    _git(env["repo"], "commit", "-qam", "work")
    return _git(env["repo"], "rev-parse", "HEAD")


def _gate(head, ok=False):
    (S.ROUNDS_DIR / RID / "gate.json").write_text(json.dumps(
        {"round_id": RID, "ok": ok, "head": head, "base": "b", "rungs": [
            {"name": "tests", "ok": ok, "detail": "pytest failed"}]}), encoding="utf-8")


def _finished(outcome=None, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": 1234, "phase": "finished",
                    "round_id": RID, "session_id": "s-1234", "stop_reason": stop_reason,
                    "outcome": outcome}, path=S.LEDGER_PATH)


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


NOT_MET = {"landed": False, "acceptance": "not_met", "deferred_to": [], "summary": "ran out",
           "spawned": [], "clause_outcomes": [{"clause": 1, "outcome": "not_met",
                                               "evidence": "", "deferred_to": []}]}


# ── it gates ────────────────────────────────────────────────────────────────

def test_a_committed_change_that_was_never_gated_is_gated_and_lands_on_pass(env):
    head = _commit(env)
    _finished(NOT_MET)
    out = I.reap_abandoned_rounds()
    assert env["aborted"] == [], "a finished change went back to the pool"
    argv = env["spawned"][0]
    assert argv[-4:] == ["gate", RID, "--land-on-pass", "reaper"]
    assert [(r["round_id"], r["verb"], r["kind"]) for r in out] == [(RID, "gating", "ungated")]
    row = _events("gate_rescued")[0]
    assert row["head"] == head and row["committed"] is False and "never gated" in row["reason"]
    assert S.gate_in_progress(RID)["by"] == "reaper"
    assert I.reap_abandoned_rounds() == [], "the running gate is left alone"


def test_uncommitted_work_is_committed_first_rather_than_deleted_by_the_abort(env):
    """The abort is `worktree remove --force`. Before this, these edits were
    simply gone."""
    (env["repo"] / "mod.py").write_text("X = 3\n")
    (env["repo"] / "new_test.py").write_text("def test_x():\n    assert True\n")
    (env["repo"] / "__pycache__").mkdir()
    (env["repo"] / "__pycache__" / "junk.pyc").write_bytes(b"\0")
    _finished(None, stop_reason="max_turns")
    I.reap_abandoned_rounds()
    assert env["aborted"] == []
    assert _git(env["repo"], "status", "--porcelain") == "", "nothing left for the abort to delete"
    tracked = _git(env["repo"], "ls-files").split()
    assert "new_test.py" in tracked and "__pycache__/junk.pyc" not in tracked
    assert _git(env["repo"], "show", "HEAD:mod.py") == "X = 3"
    assert "left uncommitted" in _git(env["repo"], "log", "-1", "--format=%s")
    assert _events("gate_rescued")[0]["committed"] is True


def test_a_fix_committed_after_a_refused_gate_is_gated_again(env):
    """The 39: review sent the round back, the turn committed the fix and
    stopped without re-gating."""
    first = _commit(env)
    _gate(first, ok=False)
    _commit(env, "X = 4\n")
    _finished(NOT_MET)
    I.reap_abandoned_rounds()
    assert env["aborted"] == [] and env["spawned"]
    assert "changed since" in _events("gate_rescued")[0]["reason"]


# ── it does not ─────────────────────────────────────────────────────────────

def _expect_close(env):
    out = I.reap_abandoned_rounds()
    assert env["spawned"] == [] and env["aborted"] == [RID]
    assert [r["round_id"] for r in out] == [RID]
    assert not [e for e in _events("gate_rescued") if e.get("kind") == "ungated"][1:]


def test_a_gate_refusal_at_the_rounds_own_commit_is_a_verdict(env):
    _gate(_commit(env), ok=False)
    _finished(NOT_MET)
    _expect_close(env)


def test_a_round_with_no_change_against_its_base_is_closed(env):
    _finished(NOT_MET)
    _expect_close(env)


def test_one_ungated_rescue_per_round(env):
    head = _commit(env)
    _finished(NOT_MET)
    I.reap_abandoned_rounds()
    S.clear_gate_marker(RID)
    env["spawned"].clear()
    _commit(env, "X = 9\n")  # moved again after the rescue's gate
    _gate(head, ok=False)
    _expect_close(env)


@pytest.mark.parametrize("verdict", ["rejected", "unnecessary"])
def test_an_item_verdict_is_the_authors_call_and_nothing_is_committed(env, verdict):
    (env["repo"] / "mod.py").write_text("X = 5\n")
    _finished({**NOT_MET, "acceptance": verdict})
    _expect_close(env)
    assert _git(env["repo"], "rev-parse", "HEAD") == env["base"]


@pytest.mark.parametrize("blocked", ["is_halted", "is_broken", "disabled", "switch"])
def test_nothing_is_started_on_a_stopped_loop(env, monkeypatch, blocked):
    _commit(env)
    _finished(NOT_MET)
    if blocked == "disabled":
        monkeypatch.setattr(S, "is_enabled", lambda repo=None: False)
    elif blocked == "switch":
        monkeypatch.setattr(I, "_source_cfg", lambda name: {"gate_ungated": False})
    else:
        monkeypatch.setattr(S, blocked, lambda: True)
    _expect_close(env)


def test_the_unreviewed_regate_also_lands_on_pass(env):
    """`_regate_if_unreviewed` is the other gate no turn waits on."""
    head = _commit(env)
    (S.ROUNDS_DIR / RID / "gate.json").write_text(json.dumps(
        {"round_id": RID, "ok": False, "head": head, "base": "b", "rungs": [
            {"name": "review", "ok": False, "detail": "review could not run",
             "data": {"external_blocker": True}}]}), encoding="utf-8")
    _finished(NOT_MET)
    I.reap_abandoned_rounds()
    assert env["spawned"][0][-2:] == ["--land-on-pass", "reaper"]


# ── the gate the reaper started lands itself ────────────────────────────────

def _report(env, ok=True):
    return {"round_id": RID, "ok": ok, "head": _git(env["repo"], "rev-parse", "HEAD")}


def test_run_gate_with_land_on_pass_starts_the_landing_and_records_it(env, monkeypatch):
    _commit(env)
    rep = _report(env)
    monkeypatch.setattr(R, "_run_gate", lambda rid, skip_smoke=False: rep)
    landed = []
    monkeypatch.setattr(R, "land_detached", lambda rid, by: landed.append((rid, by)) or {"pid": 7})
    R.run_gate(RID, land_on_pass="reaper")
    assert landed == [(RID, "reaper")]
    row = _events("land_rescued")[0]
    assert row["item_id"] == 1234 and row["by"] == "reaper" and row["head"] == rep["head"]


@pytest.mark.parametrize("why", ["failed", "moved", "halted", "not_asked"])
def test_run_gate_does_not_land_what_it_should_not(env, monkeypatch, why):
    _commit(env)
    rep = _report(env, ok=(why != "failed"))
    if why == "moved":
        rep["head"] = "0" * 40
    if why == "halted":
        monkeypatch.setattr(S, "is_halted", lambda: True)
    monkeypatch.setattr(R, "_run_gate", lambda rid, skip_smoke=False: rep)
    landed = []
    monkeypatch.setattr(R, "land_detached", lambda rid, by: landed.append(rid) or {"pid": 7})
    R.run_gate(RID, land_on_pass=None if why == "not_asked" else "reaper")
    assert landed == [] and not _events("land_rescued")


# ── and the item closes on the review, not on the pre-gate outcome ──────────

def _landed_and_settled(outcome, *, ungated=True, verdicts=("met", "met")):
    _finished(outcome)
    if ungated:
        S.append_event({"event": "gate_rescued", "kind": "ungated", "round_id": RID,
                        "item_id": 1234}, path=S.LEDGER_PATH)
    S.append_event({"event": "review", "round_id": RID, "ok": True, "blocking": False,
                    "clauses": [{"clause": i + 1, "verdict": v, "evidence": "ran it"}
                                for i, v in enumerate(verdicts)]}, path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": RID, "commit": "c0ffee"}, path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": "c0ffee"}, path=S.LEDGER_PATH)


def test_a_reaper_gated_landing_closes_on_the_reviews_all_met(env):
    _landed_and_settled(NOT_MET)
    [row] = [r for r in B.settled_landings(S.LEDGER_PATH) if r["item_id"] == 1234]
    assert row["outcome"]["acceptance"] == "met"
    assert B.implement_outcomes(S.LEDGER_PATH)[1234][0] == "spent", "re-offered a landed change"


def test_an_ordinary_landing_keeps_the_turns_own_word(env):
    _landed_and_settled(NOT_MET, ungated=False)
    [row] = [r for r in B.settled_landings(S.LEDGER_PATH) if r["item_id"] == 1234]
    assert row["outcome"]["acceptance"] == "not_met"
    assert B.implement_outcomes(S.LEDGER_PATH)[1234][0] == "partial"


def test_a_reaper_gated_landing_the_review_did_not_vouch_for_keeps_the_turns_word(env):
    _landed_and_settled(NOT_MET, verdicts=("met", "partial"))
    [row] = [r for r in B.settled_landings(S.LEDGER_PATH) if r["item_id"] == 1234]
    assert row["outcome"]["acceptance"] == "not_met"
