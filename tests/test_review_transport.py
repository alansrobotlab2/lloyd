"""2026-09-11: eighteen rounds, seventeen aborts, zero landings — and not one
review finding reached the model that could have acted on it.

The chain: a gate with the review rung runs longer than the MCP transport's
read timeout; the pool re-sent the call and gated the same round twice; the
review cap counted both; `review_disagreement` read the duplicate pair as the
author refusing to fix a clause. This file pins each link as fixed, plus the
three things the loop can now do for itself that the day showed it could not:
refuse to gate over its own running measurement, amend a clause the reviewer
calls unsatisfiable (subject to ratification), and keep human-only
conditions out of the contract it is graded on.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import annotations as A
from agent_mcp import automod as T
from scripts.automod import backlog as B, gate as G, review as RV, round as R
from scripts.automod import state as S, worktree as W
from workers.sources import _common as C
from workers.sources import autocode as I
from workers.sources import autotriage as AT


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    return d


def write_item(d: Path, item_id, *, status="up_next", clauses=None, human=None, board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": board, "tags": []}
    if clauses:
        fm["acceptance_clauses"] = list(clauses)
    if human:
        fm["human_clauses"] = list(human)
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    return path


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


def _review(rid, item_id, *, head, clauses, blocking=True, ok=True, attempt=1, findings="f"):
    S.append_event({"event": "review", "round_id": rid, "item_id": item_id, "ok": ok,
                    "blocking": blocking, "kind": "retry" if blocking else "pass",
                    "premise": "sound", "attempt": attempt, "head": head, "findings": findings,
                    "clauses": [{"clause": i, "verdict": v} for i, v in clauses.items()]},
                   path=S.LEDGER_PATH)


# ── the marker: one gate per round ────────────────────────────────────────

def test_gate_marker_round_trip_and_liveness():
    S.write_gate_marker("SM_m", pid=os.getpid(), head="abc", by="test")
    assert S.read_gate_marker("SM_m")["pid"] == os.getpid()
    assert S.gate_in_progress("SM_m")["head"] == "abc"
    S.write_gate_marker("SM_m", pid=2 ** 22 + 12345)  # almost certainly no such pid
    assert S.gate_in_progress("SM_m") is None, "a dead pid is not a running gate"
    assert S.read_gate_marker("SM_m") is not None, "…but the marker is left for the reader to judge"
    S.clear_gate_marker("SM_m")
    assert S.read_gate_marker("SM_m") is None
    S.clear_gate_marker("SM_m")  # idempotent


class _Report:
    def __init__(self, base):
        self.base = base
        self.ok = True

    def to_dict(self):
        return {"ok": True, "base": self.base, "rungs": []}


def _arm_run_gate(monkeypatch, tmp_path, *, run):
    wt = tmp_path / "wt"; wt.mkdir()
    (S.ROUNDS_DIR / "SM_g").mkdir(parents=True)
    (S.ROUNDS_DIR / "SM_g" / "run_spec.yaml").write_text(yaml.safe_dump(
        {"code": {"base_commit": "b" * 40}, "item": {"id": 7}}))
    monkeypatch.setattr(R.W, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(R.W, "head", lambda p: "h" * 40)

    class FakeGate:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return run()
    monkeypatch.setattr(R.G, "Gate", FakeGate)


def test_run_gate_owns_the_marker_for_its_lifetime_and_clears_it(monkeypatch, tmp_path):
    seen = {}

    def run():
        seen["marker"] = S.read_gate_marker("SM_g")
        return _Report("b" * 40)
    _arm_run_gate(monkeypatch, tmp_path, run=run)
    R.run_gate("SM_g")
    assert seen["marker"]["pid"] == os.getpid() and seen["marker"]["by"] == "run_gate"
    assert S.read_gate_marker("SM_g") is None
    assert (S.ROUNDS_DIR / "SM_g" / "gate.json").exists()


def test_run_gate_clears_the_marker_when_the_ladder_raises(monkeypatch, tmp_path):
    def run():
        raise RuntimeError("boom")
    _arm_run_gate(monkeypatch, tmp_path, run=run)
    with pytest.raises(RuntimeError):
        R.run_gate("SM_g")
    assert S.read_gate_marker("SM_g") is None


def test_run_gate_refuses_a_second_concurrent_gate(monkeypatch, tmp_path):
    _arm_run_gate(monkeypatch, tmp_path, run=lambda: _Report("b" * 40))
    S.write_gate_marker("SM_g", pid=os.getppid(), by="other")  # a live pid that is not us
    with pytest.raises(RuntimeError, match="already running"):
        R.run_gate("SM_g")


# ── the tool: detached start, bounded wait ────────────────────────────────

def test_automod_gate_refuses_over_a_running_background_task(monkeypatch, tmp_path):
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "wt")
    monkeypatch.setattr(T, "_background_tasks_for_session",
                        lambda: [{"task_id": "bg-1", "description": "scored run", "elapsed_s": 40}])
    spawned = []
    monkeypatch.setattr(S, "spawn_detached", lambda *a, **k: spawned.append(a) or 1)
    out = T._gate_detached("SM_t")
    assert "error" in out and out["background_tasks"][0]["task_id"] == "bg-1"
    assert not spawned


def test_automod_gate_starts_detached_writes_the_marker_and_says_wait(monkeypatch, tmp_path):
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "wt")
    monkeypatch.setattr(W, "head", lambda p: "c" * 40)
    monkeypatch.setattr(T, "_background_tasks_for_session", lambda: [])
    spawned = []
    monkeypatch.setattr(S, "spawn_detached", lambda argv, log, cwd=None: spawned.append(argv) or 4242)
    out = T._gate_detached("SM_t", skip_smoke=True)
    assert out["gate_started"] == "SM_t" and out["pid"] == 4242
    assert "automod_gate_wait" in out["next"] and "Do NOT edit" in out["next"]
    argv = [str(a) for a in spawned[0]]
    assert argv[-3:] == ["gate", "SM_t", "--skip-smoke"]
    assert S.read_gate_marker("SM_t")["pid"] == 4242


def test_automod_gate_on_a_running_round_does_not_start_a_second(monkeypatch, tmp_path):
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "wt")
    S.write_gate_marker("SM_t", pid=os.getpid(), by="x")
    spawned = []
    monkeypatch.setattr(S, "spawn_detached", lambda *a, **k: spawned.append(a) or 1)
    out = T._gate_detached("SM_t")
    assert out["running"] is True and "already running" in out["note"] and not spawned


def test_gate_wait_returns_the_report_once_the_marker_is_gone(tmp_path):
    (S.ROUNDS_DIR / "SM_w").mkdir(parents=True)
    (S.ROUNDS_DIR / "SM_w" / "gate.json").write_text(json.dumps({"ok": False, "rungs": [
        {"name": "review", "ok": False, "detail": "clause 2 unmet"}]}))
    out = T._gate_wait("SM_w", wait_seconds=0)
    assert out["gate_finished"] and out["rungs"][0]["detail"] == "clause 2 unmet"


def test_gate_wait_reports_progress_while_the_process_lives(tmp_path):
    S.write_gate_marker("SM_w", pid=os.getpid(), by="x")
    S.append_event({"event": "gate", "round_id": "SM_w", "rung": "tests", "ok": True,
                    "detail": "3000 passed", "seconds": 120}, path=S.LEDGER_PATH)
    import time
    t0 = time.time()
    out = T._gate_wait("SM_w", wait_seconds=0)
    assert time.time() - t0 < 5, "0 means look once — the first cut read it as `or 240`"
    assert out["running"] is True and out["rungs_so_far"] == ["PASS tests (120s): 3000 passed"]
    assert "again" in out["next"]


def test_gate_wait_names_a_gate_that_died_without_a_report(tmp_path):
    S.write_gate_marker("SM_w", pid=2 ** 22 + 4321, by="x")
    out = T._gate_wait("SM_w", wait_seconds=0)
    assert "died" in out["error"] and S.read_gate_marker("SM_w") is None


def test_gate_wait_with_no_gate_says_so():
    out = T._gate_wait("SM_none", wait_seconds=0)
    assert "call automod_gate first" in out["error"]


def test_land_refuses_while_a_gate_is_running(monkeypatch, tmp_path):
    (S.ROUNDS_DIR / "SM_l").mkdir(parents=True)
    (S.ROUNDS_DIR / "SM_l" / "gate.json").write_text(json.dumps({"ok": True, "rungs": []}))
    S.write_gate_marker("SM_l", pid=os.getpid(), by="x")
    assert "still running" in T._land_detached("SM_l")["error"]


# ── the rung: attempts are graded refusals of distinct commits ────────────

class _Gate(G.Gate):
    def __init__(self, item_id, changed, tmp_path, *, live=None):
        super().__init__("SM_REV", tmp_path, "a" * 40, item_id=item_id, live_root=live)
        self.report.changed_paths = changed
        self.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))


def _grader(structured, *, ok=True, error=""):
    def grade(**kw):
        grade.calls.append(kw)
        return {"ok": ok, "error": error, "session_id": "sess_r", "structured": structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}
    grade.calls = []
    return grade


def _arm(monkeypatch, tmp_path, *, grade, prior=(), head="h" * 40, contract=None):
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(prior))
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(G.W, "head", lambda p: head)
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: contract or {
        "id": iid, "title": "t", "body": "b", "clauses": ["the thing happens once"], "path": "",
        "amendments": [], "human_clauses": []})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    return events


UNMET = {"premise": "sound", "summary": "", "clauses": [
    {"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
     "test_node_id": "", "how_verified": "inferred", "note": "still"}],
    "test_honesty": [], "seams_unverified": [], "amendments_ok": True, "amendments_note": ""}


def _refusal(head, attempt, findings="f"):
    return {"event": "review", "round_id": "SM_REV", "ok": True, "blocking": True,
            "attempt": attempt, "head": head, "findings": findings}


def test_the_same_commit_refused_before_is_answered_from_the_ledger(monkeypatch, tmp_path):
    grade = _grader(UNMET)
    _arm(monkeypatch, tmp_path, grade=grade, prior=[_refusal("h" * 40, 1, "clause 1 unmet: still")])
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_same_head"] and data["review_attempt"] == 1
    assert "nothing has been committed since" in detail and "clause 1 unmet: still" in detail
    assert grade.calls == [], "no grading turn for a commit the reviewer already judged"


def test_a_grader_timeout_spends_no_attempt(monkeypatch, tmp_path):
    grade = _grader(UNMET)
    timed_out = {"event": "review", "round_id": "SM_REV", "ok": False, "blocking": False,
                 "error": "review exceeded 600.0s"}
    _arm(monkeypatch, tmp_path, grade=grade, prior=[timed_out, timed_out, timed_out])
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data["review_attempt"] == 1 and len(grade.calls) == 1
    assert "1/2" in detail


def test_two_distinct_commits_refused_exhausts_and_the_findings_are_the_last(monkeypatch, tmp_path):
    grade = _grader(UNMET)
    _arm(monkeypatch, tmp_path, grade=grade, head="c" * 40,
         prior=[_refusal("a" * 40, 1, "first"), _refusal("b" * 40, 2, "second")])
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data["review_exhausted"] and data["review_attempt"] == 3
    assert data["review_findings"] == "second" and grade.calls == []


def test_a_duplicate_pair_on_one_commit_counts_once(monkeypatch, tmp_path):
    """The 2026-09-11 shape: two review rows for the same head. One attempt."""
    grade = _grader(UNMET)
    _arm(monkeypatch, tmp_path, grade=grade, head="c" * 40,
         prior=[_refusal("a" * 40, 1), _refusal("a" * 40, 1)])
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data["review_attempt"] == 2 and len(grade.calls) == 1
    assert "abort and report" in detail


def test_the_review_event_carries_the_head_and_the_snapshot_verdict(monkeypatch, tmp_path):
    events = _arm(monkeypatch, tmp_path, grade=_grader(UNMET), head="")
    _Gate(7, ["app/x.py"], tmp_path).rung_review()
    ev = events[-1]
    assert ev["event"] == "review" and "head" in ev and ev["snapshot"] is False
    assert "working tree" in ev["snapshot_note"]


@pytest.fixture
def live_repo(tmp_path):
    repo = tmp_path / "live"; repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@t"); git(repo, "config", "user.name", "t")
    (repo / "app").mkdir(); (repo / "app" / "x.py").write_text("x = 1\n")
    git(repo, "add", "."); git(repo, "commit", "-q", "-m", "base")
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_REV", str(wt))
    (wt / "app" / "x.py").write_text("x = 2\n")
    git(wt, "commit", "-q", "-am", "change")
    return repo, wt


def test_the_grader_reads_a_detached_checkout_of_the_commit_not_the_working_tree(
        monkeypatch, tmp_path, live_repo):
    repo, wt = live_repo
    head = git(wt, "rev-parse", "HEAD").stdout.strip()
    seen = {}

    def grade(**kw):
        root = Path(kw["worktree"])
        seen["root"] = root
        seen["head"] = git(root, "rev-parse", "HEAD").stdout.strip()
        seen["content"] = (root / "app" / "x.py").read_text()
        seen["existed"] = root.exists()
        return {"ok": True, "error": "", "session_id": "s", "structured": UNMET,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1}
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [])
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b", "clauses": ["c"], "path": "",
        "amendments": [], "human_clauses": []})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    # The author keeps editing: the working tree is dirty when the review runs.
    (wt / "app" / "x.py").write_text("x = 3  # mid-write\n")
    g = _Gate(7, ["app/x.py"], wt, live=repo)
    g.rung_review()
    assert seen["existed"] and seen["root"] != wt
    assert seen["head"] == head and seen["content"] == "x = 2\n", "the commit, not the edit"
    assert not seen["root"].exists(), "the snapshot is removed after grading"
    assert "worktree" not in " ".join(git(repo, "worktree", "list").stdout.split("\n")[1:]) or \
        str(seen["root"]) not in git(repo, "worktree", "list").stdout
    assert events[-1]["snapshot"] is True and events[-1]["head"] == head


# ── the backlog: a duplicate pair is not a disagreement ───────────────────

def test_two_refusals_of_one_commit_are_not_a_disagreement(isolated):
    write_item(isolated, 578, clauses=["a", "b"])
    for rid in ("SM_1",):
        S.append_event({"event": "backlog_implement", "item_id": 578, "phase": "finished",
                        "round_id": rid, "stop_reason": "stop", "num_turns": 5}, path=S.LEDGER_PATH)
    _review("SM_1", 578, head="h" * 40, clauses={1: "met", 2: "unmet"}, attempt=1)
    _review("SM_1", 578, head="h" * 40, clauses={1: "met", 2: "unmet"}, attempt=1)
    assert B.review_disagreement(S.LEDGER_PATH, 578) is None
    _review("SM_1", 578, head="i" * 40, clauses={1: "met", 2: "unmet"}, attempt=2)
    assert B.review_disagreement(S.LEDGER_PATH, 578) == 2
    # Rows without a head — every review before 2026-09-11 — keep the old reading.
    write_item(isolated, 579, clauses=["a"])
    S.append_event({"event": "backlog_implement", "item_id": 579, "phase": "finished",
                    "round_id": "SM_2", "stop_reason": "stop", "num_turns": 5}, path=S.LEDGER_PATH)
    _review("SM_2", 579, head="", clauses={1: "unmet"})
    _review("SM_2", 579, head="", clauses={1: "unmet"})
    assert B.review_disagreement(S.LEDGER_PATH, 579) == 1


# ── amendments: the contract can move, under the second reader's eye ─────

def _unsat_review(rid, item_id, clause):
    _review(rid, item_id, head="h" * 40, clauses={clause: "unsatisfiable"})


def test_amend_refuses_a_clause_the_reviewer_did_not_call_unsatisfiable(isolated):
    path = write_item(isolated, 600, clauses=["a", "b"])
    with pytest.raises(ValueError, match="not judged clause 1 unsatisfiable"):
        B.amend_clause(600, 1, "a, narrowed", "because", round_id="SM_a")
    _unsat_review("SM_a", 600, 2)
    with pytest.raises(ValueError, match="not judged clause 1"):
        B.amend_clause(600, 1, "a, narrowed", "because", round_id="SM_a")
    assert _fm(path)["acceptance_clauses"] == ["a", "b"]


def test_amend_writes_pending_and_the_next_review_ratifies_or_restores(isolated):
    path = write_item(isolated, 601, clauses=["a", "b"])
    _unsat_review("SM_a", 601, 2)
    rec = B.amend_clause(601, 2, "b, satisfiable", "the held-out half is in USER.md", round_id="SM_a")
    fm = _fm(path)
    assert fm["acceptance_clauses"] == ["a", "b, satisfiable"]
    assert B.pending_amendments(fm) == [rec] and rec["state"] == "pending" and rec["was"] == "b"
    assert "amended clause 2" in fm["activity_log"][-1]
    with pytest.raises(ValueError, match="already amended"):
        B.amend_clause(601, 2, "again", "x", round_id="SM_a")
    # the contract the grader and implementer read is the amended one
    assert B.acceptance_clauses_of({}, fm) == ["a", "b, satisfiable"]
    assert B.settle_amendments(601, "SM_a", ratified=False, note="weaker") == [2]
    fm = _fm(path)
    assert fm["acceptance_clauses"] == ["a", "b"] and fm["clause_amendments"][0]["state"] == "refused"
    assert B.pending_amendments(fm) == []
    # …and a ratification keeps the text
    _unsat_review("SM_b", 601, 2)
    B.amend_clause(601, 2, "b, take two", "still", round_id="SM_b")
    assert B.settle_amendments(601, "SM_b", ratified=True) == [2]
    fm = _fm(path)
    assert fm["acceptance_clauses"] == ["a", "b, take two"]
    assert fm["clause_amendments"][-1]["state"] == "ratified"
    assert B.settle_amendments(601, "SM_b", ratified=True) == [], "settled once"


def test_decide_names_the_amend_tool_and_a_refused_amendment(tmp_path):
    parsed = RV.parse_review({"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "unsatisfiable", "evidence_path": "", "evidence_line": 0,
         "test_node_id": "", "how_verified": "read", "note": "the split is contaminated"}],
        "test_honesty": [], "seams_unverified": [], "amendments_ok": False,
        "amendments_note": "the new clause drops the A/B"},
        worktree=tmp_path, changed_tests=[], n_clauses=1)
    assert parsed["amendments_ok"] is False and parsed["amendments_note"].startswith("the new")
    kind, findings = RV.decide(parsed, [], amendments=[{"clause": 1}])
    assert kind == "retry"
    assert "amendment of clause(s) 1 refused" in findings and "drops the A/B" in findings
    assert "automod_amend_clause(round_id, clause=1" in findings
    # no amendments shown → the flag is inert whatever the grader wrote
    kind, findings = RV.decide(parsed, [], amendments=[])
    assert "refused" not in findings
    # absent from the object → ok
    assert RV.parse_review({"premise": "sound", "summary": "", "clauses": [], "test_honesty": [],
                            "seams_unverified": []}, worktree=tmp_path, changed_tests=[],
                           n_clauses=1)["amendments_ok"] is True


def test_the_review_schema_carries_the_new_verdict_and_the_ratification_flag():
    props = RV.REVIEW_SCHEMA["properties"]
    assert "unsatisfiable" in props["clauses"]["items"]["properties"]["verdict"]["enum"]
    assert "amendments_ok" in RV.REVIEW_SCHEMA["required"]
    assert RV.summarize_clauses({"clauses": [{"verdict": "unsatisfiable"}]}) == "1 unsatisfiable"


def test_the_rung_settles_amendments_with_the_verdict(monkeypatch, tmp_path):
    settled = []
    monkeypatch.setattr(B, "settle_amendments",
                        lambda iid, rid, *, ratified, note="": settled.append((iid, rid, ratified)) or [1])
    obj = dict(UNMET, amendments_ok=False, amendments_note="weaker")
    contract = {"id": 7, "title": "t", "body": "b", "clauses": ["c"], "path": "",
                "amendments": [{"clause": 1, "was": "c0", "now": "c", "reason": "r", "round_id": "SM_REV"}],
                "human_clauses": []}
    _arm(monkeypatch, tmp_path, grade=_grader(obj), head="", contract=contract)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert settled == [(7, "SM_REV", False)] and "refused" in detail


def test_the_prompt_shows_amendments_and_human_clauses(tmp_path):
    contract = {"id": 7, "title": "t", "body": "b", "clauses": ["c1", "c2"], "path": "",
                "amendments": [{"clause": 2, "was": "old", "now": "new", "reason": "why", "round_id": "SM_x"}],
                "human_clauses": ["Alan audits ten items"]}
    text = RV.build_prompt(contract=contract, diff="", diff_truncated=False, changed_tests=[],
                           test_counts={}, worktree=tmp_path, run_tests=tmp_path / "rt")
    assert "<amendments>" in text and "was: old" in text and "now: new" in text
    assert "<human_clauses>" in text and "Alan audits ten items" in text
    assert "graded here" in text
    assert "`unsatisfiable` if NO diff" in text
    # A clause that needs live traffic is `post_landing` — it lands and waits
    # for a person — while `unsatisfiable` stays for a clause no diff could
    # ever satisfy, whose remedy is an amendment. Telling the grader to use
    # the second for the first is what parked #859 with the mechanism
    # complete on both commits.
    assert "only be OBSERVED after the change has landed" in text
    assert "is `post_landing`" in text
    assert "does not refuse the round" in text
    assert "not `partial` and not `unsatisfiable`" in text


def test_the_new_tools_are_denied_to_the_grader_and_to_workers_and_annotated():
    for name in ("automod_gate_wait", "automod_amend_clause"):
        assert name in RV.REVIEW_DENY
        assert name in C.WORKER_AUTOMOD_BAN
    assert "automod_amend_clause" in A.REPEAT_EXPECTED
    assert "automod_gate_wait" in A.READ_ONLY and "automod_gate_wait" not in A.REPEAT_EXPECTED


def test_amend_tool_binds_the_item_from_the_run_spec(isolated, tmp_path):
    (S.ROUNDS_DIR / "SM_a").mkdir(parents=True)
    (S.ROUNDS_DIR / "SM_a" / "run_spec.yaml").write_text(yaml.safe_dump(
        {"code": {"base_commit": "b" * 40}, "item": {"id": 602}}))
    write_item(isolated, 602, clauses=["a"])
    out = T._amend_clause("SM_a", 1, "a'", "r")
    assert "not judged clause 1 unsatisfiable" in out["error"]
    _unsat_review("SM_a", 602, 1)
    out = T._amend_clause("SM_a", 1, "a'", "r")
    assert out["amended"]["now"] == "a'" and out["item_id"] == 602 and "Gate again" in out["next"]
    assert "no run spec" in T._amend_clause("SM_none", 1, "a", "r")["error"]


# ── human clauses: a person's job stays out of the contract ──────────────

def test_triage_parses_human_clauses_from_both_paths():
    text = ("VERDICT: confirmed\nSURFACE: code\nCHECK: x\nEVIDENCE: y\nACCEPTANCE: z\n"
            "ACCEPTANCE_CLAUSES:\n1. a runner exists\nHUMAN_CLAUSES:\n1. Alan audits ten items\n"
            "SPAWNED: none\n")
    parsed = AT.parse_verdict(text)
    assert parsed["acceptance_clauses"] == ["a runner exists"]
    assert parsed["human_clauses"] == ["Alan audits ten items"]
    structured = AT._from_structured({"verdict": "confirmed", "surface": "code", "check": "x",
                                      "evidence": "y", "acceptance": "z",
                                      "acceptance_clauses": ["a"], "human_clauses": ["h"],
                                      "spawned": []}, B.VERDICTS, B.SURFACES)
    assert structured["human_clauses"] == ["h"]
    assert "human_clauses" in B.TRIAGE_VERDICT_SCHEMA["required"]
    assert "HUMAN_CLAUSES:" in AT.PROMPT and "not an acceptance clause" in AT.PROMPT
    assert "can only run after landing" in AT.PROMPT


def test_record_verdict_writes_human_clauses_beside_the_contract(isolated):
    path = write_item(isolated, 610, status="draft")
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "ev", check="c", close=False, acceptance="acc",
                     acceptance_clauses=["a"], human_clauses=["Alan signs off"])
    fm = _fm(path)
    assert fm["acceptance_clauses"] == ["a"] and fm["human_clauses"] == ["Alan signs off"]
    assert "Needs a person before this closes" in path.read_text()
    assert B.human_clauses_of({}, fm) == ["Alan signs off"]
    assert B.human_clauses_for_item(path, {}) == ["Alan signs off"]
    assert B.human_clauses_for_item(None, {"human_clauses": ["x"]}) == ["x"]


def test_a_met_landing_with_human_clauses_stays_open_and_tagged(isolated):
    path = write_item(isolated, 611, status="in_progress", clauses=["a"], human=["Alan audits"])
    S.append_event({"event": "backlog_implement", "item_id": 611, "phase": "finished",
                    "round_id": "SM_h", "stop_reason": "stop", "num_turns": 5,
                    "outcome": {"landed": True, "acceptance": "met", "clause_outcomes": []}},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": "SM_h", "commit": "d" * 40}, path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": "d" * 40}, path=S.LEDGER_PATH)
    done = B.close_settled_items(S.LEDGER_PATH)
    assert done == [{"item_id": 611, "closed": False, "acceptance": "met"}]
    fm = _fm(path)
    assert fm["status"] != "done" and B.NEEDS_HUMAN_TAG in fm["tags"]
    assert "still waiting on a person for: Alan audits" in path.read_text()
    assert fm[B.LANDED_MARKER] == "d" * 40
    # …and without human clauses the same landing closes, as before
    path2 = write_item(isolated, 612, status="in_progress", clauses=["a"])
    S.append_event({"event": "backlog_implement", "item_id": 612, "phase": "finished",
                    "round_id": "SM_i", "stop_reason": "stop", "num_turns": 5,
                    "outcome": {"landed": True, "acceptance": "met", "clause_outcomes": []}},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": "SM_i", "commit": "e" * 40}, path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": "e" * 40}, path=S.LEDGER_PATH)
    assert [d for d in B.close_settled_items(S.LEDGER_PATH) if d["item_id"] == 612][0]["closed"]
    assert _fm(path2)["status"] == "done"


def test_the_implement_prompt_renders_human_clauses_as_not_yours():
    block = I._human_clauses_block(["Alan audits ten items"])
    assert "not yours to do" in block and "Alan audits ten items" in block
    assert I._human_clauses_block([]) == ""
    assert "{human_clauses}" in I.PROMPT
    # `automod_gate_wait`, `automod_amend_clause` and the do-not-edit-while-
    # gating rule are procedure and live in the skill the prompt names
    # (cut 4); `test_prompt_pacing_and_ordering` pins them there.
    assert "automod-change-own-code" in I.PROMPT


UNSAT = {"premise": "sound", "summary": "", "clauses": [
    {"clause": 1, "verdict": "unsatisfiable", "evidence_path": "", "evidence_line": 0,
     "test_node_id": "", "how_verified": "read", "note": "a live-ledger measurement"}],
    "test_honesty": [], "seams_unverified": [], "amendments_ok": True, "amendments_note": ""}


def _unsat_refusal(head, attempt):
    return {**_refusal(head, attempt, "clause 1 unsatisfiable"),
            "clauses": [{"clause": 1, "verdict": "unsatisfiable"}]}


def test_a_refusal_of_the_contract_spends_no_attempt(monkeypatch, tmp_path):
    """#860: the second review was the first `unsatisfiable`, and the rung
    said abort. Amend-and-regate has to be a move the cap allows."""
    grade = _grader(UNSAT)
    _arm(monkeypatch, tmp_path, grade=grade, head="c" * 40,
         prior=[_refusal("a" * 40, 1, "clause 2 unmet"), _unsat_refusal("b" * 40, 2)])
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data.get("review_exhausted") is None and len(grade.calls) == 1
    assert data["review_attempt"] == 2, "one real refusal spent, the unsatisfiable one did not"
    assert "spends no attempt" in detail and "automod_amend_clause" in detail
    assert "1/2 spent" in detail


def test_the_hard_ceiling_ends_a_round_that_keeps_getting_free_reviews(monkeypatch, tmp_path):
    grade = _grader(UNSAT)
    prior = [_unsat_refusal(f"{i:040d}", i + 1) for i in range(RV.REVIEW_HARD_CAP)]
    _arm(monkeypatch, tmp_path, grade=grade, head="z" * 40, prior=prior)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data["review_exhausted"] and grade.calls == [] and "ceiling" in detail
    # ...and passes count toward that ceiling too, refusals or not
    passes = [{"event": "review", "round_id": "SM_REV", "ok": True, "blocking": False,
               "head": f"{i:040d}"} for i in range(RV.REVIEW_HARD_CAP)]
    _arm(monkeypatch, tmp_path, grade=_grader(UNSAT), head="y" * 40, prior=passes)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert data["review_exhausted"]


def test_a_downgraded_partial_is_not_a_disagreement(isolated):
    """The grader said `met` twice and Python downgraded it twice for missing
    evidence; that is agreement without receipts, not the author refusing."""
    write_item(isolated, 860, clauses=["a", "b"])
    S.append_event({"event": "backlog_implement", "item_id": 860, "phase": "finished",
                    "round_id": "SM_d", "stop_reason": "stop", "num_turns": 5}, path=S.LEDGER_PATH)
    def review(head, clauses):
        S.append_event({"event": "review", "round_id": "SM_d", "item_id": 860, "ok": True,
                        "blocking": True, "kind": "retry", "head": head, "clauses": clauses},
                       path=S.LEDGER_PATH)
    down = {"clause": 2, "verdict": "partial", "downgraded": ["evidence_path missing"]}
    review("a" * 40, [{"clause": 1, "verdict": "unmet"}, down])
    review("b" * 40, [{"clause": 1, "verdict": "met"}, down])
    assert B.review_disagreement(S.LEDGER_PATH, 860) is None
    review("c" * 40, [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "partial"}])
    review("d" * 40, [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "partial"}])
    assert B.review_disagreement(S.LEDGER_PATH, 860) == 2, "a judged partial twice still is"


def test_only_a_blocking_honesty_finding_refuses_the_round(tmp_path):
    """#866: ten clauses met twice, refused twice on lists carrying a positive
    observation, two vault-note remarks and one unpinned constant beside one
    real defect. Severity decides; a non-test file is advisory whatever the
    grader called it; an object without the field keeps the old reading."""
    met = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 1,
           "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    def parsed(honesty):
        return RV.parse_review({"premise": "sound", "summary": "fine", "clauses": [met],
                                "test_honesty": honesty, "seams_unverified": [],
                                "amendments_ok": True, "amendments_note": ""},
                               worktree=tmp_path, changed_tests=["tests/test_x.py"], n_clauses=1)
    advisory_only = parsed([
        {"file": "tests/test_x.py", "line": 5, "severity": "advisory", "problem": "tolerance is loose"},
        {"file": "knowledge/ai/note.md", "line": 107, "severity": "blocking", "problem": "claim is false"},
    ])
    assert [h["severity"] for h in advisory_only["test_honesty"]] == ["advisory", "advisory"]
    kind, findings = RV.decide(advisory_only, [])
    assert kind == "pass" and "advisory tests/test_x.py:5" in findings and "note.md:107" in findings
    blocking = parsed([{"file": "tests/test_x.py", "line": 9, "severity": "blocking",
                        "problem": "cannot fail"},
                       {"file": "tests/test_x.py", "line": 5, "severity": "advisory",
                        "problem": "tolerance"}])
    kind, findings = RV.decide(blocking, [])
    assert kind == "retry" and findings.startswith("test honesty tests/test_x.py:9")
    assert "advisory tests/test_x.py:5" in findings
    legacy = parsed([{"file": "tests/test_x.py", "line": 3, "problem": "or True"}])
    assert legacy["test_honesty"][0]["severity"] == "blocking"
    assert RV.decide(legacy, [])[0] == "retry"
    # the deterministic prechecks stay blocking
    kind, _ = RV.decide(parsed([]), [{"file": "tests/test_x.py", "line": 1, "problem": "`or True`"}])
    assert kind == "retry"
    assert "severity" in RV.REVIEW_SCHEMA["properties"]["test_honesty"]["items"]["required"]
    text = RV.build_prompt(contract={"id": 1, "title": "t", "body": "b", "clauses": ["c"], "path": ""},
                           diff="", diff_truncated=False, changed_tests=[], test_counts={},
                           worktree=tmp_path, run_tests=tmp_path / "rt")
    assert "Only \\\n`blocking` entries refuse the round" in text or "`blocking` entries refuse the round" in text
    assert "does not belong" in text


def test_evidence_paths_with_symbols_anchors_or_a_dead_absolute_prefix_still_resolve(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    n = RV.normalize_evidence_path
    assert n("app/x.py::helper", tmp_path) == "app/x.py"
    assert n("app/x.py:helper", tmp_path) == "app/x.py"
    assert n("app/x.py#L12", tmp_path) == "app/x.py"
    assert n("app/x.py:12-14,", tmp_path) == "app/x.py"
    assert n("[app/x.py]", tmp_path) == "app/x.py"
    # #870's clause 2, verbatim: two locations in one field, downgraded for the semicolon
    assert n("app/x.py:377; tests/test_y.py:534", tmp_path) == "app/x.py"
    assert n("/gone/checkout/review-abc/app/x.py", tmp_path) == "app/x.py"
    assert n("app/missing.py", tmp_path) == ""
    # ...and a met that still cannot resolve records what the grader wrote
    parsed = RV.parse_review({"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "met", "evidence_path": "nowhere/y.py", "evidence_line": 3,
         "test_node_id": "tests/test_x.py::t", "how_verified": "ran", "note": "n"}],
        "test_honesty": [], "seams_unverified": []},
        worktree=tmp_path, changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "partial"
    assert "grader wrote 'nowhere/y.py'" in parsed["clauses"][0]["downgraded"][0]


def test_only_a_seam_a_test_could_cross_before_landing_refuses(tmp_path):
    """#870: eight clauses met, every honesty entry advisory, refused on two
    seams the grader raised fresh on the last attempt — one of which it
    called a post-landing check itself. A seam only production can cross is
    recorded, never a refusal; a bare-string seam keeps the old reading."""
    met = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 1,
           "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    def parsed(seams):
        return RV.parse_review({"premise": "sound", "summary": "s", "clauses": [met],
                                "test_honesty": [], "seams_unverified": seams,
                                "amendments_ok": True, "amendments_note": ""},
                               worktree=tmp_path, changed_tests=["tests/test_x.py"], n_clauses=1)
    p = parsed([{"seam": "a real pool tick", "testable_before_landing": False},
                {"seam": "loopback POST to the test server", "testable_before_landing": True}])
    assert [s["testable_before_landing"] for s in p["seams_unverified"]] == [False, True]
    kind, findings = RV.decide(p, [])
    assert kind == "retry" and findings.startswith("seam unverified: loopback POST")
    assert "post-landing seam (not refusing): a real pool tick" in findings
    p = parsed([{"seam": "a real pool tick", "testable_before_landing": False}])
    kind, findings = RV.decide(p, [])
    assert kind == "pass" and "post-landing seam" in findings
    legacy = parsed(["_meta over MCP"])
    # A bare-string seam still reads as testable — and, since the grader
    # started judging its own findings, as actionable and not a repeat.
    assert legacy["seams_unverified"] == [{"seam": "_meta over MCP", "testable_before_landing": True,
                                           "actionable_in_round": True, "same_as_prior": False}]
    assert RV.decide(legacy, [])[0] == "retry"
    items = RV.REVIEW_SCHEMA["properties"]["seams_unverified"]["items"]
    assert "testable_before_landing" in items["required"]
    text = RV.build_prompt(contract={"id": 1, "title": "t", "body": "b", "clauses": ["c"], "path": ""},
                           diff="", diff_truncated=False, changed_tests=[], test_counts={},
                           worktree=tmp_path, run_tests=tmp_path / "rt")
    assert "Only a testable seam refuses the round" in text


def test_the_review_event_records_seams_as_text_and_names_the_untestable_ones(monkeypatch, tmp_path):
    obj = dict(UNMET, clauses=[], seams_unverified=[
        {"seam": "live pool tick", "testable_before_landing": False},
        {"seam": "loopback POST", "testable_before_landing": True}])
    contract = {"id": 7, "title": "t", "body": "b", "clauses": [], "path": "",
                "amendments": [], "human_clauses": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), head="", contract={**contract, "clauses": ["c"]})
    _Gate(7, ["app/x.py"], tmp_path).rung_review()
    ev = events[-1]
    assert ev["seams_unverified"] == ["live pool tick", "loopback POST"]
    assert ev["seams_untestable"] == ["live pool tick"]


def test_a_re_offer_with_every_clause_met_runs_before_a_fresh_confirmation(isolated):
    """Five re-offers at all-clauses-met sat behind fresh umbrellas on
    2026-09-11, each of which took an hour and aborted at attempt two."""
    def confirm(iid, created_days_ago):
        created = (datetime.now(timezone.utc) - timedelta(days=created_days_ago)).isoformat()
        path = isolated / f"{iid}-x.md"
        path.write_text(f"---\nstatus: up_next\npriority: medium\ncreated: '{created}'\n"
                        f"board: lloyd\ntags: []\nacceptance_clauses:\n- a\n---\n\n# x\n\nbody\n")
        S.append_event({"event": "backlog_triage", "item_id": iid, "verdict": "confirmed",
                        "acceptance": "the check passes", "acceptance_clauses": ["a"]},
                       path=S.LEDGER_PATH)
    confirm(901, 3)   # fresh, oldest
    confirm(902, 1)   # re-offered, refused on a seam with every clause met
    confirm(903, 2)   # re-offered, refused with a clause unmet
    for iid, rid, verdict in ((902, "SM_902", "met"), (903, "SM_903", "unmet")):
        S.append_event({"event": "backlog_implement", "item_id": iid, "phase": "finished",
                        "round_id": rid, "stop_reason": "stop", "num_turns": 9}, path=S.LEDGER_PATH)
        S.append_event({"event": "review", "round_id": rid, "item_id": iid, "ok": True,
                        "blocking": True, "kind": "retry", "head": "h" * 40,
                        "clauses": [{"clause": 1, "verdict": verdict}]}, path=S.LEDGER_PATH)
        S.append_event({"event": "gate", "round_id": rid, "rung": "review", "ok": False,
                        "detail": "sent back", "review_retry": True,
                        "review_findings": "seam" if verdict == "met" else "clause 1 unmet"},
                       path=S.LEDGER_PATH)
    assert B.last_review_all_met(S.LEDGER_PATH) == {902}
    order = []
    remaining = {901, 902, 903}
    while remaining:
        pick = B.select_confirmed(S.LEDGER_PATH)
        assert pick is not None
        order.append(pick[0].id)
        remaining.discard(pick[0].id)
        (isolated / f"{pick[0].id}-x.md").unlink()
    assert order == [902, 901, 903], "nearest to landing, then fresh, then other re-offers"
