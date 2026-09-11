"""Calibrating the grader without a network: the fixture shape, the compare
rule, the stripped-tests variant, and that a backfill writes to its own file
and never to the ledger."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.automod import backlog as B, review as RV, review_tools as RT, state as S


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A repo with a landed commit, a backlog item with clauses, and a ledger
    that says the round settled."""
    repo = tmp_path / "repo"; (repo / "app").mkdir(parents=True); (repo / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    git(repo, "config", "user.email", "t@e"); git(repo, "config", "user.name", "t")
    (repo / "app" / "m.py").write_text("V = 1\n"); git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "base")
    parent = git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "app" / "m.py").write_text("V = 2\n")
    (repo / "tests" / "test_m.py").write_text("def test_v():\n    assert True or True\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "round")
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()

    backlog = tmp_path / "backlog"; backlog.mkdir()
    (backlog / "9-thing.md").write_text(
        "---\nstatus: in_progress\nboard: lloyd\nacceptance_clauses:\n- V is 2\n- a test pins it\n---\n# Thing\n\nbody\n")
    monkeypatch.setattr(B, "BACKLOG_DIR", backlog)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    for e in ({"event": "backlog_implement", "item_id": 9, "phase": "finished", "round_id": "SM_9",
               "outcome": {"acceptance": "met", "clause_outcomes": [
                   {"clause": 1, "outcome": "met"}, {"clause": 2, "outcome": "met"}]}},
              {"event": "promoted", "round_id": "SM_9", "commit": commit, "parent": parent,
               "changed_paths": ["app/m.py", "tests/test_m.py"]},
              {"event": "settled", "commit": commit}):
        S.append_event(e, path=ledger)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return {"repo": repo, "parent": parent, "commit": commit, "ledger": ledger, "tmp": tmp_path}


def _stub(structured):
    def grade(**kw):
        # Observed at call time: the detached worktree is released before
        # grade_commit returns, so existence can only be checked from inside.
        kw["_tree_present"] = (Path(kw["worktree"]) / "app" / "m.py").exists()
        grade.calls.append(kw)
        return {"ok": True, "error": "", "session_id": "sess_cal", "structured": structured}
    grade.calls = []
    return grade


def _met(path="app/m.py", node="tests/test_m.py::test_v"):
    return {"premise": "sound", "summary": "fine", "test_honesty": [], "seams_unverified": [],
            "clauses": [{"clause": i, "verdict": "met", "evidence_path": path, "evidence_line": 1,
                         "test_node_id": node, "how_verified": "ran", "note": ""} for i in (1, 2)]}


def test_landed_rounds_joins_promotion_settle_and_item(world):
    rows = RT.landed_rounds(world["ledger"])
    assert len(rows) == 1 and rows[0]["item_id"] == 9 and rows[0]["commit"] == world["commit"]
    assert rows[0]["author_outcome"]["acceptance"] == "met"


def test_grade_commit_checks_out_the_commit_and_runs_the_prechecks(world):
    grade = _stub(_met())
    out = RT.grade_commit(repo=world["repo"], item_id=9, parent=world["parent"], commit=world["commit"],
                          changed_paths=["app/m.py", "tests/test_m.py"], label="t1", grader=grade)
    kw = grade.calls[0]
    assert Path(kw["worktree"]).name == "lloyd" and kw["_tree_present"]
    assert kw["base"] == world["parent"] and kw["contract"]["clauses"] == ["V is 2", "a test pins it"]
    assert kw["child_env"]["LLOYD_VOICE_ALERTS"] == "0" and "gate-state" in kw["child_env"]["LLOYD_AUTOMOD_STATE"]
    # The stubbed grader said met everywhere, but the deterministic precheck
    # saw `or True` in the added test — and that is a retry regardless.
    assert out["kind"] == "retry" and any("or True" in p["problem"] for p in out["prechecks"])
    assert not Path(kw["worktree"]).exists(), "the detached worktree is released"


def test_strip_tests_turns_a_landing_into_a_known_bad_case(world):
    grade = _stub(_met())
    out = RT.grade_commit(repo=world["repo"], item_id=9, parent=world["parent"], commit=world["commit"],
                          changed_paths=["app/m.py", "tests/test_m.py"], label="t2", grader=grade,
                          strip_tests=True)
    kw = grade.calls[0]
    assert kw["changed_paths"] == ["app/m.py"], "the test file is gone from the diff under review"
    # Every `met` is downgraded: its test_node_id is not in a changed test file.
    assert out["kind"] == "retry" and all(c["verdict"] == "partial" for c in out["clauses"])


def test_fixture_compare_and_calibrate(world, tmp_path):
    fx = tmp_path / "fixtures"
    p = RT.write_fixture(name="nine", round_id="SM_9", expect_kind="retry", expect_unmet=[2],
                         expect_honesty=True, ledger=world["ledger"], fixture_dir=fx, note="n")
    case = json.loads(p.read_text())
    assert case["commit"] == world["commit"] and case["expect"] == {
        "kind": "retry", "unmet_clauses": [2], "honesty_finding": True}
    with pytest.raises(ValueError):
        RT.write_fixture(name="x", round_id="SM_NOPE", expect_kind="pass", ledger=world["ledger"], fixture_dir=fx)

    # A grader that says everything is met disagrees with the case on the clause,
    # even though the precheck supplies the honesty finding and the retry.
    rows = RT.calibrate(repo=world["repo"], fixture_dir=fx, grader=_stub(_met()))
    assert rows[0]["ok"] is False and "clauses [2]" in rows[0]["why"]
    # One that flags clause 2 agrees.
    obj = _met(); obj["clauses"][1]["verdict"] = "unmet"
    rows = RT.calibrate(repo=world["repo"], fixture_dir=fx, grader=_stub(obj))
    assert rows[0]["ok"] is True and rows[0]["why"] == "agrees"
    ok, why = RT.compare(case, {"error": "HTTP 503"})
    assert ok is False and "did not answer" in why


def test_backfill_writes_its_own_file_and_never_the_ledger(world):
    before = world["ledger"].read_text()
    out_path = world["tmp"] / "state" / "review_backfill.jsonl"
    rows = RT.backfill(repo=world["repo"], ledger=world["ledger"], grader=_stub(_met()), out=out_path)
    assert len(rows) == 1 and rows[0]["event"] == "review_backfill"
    assert rows[0]["author_met"] == 2 and rows[0]["grader_met"] == 2 and rows[0]["kind"] == "retry"
    assert out_path.exists() and json.loads(out_path.read_text().splitlines()[-1])["item_id"] == 9
    assert world["ledger"].read_text() == before, "a backfill is a measurement, not a verdict"
    # Idempotent: an already-graded round is skipped on the next run.
    assert RT.backfill(repo=world["repo"], ledger=world["ledger"], grader=_stub(_met()), out=out_path) == []


def test_the_shipped_fixture_for_544_expects_the_human_review():
    case = json.loads((RT.FIXTURE_DIR / "544-first-cut.json").read_text())
    assert case["commit"].startswith("0f019f90") and case["expect"]["kind"] == "retry"
    assert 4 in case["expect"]["unmet_clauses"] and case["expect"]["honesty_finding"] is True
    stripped = json.loads((RT.FIXTURE_DIR / "544-stripped-tests.json").read_text())
    assert stripped["strip_tests"] is True and stripped["expect"]["kind"] == "retry"
    names = {c["name"] for c in RT.load_fixtures()}
    assert {"544-first-cut", "544-stripped-tests"} <= names
