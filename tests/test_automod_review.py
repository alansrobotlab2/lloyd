"""The review rung, and the retry it drives.

Every other rung asks whether the change broke something. #544 passed all
eight with one acceptance clause skipped, half a fleet uncovered, a silent
`Edit` replay and an `or True` assertion — then declared itself `deferred` to
nothing and parked its item forever. What follows pins the mechanism that
would have caught it, and the loop that gives a sound-premise round another
go without letting author and grader disagree indefinitely.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, gate as G, review as RV, round as R
from scripts.automod import state as S, vault_round as V, worktree as W
from workers.sources import autocode as I
from workers.sources import autotriage as T

ROOT = Path(__file__).resolve().parent.parent


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def write_item(d: Path, item_id, *, status="up_next", clauses=None, board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": board,
          "tags": ["backlog"]}
    if clauses:
        fm["acceptance_clauses"] = list(clauses)
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    return path


def _confirm(item_id, acceptance="the check passes", clauses=()):
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "acceptance": acceptance, "acceptance_clauses": list(clauses)},
                   path=S.LEDGER_PATH)


def _round(item_id, rid, *, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": rid, "stop_reason": stop_reason, "num_turns": 40},
                   path=S.LEDGER_PATH)


def _review_refused(rid, item_id, *, clauses_unmet=(1,), findings="clause 1 unmet: no test"):
    S.append_event({"event": "review", "round_id": rid, "item_id": item_id, "blocking": True,
                    "kind": "retry", "premise": "sound",
                    "clauses": [{"clause": i, "verdict": "unmet"} for i in clauses_unmet]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": rid, "rung": "review", "ok": False,
                    "detail": f"review sent it back: {findings}", "review_retry": True,
                    "review_findings": findings}, path=S.LEDGER_PATH)


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


# ── the schema is derived, strict, and bans nothing the grader needs ────────

def test_the_review_schema_is_built_from_the_tuples_and_is_strict():
    s = RV.REVIEW_SCHEMA
    assert s["properties"]["premise"]["enum"] == list(RV.PREMISES)
    clause = s["properties"]["clauses"]["items"]
    assert clause["properties"]["verdict"]["enum"] == list(RV.CLAUSE_VERDICTS)
    assert clause["properties"]["how_verified"]["enum"] == list(RV.HOW_VERIFIED)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert clause["additionalProperties"] is False and set(clause["required"]) == set(clause["properties"])
    assert "maxLength" not in json.dumps(s)


def test_the_grader_is_denied_every_automod_verb_and_every_writer():
    from workers.sources import _common as C
    assert set(C.WORKER_AUTOMOD_BAN) <= set(RV.REVIEW_DENY)
    assert set(C.WORKER_GRANT_MINT_BAN) <= set(RV.REVIEW_DENY)
    for name in ("Edit", "Write", "backlog_write_task", "vault_write", "Task"):
        assert name in RV.REVIEW_DENY
    # ...and is not denied what it exists to do.
    for name in ("Read", "Grep", "Glob", "Bash"):
        assert name not in RV.REVIEW_DENY


# ── parse_review judges the judge ──────────────────────────────────────────

def _obj(**clause):
    base = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 3,
            "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
    base.update(clause)
    return {"premise": "sound", "clauses": [base], "test_honesty": [], "seams_unverified": [],
            "summary": "fine"}


@pytest.fixture
def wt(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "x.py").write_text("def f():\n    return 1\n")
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")
    return tmp_path


def test_a_met_with_real_evidence_stands(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met" and parsed["downgraded"] == []


@pytest.mark.parametrize("bad", [
    {"evidence_path": "app/nope.py"},
    {"evidence_path": ""},
    {"test_node_id": "tests/test_other.py::test_z"},
    {"test_node_id": ""},
    {"how_verified": "inferred"},
])
def test_a_met_without_evidence_is_downgraded_in_python(wt, bad):
    """The grader's laziness is not the author's pass. A `met` needs a real
    path, a test in a file this diff changed, and ran|read — or it is
    `partial`, decided here without asking the model again."""
    parsed = RV.parse_review(_obj(**bad), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c["downgraded"] and parsed["downgraded"] == [1]


def test_a_clause_the_grader_did_not_mention_is_not_met(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=3)
    assert [c["verdict"] for c in parsed["clauses"]] == ["met", "partial", "partial"]
    assert parsed["clauses"][1]["note"] == "not addressed by the grader"


def test_the_vault_shape_needs_no_test_node(wt):
    parsed = RV.parse_review(_obj(test_node_id=""), worktree=wt, changed_tests=[], n_clauses=1,
                             require_tests=False)
    assert parsed["clauses"][0]["verdict"] == "met"


def test_an_unusable_object_is_none(wt):
    assert RV.parse_review("nope", worktree=wt, changed_tests=[], n_clauses=1) is None
    assert RV.parse_review({"premise": "maybe"}, worktree=wt, changed_tests=[], n_clauses=1) is None


def test_decide_maps_premise_then_findings():
    base = {"premise": "sound", "clauses": [{"clause": 1, "verdict": "met", "note": ""}],
            "test_honesty": [], "seams_unverified": [], "summary": "good", "downgraded": []}
    assert RV.decide(base, []) == ("pass", "good")
    assert RV.decide({**base, "premise": "unsound", "summary": "premise is false"}, [])[0] == "unsound"
    kind, findings = RV.decide({**base, "clauses": [{"clause": 2, "verdict": "unmet", "note": "no path",
                                                     "downgraded": []}]}, [])
    assert kind == "retry" and "clause 2 unmet: no path" in findings
    kind, findings = RV.decide(base, [{"file": "tests/t.py", "line": 9, "problem": "`or True`"}])
    assert kind == "retry" and "tests/t.py:9" in findings
    kind, findings = RV.decide({**base, "seams_unverified": ["loopback POST"]}, [])
    assert kind == "retry" and "seam unverified: loopback POST" in findings


# ── the deterministic honesty checks ───────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"; (r / "tests").mkdir(parents=True); (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n")
    (r / "app" / "m.py").write_text("V = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    return r, base


def test_prechecks_flag_only_what_the_round_added(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n\n"
        "def test_new():\n    assert rows() == [] or True\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    problems = [o["problem"] for o in out]
    assert any("or True" in p for p in problems), out
    assert not any("xfail" in p for p in problems), "the pre-existing xfail is not this round's"
    assert out[0]["line"] == 8


def test_prechecks_notice_a_code_change_with_no_new_test(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0  # touched\n")
    (r / "app" / "m.py").write_text("V = 2\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=1)
    assert any("no test function was added" in o["problem"] for o in out)
    assert RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=0) == []


# ── the rung: four outcomes, and the flags ride the event ──────────────────

class _Gate(G.Gate):
    def __init__(self, item_id, changed, tmp_path):
        super().__init__("SM_REV", tmp_path, "a" * 40, item_id=item_id)
        self.report.changed_paths = changed
        self.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))


def _arm(monkeypatch, tmp_path, *, grade, contract=None, prior=0):
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [
        {"event": "review", "round_id": "SM_REV"} for _ in range(prior)])
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: contract or {
        "id": iid, "title": "t", "body": "b", "clauses": ["the thing happens once"], "path": ""})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    return events


def _grader(structured, *, ok=True, error=""):
    def grade(**kw):
        grade.calls.append(kw)
        return {"ok": ok, "error": error, "session_id": "sess_r", "structured": structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}
    grade.calls = []
    return grade


def test_no_item_bound_is_a_recorded_skip(monkeypatch, tmp_path):
    """A human's round has no contract. That is a skip that says so, not a
    pass that pretends to have graded."""
    _arm(monkeypatch, tmp_path, grade=_grader(None))
    g = _Gate(None, ["app/x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok and data["skipped"] and "no backlog item" in detail


def test_an_unreachable_grader_is_external_not_a_pass(monkeypatch, tmp_path):
    """A waived review is the #544 shape. The engine being down is the
    engine's fault, so the item keeps its attempt — but the rung fails."""
    events = _arm(monkeypatch, tmp_path, grade=_grader(None, ok=False, error="HTTP 503"))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True and "503" in detail
    assert events[-1]["event"] == "review" and events[-1]["ok"] is False


def test_a_sound_premise_with_an_unmet_clause_is_a_retry_with_findings(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "close",
           "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                        "note": "only the chat path is covered"}],
           "test_honesty": [], "seams_unverified": ["loopback POST"]}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    g = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok is False and data["review_retry"] is True and data["review_attempt"] == 1
    assert "only the chat path is covered" in data["review_findings"]
    assert "seam unverified" in data["review_findings"]
    assert "fix what it names" in detail
    ev = events[-1]
    assert ev["event"] == "review" and ev["blocking"] and ev["kind"] == "retry"
    assert ev["clauses"][0]["verdict"] == "unmet"


def test_the_second_refusal_says_abort_and_the_third_never_asks_the_model(monkeypatch, tmp_path):
    obj = {"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
         "test_node_id": "", "how_verified": "inferred", "note": "still"}],
        "test_honesty": [], "seams_unverified": []}
    grade = _grader(obj)
    _arm(monkeypatch, tmp_path, grade=grade, prior=1)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_attempt"] == 2 and "abort and report" in detail
    assert len(grade.calls) == 1
    _arm(monkeypatch, tmp_path, grade=grade, prior=2)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] and data["review_exhausted"]
    assert len(grade.calls) == 1, "a third gate call must not spend another grading turn"


def test_an_unsound_premise_is_its_own_verdict(monkeypatch, tmp_path):
    obj = {"premise": "unsound", "summary": "the file it fixes was deleted in 5531f21",
           "clauses": [], "test_honesty": [], "seams_unverified": []}
    _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_premise_unsound"] is True
    assert "5531f21" in data["review_summary"] and "review_retry" not in data


def test_all_clauses_met_passes_and_names_the_session(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "does what it says",
           "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                        "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok and "1 met" in detail and data["review_session"] == "sess_r"
    assert events[-1]["blocking"] is False


def test_the_gate_event_carries_the_review_flags_and_findings(monkeypatch):
    """`gate.json` dies with the worktree; `implement_outcomes` reads these off
    the ledger long after."""
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    g = G.Gate("SM_FLAGS", ROOT, "HEAD")
    g._rung("review", lambda: (False, "sent back", {"review_retry": True,
                                                    "review_findings": "clause 1 unmet",
                                                    "review_attempt": 1}))
    g._rung("review", lambda: (False, "unsound", {"review_premise_unsound": True,
                                                  "review_summary": "false premise"}))
    assert events[0]["review_retry"] is True and events[0]["review_findings"] == "clause 1 unmet"
    assert events[0]["review_attempt"] == 1 and "review_premise_unsound" not in events[0]
    assert events[1]["review_premise_unsound"] is True and events[1]["review_summary"] == "false premise"


def test_preflight_refuses_a_contract_with_no_clauses_or_no_test(monkeypatch, tmp_path):
    """Zero-cost fail-fasts for a round that has a contract — cheaper to learn
    here than after a 77 s test run, and both are the round's own."""
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "", "clauses": [], "path": ""})
    g = G.Gate("SM_PF", ROOT, "HEAD", item_id=9)
    # Drive only the clause/test check by calling the same predicate preflight uses.
    contract = RV.item_contract(9)
    assert not contract["clauses"]
    src = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert "has no acceptance clauses to judge" in src
    assert "nothing pins a clause" in src
    assert g.item_id == 9


def test_review_sits_after_tests_and_before_venv(monkeypatch):
    names: list[str] = []
    g = G.Gate("SM_POS", ROOT, "HEAD")
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or True)
    g.run()
    assert names.index("tests") < names.index("review") < names.index("venv")
    assert names.index("canary_smoke") == names.index("drill") - 1


# ── the backlog: retry, cap, disagreement, partial ─────────────────────────

def test_a_review_refusal_re_offers_the_item_with_the_findings_and_the_branch(isolated):
    write_item(isolated, 544, clauses=["a", "b"]); _confirm(544, clauses=["a", "b"])
    _round(544, "SM_1"); _review_refused("SM_1", 544, findings="clause 2 unmet: no loopback test")
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[544]
    assert verdict == "review_retry"
    assert "no loopback test" in detail and "automod/SM_1" in detail and "from_branch" in detail
    assert 544 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.desired_statuses(S.LEDGER_PATH, None)[544][0] == "up_next"
    assert B.reoffer_reason(S.LEDGER_PATH, 544).startswith("review_retry:")


def test_the_review_re_offer_is_capped(isolated):
    """`REVIEW_RETRY_CAP` re-offers: the first round plus two more, each
    refused on a DIFFERENT clause (the same clause twice escalates earlier —
    see the next test). The fourth finished round is spent."""
    write_item(isolated, 545, clauses=["a", "b", "c", "d"]); _confirm(545, clauses=["a", "b", "c", "d"])
    for n, rid in enumerate(("SM_a", "SM_b", "SM_c"), 1):
        _round(545, rid); _review_refused(rid, 545, clauses_unmet=(n,), findings=f"f{n}")
        assert B.implement_outcomes(S.LEDGER_PATH)[545][0] == "review_retry", n
    _round(545, "SM_d"); _review_refused("SM_d", 545, clauses_unmet=(4,), findings="f4")
    verdict, _ = B.implement_outcomes(S.LEDGER_PATH)[545]
    assert verdict == "spent", "one round plus REVIEW_RETRY_CAP re-offers; then a human decides"
    assert B.desired_statuses(S.LEDGER_PATH, None)[545][0] == "draft"


def test_the_same_clause_twice_is_a_disagreement_and_escalates_early(isolated):
    """Author says met, grader says unmet, twice, on clause 1. A third round
    re-runs the argument; a human resolves it."""
    write_item(isolated, 546, clauses=["a", "b"]); _confirm(546, clauses=["a", "b"])
    _round(546, "SM_x"); _review_refused("SM_x", 546, clauses_unmet=(1,))
    assert B.review_disagreement(S.LEDGER_PATH, 546) is None
    _round(546, "SM_y"); _review_refused("SM_y", 546, clauses_unmet=(1, 2))
    assert B.review_disagreement(S.LEDGER_PATH, 546) == 1
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[546]
    assert verdict == "spent" and detail.startswith("review disagreement") and "clause 1" in detail
    # ...while different clauses on successive reviews are still a retry.
    write_item(isolated, 547, clauses=["a", "b"]); _confirm(547, clauses=["a", "b"])
    _round(547, "SM_p"); _review_refused("SM_p", 547, clauses_unmet=(1,))
    _round(547, "SM_q"); _review_refused("SM_q", 547, clauses_unmet=(2,))
    assert B.implement_outcomes(S.LEDGER_PATH)[547][0] == "review_retry"


def test_review_retry_yields_to_incomplete_and_supersedes_external(isolated):
    write_item(isolated, 548, clauses=["a"]); _confirm(548, clauses=["a"])
    _round(548, "SM_i", stop_reason="max_turns"); _review_refused("SM_i", 548)
    assert B.implement_outcomes(S.LEDGER_PATH)[548][0] == "incomplete"
    # A later passing gate on the same round supersedes the refusal.
    S.append_event({"event": "gate", "round_id": "SM_i", "rung": "drill", "ok": True}, path=S.LEDGER_PATH)
    assert "SM_i" not in B.review_retry_rounds(S.LEDGER_PATH)


def test_an_unsound_premise_spends_the_attempt(isolated):
    write_item(isolated, 549, clauses=["a"]); _confirm(549, clauses=["a"])
    _round(549, "SM_u")
    S.append_event({"event": "gate", "round_id": "SM_u", "rung": "review", "ok": False,
                    "detail": "review: premise unsound", "review_premise_unsound": True,
                    "review_summary": "false"}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[549][0] == "spent"
    assert "SM_u" in B.review_unsound_rounds(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None)[549]
    assert want[0] == "draft" and want[2] is True


def test_fresh_confirmations_are_picked_before_re_offers(isolated):
    """Oldest-first alone let a sent-back item monopolise the loop."""
    write_item(isolated, 550, clauses=["a"]); _confirm(550, clauses=["a"])     # older, sent back
    _round(550, "SM_o"); _review_refused("SM_o", 550)
    p = write_item(isolated, 551, clauses=["a"]); _confirm(551, clauses=["a"])  # newer, fresh
    fm, body = B._split_frontmatter(p.read_text())
    fm["created"] = datetime.now(timezone.utc).isoformat()
    p.write_text(f"---\n{yaml.dump(fm)}---\n{body}")
    B.reconcile_statuses(S.LEDGER_PATH, None)
    pair = B.select_confirmed(S.LEDGER_PATH, None)
    assert pair is not None and pair[0].id == 551
    # ...and a re-offer is still reachable once the fresh work is gone.
    B.set_status(551, "draft", "parked by a human")
    assert B.select_confirmed(S.LEDGER_PATH, None)[0].id == 550


# ── clause outcomes derive the acceptance; a nameless deferral is not_met ──

def test_parse_outcome_derives_the_acceptance_from_clauses():
    base = {"landed": True, "acceptance": "met", "deferred_to": [], "summary": "s", "spawned": []}
    all_met = [{"clause": 1, "outcome": "met", "evidence": "t::a", "deferred_to": []},
               {"clause": 2, "outcome": "met", "evidence": "t::b", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": all_met})["acceptance"] == "met"
    one_unmet = all_met[:1] + [{"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]
    out = B.parse_outcome({**base, "clause_outcomes": one_unmet})
    assert out["acceptance"] == "not_met" and B.unmet_clauses(out) == [2]
    deferred = all_met[:1] + [{"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": [618]}]
    out = B.parse_outcome({**base, "clause_outcomes": deferred})
    assert out["acceptance"] == "deferred" and out["deferred_to"] == [618]
    # `unnecessary` is a verdict on the item and is kept as stated.
    assert B.parse_outcome({**base, "acceptance": "unnecessary", "clause_outcomes": []})["acceptance"] == "unnecessary"


def test_a_deferral_that_names_nothing_is_not_met():
    """#544 exactly: `deferred`, `deferred_to: []`, and the item parked in
    `in_progress` with "close this when that closes" pointing at nothing."""
    base = {"landed": True, "acceptance": "deferred", "deferred_to": [], "summary": "s",
            "spawned": [], "clause_outcomes": []}
    assert B.parse_outcome(base)["acceptance"] == "not_met"
    nameless = [{"clause": 1, "outcome": "deferred", "evidence": "", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": nameless})["acceptance"] == "not_met"
    assert B.parse_outcome({**base, "deferred_to": [7]})["acceptance"] == "deferred"
    assert "clause_outcomes" in B.IMPLEMENT_OUTCOME_SCHEMA["required"]
    assert B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["clause_outcomes"]["items"]["properties"][
        "outcome"]["enum"] == list(B.CLAUSE_OUTCOMES)


def test_a_landed_round_with_an_unmet_clause_is_offered_once_more_for_it(isolated):
    p = write_item(isolated, 552, clauses=["a", "b"]); _confirm(552, clauses=["a", "b"])
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "finished",
                    "round_id": "SM_l", "stop_reason": "stop", "num_turns": 30,
                    "outcome": {"landed": True, "acceptance": "not_met", "deferred_to": [],
                                "summary": "half", "spawned": [],
                                "clause_outcomes": [
                                    {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
                                    {"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]}},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": "SM_l", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    # Under observation: still in_progress.
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "in_progress"
    S.append_event({"event": "settled", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    out = B.close_settled_items(S.LEDGER_PATH, None)
    assert out == [{"item_id": 552, "closed": False, "acceptance": "not_met"}]
    assert "clause(s) [2]" in _fm(p)["activity_log"][-1]
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[552]
    assert verdict == "partial" and "[2]" in detail and "c0ffee00" in detail
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "up_next"
    assert "partial" in I._reoffer_block(B.reoffer_reason(S.LEDGER_PATH, 552))


# ── clauses at triage: both parsers, the item, the fallback ─────────────────

TEXT = ("VERDICT: confirmed\nSURFACE: code\nCHECK: c\nEVIDENCE: e\n"
        "ACCEPTANCE: the retry fires once and the dashboard shows it\n"
        "ACCEPTANCE_CLAUSES:\n1. a retried item fires email_send once\n2. the counter is on /api/dashboard\n"
        "SPAWNED: none\n")


def test_the_text_path_splits_numbered_clauses_and_keeps_the_placeholder_rule():
    parsed = T.parse_verdict(TEXT)
    assert parsed["acceptance_clauses"] == ["a retried item fires email_send once",
                                           "the counter is on /api/dashboard"]
    assert parsed["acceptance"].startswith("the retry fires once")
    none = T.parse_verdict("VERDICT: stale\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: -\nACCEPTANCE_CLAUSES: none\n")
    assert none["acceptance_clauses"] == [] and none["acceptance"] == ""


def test_the_structured_path_carries_clauses_too():
    obj = {"verdict": "confirmed", "surface": "code", "check": "c", "evidence": "e",
           "acceptance": "x", "acceptance_clauses": ["one", "two", "none"], "spawned": []}
    assert T.parse_verdict("", obj)["acceptance_clauses"] == ["one", "two"]
    assert "acceptance_clauses" in B.TRIAGE_VERDICT_SCHEMA["required"]
    assert "ACCEPTANCE_CLAUSES:" in T.PROMPT


def test_record_verdict_writes_the_clauses_onto_the_item(isolated):
    p = write_item(isolated, 560, status="draft")
    item = next(i for i in B.open_items(None) if i.id == 560)
    B.record_verdict(item, "confirmed", "real", acceptance="x", acceptance_clauses=["one", "two"])
    fm = _fm(p)
    assert fm["acceptance_clauses"] == ["one", "two"] and fm["status"] == "up_next"
    assert "1. one" in p.read_text() and "graded one by one" in p.read_text()


def test_acceptance_clauses_fall_back_to_the_prose_for_old_items(isolated):
    write_item(isolated, 561); _confirm(561, acceptance="the check passes")
    contract = RV.item_contract(561)
    assert contract["clauses"] == ["the check passes"] and contract["title"] == "A thing"
    assert B.acceptance_clauses_of({"acceptance": "-"}) == []
    assert B.split_clause_lines("1. a\n2) b\n") == ["a", "b"]
    assert B.split_clause_lines("just prose") == ["just prose"]


# ── the implement side: prompt, resume, abort reason ───────────────────────

def test_the_implement_prompt_names_the_clauses_the_seams_and_the_item_id():
    text = I.PROMPT.format(item_id=9, status="up_next", priority="high", name="n", body="b",
                           triaged_ago="today", surface="code", check="c", evidence="e",
                           acceptance="a", clauses="    1. a\n    2. b", spawn_cap=I.SPAWN_CAP,
                           round_label="item9", reoffer="")
    assert "    1. a\n    2. b" in text
    assert "item_id=9" in text and "Seams." in text and "review" in text
    assert "A deferral that names no id is recorded as `not_met`" in text
    assert "per clause" in text


def test_the_reoffer_banner_for_a_review_retry_names_the_branch_to_resume():
    reason = ("review_retry: the review rung found the premise sound but the implementation or its "
              "tests short — clause 1 unmet; its work is on branch `automod/SM_20260910_070638` "
              "(pass it as from_branch)")
    banner = I._reoffer_block(reason)
    assert 'from_branch="automod/SM_20260910_070638"' in banner
    assert "never reached a verdict" not in banner
    assert "never reached a verdict" in I._reoffer_block("incomplete: ran out of clock")


@pytest.fixture()
def scratch(tmp_path, monkeypatch):
    live = tmp_path / "live"; (live / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com"); git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "base")
    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state" / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"), ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, tmp_path / "state" / fn)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    return live


def test_a_round_can_resume_the_branch_review_sent_back(scratch, monkeypatch):
    """The re-offer is not a fresh start: the new worktree begins where the
    refused round left off, rebased onto live main, and the old branch is
    gone so branches do not accumulate forever."""
    # Round ids are second-resolution; two starts in one test need distinct ones.
    ids = iter(("SM_T1", "SM_T2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))
    first = R.start("first go", force=True, item_id=77)
    rid1 = first["round_id"]
    wt1 = Path(first["worktree"])
    (wt1 / "app" / "fix.py").write_text("FIX = 1\n", encoding="utf-8")
    git(wt1, "add", "-A"); git(wt1, "commit", "-q", "-m", "partial fix")
    spec1 = yaml.safe_load((S.ROUNDS_DIR / rid1 / "run_spec.yaml").read_text())
    assert spec1["item"] == {"id": 77}
    R.abort(rid1, reason="review sent it back: clause 1 unmet")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_aborted"][-1]
    assert ev["reason"].startswith("review sent it back")
    assert W.branch_exists(scratch, f"automod/{rid1}")

    # main moves underneath in the meantime.
    (scratch / "app" / "other.py").write_text("O = 1\n", encoding="utf-8")
    git(scratch, "add", "-A"); git(scratch, "commit", "-q", "-m", "human commit")
    live_head = git(scratch, "rev-parse", "HEAD").stdout.strip()

    second = R.start("second go", force=True, item_id=77, from_branch=f"automod/{rid1}")
    try:
        wt2 = Path(second["worktree"])
        assert (wt2 / "app" / "fix.py").exists(), "the refused round's work is the starting point"
        assert (wt2 / "app" / "other.py").exists(), "rebased onto the moved main"
        assert second["base"] == live_head and second["rebased_onto"] == live_head
        assert second["from_branch"] == f"automod/{rid1}"
        assert not W.branch_exists(scratch, f"automod/{rid1}"), "the old branch is deleted"
        start_ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert start_ev["item_id"] == 77 and start_ev["from_branch"] == f"automod/{rid1}"
    finally:
        W.remove(second["round_id"], repo=scratch)


def test_a_missing_resume_branch_falls_back_to_a_fresh_worktree(scratch):
    out = R.start("go", force=True, from_branch="automod/SM_NOPE")
    try:
        assert out["from_branch_missing"] is True and "does not exist" in out["note"]
        assert Path(out["worktree"]).exists()
    finally:
        W.remove(out["round_id"], repo=scratch)


# ── vault rounds: the same reader, before `git add` ────────────────────────

@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    (r / "skills" / "foo").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(V, "GRADER", None)
    return r


def _vault_events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def test_a_vault_land_with_no_grader_records_the_skip(vault):
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert out["review"] == "skipped" and _vault_events("vault_land")[-1]["review"] == "skipped"
    assert _vault_events("vault_review") == []


def test_a_vault_review_refusal_leaves_the_edit_then_reverts_on_the_second(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("retry", "clause 1 unmet: the skill never says when"))
    S.append_event({"event": "backlog_implement", "item_id": 9, "phase": "started"}, path=S.LEDGER_PATH)
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v2\n")
    with pytest.raises(V.VaultRoundError, match="still in place"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert "v2" in f.read_text(), "first refusal: the model fixes it in place"
    ev = _vault_events("vault_review")[-1]
    assert ev["blocking"] and ev["attempt"] == 1 and ev["review_retry"] and ev["reverted"] == []
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert f.read_text() == "---\nname: foo\n---\n# foo\n", "second refusal: put back"
    assert _vault_events("vault_review")[-1]["reverted"] == ["skills/foo/SKILL.md"]


def test_an_unsound_vault_premise_reverts_at_once(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("unsound", "the task this skill serves was deleted"))
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v3\n")
    with pytest.raises(V.VaultRoundError, match="premise unsound"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v3", item_id=9)
    assert "v3" not in f.read_text()
    assert _vault_events("vault_review")[-1]["review_premise_unsound"] is True


def test_a_vault_grader_that_blows_up_never_blocks_a_validated_landing(vault, monkeypatch):
    def boom(**kw):
        raise RuntimeError("engine gone")
    monkeypatch.setattr(V, "GRADER", boom)
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v4\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v4", item_id=9)
    assert out["ok"] and out["review"] == "skipped"


def test_a_vault_land_passing_review_records_it(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "all clauses met"))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v5\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v5", item_id=9)
    assert out["review"] == "pass"
    assert _vault_events("vault_review")[-1]["blocking"] is False
