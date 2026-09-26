"""The review's grading turn runs beside the tests rung (2026-09-25).

The grader needs nothing the suite computes before it starts: the counts only
colour one prompt line, and the two things that decide a verdict — a green
tests rung and which failures predate the round — are applied by
`parse_review` in Python AFTER it answers. So `Gate.run` starts the grade when
the tests rung starts and the review rung joins it. What follows pins that the
join is only ever taken when it answers the same question the serial rung
would have asked, and that a gate which stops before review records nothing.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from scripts.automod import backlog as B, gate as G, review as RV, state as S

HEAD = "c" * 40
MET = {"premise": "sound", "summary": "does what it says",
       "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                    "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                    "how_verified": "ran", "note": "ran it"}],
       "test_honesty": [], "seams_unverified": []}


def _cfg(monkeypatch, *, concurrent: bool):
    from app.config import CONFIG
    automod = dict(CONFIG.get("automod") or {})
    gate = dict(automod.get("gate") or {})
    gate.update(concurrent_review=concurrent, reuse_rungs=False)
    automod["gate"] = gate
    monkeypatch.setitem(CONFIG, "automod", automod)


class _Grader:
    """A fake `RV.grade`: records every call, names its session through
    `on_session`, and can be held until the test releases it."""

    def __init__(self, structured=MET, *, hold: bool = False):
        self.structured = structured
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def __call__(self, **kw):
        self.calls.append(kw)
        sid = f"sess_{len(self.calls)}"
        if kw.get("on_session"):
            kw["on_session"](sid)
        self.started.set()
        self.release.wait(10)
        return {"ok": True, "error": "", "session_id": sid, "structured": self.structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}


class _Gate(G.Gate):
    """The real `run` and the real review rung; every other rung a stub."""

    def __init__(self, tmp_path, tests):
        super().__init__("SM_CR", tmp_path, "a" * 40, item_id=7, live_root=tmp_path)
        self.report.changed_paths = ["app/x.py", "tests/test_x.py"]
        self._tests = tests
        self.ran: list[str] = []
        self.snapshots: list[str] = []
        self.dropped: list = []

    def _ok(name):  # noqa: N805 — a stub factory, not a method
        def rung(self):
            self.ran.append(name)
            return True, "ok", {}
        return rung

    rung_preflight = _ok("preflight")
    rung_vet = _ok("vet")
    rung_static = _ok("static")
    rung_frontend = _ok("frontend")
    rung_prompt_surface = _ok("prompt_surface")
    rung_venv = _ok("venv")
    rung_canary_boot = _ok("canary_boot")
    rung_canary_smoke = _ok("canary_smoke")
    rung_drill = _ok("drill")

    def rung_tests(self, only=None):
        self.ran.append("tests")
        return self._tests(self)

    # Never let a test touch a real repo: the snapshot is a name, not a checkout.
    def _review_snapshot(self, head, suffix=""):
        self.snapshots.append(suffix)
        return None, f"stub snapshot{suffix}"

    def _drop_snapshot(self, wt):
        self.dropped.append(wt)

    def _patch_id(self):
        return "p" * 40


@pytest.fixture
def armed(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")
    (tmp_path / "backlog").mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "GATE_TESTS_LOCK_PATH", tmp_path / "gate-tests.lock")
    monkeypatch.setattr(S, "GATE_CANARY_LOCK_PATH", tmp_path / "gate-canary.lock")
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    head = {"now": HEAD}
    monkeypatch.setattr(G.W, "head", lambda wt: head["now"])
    events: list[dict] = []
    prior: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(prior))
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b", "clauses": ["the thing happens once"], "path": ""})
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    cancelled: list[str] = []
    monkeypatch.setattr(RV, "cancel_grader", lambda sid, **k: cancelled.append(sid) or True)
    parse_calls: list[dict] = []
    real_parse = RV.parse_review

    def parse(obj, **kw):
        parse_calls.append(kw)
        return real_parse(obj, **kw)
    monkeypatch.setattr(RV, "parse_review", parse)
    _cfg(monkeypatch, concurrent=True)
    return {"tmp": tmp_path, "events": events, "prior": prior, "head": head,
            "cancelled": cancelled, "parse": parse_calls, "monkeypatch": monkeypatch}


def _reviews(events):
    return [e for e in events if e.get("event") == "review"]


def _green(g, **extra):
    return True, "pytest passed", {"passed": 10, "failed": 0, **extra}


# ── the join ──────────────────────────────────────────────────────────────

def test_the_grade_starts_before_the_tests_rung_finishes_and_a_pass_joins_it(armed):
    grader = _Grader()
    armed["monkeypatch"].setattr(RV, "grade", grader)
    seen = {}

    def tests(g):
        # The suite cannot finish until the grader has started: were the grade
        # still serial, this would time out and the rung would fail.
        seen["started_first"] = grader.started.wait(10)
        return _green(g)

    g = _Gate(armed["tmp"], tests)
    report = g.run()
    assert seen["started_first"] is True
    assert report.ok, [(r.name, r.detail) for r in report.rungs]
    assert len(grader.calls) == 1, "the prefetched grade is the one used — no second turn"
    call = grader.calls[0]
    assert call["test_counts"] == {} and call["pre_existing_failures"] == []
    assert call["scratch_dir"].name == "review-prefetch"
    assert g.snapshots == ["-prefetch"]
    rev = _reviews(armed["events"])
    assert len(rev) == 1 and rev[0]["review_concurrent"] is True
    assert rev[0]["ok"] is True and rev[0]["blocking"] is False
    assert "review_prefetch_discarded" not in rev[0]
    review_rung = next(r for r in report.rungs if r.name == "review")
    assert review_rung.data["review_concurrent"] is True
    assert armed["cancelled"] == []


def test_parse_review_is_handed_the_real_tests_rung(armed):
    armed["monkeypatch"].setattr(RV, "grade", _Grader())
    _Gate(armed["tmp"], _green).run()
    assert armed["parse"] and armed["parse"][-1]["tests_passed"] is True
    assert armed["parse"][-1]["pre_existing_failures"] == set()


def test_a_red_tests_rung_writes_no_review_and_spends_nothing(armed):
    grader = _Grader(hold=True)
    armed["monkeypatch"].setattr(RV, "grade", grader)

    def tests(g):
        grader.started.wait(10)
        return False, "pytest failed (failed=1)", {"failed": 1, "new_failures": ["tests/t.py::x"]}

    g = _Gate(armed["tmp"], tests)
    try:
        report = g.run()
    finally:
        grader.release.set()
    assert report.ok is False
    assert [r.name for r in report.rungs][-1] == "tests"
    assert "review" not in [r.name for r in report.rungs]
    assert _reviews(armed["events"]) == [], "nothing was judged, so nothing is recorded"
    assert not any(e.get("rung") == "review" for e in armed["events"])
    # The grader's turn is cancelled best effort, and its checkout dropped.
    assert armed["cancelled"] == ["sess_1"]


def test_pre_existing_failures_discard_the_prefetch_and_grade_again_told(armed):
    grader = _Grader()
    armed["monkeypatch"].setattr(RV, "grade", grader)
    red = ["tests/test_old.py::test_red"]
    report = _Gate(armed["tmp"], lambda g: _green(g, pre_existing_failures=red)).run()
    assert report.ok
    assert len(grader.calls) == 2
    serial = grader.calls[1]
    assert serial["pre_existing_failures"] == red
    assert serial["test_counts"].get("passed") == 10
    assert "on_session" not in serial and serial["scratch_dir"].name == "gate-state"
    rev = _reviews(armed["events"])
    assert len(rev) == 1 and rev[0]["review_concurrent"] is False
    assert "pre-existing" in rev[0]["review_prefetch_discarded"]
    assert armed["parse"][-1]["pre_existing_failures"] == set(red)


def test_a_head_that_moved_during_the_suite_is_graded_again(armed):
    grader = _Grader()
    armed["monkeypatch"].setattr(RV, "grade", grader)

    def tests(g):
        grader.started.wait(10)
        armed["head"]["now"] = "d" * 40
        return _green(g)

    _Gate(armed["tmp"], tests).run()
    assert len(grader.calls) == 2
    rev = _reviews(armed["events"])
    assert rev[0]["review_concurrent"] is False and rev[0]["head"] == "d" * 40
    assert "head" in rev[0]["review_prefetch_discarded"]


# ── the early answers are exactly what they were ─────────────────────────

def _prior_refusal(head):
    return {"event": "review", "round_id": "SM_CR", "ok": True, "blocking": True,
            "attempt": 1, "head": head, "findings": "clause 1 unmet", "patch_id": ""}


@pytest.mark.parametrize("case", ["patch_reuse", "same_head", "exhausted"])
def test_an_early_answer_starts_no_grade(armed, case):
    grader = _Grader()
    armed["monkeypatch"].setattr(RV, "grade", grader)
    if case == "patch_reuse":
        armed["prior"].append({"event": "review", "round_id": "SM_CR", "ok": True,
                               "blocking": False, "head": "e" * 40, "patch_id": "p" * 40})
    elif case == "same_head":
        armed["prior"].append(_prior_refusal(HEAD))
    else:
        armed["prior"].extend([_prior_refusal("1" * 40), _prior_refusal("2" * 40)])
    g = _Gate(armed["tmp"], _green)
    report = g.run()
    assert grader.calls == [], "no grading turn, beside the tests rung or after it"
    review = next(r for r in report.rungs if r.name == "review")
    if case == "patch_reuse":
        assert review.ok and review.data["review_reused"] is True
    elif case == "same_head":
        assert not review.ok and review.data["review_same_head"] is True
    else:
        assert not review.ok and review.data["review_exhausted"] is True
    assert _reviews(armed["events"]) == []
    assert g.snapshots == []


def test_prepare_writes_no_event_and_can_be_called_twice(armed):
    armed["monkeypatch"].setattr(RV, "grade", _Grader())
    g = _Gate(armed["tmp"], _green)
    a, b = g._review_prepare(), g._review_prepare()
    assert isinstance(a, dict) and G.Gate._review_key(a) == G.Gate._review_key(b)
    assert armed["events"] == []


# ── the switch ────────────────────────────────────────────────────────────

def test_switched_off_the_ladder_is_serial_and_the_event_says_so(armed):
    _cfg(armed["monkeypatch"], concurrent=False)
    grader = _Grader()
    armed["monkeypatch"].setattr(RV, "grade", grader)
    seen = {}

    def tests(g):
        seen["grading"] = grader.started.is_set()
        return _green(g)

    report = _Gate(armed["tmp"], tests).run()
    assert report.ok and seen["grading"] is False
    assert len(grader.calls) == 1
    call = grader.calls[0]
    assert call["test_counts"].get("passed") == 10 and "on_session" not in call
    rev = _reviews(armed["events"])
    assert rev[0]["review_concurrent"] is False and "review_prefetch_discarded" not in rev[0]
    assert set(rev[0]) >= {"event", "round_id", "item_id", "attempt", "head", "grader_model",
                           "snapshot", "snapshot_note", "prechecks", "validated_head",
                           "validated_worktree", "patch_id", "amendments_shown",
                           "session_id", "seconds", "retries", "waited_s"}


# ── the prompt tells the grader the truth ────────────────────────────────

def _prompt(counts):
    return RV.build_prompt(contract={"id": 1, "title": "t", "body": "b", "clauses": ["c"]},
                           diff="", diff_truncated=False, changed_tests=[],
                           test_counts=counts, worktree=Path("/wt"), run_tests=Path("/rt.sh"))


def test_the_prompt_says_the_suite_is_running_when_no_counts_were_given():
    p = _prompt({})
    assert "already ran" not in p
    assert "running beside this review" in p and "do not run the whole suite" in p
    q = _prompt({"passed": 3, "failed": 0})
    assert "The full suite already ran on this worktree: passed=3, failed=0." in q
    assert "running beside this review" not in q


def test_the_graders_pytest_excludes_what_the_gate_excludes(tmp_path):
    script = RV.write_run_tests(tmp_path, worktree=tmp_path, python=Path("/py"), env={})
    text = script.read_text()
    assert G.TESTS_MARK_EXPR == "not live_vault and not fault_injection"
    assert f"-m '{G.TESTS_MARK_EXPR}'" in text
