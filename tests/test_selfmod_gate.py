"""Gate mechanics: the ladder, its short-circuit, and its fail-closed rule.

The property worth the most here is `test_a_rung_that_errors_is_a_failed_rung`.
With no human review tier, a rung that raises and is read as "didn't fail"
would silently remove a check from the only thing standing between a proposal
and production.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.selfmod import gate as G


# ---------------------------------------------------------------------------
# pytest summary parsing — table-driven on real lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,expect", [
    ("1247 passed, 3 xfailed, 16 warnings in 33.41s",
     {"passed": 1247, "xfailed": 3, "failed": 0, "errors": 0}),
    ("5 failed, 1105 passed in 29.60s",
     {"passed": 1105, "failed": 5}),
    ("collected 1250 items\n\n1247 passed, 3 xfailed in 30.20s",
     {"collected": 1250, "passed": 1247}),
    ("2 errors in 1.20s", {"errors": 2}),
    ("no tests ran in 0.01s", {"passed": 0, "collected": 0}),
])
def test_pytest_summary_parsing(line, expect):
    got = G._parse_pytest_summary(line)
    for k, v in expect.items():
        assert got[k] == v, f"{k}: {got}"


def test_collected_falls_back_to_the_sum_when_absent():
    got = G._parse_pytest_summary("1247 passed, 3 xfailed in 30s")
    assert got["collected"] == 1250


# ---------------------------------------------------------------------------
# Ladder mechanics
# ---------------------------------------------------------------------------

class _StubGate(G.Gate):
    """Gate with the rungs replaced by scripted outcomes."""

    def __init__(self, outcomes):
        self.round_id = "SM_TEST"
        self.report = G.GateReport(round_id="SM_TEST", base="a" * 40, head="b" * 40)
        self.outcomes = outcomes
        self.called: list[str] = []
        self.skip_smoke = False
        self._canary = None

    def _make(self, name):
        def rung():
            self.called.append(name)
            outcome = self.outcomes[name]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return rung

    def run(self):
        for name in ("preflight", "static", "tests", "venv",
                     "canary_boot", "canary_smoke", "drill"):
            if not self._rung(name, self._make(name)):
                self.report.ok = False
                return self.report
        self.report.ok = True
        return self.report


ALL_PASS = {n: (True, "ok", {}) for n in
            ("preflight", "static", "tests", "venv", "canary_boot", "canary_smoke", "drill")}


def test_all_rungs_passing_is_a_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    assert g.run().ok
    assert len(g.called) == 7


def test_a_failing_rung_short_circuits_the_expensive_ones(monkeypatch):
    """An import error must cost 3 seconds, not a full canary boot."""
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    outcomes = dict(ALL_PASS)
    outcomes["static"] = (False, "import smoke failed", {})
    g = _StubGate(outcomes)
    report = g.run()
    assert not report.ok
    assert g.called == ["preflight", "static"]
    assert "canary_boot" not in g.called


def test_a_rung_that_errors_is_a_failed_rung(monkeypatch):
    """Fail closed. A raising rung must never read as 'didn't fail'."""
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    outcomes = dict(ALL_PASS)
    outcomes["tests"] = RuntimeError("subprocess exploded")
    g = _StubGate(outcomes)
    report = g.run()
    assert not report.ok
    failed = [r for r in report.rungs if not r.ok]
    assert failed[0].name == "tests"
    assert "RuntimeError" in failed[0].detail
    assert "venv" not in g.called


def test_every_rung_result_is_recorded_even_on_success(monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    report = g.run()
    assert [r.name for r in report.rungs] == [
        "preflight", "static", "tests", "venv", "canary_boot", "canary_smoke", "drill"]
    assert all(r.seconds >= 0 for r in report.rungs)


def test_the_report_serializes_for_the_round_log(monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    d = g.run().to_dict()
    assert d["ok"] is True and len(d["rungs"]) == 7
    assert set(d) >= {"round_id", "base", "head", "ok", "rungs", "changed_paths"}


# ---------------------------------------------------------------------------
# Preflight guards against a real repo
# ---------------------------------------------------------------------------

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def live_repo(tmp_path):
    r = tmp_path / "live"
    (r / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r


def _gate_for(live, worktree, base, monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(G.S, "is_halted", lambda: False)
    monkeypatch.setattr(G.S, "is_broken", lambda: False)
    g = G.Gate("SM_T", worktree, base, live_root=live)
    return g


def test_preflight_refuses_a_dirty_live_tree(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    (live_repo / "app" / "dirty.py").write_text("x = 1\n", encoding="utf-8")
    g = _gate_for(live_repo, live_repo, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "dirty" in reason


def test_preflight_refuses_when_halted(live_repo, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    g = _gate_for(live_repo, live_repo, base, monkeypatch)
    monkeypatch.setattr(G.S, "is_halted", lambda: True)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "halted" in reason


def test_preflight_refuses_when_the_guardian_is_broken(live_repo, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    g = _gate_for(live_repo, live_repo, base, monkeypatch)
    monkeypatch.setattr(G.S, "is_broken", lambda: True)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "BROKEN" in reason


def test_preflight_refuses_a_moved_base(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/x", str(wt), base)
    (live_repo / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")
    git(live_repo, "add", "-A")
    git(live_repo, "commit", "-q", "-m", "moved")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "moved" in reason


def test_preflight_refuses_a_no_op_diff(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/y", str(wt), base)
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "no changes" in reason


def test_preflight_refuses_a_denied_path(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/z", str(wt), base)
    (wt / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "touch config")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "denied" in reason


def test_preflight_accepts_an_in_scope_diff(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/ok", str(wt), base)
    (wt / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "ordinary change")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, data = g.rung_preflight()
    assert ok, reason
    assert g.report.changed_paths == ["app/m.py"]
    assert not data["buckets"]["protected"]


def test_preflight_flags_the_drill_for_a_protected_path(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/p", str(wt), base)
    d = wt / "agent-services" / "guardian"
    d.mkdir(parents=True)
    (d / "detect.py").write_text("X = 1\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "touch the guardian")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, data = g.rung_preflight()
    assert ok and "drill required" in reason
    assert data["buckets"]["protected"]


def test_preflight_refuses_a_merge_commit(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    git(live_repo, "checkout", "-q", "-b", "side")
    (live_repo / "app" / "s.py").write_text("S = 1\n", encoding="utf-8")
    git(live_repo, "add", "-A"); git(live_repo, "commit", "-q", "-m", "side")
    git(live_repo, "checkout", "-q", "main")
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "selfmod/m", str(wt), base)
    (wt / "app" / "m.py").write_text("V = 3\n", encoding="utf-8")
    git(wt, "add", "-A"); git(wt, "commit", "-q", "-m", "wt change")
    git(wt, "merge", "--no-ff", "-q", "-m", "merge side", "side")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "merge commits" in reason


# ---------------------------------------------------------------------------
# Test-deletion guard
# ---------------------------------------------------------------------------

def test_the_collected_floor_is_a_real_constant():
    """`pytest -q` exits 0 if a round deletes the test that was failing, so the
    floor is not decoration."""
    assert G.PYTEST_MIN_COLLECTED >= 1000
