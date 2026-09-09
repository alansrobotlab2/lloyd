"""Gate mechanics: the ladder, its short-circuit, and its fail-closed rule.

The property worth the most here is `test_a_rung_that_errors_is_a_failed_rung`.
With no human review tier, a rung that raises and is read as "didn't fail"
would silently remove a check from the only thing standing between a proposal
and production.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.autoimplement import gate as G


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
        for name in ("preflight", "static", "frontend", "tests", "venv",
                     "canary_boot", "canary_smoke", "drill"):
            if not self._rung(name, self._make(name)):
                self.report.ok = False
                return self.report
        self.report.ok = True
        return self.report


ALL_PASS = {n: (True, "ok", {}) for n in
            ("preflight", "static", "frontend", "tests", "venv", "canary_boot",
             "canary_smoke", "drill")}


def test_all_rungs_passing_is_a_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    assert g.run().ok
    assert len(g.called) == 8


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
        "preflight", "static", "frontend", "tests", "venv", "canary_boot", "canary_smoke", "drill"]
    assert all(r.seconds >= 0 for r in report.rungs)


def test_the_report_serializes_for_the_round_log(monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    d = g.run().to_dict()
    assert d["ok"] is True and len(d["rungs"]) == 8
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


def test_preflight_refuses_a_no_op_diff(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/y", str(wt), base)
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "no changes" in reason


def test_preflight_refuses_a_denied_path(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/z", str(wt), base)
    (wt / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "touch config")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "denied" in reason


def test_preflight_accepts_an_in_scope_diff(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/ok", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/p", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/m", str(wt), base)
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


# ---------------------------------------------------------------------------
# Rung `frontend`: tsc as a delta, vite build as a bar
# ---------------------------------------------------------------------------

from collections import Counter


def _frontend_gate(live_repo, tmp_path, monkeypatch, *, changed, head, base, build=(True, "")):
    """A gate over a real worktree with the node tooling replaced: `head` and
    `base` are the tsc findings for the worktree and the live tree."""
    base_sha = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "autoimplement/fe", str(wt), base_sha)
    (live_repo / "web" / "node_modules" / ".bin").mkdir(parents=True, exist_ok=True)
    (live_repo / "web" / "node_modules" / ".bin" / "vite").write_text("#!/bin/sh\n")
    (wt / "web").mkdir(exist_ok=True)
    g = _gate_for(live_repo, wt, base_sha, monkeypatch)
    g.report.changed_paths = list(changed)
    monkeypatch.setattr(G, "_tsc_findings",
                        lambda web, timeout=300: Counter(head if "wt" in str(web) else base))
    monkeypatch.setattr(G, "_vite_build", lambda web, out_dir, timeout=600: build)
    return g, wt


def test_parse_tsc_normalises_positions_and_counts_duplicates():
    text = ("src/a.tsx(10,2): error TS6133: 'x' is declared but its value is never read.\n"
            "src/a.tsx(40,2): error TS6133: 'x' is declared but its value is never read.\n"
            "Found 2 errors in the same file.\n")
    assert G._parse_tsc(text) == Counter({
        "src/a.tsx: error TS6133: 'x' is declared but its value is never read.": 2})


def test_frontend_rung_is_skipped_when_no_frontend_changed(live_repo, tmp_path, monkeypatch):
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["app/m.py"],
                          head={"boom": 1}, base={})
    ok, reason, _ = g.rung_frontend()
    assert ok and "no frontend changed" in reason


def test_frontend_rung_tolerates_pre_existing_tsc_errors(live_repo, tmp_path, monkeypatch):
    """The tree carried three tsc errors the day this rung was written. An
    absolute bar would have been switched off within the hour."""
    old = {"src/EntityGraph.tsx: error TS2322: bad type": 1}
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"],
                          head=old, base=old)
    ok, reason, _ = g.rung_frontend()
    assert ok and "no new errors (1 pre-existing)" in reason and "vite build ok" in reason


def test_frontend_rung_fails_on_a_new_tsc_error(live_repo, tmp_path, monkeypatch):
    old = {"src/EntityGraph.tsx: error TS2322: bad type": 1}
    new = dict(old, **{"src/x.tsx: error TS2304: Cannot find name 'foo'.": 1})
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"],
                          head=new, base=old)
    ok, reason, detail = g.rung_frontend()
    assert not ok and "1 new tsc error" in reason and "TS2304" in reason
    assert detail["new"] == ["src/x.tsx: error TS2304: Cannot find name 'foo'."]


def test_a_second_copy_of_an_existing_error_is_a_new_error(live_repo, tmp_path, monkeypatch):
    key = "src/a.tsx: error TS6133: 'x' is declared but its value is never read."
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/a.tsx"],
                          head={key: 2}, base={key: 1})
    ok, reason, _ = g.rung_frontend()
    assert not ok and "1 new tsc error" in reason


def test_frontend_rung_fails_when_the_build_fails(live_repo, tmp_path, monkeypatch):
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"],
                          head={}, base={}, build=(False, "Could not resolve './missing'"))
    ok, reason, _ = g.rung_frontend()
    assert not ok and "vite build failed" in reason and "missing" in reason


def test_frontend_rung_links_the_live_node_modules_into_the_worktree(live_repo, tmp_path, monkeypatch):
    """636 MB, gitignored, and identical by construction because package.json
    and the lockfile are denied — so the live install is the candidate's."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"],
                           head={}, base={})
    assert not (wt / "web" / "node_modules").exists()
    ok, _, _ = g.rung_frontend()
    assert ok
    link = wt / "web" / "node_modules"
    assert link.is_symlink() and link.resolve() == (live_repo / "web" / "node_modules").resolve()


def test_frontend_rung_refuses_when_the_live_tree_has_no_install(live_repo, tmp_path, monkeypatch):
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"],
                           head={}, base={})
    import shutil
    shutil.rmtree(live_repo / "web" / "node_modules")
    ok, reason, _ = g.rung_frontend()
    assert not ok and "npm install" in reason


# ---------------------------------------------------------------------------
# The base probe: is this failure the round's fault, or was the tree red?
#
# On 2026-09-08 three rounds aborted at this rung on three failures that
# reproduced on their own pristine base, and each one was recorded as its
# item's single attempt. #361, #370 and #376 became unreachable; the fix landed
# seven hours after the last of them and nothing went back. Blocking those
# promotions was right — a red tree is not a tree to land onto. Spending the
# items was not.
# ---------------------------------------------------------------------------

SUMMARY = """\
=========================== short test summary info ============================
FAILED tests/test_guardian_speak.py::test_alert_dispatches_voice_by_default - assert False
FAILED tests/test_tool_overrides.py::test_a_written_override_leaves_the_live_tree_clean
ERROR tests/test_broken_import.py
3 failed, 2044 passed, 8 skipped, 3 xfailed in 44.99s
"""


def test_failed_node_ids_reads_both_failed_and_error_lines():
    assert G._failed_node_ids(SUMMARY) == [
        "tests/test_guardian_speak.py::test_alert_dispatches_voice_by_default",
        "tests/test_tool_overrides.py::test_a_written_override_leaves_the_live_tree_clean",
        "tests/test_broken_import.py",
    ]


def test_failed_node_ids_ignores_prose_and_dedupes():
    """pytest writes the word ERROR in plenty of places that are not a node
    id, and a node id always names a file."""
    text = ("ERROR could not load plugin\n"
            "FAILED tests/a.py::t - boom\n"
            "FAILED tests/a.py::t - boom\n"
            "some ERROR tests/b.py mid-line\n")
    assert G._failed_node_ids(text) == ["tests/a.py::t"]


@pytest.mark.parametrize("node_ids,base_failed,external,new", [
    # every failure predates the round: the exemption case
    (["a::t", "b::t"], {"a::t", "b::t"}, True, []),
    # one is new — the round broke something, and one real regression is
    # enough to disqualify it however many old failures sit beside it
    (["a::t", "b::t"], {"a::t"}, False, ["b::t"]),
    # nothing reproduces: an ordinary failing round
    (["a::t"], set(), False, ["a::t"]),
    # the base probe found MORE than the round did; still external
    (["a::t"], {"a::t", "z::t"}, True, []),
    # unparseable summary is never an exemption — fail closed
    ([], {"a::t"}, False, []),
])
def test_classify_test_failure_is_a_delta(node_ids, base_failed, external, new):
    assert G._classify_test_failure(node_ids, set(base_failed)) == (external, new)


def _repo_with_failing_test(tmp_path):
    r = tmp_path / "live"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_pre.py").write_text(
        "def test_already_broken():\n    assert False\n\n"
        "def test_fine():\n    assert True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r, git(r, "rev-parse", "HEAD").stdout.strip()


def test_the_base_probe_finds_a_failure_that_predates_the_round(tmp_path):
    """The integration half, against a real git repo and a real pytest: the
    probe checks out the base into a throwaway worktree and re-runs exactly the
    failing node ids there."""
    repo, base = _repo_with_failing_test(tmp_path)
    failed, note = G._failures_at_base(
        Path(sys.executable), repo, base,
        ["tests/test_pre.py::test_already_broken", "tests/test_pre.py::test_fine"],
        tmp_path / "scratch")
    assert failed == {"tests/test_pre.py::test_already_broken"}
    assert "already failing" in note

    external, new = G._classify_test_failure(
        ["tests/test_pre.py::test_already_broken"], failed)
    assert external is True and new == []


def test_the_base_probe_cleans_up_its_worktree(tmp_path):
    """It runs inside a round that is still open, and a stray registration
    would outlive it — `promote.py` and `preflight` both refuse a dirty tree."""
    repo, base = _repo_with_failing_test(tmp_path)
    G._failures_at_base(Path(sys.executable), repo, base,
                        ["tests/test_pre.py::test_already_broken"], tmp_path / "scratch")
    listed = git(repo, "worktree", "list", "--porcelain").stdout
    assert "baseline" not in listed, listed
    assert not (tmp_path / "scratch" / "baseline").exists()


def test_the_base_probe_fails_closed_when_the_worktree_cannot_be_made(tmp_path):
    """Every failure mode returns the empty set, which classifies the failure
    as the round's own. Being wrong that way costs the status quo; being wrong
    the other way lands a change nobody checked."""
    repo, _ = _repo_with_failing_test(tmp_path)
    failed, note = G._failures_at_base(Path(sys.executable), repo,
                                       "0000000000000000000000000000000000000000",
                                       ["tests/test_pre.py::test_already_broken"],
                                       tmp_path / "scratch")
    assert failed == set()
    assert "baseline worktree failed" in note


def test_no_node_ids_is_not_an_exemption(tmp_path):
    failed, note = G._failures_at_base(Path(sys.executable), tmp_path, "HEAD", [],
                                       tmp_path / "scratch")
    assert failed == set() and "no node ids" in note


def test_rung_tests_reports_pre_existing_breakage_as_an_external_blocker(tmp_path, monkeypatch):
    """End to end through the real rung: a tree that was already red, plus a
    round that changed something unrelated. The rung still FAILS — landing onto
    a red tree would hand the guardian's observation window a broken baseline —
    but it says whose fault it is, and `backlog.implemented_ids` reads that."""
    repo, base = _repo_with_failing_test(tmp_path)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_T" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", "autoimplement/SM_T", str(wt), base)
    (wt / "unrelated.py").write_text("X = 1\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "an unrelated change")

    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    ok, detail, data = g.rung_tests()

    assert ok is False, "a red tree is still not a tree to land onto"
    assert data.get("external_blocker") is True
    assert data["external_failures"] == ["tests/test_pre.py::test_already_broken"]
    assert "PRE-EXISTING BREAKAGE" in detail
    git(repo, "worktree", "remove", "--force", str(wt))


def test_rung_tests_blames_the_round_for_a_failure_it_introduced(tmp_path, monkeypatch):
    """The counterfactual that makes the test above mean something: the same
    already-red tree, but this round also breaks a test of its own. One new
    failure disqualifies the exemption however many old ones sit beside it."""
    repo, base = _repo_with_failing_test(tmp_path)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_T2" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", "autoimplement/SM_T2", str(wt), base)
    (wt / "tests" / "test_mine.py").write_text(
        "def test_i_broke_this():\n    assert False\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "a change that breaks its own test")

    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    ok, detail, data = g.rung_tests()

    assert ok is False
    assert not data.get("external_blocker"), "one new failure is the round's own"
    assert data["new_failures"] == ["tests/test_mine.py::test_i_broke_this"]
    assert "are new in this round" in detail
    git(repo, "worktree", "remove", "--force", str(wt))


def test_a_test_the_round_added_does_not_hide_the_pre_existing_ones(tmp_path, monkeypatch):
    """Regression, found building this: probing by node id is wrong. Handed a
    node id that does not exist at base — a test the round just wrote, in a file
    that already existed — pytest exits `ERROR: not found:` and runs NOTHING, so
    the pre-existing failure beside it never reports and a red tree reads as
    green. The probe runs whole files for exactly this reason."""
    repo, base = _repo_with_failing_test(tmp_path)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_T3" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", "autoimplement/SM_T3", str(wt), base)
    # A new failing test appended to the SAME file that is already red.
    (wt / "tests" / "test_pre.py").write_text(
        "def test_already_broken():\n    assert False\n\n"
        "def test_fine():\n    assert True\n\n"
        "def test_added_by_this_round():\n    assert False\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "adds a failing test to an already-red file")

    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    ok, detail, data = g.rung_tests()

    assert ok is False
    assert not data.get("external_blocker"), "the round added a failure of its own"
    assert data["new_failures"] == ["tests/test_pre.py::test_added_by_this_round"]
    # The pre-existing one was still seen — that is what a node-id probe lost.
    assert "tests/test_pre.py::test_already_broken" in data["failed_node_ids"]
    assert "1 of 2 failures are new" in detail
    git(repo, "worktree", "remove", "--force", str(wt))


def test_a_probe_that_could_not_run_says_so_instead_of_reporting_zero(tmp_path):
    """Found by running this against real history: handed a python with no
    pytest, the probe reported "0 already failing" and blamed the round. Both
    "nothing pre-existing" and "pytest never ran" are an empty set, and only
    one of them is an answer — a silent downgrade is indistinguishable from
    success. The verdict is the same (no exemption, fail closed); the note is
    not."""
    repo, base = _repo_with_failing_test(tmp_path)
    # A python that exists and runs, but cannot import pytest — the exact
    # shape that produced the silent zero (a `.resolve()` on the venv symlink
    # landed on the bare uv interpreter).
    stub = tmp_path / "python-without-pytest"
    stub.write_text("#!/bin/sh\necho 'No module named pytest' >&2\nexit 1\n",
                    encoding="utf-8")
    stub.chmod(0o755)
    failed, note = G._failures_at_base(stub, repo, base,
                                       ["tests/test_pre.py::test_already_broken"],
                                       tmp_path / "scratch")
    assert failed == set()
    assert "INCONCLUSIVE" in note, note
    assert "already failing" not in note

    # A binary that does not exist at all takes the exception path, which is
    # also named rather than silent.
    failed, note = G._failures_at_base(Path("/nonexistent/python"), repo, base,
                                       ["tests/test_pre.py::test_already_broken"],
                                       tmp_path / "scratch")
    assert failed == set() and "baseline probe failed" in note


# ---------------------------------------------------------------------------
# Preflight: the two refusals that are about the live tree, not the diff
# ---------------------------------------------------------------------------

def test_an_empty_diff_carries_no_exemption(live_repo, tmp_path, monkeypatch):
    """The counterfactual that keeps the preflight exemption honest. "No
    changes to promote" is also a preflight failure and is entirely the
    round's own — widening an exemption is how it becomes an open door."""
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt2"
    git(live_repo, "worktree", "add", "-q", "-b", "empty", str(wt), base)
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is False and "no changes to promote" in detail
    assert not data.get("external_blocker")
    git(live_repo, "worktree", "remove", "--force", str(wt))


# ---------------------------------------------------------------------------
# The tree is shared: rebase and retest, tolerate dirt that is not ours
#
# A human commits to `main` while a round is open. Until 2026-09-09 that was
# "something landed under you; abort and re-cut" — the round's diff thrown away
# to be reapplied by hand onto a base one commit newer — and any uncommitted
# edit anywhere in production was a refusal, which cost #447 587 lines while
# an unrelated file sat modified for an hour. The rebase is the reapplication,
# done by git; the ladder below it is the retest; and dirt is a problem only
# when it is in the round's own files.
# ---------------------------------------------------------------------------

def _round(live_repo, tmp_path, name, base, edit=("app/m.py", "V = 2\n")):
    wt = tmp_path / f"wt_{name}"
    git(live_repo, "worktree", "add", "-q", "-b", f"autoimplement/{name}", str(wt), base)
    rel, body = edit
    (wt / rel).parent.mkdir(parents=True, exist_ok=True)
    (wt / rel).write_text(body, encoding="utf-8")
    git(wt, "add", "-A"); git(wt, "commit", "-q", "-m", f"round {name}")
    return wt


def _land_on_main(live_repo, rel, body, msg="landed underneath"):
    (live_repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (live_repo / rel).write_text(body, encoding="utf-8")
    git(live_repo, "add", "-A"); git(live_repo, "commit", "-q", "-m", msg)
    return git(live_repo, "rev-parse", "HEAD").stdout.strip()


def test_a_moved_main_is_rebased_onto_and_the_gate_continues(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r1", base)
    candidate = git(wt, "rev-parse", "HEAD").stdout.strip()
    new_main = _land_on_main(live_repo, "app/other.py", "y = 1\n")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()

    assert ok is True, detail
    assert data["rebased"]["from"] == base and data["rebased"]["onto"] == new_main
    assert data["rebased"]["old_head"] == candidate
    assert g.report.base == new_main, "gate.json is what land reads its base from"
    assert g.report.head == git(wt, "rev-parse", "HEAD").stdout.strip() != candidate
    assert git(live_repo, "merge-base", "--is-ancestor", new_main, g.report.head).returncode == 0
    # Only the round's own change is its diff — not the commit that landed.
    assert g.report.changed_paths == ["app/m.py"]
    assert "rebased" in detail and "on top of what landed" in detail


def test_a_rebase_conflict_names_the_files_and_leaves_the_worktree_as_it_was(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r2", base)
    candidate = git(wt, "rev-parse", "HEAD").stdout.strip()
    _land_on_main(live_repo, "app/m.py", "V = 99  # the same line, from main\n")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()

    assert ok is False
    assert data.get("external_blocker") is True, "someone else's change collided; not the round's fault"
    assert data["conflicts"] == ["app/m.py"] and "app/m.py" in detail
    assert git(wt, "rev-parse", "HEAD").stdout.strip() == candidate, "aborted, not half-applied"
    assert git(wt, "status", "--porcelain").stdout.strip() == ""
    assert not (wt / ".git" / "rebase-merge").exists() and not (wt / ".git" / "rebase-apply").exists()


def test_a_worktree_with_uncommitted_changes_is_not_rebased_and_that_is_its_own(live_repo, tmp_path, monkeypatch):
    """Autostashing would carry those changes across silently — and they are
    not in the round's diff either way. The round is told to commit."""
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r3", base)
    (wt / "app" / "half_done.py").write_text("z = 1\n", encoding="utf-8")
    _land_on_main(live_repo, "app/other.py", "y = 1\n")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is False and "commit" in detail
    assert not data.get("external_blocker"), "the worktree's own state is the round's own"


def test_uncommitted_live_edits_outside_the_diff_are_tolerated_and_recorded(live_repo, tmp_path, monkeypatch):
    """#447. A fast-forward of disjoint paths never touches the file the
    human is editing; refusing on any dirt at all is what cost 587 lines."""
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r4", base)
    (live_repo / "app" / "someone_elses_wip.py").write_text("x = 1\n", encoding="utf-8")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is True, detail
    assert data["dirty_tolerated"] == ["app/someone_elses_wip.py"]
    assert "tolerating 1 uncommitted live path" in detail


def test_uncommitted_live_edits_in_the_rounds_own_files_are_refused_by_name(live_repo, tmp_path, monkeypatch):
    """Two writers on one file. git would refuse the fast-forward anyway; the
    gate says which file, before the pool is paused for a landing."""
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r5", base)
    (live_repo / "app" / "m.py").write_text("V = 1  # being edited live\n", encoding="utf-8")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is False
    assert data.get("external_blocker") is True
    assert data["overlap"] == ["app/m.py"] and "app/m.py" in detail


def test_an_untracked_live_file_the_round_creates_is_an_overlap(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r6", base, edit=("app/new.py", "N = 1\n"))
    (live_repo / "app" / "new.py").write_text("N = 'theirs'\n", encoding="utf-8")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is False and data.get("external_blocker") is True
    assert data["overlap"] == ["app/new.py"]


def test_the_retest_judges_the_change_on_top_of_what_landed(tmp_path, monkeypatch):
    """The point of rebasing rather than refusing: the build that will be live
    is the round's change PLUS what landed, and nothing had tested that. Here
    `main` lands a test the round's change breaks. After the rebase, the tests
    rung fails — and the base probe says that test passes at the new base, so
    the failure is the round's own, not an exemption."""
    live = tmp_path / "live"
    (live / "app").mkdir(parents=True); (live / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com"); git(live, "config", "user.name", "t")
    (live / "app" / "__init__.py").write_text("", encoding="utf-8")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    (live / "tests" / "test_smoke.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "base")
    base = git(live, "rev-parse", "HEAD").stdout.strip()

    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_RT" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(live, "worktree", "add", "-q", "-b", "autoimplement/SM_RT", str(wt), base)
    (wt / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")
    git(wt, "add", "-A"); git(wt, "commit", "-q", "-m", "round: V = 2")

    # Meanwhile main lands a test that pins V == 1.
    (live / "tests" / "test_v.py").write_text("from app.m import V\n\ndef test_v_is_one():\n    assert V == 1\n",
                                              encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "pin V")

    monkeypatch.setattr(G, "PYTEST_MIN_COLLECTED", 0)
    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 0)
    g = _gate_for(live, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    ok, detail, data = g.rung_preflight()
    assert ok is True, detail
    assert (wt / "tests" / "test_v.py").exists(), "the worktree now carries what landed"

    ok, detail, data = g.rung_tests()
    assert ok is False
    assert "tests/test_v.py::test_v_is_one" in data["failed_node_ids"]
    assert not data.get("external_blocker"), "passes at the new base; the round broke it"
    assert data["new_failures"] == ["tests/test_v.py::test_v_is_one"]
    git(live, "worktree", "remove", "--force", str(wt))
