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

from scripts.automod import gate as G


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
        for name in RUNGS:
            if not self._rung(name, self._make(name)):
                self.report.ok = False
                return self.report
        self.report.ok = True
        return self.report


# The ladder, in order. `review` sits after `tests` (it trusts a green tree)
# and before `venv` (a refusal saves the build, the boot, the smoke).
# `prompt_surface` sits after `tests` and before `review`: after, so a broken
# tree fails first and cheaply; before, because a behavioural regression in
# what the model is TOLD is a fact the reviewer should be able to see.
RUNGS = ("preflight", "static", "frontend", "tests", "prompt_surface",
         "review", "venv", "canary_boot", "canary_smoke", "drill")
ALL_PASS = {n: (True, "ok", {}) for n in RUNGS}


def test_all_rungs_passing_is_a_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    assert g.run().ok
    assert len(g.called) == len(RUNGS)


def test_the_stub_ladder_is_the_real_ladder(monkeypatch):
    """The stub above is only a test of the gate while it lists the rungs the
    gate actually runs, in the order it runs them."""
    names: list[str] = []
    g = G.Gate("SM_TEST_LADDER", Path(__file__).resolve().parent.parent, "HEAD")
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or True)
    g.run()
    assert tuple(names) == RUNGS


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
    assert [r.name for r in report.rungs] == list(RUNGS)
    assert all(r.seconds >= 0 for r in report.rungs)


def test_the_report_serializes_for_the_round_log(monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    g = _StubGate(dict(ALL_PASS))
    d = g.run().to_dict()
    assert d["ok"] is True and len(d["rungs"]) == len(RUNGS)
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
    git(live_repo, "worktree", "add", "-q", "-b", "automod/y", str(wt), base)
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "no changes" in reason


def test_preflight_refuses_a_denied_path(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "automod/z", str(wt), base)
    (wt / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "touch config")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, _ = g.rung_preflight()
    assert not ok and "denied" in reason


def test_preflight_accepts_an_in_scope_diff(live_repo, tmp_path, monkeypatch):
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "automod/ok", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", "automod/p", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", "automod/m", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", "automod/fe", str(wt), base_sha)
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
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_T", str(wt), base)
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
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_T2", str(wt), base)
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


# ---------------------------------------------------------------------------
# Flaky attribution: one run is one sample (#1196)
#
# `pytest failed (…): all 1 failure(s) are new in this round` was decided from
# a single invocation on each side — one candidate run, one base probe. Round
# SM_20260916_042752 was refused that way on a node whose own file failed 1 run
# in 5 at base, and because `backlog.implemented_ids` counts any finished round
# as the item's one attempt, the item was spent by a coin toss. These tests are
# the counterfactuals: the same one-run failure, on a tree where asking again
# settles it.
# ---------------------------------------------------------------------------

FLAKY_NODE = "tests/test_flaky.py::test_fails_the_first_time_only"
BROKEN_NODE = "tests/test_broken.py::test_fails_every_time"

# The intermittent half exists at the BASE, so the base probe sees it and (this
# is the point) passes on its one run. The deterministic half is added by the
# round, so it cannot reproduce at base by construction. Each test counts its
# own executions into `$FLAKE_RUNS_FILE.<name>`, outside both trees, so a test
# here can assert how many times the gate really ran it.
_FLAKY_SRC = (
    "import os\n"
    "from pathlib import Path\n\n"
    "def _runs(name):\n"
    "    p = Path(os.environ['FLAKE_RUNS_FILE'] + '.' + name)\n"
    "    n = int(p.read_text()) if p.exists() else 0\n"
    "    p.write_text(str(n + 1))\n"
    "    return n\n\n"
    "def test_fails_the_first_time_only():\n"
    "    assert _runs('first') > 0, 'the first run of this test always fails'\n\n"
    "def test_steady():\n"
    "    assert True\n")

_BROKEN_SRC = (
    "import os\n"
    "from pathlib import Path\n\n"
    "def _runs(name):\n"
    "    p = Path(os.environ['FLAKE_RUNS_FILE'] + '.' + name)\n"
    "    n = int(p.read_text()) if p.exists() else 0\n"
    "    p.write_text(str(n + 1))\n"
    "    return n\n\n"
    "def test_fails_every_time():\n"
    "    _runs('always')\n"
    "    assert False, 'deterministic: re-running this changes nothing'\n")


def _counted_repo(tmp_path):
    """Base: one test that fails only on its first-ever run, plus a green one."""
    r = tmp_path / "live"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_flaky.py").write_text(_FLAKY_SRC, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r, git(r, "rev-parse", "HEAD").stdout.strip()


def _counted_gate(tmp_path, monkeypatch, *, add_broken: bool):
    """A round over that repo, run serially so a flake cannot be the
    parallel-load kind the rung already re-asks for itself.

    The diff also touches a file outside `tests/`, which is what makes the rung
    run the whole suite: a partial run of the round's own test files alone would
    never execute the flake that exists at base.
    """
    repo, base = _counted_repo(tmp_path)
    counter = tmp_path / "runs"
    monkeypatch.setattr(G, "_gate_cfg",
                        lambda key, default: 1 if key == "test_workers" else default)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_FLK" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_FLK", str(wt), base)
    (wt / "unrelated.py").write_text("X = 1\n", encoding="utf-8")
    if add_broken:
        (wt / "tests" / "test_broken.py").write_text(_BROKEN_SRC, encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "an unrelated change")
    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    real_env = g._child_env
    monkeypatch.setattr(g, "_child_env",
                        lambda root=None: {**real_env(root),
                                           "FLAKE_RUNS_FILE": str(counter)})
    return g, counter, repo, wt


def test_a_failure_that_passes_its_repeat_runs_is_flaky_not_the_rounds(tmp_path, monkeypatch):
    """The #1196 case. One suite run saw the failure, the base probe saw a pass,
    and the round that happened to trip it used to be blamed and spend its
    attempt."""
    g, counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=False)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False, "a red tree is not landable, flake or no flake"
    assert data.get("external_blocker") is True, "one sample is not this round's fault"
    assert data["retried_node_ids"] == [FLAKY_NODE]
    assert data["flaky_node_ids"] == [FLAKY_NODE]
    assert data["external_failures"] == [FLAKY_NODE]
    assert "flaky" in detail.lower(), detail
    assert "are new in this round" not in detail, detail
    # suite run + base probe + ONE repeat, then the node left the batch
    assert int((Path(str(counter) + ".first")).read_text()) == 3, detail


def test_a_flaky_attribution_never_passes_the_rung(tmp_path, monkeypatch):
    """The exemption is attribution and attempt-spending only. No skip, no
    xfail, no node dropped from the run: the rung's own counts still say the
    suite ran everything and came back red."""
    g, _counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=False)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False
    assert data["failed"] == 1, "the failure is still counted, not excised"
    assert data["collected"] == 2, "the flaky node was run, not skipped out of it"
    assert data["tests_skipped"] == 0 and data["xfailed"] == 0
    assert "Blocking the promotion" in detail, detail


def test_a_failure_that_fails_every_repeat_run_is_still_the_rounds(tmp_path, monkeypatch):
    """The counterfactual that keeps the mechanism honest: re-running must not
    become a get-out clause for a round that really did break something."""
    g, counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=True)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False
    assert not data.get("external_blocker"), "it failed every repeat; it is theirs"
    assert BROKEN_NODE in data["new_failures"]
    assert FLAKY_NODE in data["flaky_node_ids"]
    assert "1 of 2 failures are new" in detail, detail
    assert "are new in this round" in detail, detail
    # suite run + k=3 repeats and no more for the deterministic one; the flaky
    # node drops out of the batch as soon as it passes one.
    assert int((Path(str(counter) + ".always")).read_text()) == 4, detail
    assert int((Path(str(counter) + ".first")).read_text()) == 3, detail


def test_only_the_nodes_that_are_new_get_re_run(tmp_path, monkeypatch):
    """A node the base probe already reproduced is already attributed; re-asking
    it is minutes spent on a verdict that already exists. Only the new node ids
    are re-run, at most 3 times each, and never the whole suite again."""
    g, _counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=True)
    argv: list[list[str]] = []
    real_run = G._run

    def spy(cmd, cwd=None, env=None, timeout=900.0):
        argv.append([str(c) for c in cmd])
        return real_run(cmd, cwd=cwd, env=env, timeout=timeout)

    monkeypatch.setattr(G, "_run", spy)
    try:
        g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    suite = [c for c in argv if c[1:3] == ["-m", "pytest"] and c[-1] == "not live_vault"]
    repeats = [c for c in argv
               if c[1:3] == ["-m", "pytest"] and "--continue-on-collection-errors" not in c
               and c != suite[0]]
    assert repeats, "the candidate failures were not re-run at all"
    assert len(repeats) <= 3, f"unbounded repeats: {len(repeats)}"
    assert len(suite) == 1, "the whole suite was run more than once"
    for c in repeats:
        nodes = [a for a in c if "::" in a]
        assert nodes, f"a repeat run named no node: {c}"
        assert not [a for a in c if a.endswith(".py")], f"a repeat run took files: {c}"
        assert set(nodes) <= {FLAKY_NODE, BROKEN_NODE}, c
    # The first repeat names both new nodes; every later one drops the node that
    # already passed.
    assert set(a for a in repeats[0] if "::" in a) == {FLAKY_NODE, BROKEN_NODE}
    if len(repeats) > 1:
        assert set(a for a in repeats[1] if "::" in a) == {BROKEN_NODE}


def test_a_file_that_will_not_collect_is_the_rounds_without_a_repeat(tmp_path, monkeypatch):
    """An id with no `::` is a file pytest could not collect, and repeating it
    would re-run a whole file to ask one question — of a failure that is an import
    or syntax error far more often than an assertion is. It stays the round's on
    the first sample, with no repeat batch at all."""
    r = tmp_path / "live"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_green.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(G, "_gate_cfg",
                        lambda key, default: 1 if key == "test_workers" else default)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_UNC" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(r, "worktree", "add", "-q", "-b", "automod/SM_UNC", str(wt), base)
    (wt / "tests" / "test_uncollectable.py").write_text(
        "this is not python\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "a module that cannot be imported")
    argv: list[list[str]] = []
    real_run = G._run

    def spy(cmd, cwd=None, env=None, timeout=900.0):
        argv.append([str(c) for c in cmd])
        return real_run(cmd, cwd=cwd, env=env, timeout=timeout)

    monkeypatch.setattr(G, "_run", spy)
    g = _gate_for(r, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(r, "worktree", "remove", "--force", str(wt))
    assert ok is False and not data.get("external_blocker")
    assert "tests/test_uncollectable.py" in data["new_failures"], data
    assert "retried_node_ids" not in data, data
    assert "are new in this round" in detail, detail
    # One pytest invocation: the candidate suite. The file is new in this round,
    # so the base probe skips it (it cannot exist there), and a repeat run would
    # have been the second.
    assert len([c for c in argv if c[1:3] == ["-m", "pytest"]]) == 1, argv


def test_a_repeat_run_that_cannot_run_pytest_grants_no_pass(tmp_path):
    """`pytest did not name it as failing` is only evidence when pytest ran. A
    missing interpreter, a usage error or an empty summary stays inconclusive."""
    stub = tmp_path / "py.sh"
    stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ARGV_LOG\"\n"
                    "echo 'no tests ran in 0.00s'\nexit 5\n", encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "argv.log"
    flaky, note = G._reconfirm_candidate_failures(
        stub, tmp_path, ["tests/a.py::t"],
        env={"ARGV_LOG": str(log), "PATH": "/usr/bin:/bin"})
    assert flaky == [], "an empty summary is not a pass"
    assert "INCONCLUSIVE" in note, note
    assert len(log.read_text().splitlines()) == 1, "an inconclusive repeat stops the loop"


def test_a_collection_error_on_a_repeat_is_not_a_pass(tmp_path):
    """`ERROR tests/x.py` names the file, not the node inside it. A node whose
    file cannot even be collected has not passed."""
    stub = tmp_path / "py.sh"
    stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ARGV_LOG\"\n"
                    "echo 'collected 1 item'\n"
                    "echo 'ERROR tests/x.py - SyntaxError: invalid syntax'\n"
                    "echo '1 error in 0.10s'\nexit 1\n", encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "argv.log"
    flaky, note = G._reconfirm_candidate_failures(
        stub, tmp_path, ["tests/x.py::test_y"],
        env={"ARGV_LOG": str(log), "PATH": "/usr/bin:/bin"})
    assert flaky == [], note
    assert "INCONCLUSIVE" not in note, note
    assert len(log.read_text().splitlines()) == 3, "it kept asking; it just never passed"


def test_the_repeats_are_bounded_by_a_broken_tree(tmp_path):
    """More failing nodes than `REPEAT_MAX_NODES` is a broken tree, not a flake,
    and re-running that many nodes 3 times is minutes spent proving what the
    counts already say."""
    many = [f"tests/a.py::t{i}" for i in range(G.REPEAT_MAX_NODES + 1)]
    flaky, note = G._reconfirm_candidate_failures(
        tmp_path / "never-called", tmp_path, many, env={})
    assert flaky == [] and "broken tree" in note, note


def test_repeats_can_be_turned_off_by_config(tmp_path):
    """k=0 restores the old attribution exactly — the mechanism must be
    switchable off without a code round."""
    flaky, note = G._reconfirm_candidate_failures(
        tmp_path / "never-called", tmp_path, ["tests/a.py::t"], repeats=0, env={})
    assert flaky == [] and "off" in note, note


def test_a_round_that_breaks_a_green_tree_is_told_which_tests_by_name(tmp_path, monkeypatch):
    """#1093: this branch used to return only the last 900 characters of
    pytest's output, which began mid-name. The round that read it invented a
    test file and filed a blocker against the gate. Every new failure is named,
    whole, ahead of the tail."""
    r = tmp_path / "live"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_green.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_T4" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(r, "worktree", "add", "-q", "-b", "automod/SM_T4", str(wt), base)
    (wt / "tests" / "test_green.py").write_text(
        "def test_ok():\n    assert True\n\n"
        "def test_broken_by_the_round_one():\n    raise TypeError('x' * 400)\n\n"
        "def test_broken_by_the_round_two():\n    raise TypeError('y' * 400)\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "breaks two tests")

    g = _gate_for(r, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    ok, detail, data = g.rung_tests()

    assert ok is False and not data.get("external_blocker")
    head = detail.split("\n", 1)[0]
    assert "all 2 failure(s) are new in this round" in head
    for nid in ("tests/test_green.py::test_broken_by_the_round_one",
                "tests/test_green.py::test_broken_by_the_round_two"):
        assert nid in head, nid
    git(r, "worktree", "remove", "--force", str(wt))


def test_named_ids_are_whole_and_capped():
    ids = [f"tests/test_x.py::test_{i}" for i in range(25)]
    text = G._name_ids(ids)
    assert "tests/test_x.py::test_19" in text and "tests/test_x.py::test_20" not in text
    assert text.endswith("+5 more (all in data.failed_node_ids)")
    assert G._name_ids(ids[:2]) == "tests/test_x.py::test_0, tests/test_x.py::test_1"


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
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_T3", str(wt), base)
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
    git(live_repo, "worktree", "add", "-q", "-b", f"automod/{name}", str(wt), base)
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
    git(live, "worktree", "add", "-q", "-b", "automod/SM_RT", str(wt), base)
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


# ── vitest in the frontend rung (2026-09-17) ────────────────────────────────

def test_frontend_rung_fails_when_the_unit_tests_fail(live_repo, tmp_path, monkeypatch):
    """`web/` had no test runner until #1199 needed one: a frontend clause had
    no node a gate could grade. A failing vitest run now fails the rung."""
    g, _ = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"], head={}, base={})
    monkeypatch.setattr(G, "_vitest_run", lambda web, timeout=600: (False, "1 failed | 5 passed"))
    ok, reason, _ = g.rung_frontend()
    assert not ok and "vitest failed" in reason and "1 failed" in reason
    monkeypatch.setattr(G, "_vitest_run", lambda web, timeout=600: (True, ""))
    ok, reason, data = g.rung_frontend()
    assert ok and "vitest ok" in reason and data["vitest"] is True


def test_a_box_without_vitest_installed_skips_and_says_so(live_repo, tmp_path, monkeypatch):
    """`node_modules` is untracked. A tree that gained vitest in package.json
    before anyone ran `npm install` must not fail every frontend round — and
    must not read as tested either."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/x.tsx"], head={}, base={})
    ok, reason, data = g.rung_frontend()
    assert ok and "vitest SKIPPED (no web/vitest.config.ts)" in reason and data["vitest"] is None
    (wt / "web" / "vitest.config.ts").write_text("export default {}\n")
    ok, reason, data = g.rung_frontend()
    assert ok and "vitest is not installed" in reason and "npm install" in data["vitest_skipped"]


def test_the_web_tree_declares_the_runner_the_rung_looks_for():
    import json as _json
    web = Path(__file__).resolve().parent.parent / "web"
    pkg = _json.loads((web / "package.json").read_text())
    assert "vitest" in pkg["devDependencies"] and pkg["scripts"]["test"] == "vitest run"
    assert (web / "vitest.config.ts").exists()
    assert list((web / "src").rglob("*.test.ts")), "a runner with nothing to run passes nothing"
