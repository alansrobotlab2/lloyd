"""Gate mechanics: the ladder, its short-circuit, and its fail-closed rule.

The property worth the most here is `test_a_rung_that_errors_is_a_failed_rung`.
With no human review tier, a rung that raises and is read as "didn't fail"
would silently remove a check from the only thing standing between a proposal
and production.
"""

from __future__ import annotations

import os
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
# `vet` sits immediately behind `preflight`: it reads the same two things the
# scope check just enumerated (the resolved base, the changed-path list), costs
# about as much, and is the only rung that judges the change set itself instead
# of the behaviour it produces — so it runs before anything executes the
# candidate. It is observe-only (#679): it records, it never fails.
RUNGS = ("preflight", "vet", "static", "frontend", "tests", "prompt_surface",
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
    """An import error must cost 3 seconds, not a full canary boot.

    `vet` is in the called list because #679's clause 5 puts it on every round
    immediately behind `preflight`, which is before anything executes the
    candidate: an emptied file or a stray blob is reported even on a tree that
    then fails `static`.
    """
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    outcomes = dict(ALL_PASS)
    outcomes["static"] = (False, "import smoke failed", {})
    g = _StubGate(outcomes)
    report = g.run()
    assert not report.ok
    assert g.called == ["preflight", "vet", "static"]
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


def test_a_tests_pass_carries_its_pre_existing_failures_onto_the_ledger(monkeypatch):
    """The event, not only `gate.json`: the scorecard counts red-tree passes
    and the next round is told which ids are not its own, long after the round
    dir is gone. Lists are capped; `external_blocker` is not written."""
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, *a, **k: events.append(e))
    ids = [f"tests/test_x.py::t{i}" for i in range(60)]
    outcomes = dict(ALL_PASS)
    outcomes["tests"] = (True, "tests pass on this diff", {
        "pre_existing_failures": ids, "flaky_node_ids": ["tests/test_y.py::f"],
        "red_tree_item": 1500})
    assert _StubGate(outcomes).run().ok
    ev = next(e for e in events if e["rung"] == "tests")
    assert ev["pre_existing_failures"] == ids[:50]
    assert ev["flaky_node_ids"] == ["tests/test_y.py::f"]
    assert ev["red_tree_item"] == 1500
    assert "external_blocker" not in ev


def test_a_skipped_optional_rung_writes_no_ledger_row_but_stays_in_the_report(monkeypatch):
    """~1,900 zero-second rows a week said only "not this round". `gate.json`
    keeps every rung; the ledger keeps what something reads — and `drill` is
    always written, skipped or not, because a full pass is keyed on it
    (`backlog._last_gate_per_round`, `gate_passed_unlanded_rounds`)."""
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, *a, **k: events.append(e))
    outcomes = dict(ALL_PASS)
    for name in ("frontend", "prompt_surface", "venv", "canary_smoke", "drill", "review"):
        outcomes[name] = (True, "skipped", {"skipped": True})
    report = _StubGate(outcomes).run()
    assert report.ok and [r.name for r in report.rungs] == list(RUNGS)
    written = [e["rung"] for e in events]
    for quiet in ("frontend", "prompt_surface", "venv", "canary_smoke"):
        assert quiet not in written, quiet
    assert "drill" in written and next(e for e in events if e["rung"] == "drill")["skipped"] is True
    assert "review" in written, "only the four optional rungs go quiet"
    # A rung that RAN is written as always.
    outcomes["frontend"] = (True, "tsc delta 0, vite build ok", {})
    events.clear()
    _StubGate(outcomes).run()
    assert "frontend" in [e["rung"] for e in events]


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


@pytest.fixture(autouse=True)
def _private_red_set(tmp_path, monkeypatch):
    """Each test its own per-base red set. Two throwaway repos committed in the
    same second by the same author have the same base sha, and a red set one
    test recorded would skip another's base probe."""
    monkeypatch.setattr(G.S, "RED_SET_PATH", tmp_path / "red_set.json")


def _small_floors(monkeypatch):
    """The whole-suite floors, scaled to a two-test throwaway repo. A pass
    with pre-existing failures goes through the same floors as a green one;
    these tests are about attribution, so the floors are lowered, not
    skipped (`test_a_pass_over_pre_existing_failures_still_meets_the_floors`
    is the one that leaves them at production values)."""
    monkeypatch.setattr(G, "PYTEST_MIN_COLLECTED", 1)
    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 1)


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


def _repo_with_data_root_sensitive_test(tmp_path):
    """A test that fails only when its data root carries `poison` — the shape
    of a store the candidate's own tests provisioned into the round data root
    (#1436's 0-row kg.sqlite), which the base probe then found "at base"."""
    r = tmp_path / "live"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_data.py").write_text(
        "import os\nfrom pathlib import Path\n\n"
        "def test_data_root_is_clean():\n"
        "    assert not (Path(os.environ['LLOYD_DATA']) / 'poison').exists()\n",
        encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r, git(r, "rev-parse", "HEAD").stdout.strip()


def test_the_base_probe_does_not_inherit_the_rounds_data_root(tmp_path):
    """#1436: the round's tests rung left `poison` in the round data root and
    the same env reached the probe, so a failure the candidate caused
    reproduced "at base" and was excused as pre-existing. The probe must run
    against a data root the candidate run cannot have written."""
    repo, base = _repo_with_data_root_sensitive_test(tmp_path)
    round_data = tmp_path / "round-home" / "lloyd-data"
    (round_data / "poison").mkdir(parents=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "round-home"),
           "LLOYD_DATA": str(round_data)}
    scratch = tmp_path / "scratch"

    failed, note = G._failures_at_base(
        Path(sys.executable), repo, base,
        ["tests/test_data.py::test_data_root_is_clean"], scratch, env)

    assert failed == set(), f"the round's residue reproduced at base: {note}"
    assert "0 already failing" in note and "fresh data root" in note
    external, new = G._classify_test_failure(
        ["tests/test_data.py::test_data_root_is_clean"], failed)
    assert external is False and new == ["tests/test_data.py::test_data_root_is_clean"]
    # The round's own root is untouched — the tests rung's evidence stays —
    # and the probe's is gone with its worktree.
    assert (round_data / "poison").is_dir()
    assert not (scratch / "baseline-data").exists()


def test_a_probe_without_a_data_root_in_its_env_sets_none(tmp_path):
    """An env-less probe (the module-level tests above) is unchanged: no
    `LLOYD_DATA` is invented, and the note does not claim a fresh root."""
    repo, base = _repo_with_failing_test(tmp_path)
    failed, note = G._failures_at_base(
        Path(sys.executable), repo, base,
        ["tests/test_pre.py::test_already_broken"], tmp_path / "scratch",
        {"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    assert failed == {"tests/test_pre.py::test_already_broken"}
    assert "fresh data root" not in note


def test_rung_tests_reports_pre_existing_breakage_as_an_external_blocker(tmp_path, monkeypatch):
    """End to end through the real rung: a tree that was already red, plus a
    round that changed something unrelated. The rung still FAILS — landing onto
    a red tree would hand the guardian's observation window a broken baseline —
    but it says whose fault it is, and `backlog.implemented_ids` reads that."""


def _round_over_red_tree(tmp_path, monkeypatch, name, *, edit=("unrelated.py", "X = 1\n")):
    repo, base = _repo_with_failing_test(tmp_path)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / name / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", f"automod/{name}", str(wt), base)
    (wt / edit[0]).write_text(edit[1], encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "a change")
    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    return g, repo, wt, base


def test_rung_tests_passes_over_pre_existing_breakage_and_records_it(tmp_path, monkeypatch):
    """End to end through the real rung: a tree that was already red, plus a
    round that changed something unrelated. The rung PASSES — the failure is
    not this diff's, and refusing it killed 157 rounds in a week — and records
    whose failure it is. No `external_blocker`: the item's attempt is not in
    question, because the round is not stopped."""
    _small_floors(monkeypatch)
    g, repo, wt, base = _round_over_red_tree(tmp_path, monkeypatch, "SM_T")
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))

    assert ok is True, detail
    assert data["pre_existing_failures"] == ["tests/test_pre.py::test_already_broken"]
    assert "external_blocker" not in data and "external_failures" not in data
    assert "PRE-EXISTING" in detail and base[:8] in detail
    assert data["base_probe"].startswith("probed "), data["base_probe"]
    assert data["red_set_cached"] is False
    assert data["failed"] == 1, "the failure is counted, not excised"
    # A throwaway repo is not the production tree: no board write.
    assert "red_tree_item" not in data


def test_a_pass_over_pre_existing_failures_still_meets_the_floors(tmp_path, monkeypatch):
    """The pass tail is shared: a red-tree pass is held to the collected /
    passed / skipped floors a green run is. Two tests collected is a suite
    that was deleted, whoever's the failure is."""
    g, repo, wt, _base = _round_over_red_tree(tmp_path, monkeypatch, "SM_FL")
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False and "tests collected" in detail, detail
    assert data["pre_existing_failures"] == ["tests/test_pre.py::test_already_broken"]


def test_a_red_test_file_the_round_touches_is_the_rounds(tmp_path, monkeypatch):
    """The touched-file rule: a round that edits a red test file and leaves it
    red owns it, even though the same node also fails at base. The base probe
    is not even asked about it."""
    _small_floors(monkeypatch)
    src = ("def test_already_broken():\n    assert False\n\n"
           "def test_fine():\n    assert True  # touched by the round\n")
    g, repo, wt, _base = _round_over_red_tree(tmp_path, monkeypatch, "SM_TCH",
                                              edit=("tests/test_pre.py", src))
    probed: list = []
    real = G._failures_at_base
    monkeypatch.setattr(G, "_failures_at_base",
                        lambda *a, **k: probed.append(list(a[3])) or real(*a, **k))
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False, detail
    assert data["new_failures"] == ["tests/test_pre.py::test_already_broken"]
    assert data["touched_failures"] == ["tests/test_pre.py::test_already_broken"]
    assert "pre_existing_failures" not in data
    assert probed == [], "a node in the round's own file is never probed"


def test_an_inconclusive_base_probe_grants_nothing(tmp_path, monkeypatch):
    """Fail closed: a probe that could not run says nothing about whose
    failure this is — not even when every node then passes a repeat run."""
    _small_floors(monkeypatch)
    g, repo, wt, _base = _round_over_red_tree(tmp_path, monkeypatch, "SM_INC")
    monkeypatch.setattr(G, "_failures_at_base", lambda *a, **k: (
        set(), "baseline probe INCONCLUSIVE at base x — pytest produced no summary"))
    monkeypatch.setattr(G, "_reconfirm_candidate_failures",
                        lambda python, root, ids, **k: (list(ids), "all passed a repeat"))
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False and "failing closed" in detail, detail
    assert "pre_existing_failures" not in data
    assert G.S.read_red_set(g.base) is None, "an inconclusive probe is never cached"


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
    # The mixed case still says which failures are NOT the round's, so the
    # author does not spend the fix cycle on them.
    assert data["pre_existing_failures"] == ["tests/test_pre.py::test_already_broken"]
    assert "not yours" in detail
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
    _small_floors(monkeypatch)
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
    # `**kw` so the stub follows the real signature rather than pinning one
    # moment of it: `isolate_home` arrived on 2026-09-22 and these four tests
    # failed on the keyword, not on anything they are about.
    monkeypatch.setattr(g, "_child_env",
                        lambda root=None, **kw: {**real_env(root, **kw),
                                                 "FLAKE_RUNS_FILE": str(counter)})
    return g, counter, repo, wt


def test_a_failure_that_passes_its_repeat_runs_is_flaky_not_the_rounds(tmp_path, monkeypatch):
    """The #1196 case. One suite run saw the failure, the base probe saw a pass,
    and the round that happened to trip it used to be blamed and spend its
    attempt. Since 2026-09-24 a flake is not a reason to stop the round at all:
    the rung passes and names it."""
    g, counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=False)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is True, detail
    assert "external_blocker" not in data
    assert data["retried_node_ids"] == [FLAKY_NODE]
    assert data["flaky_node_ids"] == [FLAKY_NODE]
    assert "FLAKY" in detail, detail
    assert "are new in this round" not in detail, detail
    # suite run + base probe + ONE repeat, then the node left the batch
    assert int((Path(str(counter) + ".first")).read_text()) == 3, detail


def test_a_flaky_pass_still_counts_the_failure(tmp_path, monkeypatch):
    """The pass is attribution, not excision. No skip, no xfail, no node
    dropped from the run: the rung's own counts still say the suite ran
    everything and one test came back red."""
    g, _counter, repo, wt = _counted_gate(tmp_path, monkeypatch, add_broken=False)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is True
    assert data["failed"] == 1, "the failure is still counted, not excised"
    assert data["collected"] == 2, "the flaky node was run, not skipped out of it"
    assert data["tests_skipped"] == 0 and data["xfailed"] == 0


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
    suite = [c for c in argv if c[1:3] == ["-m", "pytest"] and c[-1] == G.TESTS_MARK_EXPR]
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


def test_a_test_the_round_added_does_not_hide_the_pre_existing_ones(tmp_path):
    """Regression, found building the base probe: probing by node id is wrong.
    Handed a node id that does not exist at base — a test the round just
    wrote, in a file that already existed — pytest exits `ERROR: not found:`
    and runs NOTHING, so the pre-existing failure beside it never reports and
    a red tree reads as green. The probe runs whole files for exactly this
    reason. (Through the rung a red file the round edits is now the round's
    own and never probed — `test_a_red_test_file_the_round_touches_is_the_rounds`
    — so this is pinned on the probe itself.)"""
    repo, base = _repo_with_failing_test(tmp_path)
    failed, note = G._failures_at_base(
        Path(sys.executable), repo, base,
        ["tests/test_pre.py::test_added_by_this_round",
         "tests/test_pre.py::test_already_broken"], tmp_path / "scratch")
    assert failed == {"tests/test_pre.py::test_already_broken"}, note


def test_a_round_that_adds_a_failure_to_a_red_file_owns_both(tmp_path, monkeypatch):
    """The same shape through the rung: the file is in the round's diff, so
    both of its failures are the round's."""
    _small_floors(monkeypatch)
    src = ("def test_already_broken():\n    assert False\n\n"
           "def test_fine():\n    assert True\n\n"
           "def test_added_by_this_round():\n    assert False\n")
    g, repo, wt, _base = _round_over_red_tree(tmp_path, monkeypatch, "SM_T3",
                                              edit=("tests/test_pre.py", src))
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))
    assert ok is False
    assert not data.get("external_blocker"), "the round added a failure of its own"
    assert set(data["new_failures"]) == {"tests/test_pre.py::test_added_by_this_round",
                                         "tests/test_pre.py::test_already_broken"}
    assert "all 2 failure(s) are new" in detail


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
    gate says which file, before the pool is paused for a landing.

    #1038 added the three assertions at the end. The refusal used to end "commit
    or stash the live edit, then gate again", and that second option is a hazard
    wearing a fix: the live checkout's stash stack is ONE global LIFO list shared
    by every author, so obeying the sentence has already destroyed work — on
    2026-09-11 a round implementing an unrelated item popped #573's recovered
    136-line diff out of it, leaving a hand-written re-stash as the only copy.
    Naming EVERY overlapping path is the other half of the pin: one path plus an
    implication is what sends a reader to the wrong editor when two people are
    mid-edit, which is exactly the case the refusal exists for."""
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = _round(live_repo, tmp_path, "r5", base)
    (wt / "app" / "second.py").write_text("S = 2\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "round r5 also changes app/second.py")
    (live_repo / "app" / "m.py").write_text("V = 1  # being edited live\n", encoding="utf-8")
    (live_repo / "app" / "second.py").write_text("S = 'theirs'\n", encoding="utf-8")

    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, detail, data = g.rung_preflight()
    assert ok is False
    assert data.get("external_blocker") is True
    assert data["overlap"] == ["app/m.py", "app/second.py"]
    assert all(p in detail for p in data["overlap"]), \
        f"the refusal names only some of the two-writer paths: {detail}"
    assert "stash" not in detail.lower(), (
        "the live tree's stash stack is shared by every author, so this refusal "
        f"may only ask for a report. Got: {detail}")


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


# ---------------------------------------------------------------------------
# The observe-only vet rung (#679)
# ---------------------------------------------------------------------------

def _vet_round(tmp_path, monkeypatch, *, mode: str = "empty"):
    """A real Gate over a scratch repo, with a head commit of the given `mode`.

    `mode="empty"` empties one non-empty tracked file (a violation);
    `mode="clean"` adds an ordinary text file (a real change set the vet must
    call clean — not the vacuous zero-file diff of a round that committed
    nothing). `app/__init__.py` is committed empty at base in both, because the
    empty check's whole discipline is the file that is *supposed* to be empty.

    Returns (gate, captured ledger events). The round dir is redirected into
    `tmp_path` so the rung cache cannot touch the live state directory, and
    `append_event` is captured rather than written.
    """
    root = tmp_path / "wt"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    for k, v in (("user.email", "vet@example.invalid"), ("user.name", "vet")):
        subprocess.run(["git", "-C", str(root), "config", k, v], check=True)
    (root / "app").mkdir()
    (root / "app" / "service.py").write_text("def handle(e):\n    return e\n")
    (root / "app" / "__init__.py").write_text("")          # legitimately empty
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "base"], check=True)
    base = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if mode == "empty":
        (root / "app" / "service.py").write_text("")
        subject = "clobbered"
    elif mode == "clean":
        (root / "app" / "new_text.py").write_text("X = 1\n")
        subject = "an ordinary change"
    else:
        raise AssertionError(f"unknown mode {mode!r}")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", subject], check=True)

    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", events.append)
    monkeypatch.setattr(G.W, "round_dir", lambda *a, **k: tmp_path / "round")
    g = G.Gate("SM_VET_ROUND", root, base)
    return g, events


def test_the_ladder_runs_the_vet_records_its_findings_and_blocks_nothing(tmp_path, monkeypatch):
    """Clause 5, end to end: the real ladder, the real rung, a real git repo.

    Only `vet` is the real method; the other ten rungs are stubbed, because what
    is under test is that the ladder itself calls the vet on every round, that
    the finding lands in the round's gate report, and that a violation cannot
    turn the round red — the soak that decides whether it may block has not run.
    """
    g, events = _vet_round(tmp_path, monkeypatch)
    for name in ("preflight", "static", "frontend", "tests", "prompt_surface",
                 "review", "venv", "canary_boot", "canary_smoke", "drill"):
        monkeypatch.setattr(g, f"rung_{name}", lambda n=name: (True, f"stub {n}", {}))

    report = g.run()

    assert report.ok, "an observe-only rung must never decide a round"
    vet_rungs = [r for r in report.rungs if r.name == "vet"]
    assert len(vet_rungs) == 1, [r.name for r in report.rungs]
    rung = vet_rungs[0]
    assert rung.ok is True
    assert rung.detail.startswith("observe-only:"), rung.detail
    assert [v["path"] for v in rung.data["vet"]["violations"]] == ["app/service.py"]
    assert rung.data["vet"]["counts"] == {"empty_file": 1}
    assert rung.data["vet"]["labels"] == ["empty_file:app/service.py"]
    assert rung.data["vet"]["observe_only"] is True
    # Recorded where the soak will read it: the event survives the worktree,
    # gate.json does not.
    ev = [e for e in events if e.get("rung") == "vet"]
    assert len(ev) == 1, events
    assert ev[0]["ok"] is True
    assert ev[0]["vet"]["counts"] == {"empty_file": 1}
    assert ev[0]["vet"]["labels"] == ["empty_file:app/service.py"]
    assert "observe-only" in ev[0]["detail"]


def test_a_clean_round_records_the_vet_ran_rather_than_recording_nothing(tmp_path, monkeypatch):
    """The soak's denominator is "the vet ran", which needs a green record too.

    A rung that recorded only findings would make an unexecuted `vet`
    indistinguishable from a clean one across a month of landings.
    """
    g, events = _vet_round(tmp_path, monkeypatch, mode="clean")
    ok, detail, data = g.rung_vet()
    assert ok is True
    assert detail.startswith("observe-only: clean"), detail
    assert data["vet"]["violations"] == []
    assert data["vet"]["totals"]["files"] >= 1
    assert data["vet"]["totals"]["max_diff_lines"] == 12_000
    # `observe_only: True` on a clean record, not just on a dirty one: the soak
    # counts landings, and a count that only appears with findings cannot be a
    # denominator.
    assert data["vet"]["observe_only"] is True
    assert data["vet"]["labels"] == []

    # And on the ladder it is a rung every round, not one that appears when it
    # has something to say. Every other rung is stubbed — the real `static` here
    # would import the candidate, which is not this test's subject.
    for name in ("preflight", "static", "frontend", "tests", "prompt_surface",
                 "review", "venv", "canary_boot", "canary_smoke", "drill"):
        monkeypatch.setattr(g, f"rung_{name}", lambda n=name: (True, f"stub {n}", {}))
    report = g.run()
    assert [r.name for r in report.rungs if r.name == "vet"] == ["vet"]
    clean_ev = [e for e in events if e.get("rung") == "vet"]
    assert len(clean_ev) == 1 and clean_ev[0]["vet"]["counts"] == {}


def test_a_vet_that_could_not_run_is_recorded_as_unevaluated_not_clean(tmp_path, monkeypatch):
    """Clause 4 at the seam that consumes it: the rung keeps the distinction."""
    g, events = _vet_round(tmp_path, monkeypatch)
    monkeypatch.setattr(G.V, "vet_change_set",
                        lambda *a, **k: G.V.VetResult(status=G.V.UNEVALUATED,
                                                      reason="git ls-tree failed"))
    ok, detail, data = g.rung_vet()
    assert ok is True, "an unevaluated vet is a finding about the gate, not a red round"
    assert data["vet"]["status"] == G.V.UNEVALUATED
    assert data["vet"]["violations"] == []
    assert "UNEVALUATED" in detail and "git ls-tree failed" in detail
    assert "clean" not in detail, "an unread change set must never read as clean"


# ---------------------------------------------------------------------------
# The review rung's own parse boundary: an unreadable verdict is the rail, not
# a judgment of the diff (#1443)
# ---------------------------------------------------------------------------

# The shape a grader actually returned on SM_20260916_032218, SM_20260922_100227
# and SM_20260924_104224: every clause graded `met`, but keyed `id`/`status`
# instead of `clause`/`verdict`, so not one clause index was readable.
_ALIAS_CLAUSES = [
    {"id": 1, "status": "met", "evidence_path": "app/x.py", "evidence_line": 3,
     "test_node_id": "tests/test_x.py::test_one", "how_verified": "ran", "note": "ran it"},
    {"id": 2, "status": "met", "evidence_path": "app/x.py", "evidence_line": 9,
     "test_node_id": "tests/test_x.py::test_two", "how_verified": "ran", "note": "ran it"},
]


class _ReviewGate(G.Gate):
    """Enough state for `rung_review` to run against a stubbed grader."""

    def __init__(self, tmp_path):
        super().__init__("SM_UNREAD", tmp_path, "a" * 40, item_id=7)
        self.report.changed_paths = ["app/x.py", "tests/test_x.py"]
        self.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))


def _stub_grader(monkeypatch, tmp_path, structured):
    """Record the rung's ledger events and the grading turns it spent."""
    from scripts.automod import review as RV
    events: list[dict] = []
    calls: list[dict] = []

    def grade(**kw):
        calls.append(kw)
        return {"ok": True, "error": "", "session_id": "sess_r", "structured": structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}

    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [])
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b",
        "clauses": ["clause one holds", "clause two holds"], "path": ""})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    return events, calls


def test_clause_entries_with_no_usable_index_are_an_external_rail_failure(tmp_path, monkeypatch):
    """Zero readable clause indexes is the rail failing, not the round refused.

    The three rounds this exists for each lost one review attempt of two to it,
    while the second reader had in fact approved every clause. An unreadable
    verdict must land on the same side of the ledger as an unreachable grader:
    the rung fails, the event is non-blocking, and nothing is charged.
    """
    obj = {"premise": "sound", "summary": "APPROVE — all four clauses met",
           "clauses": _ALIAS_CLAUSES, "test_honesty": [], "seams_unverified": []}
    events, calls = _stub_grader(monkeypatch, tmp_path, obj)

    ok, detail, data = _ReviewGate(tmp_path).rung_review()
    assert ok is False, "an unreadable review is never a pass"
    assert data["external_blocker"] is True, "the rail, not the diff"
    assert "review_retry" not in data and "review_attempt" not in data, \
        "a rail failure must not arrive as a refusal that names an attempt"
    ev = events[-1]
    assert ev["event"] == "review" and ev["ok"] is False and ev["blocking"] is False, ev

    # And the attempt really is not spent: refusal accounting counts graded
    # (`ok`) events only, so a re-gate over the recorded rail failure still
    # grades rather than reporting exhaustion.
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(events))
    ok2, detail2, data2 = _ReviewGate(tmp_path).rung_review()
    assert ok2 is False and data2["external_blocker"] is True
    assert "abort and report" not in detail2, detail2
    assert len(calls) == 2, "the grader is asked again, not written off"


def test_the_unreadable_verdict_names_the_grader_keys_not_a_refusal_to_act_on(tmp_path, monkeypatch):
    """The author reading `gate.json` has to see that nothing was graded.

    "clause N partial: not addressed by the grader" beside "fix what it names"
    is a contradiction: it sends an author to change four clauses the grader
    never judged. The finding names the keys the grader really used instead.
    """
    obj = {"premise": "sound", "summary": "APPROVE", "clauses": _ALIAS_CLAUSES,
           "test_honesty": [], "seams_unverified": []}
    _stub_grader(monkeypatch, tmp_path, obj)

    ok, detail, data = _ReviewGate(tmp_path).rung_review()
    assert ok is False
    assert "could not be read" in detail, detail
    for key in ("id", "status", "evidence_path", "test_node_id", "how_verified"):
        assert key in detail, f"the finding must name the key it found: {key} not in {detail}"
    assert "not addressed by the grader" not in detail, detail
    assert "fix what it names" not in detail, detail
    # It says what broke, in terms the author cannot mistake for work.
    assert "none carried a usable 1-based `clause` index" in detail, detail
    assert "not a judgment of the diff" in detail, detail
    assert data["external_blocker"] is True and data["review_session"] == "sess_r"
