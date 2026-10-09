"""Gate mechanics: the ladder, its short-circuit, and its fail-closed rule.

The property worth the most here is `test_a_rung_that_errors_is_a_failed_rung`.
With no human review tier, a rung that raises and is read as "didn't fail"
would silently remove a check from the only thing standing between a proposal
and production.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.automod import frontend_layout as FL
from scripts.automod import gate as G
from scripts.automod.layout_fixture import API_STUB


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
RUNGS = ("preflight", "vet", "static", "pyright", "frontend", "tests", "prompt_surface",
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


@pytest.mark.parametrize("edit, bucket", [
    (("# the cap is <= 162 lines", "# the cap is one review's diff"), "comment_only"),
    # 2026-10-05: a value under an allowed key lands in the value lane, and
    # arms the drill the way a protected path does.
    (("cap: 400", "cap: 401"), "config_value"),
    # The loop's own switch stays denied, by key.
    (("enabled: true", "enabled: false"), None),
])
def test_preflight_lands_a_comment_only_config_edit_and_no_other(live_repo, tmp_path,
                                                                 monkeypatch, edit, bucket):
    """Both sides are read from git — the base commit and the round's HEAD."""
    (live_repo / "config.yaml").write_text(
        "arch:\n  # the cap is <= 162 lines\n  cap: 400\nautomod:\n  enabled: true\n",
        encoding="utf-8")
    git(live_repo, "add", "-A")
    git(live_repo, "commit", "-q", "-m", "config")
    base = git(live_repo, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    git(live_repo, "worktree", "add", "-q", "-b", "automod/cfg", str(wt), base)
    cfg = wt / "config.yaml"
    cfg.write_text(cfg.read_text().replace(*edit), encoding="utf-8")
    git(wt, "commit", "-q", "-am", "reword")
    g = _gate_for(live_repo, wt, base, monkeypatch)
    ok, reason, data = g.rung_preflight()
    assert ok is (bucket is not None), reason
    if bucket is None:
        assert "denied" in reason and "not only comments" in reason
        assert "`automod.enabled` is under the denied key `automod.enabled`" in reason
        return
    assert data["buckets"][bucket] == ["config.yaml"]
    # What the drill rung reads back: the recorded preflight buckets, so the
    # value lane buys a drill and the comment lane does not.
    g.report.rungs.append(G.RungResult("preflight", True, reason, 0.0, data))
    armed = G.spec.requires_drill(g.report.changed_paths, buckets=g._preflight_buckets())
    assert armed is (bucket == "config_value"), (bucket, data["buckets"])
    if bucket == "config_value":
        assert "config value change → drill required" in reason, reason


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


# ── the runtime probe (#1601): observe-only, and silent about nothing ───────
# `vite build` proves the app COMPILES; it cannot see a component that throws on
# mount, because a production bundle turns that into a browser-side error. The
# rung now loads the build output it just made in a headless browser and RECORDS
# what happened. Three properties these nodes pin, and the middle one is the
# whole point of this increment:
#   - the probe is handed the directory the build just wrote, not `:5173`, whose
#     dev server serves the live tree the diff is not in;
#   - a failing verdict never fails the rung, because the false-block rate is
#     still unmeasured (`vet`'s observe-only precedent, and #623's grade-only one);
#   - anything that stops the probe being OBSERVED is named in the detail, since
#     a check whose absence reads as silence teaches the next reader to trust the
#     absence — the ledger shape #1028 is full of.
# The probe's own detection is tested with a real chromium in
# tests/test_automod_frontend_probe.py; what is stubbed here is the rung's wiring
# around a verdict, plus the two skip paths, which run through the real
# `frontend_probe.run()` and need no browser to be described.

def _probe_verdict(ok=True, checks=(), **extra):
    v = {"ok": ok, "checks": list(checks), "metrics": {"root_children": 2,
                                                       "body_text_length": 412}}
    v.update(extra)
    return v


def _fake_build(marker="<html>built</html>"):
    """A `_vite_build` stand-in that writes into the outDir it is handed, so a
    test can see WHICH directory the rung considers the build output. It writes a
    script as well as the shell, because `frontend_probe` refuses to probe an
    output with neither — that check is the build lying about its output, and a
    test that tripped it would be measuring the wrong skip reason."""
    def build(web, out_dir, timeout=600):
        (out_dir / "index.html").write_text(marker)
        (out_dir / "bundle.js").write_text("console.log('built')\n")
        return True, ""
    return build


def test_the_rung_probes_the_build_it_just_made_and_records_a_failing_verdict_without_failing(
        live_repo, tmp_path, monkeypatch):
    """Clause 4: with a `web/` path changed the rung probes the build output,
    puts the verdict in the detail and the audit record, and still PASSES — an
    unmeasured false-block rate is not worth a stalled landing. The recorded
    directory is the one the fake build wrote into, which is the assertion that
    rules out a probe pointed at something else."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    # What the probe could see AT CALL TIME. The rung `rmtree`s the build dir in
    # its own `finally`, so an assertion on the path AFTER `rung_frontend()`
    # returns can only ever prove a string was passed; the substance of clause 4 is
    # that the probe runs while that directory still holds the build. An outDir
    # handed over empty is a probe silently checking nothing, and it reads exactly
    # like a pass — which is the shape this round was most likely to introduce.
    seen: list = []

    def fake_probe(built_dir):
        d = Path(built_dir)
        seen.append({"had_index": (d / "index.html").exists(),
                     "had_asset": any(x.suffix in (".js", ".css")
                                      for x in d.rglob("*") if x.is_file())})
        return _probe_verdict(ok=False, checks=[{
            "check": "console-error",
            "problem": "a console error was logged: TypeError: boom in App",
            "screenshot": "/tmp/state/frontend_probe/broken.png"}])
    monkeypatch.setattr(g, "_frontend_probe",
                        lambda d, changed_web=None: fake_probe(d))
    ok, detail, res = g.rung_frontend()
    assert ok, "observe-only: a failing probe must never refuse the round"
    assert len(seen) == 1, seen
    assert seen[0]["had_index"] and seen[0]["had_asset"], (
        "the probe must receive the build output the rung just produced, while it "
        "still exists — not a path to a directory already cleaned up", seen)
    assert "probe FAILED: 1 check(s) [console-error]" in detail, detail
    assert "boom in App" in detail, "the evidence text belongs in the rung detail too"
    assert res["probe"]["ok"] is False and res["probe"]["checks"][0]["check"] == "console-error"


# ── the layout leg (#2130): the same observe-only contract, one level down ───
# `frontend_probe` proves the build RUNS. It watches load status, the DOM, the
# console and pageerrors, so a stylesheet that is served with a 200 and whose rule
# no longer matches the node it used to style passes it cleanly — the canary's own
# artifact records that seed as `blind` with `"checks": []`. The layout leg is the
# channel for that class, and it inherits the rung's contract in full. What these
# nodes stub is the two heavy modules (`frontend_probe.run`, `frontend_layout.run`)
# around the rung's REAL `_frontend_probe`, because the property under test is what
# the rung does with a verdict, not how a browser measures a box.

def _layout_report(*, checks=(), captured=True, baseline="absent",
                 round_id="SM_T", **extra):
    """A `frontend_layout.run` report: enough shape for the rung's own logic.

    Built through `FL.fingerprint`/`fingerprint_of` rather than written as a dict
    literal, because the rung reads keys off it (`fingerprint`, `checks`,
    `skipped`): a literal that drifted from the real return shape would let the
    rung's baseline call pass on a document the leg never produces.
    """
    view = FL.fingerprint_of(
        {"width": 1280, "sections": [{"head": h, "cw": 728, "sw": 728,
                                      "overflow": 0, "tag": "panel",
                                      "tags": ["panel"], "kidsTotal": 2,
                                      "panels": [["p", 340, 340]], "pills": [],
                                      "clipped": []}
                                     for h in ("System", "Services", "Tokens",
                                               "vLLM engines", "Lloyd agent",
                                               "Subagents & background tasks",
                                               "Automation & work")]},
        width=1280)
    report = {"captured": captured, "widths": [1280], "sections_compared": 7,
              "baseline": baseline, "checks": list(checks), "skipped": "",
              "no_verdict": [], "fingerprint": None if not captured else
              FL.fingerprint({"1280": view}, round_id=round_id)}
    report.update(extra)
    return report


def _layout_check(section="Tokens", attribution="UNTOUCHED"):
    return {"check": f"layout-changed:{section}", "ok": False,
            "section": section, "attribution": attribution,
            "detail": f"layout-changed:{section}: MOVED at 1280px "
                      f"(fields: cw) — section NOT in the round's diff",
            "delta": {"1280": {"cw": {"was": 728, "now": 700}}}}


def _layout_rung(live_repo, tmp_path, monkeypatch, *, probe_verdict, layout):
    """A frontend rung over a fake build, with the two browsers stubbed out and the
    state dir redirected so the leg's artifacts land somewhere throwaway."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch,
                           changed=["web/src/App.tsx"], head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    monkeypatch.setattr(G.S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(G._fe_probe, "run", lambda *a, **k: probe_verdict)
    monkeypatch.setattr(G._fe_layout, "run", lambda *a, **k: layout)
    return g, wt


def test_a_failing_layout_check_is_reported_in_the_detail_and_never_refuses_the_rung(
        live_repo, tmp_path, monkeypatch):
    """Clause 5 of #2130, in both directions it has to hold.

    A layout check that fires is *reported* — named, with its section and its
    attribution, in the rung's own detail line and in the verdict the ledger row
    carries — and the rung stays green. "Whatever the diff" is the half that is easy
    to get wrong by accident: an exception escaping the leg would land in the
    rung-level `except`, which returns `False`, so a broken layout TOOL would block
    a landing on layout through the back door while the checks themselves were
    correctly advisory. That is a separate node below.
    """
    layout = _layout_report(checks=[_layout_check("Tokens")], baseline="compared")
    g, wt = _layout_rung(live_repo, tmp_path, monkeypatch,
                         probe_verdict=_probe_verdict(), layout=layout)
    ok, detail, res = g.rung_frontend()
    assert ok, "record-only: a moved section is a finding, not a refusal"
    assert "LAYOUT MOVED 1 section(s) [Tokens UNTOUCHED" in detail, detail
    assert "1 UNTOUCHED" in detail, "the ripple count is the line's point\n" + detail
    laid = res["probe"]["layout"]["checks"][0]
    assert laid["attribution"] == "UNTOUCHED" and laid["section"] == "Tokens", laid
    assert "probe ok" in detail, "the load probe's own line is not displaced"


def test_a_layout_leg_that_raises_is_a_named_no_capture_and_never_refuses_the_rung(
        live_repo, tmp_path, monkeypatch):
    """The leg crashing is an OBSERVATION that did not happen, named as such.

    The rung's outer `except` returns False, so without the leg's own guard a
    playwright bug, a font that failed to load, or a bad baseline file would fail a
    landing whose diff is fine — an instrument blocking on its own health. And the
    skip has to be *in the detail*: a rung that says "frontend ok" over a leg that
    never ran is the silence #1028's ledger is full of.
    """
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch,
                           changed=["web/src/App.tsx"], head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    monkeypatch.setattr(G.S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(G._fe_probe, "run", lambda *a, **k: _probe_verdict())

    def explode(*a, **k):
        raise RuntimeError("chromium vanished")
    monkeypatch.setattr(G._fe_layout, "run", explode)
    ok, detail, res = g.rung_frontend()
    assert ok, "a broken instrument is not a broken frontend"
    assert "LAYOUT SKIPPED" in detail, detail
    assert "chromium vanished" in detail, "the reason belongs in the line"
    assert res["probe"]["layout"]["captured"] is False
    assert res["probe"]["layout"]["checks"] == [], (
        "a leg that did not run reports no verdicts, neither good nor bad")


def test_a_clean_landing_blesses_the_fingerprint_it_just_measured(
        live_repo, tmp_path, monkeypatch):
    """The pass edge of the bless rule, taken from the rung rather than from
    `maybe_advance` in isolation.

    This is the seam the whole design turns on: `maybe_advance` refuses on a failed
    report on its own, but the authority to bless is the rung's, because the rung is
    the only place that knows the load probe passed as well. Get that wiring backwards
    and the first broken landing re-blesses the baseline to its own broken layout,
    which erases the leg silently: every later round then compares against the break
    and reports the page clean.
    """
    g, wt = _layout_rung(live_repo, tmp_path, monkeypatch,
                         probe_verdict=_probe_verdict(), layout=_layout_report())
    ok, detail, res = g.rung_frontend()
    assert ok
    assert res["probe"]["layout"]["baseline_held"] == "", (
        "an empty hold reason is the one outcome that means it stored", res)
    stored = tmp_path / "state" / "frontend_layout" / "baseline.json"
    assert stored.exists(), "a clean landing blesses the baseline it just measured"
    assert json.loads(stored.read_text())["round_id"] == "SM_T"
    assert (tmp_path / "state" / "frontend_layout" / "SM_T.json").exists(), (
        "and the verdict itself is in the state dir, beside the probe's own records")
    assert not list(wt.rglob("*.layout.json")) and not list(wt.rglob("baseline.json")), (
        "no artifact of the leg lands in the tree under review")


def test_a_landing_whose_load_probe_failed_holds_the_baseline_and_says_so(
        live_repo, tmp_path, monkeypatch):
    """The fail edge, and the reason the rung's own green is not the trigger.

    The load probe is observe-only, so this rung PASSES while its probe reports a
    console error — which is precisely why the bless cannot key off the rung's `ok`.
    A landing that failed a frontend check must not advance the truth the next round
    is compared against, and the hold has to be a stated reason in the ledger row: an
    absent file proves nothing about intent.
    """
    g, wt = _layout_rung(live_repo, tmp_path, monkeypatch,
                         probe_verdict=_probe_verdict(ok=False, checks=[
                             {"check": "console-error",
                              "problem": "a console error was logged: boom"}]),
                         layout=_layout_report())
    ok, detail, res = g.rung_frontend()
    assert ok, "observe-only all the way down: a failing probe is not a refusal"
    assert "probe FAILED: 1 check(s)" in detail, detail
    assert res["probe"]["layout"]["baseline_held"].startswith("held:"), res
    assert "other checks did not pass" in res["probe"]["layout"]["baseline_held"]
    assert not (tmp_path / "state" / "frontend_layout" / "baseline.json").exists(), (
        "the failed landing stored nothing, so the previous baseline still stands")


# ── #2327: the rung's probe has to load the same document the leg fingerprints ─
#
# `maybe_advance` holds the baseline whenever the load probe did not pass
# (`advanced=bool(verdict.get("ok"))`), and the probe as wired until now was
# handed no API stub — so it loaded `Dashboard unavailable:` and reported
# `ok: false` on a build with nothing wrong with it. The six artifacts under
# `frontend_layout/` all read `baseline: "absent"`, and
# `SM_20261006_161018.json` says why: `baseline_held: "held: the frontend rung's
# other checks did not pass"` with `checks: []` and `captured: true`. The leg was
# measuring a populated dashboard two lines away (`FP.serve_build(out_dir,
# api_stub=API_STUB)`) while the probe beside it measured a hollow one. These two
# nodes pin the two halves of that: the same frozen fixture going into the probe,
# and the bless edge that only opens when the probe comes back green.

def test_the_rungs_load_probe_is_handed_the_legs_own_frozen_fixture(
        live_repo, tmp_path, monkeypatch):
    """Clause 2: the `frontend_probe.run` call made by `_frontend_probe` carries
    `api_stub=layout_fixture.API_STUB`, so the rung's probe and the layout leg load
    the same document.

    Identity, not equality: `is` is what says the probe was handed the one frozen
    snapshot rather than a copy of it that a later edit could move apart from the
    leg's. A stub that merely LOOKS like the leg's would let the two drift, which
    is the asymmetry this item exists to close.
    """
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    monkeypatch.setattr(G.S, "STATE_DIR", tmp_path / "state")
    seen: dict = {}

    def fake_probe_run(built_dir, **kw):
        seen.update(kw)
        seen["built_dir"] = Path(built_dir)
        return _probe_verdict()

    monkeypatch.setattr(G._fe_probe, "run", fake_probe_run)
    monkeypatch.setattr(G._fe_layout, "run", lambda *a, **k: _layout_report())
    ok, detail, res = g.rung_frontend()
    assert ok, detail
    assert "built_dir" in seen, "the rung never probed the build it just made"
    assert "api_stub" in seen, (
        "the rung probes with no API stub, so its page is the shell and its verdict "
        "is `Dashboard unavailable:` — the state that held the baseline for six rounds")
    assert seen["api_stub"] is API_STUB, (
        "the rung's probe must load the frozen fixture itself, not a look-alike")
    assert seen["api_stub"] is G._fe_layout.API_STUB, (
        "the leg's own default is where the probe's stub has to come from, or the "
        "two checks are reading two different pages")


def _dashboard_view(width: int = 1280):
    """One width's capture, built by the leg's own projection so the numbers below
    are the leg's, not a literal that could drift from what it returns."""
    return FL.fingerprint_of(
        {"pageOverflow": 0,
         "sections": [{"head": h, "cw": 728, "sw": 728, "overflow": 0,
                       "tag": "panel", "tags": ["panel"], "kidsTotal": 2,
                       "panels": [["p", 340, 340]], "pills": [], "clipped": []}
                      for h in ("System", "Services", "Tokens", "vLLM engines",
                                "Lloyd agent", "Subagents & background tasks",
                                "Automation & work")]},
        width=width)


def test_the_first_green_landing_blesses_the_baseline_the_next_round_reads(
        live_repo, tmp_path, monkeypatch):
    """Clause 5: the bless edge end to end, through the rung.

    Only two things are stubbed here — the browser (`capture_build`) and the load
    probe's verdict. `frontend_layout.run` is the REAL leg, so `baseline:` in the
    second round's report is the leg's own reading of the file the first round
    stored, not a value a mock was handed; and the file it reads is written by the
    real `store_baseline` behind the real `maybe_advance`. That chain is the item:
    an `ok` from the probe is the only thing that has ever stood between a green
    frontend landing and a baseline, and while the probe was red on a hollow page
    nothing downstream of it could run.
    """
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    monkeypatch.setattr(G.S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(G._fe_probe, "run", lambda *a, **k: _probe_verdict())
    monkeypatch.setattr(G._fe_layout, "capture_build",
                        lambda out_dir, **kw: {"views": {"1280": _dashboard_view()}})
    baseline = tmp_path / "state" / "frontend_layout" / "baseline.json"
    assert not baseline.exists(), "the state dir starts with no baseline, as every round to date"

    ok1, detail1, res1 = g.rung_frontend()
    assert ok1, detail1
    assert res1["probe"]["layout"]["baseline"] == "absent", res1["probe"]["layout"]
    assert res1["probe"]["layout"]["baseline_held"] == "", (
        "a green probe plus a clean capture is the one combination that stores", res1)
    assert baseline.is_file(), "the fingerprint was not stored at the leg's baseline path"
    assert FL.baseline_path() == baseline, (
        "the rung wrote somewhere the next round will not read")
    assert json.loads(baseline.read_text())["round_id"] == "SM_T"
    assert "LAYOUT BASELINE ABSENT" in detail1, detail1

    ok2, detail2, res2 = g.rung_frontend()
    assert ok2, detail2
    assert res2["probe"]["layout"]["baseline"] == "present", (
        "the round after a bless must read a baseline, or the leg has no comparator",
        res2["probe"]["layout"])
    assert res2["probe"]["layout"]["baseline_held"] == "", res2["probe"]["layout"]
    assert res2["probe"]["layout"]["sections_compared"] == 7, res2["probe"]["layout"]
    assert "LAYOUT ok (7 section(s) stable" in detail2, detail2


def test_the_rung_records_a_healthy_probe_verdict_in_the_detail(live_repo, tmp_path, monkeypatch):
    """The same path with a clean verdict: `probe ok` and the metrics, so a green
    line says what was checked rather than being indistinguishable from a probe
    that never ran."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    monkeypatch.setattr(g, "_frontend_probe",
                        lambda d, changed_web=None: _probe_verdict())
    ok, detail, res = g.rung_frontend()
    assert ok and "probe ok (2 root children, 412 chars of text" in detail, detail
    assert res["probe"]["ok"] is True and res["probe"]["checks"] == []


def test_the_rung_runs_no_probe_and_keeps_the_recorded_skip_without_a_web_change(
        live_repo, tmp_path, monkeypatch):
    """Clause 4, second half: the probe costs a browser launch, so a round that
    touched no frontend file must not pay it — and the rung's existing named
    `SKIPPED` record has to survive, not become an empty pass."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["app/router.py"],
                           head={}, base={})
    def explode(*_a, **_k):
        raise AssertionError("a non-frontend round must not launch a probe")
    monkeypatch.setattr(g, "_frontend_probe", explode)
    monkeypatch.setattr(G, "_vite_build", explode)
    ok, detail, res = g.rung_frontend()
    # The skip the rung already recorded is kept verbatim — `"no frontend changed"`
    # with `{"skipped": True, "reason": "no web/ path"}` — and the probe's word
    # appears nowhere in it. 93 rounds ran this rung non-skipped and the rest land
    # on the strength of that skip reading as honest; a probe that decorated it
    # would make an unexamined landing look examined.
    assert ok and res.get("skipped") is True and res.get("reason") == "no web/ path"
    assert detail == "no frontend changed", detail
    assert "probe" not in detail.lower(), detail


def test_an_unprobeable_build_output_is_a_named_skip_from_the_real_probe(
        live_repo, tmp_path, monkeypatch):
    """Clause 5, through the real seam: nothing is stubbed below the rung except
    the build, and the build's own output is empty — the case where `vite build`
    exited 0 and left no `index.html`. The rung passes (the build is green and
    that is what it gates) and says out loud that it could not look."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", lambda web, out_dir, timeout=600: (True, ""))
    ok, detail, res = g.rung_frontend()
    assert ok, "an unobservable probe is a skip, never a refusal"
    assert "probe SKIPPED (no build output" in detail or "index.html" in detail, detail
    assert res["probe"]["skipped"], res


def test_a_missing_chromium_is_a_named_probe_skip_and_the_rung_passes(
        live_repo, tmp_path, monkeypatch):
    """Clause 5, the other reason: the build is real, the browser is not. The
    skip must name the browser and its path, because "the box lost chromium" and
    "the build is broken" are different failures told apart at different layers —
    and neither may read as a verified frontend."""
    g, wt = _frontend_gate(live_repo, tmp_path, monkeypatch, changed=["web/src/App.tsx"],
                           head={}, base={})
    monkeypatch.setattr(G, "_vite_build", _fake_build())
    (tmp_path / "index.html")  # build stub writes into the rung's own outDir
    monkeypatch.setattr(G._fe_probe, "CHROMIUM", "/nonexistent/chromium")
    ok, detail, res = g.rung_frontend()
    assert ok, "no browser is an environment gap, not the round's defect"
    assert "probe SKIPPED (no chromium at /nonexistent/chromium)" in detail, detail
    assert "no chromium" in res["probe"]["skipped"]
    assert "ok" not in res["probe"], "a probe that never ran must not report a verdict"


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


# ── #1322: one "is this a test file?" for every site, read from pytest.ini ──

def test_the_testpath_predicate_reads_pytest_ini_and_owns_the_harness_suite(tmp_path):
    """The review rung built its changed-test set with `startswith("tests/")`
    while the tests rung ran bare `pytest`, whose `testpaths` include
    `app/harness/tests` — so every clause pinned there was downgraded
    (SM_20260921_030016, four of five). The list comes from the file, not
    from a second literal."""
    from scripts.automod import testpaths as TP
    root = Path(__file__).resolve().parents[1]
    assert TP.read_testpaths(root) == ("tests", "app/harness/tests", "scripts")
    assert TP.is_test_file("app/harness/tests/test_x.py", root)
    assert TP.is_test_file("app/harness/tests/conftest.py", root)
    assert TP.is_test_file("tests/test_x.py", root)
    assert TP.is_test_file("scripts/test_helper.py", root)
    # `scripts` is walked, not a test tree: its code is code.
    assert not TP.is_test_file("scripts/automod/review.py", root)
    assert not TP.is_test_file("app/harness/loop.py", root)
    assert TP.pick_test_files(["app/harness/loop.py", "app/harness/tests/test_x.py",
                               "tests/test_y.py", "eval/q.yaml"], root) == [
        "app/harness/tests/test_x.py", "tests/test_y.py"]

    # Read, not restated: another pytest.ini gives another answer, and none
    # at all is the root `tests/` reading the tree grew up with.
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = spec\n")
    assert TP.read_testpaths(tmp_path) == ("spec",)
    assert TP.is_test_file("spec/test_x.py", tmp_path)
    assert not TP.is_test_file("tests/test_x.py", tmp_path)
    bare = tmp_path / "bare"; bare.mkdir()
    assert TP.read_testpaths(bare) == ("tests",)
    assert not TP.is_test_file("app/harness/tests/test_x.py", bare)


def test_the_review_rung_builds_its_changed_tests_through_the_predicate():
    """Widening one site moves the refusal to the next (the item's finding:
    14 sites). No review-path module may keep a root-only literal."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("scripts/automod/gate.py", "scripts/automod/review.py",
                "scripts/automod/review_tools.py"):
        src = (root / rel).read_text()
        assert 'startswith("tests/")' not in src, rel
        assert "TP." in src, rel
    gate_src = (root / "scripts/automod/gate.py").read_text()
    assert "changed_tests = TP.pick_test_files(changed, self.worktree)" in gate_src


def test_the_tests_rung_partial_narrowing_counts_a_harness_test_as_a_test(tmp_path):
    """The partial run reads the same predicate: a delta touching only a
    harness test is a test-only delta, one touching harness code is not."""
    repo = tmp_path / "wt"
    (repo / "app" / "harness" / "tests").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                    text=True, check=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@e"); run("config", "user.name", "t")
    (repo / "pytest.ini").write_text("[pytest]\ntestpaths = tests app/harness/tests\n")
    (repo / "app" / "harness" / "loop.py").write_text("X = 1\n")
    (repo / "app" / "harness" / "tests" / "test_h.py").write_text("def test_h():\n    pass\n")
    run("add", "-A"); run("commit", "-q", "-m", "base")
    old = run("rev-parse", "HEAD").stdout.strip()
    (repo / "app" / "harness" / "tests" / "test_h.py").write_text("def test_h():\n    assert 1\n")
    run("commit", "-qam", "test only")
    head = run("rev-parse", "HEAD").stdout.strip()
    g = G.Gate.__new__(G.Gate)
    g.worktree, g.base = repo, "BASE"
    g.report = type("R", (), {"head": head})()
    g._reuse_load = lambda: {"tests": {"base": "BASE", "head": old, "ts": time.time()}}
    assert g._tests_delta_only() == ["app/harness/tests/test_h.py"]
    (repo / "app" / "harness" / "loop.py").write_text("X = 2\n")
    run("commit", "-qam", "code")
    g.report.head = run("rev-parse", "HEAD").stdout.strip()
    assert g._tests_delta_only() is None, "a code path in the delta is a full run"


# ── #1755: the honesty checker has no standing to refuse its own round ───────

_CLAIMED = [{"clause": i, "verdict": "met", "evidence_path": "app/x.py",
             "evidence_line": 3, "test_node_id": "tests/test_target.py::test_new_pattern",
             "how_verified": "read", "note": "read it"} for i in (1, 2)]


def _checker_round(tmp_path, *, touch_checker: bool):
    """A committed round that adds a test quoting a dishonesty shape in CODE.

    `assert True` and `or True` are matched by the live `_HONESTY_PATTERNS`
    exactly as written here, so the fixture needs no string concatenation to be
    found: what differs between the two cases is only whether the same commit
    also edits `scripts/automod/review.py`, which is the one fact the standing
    rule turns on.
    """
    live = tmp_path / "live"
    (live / "app").mkdir(parents=True)
    (live / "tests").mkdir()
    (live / "scripts" / "automod").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com")
    git(live, "config", "user.name", "t")
    (live / "app" / "x.py").write_text("".join(f"V{i} = {i}\n" for i in range(1, 11)),
                                       encoding="utf-8")
    (live / "scripts" / "automod" / "review.py").write_text("PATTERN = 1\n", encoding="utf-8")
    (live / "tests" / "test_target.py").write_text(
        "def test_existing():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "base")
    base = git(live, "rev-parse", "HEAD").stdout.strip()

    (live / "app" / "x.py").write_text("".join(f"V{i} = {i}\n" for i in range(1, 12)),
                                       encoding="utf-8")
    (live / "tests" / "test_target.py").write_text(
        "def test_existing():\n    assert 1 + 1 == 2\n\n"
        "def test_new_pattern():\n    ok = 1 or True\n    assert True\n", encoding="utf-8")
    changed = ["app/x.py", "tests/test_target.py"]
    if touch_checker:
        (live / "scripts" / "automod" / "review.py").write_text("PATTERN = 2\n", encoding="utf-8")
        changed.insert(0, "scripts/automod/review.py")
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "round: adds a test that quotes the pattern")
    return live, base, changed


def _review_rung(tmp_path, monkeypatch, live, base, changed):
    """Run the real review rung over `live` with only the model turn stubbed.

    `honesty_prechecks_with_standing` is deliberately NOT stubbed: which severity
    a finding ends up with, and whether `decide` may refuse on it, is what these
    two tests are about, and stubbing it would test the stub.
    """
    from scripts.automod import backlog as _BK, review as RV
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [])
    monkeypatch.setattr(G.S, "is_halted", lambda: False)
    monkeypatch.setattr(G.S, "is_broken", lambda: False)
    monkeypatch.setattr(G.S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b",
        "clauses": ["clause one holds", "clause two holds"], "path": ""})
    monkeypatch.setattr(RV, "grade", lambda **kw: {
        "ok": True, "error": "", "session_id": "sess_h", "structured_error": "",
        "structured": {"premise": "sound", "summary": "APPROVE — both clauses met",
                       "clauses": _CLAIMED, "test_honesty": [], "seams_unverified": []},
        "text": "", "stop_reason": "stop", "duration_s": 1.0})
    # `rung_review` imports `backlog` inside the method, so the patch goes on the
    # module object rather than an attribute of `gate`, and the item these rounds
    # would write is the live one.
    noted: list[list] = []
    monkeypatch.setattr(_BK, "note_review_advisories", lambda *a, **k: noted.append(list(a)))
    monkeypatch.setattr(_BK, "orphan_stale_amendments", lambda *a, **k: [])
    monkeypatch.setattr(_BK, "retriage_marks", lambda p: {})
    g = G.Gate("SM_STALE", live, base, live_root=live, item_id=7)
    g.report.changed_paths = changed
    g.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))
    ok, detail, data = g.rung_review()
    return ok, detail, data, events, noted


def test_a_round_editing_the_checker_is_not_refused_by_the_checker_it_replaces(tmp_path, monkeypatch):
    """The pre-change detector cannot convict the diff that changes it (#1755, clause 1).

    SM_20260928_210355 lost its second and last review attempt to four findings of
    exactly this kind: it was editing `honesty_prechecks`, and the LIVE module —
    the version with the rule it was adding — called its own new test fixtures
    `assert True` and `a new skip marker`. The finding is kept as advisory here,
    not dropped, and the demotion is said in the rung's recorded detail.
    """
    from scripts.automod import review as RV
    live, base, changed = _checker_round(tmp_path, touch_checker=True)
    ok, detail, data, events, _ = _review_rung(tmp_path, monkeypatch, live, base, changed)

    pre = [p for p in events[-1]["prechecks"] if p.get("demoted_from") == "blocking"]
    assert pre, f"the round's own pattern is still found, demoted: {events[-1]['prechecks']}"
    assert {p["severity"] for p in pre} == {"advisory"}, pre
    assert "scripts/automod/review.py" in RV.stale_honesty_modules(changed), \
        "the rule keys off the round's own changed paths"
    assert ok is True, f"a demoted finding cannot alone refuse the round: {detail}"
    assert "stale-by-construction" in detail, detail
    assert "scripts/automod/review.py" in detail, detail
    assert data["honesty_note"] and "demoted to advisory" in data["honesty_note"], data


def test_the_same_quoted_pattern_still_refuses_a_round_that_leaves_the_checker_alone(tmp_path, monkeypatch):
    """The other half: touching nothing but the test still costs the round (#1755, clause 2).

    Without this the fix would be an escape hatch — any round could add one line
    to `review.py` and stop answering the honesty check. The two rounds differ in
    exactly one changed path, and the verdict differs.
    """
    live, base, changed = _checker_round(tmp_path, touch_checker=False)
    ok, detail, data, events, _ = _review_rung(tmp_path, monkeypatch, live, base, changed)

    blocking = [p for p in events[-1]["prechecks"] if p["severity"] == "blocking"]
    assert blocking, f"an unedited checker keeps its standing: {events[-1]['prechecks']}"
    assert all("demoted_from" not in p for p in blocking), blocking
    assert ok is False and "review sent it back" in detail, detail
    assert data["honesty_note"] == "", "nothing to explain when nothing was demoted"



# ── the second reader on a review refusal (#1903) ──────────────────────────
#
# What the rung puts on the ledger and what it costs. Which blocks are askable,
# what the reader is shown, and what its answer may do are pinned in
# tests/test_automod_review.py; here the question is the row and the attempt.

# A refusal made of one synthesized clause verdict — the grader filed no finding
# of its own, which is the shape that refused SM_20260916_032218,
# SM_20260922_100227 and SM_20260924_104224, and the shape the reader exists for.
from scripts.automod import review as RV

# A refusal whose ONLY blocking entry is a vacuous-assertion finding: the shape
# the precheck finds by regex, which no reader looking at the diff can retire.
_UNMET_HONESTY = {"premise": "sound", "summary": "GATE_SUMMARY_SENTINEL",
                  "clauses": [{"clause": 1, "verdict": "met", "note": "as graded",
                               "evidence_path": "", "evidence_line": 0,
                               "test_node_id": "", "how_verified": ""}],
                  "test_honesty": [{"file": "tests/test_x.py", "line": 4,
                                    "pattern": "`or True`", "severity": "blocking",
                                    "actionable_in_round": True,
                                    "problem": "the assertion can never fail"}],
                  "seams_unverified": []}


_UNMET_ONE = {"premise": "sound", "summary": "GATE_SUMMARY_SENTINEL nothing else approved",
              "clauses": [{"clause": 1, "verdict": "unmet", "note": "GATE_ENTRY_SENTINEL no "
                           "second reader is wired", "evidence_path": "", "evidence_line": 0,
                           "test_node_id": "", "how_verified": ""}],
              "test_honesty": [], "seams_unverified": []}


def _confirm_reader(monkeypatch, *, retire=True, answer=None, error=""):
    """Put the policy on and replace the grader transport the reader uses.

    `RV.grade` (the first pass) stays the stub `_stub_grader` installed, so the
    only turns counted here are the reader's.
    """
    from scripts.automod import review as RV
    turns: list[dict] = []

    def run_grader(**kw):
        turns.append(kw)
        if error:
            return {"ok": False, "error": error, "text": "", "session_id": "sess_confirm"}
        return {"ok": True, "error": "", "session_id": "sess_confirm",
                "structured": answer if answer is not None
                else {"retire": retire, "reason": "checked the diff: the pin is on the line it names"}}
    monkeypatch.setattr(RV, "confirm_policy", lambda: True)
    monkeypatch.setattr(RV, "run_grader", run_grader)
    # One clause, so the refusal under test is one entry long and the reader's
    # turn count is the thing being counted rather than a missing clause.
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b", "path": "",
        "clauses": ["a grader-judgement refusal is confirmed before it costs an attempt"]})
    return turns


def test_a_bare_node_id_from_the_grader_holds_at_the_rung_that_refused_it(tmp_path,
                                                                          monkeypatch):
    """The process boundary this fix sits on, end to end. The grader answers in
    another process through `RV.grade`, and `rung_review` builds the resolver's
    whole search set itself — `changed_tests = TP.pick_test_files(changed,
    self.worktree)` — before handing it to `parse_review`. The #2177 nodes in
    test_automod_review.py stop at `parse_review`, so this is where the rung's own
    half is measured: the half that produced the 32 refusals, because
    `changed_paths` carried a test file the grader's bare node id never named.
    """
    (tmp_path / "app").mkdir(); (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "x.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_one():\n    assert 1\n\n\ndef test_two():\n    assert 2\n")
    obj = {"premise": "sound", "summary": "GATE_SUMMARY_SENTINEL both clauses met",
           "clauses": [
               {"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                "evidence_line": 3, "test_node_id": "test_one",
                "how_verified": "ran", "note": "ran it"},
               {"clause": 2, "verdict": "met", "evidence_path": "app/x.py",
                "evidence_line": 6, "test_node_id": "tests/test_x.py::test_two",
                "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    import json as _json
    sess = tmp_path / "sess.json"
    sess.write_text(_json.dumps({"messages": [{"role": "assistant",
                                               "content": "Checked both clauses."}]}))
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b", "path": "",
        "clauses": ["a bare node id holds", "a path-qualified node id holds"]})
    monkeypatch.setattr(RV, "grade", lambda *a, **k: {
        "ok": True, "structured": obj, "session_path": str(sess), "seconds": 0.01,
        "attempt": 1, "validated_head": "b" * 40})
    g = _ReviewGate(tmp_path)
    g.rung_changed_paths = lambda: ["app/x.py", "tests/test_x.py"]
    ok, detail, data = g.rung_review()
    assert ok is True, detail
    bare, qualified = data["clauses"]
    assert bare["verdict"] == "met" and "downgraded" not in bare, bare
    assert bare["accepted"] == ["bare node `test_one` resolved to tests/test_x.py, "
                                "a test file this diff changed"], bare
    # The path-qualified sibling keeps its waiver-free record: nothing about its
    # citation had to be completed, so the row carries no `accepted` at all.
    assert qualified["verdict"] == "met", qualified
    assert "downgraded" not in qualified and "accepted" not in qualified, qualified
    # The rung's `changed_paths` for this round was `app/x.py` plus
    # `tests/test_x.py` — the list `rung_changed_paths` returns and the grader's
    # bare id named none of. A bare name that list does NOT define is refused by
    # the same parse boundary two lines later in this file:
    # test_automod_review.py::test_a_bare_name_no_changed_test_defines_is_still_downgraded
    # — re-running this rung on the same commit would only be answered from the
    # ledger, so the strict half is pinned where the decision is made.


def test_a_refusal_whose_entries_are_all_retired_is_a_pass_that_spent_no_attempt(
        tmp_path, monkeypatch):
    """Clause 4, the overturn half, on the field the charging walk keys on.

    Every blocking entry is retired: the rung returns a pass, its row carries
    `blocking: false` — the field the walk at `gate.py:2485-2496` counts — and
    the vote with its one-line reason is on that same row. The detail says which
    kind of pass it is, so "1 met of 2" is not the same sentence as the round
    that was never refused.
    """
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()

    assert ok is True, "a refusal with nothing left standing is a pass"
    assert len(turns) == 1, "one blocking entry, one reader turn"
    assert turns[0]["final_schema"] is RV.CONFIRM_SCHEMA
    ev = events[-1]
    assert ev["blocking"] is False and ev["kind"] == "pass", ev
    assert ev["review_confirm"] == "overturned"
    assert ev["review_confirm_votes"][0]["verdict"] == "retired"
    assert "no review attempt spent" in detail and "GATE_SUMMARY_SENTINEL" in detail
    assert "review_retry" not in data and data.get("review_attempt") == 1

    # The attempt, measured by the walk that charges it. The walk counts the
    # rows with `blocking: true` and asks at `spent + 1`, so an overturned row
    # leaves the round at the attempt it was on before the grading turn: the
    # next grade is still attempt 1 of 2, not 2 of 2.
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(events))
    ctx = _ReviewGate(tmp_path)._review_prepare()
    assert isinstance(ctx, dict), ctx
    assert ctx["attempt"] == 1, ctx


def test_an_upheld_refusal_is_todays_refusal_plus_the_recorded_vote(tmp_path, monkeypatch):
    """Clause 4, the uphold half: the sentence and the charge do not move.

    The reader looked and the finding stood. The refusal the author reads, the
    attempt it spends, and the row's `blocking` are what the rung produces with
    the policy off; the only difference is `review_confirm: upheld` and the
    reader's reason beside it.
    """
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch, retire=False)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()

    assert ok is False
    assert len(turns) == 1
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    assert ev["review_confirm"] == "upheld"
    assert "GATE_ENTRY_SENTINEL" in ev["findings"]
    assert "clause upheld" in ev["review_confirm_reason"]
    assert data["review_retry"] is True and data["review_attempt"] == 1
    assert "review sent it back" in detail

    # And the charge: one `blocking: true` row on record, so the next grade is
    # attempt 2 of the round's two — the same number the rung charged before a
    # second reader existed.
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(events))
    ctx = _ReviewGate(tmp_path)._review_prepare()
    assert isinstance(ctx, dict), ctx
    assert ctx["attempt"] == 2, ctx


def test_a_refusal_is_never_put_to_a_second_reader_while_the_policy_is_off(
        tmp_path, monkeypatch):
    """Clause 1 at the rung: off reproduces today's single-vote behaviour.

    Same grader output, same refusal — and no reader turn, no `review_confirm`
    field on the row and none in the rung data. A field whose absence a later
    reader could mistake for a vote is worse than no field.
    """
    from scripts.automod import review as RV
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch)
    monkeypatch.setattr(RV, "confirm_policy", lambda: False)   # the shipped default
    ok, detail, data = _ReviewGate(tmp_path).rung_review()

    assert ok is False and "review sent it back" in detail
    assert turns == [], "policy off means no second grader turn, ever"
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "retry"
    assert not [k for k in ev if k.startswith("review_confirm")], ev
    assert not [k for k in data if k.startswith("review_confirm")], data


def test_a_shadow_vote_is_recorded_and_changes_nothing_the_rung_decides(tmp_path, monkeypatch):
    """#2017: in shadow the reader runs and its vote is on the row — and the
    refusal, the row's `blocking`, and the charged attempt are the shipped ones.

    The reader here retires the only blocking entry, which with the policy `on`
    is a pass that spends no attempt (the node above). In shadow the same vote is
    recorded as `overturned` beside the seconds the reader took, the rung still
    sends the round back, the row still says `blocking: true` / `kind: retry`,
    and the charging walk counts it: the next grade is attempt 2.
    """
    from scripts.automod import review as RV
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch)                      # retires what it is shown
    monkeypatch.setattr(RV, "confirm_policy", lambda: RV.CONFIRM_SHADOW)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()

    assert ok is False and "review sent it back" in detail, detail
    assert "no review attempt spent" not in detail
    assert len(turns) == 1, "shadow runs the reader"
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    assert ev["review_confirm"] == "overturned"
    assert ev["review_confirm_mode"] == "shadow"
    assert ev["review_confirm_reason"] and ev["review_confirm_votes"][0]["verdict"] == "retired"
    secs = ev["review_confirm_seconds"]
    assert isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs >= 0
    assert data["review_retry"] is True and data["review_attempt"] == 1
    assert data["review_confirm"] == "overturned" and "review_confirm_seconds" in data

    monkeypatch.setattr(G.S, "read_events", lambda limit=100: list(events))
    ctx = _ReviewGate(tmp_path)._review_prepare()
    assert isinstance(ctx, dict), ctx
    assert ctx["attempt"] == 2, "a shadow overturn refunded the attempt"


def test_the_on_and_off_states_do_not_gain_the_shadow_fields(tmp_path, monkeypatch):
    """#2017: `on` still overturns and refunds, `off` still writes no
    `review_confirm*` key — and neither carries the shadow-only fields."""
    from scripts.automod import review as RV
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch)
    monkeypatch.setattr(RV, "confirm_policy", lambda: RV.CONFIRM_ON)
    ok, _detail, _data = _ReviewGate(tmp_path).rung_review()
    ev = events[-1]
    assert ok is True and ev["blocking"] is False and ev["kind"] == "pass", ev
    assert sorted(k for k in ev if k.startswith("review_confirm")) == [
        "review_confirm", "review_confirm_reason", "review_confirm_votes"]

    for off in (RV.CONFIRM_OFF, False, "shadows", None):
        turns.clear()
        monkeypatch.setattr(RV, "confirm_policy", lambda off=off: off)
        ok, _detail, data = _ReviewGate(tmp_path).rung_review()
        ev = events[-1]
        assert ok is False and turns == [], off
        assert not [k for k in ev if k.startswith("review_confirm")], (off, ev)
        assert not [k for k in data if k.startswith("review_confirm")], (off, data)


def test_the_shipped_confirm_value_and_the_shadow_it_replaced_reach_one_row(tmp_path,
                                                                           monkeypatch):
    """#2305 clause 4: the flipped value is what the rung obeys, and the `shadow`
    arm the flip leaves in place still records its vote on the row.

    This is the seam from the shipped bytes to the ledger: the value is read out
    of the tracked `config.yaml` and pushed through `confirm_policy_state`, the
    function `confirm_policy()` itself calls, instead of being hardcoded — so the
    node fails if the file stops saying `off`, and the same node re-points the
    policy at `shadow` to show the arm is intact. As shipped, a graded block
    carries no key beginning `review_confirm` at all and asks nobody; in shadow
    the same block carries `review_confirm_mode` beside its vote. Both neighbour
    spellings are asserted here too: a round that changed the value must not have
    moved what `shadow` and `on` resolve to.
    """
    import yaml
    from scripts.automod import review as RV

    shipped_file = Path(__file__).resolve().parents[1] / "config.yaml"
    raw = yaml.safe_load(shipped_file.read_text(encoding="utf-8"))["automod"]["review"]["confirm"]
    shipped = RV.confirm_policy_state(raw)
    assert shipped == RV.CONFIRM_OFF, raw
    assert RV.confirm_policy_state(RV.CONFIRM_SHADOW) == RV.CONFIRM_SHADOW
    assert RV.confirm_policy_state(RV.CONFIRM_ON) == RV.CONFIRM_ON

    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
    turns = _confirm_reader(monkeypatch)

    monkeypatch.setattr(RV, "confirm_policy", lambda: shipped)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()
    ev = events[-1]
    assert ok is False and "review sent it back" in detail, detail
    assert turns == [], "the shipped value asks no second reader"
    assert ev["kind"] == "retry" and ev["blocking"] is True, ev
    assert not [k for k in ev if k.startswith("review_confirm")], sorted(ev)
    assert not [k for k in data if k.startswith("review_confirm")], sorted(data)

    monkeypatch.setattr(RV, "confirm_policy", lambda: RV.CONFIRM_SHADOW)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()
    ev = events[-1]
    assert ok is False and "review sent it back" in detail, detail
    assert len(turns) == 1, "shadow still runs the reader"
    assert ev["review_confirm_mode"] == "shadow", sorted(ev)
    assert ev["review_confirm"] == "overturned" and "review_confirm_seconds" in ev, sorted(ev)
    assert data["review_confirm_mode"] == "shadow", sorted(data)


def test_a_review_row_carries_its_blocking_entries_whole(tmp_path, monkeypatch):
    """#2017: the row's own `blocking_entries` is the decision's list, untouched by
    the 2000-character cap on `findings` — 118 of 365 recorded blocks ran past
    that cap and could not be replayed."""
    from scripts.automod import review as RV
    # Six clauses, each unmet with a long note: a per-clause note is itself capped,
    # so it takes a full contract to run the joined text past 2000 characters.
    note = "GATE_ENTRY_SENTINEL " + "the pin is not on the line it names " * 40
    unmet = {**_UNMET_ONE, "clauses": [
        {**_UNMET_ONE["clauses"][0], "clause": n, "note": f"{note} TAIL{n}"}
        for n in range(1, 7)]}
    events, _ = _stub_grader(monkeypatch, tmp_path, unmet)
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "b", "path": "",
        "clauses": [f"clause text {n}" for n in range(1, 7)]})
    monkeypatch.setattr(RV, "confirm_policy", lambda: RV.CONFIRM_OFF)
    ok, _detail, _data = _ReviewGate(tmp_path).rung_review()
    ev = events[-1]
    assert ok is False and len(ev["findings"]) == 2000, len(ev["findings"])
    entries = ev["blocking_entries"]
    assert len(entries) == 6 and all(set(e) == {"text", "kind"} for e in entries), entries
    assert sum(len(e["text"]) for e in entries) > 2000, "the list was capped with the text"
    assert [e["kind"] for e in entries] == ["clause"] * 6
    # The capped text re-splits to fewer whole entries than the decision made;
    # the list does not, and each stored entry re-splits to itself.
    assert len(RV.blocking_entries_from_text(ev["findings"])) < 6 or \
        RV.blocking_entries_from_text(ev["findings"])[-1]["text"] != entries[-1]["text"]
    for e in entries:
        assert RV.blocking_entries_from_text(e["text"]) == [e]


def test_a_refusal_the_code_computed_is_not_put_to_the_reader(tmp_path, monkeypatch):
    """Clause 1's exemption at the rung: a fact about the tree gets no vote.

    A vacuous assertion the precheck found is not a grader judgment about this
    diff, so with the policy on and a reader that would retire anything, the
    refusal stands, no reader turn is spent, and the row names why.
    """
    events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_HONESTY)
    turns = _confirm_reader(monkeypatch)
    ok, detail, data = _ReviewGate(tmp_path).rung_review()

    assert ok is False
    assert turns == [], "a pattern a Python check found is never offered"
    ev = events[-1]
    assert ev["blocking"] is True
    assert ev["review_confirm"] == "not_asked" and ev["review_confirm_reason"] == \
        "all_python_computed", ev
    assert "test honesty" in ev["findings"]


def test_a_reader_that_cannot_answer_leaves_the_refusal_exactly_as_it_was(
        tmp_path, monkeypatch):
    """An unreachable second reader is an outage on the record, not a pass.

    Every failure mode of the reader — a backend that refuses the turn, an object
    with no `retire` in it, a reader that raises — upholds. The refusal, its
    sentence and its charge are today's, and the reason says the reader could not
    be reached rather than that it judged.
    """
    for kwargs in ({"error": "the backend refused the turn"},
                   {"answer": {"reason": "no retire key at all"}},
                   {"answer": None, "retire": "yes please"}):
        events, _ = _stub_grader(monkeypatch, tmp_path, _UNMET_ONE)
        turns = _confirm_reader(monkeypatch, **kwargs)
        ok, detail, data = _ReviewGate(tmp_path).rung_review()
        assert ok is False, kwargs
        assert len(turns) == 1, kwargs
        ev = events[-1]
        assert ev["blocking"] is True and ev["kind"] == "retry", (kwargs, ev)
        assert ev["review_confirm"] == "upheld", (kwargs, ev)
        assert "review sent it back" in detail, kwargs


# ---------------------------------------------------------------------------
# The floors after the summary anchor (#2251): a small run is still a small run
# ---------------------------------------------------------------------------

def _repo_of_green_tests(tmp_path, count):
    """A throwaway repo whose entire suite is `count` passing tests in one file —
    the shape the collected floor exists to refuse, with nothing red to blame."""
    r = tmp_path / "greenlive"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_two_dozen.py").write_text(
        "\n".join(f"def test_green_{i}():\n    assert True\n"
                  for i in range(count)), encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r, git(r, "rev-parse", "HEAD").stdout.strip()


def test_a_genuinely_small_run_is_refused_by_the_floors_untouched(tmp_path, monkeypatch,
                                                                  _private_red_set):
    """#2251 clause 4. Anchoring the parser on one summary line must not have
    disarmed the floor it used to be fed from: a round over a repo that really has
    24 tests and really runs all 24 green is still refused, because 24 is not a
    suite. The constants are asserted at their shipped values so a later round
    cannot clear a small tree by lowering the floor and call that a fix —
    `test_the_parallel_rung_floor_is_1000` pins the same pair from the other side.
    """
    repo, base = _repo_of_green_tests(tmp_path, 24)
    monkeypatch.setattr(G.W, "WORK_ROOT", tmp_path / "work")
    wt = tmp_path / "work" / "SM_TSMALL" / "home" / "lloyd"
    wt.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "-q", "-b", "automod/SM_TSMALL", str(wt), base)
    (wt / "unrelated.py").write_text("X = 1\n", encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "a change")
    g = _gate_for(repo, wt, base, monkeypatch)
    g.python = Path(sys.executable)
    try:
        ok, detail, data = g.rung_tests()
    finally:
        git(repo, "worktree", "remove", "--force", str(wt))

    assert G.PYTEST_MIN_COLLECTED == 1000, G.PYTEST_MIN_COLLECTED
    assert G.PYTEST_MIN_PASSED == 1000, G.PYTEST_MIN_PASSED
    assert ok is False, detail
    assert "did the round delete tests?" in detail, detail
    assert f"floor {G.PYTEST_MIN_COLLECTED}" in detail, (
        "the refusal has to name the number it applied, so a changed floor cannot "
        f"quietly move the goalposts: {detail!r}")
    assert data["passed"] == 24 and data["collected"] == 24, data


def test_the_shapes_the_parser_pinned_before_the_anchor_still_parse():
    """#2251 clause 5. The anchor was added to a function three other jobs share
    (`_failures_at_base`, `_reconfirm_candidate_failures`,
    `vault_guards._run_selection`), so the parses that predate it are pinned here
    rather than left to the table at the top of this file: pytest's own collection
    report still answers for `collected` ahead of the summary line's arithmetic, a
    `no tests ran` run is still all zeros, and `pin_findings` is still on the dict
    whatever the text looks like — it is the run's only channel for "a pin did not
    execute", and `rung_tests` returns this dict as its data on every branch (the
    green rung's own check is tests/test_dashboard_responsive.py, and
    tests/test_gate_parallel_tests.py:553 and :856 read the same key off this parse).
    """
    reported = G._parse_pytest_summary(
        "collected 1250 items\n\n1247 passed, 3 xfailed in 30.20s")
    assert (reported["collected"], reported["passed"], reported["xfailed"]) == (
        1250, 1247, 3), reported

    ran_nothing = G._parse_pytest_summary("no tests ran in 0.01s")
    assert {k: v for k, v in ran_nothing.items() if k != "pin_findings"} == {
        "passed": 0, "failed": 0, "errors": 0, "xfailed": 0, "tests_skipped": 0,
        "collected": 0}, ran_nothing

    finding = f"{G.PIN_FINDING_PREFIX}: 10 of 10 pins did not run"
    assert G._parse_pytest_summary(f"1247 passed in 30.20s\n{finding}\n")[
        "pin_findings"] == [finding]
    for text in (f"{finding}\n", "INTERNALERROR> boom\n", ""):
        assert "pin_findings" in G._parse_pytest_summary(text), text[:40]


# ---------------------------------------------------------------------------
# The implement loop's pickup probe (#2385)
#
# The `tests` rung asks "did the ROUND break this". The implement loop asks a
# different question — "is the red this item filed still there" — and had no way
# to ask it, so a healed item was picked up and worked anyway: #2384 spent 22
# turns and opened no round, #1846 spent 14 the same way. `red_tree_state_at_head`
# is the same base probe with its base aimed at HEAD and its targets at the
# item's own node ids, which is where it earns the right to close anything.
# ---------------------------------------------------------------------------

FINE = "tests/test_pre.py::test_fine"
BROKEN = "tests/test_pre.py::test_already_broken"


def test_a_pickup_probe_reruns_the_nodes_it_is_given(tmp_path):
    """The seam the pickup decision depends on, against a real pytest: the probe
    checks the live tree's HEAD out into a throwaway worktree and runs exactly
    the node ids it was given there. A node that passes is not marked red because
    a neighbour in the same file fails, and a node that fails cannot be read as
    healed."""
    repo, head = _repo_with_failing_test(tmp_path)
    res = G.red_tree_state_at_head([FINE], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "green")
    assert res["head"] == head and res["probed"] == [FINE], res
    assert res["conclusive"] is True and res["unresolved"] == [], res["note"]
    assert "fresh data root" in res["note"], \
        "it runs the live venv against a checkout of HEAD, so #1436's rule applies"
    listed = git(repo, "worktree", "list", "--porcelain").stdout
    assert "baseline" not in listed, "the throwaway worktree is deregistered"

    res = G.red_tree_state_at_head([FINE, BROKEN], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "red")
    assert res["conclusive"] is True
    assert res["unresolved"] == [BROKEN], res["note"]


def test_a_pickup_probe_does_not_read_a_vanished_node_as_healed(tmp_path):
    """The node was renamed since the item was filed, so pytest exits 4 having
    collected nothing. `_failures_at_base` answers that with the empty set, which
    at the GATE means "not pre-existing" and at pickup would mean "healed — close
    it": the polarity is backwards exactly where closing is on the table, so
    nothing-passed-and-nothing-failed is an inconclusive answer, never a green
    one."""
    repo, head = _repo_with_failing_test(tmp_path)
    (repo / "tests" / "test_pre.py").write_text(
        "def test_renamed_instead():\n    assert False\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the node was renamed")
    res = G.red_tree_state_at_head([BROKEN], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "s")
    assert res["conclusive"] is False, res["note"]
    assert res["unresolved"] == [BROKEN], "no answer is the still-red answer"


def test_a_pickup_probe_calls_a_collection_error_a_node_that_is_still_red(tmp_path):
    """A collection error names its FILE and means nothing inside it ran, so the
    node inside it is unresolved — `close` requires every node to have passed,
    not merely to have gone unmentioned by the summary."""
    repo, head = _repo_with_failing_test(tmp_path)
    (repo / "tests" / "test_pre.py").write_text(
        "import definitely_not_a_real_module\n\ndef test_already_broken():\n"
        "    assert False\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test file that will not import")
    res = G.red_tree_state_at_head([BROKEN], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "s")
    assert res["conclusive"] is True, res["note"]
    assert res["unresolved"] == [BROKEN], res["note"]


def test_a_pickup_probe_calls_a_hang_no_answer(tmp_path):
    """The probe runs on the implement loop's slot, so it carries its own bound.
    A hung test costs that bound and nothing else: the item is attempted as
    today, and no red tree is retired."""
    repo, head = _repo_with_failing_test(tmp_path)
    (repo / "tests" / "test_pre.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test that hangs")
    res = G.red_tree_state_at_head(["tests/test_pre.py::test_slow"],
                                   python=Path(sys.executable), live_root=repo,
                                   scratch=tmp_path / "s", timeout=1.0)
    assert res["conclusive"] is False, res["note"]
    assert res["unresolved"] == ["tests/test_pre.py::test_slow"]
    assert "timed out" in res["note"], res["note"]


def test_a_pickup_probe_removes_the_scratch_it_made(tmp_path, monkeypatch):
    """It is called once per red-tree pickup and never asked to clean up after
    itself, so its scratch — a whole checkout — is its own to remove, and off
    `/tmp`, which this box's suite has filled twice."""
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    repo, head = _repo_with_failing_test(tmp_path)
    res = G.red_tree_state_at_head([FINE], python=Path(sys.executable), live_root=repo)
    assert res["conclusive"] is True and res["unresolved"] == [], res["note"]
    assert list((tmp_path / "work").glob("red-tree-pickup-*")) == []


@pytest.mark.parametrize("nodes,why", [
    ([], "nothing to probe"),
    (["tests/test_pre.py::test_fine"], "no tree to read HEAD from"),
])
def test_a_pickup_probe_with_nothing_to_answer_says_so(nodes, why, tmp_path):
    """Both shapes that cannot produce a verdict say `conclusive: False` before
    spending a subprocess, so the caller cannot mistake an empty `unresolved` for
    a heal: an item with no nodes, and a tree git cannot name."""
    res = G.red_tree_state_at_head(nodes, python=Path(sys.executable),
                                   live_root=tmp_path / "not-a-repo",
                                   scratch=tmp_path / "s")
    assert res["conclusive"] is False and res["unresolved"] == list(nodes), res


def test_a_pickup_probe_counts_a_node_whose_file_vanished_as_still_red(tmp_path):
    """A node whose FILE was deleted since the item was filed cannot be handed to
    pytest: asked for, pytest exits 4 with `ERROR: file or directory not found`
    and runs nothing at all (measured), which would answer for its healthy
    siblings too. So the probe never asks for it — and a green run of the
    surviving targets says nothing about it. It counts as not passed, because the
    alternative is retiring an item as healed whose test was simply removed."""
    repo, head = _repo_with_failing_test(tmp_path)
    other = "tests/test_other.py::test_sibling_also_red"
    (repo / "tests" / "test_other.py").write_text(
        "def test_sibling_also_red():\n    assert False\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a second file that is red too")
    (repo / "tests" / "test_pre.py").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one of the two red files was deleted")
    res = G.red_tree_state_at_head([BROKEN, other], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "s")
    assert res["conclusive"] is True, res["note"]
    assert res["unresolved"] == [BROKEN, other], res["note"]
    assert "whose file is gone" in res["note"], res["note"]


def test_a_pickup_probe_reads_a_parametrised_failure_as_that_nodes_red(tmp_path):
    """An item files `tests/test_beta.py::test_beta`; at HEAD the same test fails
    as `test_beta[1]` and `test_beta[2]`, which is what the short summary reports
    (measured: `FAILED tests/test_beta.py::test_beta[1] - assert 1 == 99`). Exact
    id membership reads that as a clean answer and closes the item while its own
    red sits two lines up in the same output, so a node matches a reported id on
    its whole `file::name`, bracket suffix ignored, in both directions."""
    repo, head = _repo_with_failing_test(tmp_path)
    beta = "tests/test_beta.py::test_beta"
    (repo / "tests" / "test_beta.py").write_text(
        "import pytest\n\n\n@pytest.mark.parametrize('v', [1, 2])\n"
        "def test_beta(v):\n    assert v == 99\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the red became parametrised")
    res = G.red_tree_state_at_head(["tests/test_pre.py::test_fine", beta],
                                   python=Path(sys.executable), live_root=repo,
                                   scratch=tmp_path / "a")
    assert res["conclusive"] is True, res["note"]
    assert res["unresolved"] == [beta], res["note"]
    assert "2 already failing" in res["note"], \
        "the run reported both param ids, and the item's node is one of them"

    # The other direction is not a red answer, it is no answer: asked for a param
    # id of a test that is no longer parametrised, pytest finds no match, exits 4
    # and runs nothing, so the probe has no verdict to close with and every node
    # counts unresolved — which spends the attempt, the safe reading of a drift it
    # cannot resolve into an id.
    (repo / "tests" / "test_beta.py").write_text(
        "def test_beta():\n    assert False\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the parametrization went away again")
    res = G.red_tree_state_at_head([beta + "[1]"], python=Path(sys.executable),
                                   live_root=repo, scratch=tmp_path / "b")
    assert res["conclusive"] is False, res["note"]
    assert res["unresolved"] == [beta + "[1]"], res["note"]


# ---------------------------------------------------------------------------
# The observe-only pyright type rung (#2450)
# ---------------------------------------------------------------------------

#: The dataclass and its caller, as D13 (9595ddc3) had them: `RunOptions`
#: carried `env`, and a second file kept passing it after the field went away.
#: `LIMIT: int = "not a number"` is the pre-existing finding the delta must
#: swallow — pyright sees it in both runs, and only a difference is reportable.
_TC_RUNOPTS_BASE = (
    "from dataclasses import dataclass\n"
    "\n"
    'LIMIT: int = "not a number"\n'
    "\n"
    "\n"
    "@dataclass\n"
    "class RunOptions:\n"
    "    limit: int = 0\n"
    '    env: str = ""\n'
)
_TC_CALLER = (
    "from runopts import RunOptions\n"
    "\n"
    "\n"
    "def build():\n"
    "    return RunOptions(limit=1, env=\"prod\")\n"
)
_TC_SIBLING = "def helper(x):\n    return x\n"
#: The vendored fork, which holds 68,003 of this tree's 69,840 pyright findings
#: and is not ours to type-check. It is a real inbound caller here, so excluding
#: it has to be a rule and not an accident of where the scan happened to look.
_TC_FORK_PATH = "agent-services/llm/djev-vllm-fork/fork_caller.py"
_TC_FORK = (
    "from runopts import RunOptions\n"
    "\n"
    "\n"
    "def legacy():\n"
    "    return RunOptions(env=\"prod\")\n"
)


def _tc_git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       text=True, check=True)
    return r.stdout


def _typecheck_round(tmp_path, monkeypatch, *, mode: str = "field_dropped",
                     fork: bool = False):
    """A real Gate over a scratch repo, plus the real pyright binary.

    Base commit: `runopts.py` (the dataclass, with one pre-existing
    `reportAssignmentType` finding in it), `caller.py` (passes `limit` and
    `env`), `sibling.py` (imports nothing from that module, so it must never
    enter the file set), and with `fork=True` the vendored fork's own caller.

    `mode` picks the head commit:

      `field_dropped` — the round deletes the `env` field from the dataclass
        and never opens the caller. This is the shape the item names: a
        bool→dataclass-return-style signature break that produces no finding
        anywhere in pyflakes or the graph.
      `shift` — five comment lines above the unchanged file. Nothing is new;
        a key holding line numbers would report the pre-existing finding as
        if it were.

    Returns (gate, captured ledger events). `live_root` is the scratch repo
    because that is the repo holding this round's base commit, and the rung
    takes its base run from a detached checkout of it. The interpreter the
    checker is told to resolve imports with and the checker itself are the real
    ones, so the test crosses the same process boundary the rung does.
    """
    root = tmp_path / "wt"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    for k, v in (("user.email", "tc@example.invalid"), ("user.name", "tc")):
        _tc_git(root, "config", k, v)
    (root / "runopts.py").write_text(_TC_RUNOPTS_BASE, encoding="utf-8")
    (root / "caller.py").write_text(_TC_CALLER, encoding="utf-8")
    (root / "sibling.py").write_text(_TC_SIBLING, encoding="utf-8")
    if fork:
        fork_dir = root / _TC_FORK_PATH.rsplit("/", 1)[0]
        fork_dir.mkdir(parents=True)
        (root / _TC_FORK_PATH).write_text(_TC_FORK, encoding="utf-8")
    _tc_git(root, "add", "-A")
    _tc_git(root, "commit", "-q", "-m", "base")
    base = _tc_git(root, "rev-parse", "HEAD").strip()

    if mode == "field_dropped":
        (root / "runopts.py").write_text(
            _TC_RUNOPTS_BASE.replace('    env: str = ""\n', ""), encoding="utf-8")
        subject = "dropped RunOptions.env"
    elif mode == "shift":
        (root / "runopts.py").write_text("# a note\n" * 5 + _TC_RUNOPTS_BASE,
                                         encoding="utf-8")
        subject = "five comment lines above an untouched finding"
    else:
        raise AssertionError(f"unknown mode {mode!r}")
    _tc_git(root, "add", "-A")
    _tc_git(root, "commit", "-q", "-m", subject)

    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", events.append)
    monkeypatch.setattr(G.W, "round_dir", lambda *a, **k: tmp_path / "round")
    g = G.Gate("SM_TYPECHECK_ROUND", root, base, live_root=root)
    # The real checker, found beside the interpreter running this test — the
    # sibling of `sys.executable`, and NOT of its resolved target: the venv's
    # `bin/python` is a symlink into the uv-managed base interpreter, so
    # `.resolve()` lands in ~/.local/share/uv and finds no pyright there. This
    # cannot use `Path.home()` or `LIVE_ROOT`: the gate runs the suite with HOME
    # pointed at a round home so a candidate test cannot write live automod
    # state, and `.venvs` is untracked, so neither root has the binary in a
    # worktree.
    g.python = Path(sys.executable)
    g.pyright = Path(sys.executable).parent / "pyright"
    assert g.pyright.exists(), f"no pyright at {g.pyright}: the rung has no checker"
    return g, events


def test_the_type_rung_catches_the_break_on_a_line_the_round_never_touched(tmp_path,
                                                                          monkeypatch):
    """#2450 clause 2: a caller-breakage finding, located in an unchanged file.

    The round changed one file. The finding it must report is in another one,
    which is the whole reason the file set is `changed ∪ inbound callers` and
    not the diff: `reportCallIssue` on `caller.py` is what #528's Risks section
    says no other surface in the loop can see.
    """
    g, _events = _typecheck_round(tmp_path, monkeypatch)
    changed = sorted(G.W.changed_paths(g.worktree, g.base))
    assert changed == ["runopts.py"], changed

    ok, detail, data = g.rung_pyright()
    rec = data["pyright"]

    assert ok is True, "an observe-only rung never refuses a round"
    assert rec["status"] == "evaluated", rec
    assert rec["observe_only"] is True
    assert rec["new"] == [{
        "file": "caller.py", "rule": "reportCallIssue",
        "message": 'No parameter named "env"', "count": 1,
    }], rec["new"]
    assert rec["counts"] == {"reportCallIssue": 1}
    assert rec["labels"] == ["reportCallIssue:caller.py"]
    assert detail.startswith("observe-only:"), detail
    assert "caller.py" not in changed, "the finding is in a file the round did not change"


def test_the_type_rung_reports_only_what_is_new_at_base(tmp_path, monkeypatch):
    """#2450 clause 3: the reported set is head-minus-base, so a pre-existing
    finding is not reported and inserting lines above it does not re-flag it.

    `runopts.py` carries `LIMIT: int = "not a number"` at base. In the `shift`
    round the file moves down five lines and nothing else changes, so the
    correct answer is an empty delta over a non-empty run — the positive
    control that the checker ran and saw the finding is `base_findings >= 1`.
    """
    g, _events = _typecheck_round(tmp_path, monkeypatch, mode="shift")
    ok, detail, data = g.rung_pyright()
    rec = data["pyright"]

    assert ok is True
    assert rec["status"] == "evaluated", rec
    assert rec["totals"]["base_findings"] >= 1, "positive control: it saw the old finding"
    assert rec["totals"]["head_findings"] == rec["totals"]["base_findings"]
    assert rec["new"] == []
    assert rec["counts"] == {}
    assert rec["labels"] == []
    assert "pre-existing" in detail, detail


def test_the_type_rung_checks_changed_files_plus_inbound_callers_of_what_changed(
        tmp_path, monkeypatch):
    """#2450 clause 4: the invocation's file set, and its own wall time.

    `caller.py` is in the set because it imports the module whose module-level
    `RunOptions` this round changed; `sibling.py` is not, because it imports
    nothing from there. The rung times itself, because the rung budget the
    item cites is ~8 s and pyright is a Node process doing whole-program work.
    """
    g, _events = _typecheck_round(tmp_path, monkeypatch)
    ok, detail, data = g.rung_pyright()
    rec = data["pyright"]

    assert ok is True
    assert sorted(rec["files"]) == ["caller.py", "runopts.py"], rec["files"]
    assert rec["caller_files"] == ["caller.py"]
    assert "sibling.py" not in rec["files"]
    assert rec["totals"]["seconds"] > 0
    assert rec["totals"]["head_seconds"] > 0 and rec["totals"]["base_seconds"] > 0
    assert rec["totals"]["files"] == 2 and rec["totals"]["callers"] == 1


def test_the_type_rung_never_checks_the_vendored_fork(tmp_path, monkeypatch):
    """#2450 clause 4, the exclusion half: 97% of this tree's pyright findings
    are in `agent-services/llm/djev-vllm-fork/`, and the fork IS an inbound
    caller here, so it would be checked if the file set were merely "callers".
    """
    g, _events = _typecheck_round(tmp_path, monkeypatch, fork=True)
    assert g.pyright  # the checker is the real one
    listed = _tc_git(g.worktree, "ls-files", "*.py")
    assert _TC_FORK_PATH in listed, "positive control: the fork file is tracked here"

    ok, detail, data = g.rung_pyright()
    rec = data["pyright"]

    assert ok is True
    assert rec["files"] == sorted(rec["files"])
    assert not any(p.startswith("agent-services/llm/djev-vllm-fork/") for p in rec["files"]), \
        rec["files"]
    assert _TC_FORK_PATH not in rec["caller_files"]
    assert [n["file"] for n in rec["new"]] == ["caller.py"], rec["new"]


def test_the_pyright_record_rides_the_round_event_and_cannot_lose_a_round(
        tmp_path, monkeypatch):
    """#2450 clause 5: the durable round event, beside `vet`, and observe-only
    on the ladder itself.

    Two things are pinned here and both are load-bearing for the soak. The
    record has to leave the gate process on an event: `gate.json` is deleted
    with the worktree, which is how #679's owed-check nearly lost the vet's
    denominator. And a rung that found something must still be `ok`, because
    the flag rate is what decides whether it may ever refuse a round.
    """
    g, events = _typecheck_round(tmp_path, monkeypatch)
    for name in ("preflight", "static", "frontend", "tests", "prompt_surface",
                 "review", "venv", "canary_boot", "canary_smoke", "drill"):
        monkeypatch.setattr(g, f"rung_{name}", lambda n=name: (True, f"stub {n}", {}))

    report = g.run()

    rung = [r for r in report.rungs if r.name == "pyright"]
    assert len(rung) == 1, [r.name for r in report.rungs]
    assert rung[0].ok is True
    assert rung[0].data["pyright"]["new"], "this round really does have a finding"
    assert report.ok is True, "a finding may not decide a round during the soak"

    ev = [e for e in events if e.get("rung") == "pyright"]
    assert len(ev) == 1, [e.get("rung") for e in events]
    assert ev[0]["ok"] is True
    # The same five fields the `vet` record carries, and no wider: a compact
    # record is what lets an always-on rung sit in the ledger without bloating it.
    assert sorted(ev[0]["pyright"]) == sorted(
        ["status", "observe_only", "counts", "labels", "totals"]), ev[0]["pyright"]
    assert ev[0]["pyright"]["observe_only"] is True
    assert ev[0]["pyright"]["counts"] == {"reportCallIssue": 1}
    assert ev[0]["pyright"]["labels"] == ["reportCallIssue:caller.py"]
    assert ev[0]["pyright"]["totals"]["new_findings"] == 1
