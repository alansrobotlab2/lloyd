"""The `tests` rung runs the suite in parallel, and re-asks a failure serially.

The suite grew from ~4,900 tests to ~5,900 in the week to 2026-09-18 and its
serial run from 153 s to ~600 s on a 32-core box, holding `gate-tests.lock`
the whole time — so a second gate queued another 400–470 s behind it, and a
full gate outgrew the 59-minute turn that waits on it (automod.md §3.2g).
Eight `xdist` workers run the same suite in ~70 s.

What parallelism costs is load, and the first trial showed it: a test that
asserts a 90 ms latency budget lost it three runs in three. That is a fact
about the box during the run. So a parallel failure is never believed until
the files that failed have been run again, serially, and THAT run is the
verdict.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.automod import gate as G

SUMMARY_OK = "5895 passed, 13 skipped, 2 xfailed in 70.10s\n"
# The rung's `-m` expression, as one argv element, read from the gate rather than copied: it
# widened on 2026-09-21 for #644 clause 4, and three tests below assert the argv it builds, so
# a local copy would be a second place the expression exists and a widening would redden three
# assertions that all say the same thing. `test_degradation_contract.py` pins its VALUE.
GATE_MARK_EXPR = G.TESTS_MARK_EXPR
LOAD_FLAKE = "tests/test_edit_diagnostics.py::test_a_cross_file_break_names_its_caller[rename_function]"


def _gate(tmp_path, monkeypatch, *, workers=8, xdist=True):
    """A Gate whose subprocesses are scripted. `calls` records every pytest
    command line; `script` answers them in order."""
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(G, "_gate_cfg", lambda key, default: workers if key == "test_workers" else default)
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True)
    g = G.Gate("SM_T", wt, "b" * 40, live_root=tmp_path)
    g.python = Path(sys.executable)
    monkeypatch.setattr(g, "_child_env", lambda root=None, *, isolate_home=False: {})
    calls: list[list[str]] = []
    script: list[tuple[int, str]] = []

    def run(cmd, cwd=None, env=None, timeout=900.0):
        if cmd[1:3] == ["-c", "import xdist"]:
            return subprocess.CompletedProcess(cmd, 0 if xdist else 1, "", "")
        calls.append([str(c) for c in cmd])
        rc, out = script.pop(0)
        return subprocess.CompletedProcess(cmd, rc, out, "")
    monkeypatch.setattr(G, "_run", run)
    return g, calls, script


def test_a_full_run_uses_the_configured_workers_grouped_by_file(tmp_path, monkeypatch):
    g, calls, script = _gate(tmp_path, monkeypatch)
    script.append((0, SUMMARY_OK))
    r, text, counts = g._run_suite(None)
    assert r.returncode == 0 and calls == [[sys.executable, "-m", "pytest", "-q", "-m",
                                            GATE_MARK_EXPR, "-n", "8", "--dist", "loadfile"]]
    assert counts["workers"] == 8 and counts["passed"] == 5895
    # `loadfile`, not the default: a file's module-scoped fixtures (a booted
    # uvicorn, a temp repo) are built once per file, as they always were.


def test_a_failure_under_load_that_passes_serially_is_a_pass_and_is_named(tmp_path, monkeypatch):
    """The trial's own result: one test, three runs in three, green alone."""
    g, calls, script = _gate(tmp_path, monkeypatch)
    script.append((1, f"FAILED {LOAD_FLAKE} - AssertionError\n1 failed, 5895 passed, 13 skipped in 76s\n"))
    script.append((0, "53 passed in 1.39s\n"))
    r, _text, counts = g._run_suite(None)
    assert r.returncode == 0, "a fact about the box was taken for a fact about the change"
    assert calls[1] == [sys.executable, "-m", "pytest", "-q", "-m", GATE_MARK_EXPR,
                        "tests/test_edit_diagnostics.py"], "re-asked serially, that file only"
    assert counts["failed"] == 0 and counts["passed"] == 5896
    assert counts["parallel_only_failures"] == [LOAD_FLAKE]
    assert counts["serial_retry_files"] == ["tests/test_edit_diagnostics.py"]


def test_the_rungs_pass_line_says_how_it_ran_and_what_flinched(tmp_path, monkeypatch):
    g, _calls, script = _gate(tmp_path, monkeypatch)
    script.append((1, f"FAILED {LOAD_FLAKE} - AssertionError\n1 failed, 5895 passed, 13 skipped in 76s\n"))
    script.append((0, "53 passed in 1.39s\n"))
    ok, detail, data = g.rung_tests()
    assert ok is True
    assert "(8 workers)" in detail and "failed only under parallel load and passed serially" in detail
    assert LOAD_FLAKE in detail and data["parallel_only_failures"] == [LOAD_FLAKE]


def test_a_failure_that_survives_the_serial_re_run_is_judged_as_it_always_was(tmp_path, monkeypatch):
    """Real failures reach the same classifier, by the SERIAL run's node ids —
    the load flake beside them is named, and is not among them."""
    g, _calls, script = _gate(tmp_path, monkeypatch)
    real = "tests/test_mine.py::test_i_broke_this"
    script.append((1, f"FAILED {LOAD_FLAKE} - x\nFAILED {real} - assert False\n"
                      "2 failed, 5894 passed, 13 skipped in 71s\n"))
    script.append((1, f"FAILED {real} - assert False\n1 failed, 60 passed in 3s\n"))
    probed: list = []

    def at_base(python, live, base, node_ids, scratch, env):
        probed.append(list(node_ids))
        return set(), "probed 1 file(s) at base: none failing"
    monkeypatch.setattr(G, "_failures_at_base", at_base)
    ok, detail, data = g.rung_tests()
    assert ok is False and probed == [[real]]
    assert data["new_failures"] == [real] and data["failed"] == 1 and data["passed"] == 5895
    assert data["parallel_only_failures"] == [LOAD_FLAKE]
    assert "are new in this round" in detail and real in detail


def test_a_parallel_failure_that_names_no_file_re_runs_the_whole_suite_serially(tmp_path, monkeypatch):
    """A crashed worker, an INTERNALERROR, a timeout: nothing to re-ask by name."""
    g, calls, script = _gate(tmp_path, monkeypatch)
    script.append((3, "INTERNALERROR> worker 'gw3' crashed\n"))
    script.append((0, SUMMARY_OK))
    r, _text, counts = g._run_suite(None)
    assert r.returncode == 0 and "-n" not in calls[1] and calls[1][-1] == GATE_MARK_EXPR
    assert counts["serial_rerun"] == "whole suite"


def test_too_many_failing_files_is_a_broken_tree_not_load(tmp_path, monkeypatch):
    g, calls, script = _gate(tmp_path, monkeypatch)
    many = "".join(f"FAILED tests/test_f{n}.py::test_x - boom\n" for n in range(g.PARALLEL_RETRY_MAX_FILES + 1))
    script.append((1, many + "41 failed, 5000 passed in 60s\n"))
    script.append((1, many + "41 failed, 5000 passed in 600s\n"))
    r, _text, counts = g._run_suite(None)
    assert r.returncode == 1 and "-n" not in calls[1] and len(calls[1]) == 6
    assert counts["serial_rerun"] == "whole suite" and counts["failed"] == 41


@pytest.mark.parametrize("workers, xdist, why", [
    (8, False, "pytest-xdist is not importable in the gate's venv"),   # a venv without it still gates
    (1, True, ""), (0, True, ""), ("nonsense", True, ""),              # the default, and garbage, are serial
])
def test_it_is_serial_whenever_it_cannot_or_should_not_be_parallel(tmp_path, monkeypatch, workers, xdist, why):
    g, calls, script = _gate(tmp_path, monkeypatch, workers=workers, xdist=xdist)
    script.append((0, SUMMARY_OK))
    r, _text, counts = g._run_suite(None)
    assert r.returncode == 0 and "-n" not in calls[0] and counts["workers"] == 1
    assert counts.get("parallel_unavailable", "") == why


def test_a_partial_run_stays_serial(tmp_path, monkeypatch):
    """The changed test files a re-gate gets: seconds long, and answered by
    running exactly those files."""
    g, calls, script = _gate(tmp_path, monkeypatch)
    script.append((0, "20 passed in 2.1s\n"))
    g._run_suite(["tests/test_a.py", "tests/test_b.py"])
    assert "-n" not in calls[0] and calls[0][-2:] == ["tests/test_a.py", "tests/test_b.py"]


def test_the_real_thing_end_to_end_when_xdist_is_installed(tmp_path, monkeypatch):
    """A real pytest, real workers, a real failure that a serial re-run keeps:
    the parallel path reaches the same verdict the serial one would."""
    pytest.importorskip("xdist")
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(G, "_gate_cfg", lambda key, default: 2 if key == "test_workers" else default)
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True)
    (wt / "tests" / "test_ok.py").write_text("def test_fine():\n    assert True\n")
    (wt / "tests" / "test_bad.py").write_text("def test_broken():\n    assert False\n")
    g = G.Gate("SM_T", wt, "b" * 40, live_root=tmp_path)
    g.python = Path(sys.executable)
    r, text, counts = g._run_suite(None)
    assert r.returncode != 0 and counts["workers"] == 2
    assert G._failed_node_ids(text) == ["tests/test_bad.py::test_broken"]
    assert counts["failed"] == 1 and counts["serial_retry_files"] == ["tests/test_bad.py"]


# ── #1691 clause 5: a run that skipped its pins says so in the rung detail ───────
#
# The rung's pass line is the report a reviewer reads, and until now it could only
# state totals: `12757 passed, 1 xfailed, 31 skipped (8 workers)` (`~/.local/state/
# lloyd-automod/promotions.jsonl`, rung `tests`, 2026-09-27T23:29:44Z) says nothing
# about six geometry pins asserting nothing, because six skips sit inside a
# `PYTEST_MAX_SKIPPED` ceiling of 40 that live runs already spend 31 of
# (`gate.py:77-80`, suite totals). And the partial-run branch applies no floor at
# all, so a re-gate over just the pin file returned `ok=True` on "0 passed, 6
# skipped". What is added below is the finding the file itself prints.

#: A scripted instance of what `tests/dashboard_pins.py` prints, for the four nodes
#: below that feed the rung a summary line. Only the PREFIX is a contract — it is
#: derived from `G.PIN_FINDING_PREFIX` so a rename cannot leave these four nodes
#: asserting a string the rung no longer greps for — and the counts and the reason
#: suffix are this fixture's own invention, because these nodes are grading what the
#: RUNG does with a finding, not what the helper prints. The helper's real sentence,
#: with its real reason wording, is what the scratch-tree nodes at the foot of this
#: file assert, out of a real pytest subprocess.
FINDING = (f"{G.PIN_FINDING_PREFIX}: 6 of 6 pins in "
           "test_dashboard_responsive.py did not run — vite is not installed")


def test_a_green_run_that_skipped_its_pins_names_that_in_the_detail(tmp_path, monkeypatch):
    """The finding rides on a PASS, which is the whole point: the run is green, and
    green is exactly where nobody looks. It has to appear beside the counts, not
    instead of them — a reviewer still needs to see how much of the suite ran.
    """
    g, _calls, script = _gate(tmp_path, monkeypatch)
    script.append((0, f"{FINDING}\n"
                      "5895 passed, 19 skipped, 2 xfailed in 70.10s\n"))
    ok, detail, data = g.rung_tests()
    assert ok is True, detail
    assert "5895 passed" in detail, f"the pass counts went missing: {detail}"
    assert "19 skipped" in detail, detail
    assert G.PIN_FINDING_PREFIX in detail, (
        f"the pass line hides the no-execution finding: {detail}")
    assert "6 of 6 pins" in detail, (
        f"the detail dropped how many pins did not run: {detail}")


def test_a_finding_survives_the_serial_re_run_after_a_parallel_finch(tmp_path, monkeypatch):
    """A flinch re-runs the failed files serially and reports THAT run's counts, so a
    finding from the real run would vanish at the one moment a reviewer is most likely
    to read this line. `_re_run_parallel_failures` copies the counts dict rather than
    rebuilding it from the re-run's text, which is what keeps the finding alive.
    """
    g, _calls, script = _gate(tmp_path, monkeypatch)
    script.append((1, f"{FINDING}\nFAILED {LOAD_FLAKE} - AssertionError\n"
                      "1 failed, 5895 passed, 13 skipped in 76s\n"))
    script.append((0, "53 passed in 1.39s\n"))
    ok, detail, data = g.rung_tests()
    assert ok is True, detail
    assert "failed only under parallel load and passed serially" in detail, detail
    assert G.PIN_FINDING_PREFIX in detail, (
        f"the serial re-run dropped the finding: {detail}")
    assert data["parallel_only_failures"] == [LOAD_FLAKE]


def test_a_partial_run_carries_the_finding_since_it_applies_no_floor(tmp_path, monkeypatch):
    """The partial branch is the branch that actually ran nothing: a re-gate whose only
    changed test file is the pin file runs just that file, reports "0 passed, 6
    skipped", and skips every suite floor by construction. Without the finding that run
    is a green light over zero assertions.
    """
    g, _calls, script = _gate(tmp_path, monkeypatch)
    script.append((0, f"{FINDING}\n6 skipped in 0.10s\n"))
    ok, detail, data = g.rung_tests(only=["tests/test_dashboard_responsive.py"])
    assert ok is True and data.get("partial") is True, detail
    assert "6 skipped" in detail, detail
    assert G.PIN_FINDING_PREFIX in detail, (
        f"a partial run of the pin file alone reported no finding: {detail}")


def test_a_run_with_no_finding_adds_nothing_to_the_detail(tmp_path, monkeypatch):
    """The negative control, and the difference between a finding and noise: an ordinary
    green run's detail is exactly what it was before, so the clause earns its place in
    the line instead of padding every round with a sentence that never fires.
    """
    g, _calls, script = _gate(tmp_path, monkeypatch)
    script.append((0, SUMMARY_OK))
    ok, detail, _data = g.rung_tests()
    assert ok is True, detail
    assert G.PIN_FINDING_PREFIX not in detail, detail
    assert detail.startswith("5895 passed"), detail


def test_the_gate_scans_for_the_prefix_the_helper_actually_prints():
    """The seam between the two files, which cross a process boundary and a repo
    boundary: `tests/dashboard_pins.py` PRINTS the finding, `scripts/automod/gate.py`
    SCANS for it. A prefix that drifts on either side yields a rung detail that quietly
    stops carrying it — a guard reading its own missing input, undetectable from inside
    either file. So the constant is compared against the text the helper emits, read
    from the helper's own source.
    """
    helper = (Path(__file__).resolve().parent / "dashboard_pins.py").read_text()
    assert G.PIN_FINDING_PREFIX in helper, (
        f"gate.py scans for {G.PIN_FINDING_PREFIX!r}, which "
        "tests/dashboard_pins.py never prints — the rung detail would silently stop "
        "surfacing no-execution findings")
    assert G.PIN_FINDING_PREFIX in FINDING, (
        "the fixture text this file scripts no longer starts with the scanned prefix")


def test_the_gate_declares_the_frontend_only_where_it_made_it_reachable(tmp_path, monkeypatch):
    """The declaration is a promise, so the gate sets it exactly where the gate itself
    put `web/node_modules` there — `rung_frontend` symlinks it for a round that touched
    a `web/` path (`gate.py:1226-1235`) — and the check is the real `vite` BINARY, not
    the directory, because a symlink whose target vanished promises nothing.

    Both directions are asserted. Loose is the failure #1691's triage ruled out: a
    gate that declared everywhere would make every non-frontend round red for an
    install that was never going to be there. Strict is the silent one: a gate that
    never declared would leave a zero-assertion pin run a quiet pass, which is the
    defect this item was filed for. The env the child pytest ACTUALLY receives is
    recorded here, because `_gate`'s shared harness answers commands and drops env.
    """
    declared = G.FRONTEND_PINS_ENV
    seen: list[dict] = []

    def run(cmd, cwd=None, env=None, timeout=900.0):
        if cmd[1:3] == ["-c", "import xdist"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")   # serial path
        seen.append(dict(env or {}))
        return subprocess.CompletedProcess(cmd, 0, SUMMARY_OK, "")

    g, _calls, script = _gate(tmp_path, monkeypatch)
    monkeypatch.setattr(G, "_run", run)
    script.append((0, SUMMARY_OK))
    assert g.rung_tests()[0] is True
    assert seen, "the rung never launched pytest, so the env assertions below prove nothing"
    assert declared not in seen[-1], (
        "the gate declared the frontend for a worktree with no vite, which would "
        "fail every round that touched no web/ path")

    vite = g.worktree / "web" / "node_modules" / ".bin" / "vite"
    vite.parent.mkdir(parents=True)
    vite.write_text("#!/bin/sh\n")
    script.append((0, SUMMARY_OK))
    assert g.rung_tests()[0] is True
    assert len(seen) == 2, f"the second run was not launched: {len(seen)}"
    assert seen[-1].get(declared) == "1", (
        "the gate ran the suite in a worktree whose vite it had just made reachable "
        "without declaring it, so a pin run that skipped there stayed a silent pass")


def test_a_dangling_vite_symlink_declares_nothing(tmp_path, monkeypatch):
    """The strict direction, spelled out: `rung_frontend` links `node_modules` from the
    LIVE tree, and a link whose target was removed is not an install. A declaration
    made on the directory's existence would fail a round whose pins could not run —
    red for a promise nobody can keep, which is exactly what the fix must not do.
    """
    declared = G.FRONTEND_PINS_ENV
    seen: list[dict] = []

    def run(cmd, cwd=None, env=None, timeout=900.0):
        if cmd[1:3] == ["-c", "import xdist"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        seen.append(dict(env or {}))
        return subprocess.CompletedProcess(cmd, 0, SUMMARY_OK, "")

    g, _calls, script = _gate(tmp_path, monkeypatch)
    monkeypatch.setattr(G, "_run", run)
    target = tmp_path / "gone" / "node_modules"
    (g.worktree / "web").mkdir(parents=True, exist_ok=True)
    (g.worktree / "web" / "node_modules").symlink_to(target)   # target does not exist
    script.append((0, SUMMARY_OK))
    assert g.rung_tests()[0] is True
    assert declared not in seen[-1], (
        "a dangling web/node_modules symlink declared the frontend available")


def _helper_constant(name: str) -> str:
    """The value of a module-level `NAME = "literal"` in `tests/dashboard_pins.py`.

    Read out of the helper's own source rather than imported. This file is the
    gate's suite and the helper is the pytest side of a process boundary, so the
    two can never share an import — which is exactly why the strings they must
    agree on are worth a test. Importing it would also mean this file's fixture
    depended on loading candidate test code as a module.
    """
    src = (Path(__file__).resolve().parent / "dashboard_pins.py").read_text()
    m = re.search(rf'^{name} = "([^"]+)"', src, re.M)
    assert m, (f"{name} is no longer a plain module-level string constant in "
               "tests/dashboard_pins.py — the seam below has nothing to compare")
    return m.group(1)


def test_the_two_ends_of_the_seam_are_named_in_both_files():
    """The gate and the helper have to agree on two strings, and neither can see the
    other's copy: the finding prefix the rung SCANS for, and the environment variable
    the rung WRITES and the helper READS.

    The prefix already had a pin; the declaration did not, and that half is the
    quieter failure. Rename either end alone and nothing reddens: the gate stops
    declaring, every pin module takes its skip branch again, and the suite reports
    green over zero executed assertions — the exact defect #1691 was filed for, back
    by a rename rather than by a bug. The literals therefore live behind one name on
    each side (`gate.FRONTEND_PINS_ENV`, `dp.DECLARED_ENV`) and are compared here.
    """
    assert G.FRONTEND_PINS_ENV == _helper_constant("DECLARED_ENV"), (
        f"gate.py sets {G.FRONTEND_PINS_ENV!r} but tests/dashboard_pins.py reads "
        f"{_helper_constant('DECLARED_ENV')!r}: the declaration would reach a pytest "
        "child nobody reads, and every pin run goes back to a silent skip")
    assert G.PIN_FINDING_PREFIX == _helper_constant("FINDING"), (
        f"gate.py scans for {G.PIN_FINDING_PREFIX!r} but the helper prints "
        f"{_helper_constant('FINDING')!r}: the rung detail would quietly stop "
        "carrying no-execution findings")


def _scratch_gate_run(tmp_path, monkeypatch):
    """A scratch worktree with the shipped trio, driven through the MODIFIED gate by a
    real pytest subprocess — no scripted summary anywhere in this pair of nodes.

    The four nodes above feed the rung a text this file wrote, which is fair to
    distrust. Here the finding in the rung detail is text a real pytest printed about
    six pins it really skipped, and the counts beside it are the counts it really
    reported. `_run` is pointed at a subprocess because `Gate._run` uses the worktree's
    own venv, which a scratch tree has no: the code under test is this diff's
    `_run_suite`/`_tests_pass`, not the choice of interpreter.
    """
    trio = ("tests/dashboard_pins.py", "tests/test_dashboard_responsive.py",
            "scripts/maintenance/dashboard_mobile_probe.py")
    repo = Path(__file__).resolve().parent.parent
    root = tmp_path / "worktree"
    for rel in trio:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)

    # The scratch tree has no frontend, so the child must not inherit a declaration from
    # THIS process. `Gate._child_env` starts from `os.environ`, and the gate puts
    # `LLOYD_FRONTEND_PINS_AVAILABLE` on the whole suite child in any round that touched
    # `web/` — so a node here that inherited it would watch a six-pin file FAIL and blame
    # the code under test. Clearing it also makes the declared-direction node's
    # observation unambiguous: a declaration in that child's environment can only have
    # come from `_run_suite`.
    monkeypatch.delenv(G.FRONTEND_PINS_ENV, raising=False)
    monkeypatch.setattr(
        G, "_run",
        lambda cmd, cwd=None, env=None, timeout=900.0: subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, env=env, capture_output=True,
            text=True, timeout=timeout, check=False))
    g = G.Gate("SM_PINS", root, "b" * 40, live_root=tmp_path)
    g.python = Path(sys.executable)
    return g


def test_a_real_run_of_the_pin_file_puts_the_finding_in_the_full_run_detail(tmp_path, monkeypatch):
    """The branch the ladder actually takes today (`rung_tests` with no `only=`), fed
    by a real pytest subprocess. The suite floors are lowered to the scratch tree's
    size because they are whole-suite numbers and this tree holds fifteen tests; the
    assertion is about the finding riding on the same line as the counts, not about the
    floors.
    """
    g = _scratch_gate_run(tmp_path, monkeypatch)
    for name, value in (("PYTEST_MIN_COLLECTED", 1), ("PYTEST_MIN_PASSED", 1),
                        ("PYTEST_MAX_SKIPPED", 100)):
        monkeypatch.setattr(G, name, value)

    ok, detail, data = g.rung_tests()
    assert ok is True, detail
    assert data["passed"] >= 1, (
        f"the scratch run executed nothing, so its detail proves nothing: {detail}")
    assert "DASHBOARD_PINS_NOT_EXECUTED: 6 of 6 pins in " in detail, (
        "a real pytest run that skipped all six geometry pins did not put the finding "
        f"in the rung detail the reviewer reads:\n{detail}")
    assert " skipped" in detail, (
        f"the counts vanished from a line that now carries a finding: {detail}")


def test_a_real_partial_run_of_the_pin_file_carries_the_finding_too(tmp_path, monkeypatch):
    """Same real subprocess, the partial branch — the one that applies NO floor, and
    so is the run where a green light over zero assertions is easiest to grant.

    No ladder caller passes `only=` today (`rung_tests` is reached with `None` from the
    ladder list), which is exactly why this is pinned here rather than observed in a
    ledger row: the branch is live code a re-gate route can take, and it must already
    carry the finding when it does.
    """
    g = _scratch_gate_run(tmp_path, monkeypatch)
    g.report.changed_paths = ["tests/test_dashboard_responsive.py"]
    ok, detail, data = g.rung_tests(only=["tests/test_dashboard_responsive.py"])
    assert ok is True and data.get("partial") is True, detail
    assert "DASHBOARD_PINS_NOT_EXECUTED: 6 of 6 pins in " in detail, (
        f"a real partial run of the pin file alone reported no finding:\n{detail}")


def test_the_declaration_the_gate_makes_is_the_one_the_pytest_child_acts_on(tmp_path, monkeypatch):
    """The process boundary itself, in the direction that RAISES: the gate parent sets
    `FRONTEND_PINS_ENV` on the environment of the pytest child it spawns, and that
    child — a different process, which has never imported `gate.py` — reads the name
    out of `tests/dashboard_pins.py` and turns its own unexecuted pins into a failure.

    The review rung flagged this seam as unverified on round SM_20260928_000124, and
    the two nodes above only prove the UNDECLARED half: there the helper prints a line
    and skips, which it would do identically had the parent never set anything. This
    one cannot pass on a rename, because it asserts both ends of one handshake — the
    environment the real child was launched with, and the non-zero exit that
    environment caused. A name that matches only against a string read out of a file
    (`test_the_two_ends_of_the_seam_are_named_in_both_files`) is the cheap version of
    this check; this is the version that runs.

    Stops at `_run_suite` rather than going on to `rung_tests`: a non-zero child is
    already the rung's red branch (`if r.returncode != 0: return self._tests_failed(...)`,
    gate.py:1543), and `_tests_failed` would spend a second real run re-probing a base
    tree this scratch fixture does not have. The stub `vite` exits immediately, so the
    pins die at readiness — the same declared path, without needing a real frontend.
    """
    g = _scratch_gate_run(tmp_path, monkeypatch)
    # One worker: the parallel branch would answer a `import xdist` probe first, which
    # is a different subprocess and not the run whose environment is under test.
    monkeypatch.setattr(G, "_gate_cfg",
                        lambda key, default: 1 if key == "test_workers" else default)
    seen: list[dict] = []

    def run(cmd, cwd=None, env=None, timeout=900.0):
        seen.append(dict(env or {}))
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env,
                              capture_output=True, text=True, timeout=timeout,
                              check=False)
    monkeypatch.setattr(G, "_run", run)

    vite = g.worktree / "web" / "node_modules" / ".bin" / "vite"
    vite.parent.mkdir(parents=True, exist_ok=True)
    vite.write_text("#!/bin/sh\nexit 0\n")
    vite.chmod(0o755)

    done, text, counts = g._run_suite(None)
    assert done.returncode != 0, (
        "the gate declared the frontend reachable and the child still went green over "
        f"six unexecuted pins:\n{text[-1500:]}")
    assert seen, "_run_suite spawned no subprocess, so there was no seam to check"
    assert seen[-1].get(G.FRONTEND_PINS_ENV) == "1", (
        "the gate did not put its declaration on the environment of the pytest child "
        f"it spawned (keys naming a PIN: "
        f"{sorted(k for k in seen[-1] if 'PIN' in k)})")
    assert counts["pin_findings"], (
        f"a declared run that failed its pins produced no finding for the rung:\n{text[-1500:]}")
    assert "6 of 6 pins" in text, (
        f"the child failed, but not for the pins: {counts['pin_findings']}")
