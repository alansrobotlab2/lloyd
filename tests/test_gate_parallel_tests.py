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
import os
import shutil
import subprocess
import sys
import threading
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


#: `DASHBOARD_PINS_NOT_EXECUTED: N of M pins in <file> did not run …`
#:
#: Matched as a shape rather than as a literal with a number in it, because that number
#: is the count of the pin file's frontend pins and #2202 moved it the moment the seeded
#: pins landed. Three nodes below used to assert a literal carrying that number,
#: so the
#: pin added to that file would redden three gate-plumbing tests at once with messages
#: that blame the gate. What is worth pinning is that the finding NAMES EVERY PIN
#: (named == total, and total > 0, since a finding enumerating an empty set is the
#: vacuous zero again) — not how many pins happened to exist the day this was written.
_PINS_NAMED_RX = re.compile(r"DASHBOARD_PINS_NOT_EXECUTED: (\d+) of (\d+) pins in ")


def _all_pins_named(text: str) -> tuple[int, int]:
    """`(named, total)` from the finding printed in `text`, asserting there is one.

    Callers assert `named == total > 0`: a run in which nothing measured has to name
    every pin it skipped, and it cannot do that over a set it never enumerated.
    """
    m = _PINS_NAMED_RX.search(text)
    assert m, (
        "no pin-naming finding in the output at all — expected "
        f"{G.PIN_FINDING_PREFIX!r} followed by `N of M pins in …`, got:\n{text[-1200:]}")
    return int(m.group(1)), int(m.group(2))


def _scratch_gate_run(tmp_path, monkeypatch):
    """A scratch worktree with the shipped frontend-pin files, driven through the
    MODIFIED gate by a real pytest subprocess — no scripted summary anywhere in this
    pair of nodes.

    The four nodes above feed the rung a text this file wrote, which is fair to
    distrust. Here the finding in the rung detail is text a real pytest printed about
    pins it really skipped, and the counts beside it are the counts it really
    reported. `_run` is pointed at a subprocess because `Gate._run` uses the worktree's
    own venv, which a scratch tree has no: the code under test is this diff's
    `_run_suite`/`_tests_pass`, not the choice of interpreter.

    The tree is a plain directory, not a linked worktree, and it has no `web/` at all:
    that is the shape these two nodes have always had, and it is also the one case
    #2233's provisioning hook must answer with a plain no — no parent checkout to
    inherit an install from, so the pins keep skipping and the finding these nodes read
    is still printed. A tree that IS a linked worktree of a checkout with a vite is
    `test_the_parallel_rung_reaches_the_parent_checkouts_install` below.
    """
    shipped = ("tests/dashboard_pins.py", "tests/test_dashboard_responsive.py",
               "tests/frontend_deps.py",
               "scripts/maintenance/dashboard_mobile_probe.py")
    repo = Path(__file__).resolve().parent.parent
    root = tmp_path / "worktree"
    for rel in shipped:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)

    # The scratch tree has no frontend, so the child must not inherit a declaration from
    # THIS process. `Gate._child_env` starts from `os.environ`, and the gate puts
    # `LLOYD_FRONTEND_PINS_AVAILABLE` on the whole suite child in any round that touched
    # `web/` — so a node here that inherited it would watch the whole pin file FAIL
    # and blame
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
    named, total = _all_pins_named(detail)
    assert named == total > 0, (
        f"a real pytest run that skipped every pin reported {named} of {total} in the "
        f"rung detail the reviewer reads:\n{detail}")
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
    named, total = _all_pins_named(detail)
    assert named == total > 0, (
        f"a real partial run of the pin file alone named {named} of {total} pins:\n"
        f"{detail}")


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
        f"every unexecuted pin:\n{text[-1500:]}")
    assert seen, "_run_suite spawned no subprocess, so there was no seam to check"
    assert seen[-1].get(G.FRONTEND_PINS_ENV) == "1", (
        "the gate did not put its declaration on the environment of the pytest child "
        f"it spawned (keys naming a PIN: "
        f"{sorted(k for k in seen[-1] if 'PIN' in k)})")
    assert counts["pin_findings"], (
        f"a declared run that failed its pins produced no finding for the rung:\n{text[-1500:]}")
    named, total = _all_pins_named(text)
    assert named == total > 0, (
        f"the child failed, but not over every pin it skipped ({named} of {total}): "
        f"{counts['pin_findings']}")


# --------------------------------------------------------------------------- #
# #2233 — the ten dashboard pins must RUN in a round worktree, not skip, and the
# suite-wide skip ceiling of 40 is unchanged by that. Every node below is about
# one edge of that: the race the parallel rung creates (`tests/frontend_deps.py`
# is called by every xdist worker through `tests/conftest.py`), the real
# measurement in a real linked worktree, and the ceiling the pins now live under.
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_deps as FD  # noqa: E402


def _scratch_linked_pair(tmp_path):
    """A scratch parent checkout with a vite, plus a real linked worktree of it with none.

    The geometry of a round: a round's tree is `git worktree add <round>/home/lloyd` off
    the live checkout (`scripts/automod/worktree.py`), and `web/node_modules` is
    gitignored (`.gitignore:13`), so the new tree arrives without one. `git worktree
    add` is run for real because finding the parent IS reading this tree's `.git`
    pointer file — a plain `tmp_path` has no parent to find, and the race below would be
    a race about nothing.

    The parent's vite is a shell stub, not a dev server: these nodes are about who
    creates the link and in what order, and the vite's behaviour is irrelevant to that.
    Whether the linked install then MEASURES a real dashboard is the node below this
    one, which uses the real checkout and the real browser.
    """
    repo = Path(__file__).resolve().parent.parent
    parent = tmp_path / "parent"
    (parent / "tests").mkdir(parents=True)
    shutil.copy2(repo / "tests" / "frontend_deps.py", parent / "tests" / "frontend_deps.py")
    (parent / "web").mkdir()
    (parent / "web" / "index.html").write_text("<!doctype html><html></html>\n")
    (parent / ".gitignore").write_text("/web/node_modules\n")
    for args in (["init", "-q"], ["add", "-A"]):
        r = subprocess.run(["git", *args], cwd=str(parent), capture_output=True,
                           text=True, timeout=180)
        assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr[-400:]}"
    r = subprocess.run(
        ["git", "-c", "user.email=pins@example.invalid", "-c", "user.name=pins",
         "commit", "-q", "-m", "scratch parent with an ignored web/node_modules"],
        cwd=str(parent), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-400:]

    vite = parent / "web" / "node_modules" / ".bin" / "vite"
    vite.parent.mkdir(parents=True)
    vite.write_text("#!/bin/sh\nexit 0\n")
    vite.chmod(0o755)

    child = tmp_path / "child"
    r = subprocess.run(["git", "-C", str(parent), "worktree", "add", "--detach",
                        str(child), "HEAD"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-400:]
    assert not (child / "web" / "node_modules").exists(), (
        "the linked tree arrived with its own node_modules, so the race below has "
        "nothing to race for")
    return parent, child


def test_two_workers_racing_the_link_leave_exactly_one_creator(tmp_path):
    """Clause 4, the racing half: two workers importing the conftest at the same instant
    must not produce a `FileExistsError`, and must leave exactly one link.

    The tests rung runs `-n <workers> --dist loadfile`, and every xdist worker imports
    `tests/conftest.py`, which is where `#2233`'s provisioning hook is called — so the
    first two workers to boot ask for `<tree>/web/node_modules` in the same millisecond.
    `Path.symlink_to` on a path another worker just created raises `FileExistsError`,
    and an exception raised from a conftest at import is not a skip: it is a collection
    error in that worker, which the rung reads as failing tests the round did not write.
    That is the bug this node is written against, and it is why the helper swallows the
    error and reports the state instead.

    Exactly one creator per round is not a nicety, it is the proof that the swallow did
    not happen by both callers giving up: `linked` means "I made it", and two workers
    both making it is the double-create the check exists to prevent.
    """
    parent, child = _scratch_linked_pair(tmp_path)
    link = child / "web" / "node_modules"
    threads, rounds = 8, 12

    for round_no in range(rounds):
        if os.path.lexists(link):
            os.unlink(link)          # only ever the symlink this node created
        results: list[str] = []
        gate = threading.Barrier(threads)

        def ask():
            try:
                gate.wait(timeout=60)
                results.append(FD.ensure_node_modules_link(child))
            except BaseException as exc:                       # noqa: BLE001
                results.append(f"raised {type(exc).__name__}: {exc}")

        workers = [threading.Thread(target=ask) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=120)
        assert len(results) == threads, f"round {round_no}: a worker never reported"
        assert not any(r.startswith("raised") for r in results), (
            f"round {round_no}: a caller raised instead of reporting, which in a real "
            f"worker is a collection error: {results}")
        assert results.count(FD.LINKED) == 1, (
            f"round {round_no}: expected exactly one creator among {threads} racing "
            f"callers, got {results}")
        assert set(results) == {FD.LINKED, FD.PRESENT}, results
        assert link.is_symlink(), f"round {round_no}: the link is not a symlink: {results}"
        assert Path(os.readlink(link)) == (parent / "web" / "node_modules")


#: A second test file for a two-worker run, whose only job is to BE the second file.
#:
#: `--dist loadfile` hands one file to one worker, so with the pin file this probe is
#: what puts two pytest processes into the race for `<tree>/web/node_modules` — the race
#: the tests rung really creates, because each worker imports `tests/conftest.py` and the
#: conftest is what calls the hook. Each worker writes what it saw to `link-statuses/`
#: beside the tree under its own xdist name, so the driving process can read both answers
#: after the run instead of trusting one worker's view of the other.
LINK_PROBE_SRC = '''"""Generated by tests/test_gate_parallel_tests.py for #2233; not a shipped file."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_deps

ROOT = Path(__file__).resolve().parent.parent
STATUS = frontend_deps.ensure_node_modules_link(ROOT)
STATUSES = ROOT / "link-statuses"
STATUSES.mkdir(exist_ok=True)
NAME = "%s.txt" % os.environ.get("PYTEST_XDIST_WORKER", "serial")
(STATUSES / NAME).write_text(STATUS)


def test_frontend_link_probe_finds_the_parents_install_through_a_link():
    link = ROOT / "web" / "node_modules"
    assert STATUS in (frontend_deps.LINKED, frontend_deps.PRESENT), STATUS
    assert os.path.islink(str(link)), "the probe's tree has no link to the parent install"
    assert (link / ".bin" / "vite").exists(), (
        "the link exists but does not resolve to a vite: %s" % os.readlink(str(link)))
'''


@pytest.fixture()
def real_linked_worktree(tmp_path):
    """A real linked worktree of THIS checkout, with the working copies of the frontend files.

    A round's tree is exactly this: a linked worktree whose parent has the 641 MB
    `web/node_modules` the pins need and which arrives with none of its own. Nothing here
    copies or moves an install — the whole point is that the child reaches the parent's
    through one symlink.

    The worktree is checked out at `HEAD` (that is what `git worktree add` does, and what
    `scripts/automod/worktree.py` does for a round), then the five files the frontend pins
    are made of are overwritten from the working tree — the same thing
    `_scratch_gate_run` does, so the node measures the files under review rather than
    whatever was last committed, and stays runnable mid-edit.
    """
    repo = Path(__file__).resolve().parent.parent
    wt = tmp_path / "round-worktree"
    r = subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(wt),
                        "HEAD"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"git worktree add: {r.stderr[-600:]}"
    for rel in ("tests/conftest.py", "tests/dashboard_pins.py", "tests/frontend_deps.py",
                "tests/test_dashboard_responsive.py",
                "scripts/maintenance/dashboard_mobile_probe.py"):
        shutil.copy2(repo / rel, wt / rel)
    (wt / "tests" / "test_frontend_link_probe.py").write_text(LINK_PROBE_SRC)
    assert not (wt / "web" / "node_modules").exists(), (
        "git put a web/node_modules into a fresh worktree, which contradicts the "
        "premise #2233 was filed from")
    yield repo, wt
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)],
                   capture_output=True, text=True, timeout=600)


def _link_probe_command(parallel: bool) -> list[str]:
    """The tests rung's own invocation, narrowed to one pin and the link probe.

    `Gate._run_suite` builds the parallel branch as `-q -m <TESTS_MARK_EXPR> -n
    <workers> --dist loadfile`, and what the hook under test has to survive is exactly
    that shape; `test_a_full_run_uses_the_configured_workers_grouped_by_file` pins the
    argv the rung builds, so this is the same command with two files and a `-k` that
    selects one real pin (`test_no_section_overflows_its_box_on_a_phone[320]`) plus the
    probe, which is the difference between ten seconds in this node and ninety-four.
    Two files, because `--dist loadfile` gives one file to one worker, and two workers
    booting at once is the race.
    """
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if parallel:
        cmd += ["-n", "2", "--dist", "loadfile"]
    cmd += ["-m", GATE_MARK_EXPR,
            "tests/test_dashboard_responsive.py", "tests/test_frontend_link_probe.py",
            "-k", "(phone and 320) or frontend_link_probe", "-p", "no:cacheprovider"]
    return cmd


def test_the_parallel_rung_reports_the_pins_measured_in_a_linked_worktree(real_linked_worktree):
    """Clause 4, the measuring half: under the parallel invocation, in a real linked
    worktree of a checkout that HAS the frontend, the pins report themselves as measured
    and no worker dies on the link.

    This is #2233's acceptance check inside the suite. Before the change the same command
    in the same geometry reported `12 passed, 10 skipped` with
    `DASHBOARD_PINS_NOT_EXECUTED: 10 of 10 pins in test_dashboard_responsive.py did not
    run — web/node_modules has no vite`, and those ten skips on a suite already at 31-35
    are what failed every non-`web/` round against `PYTEST_MAX_SKIPPED = 40`. What is
    asserted now is `tests_skipped == 0` for the nodes that ran plus an empty
    `pin_findings`: a skip cannot satisfy either, which is the only reason the pair is
    worth ten seconds of real browser.

    The three assertions about the link and the workers are taken from the PARALLEL run
    unconditionally — a hook that raced badly cannot hide behind anything. The pin's own
    verdict is held to this file's standing doctrine: a parallel failure is not believed
    until the same files have been run serially, and that run is the verdict. That is not
    decoration. `web/node_modules` is shared with the live checkout by design, and so is
    vite's own dependency cache inside it, which the live dev server also writes; a child
    that arrives while that cache is being rebuilt takes a cold-start page load, and this
    file's opening paragraphs already say what to do about a result that is a fact about
    the box during the run rather than a fact about the diff. The serial re-ask cannot
    excuse a missing link, a raced link or a skipped pin on a warm box, because those
    assertions are all made above it and again on the serial run.
    """
    repo, wt = real_linked_worktree
    env = dict(os.environ)
    # Undeclared on purpose: this node measures the pins turning from skips into passes
    # in a tree that has no frontend of its own, which is a round's ordinary state. A
    # declaration would turn every skip into a failure and hide WHICH dependency stopped
    # the run.
    env.pop(G.FRONTEND_PINS_ENV, None)

    res = subprocess.run(_link_probe_command(True), cwd=str(wt), env=env,
                         capture_output=True, text=True, timeout=900)
    text = res.stdout + "\n" + res.stderr

    assert "FileExistsError" not in text, (
        "two workers raced the same symlink and one of them died on it — in a real "
        "rung that is a collection error, not a skip:\n"
        + "\n".join(ln for ln in text.splitlines() if "FileExistsError" in ln))

    link = wt / "web" / "node_modules"
    assert link.is_symlink(), f"no link was created in a real linked worktree:\n{text[-1500:]}"
    # The target is the MAIN checkout, which is what a linked worktree's `.git` pointer
    # names even when the worktree was added from another linked worktree: a round's tree
    # is a linked worktree of the live checkout, so a round's `web/node_modules` lands on
    # the live install directly instead of on another symlink.
    main = FD.main_checkout(wt)
    assert main is not None, (
        f"{wt} is not a linked worktree any more, so this node has lost the geometry "
        "#2233 is about")
    assert Path(os.readlink(link)) == (main / "web" / "node_modules"), (
        f"the link points at {os.readlink(link)!r}, not at the main checkout's install")
    assert (link / ".bin" / "vite").exists(), (
        f"the link resolves to an install with no vite: {os.readlink(link)!r}")

    statuses = {p.name: p.read_text() for p in (wt / "link-statuses").glob("*.txt")}
    assert statuses, f"no worker recorded what it saw, so the probe never ran:\n{text[-1500:]}"
    assert any(name.startswith("gw") for name in statuses), (
        f"nothing ran in an xdist worker (files: {sorted(statuses)}), so this was not a "
        "parallel run at all")
    seen = sorted(statuses.values())
    # What each worker can honestly report is the STATE it found, not who made it: the
    # conftest that creates the link runs at import, before this probe module is even
    # collected, so the worker that won the race may well report `present` a moment later.
    # The exactly-one-creator claim is the threaded node's, where it is observable and
    # deterministic. What must hold here is that no worker fell back to `unavailable` —
    # the answer a hook that could not find the main checkout would leave both files
    # skipping over — and that neither of them died.
    assert FD.UNAVAILABLE not in seen, (
        f"a worker could not reach the parent checkout's install: {statuses}")
    assert set(seen) <= {FD.LINKED, FD.PRESENT}, statuses

    counts = G._parse_pytest_summary(text)
    if res.returncode != 0 or counts["tests_skipped"] or counts["passed"] != 2:
        # Only a box with no Chromium may excuse the parallel run. A pin that skipped
        # under `-n 2` WHILE THE BROWSER WAS PRESENT is exactly the failure clause 4
        # exists to catch — a link that only works for one worker, say — and re-asking
        # serially over that would excuse it. So the excuse has to be named first.
        assert "chromium is not available" in text, (
            f"the parallel run did not answer for itself ({counts}) and did not stop at "
            f"the browser either, so no re-ask may excuse it:\n{text[-2500:]}")
        serial = subprocess.run(_link_probe_command(False), cwd=str(wt), env=env,
                                capture_output=True, text=True, timeout=900)
        stext = serial.stdout + "\n" + serial.stderr
        scounts = G._parse_pytest_summary(stext)
        assert serial.returncode == 0, (
            f"the parallel run was not green ({counts}) and the serial re-ask, which is "
            f"the verdict, was not green either:\n--- parallel ---\n{text[-2000:]}"
            f"\n--- serial ---\n{stext[-2000:]}")
        counts, text = scounts, stext

    assert counts["tests_skipped"] == 0, (
        f"the pins skipped again — the whole of #2233: {counts}\n{text[-2000:]}")
    assert counts["passed"] == 2, (
        f"expected the selected pin and the probe to pass, got {counts}\n{text[-2000:]}")
    assert counts["pin_findings"] == [], (
        f"the ledger still says a pin went unexecuted: {counts['pin_findings']}")


def test_the_skip_ceiling_the_pins_now_live_under_is_still_forty(tmp_path, monkeypatch):
    """Clause 5: #2233 removes ten skips, it does not move the ceiling — and the ceiling
    still bites at forty-one.

    This matters because the ceiling is the thing three sibling items (#2205, #2211,
    #2214) proposed raising to 55 instead, and #2214's `PYTEST_MAX_SKIPPED = 55`
    (`dbfddc60`) is a human-promotion route this change is the alternative to. Running
    the pins is the route that keeps the number at 40, so the number being 40 is part of
    this diff's contract, not an absence of change: if a later edit quietly widened it,
    the round that did it would have to say so here.

    Forty-one skips fails and forty passes, both read out of the rung's own sentence, so
    the boundary is pinned at the value rather than beside it. The vacuity guard's
    textual mutant — `scripts/maintenance/guard_vacuity.py::
    _m_mutate_pytest_skip_ceiling`, which rewrites the constant to 10**12 and requires
    that SOME checker then BLOCK — stays in force because the constant is still written
    the one way it matches, and `tests/test_guard_vacuity.py` runs that mutant over the
    real tree on every round.
    """
    g, _calls, _script = _gate(tmp_path, monkeypatch)
    assert G.PYTEST_MAX_SKIPPED == 40, (
        "#2233's fix is that the pins run, not that the ceiling rises; raising it here "
        "would be the #2214 route wearing this item's clothes")

    def counts_with(skipped: int) -> dict:
        return {"passed": 5800, "failed": 0, "errors": 0, "xfailed": 0,
                "tests_skipped": skipped, "collected": 5800 + skipped,
                "workers": 8, "parallel_only_failures": 0}

    ok, detail, _data = g._tests_pass(counts_with(G.PYTEST_MAX_SKIPPED + 1), None, {})
    assert ok is False, (
        f"{G.PYTEST_MAX_SKIPPED + 1} skipped passed a rung whose limit is "
        f"{G.PYTEST_MAX_SKIPPED}: {detail}")
    assert f"{G.PYTEST_MAX_SKIPPED + 1} tests skipped (limit {G.PYTEST_MAX_SKIPPED})" in detail, (
        f"the rung failed, but not with the ceiling sentence a reviewer reads: {detail}")

    ok2, detail2, _data2 = g._tests_pass(counts_with(G.PYTEST_MAX_SKIPPED), None, {})
    assert ok2 is True, (
        f"a run at exactly the ceiling — the shape a suite at its non-pin baseline "
        f"lands in once the pins execute — was failed: {detail2}")
