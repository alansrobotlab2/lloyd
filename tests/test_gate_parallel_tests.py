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
