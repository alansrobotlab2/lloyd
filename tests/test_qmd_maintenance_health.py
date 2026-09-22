"""#958: a no-op run of the qmd maintenance job still has to measure daemon health.

`scripts/maintenance/qmd_index_maintenance.py` decided its exit code from
`report.get("daemon_healthy", True)`, and the only call of `daemon_healthy()` sat
in the `finally` of the *mutating* section. A run that found nothing to prune and
nothing to embed therefore returned 0 having never probed retrieval, and `_emit`
printed no health line at all because the key was absent from the report. That is
the catalogued class — `graph-baseline.json`, `_is_dependency_met`'s
`if not dep_task: return True`, the dream-consolidation lock: **a check whose input
is absent defaults to a pass.**

The no-op path is the job's normal path, not a hypothetical. Measured at triage
(2026-09-17, re-run on the base commit of this round): the script printed
`prune needed      False   embed needed False` and `· dry-run` with
`EXIT_CODE=0` and no `daemon healthy` line. Task #81 runs it daily and has been
paying a manual workaround instead — `autonomy-runs/81/run_81_20260917_120039.md`
states "Because a no-work run never touches the daemon and prints no `daemon
healthy` line, I probed it independently rather than inferring health from exit 0"
and then records `status: success`.

Scope, stated because the neighbouring items cover the other halves: this is the
exit-code/measurement half. The artifact half (a skipped day leaving no dated JSON
under `_pipeline/reflection`) is #844 and is already landed — the report is written
on this path, and only the `--dry-run` path still declines to write. What the probe
proves is *"the daemon answers an HTTP request on :8181"*: `daemon_healthy()`
returns True on any `curl` exit 0, which includes the 405 the MCP endpoint answers
to a GET, so a listening-but-wedged daemon still passes and no test here claims
otherwise. A decisive retrieval probe (`qmd vsearch`) is a different change, as
#958 says.

Nothing in this file stops, starts or restarts the daemon, opens the live index, or
reads `~/.config/qmd/index.yml`. `daemon_healthy` is patched except in the one case
that exists to prove the health line is backed by a real `curl`, and that case
stubs `_sh` as well, so it is green on a machine with no qmd at all.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.maintenance import qmd_index_maintenance as m  # noqa: E402

#: A template/live pair with nothing to compare: identical contents, so
#: `config_drift` is 0 and no case here is secretly about #1298.
CONFIG = """\
collections:
  memory:
    path: /home/me/obsidian/memory
    pattern: "**/*.md"
"""

#: The line `_emit` prints for it — the label column `_emit` pads, then the value.
HEALTH_LINE = re.compile(r"^  daemon healthy\s+(True|False)$", re.M)


class _Supervisor:
    """Records every supervisorctl action `main()` would have taken."""

    def __init__(self):
        self.actions: list[str] = []

    def __call__(self, action: str, timeout: int = 120) -> tuple[int, str]:
        self.actions.append(action)
        return 0, "stubbed"


class _StubSh:
    """Records every subprocess `main()` would have run, and answers with `rc`.

    The claim under test on the real-probe cases is "the health line came from a
    `curl` to the daemon's port", which is only checkable on the command line.
    `rc=7` is curl's CONNECTION_REFUSED, i.e. the down case.
    """

    def __init__(self, rc: int = 0):
        self.rc = rc
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, timeout, env=None):
        self.cmds.append([str(c) for c in cmd])
        return self.rc, "stubbed"

    @property
    def curls(self) -> list[list[str]]:
        return [c for c in self.cmds if c[0] == "curl"]


def _no_op_run(monkeypatch, tmp_path, *, healthy: bool, argv=(), probe_calls=None,
               supervisor=None):
    """Run `main()` on a run with nothing to prune and nothing to embed.

    Forces `need_prune` False (orphan ratio 0, under both triggers) and
    `pending_embeddings` 0, which is the state triage measured live — so the run
    takes the early-exit branch and the *only* thing that can set `daemon_healthy`
    is the probe this run is meant to add. Returns `(rc, report, probe_calls)`,
    `report` read back off the dated file the run wrote, not from memory.
    """
    tmpl, live = tmp_path / "qmd-index.yml", tmp_path / "index.yml"
    tmpl.write_text(CONFIG)
    live.write_text(CONFIG)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", tmpl)
    monkeypatch.setattr(m, "LIVE_CONFIG", live)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {
        "orphan_ratio": 0.0, "vectors_orphaned": 0, "vectors_total": 40_000,
        "documents": 16_000,
    })
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)

    calls = probe_calls if probe_calls is not None else []

    def probe(retries: int = 10) -> bool:
        calls.append(retries)
        return healthy

    monkeypatch.setattr(m, "daemon_healthy", probe)
    monkeypatch.setattr(m, "supervisor", supervisor or _Supervisor())
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py", *argv])
    rc = m.main()
    reports = sorted((tmp_path / "reflection").glob("qmd-index-maintenance-*.json"))
    report = json.loads(reports[0].read_text()) if reports else None
    return rc, report, calls


# --- clause 1: the probe's boolean reaches the report ------------------------

def test_a_no_op_run_records_the_probe_in_the_report(monkeypatch, tmp_path):
    """Before: the key was never written on this path, so `main()`'s `True`
    default decided the exit code with nothing measured."""
    rc, report, calls = _no_op_run(monkeypatch, tmp_path, healthy=True)
    assert report is not None, "the dated JSON under REPORT_DIR is the readable verdict"
    assert report["need_prune"] is False and report["need_embed"] is False
    assert report["daemon_healthy"] is True
    assert len(calls) == 1, f"the value must come from exactly one probe, got {calls}"


def test_a_failing_probe_is_recorded_as_false_not_silently_dropped(monkeypatch, tmp_path):
    rc, report, calls = _no_op_run(monkeypatch, tmp_path, healthy=False)
    assert report["daemon_healthy"] is False
    assert len(calls) == 1


def test_the_exit_path_no_longer_reads_the_value_with_a_true_default():
    """The clause is about the default, so pin the default's absence in the source.

    `report.get("daemon_healthy", True)` is what made an unmeasured run read as
    healthy; with the probe on both paths the key is always set, so the exit line
    can index it directly and a missing key becomes a crash instead of a pass.
    """
    src = (ROOT / "scripts/maintenance/qmd_index_maintenance.py").read_text(encoding="utf-8")
    assert '.get("daemon_healthy"' not in src, (
        "the exit code must follow a measured boolean, never a defaulted one"
    )


# --- clause 2: the line printed to stdout equals the probe -------------------

@pytest.mark.parametrize("healthy,printed", [(True, "True"), (False, "False")])
def test_the_no_op_run_prints_the_health_line_of_its_probe(
        monkeypatch, tmp_path, capsys, healthy, printed):
    _no_op_run(monkeypatch, tmp_path, healthy=healthy)
    out = capsys.readouterr().out
    hit = HEALTH_LINE.search(out)
    assert hit, f"expected a '  daemon healthy <bool>' line, got:\n{out}"
    assert hit.group(1) == printed, f"the printed value must be the probe's, not a default:\n{out}"


def test_the_dry_run_shares_the_probe_and_still_writes_no_file(monkeypatch, tmp_path, capsys):
    """#958 asked for one early-exit branch, not bespoke dry-run semantics: a
    `--dry-run` now measures health too (a curl changes nothing), but its promise
    to write nothing still holds."""
    rc, report, calls = _no_op_run(monkeypatch, tmp_path, healthy=True, argv=("--dry-run",))
    assert rc == 0
    assert len(calls) == 1
    assert report is None, "--dry-run promises to change nothing"
    assert HEALTH_LINE.search(capsys.readouterr().out)


# --- clauses 3 and 4: the exit code follows the probe -----------------------

def test_a_failing_probe_on_a_no_op_run_exits_non_zero(monkeypatch, tmp_path):
    """The whole point: retrieval down on a day with no work must not be a
    green run."""
    rc, report, _ = _no_op_run(monkeypatch, tmp_path, healthy=False)
    assert rc == 1, f"an unhealthy daemon is the one non-zero outcome; got rc={rc}"
    assert report["daemon_healthy"] is False


def test_a_passing_probe_on_a_no_op_run_exits_zero(monkeypatch, tmp_path):
    rc, report, _ = _no_op_run(monkeypatch, tmp_path, healthy=True)
    assert rc == 0, "a healthy no-op run must not start failing — the probe, not the exit, was the gap"
    assert report["daemon_healthy"] is True


# --- clause 5: measuring is not mutating ------------------------------------

@pytest.mark.parametrize("healthy", [True, False])
def test_the_no_op_path_still_calls_no_supervisor_action(monkeypatch, tmp_path, healthy):
    """Probe, report, exit — nothing else. A down daemon on a no-work day is
    reported and exited on, not restarted: this run never touched the index, so a
    restart here would be an unexplained retrieval outage in the middle of a job
    that decided to do nothing. (The mutating path's `supervisor("restart")` is
    pinned by test_qmd_single_build.py and is untouched.)"""
    sup = _Supervisor()
    rc, report, _ = _no_op_run(monkeypatch, tmp_path, healthy=healthy, supervisor=sup)
    assert sup.actions == [], f"the no-op path mutated the daemon: {sup.actions}"
    assert report["daemon_healthy"] is healthy


# --- the cost of the new probe on the branch where it fails -----------------

def test_the_no_op_probe_does_not_inherit_the_mutating_path_s_ten_retries(
        monkeypatch, tmp_path):
    """`daemon_healthy(retries=10)` sleeps 3 s between tries, because the mutating
    path is waiting out a restart it just performed. The no-op path restarts
    nothing, so waiting ~30 s per night for a daemon that was already down is the
    cost #958 warned about; a few tries still ride out a transient blip."""
    calls: list[int] = []
    _no_op_run(monkeypatch, tmp_path, healthy=True, probe_calls=calls)
    assert len(calls) == 1
    assert calls[0] < 10, f"no-op probe waited the mutating path's {calls[0]} retries"
    assert calls[0] >= 2, "one try would turn a 3 s hiccup into a failed nightly run"


# --- the seam: the health line is backed by a real curl, not a literal -------

def test_the_no_op_health_line_comes_from_a_curl_to_the_daemon_port(
        monkeypatch, tmp_path, capsys):
    """Patching `daemon_healthy` can prove the wiring but not that anything was
    probed. Here the probe is the real one and only `_sh` is stubbed, so the
    assertion is on the command line: a `curl` at the daemon's MCP endpoint, and
    nothing else issued by a run that decided to do nothing."""
    stub = _StubSh(rc=0)
    monkeypatch.setattr(m, "_sh", stub)
    sup = _Supervisor()
    monkeypatch.setattr(m, "supervisor", sup)
    tmpl, live = tmp_path / "qmd-index.yml", tmp_path / "index.yml"
    tmpl.write_text(CONFIG)
    live.write_text(CONFIG)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", tmpl)
    monkeypatch.setattr(m, "LIVE_CONFIG", live)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {
        "orphan_ratio": 0.0, "vectors_orphaned": 0, "documents": 16_000})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])

    rc = m.main()
    out = capsys.readouterr().out
    assert rc == 0, f"rc={rc}\n{out}"
    assert stub.curls, f"the health line must be backed by a probe; commands: {stub.cmds}"
    assert m.DAEMON_PROBE_URL in stub.curls[0], stub.curls[0]
    assert len(stub.curls) == 1, "a passing probe must not keep retrying"
    assert [c for c in stub.cmds if c[0] != "curl"] == [], (
        "a no-op run issued a non-probe subprocess")
    assert sup.actions == []
    assert HEALTH_LINE.search(out).group(1) == "True"


def test_a_refused_curl_makes_the_no_op_run_fail_after_bounded_retries(
        monkeypatch, tmp_path, capsys):
    """The down case end to end inside the process boundary: curl exits non-zero,
    the probe retries a bounded number of times, the line says False, and the run
    exits 1 — without a restart and without writing anything but its report."""
    stub = _StubSh(rc=7)  # curl CONNECTION_REFUSED
    monkeypatch.setattr(m, "_sh", stub)
    sup = _Supervisor()
    monkeypatch.setattr(m, "supervisor", sup)
    # The retry wait is not what is under test, and 3 s x N would only slow CI.
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    tmpl, live = tmp_path / "qmd-index.yml", tmp_path / "index.yml"
    tmpl.write_text(CONFIG)
    live.write_text(CONFIG)
    monkeypatch.setattr(m, "TEMPLATE_CONFIG", tmpl)
    monkeypatch.setattr(m, "LIVE_CONFIG", live)
    monkeypatch.setattr(m, "REPORT_DIR", tmp_path / "reflection")
    monkeypatch.setattr(m, "inspect_index", lambda: {
        "orphan_ratio": 0.0, "vectors_orphaned": 0, "documents": 16_000})
    monkeypatch.setattr(m, "pending_embeddings", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["qmd_index_maintenance.py"])

    rc = m.main()
    out = capsys.readouterr().out
    assert rc == 1, f"a refused curl on a no-op run must not exit 0\n{out}"
    assert HEALTH_LINE.search(out).group(1) == "False"
    assert len(stub.curls) == m.NOOP_HEALTH_RETRIES, (
        f"expected {m.NOOP_HEALTH_RETRIES} bounded tries, got {len(stub.curls)}")
    assert sup.actions == [], "an unhealthy no-op run reports, it does not restart"
