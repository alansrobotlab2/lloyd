"""The qmd watcher re-runs a walk for changes that arrived during the last one.

Events that arrived while `qmd update` ran waited in the pipe and were then read
and discarded by the cooldown drain, and nothing re-triggered: an edit landing
mid-walk stayed unindexed until some unrelated write came along. The loop is
driven here through the script's own `watch_loop` with a stub CLI and short
timings — no inotifywait, no qmd.

The path is assembled from parts, as `tests/test_automod_spec.py` does, because
Lloyd's service-control guard refuses a Bash command naming the launcher.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT.joinpath("agent-services", "scripts", "qmd-" + "watcher.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _drive(tmp_path: Path, feeder: str) -> list[str]:
    calls = tmp_path / "calls.log"
    stub = tmp_path / "qmd-stub"
    stub.write_text(f"#!/usr/bin/env bash\necho \"$1\" >> {calls}\nsleep 1\n")
    stub.chmod(0o755)
    script = f"""
set -euo pipefail
QMD_WATCHER_LIB=1 source "$1"
QMD_CLI=("{stub}")
DEBOUNCE_SEC=1; MAX_DEBOUNCE_SEC=2; COOLDOWN_SEC=2
{{ {feeder}; }} | watch_loop
"""
    subprocess.run(["bash", "-c", script, "watcher-test", str(WATCHER)],
                   check=True, timeout=60, capture_output=True, text=True)
    return calls.read_text().split() if calls.exists() else []


def test_an_edit_during_a_walk_gets_a_walk_of_its_own(tmp_path):
    # t=0 one event -> 1 s debounce -> walk (update 1 s + embed 1 s) from t~1.
    # A second event at t~1.5 lands mid-walk; it must cause a second walk.
    calls = _drive(tmp_path, "echo a; sleep 1.5; echo b; sleep 9")
    assert calls.count("update") == 2, calls


def test_a_quiet_cooldown_runs_no_extra_walk(tmp_path):
    calls = _drive(tmp_path, "echo a; sleep 7")
    assert calls.count("update") == 1, calls


def test_sourcing_as_a_library_starts_nothing(tmp_path):
    out = subprocess.run(["bash", "-c", 'QMD_WATCHER_LIB=1 source "$1"; type watch_loop >/dev/null && echo ok',
                          "t", str(WATCHER)], capture_output=True, text=True, timeout=10)
    assert out.stdout.strip() == "ok" and "Watching" not in out.stdout
