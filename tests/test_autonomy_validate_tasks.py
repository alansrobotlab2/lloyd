"""`validate_tasks.py` lints `frequency:` against the scheduler's own domain (#815).

A `frequency` outside `autonomy.FREQUENCY_INTERVALS` with no `runs_per_day`
resolves to no interval, and a task with no interval is never due — until
2026-09-24 without a line in any log. #24 carries `frequency: 6x-daily` and
dispatches only because `runs_per_day: 6` is read first; delete that one field
and a nightly pipeline is parked unannounced. The linter is where that is
cheapest to catch, and it imports the domain rather than restating it.

Runs the script as a subprocess like `tests/test_validate_tasks_linter.py`,
whose `Linter` fixture it reuses: the exit code is the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_validate_tasks_linter import Linter  # noqa: E402

LIVE_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


def _task(linter: Linter, filename: str, **fields) -> None:
    linter.add(filename, id=int(filename.split("-")[0]), status="up_next",
               skill_name="heartbeat", **fields)


def test_a_frequency_outside_the_domain_with_no_runs_per_day_warns_naming_the_file(tmp_path):
    linter = Linter(tmp_path)
    _task(linter, "24-pipeline.md", frequency="6x-daily")
    rc, out = linter.run()
    assert rc == 0, out
    assert "24-pipeline.md: frequency '6x-daily'" in out, out
    assert "runs_per_day" in out and "never dispatch" in out, out
    # The warning names the domain it was checked against.
    from autonomy import FREQUENCY_INTERVALS
    for word in FREQUENCY_INTERVALS:
        assert word in out, (word, out)


def test_the_frequency_warning_fails_the_strict_rung(tmp_path):
    linter = Linter(tmp_path)
    _task(linter, "24-pipeline.md", frequency="6x-daily")
    rc, out = linter.run("--strict")
    assert rc == 2, out


@pytest.mark.parametrize("fields", [
    {"frequency": "6x-daily", "runs_per_day": 6},
    {"frequency": "daily"},
    {"frequency": "every-15min"},
    {"runs_per_day": 4},
], ids=["outside-with-rpd", "daily", "every-15min", "rpd-only"])
def test_a_resolvable_schedule_raises_no_frequency_warning(tmp_path, fields):
    linter = Linter(tmp_path)
    _task(linter, "1-ok.md", **fields)
    rc, out = linter.run("--strict")
    assert rc == 0, out
    assert "frequency '" not in out, out


def test_the_check_reads_the_scheduler_domain_not_a_copy():
    src = (ROOT / "scripts" / "autonomy" / "validate_tasks.py").read_text()
    assert "from autonomy import FREQUENCY_INTERVALS" in src
    assert '"every-15min"' not in src, "the linter must not restate the vocabulary"


@pytest.mark.live_vault
def test_the_live_autonomy_dir_raises_no_frequency_warning_today():
    """Every live task file either uses a word in the domain or carries
    `runs_per_day` (#24). Other warning classes are somebody else's clause."""
    if not LIVE_AUTONOMY_DIR.is_dir():
        pytest.skip("live ~/obsidian/autonomy not readable from this tree")
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "autonomy" / "validate_tasks.py"),
         "--autonomy-dir", str(LIVE_AUTONOMY_DIR)],
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    offenders = [ln for ln in out.splitlines() if ": frequency '" in ln]
    assert offenders == [], offenders
