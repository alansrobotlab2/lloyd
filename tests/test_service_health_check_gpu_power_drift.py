"""#1108: the GPU power clamp's deployed copies are checked against the tree.

`nvidia-power-limit.service` runs `nvidia-smi -pl`, which needs root, so its
unit and script are not symlinked from the tree like every other unit — they
are a hand `install` to `/etc/systemd/system/` and `/usr/local/sbin/`. Twice
in one week (`bb0dbca` 09-11, `36ba27e` 09-17) the tracked copy was edited and
the deployed one re-installed by hand the same minute, and nothing would have
said so had the second step been skipped: `grep -rl etc/systemd/system
scripts/ tests/` was empty. `check_deployed_copies` is the gate. Each test
names the acceptance clause it pins; the pairs are fed as temp files, so no
test reads root-owned paths or depends on what is installed on this box.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"


def _load_module():
    """Load the CLI script by path: it has no package and nothing imports it."""
    spec = importlib.util.spec_from_file_location("shc_gpu_power_drift", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()

UNIT = "nvidia-power-limit.service"
SCRIPT_NAME = "set-gpu-power-limit.sh"


def _pair(tmp_path, name, tracked_text, deployed_text):
    """A (tracked, deployed) pair under tmp_path; `deployed_text=None` leaves
    the deployed side absent."""
    tracked = tmp_path / "tree" / name
    deployed = tmp_path / "deployed" / name
    tracked.parent.mkdir(parents=True, exist_ok=True)
    deployed.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(tracked_text)
    if deployed_text is not None:
        deployed.write_text(deployed_text)
    return tracked, deployed


def _only(results):
    assert len(results) == 1, results
    return results[0]


def test_the_default_pairs_are_the_two_hand_installed_files():
    """Clauses 1 and 2 name the pairs; the constant has to name them too, or a
    green suite over temp files says nothing about the real deploy."""
    tracked = {t.name: (t, d) for t, d in shc.DEPLOYED_COPIES}
    assert set(tracked) == {UNIT, SCRIPT_NAME}
    assert tracked[UNIT][1] == Path("/etc/systemd/system") / UNIT
    assert tracked[SCRIPT_NAME][1] == Path("/usr/local/sbin") / SCRIPT_NAME
    for t, _ in shc.DEPLOYED_COPIES:
        assert t.is_file(), f"tracked copy {t} is not in the tree"


def test_an_identical_pair_reads_ok(tmp_path):
    pair = _pair(tmp_path, UNIT, "[Unit]\nx=1\n", "[Unit]\nx=1\n")
    r = _only(shc.check_deployed_copies([pair]))
    assert r["healthy"] is True
    assert r["status"].startswith("ok:")
    assert r["category"] == shc.DEPLOY_CATEGORY
    assert r["name"] == f"deployed:{UNIT}"


def test_a_differing_unit_reports_drift(tmp_path):
    """Clause 1: the unit pair, deliberately different by one Environment line
    — the exact edit #1107 made by hand on 09-17."""
    tracked, deployed = _pair(
        tmp_path, UNIT,
        "[Service]\nEnvironment=GPU_POWER_LIMIT_W_1=450\n",
        "[Service]\nEnvironment=GPU_POWER_LIMIT_W_1=400\n")
    r = _only(shc.check_deployed_copies([(tracked, deployed)]))
    assert r["healthy"] is False
    assert r["status"].startswith("drift:")
    assert str(deployed) in r["status"] and str(tracked) in r["status"], (
        "a drift line has to say which pair, or the reader cannot act on it")


def test_a_differing_script_reports_drift(tmp_path):
    """Clause 2: the script pair, proven the same way."""
    tracked, deployed = _pair(
        tmp_path, SCRIPT_NAME,
        "#!/bin/bash\nnvidia-smi -pl 450\n",
        "#!/bin/bash\nnvidia-smi -pl 400\n")
    r = _only(shc.check_deployed_copies([(tracked, deployed)]))
    assert r["healthy"] is False
    assert r["status"].startswith("drift:")
    assert str(deployed) in r["status"]


def test_an_absent_deployed_copy_is_a_named_failure_never_ok(tmp_path):
    """Clause 3: nothing installed at the target."""
    tracked, deployed = _pair(tmp_path, UNIT, "[Unit]\n", None)
    assert not deployed.exists()
    r = _only(shc.check_deployed_copies([(tracked, deployed)]))
    assert r["healthy"] is False
    assert r["status"].startswith("missing:")
    assert str(deployed) in r["status"]
    assert "ok" not in r["status"].split(":")[0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_an_unreadable_deployed_copy_is_a_named_failure_never_ok(tmp_path):
    """Clause 3: the target exists but cannot be read. A check that read
    nothing must not answer `ok` — that is the false comfort it exists to
    remove."""
    tracked, deployed = _pair(tmp_path, SCRIPT_NAME, "x\n", "x\n")
    deployed.chmod(0)
    try:
        r = _only(shc.check_deployed_copies([(tracked, deployed)]))
    finally:
        deployed.chmod(0o644)
    assert r["healthy"] is False
    assert r["status"].startswith("missing:")
    assert str(deployed) in r["status"]


def test_a_mixed_set_degrades_the_category_and_rides_both_formatters(tmp_path):
    """The result shape is `check_service`'s, so the summary and formatters need
    no special case: one ok and one drift is a degraded `deploy` category."""
    ok = _pair(tmp_path, UNIT, "a\n", "a\n")
    bad = _pair(tmp_path, SCRIPT_NAME, "a\n", "b\n")
    results = shc.check_deployed_copies([ok, bad])
    healthy = [r for r in results if r["healthy"]]
    assert len(healthy) == 1
    text = shc.format_text(results, {shc.DEPLOY_CATEGORY: "degraded"})
    assert "[✗] deployed:set-gpu-power-limit.sh" in text
    assert "[✓] deployed:nvidia-power-limit.service" in text
    payload = json.loads(shc.format_json(results, {shc.DEPLOY_CATEGORY: "degraded"}))
    assert payload["unhealthy"] == 1
    assert {s["name"] for s in payload["services"]} == {
        f"deployed:{UNIT}", f"deployed:{SCRIPT_NAME}"}


def test_the_cli_runs_the_pairs_under_its_own_category():
    """`--category deploy --format json` reaches the real pairs. Only the shape
    is asserted: what is installed on the box running the suite is not the
    suite's to decide, and every verdict is one of the three named words."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--category", shc.DEPLOY_CATEGORY, "--format", "json"],
        capture_output=True, text=True, timeout=30, check=True)
    payload = json.loads(proc.stdout)
    rows = {s["name"]: s for s in payload["services"]}
    assert set(rows) == {f"deployed:{UNIT}", f"deployed:{SCRIPT_NAME}"}
    for row in rows.values():
        assert row["category"] == shc.DEPLOY_CATEGORY
        assert row["status"].split(":")[0] in {"ok", "drift", "missing"}, row["status"]
        assert row["healthy"] is row["status"].startswith("ok:")
