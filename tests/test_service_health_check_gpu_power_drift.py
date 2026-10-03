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



# ---------------------------------------------------------------------------
# #2135: the LIVE wattage versus the wattage the unit DECLARES
# ---------------------------------------------------------------------------
#
# The section above grades `check_deployed_copies`, which compares sha256 of the
# tracked unit against the installed one. It structurally cannot see the failure
# this one exists for: a hand `nvidia-smi -pl 400` leaves both files byte-perfect
# and moves the silicon, which happened twice on this box and was invisible both
# times (#1107). Declared watts come from systemd — the same source the fixer
# script reads at its :32 — and live watts from nvidia-smi's
# `enforced.power.limit`, a field `app/host_metrics.py` does not query (it publishes
# `power.limit`, the ceiling, which does not move when someone clamps a card).
#
# Nothing below shells out. Both sides are injected, so no assertion says anything
# about the hardware of the machine running the suite — that is this item's own
# clause 5, and the first real observation belongs to the owed-check run.

#: `(per-index dict, bare fallback)` — what `_declared_power_limits` returns on a
#: box wired like `agent-services/systemd/nvidia-power-limit.service:23-26`.
DECL_275_450_275 = ({0: 275.0, 1: 450.0, 2: 275.0}, 300.0)

#: What nvidia-smi answers on the same box, as the raw CSV `275.00 W` rows parse to.
LIVE_275_450_275 = {0: 275.0, 1: 450.0, 2: 275.0}

#: Everything the power row shares a default run with, stubbed to nothing so a
#: driven `main()` yields the power rows alone.
# Everything the power row would otherwise share the default run with. `main`
# builds its service rows out of `supervisor_fleet()` / `declared_programs()`, so an
# empty fleet empties that whole branch; the other three are the checks that shell
# out or read disk, each of which would put a row of its own in the payload.
SIBLING_CHECKS = ("check_deployed_copies", "check_ca_trust",
                  "check_unit_enabledness")


def _fresh():
    """The module-scope instance of the script that `shc` already returns.

    A function, not the `shc` fixture: three nodes below need the function-scoped
    `monkeypatch` and `capsys`, and pytest refuses a session-scoped fixture
    requested beside them. Same object either way — the loader at :29 execs the
    script once and this file already holds the result at :37 — and every stub put
    on it goes through `monkeypatch`, so nothing leaks into the section above.
    """
    return shc


def _drive(shc, monkeypatch, capsys, declared, live, argv_extra=()):
    """Run `main()` over injected sides and return its parsed `--format json`.

    Through `main()` rather than the check alone because the two things clause 5
    promises are `main`'s: that the row rides the DEFAULT run (no `--category`, so
    the whole wiring rule at :1169-1199 applies to it) and that it reaches the JSON
    payload with no new flag. The clause-1 and clause-2 summaries come from `main`'s
    own arithmetic for the same reason — a word this file typed would prove nothing.
    """
    for name in SIBLING_CHECKS:
        monkeypatch.setattr(shc, name, lambda *a, **k: [])
    monkeypatch.setattr(shc, "derived_supervisor_rows", lambda *a, **k: [])
    monkeypatch.setattr(shc, "_switched_off", lambda *a, **k: set())
    # `all` empty makes main's per-service loop body never run; the other keys stay
    # so `--category`'s argparse choices still name gpu-power.
    monkeypatch.setattr(shc, "CATEGORIES", dict(shc.CATEGORIES, **{"all": []}))
    monkeypatch.setattr(shc, "_declared_power_limits", lambda: declared)
    monkeypatch.setattr(shc, "_live_power_limits", lambda: live)
    monkeypatch.setattr(sys, "argv", [str(shc.__file__), "--format", "json", *argv_extra])
    shc.main()
    return json.loads(capsys.readouterr().out)


def test_a_hand_raised_card_names_its_index_live_and_declared_watts(monkeypatch,
                                                                    capsys):
    """#2135 clause 1: one mismatching card is an unhealthy row naming that index,
    the watts the driver enforces and the watts the unit declares, and the category
    summary reads `degraded`.

    Index 1 is the card that drifted here twice (#1107: clamped to 400 while the
    unit says 450), so the fixture moves exactly that one. The two matching cards
    are still returned: rows are per card, not per fault, so the payload names which
    cards did NOT move — the difference between one clamp slipping and the box being
    reconfigured.
    """
    shc = _fresh()
    doc = _drive(shc, monkeypatch, capsys, DECL_275_450_275,
                 {0: 275.0, 1: 400.0, 2: 275.0})
    by = {r["index"]: r for r in doc["services"] if r["index"] is not None}
    assert sorted(by) == [0, 1, 2], doc["services"]
    moved = by[1]
    assert moved["verdict"] == "drift" and moved["healthy"] is False, moved
    assert moved["exit_code"] == 1, moved
    assert moved.get("advisory") is None, "a real fault must never be advisory"
    assert (moved["live_w"], moved["declared_w"]) == (400.0, 450.0), moved
    assert moved["declared_source"] == "per-index", moved
    assert moved["category"] == shc.GPU_POWER_CATEGORY
    assert shc.DEPLOY_CATEGORY not in moved["category"], (
        "filing silicon drift under `deploy` would make the heading that means "
        "'the installed bytes are stale' also mean 'the wattage moved' (#1726)")
    out = moved["output"]
    assert "400" in out and "450" in out, out
    assert "nvidia-power-limit.service" in out and "set-gpu-power-limit.sh" in out, out
    assert doc["summary"][shc.GPU_POWER_CATEGORY] == "degraded", doc["summary"]
    assert shc.DEPLOY_CATEGORY not in doc["summary"], (
        "the power row must not create or move a `deploy` heading: its own rows are "
        "stubbed to nothing here, so any deploy word can only be the power row's: "
        "%s" % doc["summary"])


def test_matching_wattages_are_healthy_rows_only_with_no_root_and_no_gpu(monkeypatch,
                                                                         capsys):
    """#2135 clause 2: 275/450/275 live against 275/450/275 declared yields three
    healthy rows, no advisory row, and a `healthy` category word — from injected
    sides, so CI needs neither root nor a GPU.

    Neither reader is reached: `systemctl show` and `nvidia-smi` are read-only
    queries that need no root on this box, and here they are not even called, so
    nothing in this file depends on what the machine running it has in it.
    """
    shc = _fresh()
    doc = _drive(shc, monkeypatch, capsys, DECL_275_450_275, LIVE_275_450_275)
    rows = doc["services"]
    assert len(rows) == 3, rows
    assert all(r["healthy"] is True and r["verdict"] == "ok" for r in rows), rows
    assert all(r.get("advisory") is None for r in rows), rows
    assert doc["summary"][shc.GPU_POWER_CATEGORY] == "healthy", doc["summary"]
    assert doc["unhealthy"] == 0 and doc["advisory"] == 0, doc
    assert {r["index"] for r in rows} == {0, 1, 2}, rows


def test_the_bare_default_covers_an_index_and_two_decimals_are_not_drift():
    """#2135 clause 3, both halves.

    Fallback: index 2 has no `GPU_POWER_LIMIT_W_2`, so the bare
    `GPU_POWER_LIMIT_W=300` is what it is compared against — 300 W live is `ok`, and
    450 W live on that same card is a real mismatch that must not be waved through as
    "no declaration for this card, nothing to say". Its `declared_source` says
    `default`, which is what tells an operator which line of the unit to edit.

    Numeric: the live side arrives as `275.00 W` against a declared `275` —
    decimals and a ` W` suffix, exactly how nvidia-smi answers. A string compare
    reports drift on every healthy card of every healthy box forever, so the whole
    check would be noise. The fixtures below differ from the declared set only in
    that formatting, and the row still says `ok`.
    """
    shc = _fresh()
    declared = ({0: 275.0, 1: 450.0}, 300.0)
    rows = shc.check_gpu_power_limit(declared=lambda: declared,
                                     live=lambda: {0: 275.0, 1: 450.0, 2: 300.0})
    by = {r["index"]: r for r in rows}
    assert by[2]["verdict"] == "ok" and by[2]["healthy"] is True, by[2]
    assert (by[2]["declared_w"], by[2]["declared_source"]) == (300.0, "default"), by[2]
    assert by[0]["declared_source"] == "per-index" and by[1]["declared_source"] == "per-index"

    rows = shc.check_gpu_power_limit(declared=lambda: declared,
                                     live=lambda: {0: 275.0, 1: 450.0, 2: 450.0})
    moved = {r["index"]: r for r in rows}[2]
    assert moved["verdict"] == "drift" and moved["healthy"] is False, moved
    assert (moved["live_w"], moved["declared_w"], moved["declared_source"]) == (
        450.0, 300.0, "default"), moved

    # And the format-only case, straight off the reader's parse of real nvidia-smi
    # text: three cards, all matching, all `ok`.
    parsed = shc._live_power_limits(runner=lambda cmd, **kw: _Csv(
        "0, 275.00 W\n1, 450.00 W\n2, 300.00 W\n"))
    assert parsed == {0: 275.0, 1: 450.0, 2: 300.0}, parsed
    rows = shc.check_gpu_power_limit(declared=lambda: declared, live=lambda: parsed)
    assert [r["verdict"] for r in rows] == ["ok", "ok", "ok"], rows


class _Csv:
    """The one method `_live_power_limits` reads off a completed process."""

    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_the_reader_turns_a_real_csv_row_into_a_wattage():
    """The parse the whole probe rests on, pinned on its own against the reader.

    Clause 3's numeric comparison is only reached if the reader extracts `275.0`
    from `0, 275.00 W` and keys it by index, and a reader that got the field order
    or the suffix wrong would make every downstream assertion in this section
    vacuous — `check_gpu_power_limit` would be comparing an empty dict to an empty
    dict and reporting nothing at all. So the argv it asks for is pinned too: a
    query for `power.limit` instead of `enforced.power.limit` answers the ceiling,
    which does not move when someone clamps a card, and the check would report a
    healthy box while the silicon drifted.
    """
    shc = _fresh()
    seen = []

    def runner(cmd, **kw):
        seen.append(list(cmd))
        return _Csv("0, 275.00 W\n1, 450.00 W\n")

    assert shc._live_power_limits(runner=runner) == {0: 275.0, 1: 450.0}
    assert seen[0][:2] == ["nvidia-smi", "--query-gpu=index,enforced.power.limit"], seen
    assert "--format=csv,noheader" in seen[0], seen


def test_no_way_to_answer_is_unknown_neither_drift_nor_a_fault():
    """#2135 clause 4: five ways the probe cannot answer, and none of them is drift
    or a fault, so a machine with no GPU cannot go red.

    The declared read raising; the declared side answering with no power variable at
    all (a box that never installed the unit — not a box whose hardware moved);
    nvidia-smi missing from PATH; nvidia-smi answering non-zero (a wedged driver);
    and nvidia-smi listing no GPU. Each yields `verdict == "unknown"` rows with
    `advisory` set, which `_scored_rows` keeps out of the verdict arithmetic and
    `format_text` names on its own `Advisory:` line — so an unknown can neither be
    read as a fault nor silently graded as one. `--category gpu-power` on such a box
    must produce a `healthy`-or-`unknown` summary, never `unhealthy`.
    """
    shc = _fresh()

    def raiser(exc):
        def go():
            raise exc
        return go

    declared_ok = DECL_275_450_275
    live_ok = LIVE_275_450_275
    cases = [
        ("systemctl read raises", dict(declared=raiser(FileNotFoundError("systemctl")),
                                       live=lambda: live_ok)),
        ("unit declares no watts", dict(declared=lambda: None,
                                        live=lambda: live_ok)),
        ("nvidia-smi not on PATH", dict(declared=lambda: declared_ok,
                                        live=raiser(FileNotFoundError("nvidia-smi")))),
        ("nvidia-smi answers non-zero", dict(declared=lambda: declared_ok,
                                             live=raiser(RuntimeError("NVRM: down")))),
        ("no GPU enumerated", dict(declared=lambda: declared_ok,
                                   live=lambda: {})),
    ]
    for why, sides in cases:
        rows = shc.check_gpu_power_limit(**sides)
        assert rows, why
        for row in rows:
            assert row["verdict"] == "unknown", (why, row)
            assert row["advisory"] is True, (why, row)
            assert row["drift" if "drift" in row else "verdict"] != "drift", (why, row)
            assert shc._scored_rows([row]) == [], (why, row)
            assert shc._advisory_rows([row]) == [row], why
        text = shc.format_text(rows, {shc.GPU_POWER_CATEGORY: "unknown"})
        assert "Advisory:" in text, (why, text)
        assert "[\u2717]" not in text, (why, text)


def test_the_row_rides_the_default_run_and_its_json_with_no_new_flag(monkeypatch,
                                                                     capsys):
    """#2135 clause 5, json half: the default run — no `--category`, no
    `--services`, no new flag — puts the power rows in the `--format json` payload.

    The sibling checks are stubbed to nothing so the payload is the probe's alone.
    Then the same drive with card 1 moved: the row appears unhealthy and
    `summary["gpu-power"]` reads `degraded`, which is the difference between a row
    that is wired into the verdict and one that merely prints. The `--category
    gpu-power` drive is the other half of the wiring rule: a `--services` ask must
    NOT get the row, which is what the loop at :1169-1199 is for.
    """
    shc = _fresh()
    doc = _drive(shc, monkeypatch, capsys, DECL_275_450_275, LIVE_275_450_275)
    assert [r["name"] for r in doc["services"]] == ["gpu-power:0", "gpu-power:1",
                                                   "gpu-power:2"], doc["services"]
    assert doc["summary"][shc.GPU_POWER_CATEGORY] == "healthy", doc["summary"]

    doc = _drive(shc, monkeypatch, capsys, DECL_275_450_275,
                 {0: 275.0, 1: 400.0, 2: 275.0})
    moved = [r for r in doc["services"] if r["index"] == 1][0]
    assert moved["healthy"] is False and moved["live_w"] == 400.0, moved
    assert doc["summary"][shc.GPU_POWER_CATEGORY] == "degraded", doc["summary"]

    # `check_service` is stubbed because the real one shells out to supervisorctl:
    # this assertion is about which rows main COLLECTS for an ask, not about what
    # supervisor answers.
    monkeypatch.setattr(shc, "check_service",
                        lambda name, spec: {"name": name, "status": "RUNNING",
                                            "healthy": True, "exit_code": 0,
                                            "output": "stubbed for the wiring rule",
                                            "category": "lloyd"})
    monkeypatch.setattr(sys, "argv", [str(shc.__file__), "--format", "json",
                                      "--services", "lloyd-backend"])
    shc.main()
    doc = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in doc["services"]] == ["lloyd-backend"], (
        "a `--services` ask must get supervisor rows only — the power row belongs to "
        "the whole-fleet run and to `--category gpu-power`, never to an ask that "
        "names programs: %s" % doc["services"])


def test_the_skill_names_the_command_that_shows_the_declared_watts():
    """#2135 clause 5, doc half: the skill documents the check and names the
    systemctl command that shows the declared side.

    Read from the vault, not the repo: the skill lives at
    `~/obsidian/skills/service-health-check/SKILL.md`, and its front matter's
    `repo_path` points at the repo copy. Skipped rather than failed when the vault
    is not mounted, the way the skill-lint tests read the library.

    The reason is the one every row in this file exists for: the probe is worthless
    if nobody knows the declared side is readable without root. An operator told
    "the GPU power drifted" who does not know that command reaches for
    `nvidia-smi -q -d POWER` alone, which answers one of the two numbers and so
    cannot tell drift apart from a unit that was always wrong.
    """
    path = Path.home() / "obsidian" / "skills" / "service-health-check" / "SKILL.md"
    if not path.is_file():
        pytest.skip("vault not mounted")
    text = path.read_text()
    assert "systemctl show nvidia-power-limit.service --property=Environment" in text
    assert "enforced.power.limit" in text
    assert "gpu-power" in text
    assert "--format json" in text


def test_the_declared_side_is_read_through_the_shell_guard():
    """The other reader's seam, pinned beside the live one.

    The declared watts arrive through `scripts/maintenance/read-gpu-power-limit.sh`
    rather than a call made in this file, and that is not a style choice:
    `tests/test_unit_enabledness_route.py::test_no_enabledness_assertion_is_reimplemented_in_python`
    walks `service_health_check.py`'s string constants and refuses the systemd verb in
    any of them, because a unit question answered in Python is a second
    implementation of one with no guard in front of it (#1951, the same route
    `check_unit_enabledness` takes). So this node pins the three things that route has
    to keep doing: it runs `bash <the guard>`, it parses the guard's
    `Environment=` answer into per-index watts plus the bare default, and a non-zero
    answer — the unit not being installed is the ordinary one — yields `None`, which
    the check turns into an unknown row and never into drift.

    Nothing here reads the host's unit state: the runner is a stub, on purpose, for
    the same reason the guard itself takes a stand-in under pytest.
    """
    shc = _fresh()
    seen = []

    def runner(cmd, **kw):
        seen.append(list(cmd))
        return _Csv(DECL_275_450_275_TEXT)

    assert shc._declared_power_limits(runner=runner) == DECL_275_450_275
    assert seen[0] == ["bash", str(shc.GPU_POWER_GUARD)], seen
    assert shc.GPU_POWER_GUARD.name == "read-gpu-power-limit.sh", seen

    assert shc._declared_power_limits(
        runner=lambda cmd, **kw: _Csv("Environment=GPU_POWER_LIMIT_W=300\n")) == (
        {}, 300.0)
    assert shc._declared_power_limits(
        runner=lambda cmd, **kw: _Csv("", returncode=1)) is None
    assert shc._declared_power_limits(
        runner=lambda cmd, **kw: _Csv("MainPID=0\n")) == ({}, None)
    assert shc._declared_power_limits(
        runner=lambda cmd, **kw: (_ for _ in ()).throw(
            FileNotFoundError("no systemd here"))) is None


#: The guard's real answer shape on a box wired like the unit in the tree.
DECL_275_450_275_TEXT = ("Environment=GPU_POWER_LIMIT_W=300 GPU_POWER_LIMIT_W_0=275 "
                         "GPU_POWER_LIMIT_W_1=450 GPU_POWER_LIMIT_W_2=275\n")
