"""`service_health_check.py` is the caller `check-unit-enabledness.sh` lacked (#1989).

The guard landed in #1891 with a test and no route, so a placed-but-disabled timer
was still found only by someone remembering to run it. These nodes pin the route:
category `units`, graded rows, the script's own denominator, skipped for a
`--services` ask.

Nothing here executes the host `systemctl`. The real guard is never run: every node
either hands `check_unit_enabledness` a fake runner or points the
PYTEST_CURRENT_TEST-guarded `UNIT_ENABLEDNESS_SCRIPT` seam at a stand-in script that
prints what the real one prints and exits as told.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"
GUARD = ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh"


def _load():
    spec = importlib.util.spec_from_file_location("shc_units", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shc = _load()

# What the real guard prints (scripts/maintenance/check-unit-enabledness.sh): one
# `<unit>\t<verdict>` line per timer, the denominator, then a FAIL line per miss.
ALL_ENABLED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tenabled\n"
    "checked 2 timers: 2 enabled, 0 not enabled\n"
)
ONE_DISABLED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tdisabled\n"
    "checked 2 timers: 1 enabled, 1 not enabled\n"
    "FAIL beta.timer is not enabled — run: systemctl --user enable beta.timer (SETUP.md:1520)\n"
)
NOTHING_LOOKED_AT = (
    "FAIL 0 *.timer in /nowhere — the check looked at nothing, which is not a pass\n"
    "checked 0 timers: 0 enabled, 0 not enabled\n"
)


def _stand_in(tmp_path: Path, text: str, code: int) -> Path:
    """A script that prints `text` and exits `code`; it never calls systemctl."""
    out = tmp_path / f"out-{code}.txt"
    out.write_text(text, encoding="utf-8")
    stub = tmp_path / f"stand-in-{code}.sh"
    stub.write_text(f'#!/usr/bin/env bash\ncat "{out}"\nexit {code}\n', encoding="utf-8")
    return stub


def _run_main(monkeypatch, capsys, *argv) -> dict:
    """Drive `main()` with every other row source silenced, so only the route under
    test can reach a subprocess — and that one only through the stand-in."""
    monkeypatch.setattr(shc, "derived_supervisor_rows", lambda: [])
    monkeypatch.setattr(shc, "check_service",
                        lambda name, cfg: {"name": name, "status": "RUNNING", "healthy": True,
                                           "exit_code": 0, "output": "", "category": "stub"})
    monkeypatch.setattr(shc, "check_deployed_copies", lambda: [])
    monkeypatch.setattr(shc, "check_ca_trust", lambda: [])
    monkeypatch.setattr(shc, "_switched_off", lambda: set())
    monkeypatch.setattr(sys, "argv", ["service_health_check.py", "--format", "json", *argv])
    shc.main()
    return json.loads(capsys.readouterr().out)


def _unit_rows(doc: dict) -> list:
    return [r for r in doc["services"] if r["category"] == shc.UNITS_CATEGORY]


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    def use(text: str, code: int) -> Path:
        stub = _stand_in(tmp_path, text, code)
        monkeypatch.setenv("UNIT_ENABLEDNESS_SCRIPT", str(stub))
        return stub
    return use


def test_the_category_is_selectable_and_has_no_supervisor_program():
    """Clause 1: argparse takes its choices from CATEGORIES, so the key is all it needs."""
    assert shc.UNITS_CATEGORY in shc.CATEGORIES
    assert shc.CATEGORIES[shc.UNITS_CATEGORY] == []
    assert shc.UNITS_CATEGORY not in (shc.DEPLOY_CATEGORY, shc.CA_TRUST_CATEGORY)
    assert shc.UNIT_ENABLEDNESS == GUARD and GUARD.is_file()


def test_the_rows_come_from_running_the_guard_script(monkeypatch):
    """Clause 1: the argv is `bash <the tracked guard>`, and outside pytest the
    override is inert — an environment variable must not pick the guard on a live run."""
    monkeypatch.delenv("UNIT_ENABLEDNESS_SCRIPT", raising=False)
    assert shc._unit_enabledness_argv() == ["bash", str(GUARD)]

    monkeypatch.setenv("UNIT_ENABLEDNESS_SCRIPT", "/somewhere/else.sh")
    assert shc._unit_enabledness_argv() == ["bash", "/somewhere/else.sh"]
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert shc._unit_enabledness_argv() == ["bash", str(GUARD)]


def test_no_enabledness_assertion_is_reimplemented_in_python():
    """Clause 1: the health script never names systemctl or `is-enabled` in code.

    Comments and docstrings may explain; a string or name in executable code may not,
    which is what a second copy of the assertion would need.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docstrings]
    assert strings, "the walk found no string constant, so the check below is vacuous"
    for needle in ("systemctl", "is-enabled"):
        hits = [s for s in strings if needle in s]
        assert not hits, f"{needle!r} appears in code: {hits}"


def test_a_whole_fleet_run_and_the_category_carry_the_rows(monkeypatch, capsys, stand_in):
    """Clause 2: present on a default run and on `--category units`."""
    stand_in(ALL_ENABLED, 0)
    fleet = _run_main(monkeypatch, capsys)
    assert [r["name"] for r in _unit_rows(fleet)] == ["timer:alpha.timer", "timer:beta.timer"]

    only = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    assert [r["name"] for r in only["services"]] == ["timer:alpha.timer", "timer:beta.timer"]


def test_a_services_ask_and_another_category_get_no_unit_rows(monkeypatch, capsys, stand_in):
    """Clause 2: the DEPLOY/CA rule — an ask naming supervisor programs gets only those.

    The stand-in exits 1, so a row that leaked through would also be visible as a fault.
    """
    stand_in(ONE_DISABLED, 1)
    asked = _run_main(monkeypatch, capsys, "--services", "lloyd-backend")
    assert _unit_rows(asked) == []
    assert [r["name"] for r in asked["services"]] == ["lloyd-backend"]

    other = _run_main(monkeypatch, capsys, "--category", "lloyd")
    assert _unit_rows(other) == []


def test_exit_zero_is_healthy_and_graded(monkeypatch, capsys, stand_in):
    """Clause 3: a stubbed exit 0 yields healthy rows that are counted, not advisory."""
    stand_in(ALL_ENABLED, 0)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert rows and all(r["healthy"] for r in rows)
    assert all("advisory" not in r for r in rows)
    assert doc["summary"][shc.UNITS_CATEGORY] == "healthy"
    assert doc["advisory"] == 0 and doc["healthy"] == len(rows)


def test_a_non_zero_exit_degrades_the_category(monkeypatch, capsys, stand_in):
    """Clause 3: one timer the guard failed turns the word `degraded`, through the
    graded arithmetic and never through `advisory`."""
    stand_in(ONE_DISABLED, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = {r["name"]: r for r in _unit_rows(doc)}
    assert rows["timer:alpha.timer"]["healthy"] is True
    assert rows["timer:beta.timer"]["healthy"] is False
    assert rows["timer:beta.timer"]["status"].startswith("disabled")
    assert all("advisory" not in r for r in rows.values())
    assert doc["summary"][shc.UNITS_CATEGORY] == "degraded"
    assert doc["advisory"] == 0 and doc["unhealthy"] == 1


def test_a_guard_that_looked_at_nothing_is_never_healthy(monkeypatch, capsys, stand_in):
    """A non-zero exit that names no timer still produces a fault row."""
    stand_in(NOTHING_LOOKED_AT, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert [r["name"] for r in rows] == ["unit-enabledness"]
    assert rows[0]["healthy"] is False
    assert "looked at nothing" in rows[0]["status"]
    assert doc["summary"][shc.UNITS_CATEGORY] != "healthy"


def test_the_verdict_is_the_scripts_and_not_the_word_on_the_line():
    """The exit code and the script's own FAIL lines decide; Python does not judge
    the verdict word. A line reading `disabled` under exit 0 is still the script's pass."""
    class Answer:
        returncode = 0
        stdout = "alpha.timer\tdisabled\nchecked 1 timers: 1 enabled, 0 not enabled\n"
        stderr = ""
    rows = shc.check_unit_enabledness(runner=lambda *a, **k: Answer())
    assert [r["healthy"] for r in rows] == [True]


def test_a_guard_that_cannot_be_run_is_a_fault_row():
    def boom(*a, **k):
        raise OSError("no bash")
    rows = shc.check_unit_enabledness(runner=boom)
    assert len(rows) == 1 and rows[0]["healthy"] is False
    assert rows[0]["status"].startswith("unknown:")
    assert "advisory" not in rows[0]


@pytest.mark.parametrize("text,code,line", [
    (ALL_ENABLED, 0, "checked 2 timers: 2 enabled, 0 not enabled"),
    (ONE_DISABLED, 1, "checked 2 timers: 1 enabled, 1 not enabled"),
    ("a.timer\tenabled\nb.timer\tstatic\nc.timer\tmasked\n"
     "checked 3 timers: 1 enabled, 2 not enabled\n"
     "FAIL b.timer is not enabled — run: x\nFAIL c.timer is not enabled — run: x\n",
     1, "checked 3 timers: 1 enabled, 2 not enabled"),
])
def test_every_row_carries_the_scripts_own_denominator(monkeypatch, capsys, stand_in,
                                                       text, code, line):
    """Clause 4: the count in the status is the line the stand-in printed, verbatim."""
    stand_in(text, code)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert rows
    for row in rows:
        assert line in row["status"], row["status"]
    assert sum(1 for r in rows if not r["healthy"]) == text.count("\nFAIL ")
