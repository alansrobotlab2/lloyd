"""#1141 clause 4 — the services check has no inventory, and says so.

`SERVICES` was a ten-name list defined once and read nowhere, while
`check_services()` graded every line `supervisorctl status` printed. A reader of
the list believed the check was scoped to ten programs. It was not, and the fix
chosen is to delete the list rather than start honouring it: a program added
to supervisor next month is graded the day it appears, and nobody has to
remember to add it anywhere.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(os.environ.get("LLOYD_SHC_SCRIPT") or
              Path.home() / "obsidian" / "skills" / "system-health-check" / "system_health_check.py")


def test_there_is_no_services_inventory_in_the_script():
    assert re.search(r"\bSERVICES\b", SCRIPT.read_text()) is None


def test_a_program_nobody_declared_is_graded_and_named(tmp_path):
    stub = tmp_path / "supervisorctl"
    stub.write_text(textwrap.dedent("""\
        #!/bin/sh
        echo "lloyd-mc:lloyd-backend           RUNNING   pid 11, uptime 1:00:00"
        echo "agent-brand-new-thing            FATAL     Exited too quickly"
        echo "agent-voice-mode                 STOPPED   Sep 14 12:00 PM"
        """))
    stub.chmod(0o755)
    env = {**os.environ, "LLOYD_HEALTH_SUPERVISORCTL": str(stub)}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--format", "json",
                           "--component", "services"],
                          capture_output=True, text=True, env=env, timeout=60)
    payload = json.loads(proc.stdout)
    rows = {r["name"]: r for r in payload["services"]["details"]}
    assert rows["agent-brand-new-thing"]["healthy"] is False, rows
    assert rows["lloyd-mc:lloyd-backend"]["healthy"] is True, rows
    assert rows["agent-voice-mode"]["expected_stopped"] is True, rows
    assert payload["overall_status"] == "degraded" and proc.returncode == 3
    text = subprocess.run([sys.executable, str(SCRIPT), "--component", "services"],
                          capture_output=True, text=True, env=env, timeout=60).stdout
    assert "[✗] agent-brand-new-thing — FATAL" in text, text
