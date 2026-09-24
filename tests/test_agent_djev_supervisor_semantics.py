"""djev's supervisor conf must not call a doomed boot RUNNING (#1359).

`startsecs` is the only readiness supervisord knows: a process alive that long
is RUNNING, and a later exit is respawned with a fresh `startretries` budget, so
it can never go FATAL. At `startsecs=30` every one of nine OOM boots on
2026-09-21 cleared it (each died at spawn+65..155 s), and supervisor reported
djev RUNNING for 57 minutes of a crash loop. The value must stay above the
measured worst-case time-to-death, and the conf must say why beside it.
"""

from __future__ import annotations

import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "agent-services" / "supervisor" / "conf.d" / "agent-djev.conf"

#: The latest death observed after spawn on 2026-09-21 (supervisord.log).
MEASURED_TIME_TO_DEATH_S = 155


def _program() -> configparser.SectionProxy:
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
    cp.read(CONF)
    return cp["program:agent-djev"]


def test_startsecs_is_above_the_measured_time_to_death():
    assert int(_program()["startsecs"]) > MEASURED_TIME_TO_DEATH_S


def test_startretries_can_trip_to_fatal():
    """A boot that dies inside startsecs is a failed start; a finite retry
    budget is what turns a crash loop into FATAL instead of a RUNNING label."""
    assert 1 <= int(_program()["startretries"]) <= 5


def test_the_conf_states_the_measurement_and_the_cold_boot_ceiling():
    text = CONF.read_text()
    head = text.split("startsecs=", 1)[0]
    assert "#1359" in head and "time-to-death" in head
    assert "cold" in head.lower() and "FATAL" in head
