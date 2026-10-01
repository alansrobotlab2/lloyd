"""`_FALLBACK_GROUPS` must agree with the tracked supervisord conf.d (#1940).

`app.supervisor_client.qualify()` asks supervisord which group a program is in;
when the socket is unreachable it falls back to a hand-maintained copy. A bare
name for a grouped program is `Fault 10 BAD_NAME` (HTTP 500 on
`POST /api/services/action` before 2026-09-06), so a regrouping in conf.d that
the copy does not follow breaks every service action exactly when supervisord
is hardest to reach. The comment above the dict cites this file as the guard;
until #1940 the file did not exist.

The expected mapping is read from conf.d, never restated here.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

from app import supervisor_client

ROOT = Path(__file__).resolve().parent.parent
CONF_D = ROOT / "agent-services" / "supervisor" / "conf.d"
CLIENT_SRC = ROOT / "app" / "supervisor_client.py"


def _conf_groups() -> tuple[dict[str, str], set[str]]:
    """({program: group} for every `[group:*]`, every `[program:*]` name)."""
    grouped: dict[str, str] = {}
    programs: set[str] = set()
    for conf in sorted(CONF_D.glob("*.conf")):
        cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
        cp.read(conf)
        for section in cp.sections():
            kind, _, name = section.partition(":")
            if kind == "program":
                programs.add(name)
            elif kind == "group":
                for prog in cp[section].get("programs", "").split(","):
                    if prog.strip():
                        grouped[prog.strip()] = name
    return grouped, programs


def test_conf_d_declares_at_least_one_group():
    """A parse that found nothing would make every assertion below vacuous."""
    grouped, programs = _conf_groups()
    assert grouped, f"no [group:*] section parsed under {CONF_D}"
    assert programs, f"no [program:*] section parsed under {CONF_D}"


def test_every_fallback_entry_names_the_group_conf_d_declares():
    grouped, _ = _conf_groups()
    wrong = {
        prog: (group, grouped.get(prog))
        for prog, group in supervisor_client._FALLBACK_GROUPS.items()
        if grouped.get(prog) != group
    }
    assert not wrong, f"_FALLBACK_GROUPS disagrees with conf.d (fallback, conf.d): {wrong}"


def test_every_grouped_program_is_in_the_fallback():
    grouped, _ = _conf_groups()
    missing = {
        prog: group
        for prog, group in grouped.items()
        if supervisor_client._FALLBACK_GROUPS.get(prog) != group
    }
    assert not missing, f"conf.d groups these programs but _FALLBACK_GROUPS does not: {missing}"


def test_the_fallback_qualifies_names_when_supervisord_is_unreachable(monkeypatch):
    def unreachable():
        raise supervisor_client.SupervisordUnreachable("no socket")

    monkeypatch.setattr(supervisor_client, "_supervisor_proxy", unreachable)
    grouped, programs = _conf_groups()
    for prog, group in grouped.items():
        assert supervisor_client.qualify(prog) == f"{group}:{prog}"
    ungrouped = sorted(programs - set(grouped))
    assert ungrouped
    assert supervisor_client.qualify(ungrouped[0]) == ungrouped[0]


def test_the_test_file_the_fallback_comment_cites_exists():
    src = CLIENT_SRC.read_text()
    head = src[: src.index("_FALLBACK_GROUPS = {")]
    comment = head[head.rindex("\n\n"):]
    cited = re.findall(r"`(tests/[\w/.-]+\.py)`", comment)
    assert cited, "the _FALLBACK_GROUPS comment no longer cites its guard"
    for rel in cited:
        assert (ROOT / rel).is_file(), f"{rel} is cited by app/supervisor_client.py but is not on disk"
