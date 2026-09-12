"""The services-only health check has to be able to see qmd (#406).

`scripts/service_health_check.py` carried a TCP prober — `http_check`, wired to
an optional `port` key on each `SERVICES` entry — but no entry ever declared a
port, so that machinery had never once run. The skill that documents the script
(`~/obsidian/skills/service-health-check/SKILL.md`) meanwhile asserted a QMD row
on a port the daemon has never listened on, and told the agent to probe the
Mission Control frontend over plain HTTP on a port nothing binds.

So a qmd outage — the retrieval path behind every `vault_search` /
`vault_recall` — was invisible to the one check whose entire job is service
liveness, while the skill text claimed it was covered. Same class for the
frontend row: `curl http://localhost:3000/health` reports a dead service
whenever it is followed, because the frontend is an HTTPS vite dev server on
5173 (`httpsConfig` in `web/vite.config.ts`), and a plain-HTTP request fails on
the scheme alone (curl exit 52) — which had already been written up as
"frontend is down" while the user was typing into it.

Each test names the acceptance clause it pins.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"
SKILL = Path.home() / "obsidian" / "skills" / "service-health-check" / "SKILL.md"

QMD = "agent-qmd-daemon"
QMD_PORT = 8181


def _load_module():
    """Load the CLI script as a module.

    It has no package (no `scripts/__init__.py`) and nothing in the tree
    imports it — it is invoked as a subprocess from Bash. Loading it by path is
    how the unit-level tests reach `SERVICES` / `check_service` without
    inventing an import surface.
    """
    spec = importlib.util.spec_from_file_location("shc_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()


# ---------------------------------------------------------------- clause 1
# "JSON output of service_health_check.py --format json includes a service
#  named agent-qmd-daemon."
#
# Run the real CLI as a subprocess: the acceptance check is a Bash → python →
# stdout-JSON contract across a process boundary, so it is tested across that
# boundary rather than by importing the module.
def test_cli_json_output_lists_the_qmd_daemon():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    names = sorted(s["name"] for s in payload["services"])
    assert QMD in names, f"{QMD} absent from the services list: {names}"


def test_live_qmd_port_probe_does_not_report_a_failure():
    """`http_check` reports port 8181 reachable (acceptance clause 2, live half).

    Only the negative assertion is made about the live box: the entry must not
    report a port failure while the daemon is resident. The unhealthy path is
    pinned deterministically below, without depending on the box.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)
    entry = next(s for s in payload["services"] if s["name"] == QMD)
    assert "FAIL" not in entry["status"], (
        f"port {QMD_PORT} reported unreachable by the live check: {entry}"
    )


# ---------------------------------------------------------------- clause 2
# "The qmd SERVICES entry has a `port` key equal to 8181..."
def test_qmd_entry_declares_port_8181():
    entry = shc.SERVICES[QMD]
    assert entry.get("port") == QMD_PORT, entry


def test_qmd_entry_is_reachable_through_a_category():
    """It is in a category, and in the default run — an entry nobody selects
    would be as invisible in practice as one nobody wrote."""
    category = shc.SERVICES[QMD]["category"]
    assert category in shc.CATEGORIES, f"category {category!r} is not selectable"
    assert QMD in shc.CATEGORIES[category]
    assert QMD in shc.CATEGORIES["all"]


def test_http_check_uses_a_real_socket_on_both_verdicts():
    """The prober itself, with no mocks: a bound port connects, a closed one does not."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    open_port = listener.getsockname()[1]
    try:
        assert shc.http_check(open_port) == (True, 0)
    finally:
        listener.close()

    closed_port = open_port  # released; nothing is listening on it now
    connected, err = shc.http_check(closed_port)
    assert connected is False, f"closed port {closed_port} read as reachable: {err}"
    assert err != 0


def test_http_check_reaches_an_ipv6_only_listener():
    """The reason the prober cannot hardcode 127.0.0.1.

    qmd binds `[::1]:8181` and nothing else — `ss -ltn | grep :8181` shows
    `[::1]:8181`, `curl http://127.0.0.1:8181/health` exits 7, `curl
    http://[::1]:8181/health` returns 200. A v6-only listener here reproduces
    that without depending on the live daemon: a check that connects only over
    IPv4 reports this port dead while it is up.
    """
    if not hasattr(socket, "AF_INET6"):
        raise AssertionError("this test assumes an IPv6-capable host")
    listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    listener.bind(("::1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        connected, err = shc.http_check(port)
        assert connected is True, (
            f"IPv6-only listener on port {port} read as unreachable ({err!r}) — "
            "the prober must try every address the host resolves to, as qmd does"
        )
    finally:
        listener.close()


class _StubSocket:
    def __init__(self, owner):
        self._owner = owner

    def settimeout(self, _timeout):
        pass

    def connect_ex(self, addr):
        self._owner.probed.append(addr)
        return self._owner.result

    def close(self):
        pass


class _StubSocketModule:
    """Replaces `shc.socket` for one test, so only the module under test sees it.

    `getaddrinfo` mirrors what this box really returns for "localhost" — IPv6
    first, then IPv4 — so a prober that tries only one family, or only the first
    answer, is still caught here.
    """

    AF_INET = socket.AF_INET
    AF_INET6 = socket.AF_INET6
    SOCK_STREAM = socket.SOCK_STREAM

    def __init__(self, result):
        self.result = result
        self.probed: list[tuple] = []
        self.queries: list[tuple] = []

    def getaddrinfo(self, host, port, *_rest):
        self.queries.append((host, port))
        return [
            (self.AF_INET6, self.SOCK_STREAM, 6, "", ("::1", port, 0, 0)),
            (self.AF_INET, self.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    def socket(self, *_family, **_kw):
        return _StubSocket(self)


class _StubSubprocessModule:
    """Replaces `shc.subprocess` so the supervisor answer is fixed without
    touching the daemon or the real subprocess module."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, output):
        self.output = output
        self.calls: list[list] = []

    def run(self, command, **_kwargs):
        self.calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, self.output, "")


def _qmd_result(monkeypatch, *, supervisor_output: str, connect_result: int):
    """Run the real `check_service` against the real SERVICES entry, faking only
    the two genuinely external answers: what supervisor says, and whether the
    TCP connect succeeded."""
    sockets = _StubSocketModule(connect_result)
    monkeypatch.setattr(shc, "socket", sockets)
    monkeypatch.setattr(
        shc, "subprocess", _StubSubprocessModule(supervisor_output)
    )
    result = shc.check_service(QMD, shc.SERVICES[QMD])
    return result, sockets


# "...and running the script while qmd is unreachable reports that entry as
#  unhealthy."
def test_qmd_unreachable_reports_the_entry_unhealthy(monkeypatch):
    result, sockets = _qmd_result(
        monkeypatch,
        supervisor_output=f"{QMD}  RUNNING   pid 3129856, uptime 18:46:25",
        connect_result=111,  # ECONNREFUSED
    )
    assert sockets.queries == [("localhost", QMD_PORT)], sockets.queries
    assert {a[1] for a in sockets.probed} == {QMD_PORT}, (
        f"the prober never consulted the declared port: {sockets.probed}"
    )
    assert result["healthy"] is False, (
        f"supervisor RUNNING masked an unreachable port: {result}"
    )
    assert f"port {QMD_PORT} FAIL" in result["status"], result


def test_qmd_reachable_keeps_the_entry_healthy(monkeypatch):
    """The other direction, so the test above cannot be satisfied by an entry
    that is simply always unhealthy."""
    result, sockets = _qmd_result(
        monkeypatch,
        supervisor_output=f"{QMD}  RUNNING   pid 3129856, uptime 18:46:25",
        connect_result=0,
    )
    assert sockets.queries == [("localhost", QMD_PORT)], sockets.queries
    assert {a[1] for a in sockets.probed} == {QMD_PORT}, sockets.probed
    assert result["healthy"] is True, result


def test_a_held_closed_port_flips_the_entry_unhealthy_with_real_sockets(monkeypatch):
    """Acceptance clause 2's unhealthy half, through the real prober and real TCP.

    The connect is really attempted, over both families, at a port that is
    provably refusing connections — so the unhealthy verdict is demonstrated
    without taking the resident daemon down, on the code path qmd would take if
    its listener really went away while supervisor still believed it was up.

    The port is *held* rather than allocated-then-released: a socket bound to
    `::1:p` (v6only) and `127.0.0.1:p` with no `listen()` behind it answers
    ECONNREFUSED on both families, and the bind keeps another process from
    taking that number mid-test. The obvious version — bind, read the ephemeral
    port, `close()`, then probe it — passed standalone and then failed in the
    full-suite gate run of round SM_20260912_064601, because a released
    ephemeral port belongs to the suite again (this suite starts uvicorn
    servers) and because this very assertion also sat on a real `supervisorctl`
    call that the script caps at 5 s. Both of those are environment, not
    behaviour. What is left as external is supervisor's answer, which is a fact
    about the box rather than about this code, so it is the one thing stubbed.
    """
    v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    v6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    v6.bind(("::1", 0))
    closed_port = v6.getsockname()[1]
    v4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        v4.bind(("127.0.0.1", closed_port))
    except OSError as exc:  # pragma: no cover - number already used on ::1 side
        v6.close()
        raise AssertionError(
            f"cannot hold both loopback families on port {closed_port}: {exc}"
        )
    monkeypatch.setattr(
        shc,
        "subprocess",
        _StubSubprocessModule(f"{QMD}  RUNNING   pid 3129856, uptime 18:46:25"),
    )
    try:
        entry = dict(shc.SERVICES[QMD])
        entry["port"] = closed_port
        result = shc.check_service(QMD, entry)
    finally:
        v4.close()
        v6.close()

    assert result["healthy"] is False, (
        f"port {closed_port} refuses connections yet the entry reported healthy: "
        f"{result}"
    )
    assert f"port {closed_port} FAIL" in result["status"], result


def test_qmd_stopped_supervisor_state_is_not_masked_as_healthy(monkeypatch):
    """The port probe runs only on a service supervisor already believes is
    up; a STOPPED daemon must stay unhealthy on its own."""
    result, sockets = _qmd_result(
        monkeypatch, supervisor_output=f"{QMD}  STOPPED", connect_result=0
    )
    assert result["healthy"] is False, result
    assert sockets.queries == [], "port resolved for a service that is not running"
    assert sockets.probed == [], "port probed for a service that is not running"


# ------------------------------------------- clauses 3, 4, 5 (skill surface)
@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL.is_file(), f"skill file missing: {SKILL}"
    return SKILL.read_text()


def test_skill_no_longer_claims_the_wrong_qmd_port(skill_text):
    """Clause 3: `grep '5003'` over the skill returns 0 lines."""
    assert "5003" not in skill_text


def test_skill_probes_qmd_on_8181(skill_text):
    """The corrected port has to actually appear, not merely replace 5003."""
    assert str(QMD_PORT) in skill_text


def test_skill_no_longer_probes_the_frontend_over_plain_http(skill_text):
    """Clause 4: `grep 'localhost:3000'` over the skill returns 0 lines."""
    assert "localhost:3000" not in skill_text


def test_skill_step_four_probes_the_https_frontend_with_k(skill_text):
    """Clause 5: Step 4's curl targets https://localhost:5173/ with -k."""
    marker = "### Step 4"
    assert marker in skill_text
    step4 = skill_text.split(marker, 1)[1].split("### Step 5", 1)[0]
    assert "https://localhost:5173/" in step4, step4
    line = next(
        (l for l in step4.splitlines() if "https://localhost:5173/" in l), ""
    )
    # `-k` may arrive alone or combined (`-sk`), so match the flag inside the
    # option token rather than the string "-k".
    flags = [t for t in line.split() if t.startswith("-")]
    assert any("k" in f for f in flags), (
        f"self-signed cert needs -k, options were {flags}: {line!r}"
    )
    assert "http://localhost:5173" not in step4, "plain-HTTP probe of a TLS service"
