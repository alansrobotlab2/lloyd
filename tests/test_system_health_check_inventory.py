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


# ---------------------------------------------------------------------------
# #3025 — the /api/* trust boundary has a witness, and the witness can fail.
#
# The component lives in the vault skill (its own tree, driven as a subprocess
# through `LLOYD_SHC_SCRIPT` exactly as the two nodes above drive it); this file
# is why it is pinned. What is pinned here is the process boundary. Every leg of
# the probe leaves the interpreter over a real socket, so the stub below is a
# real `ThreadingHTTPServer` on 127.0.0.1 and not an in-process transport:
# `tests/test_api_client_gating.py` already covers what the gate answers to a
# peer injected into an ASGI scope, and the gap #3025 names is that the answer
# was never asked over a socket. No LAN address is needed to test a probe that
# binds one — the peer seam chooses the SOURCE address, the URL seam chooses the
# destination, and loopback accepts a connection opened from any source a caller
# cares to name.
#
# `server.py` and `config.yaml` are faked per test, and the fakes deliberately
# carry values DIFFERENT from the shipped ones: the default trusted set below is
# not the network `server.py` ships, and the refusal sentence below is not the
# sentence it defines. A row that reported a hard-coded copy instead of reading
# the deployed files would name a value no fake file contains, and the nodes
# below would go red.
# ---------------------------------------------------------------------------

FAKE_DEFAULT_NETWORKS = '10.240.0.0/16'          # NOT the network server.py ships
FAKE_REFUSAL = 'the fake refuses: /api/* is gated'   # NOT the sentence server.py defines
FAKE_COMMIT_SHA = 'deadbeefcafebabe'
WIDENED_NETWORK = '192.168.50.0/24'  # sanctioned by server.py's comment: advisory, never red


class _StubBackend:
    """A backend on a real loopback socket, answering the two legs as a test says.

    Not an in-process ASGI transport: the component's claim is that it crosses a
    process boundary over a socket whose source address it chose, so the stub has
    to be a socket. `seen` records what the SERVER observed as the peer, which is
    the only evidence available that a row names the address its request really
    came from rather than one it merely was told to mention.
    """

    def __init__(self, gated_status=403, gated_body=None,
                 control_status=200, control_body=None):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        refusal = gated_body if gated_body is not None else json.dumps(
            {'detail': FAKE_REFUSAL})
        health = control_body if control_body is not None else json.dumps(
            {'status': 'ok', 'commit': FAKE_COMMIT_SHA})
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, self.client_address[0]))
                if self.path.startswith('/api/'):
                    status, body = gated_status, refusal
                elif self.path == '/health':
                    if control_status is None:
                        # Answer NOTHING: drop the connection with no status line,
                        # which is what a half-alive listener does, as distinct
                        # from answering wrongly.
                        self.connection.close()
                        return
                    status, body = control_status, health
                else:
                    status, body = 404, '{"detail":"not found"}'
                payload = body.encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.seen = seen
        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self._server.server_address[1]}'

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _fake_server_py(tmp_path, default_networks=FAKE_DEFAULT_NETWORKS,
                    refusal=FAKE_REFUSAL):
    """A `server.py` holding only the two constants the witness reads.

    Only those two, because the witness parses the module with `ast`: a health
    check must not have to boot the FastAPI app to ask one question.
    """
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'server.py').write_text(
        f'DEFAULT_TRUSTED_NETWORKS = {default_networks!r}\n'
        f'REFUSAL_DETAIL = {refusal!r}\n', encoding='utf-8')
    return root


def _fake_config(tmp_path, host='0.0.0.0', trusted=None):
    """A deployed config: the bind, and a trusted set only when the test asks."""
    lines = ['server:', f'  host: {host}']
    if trusted is not None:
        lines.append('  trusted_networks: [' + ', '.join(trusted) + ']')
    path = tmp_path / 'config.yaml'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _api_gate(env_overrides, fmt='json'):
    """One `--component api_trust_boundary` run: (row-or-text, process).

    A minimal environment, never a copy of `os.environ`: whoever invokes pytest is
    often an agent turn that may itself have `LLOYD_HEALTH_*` exported, and a
    witness under test must not inherit a peer, a URL or an `ip` from it.
    """
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': os.path.expanduser('~')}
    env.update({key: str(value) for key, value in env_overrides.items()})
    cmd = [sys.executable, str(SCRIPT)]
    if fmt == 'json':
        cmd += ['--format', 'json']
    cmd += ['--component', 'api_trust_boundary']
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
    if fmt == 'json':
        return json.loads(proc.stdout)['api_trust_boundary'], proc
    return proc.stdout, proc


def _env(root, config, stub, peer=None, ip_cmd=None):
    """The seams for one run — URL and peer always, config and `ip` when given."""
    env = {'LLOYD_HEALTH_LLOYD_ROOT': str(root),
           'LLOYD_HEALTH_CONFIG_PATH': str(config),
           'LLOYD_HEALTH_API_GATE_URL': stub.url}
    if peer is not None:
        env['LLOYD_HEALTH_API_GATE_PEER'] = peer
    if ip_cmd is not None:
        env['LLOYD_HEALTH_IP_CMD'] = str(ip_cmd)
    return env


def test_the_witness_is_green_only_on_the_gates_own_refusal(tmp_path):
    """Clause 1, and the named-bind half of clause 4.

    The stub refuses with the wording the fake `server.py` defines and answers the
    control with a commit, so the row is `refused`, healthy, and exits 0. The fake
    config binds every interface, which must appear as `server-host-unrestricted`:
    named, and still not an outage — #683 declined the rebind twice because the
    gate, not the bind, is the control.
    """
    root, config, stub = _fake_server_py(tmp_path), _fake_config(tmp_path), _StubBackend()
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['state'] == 'refused', row
    assert row['kind'] == 'boundary-held', row
    assert row['healthy'] is True, row
    assert row['gated_leg']['status'] == 403, row['gated_leg']
    assert row['control_leg']['status'] == 200, row['control_leg']
    assert 'server-host-unrestricted' in row['advisories'], row['advisories']
    # The peer here is loopback, which the rule under test trusts by definition, so
    # the row has to carry that limit too rather than print an unqualified green.
    assert 'probe-peer-is-loopback' in row['advisories'], row['advisories']
    assert proc.returncode == 0, proc.stderr[-800:]


def test_the_witness_is_an_outage_only_when_the_gated_route_answers(tmp_path):
    """Clause 1's `not green unless` and clause 4's `only when`, from one stub.

    The gated route answers 200 to a peer the gate exists to refuse. That is #683
    regressing and it is the ONLY condition this component may print `DEGRADED`
    for, so the node pins the red and its exit code together — and pins that the
    control leg was never asked, because a leak is decided on the leak.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend(gated_status=200, gated_body='{"sessions": []}')
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['state'] == 'refusal-lost', row
    assert row['healthy'] is False, row
    assert row['control_leg'] is None, row
    assert [path for path, _ in stub.seen] == ['/api/sessions'], stub.seen
    assert '/api/sessions' in row['detail'] and '127.0.0.1' in row['detail'], row['detail']
    assert json.loads(proc.stdout)['overall_status'] == 'degraded', proc.stdout[-800:]
    assert proc.returncode == 3, proc.returncode


def test_a_refusal_that_is_not_the_gate_refusing_is_not_a_verdict(tmp_path):
    """A 403 from anything else is neither green nor the outage.

    A proxy, or a different process holding the port, can answer 403 with words of
    its own. Reading that as `refused` would have the row trust an accident;
    reading it as `refusal-lost` would cry wolf about a leak that did not happen.
    It is `not-measured`, it is not green, and the box stays healthy.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend(gated_status=403, gated_body='{"detail":"nope"}')
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['state'] == 'not-measured', row
    assert row['healthy'] is False, row
    assert row['reason'] == 'gated-answer-not-a-refusal', row
    assert json.loads(proc.stdout)['overall_status'] == 'healthy', proc.stdout[-800:]
    assert proc.returncode == 0, proc.returncode


def test_the_control_leg_is_what_licenses_the_green(tmp_path):
    """Clause 2: a refusal nobody can attribute is not a proof.

    The gated leg refuses exactly as it should and the control leg answers NOTHING
    — the stub drops that connection without a status line, which is a different
    fact from answering wrongly and is the case the clause names. With nothing
    replying to the same peer there is nothing tying that 403 to the backend
    process that owns the gate, so the row is the named not-measured state: neither
    green nor the refusal-lost outage.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend(control_status=None)
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['state'] == 'not-measured', row
    assert row['reason'] == 'control-leg-unreachable', row
    assert row['healthy'] is False, row
    assert json.loads(proc.stdout)['overall_status'] == 'healthy', proc.stdout[-800:]
    assert proc.returncode == 0, proc.returncode


def test_a_control_answer_without_a_commit_is_not_a_control(tmp_path):
    """Clause 2's second half: a 200 alone does not say who answered.

    `commit` is the field that identifies the answering process as the backend —
    the same field the watchdog reads out of `/health` — so a 200 from something
    else holding the port must not turn the row green.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend(control_body='{"status":"ok"}')
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['state'] == 'not-measured', row
    assert row['reason'] == 'control-leg-unexpected', row
    assert row['healthy'] is False, row
    assert proc.returncode == 0, proc.returncode


def test_a_box_with_no_scope_global_address_reports_an_absence(tmp_path):
    """Clause 2's other named state: no source address, no verdict, no alarm.

    This is what an agent turn under `bwrap --unshare-all` really is — no
    scope-global interface, no route to the backend — and the run has to name the
    absence instead of printing `Overall: DEGRADED`, or the one real alarm this
    component owns stops being believed. `LLOYD_HEALTH_IP_CMD` stands in for the
    missing interface: the fake `ip` succeeds and lists nothing, and the stub's own
    hit list is the proof that no request went out to be judged.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend()
    try:
        row, proc = _api_gate(_env(root, config, stub, ip_cmd='/bin/true'))
    finally:
        stub.close()
    assert row['state'] == 'not-measured', row
    assert row['reason'] == 'no-scope-global-address', row
    assert row['healthy'] is False, row
    assert stub.seen == [], stub.seen
    assert json.loads(proc.stdout)['overall_status'] == 'healthy', proc.stdout[-800:]
    assert proc.returncode == 0, proc.returncode


def test_a_widened_trusted_set_is_named_not_red(tmp_path):
    """Clauses 3 and 4 together, on the value server.py itself recommends.

    The fake config widens `server.trusted_networks` to a LAN network while the
    fake `server.py` defaults to a different one, so the row can only report
    `192.168.50.0/24` by having read the config. The stub still refuses, so the
    row stays green and the box stays healthy: a widened set is how a LAN-only
    device is kept working, and losing the refusal is the only red.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path, trusted=[WIDENED_NETWORK])
    stub = _StubBackend()
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['trusted_networks'] == [WIDENED_NETWORK], row['trusted_networks']
    assert 'config.yaml' in row['trusted_set_source'], row['trusted_set_source']
    assert 'trusted-set-widened' in row['advisories'], row['advisories']
    assert row['state'] == 'refused', row
    assert json.loads(proc.stdout)['overall_status'] == 'healthy', proc.stdout[-800:]
    assert proc.returncode == 0, proc.returncode


def test_the_default_trusted_set_comes_from_server_py(tmp_path):
    """Clause 3: with no `trusted_networks` key the set is the one server.py defines.

    The fake `server.py` ships a network this box would never use and the fake
    config has no `trusted_networks` key, so a row reporting the fake value read
    it from the source; a row reporting the network actually deployed here would
    be reporting a copy, which is what the clause forbids.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path, trusted=None)
    stub = _StubBackend()
    try:
        row, proc = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
    finally:
        stub.close()
    assert row['trusted_networks'] == [FAKE_DEFAULT_NETWORKS], row['trusted_networks']
    assert 'server.py' in row['trusted_set_source'], row['trusted_set_source']
    assert row['state'] == 'refused', row


def test_the_script_carries_no_copy_of_the_values_it_reads(tmp_path):
    """Clause 3's other half, read off the source rather than off a run.

    The script defines no CIDR of its own and holds no copy of the sentence the
    gate answers with: either copy would keep printing a confident, stale verdict
    the day the real value changed. Naming the sentence here is deliberate — a test
    is allowed to quote the value precisely so the script is not.
    """
    source = SCRIPT.read_text(encoding='utf-8')
    cidrs = re.findall(r'(?<![\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}', source)
    assert cidrs == [], f'the script hard-codes networks: {sorted(set(cidrs))}'
    assert 'peer not permitted' not in source
    assert 'DEFAULT_TRUSTED_NETWORKS' in source, 'the default is read, not restated'


def test_the_row_names_the_peer_the_socket_actually_came_from(tmp_path):
    """Clause 5: attribution, checked against what the server observed.

    The witness provokes `api-gate: refused … from peer` warnings in `server.err`
    by design and no counter or series reads them, so the address printed in the
    row is the only thing that lets whoever reads that log separate the witness's
    own refusals from an outsider's. The stub records the peer its socket saw, and
    the text report has to print the same address on the verdict line.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    stub = _StubBackend()
    try:
        row, _ = _api_gate(_env(root, config, stub, peer='127.0.0.1'))
        text, _ = _api_gate(_env(root, config, stub, peer='127.0.0.1'), fmt='text')
    finally:
        stub.close()
    assert row['probe_peer'] == '127.0.0.1', row['probe_peer']
    assert row['probe_peer_source'] == 'LLOYD_HEALTH_API_GATE_PEER', row['probe_peer_source']
    assert row['gated_leg']['peer'] == '127.0.0.1', row['gated_leg']
    assert {peer for _, peer in stub.seen} == {'127.0.0.1'}, stub.seen
    # Loopback is trusted by the rule under test, so the row also has to admit that
    # what it witnessed here was the mechanism, not the boundary's reach.
    assert 'probe-peer-is-loopback' in row['advisories'], row['advisories']
    assert 'api_trust_boundary — refused (boundary-held, peer 127.0.0.1)' in text, text


def test_discovery_asks_from_an_address_the_trusted_set_does_not_cover(tmp_path):
    """The peer the witness chooses, not the peer it was told.

    Two scope-global addresses, one inside the fake default set and one outside
    it. The trusted one has to be skipped: the gate is supposed to ANSWER it, and
    booking that 200 as `refusal-lost` would raise the component's only alarm about
    a device that was entitled to its data. The stub's own socket log is the proof
    of which address asked.
    """
    root = _fake_server_py(tmp_path)
    config = _fake_config(tmp_path)
    ip_stub = tmp_path / 'ip'
    ip_stub.write_text(textwrap.dedent("""\
        #!/bin/sh
        echo '3: eth0    inet 10.240.1.7/16 scope global eth0'
        echo '5: eth1    inet 127.0.0.2/8 scope global eth1'
        """))
    ip_stub.chmod(0o755)
    stub = _StubBackend()
    try:
        row, proc = _api_gate(_env(root, config, stub, ip_cmd=ip_stub))
    finally:
        stub.close()
    assert row['probe_peer'] == '127.0.0.2', row
    assert row['state'] == 'refused', row
    assert {peer for _, peer in stub.seen} == {'127.0.0.2'}, stub.seen
    assert proc.returncode == 0, proc.returncode
