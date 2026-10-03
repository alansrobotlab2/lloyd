"""#2129 — the `:5173` row's four scheme/certificate verdicts, one node each.

`system_health_check.py` gives a reader four different words for four different
things that can be true of the Mission Control port at the same moment, and
SKILL.md:63-66 tells an agent which sentence each one licenses: a refused
connection is the only signature that means DOWN; a certificate the client will
not accept is `TLS TRUST FAILED` and **never** DOWN; a leaf inside the
threshold is `CERT EXPIRING` even while HTTP answers 200; a plain-HTTP probe of
a TLS port is a scheme mismatch and **never** DOWN. The last three of those are
the false-"frontend is down" class #575 and #1044 exist to stop, and until this
file nothing in `tests/` pinned any of them: the only hit for the cert words
anywhere in the suite is `tests/test_degradation_contract.py:588`, which asserts
`CERTIFICATE_VERIFY_FAILED` on a raw `HTTPSConnection` to prove
`FixtureListener(tls=True)` really injected a trust fault — a pin on the
fixture, never on the verdict the script prints.

Every state here is a fixture, not the machine: the vite config the script reads
is written into a `tmp_path` tree, the TLS pair it serves is generated into that
same tree by `openssl req -x509`, it is trusted only because the run's
`SSL_CERT_FILE` names it, and the port is a `FixtureListener`/released fixture
socket. Nothing binds or connects to 5173, nothing reads the live
`~/lloyd/web/vite.config.ts`, and no real certificate is involved.

The asserted words are string literals, deliberately not imported from
`system_health_check.py`: comparing the row against `CERT_EXPIRING_PREFIX` or
`TLS_TRUST_PREFIX` would follow a rename and stay green, which is a call-through
rather than a pin. The literals are the words SKILL.md promises a human reader.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from degradation.runner import DeadPort, FixtureListener  # noqa: E402

# The same seam `tests/test_system_health_check_inventory.py:21` uses: the skill
# script lives in the vault, so the suite points at it rather than assuming a path.
SCRIPT = Path(os.environ.get("LLOYD_SHC_SCRIPT") or
              Path.home() / "obsidian" / "skills" / "system-health-check" / "system_health_check.py")

#: `vite_tls_material` (`system_health_check.py:451`) reads exactly two literals out of
#: the config: `tsHost`, and the ONE string literal inside `path.resolve(__dirname, …)`.
#: `certDir` is therefore written as a single relative literal here, because the
#: multi-argument form would hand the probe `..` and it would look for `lloyd.crt` one
#: directory above the cert dir — a fixture that silently tests the wrong path.
VITE_CONFIG = """\
import path from 'node:path'
const tsHost = 'box.ts.net'
const certDir = path.resolve(__dirname, '../agent-services/cert')
export default { server: { host: '0.0.0.0', https: {
  cert: path.resolve(certDir, `${tsHost}.crt`), key: path.resolve(certDir, `${tsHost}.key`) } } }
"""

CERT_DIR_RELPATH = "agent-services/cert"


def _fixture_tree(tmp_path: Path, *, serves_tls_pair: bool) -> Path:
    """A throwaway `~/lloyd` whose `web/vite.config.ts` says what the port serves.

    `serves_tls_pair=True` puts `lloyd.crt`/`lloyd.key` in the cert dir, which is what
    makes `vite_tls_material` answer `have_server=True` and the probe derive `https`
    against `localhost`; `False` leaves the pair away, the config's own `haveServer`
    false branch, so the derived scheme is `http` — the state a TLS port meets when
    somebody is read as plain HTTP. The files' bytes are irrelevant: the derivation
    only asks `is_file()`, which is why they are one word each and not certificates.
    The Tailscale-named pair is deliberately never created, so the derived verify host
    is `localhost` — the name the fixture leaf is valid for.
    """
    cert_dir = tmp_path / CERT_DIR_RELPATH
    cert_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "web" / "vite.config.ts").write_text(VITE_CONFIG, encoding="utf-8")
    if serves_tls_pair:
        (cert_dir / "lloyd.crt").write_text("fixture pair\n", encoding="utf-8")
        (cert_dir / "lloyd.key").write_text("fixture pair\n", encoding="utf-8")
    return tmp_path


def _throwaway_pair(tmp_path: Path, days: int) -> tuple[Path, Path]:
    """A self-signed `CN=localhost` pair valid for `days`, generated for this test.

    The name has to be `localhost` because `cert_expiry` verifies the handshake, and
    verifying means the leaf names the host it was reached by
    (`system_health_check.py:672-679`). `openssl` is the same issuer
    `tests/degradation/runner.py::_tls_material` shells out to, and the pair lands in
    this test's own `tmp_path`, so no certificate this box owns is read, trusted, or
    renewed anywhere in here.
    """
    cert, key = tmp_path / "leaf.crt", tmp_path / "leaf.key"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-days", str(days), "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=DNS:localhost",
                    "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return cert, key


@contextlib.contextmanager
def _tls_fixture(tmp_path: Path, days: int, code: int = 200):
    """A loopback listener serving a pair of known age, with that pair as its CA.

    `FixtureListener` serves whatever its module-level `_TLS_CERT`/`_TLS_KEY` name
    holds when its handshake runs (`context.load_cert_chain` reads them at
    tests/degradation/runner.py:309; declared at `:334`, minted by `_tls_material` at
    `:337`), so the fixture's age is set here rather
    than inherited: the runner's own throwaway pair is `-days 1`, which cannot
    express "outside the threshold", and the whole point of nodes 3 and 4 is which
    side of 30 days the served leaf falls on. Yields `(port, cert)` — handing back
    the certificate so the caller can make it the trust anchor through
    `SSL_CERT_FILE` and nothing else.
    """
    import degradation.runner as runner

    cert, key = _throwaway_pair(tmp_path, days)
    saved = runner._TLS_CERT, runner._TLS_KEY
    runner._TLS_CERT, runner._TLS_KEY = cert, key
    listener = FixtureListener(code=code, tls=True)
    try:
        yield listener.port, cert
    finally:
        listener.close()
        runner._TLS_CERT, runner._TLS_KEY = saved


def _row(tmp_path: Path, port: int, *, ssl_cert_file: Path | None = None,
         serves_tls_pair: bool = True) -> dict:
    """The `:5173` row the skill produces for one fixture state, JSON and text alike.

    `LLOYD_HEALTH_ENDPOINT_PORTS` narrows the `tools` component to this one endpoint
    so the verdict cannot come from whether production `:8080`/`:8096` happen to be
    up, and `LLOYD_HEALTH_FRONTEND_PORT` is what makes the fixture port the frontend
    port — so every assertion below can say 5173's row and mean it. `text` is the
    human report, asserted alongside the JSON because the words a human reads are the
    thing SKILL.md:63-66 is about.
    """
    env = {**os.environ,
           "LLOYD_HEALTH_LLOYD_ROOT": str(_fixture_tree(tmp_path, serves_tls_pair=serves_tls_pair)),
           "LLOYD_HEALTH_FRONTEND_PORT": str(port),
           "LLOYD_HEALTH_ENDPOINT_PORTS": str(port)}
    env.pop("SSL_CERT_DIR", None)
    if ssl_cert_file is None:
        env.pop("SSL_CERT_FILE", None)
    else:
        env["SSL_CERT_FILE"] = str(ssl_cert_file)
    argv = [sys.executable, str(SCRIPT), "--component", "tools"]
    json_run = subprocess.run(argv + ["--format", "json"], capture_output=True,
                              text=True, env=env, timeout=120)
    text_run = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)
    rows = json.loads(json_run.stdout)["tools"]["endpoints"]
    assert len(rows) == 1, f"expected one narrowed endpoint row, got {rows}"
    row = rows[0]
    line = [ln for ln in text_run.stdout.splitlines() if f":{port} —" in ln]
    assert len(line) == 1, f"the report should print exactly one line for :{port}: {text_run.stdout}"
    return {"row": row, "line": line[0], "text": text_run.stdout,
            "reason": [ln for ln in text_run.stdout.splitlines() if ln.startswith("[!] tools:")]}


def test_a_refused_frontend_port_is_the_one_signature_that_means_down(tmp_path):
    """Clause 1 — nothing is listening, and the row says so in the DOWN shape.

    The port is `DeadPort`: bound, its number read, then released, so connecting is
    refused by the kernel rather than by whoever owns a service. SKILL.md:63 is exact
    that a `ConnectionRefusedError` is "the one signature that means DOWN, and the
    frontend row is unhealthy on it" — and the script has no literal `DOWN` token to
    print, so the shape this pins is the three things that sentence is made of: state
    `no-answer` with `healthy: false`, the refusal as the headline error, and the
    component reason "an HTTP endpoint is not responding".

    The last two asserts are the half that stops this node passing for the wrong
    reason: a refusal is never retried on the other scheme, so exactly one candidate
    URL was attempted, and no scheme-mismatch note is attached. A row that reached the
    fallback had a listener answering on some scheme, which is not DOWN.
    """
    closed = DeadPort()
    got = _row(tmp_path, closed.port)
    row = got["row"]
    assert row["state"] == "no-answer", row
    assert row["healthy"] is False, row
    assert "Connection refused" in row["error"], row
    assert not row.get("response_code"), row
    assert got["reason"] and "an HTTP endpoint is not responding" in got["reason"][0], got["reason"]
    assert row["schemes_tried"] == [f"https://localhost:{closed.port}/"], row
    assert "scheme mismatch" not in json.dumps(row), row


def test_an_unverifiable_certificate_prints_tls_trust_failed_and_is_never_down(tmp_path):
    """Clause 2 — a leaf the client will not accept is a third state, in its own words.

    `FixtureListener(tls=True)` serves a pair no trust store on this box names, and
    `SSL_CERT_FILE` is deliberately popped so the probe's default context is the real
    one: the refusal is the fixture's doing, not the machine's. What the row must
    carry is `TLS TRUST FAILED`, and what it must not carry is the refusal signature —
    SKILL.md:64: "A certificate the client will not accept is a third state, printed as
    `TLS TRUST FAILED` and never as DOWN: it proves a TLS server is answering".

    `schemes_tried` is the assert that makes "never DOWN" structural rather than
    cosmetic: the probe attempted its one derived scheme and stopped. Chasing the
    other scheme is what a scheme fault earns; a refused certificate is proof
    something speaks TLS on this port, so a second candidate could only ever produce
    a different wrong verdict.
    """
    with _tls_fixture(tmp_path, days=120) as (port, _leaf):
        got = _row(tmp_path, port, ssl_cert_file=None)
    row = got["row"]
    assert row["error"].startswith("TLS TRUST FAILED:"), row
    assert "CERTIFICATE_VERIFY_FAILED" in row["error"], row
    assert "TLS TRUST FAILED" in got["line"], got["line"]
    assert "Connection refused" not in row["error"], row
    assert not row.get("response_code"), row
    assert "scheme mismatch" not in json.dumps(row), row
    assert row["schemes_tried"] == [f"https://localhost:{port}/"], row


def test_a_leaf_inside_the_threshold_prints_cert_expiring_although_http_answers_200(tmp_path):
    """Clause 3 — the port is fine today, the leaf has a scheduled event, and the row
    says both in one sentence.

    The served pair is issued for 12 days, so its `notAfter`
    sits inside `CERT_EXPIRY_THRESHOLD_DAYS = 30` (`system_health_check.py:273`), and
    `SSL_CERT_FILE` is that very certificate, which is what makes the handshake
    verify and the date readable at all — `cert_expiry` refuses to report a date it
    did not verify (`system_health_check.py:676-679`). The listener answers 200 to the
    liveness probe, which is the whole point: HTTP working and the leaf nearly dead
    are simultaneously true, and #1044's finding was that a 200 silenced the date.

    So the wording pinned here is the conjunction, verbatim from
    `system_health_check.py:759-764`: `CERT EXPIRING IN <n> DAYS`, the served
    `notAfter=` echoed back, and `(HTTP 200 still answers; a client that verifies will
    refuse this leaf)`. The reason line is checked too, because it is where a reader
    is told to look at a certificate rather than restart a vite.
    """
    with _tls_fixture(tmp_path, days=12) as (port, leaf):
        got = _row(tmp_path, port, ssl_cert_file=leaf)
    row = got["row"]
    assert row["state"] == "cert-expiring", row
    assert row["healthy"] is False, row
    assert row["response_code"] == 200, row
    assert re.search(r"CERT EXPIRING IN \d+ DAYS", row["error"]), row
    assert "notAfter=" in row["error"] and row["cert_not_after"] in row["error"], row
    assert "(HTTP 200 still answers; a client that verifies will refuse this leaf)" in row["error"], row
    assert 0 <= row["cert_days_left"] < 30, row
    assert got["reason"] and "a served TLS certificate is inside the 30-day expiry" in got["reason"][0], got["reason"]


def test_a_plain_http_probe_of_the_tls_port_is_a_scheme_mismatch_and_never_down(tmp_path):
    """Clause 4 — the config says plain HTTP, the port speaks TLS, and the row must
    name the mismatch instead of the outage.

    This is the 2026-09-09 write-up this skill carries: an `http://` probe of the
    frontend returned nothing (`http_code=000`, curl exit 52) and was reported as
    "Frontend is down, MC UI unusable" *while the user was typing into it*. The
    fixture reproduces it from the config side: `lloyd.crt`/`lloyd.key` are absent, so
    `vite_tls_material` says `have_server=False` and the derived scheme is `http`
    while the listener serves TLS — so `serves_tls_pair=False` is the fault, not
    something about the socket.

    The probe then does the thing the false DOWN made impossible: the mismatch
    signature sends it to `https` on the same name, which answers. The pinned words are
    the note (`system_health_check.py:910-913`) and the shape: `answered`, healthy,
    both candidates listed in `schemes_tried`, `scheme: https` reporting what actually
    answered, no `TLS TRUST FAILED` (the pair is trusted through `SSL_CERT_FILE`), no
    cert finding at all (this leaf has 120 days), and no refusal signature anywhere.
    """
    with _tls_fixture(tmp_path, days=120) as (port, leaf):
        got = _row(tmp_path, port, ssl_cert_file=leaf, serves_tls_pair=False)
    row = got["row"]
    assert row["state"] == "answered", row
    assert row["healthy"] is True, row
    assert row["scheme_basis"] == "no-cert-material", row
    assert "scheme mismatch" in row["note"], row
    assert row["schemes_tried"] == [f"http://localhost:{port}/", f"https://localhost:{port}/"], row
    assert row["scheme"] == "https" and row["host"] == "localhost", row
    assert row["cert_expiry_state"] == "ok" and row["cert_days_left"] > 30, row
    assert "scheme mismatch" in got["line"], got["line"]
    assert "Connection refused" not in got["line"] and "TLS TRUST FAILED" not in got["line"], got["line"]
