"""Backlog #1241 — provisioning the minted Lloyd CA into the invoking user's NSS store.

``agent_mcp/browser.py`` launches Chromium with no ``ignore_https_errors``, so the
browser arm *verifies* whatever certificate the frontend presents. When
``agent-services/cert`` holds no fresh Tailscale cert, ``web/vite.config.ts`` serves
``lloyd.crt`` — a leaf signed by ``agent-services/cert/ca.crt``, a private CA — and
that fallback is the path this CA has to be trusted on. Until this file existed,
nothing in the checkout wrote that CA into a trust store: the trust behind loading that
leaf was hand-seeded state (``certutil -A`` run by a person; ``git grep -ln -E "certutil|nssdb|update-ca-certificates" HEAD``
returned no files), so a rebuilt box reached the MC frontend as
``net::ERR_CERT_AUTHORITY_INVALID`` with nothing naming the cause.

The store that matters for Chromium is the NSS shared database at ``$HOME/.pki/nssdb``.
Measured on this box on 2026-09-23 with the throwaway CA + ``CN=lloyd`` leaf this
suite mints (``/usr/bin/chromium``, headless, ``new_context()`` with no trust flags,
``HOME`` pointed at a temp dir):

* CA installed with ``-t "CT,C,C"``          -> ``status=200``, page title as served
* temp ``HOME`` with no installed CA          -> ``net::ERR_CERT_AUTHORITY_INVALID``

The live frontend cannot be the assertion, which is why the fixture is the instrument.
As of 2026-09-22 13:17 ``agent-services/cert/goliath.taile37041.ts.net.crt`` exists, so
:5173 is serving a Let's Encrypt leaf that names only the tailnet host — loading it by
``localhost`` fails ``ERR_CERT_COMMON_NAME_INVALID`` with or without this CA installed,
and loading it by its tailnet name succeeds with an empty store. And ``trust list``
here carries a machine-wide ``Lloyd CA`` anchor that no script provisioned either. A
throwaway CA is in no system store, so it is the only thing on this box that
discriminates a real install from no install at all.

So a rootless, user-level install is sufficient for the browser arm; the machine-wide
half (a real anchor under ``/etc/ca-certificates/trust-source/anchors/`` plus
``update-ca-trust``) needs root and stays a person's job — see the item's
needs-a-person clause.

These tests drive the scripts themselves: the seam is python -> bash ->
(``certutil``, ``openssl``, ``sha256sum``), and for the end-to-end clause
bash -> Chromium -> TLS handshake, which no grep can substitute for. ``HOME`` is a
temp dir in every run and ``LLOYD_NSS_DB`` is used where a test only needs *a*
store, so no test here can reach the real ``~/.pki/nssdb``; ``LLOYD_CERT_DIR`` parks
the mint in a temp tree, so no test can reach the live ``agent-services/cert``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import threading
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_CERT = REPO / "scripts" / "gen-cert.sh"
INSTALL_CA = REPO / "scripts" / "install-ca.sh"

BASH = shutil.which("bash")
CERTUTIL = shutil.which("certutil")
CHROMIUM = "/usr/bin/chromium"

# What scripts/install-ca.sh installs, and the form the hand-seeded store on this
# box carries: trusted CA for SSL, e-mail and object signing.
NSS_NICK = "Lloyd CA"
NSS_TRUST = "CT,C,C"

# Same host shape scripts/test_gen_cert_sans.py pins, so the leaf names 127.0.0.1
# (which the fixture navigates to) and nothing depends on this box's routing table.
HOST = "goliath"
LAN_IP = "192.168.50.108"

FIXTURE_TITLE = "Lloyd -- Mission Control (CA fixture)"
FIXTURE_BODY = b"<html><head><title>%s</title></head><body>ok</body></html>" % (
    FIXTURE_TITLE.encode(),)

# Everything the two scripts execute, resolved from the running system and symlinked
# into the directory used as the whole PATH. `certutil` is listed separately because
# the "certutil is unavailable" clause needs a PATH that genuinely lacks it —
# prepending a directory could never hide /usr/bin/certutil.
TOOLCHAIN = (
    "awk", "bash", "cat", "chmod", "dirname", "grep", "head", "hostname", "ls",
    "mkdir", "mktemp", "openssl", "python3", "rm", "sed", "sha256sum", "sort", "tr",
    "uname",
)

Minted = namedtuple("Minted", "bindir cert_dir home run")


# ── harness ─────────────────────────────────────────────────────────────────────

def _make_sandbox(root: Path, *, with_certutil: bool = True) -> Path:
    """A PATH holding only the scripts' own toolchain — no tailscale, and no
    certutil when *with_certutil* is false."""
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    tools = list(TOOLCHAIN) + (["certutil"] if with_certutil else [])
    for tool in tools:
        src = shutil.which(tool)
        assert src is not None, f"these tests need {tool} on PATH"
        (bindir / tool).symlink_to(src)
    # The exact format `ip -4 -o route get 1.1.1.1` prints, so the script's awk
    # extraction runs and the minted SAN set does not depend on this box's routes.
    ip_stub = bindir / "ip"
    ip_stub.write_text(
        "#!/usr/bin/env bash\n"
        f"echo '1.1.1.1 via 192.168.50.1 dev eth0 src {LAN_IP} uid 1000'\n"
    )
    ip_stub.chmod(0o755)
    return bindir


def _env(bindir: Path, *, cert_dir: Path | None = None, home: Path | None = None,
         nss_db: Path | None = None) -> dict[str, str]:
    """A minimal environment. HOME is always a temp dir: the default store location
    is ``$HOME/.pki/nssdb``, and this suite must never resolve that to the real one."""
    env = {"PATH": str(bindir), "LC_ALL": "C", "HOSTNAME": HOST}
    if home is not None:
        env["HOME"] = str(home)
    if cert_dir is not None:
        env["LLOYD_CERT_DIR"] = str(cert_dir)
    if nss_db is not None:
        env["LLOYD_NSS_DB"] = str(nss_db)
    return env


def _run(script: Path, bindir: Path, args: list[str] = (), **env_kw) -> subprocess.CompletedProcess:
    assert BASH is not None, "bash is needed to run the scripts under test"
    return subprocess.run(
        [BASH, str(script), *args],
        env=_env(bindir, **env_kw),
        capture_output=True,
        text=True,
        timeout=300,
    )


def _gen_cert(bindir: Path, cert_dir: Path, home: Path, args: list[str] = ()):
    return _run(GEN_CERT, bindir, list(args), cert_dir=cert_dir, home=home)


def _install_ca(bindir: Path, ca_crt: Path, *, home: Path | None = None,
                nss_db: Path | None = None):
    return _run(INSTALL_CA, bindir, [str(ca_crt)], home=home, nss_db=nss_db)


def _certutil(db: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    assert CERTUTIL is not None, "certutil (nss-tools / libnss3-tools) is needed"
    run = subprocess.run(
        [CERTUTIL, "-d", f"sql:{db}", *args], capture_output=True, text=True, timeout=60)
    if check and run.returncode != 0:
        raise AssertionError(f"certutil {' '.join(args)} failed: {run.stdout}\n{run.stderr}")
    return run


def _certutil_bytes(db: Path, *args: str) -> bytes:
    """`certutil` in binary mode. `certutil -L -r` writes the stored certificate as
    raw DER, which no text mode survives."""
    assert CERTUTIL is not None, "certutil (nss-tools / libnss3-tools) is needed"
    run = subprocess.run([CERTUTIL, "-d", f"sql:{db}", *args], capture_output=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(
            f"certutil {' '.join(args)} failed: {run.stderr.decode(errors='replace')}")
    assert run.stdout, f"certutil {' '.join(args)} produced no output"
    return run.stdout


_TRUST_SHAPE = re.compile(r"[A-Za-zuU]{0,2},[A-Za-zuU]{0,2},[A-Za-zuU]{0,2}")


def _nss_rows(db: Path) -> list[tuple[str, str]]:
    """(nickname, trust) per entry of an NSS store, read from `certutil -L`.

    A store that was never created has no entries — that is the claim every clause-1
    assertion is making, so it reads as an empty listing rather than as certutil's
    SEC_ERROR_BAD_DATABASE. (test_a_store_that_does_not_exist_is_created is what pins
    the directory itself.)

    Nicknames hold spaces ("Lloyd CA") and the trust column is padded, so a row is
    'text, two or more spaces, then a non-space run'. The two header lines are
    dropped by the trust-shape filter — `Trust Attributes` is not shaped like one.
    """
    if not db.exists():
        return []
    out = _certutil(db, "-L").stdout
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}(\S+)\s*$", line)
        if m and _TRUST_SHAPE.fullmatch(m.group(2)):
            rows.append((m.group(1).strip(), m.group(2)))
    return rows


def _trusts(db: Path, nick: str = NSS_NICK) -> list[str]:
    """Trust attributes of every entry stored under *nick* — length is also the
    duplicate-nickname check (a second row is how a botched re-install shows up)."""
    return [trust for nickname, trust in _nss_rows(db) if nickname == nick]


def _der_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ca_der_sha(ca_crt: Path) -> str:
    """DER of a PEM certificate file, via openssl."""
    run = subprocess.run(["openssl", "x509", "-in", str(ca_crt), "-outform", "DER"],
                         capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr.decode()
    return _der_sha(run.stdout)


def _stored_der_sha(db: Path, nick: str = NSS_NICK) -> str:
    """DER of the certificate NSS actually holds under *nick* (`certutil -L -r`, raw
    DER on stdout), so identity is compared byte for byte rather than by nickname
    alone — a re-mint leaves the nickname standing and the key changed."""
    return _der_sha(_certutil_bytes(db, "-L", "-n", nick, "-r"))


def _create_empty_store(db: Path) -> None:
    """An NSS store with no certificates. `certutil -N` refuses a missing directory
    (SEC_ERROR_BAD_DATABASE), which is why scripts/install-ca.sh has to mkdir first."""
    db.mkdir(parents=True, exist_ok=True)
    _certutil(db, "-N", "--empty-password")


# ── clause 1: gen-cert.sh provisions the CA, on the mint path and the skip path ──

@pytest.fixture(scope="module")
def minted(tmp_path_factory) -> Minted:
    """One CA + leaf minted by scripts/gen-cert.sh into a temp cert dir, with HOME a
    temp dir — so the run's own trust install is what the clause-1 assertions read.
    Module-scoped: the 4096-bit CA keygen is the slow part."""
    root = tmp_path_factory.mktemp("ca-install")
    bindir = _make_sandbox(root)
    cert_dir = root / "agent-services" / "cert"
    home = root / "home"
    home.mkdir()
    run = _gen_cert(bindir, cert_dir, home)
    assert run.returncode == 0, run.stdout + run.stderr
    return Minted(bindir=bindir, cert_dir=cert_dir, home=home, run=run)


def test_gen_cert_installs_the_minted_ca_into_the_default_nss_store(minted: Minted) -> None:
    """Clause 1, mint path: after `bash scripts/gen-cert.sh` the CA is in the
    invoking user's store (``$HOME/.pki/nssdb``, nickname ``Lloyd CA``) and it is
    *this* CA — same bytes, not merely same nickname."""
    db = minted.home / ".pki" / "nssdb"
    assert _trusts(db) == [NSS_TRUST], (
        f"gen-cert.sh did not install {NSS_NICK!r} into {db}; store holds {_nss_rows(db)}"
    )
    assert _stored_der_sha(db) == _ca_der_sha(minted.cert_dir / "ca.crt"), (
        "the installed certificate is not the CA that was just minted"
    )


def test_the_certs_already_exist_skip_branch_installs_the_ca_too(minted: Minted, tmp_path) -> None:
    """Clause 1's discriminating half. `scripts/gen-cert.sh` exits early when CA and
    server cert are present, so on the box that already has certs — every box that
    was ever set up — a provisioning step appended at the end of the file never runs.
    A brand-new user on such a box must still be provisioned: same cert dir, fresh
    HOME, and the run must prove it took the early exit while doing the install."""
    fresh_home = tmp_path / "fresh-user"
    fresh_home.mkdir()

    run = _gen_cert(minted.bindir, minted.cert_dir, fresh_home)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "already exist" in run.stdout, "the run did not take the early exit this clause names"
    assert _trusts(fresh_home / ".pki" / "nssdb") == [NSS_TRUST], (
        "the skip branch reported the certs as present and provisioned nobody"
    )


# ── clause 2: idempotent, one nickname, trust never rewritten ───────────────────

def test_running_the_provisioning_step_twice_changes_nothing(minted: Minted, tmp_path) -> None:
    """Clause 2: twice against one store leaves exactly one ``Lloyd CA`` entry with
    unchanged trust attributes. The second row a naive ``certutil -A`` loop can
    produce shows up here as a longer listing, not as a passing run."""
    ca_crt = minted.cert_dir / "ca.crt"
    db = tmp_path / "nssdb"

    first = _install_ca(minted.bindir, ca_crt, nss_db=db)
    assert first.returncode == 0, first.stdout + first.stderr
    after_first = _nss_rows(db)
    assert after_first == [(NSS_NICK, NSS_TRUST)], f"first install is not one clean entry: {after_first}"

    second = _install_ca(minted.bindir, ca_crt, nss_db=db)
    assert second.returncode == 0, second.stdout + second.stderr
    assert _nss_rows(db) == after_first, "the second run rewrote the store"
    assert _trusts(db) == [NSS_TRUST], "duplicate or re-trusted entries under one nickname"


def test_an_existing_entry_for_the_same_ca_is_left_alone_not_rewritten(minted: Minted, tmp_path) -> None:
    """Clause 2's mechanism. #1241's triage measured that ``certutil -A`` for a
    subject the store already holds does not add a row — it overwrites that row's
    trust bits, so a re-run can silently downgrade an entry a person set by hand
    (``-t "C,,"`` is the form documented at
    ``architecture/tailscale-mtls-remote-access.md:208`` and Chromium accepts it).
    Provisioning therefore checks before it adds."""
    ca_crt = minted.cert_dir / "ca.crt"
    db = tmp_path / "nssdb"
    _create_empty_store(db)
    _certutil(db, "-A", "-n", NSS_NICK, "-t", "C,,", "-i", str(ca_crt))
    before = _nss_rows(db)
    assert before == [(NSS_NICK, "C,,")], f"fixture store is not the hand-seeded shape: {before}"

    run = _install_ca(minted.bindir, ca_crt, nss_db=db)
    assert run.returncode == 0, run.stdout + run.stderr
    assert _nss_rows(db) == before, (
        f"provisioning rewrote trust bits of a certificate it was already satisfied by: {_nss_rows(db)}"
    )


# ── clause 3: overridable store, and a missing certutil is a warning not a failure ─

def test_lloyd_nss_db_redirects_the_install_away_from_home(minted: Minted, tmp_path) -> None:
    """Clause 3: the store location is overridable, so the step can be exercised
    against a temp nssdb — which is what makes every other test in this file safe to
    run on a machine whose real ``~/.pki/nssdb`` holds the live Lloyd CA."""
    other = tmp_path / "sandbox-nssdb"
    home = tmp_path / "home"
    home.mkdir()

    run = _install_ca(minted.bindir, minted.cert_dir / "ca.crt", home=home, nss_db=other)
    assert run.returncode == 0, run.stdout + run.stderr
    assert _trusts(other) == [NSS_TRUST], f"LLOYD_NSS_DB was set but nothing landed in {other}"
    assert not (home / ".pki").exists(), "the override was ignored and $HOME/.pki was written"


def test_gen_cert_honours_the_store_override_as_well(minted: Minted, tmp_path) -> None:
    """The acceptance names ``bash scripts/gen-cert.sh`` as the entry point, so the
    override has to survive the hand-off from gen-cert.sh to the install step too."""
    other = tmp_path / "from-gen-cert"
    home = tmp_path / "home"
    home.mkdir()

    run = _run(GEN_CERT, minted.bindir, [], cert_dir=minted.cert_dir, home=home, nss_db=other)
    assert run.returncode == 0, run.stdout + run.stderr
    assert _trusts(other) == [NSS_TRUST], "gen-cert.sh installed into the default location instead"
    assert not (home / ".pki").exists(), "$HOME/.pki was written despite LLOYD_NSS_DB"


def test_a_store_that_does_not_exist_is_created(minted: Minted, tmp_path) -> None:
    """The rebuilt-box case: no ``~/.pki/nssdb`` at all. ``certutil -N`` refuses a
    missing directory with SEC_ERROR_BAD_DATABASE, so creating it is the script's job."""
    home = tmp_path / "home"
    home.mkdir()
    db = home / ".pki" / "nssdb"
    assert not db.exists()

    run = _gen_cert(minted.bindir, minted.cert_dir, home)
    assert run.returncode == 0, run.stdout + run.stderr
    assert db.is_dir(), f"no NSS store was created at {db}"
    assert _trusts(db) == [NSS_TRUST]


def test_missing_certutil_warns_and_exits_zero(minted: Minted, tmp_path) -> None:
    """Clause 3: without ``certutil`` (nss-tools / libnss3-tools not installed) the
    step warns and exits 0 — cert-minting is not the thing that should break."""
    bindir = _make_sandbox(tmp_path / "no-certutil", with_certutil=False)
    assert not (bindir / "certutil").exists(), "the sandbox PATH still resolves certutil"
    home = tmp_path / "home"
    home.mkdir()

    run = _install_ca(bindir, minted.cert_dir / "ca.crt", home=home)
    assert run.returncode == 0, f"exit {run.returncode}: {run.stdout}\n{run.stderr}"
    assert "WARNING" in run.stderr and "certutil" in run.stderr, run.stderr
    assert not (home / ".pki").exists(), "a warning-only run still wrote a store"


def test_gen_cert_still_succeeds_when_certutil_is_missing(minted: Minted, tmp_path) -> None:
    """The caller must not inherit the failure: on this box the certs exist, so the
    run reaches the trust step from the skip branch and must still report success."""
    bindir = _make_sandbox(tmp_path / "no-certutil-2", with_certutil=False)
    home = tmp_path / "home"
    home.mkdir()

    run = _run(GEN_CERT, bindir, [], cert_dir=minted.cert_dir, home=home)
    assert run.returncode == 0, f"exit {run.returncode}: {run.stdout}\n{run.stderr}"
    assert "already exist" in run.stdout
    assert "WARNING" in run.stderr and "certutil" in run.stderr, run.stderr


# ── clause 4: end to end — the install is what makes Chromium load the page ───────

def _browser_arm_available() -> bool:
    """Everything clause 4 needs to exist: the NSS CLI, the same Chromium binary
    agent_mcp/browser.py names, and playwright to drive it."""
    if CERTUTIL is None or not Path(CHROMIUM).is_file():
        return False
    return importlib.util.find_spec("playwright") is not None


requires_browser = pytest.mark.skipif(
    not _browser_arm_available(),
    reason="needs certutil, playwright and /usr/bin/chromium to cross the TLS trust seam",
)


class _FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler spelling
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(FIXTURE_BODY)))
        self.end_headers()
        self.wfile.write(FIXTURE_BODY)

    def log_message(self, *args):  # stay quiet in the gate log
        pass


@pytest.fixture(scope="module")
def tls_fixture(minted: Minted):
    """The minted leaf (``CN=lloyd``, SAN names 127.0.0.1) served over TLS on
    loopback — the same shape ``lloyd-frontend`` has, minus the vite process."""
    import ssl

    httpd = HTTPServer(("127.0.0.1", 0), _FixtureHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(minted.cert_dir / "lloyd.crt"), str(minted.cert_dir / "lloyd.key"))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{httpd.server_address[1]}/"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


# The four flags agent_mcp/browser.py passes today, repeated here rather than
# imported so this fixture keeps working if browser.py grows: clause 5's own test in
# tests/test_browser_panel.py is what pins those args.
BROWSER_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]


async def _navigate(url: str, home: Path) -> str:
    """Load *url* in the same browser the arm uses — ``/usr/bin/chromium`` through
    playwright, ``new_context()`` with no trust flags, exactly as
    ``agent_mcp/browser.py`` builds it — and describe the verdict as one string,
    whether it was a status line or a Chromium error page."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=list(BROWSER_LAUNCH_ARGS),
            env={**os.environ, "HOME": str(home)},
        )
        try:
            context = await browser.new_context()
            page = await context.new_page()
            try:
                response = await page.goto(url, wait_until="domcontentloaded")
                return f"status={response.status} title={await page.title()!r}"
            except Exception as exc:  # noqa: BLE001 - the verdict string IS the result
                return f"failed: {str(exc).splitlines()[0]}"
        finally:
            await browser.close()
    finally:
        await pw.stop()


@requires_browser
def test_headless_chromium_loads_the_fixture_once_the_ca_is_installed(
    minted: Minted, tls_fixture: str
) -> None:
    """Clause 4, positive half: with the generated CA in an otherwise-empty HOME's
    nssdb — the store ``scripts/gen-cert.sh`` wrote in the module fixture — headless
    Chromium loads an HTTPS fixture signed by that CA with status 200. The HOME holds
    nothing else, so the only way this passes is the install."""
    verdict = asyncio.run(_navigate(tls_fixture, minted.home))
    assert verdict == f"status=200 title={FIXTURE_TITLE!r}", verdict


@requires_browser
def test_the_identical_fixture_is_refused_by_an_empty_nss_store(
    minted: Minted, tls_fixture: str, tmp_path
) -> None:
    """Clause 4, negative half and the positive control for the whole item: the same
    fixture, the same browser, an empty nssdb — net::ERR_CERT_AUTHORITY_INVALID.
    Without this arm the positive test would only prove Chromium trusts anything."""
    home = tmp_path / "empty-store"
    (home / ".pki").mkdir(parents=True)
    _create_empty_store(home / ".pki" / "nssdb")
    assert _nss_rows(home / ".pki" / "nssdb") == [], "the negative arm's store is not empty"

    verdict = asyncio.run(_navigate(tls_fixture, home))
    assert "failed:" in verdict and "net::ERR_CERT_AUTHORITY_INVALID" in verdict, verdict


# ── #1668 clause 1-5: `install-ca.sh --check`, the drift guard the re-mint lacked ──
#
# The live store on this box carried the retired Aug-22 CA (`98:1E:A0:F3...`) five
# days after agent-services/cert/ca.crt was re-minted to `C5:A0:C2:DC...`, and
# nothing on the box could say so: the only caller of install-ca.sh is gen-cert.sh
# (scripts/gen-cert.sh:50), which normal ops never run. These tests drive `--check`
# through the same sandboxed-PATH harness as the install path, against temp stores
# only — the real ~/.pki/nssdb is not opened by any test here.

PK12UTIL = shutil.which("pk12util")

# `certutil -A` cannot carry a private key, and NSS downgrades a keyless `-t u,u,u`
# entry to `,,`, so the leaf-and-key nickname the box actually holds (`Lloyd-goliath
# u,u,u`) can only be reproduced with pk12util. It ships in the same nss package as
# certutil, so wherever this file's other tests run, this one runs too.
requires_pk12 = pytest.mark.skipif(
    CERTUTIL is None or PK12UTIL is None,
    reason="certutil/pk12util (nss-tools) not installed — cannot build an NSS store",
)


def _fingerprint(path: Path) -> str:
    """`openssl x509 -in <path> -noout -fingerprint -sha256`, value only — the form
    #1668's premise check was measured in, so a test failure reads straight across
    to the command an operator would run by hand."""
    openssl = shutil.which("openssl")
    assert openssl is not None, "openssl not available for the trust-step tests"
    out = subprocess.run([openssl, "x509", "-in", str(path), "-noout",
                          "-fingerprint", "-sha256"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().rsplit("=", 1)[-1]


def _mint_ca(dest: Path) -> Path:
    """A second CA — a different key, not a re-signed copy of the same one.

    The drift `--check` exists to catch is the store holding a *different key*, which
    a re-export of the same key cannot produce. 2048 bits because this fixture's job
    is a mismatching DER, and the re-mint this models is already exercised end to end
    by `minted`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    openssl = shutil.which("openssl")
    assert openssl is not None, "openssl not available for the trust-step tests"
    crt = dest / "other-ca.crt"
    out = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout",
         str(dest / "other-ca.key"), "-out", str(crt), "-days", "400",
         "-subj", "/CN=Lloyd other CA (test fixture)"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return crt


def _check(bindir: Path, cert_dir: Path, ca: Path | None, *, home: Path,
           nss_db: Path | None):
    """`bash scripts/install-ca.sh --check [ca]`, sandboxed like every other call.

    With *ca* None the script resolves the CA from `$LLOYD_CERT_DIR/ca.crt` — the
    other half of the clause, and the form an operator runs after a re-mint.
    """
    argv = ["--check"] if ca is None else ["--check", str(ca)]
    return _run(INSTALL_CA, bindir, argv, cert_dir=cert_dir, home=home,
                nss_db=nss_db)


def test_check_exits_zero_for_the_ca_it_would_install(minted: Minted, tmp_path) -> None:
    """Clause 1. An installed store passes the check, by both ways of naming the CA.

    The store ends up holding the CA through the real install path, so the match is
    `certutil`'s stored DER against `openssl`'s read of the file, not two reads of the
    same file by one tool. Then the check runs twice: with `ca.crt` as the positional
    argument, and with only `$LLOYD_CERT_DIR` set (`_run` points that at the minted
    cert dir and passes no argument), because the operator after a re-mint types the
    second form. Both must exit 0 and print the fingerprint that
    `openssl x509 -in ca.crt -noout -fingerprint -sha256` prints for the same file —
    the value #1668's premise check compares against the store.
    """
    store = tmp_path / "nssdb"
    home = tmp_path / "home"
    home.mkdir()

    install = _run(INSTALL_CA, minted.bindir, [], cert_dir=minted.cert_dir,
                   home=home, nss_db=store)
    assert install.returncode == 0, f"exit {install.returncode}:\n{install.stderr}"

    want = _fingerprint(minted.cert_dir / "ca.crt")
    positional = _check(minted.bindir, minted.cert_dir, minted.cert_dir / "ca.crt",
                        home=home, nss_db=store)
    assert positional.returncode == 0, (
        f"a store holding the CA failed its own check: exit "
        f"{positional.returncode}\n{positional.stdout}\n{positional.stderr}")
    assert want in positional.stdout, f"{want!r} missing from:\n{positional.stdout}"

    by_env = _check(minted.bindir, minted.cert_dir, None, home=home, nss_db=store)
    assert by_env.returncode == 0, (
        f"--check with only $LLOYD_CERT_DIR set failed: exit {by_env.returncode}\n"
        f"{by_env.stdout}\n{by_env.stderr}")
    assert want in by_env.stdout, f"{want!r} missing from:\n{by_env.stdout}"


def test_check_fails_naming_both_sides_when_the_nickname_is_absent(
        minted: Minted, tmp_path) -> None:
    """Clause 2, first half. No nickname at all is the state a fresh box is in, and
    it is also the state a store left half-written lands in; either way the answer is
    a failure that prints the fingerprint the store *should* hold, so the operator has
    something to compare. `_run` gives every call a temp HOME, and the store override
    points at a path that does not exist: the run must not have to create anything to
    discover the absence.
    """
    home = tmp_path / "home"
    home.mkdir()
    want = _fingerprint(minted.cert_dir / "ca.crt")

    run = _check(minted.bindir, minted.cert_dir, minted.cert_dir / "ca.crt",
                 home=home, nss_db=tmp_path / "absent" / "nssdb")
    assert run.returncode != 0, (
        f"an absent 'Lloyd CA' nickname reported no drift:\n{run.stdout}")
    both = run.stdout + run.stderr
    assert want in both, f"expected fingerprint {want!r} missing:\n{both}"
    assert "stored" in both.lower() and "expected" in both.lower(), (
        f"the failure named neither side of the comparison:\n{both}")


def test_check_fails_naming_both_fingerprints_when_the_store_holds_another_key(
        minted: Minted, tmp_path) -> None:
    """Clause 2, second half — the exact drift that sat on this box for five days.

    The store holds one CA, `ca.crt` is another. Both fingerprints must appear, because
    the operator's next question is "which of my two boxes is the stale one?", and this
    is the state the retired Aug-22 CA left: a nickname present, trusted, and wrong.
    """
    store = tmp_path / "nssdb"
    home = tmp_path / "home"
    home.mkdir()
    other = _mint_ca(tmp_path / "other")

    install = _run(INSTALL_CA, minted.bindir, [str(other)], cert_dir=minted.cert_dir,
                   home=home, nss_db=store)
    assert install.returncode == 0, f"exit {install.returncode}:\n{install.stderr}"

    stored, want = _fingerprint(other), _fingerprint(minted.cert_dir / "ca.crt")
    assert stored != want, "the two fixture CAs are the same key; nothing to detect"

    run = _check(minted.bindir, minted.cert_dir, minted.cert_dir / "ca.crt",
                 home=home, nss_db=store)
    assert run.returncode != 0, (
        f"a store holding a different CA reported no drift:\n{run.stdout}")
    both = run.stdout + run.stderr
    assert stored in both, f"stored fingerprint {stored!r} missing:\n{both}"
    assert want in both, f"expected fingerprint {want!r} missing:\n{both}"


def test_check_never_writes_to_the_store_it_reports_on(minted: Minted,
                                                      tmp_path) -> None:
    """Clause 3. A guard that repairs while it judges cannot be run from a health
    check or by an operator who is only looking, which is why this mode exists at all:
    #1241's ruling of 2026-09-27 declined the machine-wide anchor, so a `--check` that
    auto-installed would have quietly done the per-store half of the thing that ruling
    asked someone to look at first.

    Two stores, both with HOME a temp dir. The fresh one must come back with no
    database at all — not an empty nickname list, which a `certutil -N` on the read
    path would also leave behind. The stale one must be byte-identical afterwards:
    same stored DER, same trust bits, same `cert9.db` file bytes.
    """
    home = tmp_path / "home"
    home.mkdir()

    fresh = tmp_path / "fresh" / "nssdb"
    first = _check(minted.bindir, minted.cert_dir, minted.cert_dir / "ca.crt",
                   home=home, nss_db=fresh)
    assert first.returncode != 0
    assert not (fresh / "cert9.db").exists(), (
        "--check created a certificate database while reporting on it")
    assert not (home / ".pki").exists(), (
        "--check fell back to $HOME/.pki and wrote there")

    store = tmp_path / "stale" / "nssdb"
    other = _mint_ca(tmp_path / "other2")
    install = _run(INSTALL_CA, minted.bindir, [str(other)], cert_dir=minted.cert_dir,
                   home=home, nss_db=store)
    assert install.returncode == 0, f"exit {install.returncode}:\n{install.stderr}"

    db = store / "cert9.db"
    assert db.is_file(), "install did not leave the store this half reads back"
    before_der = _stored_der_sha(store)
    before_trust = _trusts(store, NSS_NICK)
    before_bytes = hashlib.sha256(db.read_bytes()).hexdigest()
    assert before_der == _ca_der_sha(other) and before_trust == [NSS_TRUST], (
        f"the stale fixture is not the shape the box holds: {before_trust!r}")

    run = _check(minted.bindir, minted.cert_dir, minted.cert_dir / "ca.crt",
                 home=home, nss_db=store)
    assert run.returncode != 0
    assert _stored_der_sha(store) == before_der, (
        "--check rewrote the stored DER it was only meant to compare")
    assert _trusts(store, NSS_NICK) == before_trust, "--check changed the trust bits"
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_bytes, (
        "cert9.db bytes changed across a read-only check")


@requires_pk12
def test_install_replaces_only_the_lloyd_ca_nickname(minted: Minted,
                                                    tmp_path) -> None:
    """Clause 4. The store on this box also holds `Lloyd-goliath  u,u,u` — the server
    leaf and its key, which is what the browser arm and vite read for the fallback
    cert. An install that swept it would break the page the CA install exists to make
    loadable, so the only removal this script may perform is the same-nickname
    `certutil -D` (scripts/install-ca.sh) that replaces `Lloyd CA` with the new CA.

    Asserted two ways because the comment this clause retires claimed a broader
    retirement that never shipped: the surviving nickname and its trust bits after an
    install, and that the script still contains exactly one `certutil -D` call. The
    nickname is imported with pk12util because `certutil -A` cannot carry a key, and
    NSS downgrades a keyless `u,u,u` entry to `,,`.
    """
    store = tmp_path / "nssdb"
    _create_empty_store(store)
    leaf_crt, leaf_key = minted.cert_dir / "lloyd.crt", minted.cert_dir / "lloyd.key"
    p12 = tmp_path / "leaf.p12"
    openssl = shutil.which("openssl")
    exported = subprocess.run(
        [openssl, "pkcs12", "-export", "-inkey", str(leaf_key), "-in", str(leaf_crt),
         "-name", "Lloyd-goliath", "-out", str(p12), "-passout", "pass:testpw"],
        capture_output=True, text=True)
    assert exported.returncode == 0, exported.stderr
    imported = subprocess.run([PK12UTIL, "-i", str(p12), "-d", f"sql:{store}",
                               "-W", "testpw"], capture_output=True, text=True,
                              timeout=60)
    assert imported.returncode == 0, imported.stderr

    before = _nss_rows(store)
    assert ("Lloyd-goliath", "u,u,u") in before, (
        f"the fixture store does not reproduce the box's leaf nickname: {before}")

    home = tmp_path / "home"
    home.mkdir()
    install = _run(INSTALL_CA, minted.bindir, [], cert_dir=minted.cert_dir,
                   home=home, nss_db=store)
    assert install.returncode == 0, f"exit {install.returncode}:\n{install.stderr}"

    rows = _nss_rows(store)
    assert ("Lloyd-goliath", "u,u,u") in rows, (
        f"install removed or retrusted the leaf nickname: {rows}")
    assert (NSS_NICK, NSS_TRUST) in rows, f"the CA was not installed: {rows}"

    grep = shutil.which("grep")
    counting = subprocess.run([grep, "-c", "certutil -D", str(INSTALL_CA)],
                              capture_output=True, text=True)
    assert counting.stdout.strip() == "1", (
        f"install-ca.sh no longer holds exactly one deletion call: "
        f"{counting.stdout.strip()!r}\n{INSTALL_CA.read_text()}")


def test_the_script_header_tells_an_operator_to_run_the_check_after_a_re_mint() -> None:
    """Clause 5. `gen-cert.sh` is the only caller of this script and normal ops never
    run it, so a CA re-minted out of band is invisible to every scheduled job on the
    box — the drift in this item was found by hand, five days late. The knowledge that
    `--check` exists therefore has to live where the person who just re-minted a CA is
    looking, which is the top of the script itself, not a vault note.

    Read out of the header block only (everything up to the first non-comment line),
    so a mention buried in a branch further down does not pass: three months from now
    the reader skims the usage comment and nothing else.
    """
    header_lines = []
    for line in INSTALL_CA.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            break
        header_lines.append(line)
    header = "\n".join(header_lines)
    assert header.strip(), "install-ca.sh has no header block to read"

    low = header.lower()
    assert "--check" in low, "the header never mentions --check"
    assert "re-mint" in low, "the header does not tie --check to a CA re-mint"
    assert "never writes" in low or "write nothing" in low, (
        "the header does not say --check is read-only")
    assert "/etc" in low, (
        "the header does not scope --check away from /etc, which #1241's ruling "
        "requires of it")


# ── #1726 clause 5: gaining a production reader must not move these boundaries ──

HEALTH_CHECK = REPO / "scripts" / "service_health_check.py"


def test_the_new_caller_left_the_deletion_path_at_one_call() -> None:
    """Clause 5, first half. `certutil -D` stays at exactly one call in this script.

    The single removal is the same-nickname replacement at the install path. A second
    deletion call — and above all one reachable from a health check, which a person or
    an agent may run unattended — would let an automated read destroy the trust it was
    asked to measure. Counted rather than diffed, so the invariant holds for whatever
    the next round does to this file too.
    """
    src = INSTALL_CA.read_text(encoding="utf-8")

    assert src.count("certutil -D") == 1, (
        f"{src.count('certutil -D')} `certutil -D` calls in install-ca.sh; the "
        "contract this file's own clause 4 test pins is exactly one")


def test_the_health_checker_that_calls_this_script_never_asks_it_to_write() -> None:
    """Clause 5, second half: the caller #1726 added is read-only against the machine.

    Read-only against the *store* is `tests/test_service_health_check_ca_trust.py`'s
    job, asserted against the argv it builds. This is the other sense: #1241's ruling of
    2026-09-27T18:56Z moved the system half of the trust story out of every automated
    path, so an entry that ran `update-ca-trust`, or wrote an anchor under
    `/etc/ca-certificates/`, or shelled to `certutil` itself, would be a health check
    that needs root and reimplements the verdict it is reading. The whole file is
    scanned, comments included, because the clause says "anywhere in the diff".
    """
    src = HEALTH_CHECK.read_text(encoding="utf-8")
    assert src, f"{HEALTH_CHECK} vanished"

    assert "update-ca-trust" not in src, (
        "a health check cannot run update-ca-trust — it needs root (#1241's ruling)")
    assert "/etc/ca-certificates" not in src, (
        "nothing automated writes a system trust anchor (#1241's ruling)")
    assert re.search(r"[\"']certutil", src) is None, (
        "the checker reaches the store only through `install-ca.sh --check`; a quoted "
        "`certutil` token is an argv entry, which would be a second implementation of "
        "the verdict this row reads")
    assert '"--check"' in src, "the caller must build the read-only form"


def test_no_executable_line_under_scripts_reaches_system_trust() -> None:
    """Every `update-ca-trust` under `scripts/` is a comment recording that it is NOT run.

    Three shell scripts mention the command to say why they do not call it, so the
    assertion ignores comment lines — a test that could not tell a prohibition from an
    invocation would have to be deleted the first time someone wrote the prohibition
    down. What is left must be empty: system trust is a person's job (#1241).
    """
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.sh")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "update-ca-trust" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{n}: {line.strip()}")

    assert not offenders, (
        "these lines execute the machine-wide trust update that #1241 ruled a "
        "person's job: " + "; ".join(offenders))
