"""Backlog #1045 — the SAN set written into the lloyd-frontend TLS leaf.

`lloyd-frontend` serves `agent-services/cert/lloyd.crt`, and the leaf measured on
this host named ``DNS:localhost, DNS:goliath, IP:127.0.0.1, IP:192.168.50.108``.
Every live client connects over the tailnet instead — ``ss -tnp`` showed the :5173
sockets terminating on the Tailscale address 100.105.113.88 — so

    curl -s --cacert agent-services/cert/ca.crt -o /dev/null \\
         -w '%{http_code}' https://$(tailscale ip -4):5173/

returned code 000 with **curl exit 60** (peer name mismatch) while the same
command against https://localhost:5173/ returned 200 exit 0. The CA was trusted;
only the *name* failed, which is why installing the CA on a device never helped.
`scripts/gen-cert.sh` built its SAN list from the hostname and the route-to-1.1.1.1
source address alone and never consulted `tailscale`, so every re-mint reproduced
the defect silently — and every existing probe passes `-k` or targets `localhost`,
so none of them can see it.

These tests drive the script itself: the seam is python -> bash -> (tailscale CLI,
openssl), so grepping the script is not a test. The tailnet comes from a stub
`tailscale` on PATH, the LAN address from a stub `ip` printing the real format, and
the leaf from real `openssl`, minted into a temp tree via `LLOYD_CERT_DIR` so no
test can reach the live `agent-services/cert`.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gen-cert.sh"
BASH = shutil.which("bash")
OPENSSL = shutil.which("openssl")

# All four taken from backlog #1045's measurement of this host: the Tailscale
# address the live clients connect to, the ts.net MagicDNS name that
# `tailscale status --json` reports under CertDomains, and the two addresses the
# broken leaf did name.
TS_IP = "100.105.113.88"
TS_HOST = "goliath.taile37041.ts.net"
HOST = "goliath"
LAN_IP = "192.168.50.108"

# The exact set scripts/gen-cert.sh produced before the tailnet entries existed —
# what the served leaf carried on 2026-09-12, 2026-09-16 and at triage. A host
# that is not on a tailnet must still mint this and nothing else.
BASELINE_SANS = f"DNS:localhost,DNS:{HOST},IP:127.0.0.1,IP:{LAN_IP}"

# scripts/gen-cert.sh signs the leaf with `-days 397`: Apple platforms reject TLS
# server certs valid for more than 398 days even under a trusted CA, which is what
# makes the iOS home-screen app work. Cap, not target — anything over 397 fails.
APPLE_MAX_LEAF_DAYS = 397

# Everything the script executes, resolved from the running system and symlinked
# into a directory used as the whole PATH. That is what makes the "no tailscale on
# PATH" case reachable: this host has /usr/bin/tailscale, and a PATH that merely
# prepends a directory could never hide it.
TOOLCHAIN = (
    "awk", "bash", "cat", "chmod", "date", "dirname", "grep", "head", "hostname",
    "ln", "ls", "mkdir", "mktemp", "openssl", "python3", "rm", "sed", "sort", "tr",
    "uname",
)

# The two `tailscale` calls scripts/gen-cert.sh makes. Nothing else is faked: an
# unhandled subcommand prints nothing, like a tool that is there but not useful.
TS_STUB = """#!/usr/bin/env bash
case "${1:-}" in
  ip)     __IP__ ;;
  status) __STATUS__ ;;
esac
exit __RC__
"""

Minted = namedtuple("Minted", "bindir cert_dir print_sans mint")


def _make_sandbox(root: Path) -> Path:
    """A PATH holding only the script's own toolchain — notably no tailscale."""
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for tool in TOOLCHAIN:
        src = shutil.which(tool)
        assert src is not None, f"scripts/gen-cert.sh needs {tool} on PATH to be tested"
        (bindir / tool).symlink_to(src)
    # Faked in the exact format `ip -4 -o route get 1.1.1.1` prints, so the awk
    # src-address extraction is exercised and the expected SAN set does not depend
    # on the routing table of whatever box the gate runs on.
    ip_stub = bindir / "ip"
    ip_stub.write_text(
        "#!/usr/bin/env bash\n"
        f"echo '1.1.1.1 via 192.168.50.1 dev eth0 src {LAN_IP} uid 1000'\n"
    )
    ip_stub.chmod(0o755)
    return bindir


def _stub_tailscale(bindir: Path, *, ipv4: str | None, cert_domains: list[str], rc: int = 0) -> None:
    """Write a `tailscale` stub. With rc=0, `ip -4` prints *ipv4* (nothing if None) and
    `status --json` prints those *cert_domains*; with a non-zero rc both print nothing,
    which is what a logged-out node does — its complaint goes to stderr."""
    if rc:
        ip_body = status_body = ":"
    else:
        ip_body = f"printf '{ipv4}\\n'" if ipv4 else ":"
        # The JSON goes in a sibling file rather than a heredoc: inside a `case`
        # arm the closing delimiter has to sit alone on its line, and the arm
        # terminator `;;` would have to go somewhere else.
        status_file = bindir.parent / "tailscale-status.json"
        status_file.write_text(json.dumps({"CertDomains": cert_domains}, indent=2) + "\n")
        status_body = f"cat '{status_file}'"
    stub = TS_STUB.replace("__IP__", ip_body).replace("__STATUS__", status_body).replace("__RC__", str(rc))
    (bindir / "tailscale").write_text(stub)
    (bindir / "tailscale").chmod(0o755)


def _run(bindir: Path, cert_dir: Path, args: list[str] = ()) -> subprocess.CompletedProcess:
    """Run scripts/gen-cert.sh with the sandbox PATH and the cert dir parked in tmp."""
    assert BASH is not None
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        env={
            "PATH": str(bindir),
            "LC_ALL": "C",        # openssl prints cert dates as `Sep 19 ... GMT`
            "HOSTNAME": HOST,     # the script's first choice for the DNS: entry
            "LLOYD_CERT_DIR": str(cert_dir),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )


def _openssl(*args: str) -> str:
    assert OPENSSL is not None
    run = subprocess.run([OPENSSL, *args], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return run.stdout


def _leaf_san_entries(cert_path: Path) -> list[str]:
    """The SAN entries of a minted leaf, normalised to the script's `IP:` spelling."""
    out = _openssl("x509", "-in", str(cert_path), "-noout", "-ext", "subjectAltName")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert len(lines) > 1, f"leaf carries no subjectAltName at all: {out!r}"
    return [e.strip().replace("IP Address:", "IP:") for e in " ".join(lines[1:]).split(",")]


def _leaf_dates(cert_path: Path) -> tuple[datetime, datetime]:
    out = _openssl("x509", "-in", str(cert_path), "-noout", "-startdate", "-enddate")
    fields = dict(line.split("=", 1) for line in out.splitlines())
    return tuple(  # type: ignore[return-value]
        datetime.strptime(fields[k].strip(), "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        for k in ("notBefore", "notAfter")
    )


@pytest.fixture(scope="module")
def minted(tmp_path_factory) -> Minted:
    """One CA + leaf, minted with a stubbed tailnet; the keygen is the slow part."""
    root = tmp_path_factory.mktemp("tailnet-mint")
    bindir = _make_sandbox(root)
    _stub_tailscale(bindir, ipv4=TS_IP, cert_domains=[TS_HOST])
    cert_dir = root / "agent-services" / "cert"

    printed = _run(bindir, cert_dir, ["--print-sans"])
    assert printed.returncode == 0, printed.stderr
    mint = _run(bindir, cert_dir)
    assert mint.returncode == 0, mint.stderr
    return Minted(bindir=bindir, cert_dir=cert_dir, print_sans=printed.stdout.strip(), mint=mint)


# ── clause 1: --print-sans reports the tailnet, and the leaf carries it ─────────

def test_print_sans_adds_the_tailnet_address_and_ts_net_name(minted: Minted) -> None:
    """Clause 1: with a tailnet up, the printed SAN string holds both new entries
    *in addition to* the ones the broken leaf already had."""
    assert minted.print_sans.split("\n") == [
        f"{BASELINE_SANS},IP:{TS_IP},DNS:{TS_HOST}"
    ], "--print-sans must put the SAN string alone on stdout"
    entries = minted.print_sans.split(",")
    for required in ("DNS:localhost", f"DNS:{HOST}", "IP:127.0.0.1", f"IP:{LAN_IP}",
                     f"IP:{TS_IP}", f"DNS:{TS_HOST}"):
        assert required in entries, f"{required} missing from {minted.print_sans!r}"


def test_minted_leaf_names_the_tailnet_address_and_ts_net_name(minted: Minted) -> None:
    """Acceptance: a freshly generated leaf — not just the string it was told to
    use — names the tailnet. This is what stops a re-mint silently reproducing the
    exit-60 defect that #1045 filed."""
    entries = _leaf_san_entries(minted.cert_dir / "lloyd.crt")
    assert f"IP:{TS_IP}" in entries, f"leaf SANs name no tailnet address: {entries}"
    assert f"DNS:{TS_HOST}" in entries, f"leaf SANs name no ts.net host: {entries}"


def test_print_sans_is_exactly_the_string_the_minted_leaf_received(minted: Minted) -> None:
    """`--print-sans` promises the SAN string the server leaf would receive, so the
    two must be the same set — no entry printed that openssl dropped, none signed
    that was never printed."""
    assert sorted(_leaf_san_entries(minted.cert_dir / "lloyd.crt")) == sorted(
        minted.print_sans.split(",")
    )


# ── clause 2: no tailnet mints exactly what it minted before ───────────────────

@pytest.mark.parametrize(
    "ipv4,cert_domains,rc,why",
    [
        (None, [], 0, "tailscale up but reporting no address and no CertDomains"),
        (None, [], 1, "tailscale installed and logged out"),
    ],
    ids=["idle-tailnet", "logged-out"],
)
def test_print_sans_without_a_tailnet_is_the_unchanged_set(
    tmp_path, ipv4, cert_domains, rc, why
) -> None:
    """Clause 2: a host that is not on a tailnet gets today's SAN set, and never an
    empty `IP:`/`DNS:` entry standing in for a name it could not find."""
    bindir = _make_sandbox(tmp_path)
    _stub_tailscale(bindir, ipv4=ipv4, cert_domains=cert_domains, rc=rc)
    cert_dir = tmp_path / "agent-services" / "cert"

    run = _run(bindir, cert_dir, ["--print-sans"])
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == BASELINE_SANS, f"changed on a host with {why}"
    assert not run.stdout.strip().startswith(","), "leading comma — an entry came out empty"
    assert not run.stdout.strip().endswith(","), "trailing comma — an entry came out empty"
    for entry in run.stdout.strip().split(","):
        assert re.fullmatch(r"(?:DNS|IP):\S+", entry), f"empty or malformed entry {entry!r}"
    assert not cert_dir.parent.exists(), "--print-sans created the cert tree"


def test_print_sans_without_the_tailscale_binary_is_the_unchanged_set(tmp_path) -> None:
    """Clause 2, first half: `tailscale` absent from PATH entirely — the fixture
    PATH is exactly the toolchain, so this host's /usr/bin/tailscale cannot leak in."""
    bindir = _make_sandbox(tmp_path)
    assert (bindir / "tailscale").exists() is False
    cert_dir = tmp_path / "agent-services" / "cert"

    run = _run(bindir, cert_dir, ["--print-sans"])
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == BASELINE_SANS
    assert not cert_dir.parent.exists(), "--print-sans created the cert tree"


# ── clause 3: inspecting the SANs can never invalidate a device's trust ─────────

def test_print_sans_writes_nothing_and_needs_no_force(minted: Minted) -> None:
    """Clause 3: with a full CA + leaf already in place, `--print-sans` must not
    write, and must not need `--force` — it exits before the existence check that
    prints the skip notice and before the --force warning about voiding every
    client cert, so it cannot disturb a device that trusts this CA."""
    cert_dir = minted.cert_dir
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(cert_dir.iterdir())}
    assert before, "the minted fixture produced no files"

    run = _run(minted.bindir, cert_dir, ["--print-sans"])
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == minted.print_sans, "answer changed once certs existed"
    assert "already exist" not in run.stdout, "--print-sans reached the skip branch"
    assert "invalidate" not in run.stdout + run.stderr, "--print-sans reached the --force warning"
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(cert_dir.iterdir())}
    assert after == before, "--print-sans touched a file in the cert dir"


# ── clause 4: Apple's 397-day ceiling on the signed leaf ───────────────────────

def test_minted_server_leaf_is_capped_at_397_days(minted: Minted) -> None:
    """Clause 4: the signing invocation still caps the server leaf at <=397 days.
    Apple platforms reject TLS server certs valid for >398 days even when the
    signing CA is installed, so a tailnet-covering leaf that outlives the cap is
    still refused by the iOS home-screen app — the client this item is about."""
    start, end = _leaf_dates(minted.cert_dir / "lloyd.crt")
    days = (end - start).total_seconds() / 86400
    assert days <= APPLE_MAX_LEAF_DAYS, f"leaf valid {days:.1f} days — Apple clients reject it"
    assert days >= APPLE_MAX_LEAF_DAYS - 1, (
        f"leaf valid only {days:.1f} days — the mint stopped using the -days cap"
    )


def test_minted_server_leaf_keeps_the_lloyd_common_name(minted: Minted) -> None:
    """#683 gates non-loopback /api/* on the peer cert CN, and the item requires a
    re-mint that leaves `CN=lloyd` alone; the SANs may grow, the subject may not."""
    subject = _openssl("x509", "-in", str(minted.cert_dir / "lloyd.crt"), "-noout", "-subject")
    assert re.search(r"CN\s*=\s*lloyd\s*$", subject.strip()), subject.strip()


# ── the argv seam the two flags now share ──────────────────────────────────────

def test_unknown_argument_is_refused_rather_than_ignored(tmp_path) -> None:
    """Both --force and --print-sans are now parsed from one loop, so a mistyped
    flag has to be an error: ignoring it would mint (or decline to mint) silently."""
    bindir = _make_sandbox(tmp_path)
    cert_dir = tmp_path / "agent-services" / "cert"

    run = _run(bindir, cert_dir, ["--print-sanes"])
    assert run.returncode == 2, f"exit {run.returncode}, stderr {run.stderr!r}"
    assert "--print-sans" in run.stderr, run.stderr
    assert not cert_dir.parent.exists(), "a rejected argument still created the cert tree"
