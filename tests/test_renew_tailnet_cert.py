"""#1727 — scripts/renew-tailnet-cert.sh, the owner of the tailnet TLS leaf.

The MC frontend's Vite server picks `agent-services/cert/goliath.taile37041.ts.net.crt`
by `fs.existsSync` (the `haveTs` selection in `web/vite.config.ts`), not by
validity, so when that 90-day
Let's Encrypt leaf expires it keeps serving the dead one — the valid `lloyd.crt`
fallback is never reached — and every tailnet client fails TLS. Nothing renewed
it: 5 user timers and none of them a cert one, no crontab, no supervisor program.

The script is driven here through `CERT_DIR` and a stub `tailscale` /
`supervisorctl` pair on `PATH` that records its argv, so nothing is minted and
nothing is restarted: the assertions are on the commands the script issued and on
the bytes in the cert dir. The leaves are real — minted by the same `openssl` the
script reads them with, at chosen expiry dates — because the window arithmetic is
the thing under test and a fabricated `notAfter=` string would only prove the
parser agrees with itself.

#2209 added the other half of that story: when a mint attempt has failed and the
pair still on disk is inside 24 h of its `notAfter`, the script moves it out of the
cert dir into `.expired-<host>` beside it and restarts the frontend once, so
`fs.existsSync` is false and Vite serves `lloyd.crt`. Those tests assert on which
files left the cert dir, where they landed, the argv issued after the mint, and
that the fallback pair's bytes were never touched.

Run: .venvs/lloyd/bin/python -m pytest tests/test_renew_tailnet_cert.py
"""
import json
import os
import pwd
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "renew-tailnet-cert.sh"
CONF = "agent-services/supervisor/supervisord.conf"
PROGRAM = "lloyd-mc:lloyd-frontend"
HOST = "goliath.taile37041.ts.net"
DAY = 86400

STUB = """#!/usr/bin/env python3
import json, os, shutil, sys

# The kernel hands the interpreter the PATH-resolved script path, so argv[0] is
# restored to the command word the shell actually typed: the script under test
# calls `tailscale` and `supervisorctl` by bare name, and that is the argv the
# clauses are about.
argv = [os.path.basename(sys.argv[0])] + sys.argv[1:]
rec = {"name": argv[0], "argv": argv}
with open(os.environ["ARGV_LOG"], "a") as fh:
    fh.write(json.dumps(rec) + "\\n")

name = rec["name"]
if name == "supervisorctl":
    if os.environ.get("STUB_RESTART_RC") == "1":
        sys.stderr.write("ERROR (aborted): connection refused\\n")
        raise SystemExit(1)
    print("lloyd-mc:lloyd-frontend: stopped")
    print("lloyd-mc:lloyd-frontend: started")
    raise SystemExit(0)

behaviour = os.environ.get("STUB_TAILSCALE", "ok")
args = argv[1:]
cert = args[args.index("--cert-file") + 1]
key = args[args.index("--key-file") + 1]
if behaviour == "refuse":
    sys.stderr.write(os.environ.get("STUB_REFUSE_MSG",
                                   "Error: CertDomains is empty\\n") + "\\n")
    raise SystemExit(1)
if behaviour == "empty":
    open(cert, "w").close()
    shutil.copy(os.environ["FIX_KEY"], key)
elif behaviour == "garbage":
    open(cert, "w").write("-----BEGIN NOTHING-----\\nnot a certificate\\n")
    shutil.copy(os.environ["FIX_KEY"], key)
else:
    shutil.copy(os.environ["FIX_CRT"], cert)
    shutil.copy(os.environ["FIX_KEY"], key)
raise SystemExit(0)
"""


@pytest.fixture(scope="module")
def key_pem(tmp_path_factory):
    """One RSA key for every leaf this module mints: `openssl req -x509 -key`
    costs milliseconds once, `-newkey` costs a keygen per leaf."""
    key = tmp_path_factory.mktemp("keys") / "leaf.key"
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True,
                   capture_output=True)
    return key


def _stamp(ts: float) -> str:
    return time.strftime("%Y%m%d%H%M%SZ", time.gmtime(ts))


def _leaf(dir_: Path, name: str, key_pem: Path, not_after_days: float) -> Path:
    """A real self-signed leaf whose notAfter is `not_after_days` from now.

    Written with openssl's own `-not_after`, so what the script parses out of it
    is the string `openssl x509 -enddate` produces in production, not one
    assembled in this file.
    """
    out = dir_ / name
    now = time.time()
    subprocess.run(["openssl", "req", "-x509", "-key", str(key_pem), "-out", str(out),
                    "-subj", f"/CN={HOST}",
                    "-not_before", _stamp(now - DAY),
                    "-not_after", _stamp(now + not_after_days * DAY)],
                   check=True, capture_output=True)
    return out


def _bin(tmp_path: Path) -> Path:
    """The stub `tailscale` / `supervisorctl` pair, first on PATH."""
    d = tmp_path / "bin"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("tailscale", "supervisorctl"):
        p = d / name
        p.write_text(STUB, encoding="utf-8")
        p.chmod(0o755)
    return d


class Run:
    """One invocation of the script, with its captured argv and stderr."""

    def __init__(self, proc, calls, cert_dir, before, after):
        self.proc = proc
        self.calls = calls
        self.cert_dir = cert_dir
        self.before = before
        self.after = after
        self.out = proc.stdout + proc.stderr

    def minted(self):
        return [c for c in self.calls if c["name"] == "tailscale"]

    def restarted(self):
        return [c for c in self.calls if c["name"] == "supervisorctl"]

    def operator_lines(self):
        return [l for l in self.out.splitlines() if "tailscale set --operator=" in l]

    def quarantine_dir(self) -> Path:
        """Where #2209 moves a pair it could not renew: beside the cert dir, and
        named by the same `$CERT_DIR/../.expired-$HOST` the script builds."""
        return self.cert_dir.parent / f".expired-{HOST}"


def _notafter_iso(path: Path) -> str:
    """The leaf's notAfter as the UTC ISO string the script prints, read back out
    of the file with `openssl x509 -enddate` — the same command the script parses,
    so a log line asserting this value is checked against the certificate rather
    than against a string this file assembled."""
    raw = subprocess.run(["openssl", "x509", "-enddate", "-noout", "-in", str(path)],
                         capture_output=True, text=True)
    assert raw.returncode == 0 and raw.stdout.startswith("notAfter="), raw
    gm = datetime.strptime(raw.stdout.strip()[len("notAfter="):], "%b %d %H:%M:%S %Y %Z")
    return gm.strftime("%Y-%m-%dT%H:%M:%SZ")


def _account() -> str:
    """The account whose name a refused mint has to name: the uid running the
    script, taken from the password database.

    Not `$USER`. A systemd user unit's environment carries HOME, LOGNAME and PATH
    and no USER (`systemctl --user show-environment` on this box), and the gate's
    own pytest environment is the same — so a test that read `$USER` to build its
    expectation would assert on a value that is absent exactly when the script
    needs its fallback. The uid is what both agree on."""
    return pwd.getpwuid(os.getuid()).pw_name


_AMBIENT = object()


def _run(tmp_path: Path, key_pem: Path, *, current_days: float, behaviour: str = "ok",
         fixture_days: float = 90.0, window_days: str | None = None,
         restart_rc: str | None = None, refuse_msg: str | None = None,
         current_garbage: bool = False, no_leaf: bool = False,
         fallback_pair: bool = False,
         user: str | None | object = _AMBIENT) -> Run:
    """Run the script against a temp cert dir holding a leaf expiring in
    `current_days`, with the stub's mint behaviour forced.

    `current_garbage` writes a non-certificate where the leaf should be;
    `no_leaf` leaves the path absent, which is the case the timer must mint into
    rather than the one it must refuse. `fallback_pair` also puts a `lloyd.crt` /
    `lloyd.key` in the cert dir, the way the real one has both: it is what Vite
    selects once the tailnet pair is gone, so a quarantining run has a byte-for-
    byte obligation to it. `user` is the `USER` the script sees: a name sets it,
    `None` removes `USER` and `LOGNAME` to reproduce a systemd user unit's
    environment, and the default passes the ambient environment through.
    """
    cert_dir = tmp_path / "cert"
    cert_dir.mkdir(parents=True, exist_ok=True)
    crt, key = cert_dir / f"{HOST}.crt", cert_dir / f"{HOST}.key"
    if no_leaf:
        pass
    elif current_garbage:
        crt.write_text("-----BEGIN NOTHING-----\nnot a certificate\n", encoding="utf-8")
        key.write_bytes(b"stub key")
    else:
        shutil.copy(_leaf(tmp_path, "current.crt", key_pem, current_days), crt)
        shutil.copy(key_pem, key)
    if fallback_pair:
        # Minted outside the cert dir and copied in, so nothing but the fallback
        # pair itself is in the directory when the run starts.
        src = tmp_path / "src"
        src.mkdir(parents=True, exist_ok=True)
        shutil.copy(_leaf(src, "lloyd-leaf.crt", key_pem, 300.0), cert_dir / "lloyd.crt")
        shutil.copy(key_pem, cert_dir / "lloyd.key")
    before = {p.name: p.read_bytes() for p in sorted(cert_dir.iterdir())}

    fixture = tmp_path / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    fix_crt = _leaf(fixture, "new.crt", key_pem, fixture_days)

    env = dict(os.environ)
    env.update({
        "PATH": f"{_bin(tmp_path)}{os.pathsep}{os.environ['PATH']}",
        "ARGV_LOG": str(tmp_path / "argv.jsonl"),
        "CERT_DIR": str(cert_dir),
        "TAILNET_CERT_HOST": HOST,
        "STUB_TAILSCALE": behaviour,
        "FIX_CRT": str(fix_crt),
        "FIX_KEY": str(key_pem),
    })
    if window_days is not None:
        env["CERT_RENEW_WINDOW_DAYS"] = window_days
    if restart_rc is not None:
        env["STUB_RESTART_RC"] = restart_rc
    if refuse_msg is not None:
        env["STUB_REFUSE_MSG"] = refuse_msg
    if user is None:
        env.pop("USER", None)
        env.pop("LOGNAME", None)
    elif user is not _AMBIENT:
        env["USER"] = str(user)

    # Truncated per run, not appended to: the two-run tests drive the same temp
    # cert dir twice, and a `Run` has to report only the commands its own
    # invocation issued.
    log = tmp_path / "argv.jsonl"
    log.write_text("", encoding="utf-8")

    proc = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    calls = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    after = {p.name: p.read_bytes() for p in sorted(cert_dir.iterdir())}
    return Run(proc, calls, cert_dir, before, after)

def test_outside_the_window_it_reads_the_leaf_and_touches_nothing(tmp_path, key_pem):
    """Clause 1: at 82 days out — the real distance to the leaf on this box on
    2026-09-28 — the run exits 0, issues no `tailscale cert`, and leaves the pair
    byte-identical."""
    run = _run(tmp_path, key_pem, current_days=82)
    assert run.proc.returncode == 0, run.out
    assert run.minted() == [], "minted outside the window"
    assert run.restarted() == [], "restarted outside the window"
    assert run.after == run.before
    assert not list(run.cert_dir.parent.glob(f".renew-{HOST}.*")), "staging dir left behind"
    assert run.out.strip(), "a no-op said nothing at all"


def test_the_window_is_fourteen_days_not_thirty(tmp_path, key_pem):
    """The 14-day boundary from both sides: a leaf 15 days out is not this run's
    business, one 13 days out is. Without the boundary test, `current_days=82`
    would pass against a script that never minted at all."""
    late = _run(tmp_path / "late", key_pem, current_days=15)
    assert late.proc.returncode == 0 and late.minted() == [] and late.restarted() == []
    assert late.after == late.before
    soon = _run(tmp_path / "soon", key_pem, current_days=13)
    assert soon.proc.returncode == 0, soon.out
    assert len(soon.minted()) == 1, soon.out


def test_inside_the_window_it_mints_then_restarts_once(tmp_path, key_pem):
    """Clause 2: exactly the documented mint command, then exactly one restart of
    the MC frontend, in that order, read back from the argv the stub captured.

    The mint names the staging copy, never the live pair: `agent-services/cert/
    <host>.crt` and `.key` are the last components of its arguments, rooted at a
    temp staging tree beside the cert dir, so an unverified leaf cannot reach the
    directory Vite reads. The restart is issued only after the new pair is in
    place, and the pair on disk is the minted one by then.
    """
    run = _run(tmp_path, key_pem, current_days=3, fixture_days=90)
    assert run.proc.returncode == 0, run.out
    assert len(run.calls) == 2, [c["argv"] for c in run.calls]
    mint, restart = run.calls
    assert mint["argv"][0] == "tailscale" and mint["argv"][1] == "cert"
    assert mint["argv"][2] == "--cert-file" and mint["argv"][4] == "--key-file"
    cert_arg, key_arg = mint["argv"][3], mint["argv"][5]
    assert cert_arg.endswith(f"agent-services/cert/{HOST}.crt"), cert_arg
    assert key_arg.endswith(f"agent-services/cert/{HOST}.key"), key_arg
    assert mint["argv"][6] == HOST and len(mint["argv"]) == 7, mint["argv"]
    live = str(run.cert_dir / f"{HOST}.crt")
    assert cert_arg != live and key_arg != str(run.cert_dir / f"{HOST}.key")
    assert Path(cert_arg).parent != run.cert_dir, "minted straight into the cert dir"

    assert restart["argv"][0] == "supervisorctl"
    assert restart["argv"][1:] == ["-c", str(ROOT / CONF), "restart", PROGRAM], restart["argv"]
    assert restart["argv"][2].endswith(CONF)
    assert run.out.index("leaf renewed") < run.out.index("done")

    published = (run.cert_dir / f"{HOST}.crt").read_bytes()
    assert published != run.before[f"{HOST}.crt"], "the live cert was not replaced"
    # Read back with the same command the script reads the leaf with, so what is
    # asserted is that the file Vite would serve is a cert with a later notAfter
    # than the one that was there, not merely a different file.
    enddate = subprocess.run(["openssl", "x509", "-enddate", "-noout",
                              "-in", str(run.cert_dir / f"{HOST}.crt")],
                             capture_output=True, text=True)
    assert enddate.returncode == 0 and enddate.stdout.startswith("notAfter="), enddate
    # The fixture reuses one key across every leaf it mints, so "the published
    # pair is the staged one" is equality with that key, not difference from the
    # bytes that were there before.
    assert (run.cert_dir / f"{HOST}.key").read_bytes() == key_pem.read_bytes()
    assert not list(run.cert_dir.parent.glob(f".renew-{HOST}.*")), "staging dir left behind"


def test_the_key_is_never_published_world_readable(tmp_path, key_pem):
    """The pair lands with the key owner-only, because tailscale's staging mode
    is not what should end up at the live path."""
    run = _run(tmp_path, key_pem, current_days=3)
    assert run.proc.returncode == 0, run.out
    assert (run.cert_dir / f"{HOST}.key").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("behaviour,fixture_days", [
    ("refuse", 90.0),      # the mint itself was refused
    ("empty", 90.0),       # a zero-byte cert came back
    ("garbage", 90.0),     # something that is not a certificate came back
    ("ok", 3.0),           # a mint that succeeded but is not later than the leaf
])
def test_a_bad_mint_never_reaches_the_cert_dir(tmp_path, key_pem, behaviour, fixture_days):
    """Clause 3: for a refused mint, a zero-byte file, an unparseable leaf and a
    notAfter that is not later than the current one, the existing pair stays
    byte-identical, no new file appears in the cert dir, nothing is restarted,
    and the run exits non-zero so the unit shows as failed."""
    run = _run(tmp_path, key_pem, current_days=5, behaviour=behaviour,
               fixture_days=fixture_days)
    assert run.proc.returncode != 0, f"{behaviour} was accepted: {run.out}"
    assert run.after == run.before, f"{behaviour} disturbed the cert dir"
    assert run.restarted() == [], f"{behaviour} still restarted the frontend"
    assert not list(run.cert_dir.parent.glob(f".renew-{HOST}.*")), f"{behaviour} left staging behind"
    assert run.out.strip(), f"{behaviour} failed silently"


def test_an_unreadable_leaf_is_left_alone_and_not_minted_over(tmp_path, key_pem):
    """Clause 3's fourth input is the leaf on disk: if `openssl x509 -enddate`
    cannot read it, the window arithmetic is meaningless, so the script refuses
    before it issues any command at all."""
    run = _run(tmp_path, key_pem, current_days=5, current_garbage=True)
    assert run.proc.returncode != 0, run.out
    assert run.calls == [], [c["argv"] for c in run.calls]
    assert run.after == run.before
    assert "cannot read notAfter" in run.out


def test_a_missing_leaf_is_minted_into(tmp_path, key_pem):
    """The one case with nothing to preserve: with no leaf at all Vite is already
    serving lloyd.crt, so the window check treats it as inside the window."""
    run = _run(tmp_path, key_pem, current_days=0, no_leaf=True)
    assert run.proc.returncode == 0, run.out
    assert len(run.minted()) == 1 and len(run.restarted()) == 1
    assert (run.cert_dir / f"{HOST}.crt").read_bytes()


def test_an_operator_refusal_names_the_grant_once_and_never_runs_sudo(tmp_path, key_pem):
    """Clause 4: when the mint is refused for want of the operator grant
    (SETUP.md:1504, already in place on this box), the run says so in exactly one
    line naming `sudo tailscale set --operator=<the invoking account>`, exits
    non-zero, and the captured argv contains no sudo anywhere — the script reports
    the fix, it does not apply it.

    `USER` is set explicitly rather than inherited: the point of the line is that
    it names the account that has to be granted, and the ambient environment is
    not guaranteed to carry the variable at all (see `_account`)."""
    account = _account()
    run = _run(tmp_path, key_pem, current_days=3, behaviour="refuse",
               refuse_msg="Error: You must be the tailscale operator to mint a cert",
               user=account)
    assert run.proc.returncode != 0, run.out
    assert account in run.out
    lines = run.operator_lines()
    assert len(lines) == 1, lines
    assert lines[0].strip().endswith(f"sudo tailscale set --operator={account}"), lines[0]
    for call in run.calls:
        for arg in call["argv"]:
            assert "sudo" not in arg, call["argv"]
    assert run.restarted() == []
    assert run.after == run.before


def test_the_grant_line_survives_an_environment_with_no_USER(tmp_path, key_pem):
    """The same refusal in a systemd user unit's environment, which carries HOME,
    LOGNAME and PATH but no USER — the shape the timer actually runs in. A
    message that interpolated `$USER` under `set -u` would die with an unbound
    variable at the one moment it is supposed to tell a human the fix, so the name
    has to come from the uid and the line still has to be exactly one."""
    account = _account()
    run = _run(tmp_path, key_pem, current_days=3, behaviour="refuse",
               refuse_msg="Error: You must be the tailscale operator to mint a cert",
               user=None)
    assert run.proc.returncode != 0, run.out
    lines = run.operator_lines()
    assert len(lines) == 1, lines
    assert lines[0].strip().endswith(f"sudo tailscale set --operator={account}"), lines[0]
    assert "$" not in lines[0], f"an unexpanded variable reached the operator: {lines[0]}"
    assert run.after == run.before


def test_a_refusal_that_is_not_about_the_operator_does_not_name_the_grant(tmp_path, key_pem):
    """The same path with the other diagnosis: a refusal that says nothing about
    the operator must not send the reader to a sudo command that would not fix
    anything — and still must not run sudo itself."""
    run = _run(tmp_path, key_pem, current_days=3, behaviour="refuse",
               refuse_msg="Error: CertDomains is empty for this node")
    assert run.proc.returncode != 0, run.out
    assert run.operator_lines() == []
    assert "CertDomains is empty" in run.out
    assert all("sudo" not in arg for c in run.calls for arg in c["argv"])


def test_a_failed_restart_says_the_new_leaf_is_on_disk(tmp_path, key_pem):
    """The restart is the half that actually changes what Vite serves, so its
    failure is not inert: the pair has already been published and the run has to
    say that Vite is still serving the previous cert until the program
    restarts."""
    run = _run(tmp_path, key_pem, current_days=3, restart_rc="1")
    assert run.proc.returncode != 0, run.out
    assert len(run.restarted()) == 1
    assert (run.cert_dir / f"{HOST}.crt").read_bytes() != run.before[f"{HOST}.crt"]
    assert "still serving the previous cert" in run.out


# --------------------------------------------------------------------------- #
# #2209 — the fallback for a mint that did not happen.
#
# Vite picks the tailnet pair with `fs.existsSync(tsCert) && fs.existsSync(tsKey)`
# (the `haveTs` selection in `web/vite.config.ts`) and reads no date, so a pair
# that is no longer valid keeps being served tailnet-wide for as long as it is in
# the directory, and the still-valid lloyd.crt is never reached.
# The only lever there is to remove the
# pair, so after a mint attempt has failed the script moves a pair that is inside
# 24 h of its notAfter out to `.expired-<host>` beside the cert dir and restarts
# the frontend once — and still exits non-zero, because the renewal did not
# happen.


@pytest.mark.parametrize("behaviour,fixture_days", [
    ("refuse", 90.0),      # the mint itself was refused
    ("empty", 90.0),       # a zero-byte cert came back
    ("garbage", 90.0),     # something that is not a certificate came back
    ("ok", 0.2),           # a mint that succeeded but is not later than the leaf
])
def test_a_dead_leaf_is_quarantined_when_no_better_pair_is_published(
        tmp_path, key_pem, behaviour, fixture_days):
    """Clause 1: 12 hours from notAfter and every one of the four failed-mint
    shapes — refused, empty, unparseable, not-later — ends with the pair gone from
    the cert dir, present under `cert_dir/../.expired-<host>` byte-for-byte, the
    `lloyd.crt` fallback untouched, and a non-zero exit."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour=behaviour,
               fixture_days=fixture_days, fallback_pair=True)
    assert run.proc.returncode != 0, f"{behaviour} exited 0: {run.out}"
    assert f"{HOST}.crt" not in run.after, f"{behaviour} left the leaf in the cert dir"
    assert f"{HOST}.key" not in run.after, f"{behaviour} left the key in the cert dir"
    qdir = run.quarantine_dir()
    assert (qdir / f"{HOST}.crt").read_bytes() == run.before[f"{HOST}.crt"], \
        f"{behaviour}: the quarantined leaf is not the bytes that were there"
    assert (qdir / f"{HOST}.key").read_bytes() == run.before[f"{HOST}.key"], \
        f"{behaviour}: the quarantined key is not the bytes that were there"
    assert run.after["lloyd.crt"] == run.before["lloyd.crt"], \
        f"{behaviour} rewrote the fallback cert"
    assert run.after["lloyd.key"] == run.before["lloyd.key"], \
        f"{behaviour} rewrote the fallback key"
    assert not list(run.cert_dir.parent.glob(f".renew-{HOST}.*")), \
        f"{behaviour} left staging behind"


def test_quarantining_restarts_the_frontend_exactly_once_after_the_mint(
        tmp_path, key_pem):
    """Clause 2, and the seam it sits on: the captured argv for a quarantining run
    is exactly two commands — the mint, then one `supervisorctl -c <conf> restart
    <program>` and nothing else. The restart is what makes the fallback real,
    because Vite reads the cert at startup; a pair moved with no restart is still
    the pair being served. It comes second because the quarantine only exists as
    the aftermath of a mint attempt."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse",
               fallback_pair=True)
    assert run.proc.returncode != 0, run.out
    assert [c["name"] for c in run.calls] == ["tailscale", "supervisorctl"], \
        [c["argv"] for c in run.calls]
    assert run.minted()[0]["argv"][1] == "cert"
    restart = run.restarted()[0]
    assert restart["argv"] == ["supervisorctl", "-c", str(ROOT / CONF), "restart",
                               PROGRAM], restart["argv"]


def test_a_quarantining_run_with_a_failing_restart_still_exits_non_zero(
        tmp_path, key_pem):
    """A restart that fails does not turn the fallback into a success: the pair is
    out of the cert dir either way, the run says the restart failed, and the unit
    still reports failure — the renewal never happened."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse", restart_rc="1")
    assert run.proc.returncode != 0, run.out
    assert len(run.restarted()) == 1, [c["argv"] for c in run.calls]
    assert f"{HOST}.crt" not in run.after
    assert (run.quarantine_dir() / f"{HOST}.crt").exists()
    assert "restart failed" in run.out


def test_the_quarantining_run_names_where_the_pair_went_and_its_notafter(
        tmp_path, key_pem):
    """Clause 4: the journal is the only surface a nightly unit has, so the run
    states which files left, to where, and the notAfter of the leaf it gave up on —
    read here back out of the quarantined file with `openssl x509 -enddate`, not
    from a string this file made up."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse")
    qdir = run.quarantine_dir()
    for name in (f"{HOST}.crt", f"{HOST}.key"):
        assert str(qdir / name) in run.out, f"{name}'s destination is not in: {run.out}"
    assert _notafter_iso(qdir / f"{HOST}.crt") in run.out, \
        f"the quarantined leaf's notAfter is not in: {run.out}"


def test_a_leaf_more_than_a_day_out_is_never_quarantined(tmp_path, key_pem):
    """Clause 3's first input, on the near side of the old 14-day window: a leaf
    two days out with a refused mint is a failed renewal, not a cert Vite is
    serving wrongly. Moving it would downgrade every tailnet client to the
    CA-signed fallback to solve a problem that is not happening today, so the run
    exits non-zero having moved nothing and restarted nobody."""
    run = _run(tmp_path, key_pem, current_days=2, behaviour="refuse",
               fallback_pair=True)
    assert run.proc.returncode != 0, run.out
    assert run.after == run.before, "a 2-day leaf was quarantined"
    assert run.restarted() == [], "restarted for a pair Vite is serving correctly"
    assert not run.quarantine_dir().exists(), "a quarantine dir appeared for a live leaf"


def test_an_absent_pair_is_never_quarantined(tmp_path, key_pem):
    """Clause 3's second input: with no tailnet pair at all Vite has already
    fallen back, so there is nothing to move and no restart to ask for. This is the
    state the run *after* a quarantine is in, which is why a quarantine must not be
    something that happens twice."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse", no_leaf=True)
    assert run.proc.returncode != 0, run.out
    assert len(run.minted()) == 1, "the mint was not attempted with no leaf on disk"
    assert run.restarted() == [], [c["argv"] for c in run.calls]
    assert run.after == run.before == {}
    assert not run.quarantine_dir().exists()


def test_the_run_after_a_quarantine_mints_and_moves_nothing(tmp_path, key_pem):
    """Two consecutive runs over one cert dir, which is what the daily timer does.
    Day one: a 12-hour leaf and a refused mint, so the pair is quarantined and the
    run fails. Day two: nothing in the cert dir and a mint that works, so the pair
    is published, the run exits 0, and the quarantined leaf from day one is left
    alone where it was — the fallback is not a loop that eats its own evidence."""
    day_one = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse",
                   fallback_pair=True)
    assert day_one.proc.returncode != 0, day_one.out
    qdir = day_one.quarantine_dir()
    quarantined = (qdir / f"{HOST}.crt").read_bytes()
    assert quarantined == day_one.before[f"{HOST}.crt"]

    day_two = _run(tmp_path, key_pem, current_days=0.5, behaviour="ok",
                   fixture_days=90.0, no_leaf=True, fallback_pair=False)
    assert day_two.proc.returncode == 0, day_two.out
    assert (day_two.cert_dir / f"{HOST}.crt").read_bytes() != quarantined, \
        "day two did not publish a new leaf"
    assert (qdir / f"{HOST}.crt").read_bytes() == quarantined, \
        "day two disturbed the quarantined leaf"
    assert [c["name"] for c in day_two.calls] == ["tailscale", "supervisorctl"], \
        [c["argv"] for c in day_two.calls]


def test_quarantining_writes_nothing_outside_moving_the_pair(tmp_path, key_pem):
    """Clause 5 as far as a temp tree can pin it: the quarantining run leaves the
    cert dir holding exactly the untouched fallback pair, creates nothing anywhere
    under the temp root but the quarantine dir itself, and the only two commands it
    issued name neither `gen-cert.sh`, the CA, the trust stores, nor the fallback
    pair — which is the whole of what this script is allowed to touch."""
    run = _run(tmp_path, key_pem, current_days=0.5, behaviour="refuse",
               fallback_pair=True)
    assert run.proc.returncode != 0, run.out
    assert set(run.after) == {"lloyd.crt", "lloyd.key"}, run.after
    assert run.after["lloyd.crt"] == run.before["lloyd.crt"]
    assert run.after["lloyd.key"] == run.before["lloyd.key"]
    # Everything here but the quarantine dir is this file's own scaffolding: the
    # stub pair (`bin`), the cert dir, the day's leaf before it was copied in
    # (`current.crt`), the mint fixture, the fallback leaf's source and the argv
    # log. A quarantining run adds exactly one entry, and it is the quarantine.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([
        f".expired-{HOST}", "argv.jsonl", "bin", "cert", "current.crt", "fixture",
        "src"]), sorted(p.name for p in tmp_path.iterdir())
    assert [c["name"] for c in run.calls] == ["tailscale", "supervisorctl"]
    for call in run.calls:
        joined = " ".join(call["argv"])
        for forbidden in ("gen-cert", "ca.crt", "update-ca", "lloyd.crt", "lloyd.key",
                          "/etc/ca-certificates"):
            assert forbidden not in joined, f"{forbidden} appeared in {call['argv']}"
