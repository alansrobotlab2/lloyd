"""Degradation-matrix fault injector and runner (#644).

`matrix.yaml` declares, for each (consumer, dependency, fault), the behavior a
human wrote down. This module makes each row's fault **happen** — a socket that
answers 404, a vault whose `.git/objects` was repacked, a lock file that is
missing, a process table with no media socket — and then calls the real consumer
and reports what it said. No dependency is mocked: the only thing a row can lean
on is a fault that was actually produced.

The method is Eskildsen's at Shopify: write the expected behavior down first, then
inject at the layer where the dependency is actually reached. His two lessons that
survive translation are both implemented here. The first is that mocks prove the
mock. The second is that a general-purpose proxy only covers part of the graph —
Toxiproxy speaks TCP, and most of Lloyd's gate bugs live in *state files* and
*endpoint schemes*, which no TCP proxy can express. So `injection` names one of
three fixture kinds, and none of them is a proxy:

    fixture-listener   a socket the runner owns, answering badly on purpose
    fixture-state      a temp directory holding the file a gate reads
    fixture-proc       a synthesised ``/proc`` layout for the media-plane probe

**Never a live target.** The box this runs on is serving Alan while the suite
runs, and a false "X is down" report is the exact defect the matrix catalogues —
so before any fault is applied the runner proves the target is its own: it bound
the port itself, or the path lives under its temp root. There is no flag that
injects against a live port, and `tests/test_degradation_matrix.py` holds a port
and a live state path to prove the refusal is real rather than a docstring.

Run it::

    python3 -m tests.degradation.runner --check            # every row
    python3 -m tests.degradation.runner --row D-VOICE-PROC-UNREADABLE
    python3 -m tests.degradation.runner --check --json

Exit status is 0 only when every row's declared behavior held.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MATRIX = HERE / "matrix.yaml"
VAULT = Path(os.environ.get("LLOYD_VAULT") or (Path.home() / "obsidian"))

# Fault-injection budget for the whole matrix. The acceptance bar is "headless,
# under ten minutes"; this is the same number, stated so a row that starts
# hanging is a visible timeout rather than a run that dies in a gate at 9 minutes.
SUITE_TIMEOUT_S = 480
# Per-row ceiling, so one wedged listener cannot eat the suite budget.
ROW_TIMEOUT_S = 30

# The name the matrix's rows are marked with, in one place. `pytest.ini` registers nothing —
# #644's acceptance check forbids editing that file, and it is a path the loop may not write —
# so the only thing standing between a bare `pytest -q` and these nodes binding ports is the
# gate's `-m` expression. `test_degradation_contract.py` asserts that expression carries a
# `not <MARKER_NAME>` term, which is a real claim about two modules, rather than comparing two
# hand-copied strings that can each drift without the other noticing.
MARKER_NAME = "fault_injection"

# Ports the runner may bind. A free port picked by the kernel (`bind(0)`) is
# always used; this range exists only so the refusal check has something to
# compare a *live* port against when a row ever names one explicitly.
FIXTURE_PORT_RANGE = range(41000, 42000)


class InjectRefused(Exception):
    """A fault whose target a live process already owns — or a fault the runner
    cannot prove is isolated, which is treated the same way."""


class RowError(Exception):
    """A row could not be executed at all (missing consumer, bad args)."""


# ---------------------------------------------------------------- matrix load
def load_matrix(path=MATRIX):
    """The rows, as data. Kept separate from validation so a test can load a
    deliberately broken copy without the validator getting a say."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["rows"]


def _incident_exists(incident):
    """An `incident` must resolve to a real artifact, or be the literal
    `unmotivated`. A row citing a note that no longer exists is a row that will be
    trusted by nobody, which is the same as no citation at all."""
    if incident == "unmotivated":
        return True
    p = VAULT / incident
    if p.exists():
        return True
    # A code-side citation (no vault path) is allowed to name a file in the checkout.
    return (REPO / incident).exists()


def validate_rows(rows):
    """Structural rules only — no faults, no consumers. Cheap enough to run in a
    plain `pytest -q` on every commit, which is the point: the shape of the spec is
    a claim that should fail loudly and instantly."""
    errors = []
    required = ("id", "dependency", "consumer", "fault", "injection",
                "declared_behavior", "expect", "incident")
    for row in rows:
        missing = [k for k in required if k not in row or row[k] in (None, "")]
        if missing:
            errors.append(f"{row.get('id', '?')}: missing {', '.join(missing)}")
            continue
        if "reported" not in row["expect"]:
            errors.append(f"{row['id']}: expect.reported is the assertion; expect has "
                          f"{sorted(row['expect'])}")
        if not _incident_exists(row["incident"]):
            errors.append(f"{row['id']}: incident {row['incident']!r} resolves to no file "
                          f"in the vault or the checkout, and is not the literal 'unmotivated'")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate row ids: {', '.join(dupes)}")
    return errors


def resolve_consumer(ref):
    """`path::attribute` -> the callable, imported from wherever it really lives.

    The consumer strings in the matrix are not documentation: each one is resolved
    on every run, so a renamed or deleted function fails the row rather than quietly
    testing a copy. Vault paths are imported from the vault, repository paths through
    the normal package path.
    """
    import importlib.util

    path, _, attr = ref.partition("::")
    if not attr:
        raise RowError(f"consumer {ref!r} is not path::function")
    target = (VAULT / path) if not (REPO / path).exists() else (REPO / path)
    if not target.exists():
        raise RowError(f"consumer file {target} does not exist")
    mod_name = "lloyd_matrix_" + path.replace("/", "_").replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, target)
    if spec is None or spec.loader is None:
        raise RowError(f"cannot build an import spec for {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, attr, None)
    if fn is None:
        raise RowError(f"{target} has no attribute {attr!r}")
    return module, fn


# ------------------------------------------------------------ live-target guard
def port_in_use(port, host="127.0.0.1"):
    """Is something else already listening? That something is a live service, and a
    fault injected against it would be reported by the daily health check as an
    outage of the real dependency."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def bind_fixture_socket(preferred=None, host="127.0.0.1"):
    """A TCP socket this process owns, refused if anything else already answers.

    The order is the whole mechanism. A port that is **already answering** belongs
    to a live service, and a fault injected into it would be reported by the daily
    health check as an outage of the real dependency — the very defect this matrix
    catalogues, manufactured by the suite meant to catch it. So liveness is checked
    *before* the bind, and the successful bind is what proves the port is ours.
    Checking afterwards would refuse every fixture, because a listening socket is by
    then answering on its own port.

    Every probe here binds with `preferred=None`, so the kernel picks a free port
    and no row can name a live one. `preferred` exists so the refusal is testable
    rather than merely asserted: `tests/test_degradation_matrix.py` holds a port,
    asks for that same number, and the run has to say no.
    """
    if preferred is not None and port_in_use(preferred, host):
        raise InjectRefused(
            f"port {preferred} is already answering — a live instance owns it, so the "
            f"fault will not be injected there. Every row targets a port this process "
            f"binds itself; there is no flag that injects against a live dependency.")
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, preferred or 0))
    except OSError as exc:
        sock.close()
        raise InjectRefused(f"cannot take a fixture port ({preferred}): {exc}") from exc
    return sock


def assert_fixture_path(path: Path, root: Path):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if not str(resolved).startswith(str(root_resolved) + os.sep):
        raise InjectRefused(
            f"{resolved} is outside this run's temp root {root_resolved} — injecting a "
            f"state-file fault into a live path would delete or age a file a running job "
            f"is reading. The gate bugs this matrix catalogues are all about state files; "
            f"that is a reason for isolation, not for touching them.")
    return resolved


@contextlib.contextmanager
def fixture_root(name="degradation"):
    root = Path(tempfile.mkdtemp(prefix=f"lloyd-{name}-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def env(**values):
    """Set environment variables for the duration of a probe and restore exactly.

    The vault consumers take their seams from the environment
    (`LLOYD_HEALTH_PROC_ROOT`, `AUTONOMY_DIR`, `LLOYD_HEALTH_SUPERVISORCTL`), which is
    how a fixture reaches code that was written to read the real machine.
    """
    saved = {k: os.environ.get(k) for k in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ------------------------------------------------------------------- fixtures
class DeadPort:
    """A port number that was bound and then released, so connecting is refused.

    The refused-connection fault needs a number that is *plausibly* a service and is
    guaranteed empty; `bind(0)` then `close()` is the only honest way to get one.
    """

    def __init__(self):
        # bind, read the number, release: the port was ours, so connecting to it now
        # is refused by the kernel rather than by whoever owns the service.
        with contextlib.closing(bind_fixture_socket()) as sock:
            self.port = sock.getsockname()[1]

    def urls(self):
        return [f"http://localhost:{self.port}/"]


class FixtureListener:
    """A socket that answers badly on purpose.

    `silence` accepts and never replies — the hung-worker fault, which no amount of
    status-code handling will ever see coming, and which the timeout row exists to
    bound. `code`/`body` reply with a chosen status and bytes. `tls` wraps with a
    throwaway self-signed certificate, which is how a trust failure is produced
    without a certificate authority, a hostname, or the live frontend.
    """

    def __init__(self, code=200, body=b"", silence=False, tls=False, schemes=("http",)):
        self.code, self.body, self.silence = code, body, silence
        self.tls, self.schemes = tls, tuple(schemes)
        self._sock = bind_fixture_socket()
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def base(self):
        return self.schemes[0]

    def urls(self):
        return [f"{scheme}://localhost:{self.port}/" for scheme in self.schemes]

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            sock = conn
            if self.tls:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=_TLS_CERT, keyfile=_TLS_KEY)
                sock = context.wrap_socket(conn, server_side=True)
            sock.recv(4096)  # the request line is all we need; the fault is the reply
            if self.silence:
                time.sleep(10)  # never answers, exactly like a wedged worker pool
                return
            reason = _HTTP_REASONS.get(self.code, "Status")
            head = (f"HTTP/1.1 {self.code} {reason}\r\n"
                    f"Content-Length: {len(self.body)}\r\n"
                    "Content-Type: text/plain\r\nConnection: close\r\n\r\n").encode()
            sock.sendall(head + self.body)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def close(self):
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()


_HTTP_REASONS = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}
_TLS_DIR = None
_TLS_CERT = _TLS_KEY = None


def _tls_material():
    """A throwaway self-signed certificate, generated once per process.

    `ssl` cannot offer a server without a key pair, and issuing one is the only way
    to inject a *trust* fault — as opposed to a scheme fault or a refusal — without
    touching the box's real certificate. It is generated into a temp directory and
    never referenced by anything else.
    """
    global _TLS_DIR, _TLS_CERT, _TLS_KEY
    if _TLS_CERT:
        return _TLS_CERT, _TLS_KEY
    _TLS_DIR = Path(tempfile.mkdtemp(prefix="lloyd-degradation-tls-"))
    _TLS_CERT, _TLS_KEY = _TLS_DIR / "c.pem", _TLS_DIR / "k.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-keyout", str(_TLS_KEY), "-out", str(_TLS_CERT), "-subj", "/CN=localhost"],
        check=True, capture_output=True)
    return _TLS_CERT, _TLS_KEY


# ------------------------------------------------------------------- adapters
# Each adapter: build the fault, call the real consumer, return the token the row
# asserts plus the evidence a human reads. An adapter that could not produce the
# fault raises; it never reports a verdict it could not back.
def _probe_http(row):
    args = row.get("args", {})
    module, check_endpoints = resolve_consumer(row["consumer"])
    if args.get("closed"):
        target = DeadPort()
        endpoint = {"name": row["id"], "port": target.port, "url": target.urls()[0]}
        listener = None
    else:
        schemes = tuple(args.get("urls") or ("http",))
        # TLS is chosen by the row, never inferred from the probe list. The
        # scheme-mismatch row is precisely "the list says https, the service says
        # http", and inferring TLS from the list would serve TLS and test nothing.
        tls = args.get("tls") == "self-signed"
        if tls:
            _tls_material()
        listener = FixtureListener(code=args.get("code", 200),
                                   body=str(args.get("body", "")).encode(),
                                   silence=args.get("silence", False),
                                   tls=tls,
                                   schemes=schemes)
        endpoint = {"name": row["id"], "port": listener.port, "schemes": list(schemes)}
    if args.get("ignore_404"):
        endpoint["ignore_404"] = True
    if args.get("expect_payload"):
        endpoint["expect"] = args["expect_payload"]
    if args.get("timeout") is not None:
        endpoint["timeout"] = float(args["timeout"])
    try:
        rows = check_endpoints([endpoint])
    finally:
        if listener:
            listener.close()
    if not rows:
        raise RowError("check_endpoints returned no rows, so nothing was probed")
    result = rows[0]
    evidence = (f"state={result.get('state')} healthy={result.get('healthy')} "
                f"{result.get('error') or result.get('note') or ''}".strip()
                .replace("\n", " "))
    return str(result.get("state")), evidence


def proc_tables(root, pids=(900001,), media_ports=(), signal_clients=0,
                signal_port=7880):
    """Write a `/proc` shaped exactly like the kernel's under `root`, and return `root`.

    One definition, used by the voice rows and by `tests/test_degradation_matrix.py`, so a
    fixture cannot drift from the format the probe is asserted against. Field order is the
    kernel's: field 1 is `local_address` (`hex(ip):hex(port)`), field 2 the peer, field 3
    `st`. The media pool is per-namespace and read through `/proc/<pid>/net/udp{,6}`; the
    signalling connections are machine-wide, from `/proc/net/tcp{,6}`. `01` is ESTABLISHED.

    A `cmdline` naming `livekit-server` is written for each pid because that is how the probe
    finds the process — a fixture without it exercises the no-process state, not the media
    state, which is the difference between two different rows.
    """
    root = Path(root)
    header = ("   sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
              "retrnsmt   uid  timeout inode")
    for pid in pids:
        pid_dir = root / str(pid)
        (pid_dir / "net").mkdir(parents=True)
        (pid_dir / "cmdline").write_bytes(b"livekit-server\x00--config\x00x")
        lines = [header]
        for port in media_ports:
            lines.append(f"  0: 00000000:{port:04X} 00000000:0000 07 00000000:00000000 "
                         "00:00000000     0        0 1 2 3 4 5 6")
        for name in ("udp", "udp6"):
            (pid_dir / "net" / name).write_text("\n".join(lines) + "\n")
    net = root / "net"
    net.mkdir(parents=True, exist_ok=True)
    tcp = [header]
    for index in range(int(signal_clients)):
        # A client holding a signalling connection was promised a media plane, which is
        # what separates a broken SFU from an idle one.
        tcp.append(f" {index}: 0100007F:{signal_port:04X} 0100007F:C000 01 "
                   "00000000:00000000 00:00000000     0        0 1 2 3 4 5 6")
    for name in ("tcp", "tcp6"):
        (net / name).write_text("\n".join(tcp) + "\n")
    return root



def _probe_voice(row):
    """Synthesise the kernel's answer and ask the health checker what it thinks.

    LiveKit's media-plane evidence is two socket tables: the UDP ports in the media
    pool, which live in the SFU's network namespace and are read through
    `/proc/<pid>/net/udp`, and the TCP signalling connections on 7880, read from the
    machine-wide tables. The fixture writes both, in the kernel's own hex format,
    under a temp `proc_root` the probe is pointed at by `LLOYD_HEALTH_PROC_ROOT`.

    Nothing about the real SFU is touched. The probe reads a directory this run made;
    if it ever read the real `/proc` the fixture root would be unused and the row
    would report the live machine, which is why the reported `proc_root` is part of
    the evidence line and asserted to be the fixture.
    """
    args = row.get("args", {})
    module, check_voice_media = resolve_consumer(row["consumer"])
    outage = getattr(module, "voice_media_is_outage", None)
    if outage is None:
        raise RowError(f"{row['consumer']} has no voice_media_is_outage to reduce with")
    signal_port = int(getattr(module, "VOICE_MEDIA_SIGNAL_PORT", 7880))
    with fixture_root("proc") as root:
        proc_root = root / "proc"
        proc_root.mkdir()
        if args.get("unreadable_proc"):
            pid_dir = proc_root / "900001"
            (pid_dir / "net").mkdir(parents=True)
            (pid_dir / "cmdline").write_bytes(b"livekit-server\x00--config\x00x")
            (proc_root / "net").mkdir()
            # Unreadable means unreadable, not merely empty. Mode 000 on the tables the
            # probe has to open is how a socket table in a namespace this user cannot
            # enter behaves. An empty table is a different fault with a different
            # correct answer (idle), so this fixture does not stand in for it.
            for name in ("net/tcp", "net/tcp6"):
                (proc_root / name).write_text("x\n")
                (proc_root / name).chmod(0o000)
            for name in ("udp", "udp6"):
                (pid_dir / "net" / name).write_text("x\n")
                (pid_dir / "net" / name).chmod(0o000)
        else:
            proc_tables(proc_root, pids=tuple(args.get("pids", [900001])),
                        media_ports=tuple(args.get("media_ports", ())),
                        signal_clients=int(args.get("signal_clients", 0)),
                        signal_port=signal_port)
        with env(LLOYD_HEALTH_PROC_ROOT=proc_root):
            result = check_voice_media()
    if not result:
        raise RowError("check_voice_media returned no row")
    if str(result.get("proc_root", "")).startswith(str(proc_root)) is False:
        raise RowError(f"the probe read {result.get('proc_root')!r}, not this fixture's "
                       f"{proc_root} — the row would be reporting the live machine")
    evidence = (f"state={result.get('state')} healthy={result.get('healthy')} "
                f"outage={bool(outage(result))} "
                f"{result.get('error', '')}".strip().replace("\n", " "))
    return str(result.get("state")), evidence


def git_in(vault, *args, check=False):
    """One git invocation inside a fixture vault, output kept and errors swallowed by
    default so a `check=False` call stays a one-liner."""
    return subprocess.run(["git", *[str(a) for a in args]], cwd=str(vault),
                          capture_output=True, text=True, check=check)


def _commit(vault, message, changes=None):
    for name, body in (changes or {}).items():
        target = Path(vault) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    git_in(vault, "add", "-A")
    git_in(vault, "-c", "user.email=matrix@example.invalid", "-c", "user.name=matrix",
           "commit", "-q", "-m", message)


def init_fixture_repo(vault, history_commits=40):
    """Give the fixture real history, so its `.git` holds real loose objects.

    40 commits that each change one scratch file write roughly 120 loose objects — a blob,
    a tree and a commit each — which is what a `git repack -a -d` then folds into a single
    pack file. That is the exact mechanism of the 2026-09-17 false `data_damage` trip
    ("vault files dropped 6.7% (6075 → 5667)" while the note count sat at 5622 the whole
    time). Fabricating `objects/xx/hash` files by hand instead would look convincing and
    prove nothing: `git gc` does not touch directories it does not recognise, so a
    hand-made `.git` does not shrink on repack and the control that distinguishes "the
    filter held" from "the fixture never moved" would report the former either way.
    """
    git_in(vault, "init", "-q", check=True)
    _commit(vault, "initial")
    for i in range(history_commits):
        _commit(vault, f"scratch {i}", {f".matrix/scratch-{i % 4}.txt": f"{i}\n"})
    return vault


def remove_loose_objects(vault):
    """Pack the loose objects with zero notes touched. `git status --porcelain` stays
    empty; only the number of files under `.git` moves."""
    git_in(vault, "-c", "gc.auto=0", "repack", "-a", "-d", "-q", check=True)


def git_porcelain(vault) -> str:
    """`git status --porcelain`, which is the whole point of calibration 2: the working
    tree is clean before and after the repack, so any movement in a content count is a
    count that was never counting content."""
    return git_in(vault, "status", "--porcelain").stdout


def unfiltered_count(vault) -> int:
    """Every file under the vault, `.git/**` included — the pre-fix reading, used only as
    the control that proves the guarded assertion is holding something up."""
    return sum(1 for path in Path(vault).rglob("*") if path.is_file())


def guarded_count(vault) -> int:
    """The promoter's own count, the number the pre-landing baseline is written from."""
    from scripts.automod import promote as P
    return P.count_vault_files(Path(vault))


def delete_notes(vault, notes=30):
    """The fault the data_damage trip exists for, injected for real."""
    removed = 0
    for path in sorted(Path(vault).rglob("*.md")):
        if path.relative_to(vault).parts[0] in ("knowledge", "backlog") and removed < notes:
            path.unlink()
            removed += 1
    return removed


def loose_object_count(vault) -> int:
    """Files under `.git/objects` — loose objects plus the pack and index files."""
    fanout = Path(vault) / ".git" / "objects"
    return sum(1 for path in fanout.rglob("*") if path.is_file())


def _fixture_vault(root, notes=30, loose_objects=400):
    """A vault-shaped directory with `notes` notes and a git repo holding ~`loose_objects`
    loose objects. The count is approximate by design — the assertion that matters is that
    repacking moves the `.git` file count while the note count does not, and both sides of
    that are asserted, not assumed."""
    vault = root / "obsidian"
    for i in range(notes):
        folder = vault / ("knowledge" if i % 2 else "backlog")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{i:03d}.md").write_text(f"# note {i}\n")
    init_fixture_repo(vault, history_commits=max(1, int(loose_objects // 3)))
    return vault


def _probe_vault_count(row):
    """Repack `.git` with zero notes lost, then ask both counters what happened.

    The mutation is real: `git init`, one commit, `git rebase` to rewrite history so
    the old objects become garbage, then `git gc --prune=now` deletes them. Nothing
    in the tree that is a *note* changes, and `git status --porcelain` stays empty —
    the exact condition under which the guardian's `data_damage` trip fired twice and
    spent two items' unattended attempts.
    """
    args = row.get("args", {})
    promote, promote_count = resolve_consumer(row["consumer"])
    guardian = resolve_consumer("agent-services/guardian/guardian.py::count_vault_files")[1]
    with fixture_root("vault") as root:
        vault = _fixture_vault(root, notes=int(args.get("notes", 30)),
                              loose_objects=int(args.get("loose_objects", 400)))

        def git(*a, check=True):
            return subprocess.run(["git", *a], cwd=str(vault), capture_output=True,
                                  text=True, check=check)

        before = (promote_count(vault), guardian(str(vault)))
        if args.get("mutation") == "git_repack":
            remove_loose_objects(vault)
        elif args.get("mutation") == "delete_notes":
            for path in sorted(vault.glob("*/*.md"))[: int(args.get("delete", 30))]:
                assert_fixture_path(path, root)
                path.unlink()
            git("add", "-A", check=False)
            _commit(vault, "notes removed by the matrix")
        else:
            raise RowError(f"unknown vault mutation {args.get('mutation')!r}")
        after = (promote_count(vault), guardian(str(vault)))
        porcelain = git("status", "--porcelain").stdout.strip()
    delta = after[0] - before[0]
    guardian_delta = after[1] - before[1]
    token = "unchanged" if delta == 0 and guardian_delta == 0 else (
        "dropped" if delta < 0 or guardian_delta < 0 else "grew")
    evidence = (f"promoted={delta} guardian={guardian_delta} "
                f"notes {before[0]}->{after[0]} porcelain_empty={not porcelain}")
    return token, evidence


def _consolidation_gate_source():
    """The Phase-0 gate exactly as the skill tells a run to execute it.

    Lifted out of `SKILL.md` rather than re-implemented: a fixture that carries its
    own copy of the gate keeps passing after someone edits the skill, which is how
    this gate stayed permanently open for six days in September — the text that was
    reviewed and the text that ran were not the same text. If the skill stops looking
    like a python heredoc this raises, and the row fails loudly instead of quietly
    testing a copy.
    """
    skill = VAULT / "skills" / "dream-consolidation" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    blocks = [block for block in text.split("```") if "LOCK_FILE" in block]
    if not blocks:
        raise RowError(f"no Phase-0 lock gate found in {skill}")
    source = blocks[0]
    start = source.find("import ")
    end = source.rfind("PYEOF")
    if start < 0 or end < 0:
        raise RowError(f"the Phase-0 gate in {skill} is no longer a python heredoc this "
                       f"extractor can lift (first import at {start}, marker at {end})")
    return source[start:end]


def _probe_consolidation(row):
    args = row.get("args", {})
    with fixture_root("consolidation") as root:
        memory_root = root / "lloyd"
        memory_root.mkdir()
        sessions = root / "sessions"
        sessions.mkdir()
        if args.get("lock") == "fresh":
            (memory_root / ".consolidate-lock").write_text("1")
        elif args.get("lock") not in (None, "absent"):
            raise RowError(f"unknown lock fault {args.get('lock')!r}")
        source = (_consolidation_gate_source()
                  .replace('MEMORY_ROOT = Path.home() / "obsidian" / "lloyd"',
                           f'MEMORY_ROOT = Path({str(memory_root)!r})')
                  .replace('SESSIONS_DIR = Path.home() / "lloyd" / "sessions"',
                           f'SESSIONS_DIR = Path({str(sessions)!r})'))
        if "MEMORY_ROOT = Path(" not in source or "SESSIONS_DIR = Path(" not in source:
            raise RowError("the Phase-0 gate no longer assigns MEMORY_ROOT/SESSIONS_DIR "
                           "in the shape the extractor rewrites; update the extractor "
                           "rather than trusting this row")
        script = root / "phase0.py"
        script.write_text(source)
        result = subprocess.run([sys.executable, str(script)], capture_output=True,
                                text=True, timeout=30)
        assert_fixture_path(memory_root / ".consolidate-lock", root)
        output = (result.stdout + result.stderr).strip()
    for token in ("LOCK_MISSING", "GATE_FAIL", "GATE_PASS"):
        if token in output:
            first = next(line for line in output.splitlines() if token in line)
            return token, first.strip()[:200]
    return "no-verdict", (output.splitlines() or ["the gate printed nothing"])[0][:200]


def _probe_dependency(row):
    """Ask the scheduler's own due gate whether a dependent may run.

    The fault is a task file in a status the resolution set used not to contain, in a
    temp autonomy directory; `dependency_resolution_set(directory)` and
    `_all_board_tasks(directory)` both take that directory, which is the seam #870
    left for exactly this. The real `_is_task_due` decides — the world is injected,
    never the answer.

    `next_run` is pinned an hour in the past and `now` is passed explicitly, so the
    status, skill and hours gates are satisfied by construction and the dependency
    gate is the only one that can hold the task. `hold_reason` calls the same
    predicates in the same order, and its text is the evidence: a reader can see
    *which* gate answered, not just that something did.
    """
    args = row.get("args", {})
    autonomy, is_task_due = resolve_consumer("autonomy.py::_is_task_due")
    resolution_set = getattr(autonomy, "dependency_resolution_set", None)
    board = getattr(autonomy, "_all_board_tasks", None)
    if resolution_set is None or board is None:
        raise RowError("autonomy.py no longer exposes dependency_resolution_set and "
                       "_all_board_tasks; #870 moved this seam and the adapter has to "
                       "move with it rather than pass on a guess")
    with fixture_root("autonomy") as root:
        (root / "40-dependent.md").write_text(
            "---\nid: 40\nname: dependent\ntask: dependent\nstatus: up_next\n"
            "skill_name: nightly-reflection-signals\nfrequency: hourly\nagent: a1\n"
            "depends_on: 42\n---\n")
        if args.get("dangling"):
            (root / "39-upstream.md").write_text(
                "---\nid: 39\nname: upstream\ntask: upstream\nstatus: up_next\n"
                "skill_name: nightly-reflection-knowledge-write\nfrequency: hourly\n"
                "agent: a1\n---\n")
            for path in root.glob("*.md"):
                path.write_text(path.read_text().replace(
                    "depends_on: 42", f"depends_on: {int(args['dangling'])}"))
        else:
            status = args.get("upstream_status", "paused")
            (root / "42-upstream.md").write_text(
                f"---\nid: 42\nname: upstream\ntask: upstream\nstatus: {status}\n"
                "skill_name: nightly-vault-maintenance\nfrequency: hourly\nagent: a1\n---\n")
        resolution = resolution_set(root)
        dependent = next((t for t in board(root) if str(t.get("id")) == "40"), None)
        if dependent is None:
            raise RowError("the fixture dependent task did not parse; the task-file schema "
                           "moved and this row is no longer testing what it names")
        now = datetime.datetime.now(datetime.timezone.utc)
        dependent["next_run"] = (now - datetime.timedelta(hours=1)).isoformat()
        held = not is_task_due(dependent, resolution, now=now)
        reason = autonomy.hold_reason(dependent, resolution, now=now) or "due"
    what = (f"depends_on={args['dangling']} (no such file)" if args.get("dangling")
            else f"upstream 42 status={args.get('upstream_status', 'paused')}")
    return ("held" if held else "runs-anyway", f"{what} hold_reason={str(reason)[:80]}")


def _probe_kg(row):
    args = row.get("args", {})
    module, _store = resolve_consumer("app/kg_store.py::KGStore")
    with fixture_root("kg") as root:
        db = root / "kg.sqlite"
        if args.get("mutation") == "corrupt":
            db.write_bytes(b"this is not a database, it is a truncated write")
        elif args.get("mutation") == "absent":
            pass
        else:
            raise RowError(f"unknown kg mutation {args.get('mutation')!r}")
        try:
            store = module.KGStore(db)
            count = store.entities.count() if hasattr(store.entities, "count") else None
            token, detail = "opened", f"entities={count}"
        except module.StoreUnavailable as exc:
            return "store-unavailable", f"StoreUnavailable: {str(exc)[:160]}"
        except Exception as exc:  # noqa: BLE001 - the point is what the store raises
            return f"opened-but-{type(exc).__name__}", str(exc)[:160]
    return token, detail


def _probe_supervisor(row):
    """Ask `check_services` about a supervisor it cannot reach, two ways.

    `args.real_binary` invokes the REAL `supervisorctl` against a conf whose
    `serverurl` names a socket that does not exist. The other half of the matrix injects
    through a fixture, and here that would be the wrong seam: a shim decides its own exit
    code and its own stdout, so the row would only ever test `check_services` against
    output this file invented. The real binary answers `rc=4` and one prose line —
    `unix:///…/absent.sock no such file` on stdout, not stderr, not empty, not the shape a
    shim would have produced — and only that response can say whether the consumer's
    parser treats an unrecognised status token as a broken reading. That distinction is
    #1191: an unrecognised token must not read as a stopped program, and it cannot be
    discovered by a fixture that was told what to print.

    Blast radius: the conf and its socket are inside this run's fixture root, never
    `/tmp/agent-supervisor.sock`, and `status` is the only verb, so the daemon serving Alan
    is neither addressed nor changed. `assert_fixture_path` is what holds that line.
    """
    args = row.get("args", {})
    _module, check_services = resolve_consumer(row["consumer"])
    services_healthy = resolve_consumer(
        "skills/system-health-check/system_health_check.py::services_healthy")[1]
    services_reason = resolve_consumer(
        "skills/system-health-check/system_health_check.py::services_reason")[1]
    with fixture_root("supervisor") as root:
        missing_sock = root / "agent-supervisor.sock"
        conf = root / "supervisord.conf"
        if args.get("real_binary"):
            binary = "unset"  # pop the override so supervisorctl_command() uses its own default
            conf.write_text(f"[unix_http_server]\nfile={missing_sock}\n\n"
                            f"[supervisorctl]\nserverurl=unix://{missing_sock}\n")
        else:
            binary = root / "supervisorctl"
            binary.write_text(f"#!/bin/sh\ncat <<'EOF'\n{args.get('stdout', '')}EOF\n"
                              f"exit {int(args.get('rc', 0))}\n")
            binary.chmod(0o755)
            conf.write_text(f"[unix_http_server]\nfile={missing_sock}\n")
        assert_fixture_path(conf, root)
        assert not missing_sock.exists(), f"{missing_sock} exists; the fault is not injected"
        with env(LLOYD_HEALTH_SUPERVISOR_CONF=str(conf),
                 LLOYD_HEALTH_SUPERVISORCTL=(None if binary == "unset" else str(binary))):
            # Read the command back through the consumer's own resolver rather than
            # restating it: this is the control that says the seam crossed to a real
            # executable, and that the only thing redirected is the conf — the verb and the
            # argument order are the ones production uses.
            argv = _module.supervisorctl_command()
            assert Path(argv[0]).is_file() and os.access(argv[0], os.X_OK), (
                f"{row['id']}: seam would invoke {argv[0]!r}, which is not an executable")
            assert argv[argv.index("-c") + 1] == str(conf), (
                f"{row['id']}: the binary was handed a conf that is not this fixture's: {argv}")
            assert argv[-1] == "status", f"{row['id']}: {argv} is not a read-only verb"
            results = check_services()
        healthy = services_healthy(results)
        reason = services_reason(results) if not healthy else ""
    token = "healthy" if healthy else "not-healthy"
    invoked = "supervisorctl(default)" if binary == "unset" else "supervisorctl(shim)"
    return token, f"rows={len(results)} invoked={invoked} {reason}".strip()[:200]


INJECTIONS = None  # set to frozenset(PROBES) below, once PROBES exists
PROBES = {
    "http": _probe_http,
    "voice": _probe_voice,
    "vault_count": _probe_vault_count,
    "consolidation": _probe_consolidation,
    "dependency": _probe_dependency,
    "kg": _probe_kg,
    "supervisor": _probe_supervisor,
}


# -------------------------------------------------------------------- engine
INJECTIONS = frozenset(PROBES)

# How each row's fault is produced, as a closed set. Every value names a fixture mechanism
# and nothing else: `fixture-listener*` binds a port this process owns, `fixture-state`
# builds a temp tree, `fixture-proc` writes a `/proc` shaped like the kernel's under a temp
# root. There is no class that means "against the running service" — that is the blast-radius
# clause of #644, and naming it here is what keeps a fourth row from quietly reaching out.
INJECTION_CLASSES = frozenset({
    "fixture-listener",            # a real socket, answering what the row asks
    "fixture-listener-closed",     # a real socket, deliberately torn down: connection refused
    "fixture-state",               # a temp tree: repos, lock files, task files, databases
    "fixture-proc",                # a temp /proc: socket tables the probe has to parse
})


def run_row(row):
    """Execute one row. Returns the record whether the declared behavior held."""
    probe = row.get("probe")
    if probe not in PROBES:
        return {"row_id": row["id"], "fault": row["fault"],
                "reported": f"row-error: no probe adapter {probe!r}", "passed": False,
                "evidence": "the row names a probe this runner does not implement"}
    try:
        reported, evidence = PROBES[probe](row)
    except InjectRefused as exc:
        return {"row_id": row["id"], "fault": row["fault"],
                "reported": "refused: live target", "passed": False, "evidence": str(exc)}
    except RowError as exc:
        return {"row_id": row["id"], "fault": row["fault"],
                "reported": f"row-error: {exc}", "passed": False, "evidence": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # One row's consumer blowing up is that row's finding, not the suite's. An
        # aborting runner reports nothing at all, which is the least useful possible
        # output from a robustness harness.
        return {"row_id": row["id"], "fault": row["fault"],
                "reported": f"consumer-raised: {type(exc).__name__}", "passed": False,
                "evidence": str(exc)[:300]}
    problems = verdict_problems(row, reported, evidence)
    return {"row_id": row["id"], "fault": row["fault"], "reported": reported,
            "passed": not problems, "evidence": evidence,
            **({"problems": problems} if problems else {})}


# Words a consumer may emit that mean "this dependency is fine". A row flagged
# `expect.non_healthy` may never resolve to one of them, which is what makes clause 6's
# "a silent healthy is not representable" a machine check rather than an aspiration: the
# token is checked on both sides, the reported state and the evidence text.
HEALTHY_TOKENS = frozenset({"answered", "healthy", "ok", "green", "unchanged", "held-ok"})
HEALTHY_EVIDENCE_RE = re.compile(r"\bhealthy\s*[=:]\s*True\b", re.IGNORECASE)


def verdict_problems(row, reported, evidence):
    """Why this verdict does not satisfy the row's declared behavior; empty when it does.

    Public because the calibration clause has to be testable against a verdict the code no
    longer produces. All three still-live incidents were fixed before this matrix's code half
    landed (vault `41f545e5` and `2910b05c` for the two health-checker rows, `add6692` for the
    vault-count row), so "this row fails on unmodified code" cannot be shown by running the old
    code at test time — the pre-fix consumer is not on disk any more. It is shown by feeding
    the recorded old verdict through this same matcher and watching it fail, which
    `tests/test_degradation_contract.py::test_each_calibrated_row_rejects_the_verdict_the_old_code_gave`
    does, one case per incident, each string quoted from the pre-fix bytes. A matcher sealed
    inside `run_row` could not be pointed at that.
    """
    expect = row["expect"]
    problems = []
    if reported != expect["reported"]:
        problems.append(f"reported {reported!r}, declared {expect['reported']!r}")
    for needle in expect.get("contains", []):
        if needle not in evidence:
            problems.append(f"evidence does not contain {needle!r}")

    # Clause 6 of #644: where the fault removes the input rather than degrading it, a healthy
    # verdict is not one of the answers this row can produce — checked on the token and on the
    # evidence text, because a consumer that writes `healthy=True` into its note while naming
    # some other state is exactly the shape `ignore_404` had (`state` was None, `healthy` was
    # True). An empty verdict is rejected too: "the probe said nothing" is how the media-socket
    # incident reached a green daily report.
    if expect.get("non_healthy"):
        if not str(reported).strip():
            problems.append("a non-healthy row cannot be satisfied by an empty verdict")
        elif str(reported).strip() in HEALTHY_TOKENS:
            problems.append(f"reported {reported!r}, which reads as healthy on a row whose "
                            f"fault removed the probe")
        if HEALTHY_EVIDENCE_RE.search(evidence or ""):
            problems.append("evidence asserts healthy=True on a row whose fault removed the "
                            "probe — a silent healthy must not be representable")
    return problems


def verdict_holds(row, reported, evidence) -> bool:
    return not verdict_problems(row, reported, evidence)


def guardian_count(vault):
    """The guardian's own count of the same tree, through the matrix's consumer resolver.

    The two counters are one definition on purpose (`vaultwatch.measure`). A test that checked
    only the promoter's would pass on a one-sided patch, which is how the 2026-09-17 false
    `data_damage` trip survived for a day: the promoter and the guardian held private copies
    of the same `rglob`, and fixing one of them changed nothing.
    """
    _module, fn = resolve_consumer("agent-services/guardian/guardian.py::count_vault_files")
    return fn(str(vault))


def run_matrix(rows=None, only=None, stream=None):
    """Run every row and print exactly one evidence line each: `row | fault | reported`.

    The line is the deliverable, not a log. It is what makes "the consumer said
    healthy" a claim a reader can check instead of an assertion, and the reason a
    healthy verdict with no probe behind it stops being silently emittable: the row
    has to say what it measured, in the same line that says it passed.
    """
    out = stream or sys.stdout
    rows = load_matrix() if rows is None else rows
    errors = validate_rows(rows)
    if errors:
        for error in errors:
            print(f"MATRIX INVALID | {error}", file=out)
        return False
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["id"] in wanted]
        missing = wanted - {row["id"] for row in rows}
        if missing:
            for name in sorted(missing):
                print(f"MATRIX INVALID | no row {name}", file=out)
            return False
    started = time.monotonic()
    records = []
    for row in rows:
        deadline = started + SUITE_TIMEOUT_S
        if time.monotonic() >= deadline:
            print(f"{row['id']} | {row['fault']} | SKIPPED: suite budget of "
                  f"{SUITE_TIMEOUT_S}s spent, nothing else was injected", file=out)
            records.append({"row_id": row["id"], "passed": False})
            continue
        with contextlib.redirect_stderr(_Quiet()):
            record = run_row(row)
        records.append(record)
        verdict = "ok" if record["passed"] else "FAIL"
        print(f"{record['row_id']} | {record['fault']} | {record['reported']} "
              f"| {verdict} | {record['evidence']}", file=out)
    elapsed = time.monotonic() - started
    failed = [record["row_id"] for record in records if not record["passed"]]
    print(f"--- {len(records) - len(failed)}/{len(records)} rows held in "
          f"{elapsed:.1f}s (budget {SUITE_TIMEOUT_S}s) ---", file=out)
    if failed:
        print("FAILED: " + ", ".join(failed, ), file=out)
    return not failed


class _Quiet:
    """Swallow a consumer's stderr noise.

    The health checker prints; a report interleaved with a probe's own chatter stops
    being one line per row. Failures still surface through the row's evidence.
    """

    def write(self, text):
        return len(text)

    def flush(self):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the #644 degradation matrix.")
    parser.add_argument("--check", action="store_true", help="run every row")
    parser.add_argument("--row", action="append", default=[], help="run one row by id")
    parser.add_argument("--list", action="store_true", help="list row ids and exit")
    parser.add_argument("--json", action="store_true", help="print machine-readable records")
    args = parser.parse_args(argv)
    if args.list:
        for row in load_matrix():
            print(f"{row['id']}\t{row['dependency']}\t{row['fault']}")
        return 0
    if not args.check and not args.row:
        parser.error("nothing to do: pass --check, --row ID, or --list")
    ok = run_matrix(only=args.row or None)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
