"""A frozen document corpus for the paired quality comparison.

The paired A/B holds the fact tree and the knowledge graph still — both arms
run with `LLOYD_FACTS_ROOT` and `LLOYD_KG_DB` pointed at the same data, so the
only difference between them is the code. The document half was never held
still. It comes from the qmd daemon on :8181, whose index covers a vault being
rewritten continuously by nightly jobs and session capture, and the two arms
run minutes apart.

Measured 2026-09-06, three runs of identical code and data against the live
daemon: every entity metric moved 0.0000 and `doc_recall_avg` moved **0.0250**,
against its own 3-sigma tolerance of 0.0030. The first real run of the check
then demanded a rollback of a promotion whose entire diff was text inside an
inject string. The four document metrics had to be disarmed, which left every
graph-to-document change unfalsifiable.

This closes that. qmd keeps its whole index in one SQLite file and takes a
`--index <name>` flag naming it, so a frozen corpus is a `VACUUM INTO` copy
served on a second port:

    snapshot   1.5s for 1.0 GB   (measured; VACUUM INTO, source read-only)
    startup    ~4s
    results    identical to live at snapshot time

The snapshot is retaken per check rather than kept. The property that matters
is that the two arms see the SAME corpus, not that the corpus never changes —
a stale pin would answer today's question with last week's documents.

`VACUUM INTO` rather than `cp`: the live daemon is mid-write on every tick and
holds a WAL, so a byte copy can tear. VACUUM INTO reads through a read-only
connection and writes a consistent standalone database.

**What this costs, measured 2026-09-07.**

    one full eval run    82s wall, 128 seconds of qmd CPU
    one qmd request      305ms warm, 734ms cold, ~1 CPU-second
    first request idle   3.7s, while the embedding model loads onto the GPU

qmd *does* use the GPU: 3.6 GB resident on GPU 0, utilisation spiking to 93%
during a vector leg. An earlier note here said it was CPU-only. That was
wrong, and wrong for an embarrassing reason — the check was
`nvidia-smi --query-compute-apps ... | head`, and qmd is the fourteenth of
nineteen entries, so it was cut off by the pipe. The conclusion was an
artifact of truncated evidence.

The cost is not the embedding, it is the **fan-out**. `_vault_recall` queries
each of the twelve `VAULT_SEGMENTS` separately, four at a time, so one eval
question becomes twelve qmd requests and a twenty-question run becomes 240.

**And what it costs now, measured 2026-09-18.** The fan-out is gone — #504
sends one request naming every segment — and the cost moved into the
cross-encoder, which now scores a 240-row pool per recall instead of 40:
4.6-5.5 s a recall on production's daemon settings, one at a time, so a
twenty-question arm is about two minutes. See `QMD_PROGRAM_CONF` below for
what the same recall cost on the settings this module used to restate.

Affordable once per promotion, which is what the dedup enforces. Not
affordable in a loop: re-measuring the noise floor repeatedly during
development is what loaded the live daemon and slowed real retrieval for
everything else on the box. Prefer a single `--label` run against the live
daemon when you only need a number; reach for the pin when two arms have to be
comparable.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# The fallback for a daemon conf that cannot be read: the fork, which is the
# only qmd installed since 2026-09-19. It named the published bun-global package
# until then -- a retriever 57x slower on the fan-out and deaf to `rerank`, so a
# pin that fell back compared two arms of something production never runs.
QMD_CLI = Path.home() / "lloyd/qmd/dist/cli/qmd.js"
QMD_CACHE = Path.home() / ".cache" / "qmd"
LIVE_INDEX = QMD_CACHE / "index.sqlite"
# The pin serves production's retriever or it measures somebody else's. Which
# build answers on :8181, and under which settings, is decided by this file and
# nothing else; the pin used to restate it (the published CLI above, three CUDA
# variables) and the restatement stopped being true the day it was written:
# the daemon moved to the fork in `~/lloyd/qmd` on 2026-09-07 with
# `QMD_RERANK_PARALLELISM=4` and a 1200-char rerank window, and the pin kept
# serving published 2.8.3, whole chunks, a reranker pool sized from whatever
# VRAM was free. Nobody could see it while a recall reranked 40 rows. #504 made
# it 240 (2026-09-14). Measured 2026-09-18 on one snapshot, a fresh question per
# sample, one request at a time, against a retriever that gives up at 15 s:
#
#     published build, the pin's old environment    16.3-20.1 s a recall
#     the fork, the pin's old environment           14.3-15.7 s
#     the fork, production's environment             4.6-5.5 s   (4.4 GB on GPU 0)
#
# So it is the settings, not the build, and every recall of every arm timed out.
QMD_PROGRAM_CONF = (Path(__file__).resolve().parents[2] / "agent-services" / "supervisor"
                    / "conf.d" / "agent-qmd-daemon.conf")
QMD_PROGRAM_SECTION = "program:agent-qmd-daemon"

PIN_INDEX_NAME = os.environ.get("LLOYD_EVALPIN_INDEX", "evalpin")
PIN_PORT = int(os.environ.get("LLOYD_EVALPIN_PORT", "8182"))
STARTUP_TIMEOUT = 120.0
# Warm-up: production's recall, answered fast, before anything is timed against
# it. A fresh daemon's first recall also loads the embedding and rerank models
# (7.4 s measured, then 4.6-5.5 s); the retriever's own client gives up at 15 s
# and qmd keeps working on what was abandoned, so a recall slower than that
# leaves every later one queued behind it. On 2026-09-18 that cascade ran a
# whole arm at 116-160 s a query against a 15 s timeout: twenty questions,
# twenty empty answers. The target is two thirds of that timeout.
WARM_TIMEOUT = 180.0
WARM_TARGET_S = 10.0
WARM_TRIES = 4
# One per try, and none of them an eval question: a warm-up that asked the
# eval's own questions would leave their rerank scores cached for both arms.
WARM_QUESTIONS = (
    "what did the household decide about the garden irrigation schedule",
    "notes on soldering the robot arm wrist connector",
    "how is the weekly grocery budget tracked",
    "which books were recommended for learning control theory",
)


class PinError(RuntimeError):
    """The pinned corpus could not be prepared or served."""


def pin_index_path(name: str = PIN_INDEX_NAME) -> Path:
    return QMD_CACHE / f"{name}.sqlite"


def _program_environment(raw: str) -> dict[str, str]:
    """supervisord's `environment=KEY="value",KEY2="value2"`, as a dict."""
    import shlex
    lex = shlex.shlex(raw, posix=True)
    lex.whitespace = ","
    lex.whitespace_split = True
    out: dict[str, str] = {}
    for token in lex:
        key, sep, value = token.strip().partition("=")
        if sep and key.strip():
            out[key.strip()] = value
    return out


def production_daemon(conf: Path | None = None) -> tuple[list[str], dict[str, str], str]:
    """`(argv, environment, source)` of the qmd daemon production runs.

    Read from the daemon's own supervisord program, so the pin follows a swap
    of the build or a changed knob without anyone remembering it exists. The
    fallback is the published CLI and the three CUDA variables the pin always
    set, and `source` says so: a pin that fell back is a different retriever
    from production's, and the production-shaped `warm_up` decides whether it
    is even fast enough to be timed against.
    """
    import configparser
    import shlex
    conf = Path(conf) if conf is not None else QMD_PROGRAM_CONF
    fallback = (["/usr/bin/node", str(QMD_CLI), "mcp", "--http"], {})
    try:
        cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(" ;",),
                                       strict=False)
        if not cp.read(conf, encoding="utf-8") or not cp.has_section(QMD_PROGRAM_SECTION):
            return (*fallback, f"fallback: no [{QMD_PROGRAM_SECTION}] in {conf}")
        argv = shlex.split(cp.get(QMD_PROGRAM_SECTION, "command", fallback=""))
        cli = next((a for a in argv if a.endswith("qmd.js")), "")
        if not cli:
            return (*fallback, f"fallback: {conf.name} names no qmd.js")
        if not Path(cli).exists():
            return (*fallback, f"fallback: {cli} does not exist")
        env = _program_environment(cp.get(QMD_PROGRAM_SECTION, "environment", fallback=""))
        return argv, env, str(conf)
    except Exception as exc:  # noqa: BLE001 — a conf nobody can read is a fallback, not a crash
        return (*fallback, f"fallback: {conf} unreadable ({exc!r})")


def pin_command(argv: list[str], port: int, name: str) -> list[str]:
    """Production's command line, on the pin's port and serving the pin's index."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ("--port", "--index"):
            skip = True
            continue
        if arg.startswith(("--port=", "--index=")):
            continue
        out.append(arg)
    return [*out, "--port", str(port), "--index", name]


def production_payload(text: str) -> dict:
    """The request `_vault_recall`'s document leg sends for `text`: its pool and
    its collections, read from the retriever rather than restated. The pool is
    what a recall costs — the cross-encoder scores every row of it — and it has
    moved once already (40 to 240, #504) without the pin's probe following."""
    from agent_mcp import vault as V
    # `recall_doc_leg_shape` where the tree has it (#1336: the djev ranker asks
    # for a fused head plus floors with the cross-encoder off); the baseline arm
    # may be an older tree, which only knows a pool.
    if hasattr(V, "recall_doc_leg_shape"):
        shape = V.recall_doc_leg_shape()
        payload = {"searches": [{"type": "lex", "query": text}, {"type": "vec", "query": text}],
                   "limit": int(shape["limit"]), "candidateLimit": int(shape["candidateLimit"]),
                   "collections": list(V.VAULT_SEGMENTS), "rerank": bool(shape["rerank"])}
        if shape.get("lexMode") and shape["lexMode"] != "and":
            payload["lexMode"] = shape["lexMode"]
        if getattr(V, "RECALL_QMD_FUSION", "collection") == "global":
            payload["fusion"] = "global"
            floor = {c: n for c, n in shape["floor"].items() if c in payload["collections"]}
            if floor:
                payload["collectionFloor"] = floor
        return payload
    # `recall_doc_pool` where the tree has it (global fusion, 2026-09-19); an
    # older tree's pool is the constant.
    pool = int(V.recall_doc_pool()) if hasattr(V, "recall_doc_pool") else int(V.RECALL_DOC_POOL)
    payload = {"searches": [{"type": "lex", "query": text}, {"type": "vec", "query": text}],
               "limit": pool, "candidateLimit": pool,
               "collections": list(V.VAULT_SEGMENTS), "rerank": True}
    if getattr(V, "RECALL_QMD_FUSION", "collection") == "global":
        payload["fusion"] = "global"
        floor = {c: n for c, n in getattr(V, "RECALL_COLLECTION_FLOOR", {}).items() if c in payload["collections"]}
        if floor:
            payload["collectionFloor"] = floor
    return payload


# "Nobody is there", as opposed to "somebody is there and busy" (a timeout, a
# full backlog) or "this box has no such loopback".
_NOBODY = {errno.ECONNREFUSED}
_NO_SUCH_LOOPBACK = {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.ENETUNREACH}


def port_free(port: int) -> bool:
    """Nothing is listening on `port`, on EITHER loopback.

    qmd binds `localhost`, which resolves to `::1` first on this box, so a qmd
    daemon listens on `[::1]` and nowhere else — and for as long as the pin has
    existed this probed `127.0.0.1` alone. It could not see a qmd daemon at
    all. "Refusing to compare against something I did not start" has never
    once been recorded; what an orphaned pin on :8182 produced instead was a
    second daemon started on top of it, dead on EADDRINUSE, and four skips
    reading `pinned qmd exited immediately (rc=1); see <a log already deleted>`.
    """
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                rc = s.connect_ex((host, port))
        except OSError:
            continue
        if rc in _NOBODY or rc in _NO_SUCH_LOOPBACK:
            continue
        return False
    return True


def _die_with_parent():
    """A `preexec_fn` that has the kernel SIGTERM the child when its parent dies.

    "Still a child of this process" was the whole of the old guarantee that a
    pin does not outlive its owner, and it guarantees nothing: a child whose
    parent dies is re-parented, not killed. The pin also sits in its own process
    group (so `stop` can take the whole tree), which is exactly what a group
    signal cannot reach — so when a landing restarted the backend under a
    running check, supervisord's group kill took the check and left its daemon.
    On 2026-09-18 one such orphan held :8182 for 5 h 30 min and burned two
    CPU-hours; every check after it either started a second daemon on top of
    it (dead on EADDRINUSE — see `port_free`) or shared GPU 0 with it.

    `PR_SET_PDEATHSIG` is the kernel's version of the promise. libc is resolved
    HERE, before the fork, so the child does no import and takes no lock; the
    ppid re-check closes the race where the parent died before the call landed.
    Returns None where prctl is unavailable, and the pin is then only as safe as
    `reap_stale` makes it.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        prctl = libc.prctl
    except Exception:
        return None
    parent = os.getpid()
    PR_SET_PDEATHSIG = 1

    def _arm():
        prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
        if os.getppid() != parent:
            os._exit(1)
    return _arm


def _pin_processes(port: int, name: str) -> list[tuple[int, int]]:
    """`(pid, ppid)` of every process of this user serving `--index name` on
    `--port port`. Read from /proc: the stdlib has no process table."""
    found: list[tuple[int, int]] = []
    me = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != me:
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            args = [a.decode("utf-8", "replace") for a in argv if a]
            if not any(a.endswith("qmd.js") for a in args):
                continue
            if "--index" not in args or args[args.index("--index") + 1] != name:
                continue
            if "--port" not in args or args[args.index("--port") + 1] != str(port):
                continue
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            found.append((int(entry.name), ppid))
        except (OSError, ValueError, IndexError):
            continue
    return found


def _is_orphan(ppid: int) -> bool:
    """Re-parented to init or to the per-user systemd: nobody owns it any more."""
    if ppid <= 1:
        return True
    try:
        return (Path("/proc") / str(ppid) / "comm").read_text().strip() == "systemd"
    except OSError:
        return True


def reap_stale(port: int = PIN_PORT, name: str = PIN_INDEX_NAME, *, wait: float = 20.0) -> list[int]:
    """Kill ORPHANED pinned daemons on `port`; the pids reaped.

    The pin used to meet a busy port with "a stale pinned daemon? Refusing to
    compare against something I did not start" — right about the comparison,
    and a dead end about the daemon: nothing ever removed it, so one orphan
    switched the regression check off until a person noticed. An orphan is a
    daemon whose owner is gone (`_is_orphan`); a pin that still has a live
    parent belongs to a run in progress and is left alone, as is anything on
    the port that is not a pin at all — that is still refused.
    """
    reaped: list[int] = []
    for pid, ppid in _pin_processes(port, name):
        if not _is_orphan(ppid):
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            deadline = time.time() + (wait if sig == signal.SIGTERM else 5.0)
            while time.time() < deadline and Path(f"/proc/{pid}").exists():
                time.sleep(0.25)
            if not Path(f"/proc/{pid}").exists():
                break
        reaped.append(pid)
    deadline = time.time() + 10.0
    while reaped and not port_free(port) and time.time() < deadline:
        time.sleep(0.25)
    return reaped


def snapshot(name: str = PIN_INDEX_NAME) -> dict:
    """Freeze the live qmd index into a named copy. Returns its provenance.

    Raises rather than degrading: a comparison that silently fell back to the
    live daemon would be exactly the unpinned comparison this exists to
    replace, and it would look like a successful pinned one.
    """
    import sqlite3

    if not LIVE_INDEX.exists():
        raise PinError(f"no live qmd index at {LIVE_INDEX}")
    dest = pin_index_path(name)
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(dest) + suffix).unlink()
        except FileNotFoundError:
            pass

    started = time.time()
    con = sqlite3.connect(f"file:{LIVE_INDEX}?mode=ro", uri=True, timeout=120)
    try:
        con.execute(f"VACUUM INTO '{dest}'")
    finally:
        con.close()
    if not dest.exists():
        raise PinError(f"VACUUM INTO produced no file at {dest}")

    docs = None
    try:
        c = sqlite3.connect(f"file:{dest}?mode=ro", uri=True, timeout=30)
        try:
            names = {r[0] for r in c.execute(
                "select name from sqlite_master where type='table'")}
            for candidate in ("documents", "docs", "files"):
                if candidate in names:
                    docs = c.execute(f"select count(*) from {candidate}").fetchone()[0]
                    break
        finally:
            c.close()
    except Exception:
        docs = None

    return {
        "index": str(dest),
        "bytes": dest.stat().st_size,
        "documents": docs,
        "snapshot_seconds": round(time.time() - started, 2),
        "taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(LIVE_INDEX),
    }


def write_overlay(path: Path, port: int = PIN_PORT) -> Path:
    """A config overlay pointing `services.qmd` at the pinned daemon.

    The same mechanism the gate's canary uses. `app.config.service_url` reads
    `services.qmd`, so both arms of the comparison resolve the pinned port
    without either worktree's tracked config.yaml being touched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    path.write_text(yaml.safe_dump(
        {"services": {"qmd": f"http://localhost:{port}/query"}}, sort_keys=False),
        encoding="utf-8")
    return path


def _probe(port: int, timeout: float = 5.0) -> bool:
    payload = json.dumps({
        "searches": [{"type": "lex", "query": "lloyd"}],
        "limit": 1, "collections": [], "skipRerank": True,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/query", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


class PinnedCorpus:
    """Snapshot the qmd index and serve it, for the life of the `with` block.

    Stopped on exit rather than left running: it holds an embedding model on
    GPU 0 beside the live daemon, and nothing outside a comparison should be
    reading a frozen corpus.
    """

    def __init__(self, workdir: Path, *, name: str = PIN_INDEX_NAME,
                 port: int = PIN_PORT):
        self.workdir = Path(workdir)
        self.name = name
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.overlay: Path | None = None
        self.provenance: dict = {}

    def __enter__(self) -> "PinnedCorpus":
        argv, program_env, source = production_daemon()
        cli = next((a for a in argv if a.endswith("qmd.js")), "")
        if not cli or not Path(cli).exists():
            raise PinError(f"qmd CLI not found at {cli or QMD_CLI} ({source})")
        # Orphans first, found by what they ARE (a process serving this index on
        # this port whose owner is gone) rather than by whether the port
        # answers: the port probe is the check that could not see them. Then
        # the port is probed ONCE — a listener with a short backlog stops
        # accepting after a few un-accepted connects, and a port that "became
        # free" that way is how a second daemon gets started on top of
        # somebody else's.
        self.reaped = reap_stale(self.port, self.name)
        if not port_free(self.port):
            raise PinError(
                f"port {self.port} is already in use — and not by an orphaned pin "
                "I could remove. Refusing to compare against something I did not start.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.provenance = snapshot(self.name)

        # Production's program environment LAST, so what it says wins: the three
        # CUDA variables are the floor the pin always had, for a conf that
        # could not be read.
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "QMD_LLAMA_GPU": "cuda",
            **program_env,
        }
        if self.reaped:
            self.provenance["reaped_orphans"] = list(self.reaped)
        self.provenance["daemon"] = {
            "cli": cli, "source": source,
            "settings": {k: v for k, v in sorted(program_env.items()) if k.startswith("QMD_")},
        }
        log = self.workdir / "qmd-pin.log"
        self._log_fh = open(log, "wb")
        # Deliberately NOT `start_new_session=True`. The promoter spawns that
        # way because it must outlive the process that started it; a pinned
        # corpus is the opposite — it exists only for this comparison, holds an
        # embedding model, and an orphan would keep answering on :8182 under
        # every later run. Its own group so `stop` can take the
        # whole tree — which is also why a group signal to its owner misses it,
        # so the kernel is asked to end it when the owner ends (`_die_with_parent`).
        self.proc = subprocess.Popen(
            pin_command(argv, self.port, self.name),
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, process_group=0, env=env,
            preexec_fn=_die_with_parent(),
        )

        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                # Its last words go INTO the error: the caller removes the work
                # dir on the way out, and "see <log>" named a deleted file on
                # every one of the four skips that said it.
                rc = self.proc.returncode
                raise PinError(f"pinned qmd exited immediately (rc={rc}): {self._abandon(log)}")
            if _probe(self.port):
                # An answer counts only while OUR daemon is alive to have given
                # it. One that lost the bind takes a second to die, and in that
                # second an orphan on the port answers the probe: the check then
                # runs both arms against a daemon it did not start, serving an
                # index it did not snapshot. That race is the other half of
                # 2026-09-18 — the same orphan produced four "exited
                # immediately" skips AND five checks that compared 0.0 with 0.0.
                if self.proc.poll() is None:
                    break
                continue
            time.sleep(1.0)
        else:
            raise PinError(f"pinned qmd never answered on :{self.port}: {self._abandon(log)}")

        self.overlay = write_overlay(self.workdir / "qmd-pin-overlay.yaml", self.port)
        self.provenance["port"] = self.port
        self.provenance["overlay"] = str(self.overlay)
        return self

    def _abandon(self, log: Path) -> str:
        """A start that failed: the daemon's last words, with the daemon stopped
        and the 1 GB snapshot removed. `__exit__` never runs for a `with` whose
        `__enter__` raised, so nothing else would."""
        try:
            self._log_fh.flush()
            said = " ".join(log.read_text(errors="replace").split())[-300:]
        except Exception:  # noqa: BLE001
            said = ""
        self.stop()
        self.discard()
        return said or "it said nothing"

    def env_for(self, base: dict | None = None, *, code_root=None) -> dict:
        """Environment that points a child process at the pinned corpus.

        `code_root` pins the OTHER half of the document corpus. This retriever
        greps the repository it ships in, so without it each arm of a paired
        comparison searches its own source and any commit that adds prose to
        `app/` or `scripts/` moves the document metrics on its own.
        """
        env = dict(base if base is not None else os.environ)
        env["LLOYD_CONFIG_OVERLAY"] = str(self.overlay)
        # Every child that runs against the pinned corpus has the djev shadow
        # recorder muted, and it is set HERE rather than at each caller
        # because this is the one function they all pass through — the two
        # regression arms, the noise runs and the warm-up. The lead shadow
        # seam lives inside `_vault_recall`, so a pinned-corpus arm would
        # otherwise write rows indistinguishable from production traffic into
        # the distribution those seams' `label_mass` floors are derived from.
        env["LLOYD_DJEV_SHADOW"] = "0"
        if code_root is not None:
            env["LLOYD_CODE_ROOT"] = str(code_root)
        return env

    def warm_up(self, *, timeout: float = WARM_TIMEOUT, target_s: float = WARM_TARGET_S,
                tries: int = WARM_TRIES) -> float:
        """Ask the recall production asks until it comes back fast. Seconds the
        last one took. Raises `PinError` if the daemon never gets there: an arm
        timed against a daemon that cannot answer scores zero on every question,
        in BOTH arms, and zero against zero reads as "no regression" (five
        checks in a row did, 2026-09-18).

        Two things make it a measurement rather than a greeting. The request is
        `production_payload` — the first cut asked for 30 rows, came back in
        3.4 s, and waved through a daemon that then took 18 s for each of the
        240-row recalls the eval sends. And every try asks a DIFFERENT question:
        qmd caches a rerank score per (query, chunk) in the index it serves, so
        the same question asked twice is answered from that cache in ~0.2 s and
        the slowest daemon passes on its second try."""
        took = float("inf")
        samples: list[float] = []
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for n in range(max(1, tries)):
            try:
                payload = json.dumps(production_payload(WARM_QUESTIONS[n % len(WARM_QUESTIONS)])).encode()
            except Exception as exc:  # noqa: BLE001
                raise PinError(f"cannot build production's recall request: {exc!r}") from exc
            req = urllib.request.Request(f"http://localhost:{self.port}/query", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            started = time.time()
            try:
                with opener.open(req, timeout=timeout) as resp:
                    json.loads(resp.read())
            except Exception as exc:  # noqa: BLE001 — every failure is the same finding
                raise PinError(f"pinned qmd did not answer its warm-up query: {exc!r}") from exc
            took = time.time() - started
            samples.append(round(took, 2))
            self.provenance["warm_samples"] = samples
            if took <= target_s:
                self.provenance["warm_seconds"] = round(took, 2)
                return took
        raise PinError(f"pinned qmd is too slow to time anything against: production's recall "
                       f"still took {took:.1f}s after {tries} tries (target {target_s:.0f}s, "
                       f"samples {samples}; daemon {self.provenance.get('daemon', {}).get('source')})")

    def stop(self) -> None:
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.proc = None
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._log_fh = None

    def discard(self) -> None:
        """Remove the snapshot. ~1 GB, and the next check takes a fresh one."""
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(pin_index_path(self.name)) + suffix).unlink()
            except FileNotFoundError:
                pass

    def __exit__(self, *exc) -> None:
        # Both, always. `discard` used to be the caller's job on the happy
        # path, so an interrupted run left a 1 GB snapshot behind — which is
        # exactly what happened on 2026-09-07.
        self.stop()
        self.discard()


def main(argv=None) -> int:
    """Operator entry point: prove the pin works, then tear it down."""
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Snapshot and serve a frozen qmd corpus")
    ap.add_argument("--keep", action="store_true", help="leave the snapshot on disk")
    ap.add_argument("--port", type=int, default=PIN_PORT)
    args = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="evalpin-"))
    try:
        with PinnedCorpus(work, port=args.port) as pin:
            print(json.dumps(pin.provenance, indent=2))
            print(f"serving on :{pin.port}; overlay {pin.overlay}")
            if not args.keep:
                pin.discard()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
