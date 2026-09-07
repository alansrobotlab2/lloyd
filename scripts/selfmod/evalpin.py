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

**What this costs, measured 2026-09-07.** qmd embeds on the CPU — it appears
on no GPU despite `QMD_LLAMA_GPU=cuda` — and one vector-leg query burns about
**7 seconds of CPU** against 3.7s wall, roughly two cores saturated. A lex-only
query costs 1s. The eval is 20 queries with both legs, so ONE run is around
2.7 CPU-minutes, a paired comparison is two runs, and a pinned comparison also
loads a second embedding model beside the live daemon.

That is affordable once per promotion, which is what the check does. It is not
affordable in a loop: repeatedly re-measuring the noise floor during
development is what pegged the live daemon and slowed real retrieval for
everything else on the box. Prefer `--label` runs against the live daemon when
you only need a number, and reach for the pin when you need two arms to be
comparable.
"""

from __future__ import annotations

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

QMD_CLI = Path.home() / ".bun/install/global/node_modules/@tobilu/qmd/dist/cli/qmd.js"
QMD_CACHE = Path.home() / ".cache" / "qmd"
LIVE_INDEX = QMD_CACHE / "index.sqlite"

PIN_INDEX_NAME = os.environ.get("LLOYD_EVALPIN_INDEX", "evalpin")
PIN_PORT = int(os.environ.get("LLOYD_EVALPIN_PORT", "8182"))
STARTUP_TIMEOUT = 120.0


class PinError(RuntimeError):
    """The pinned corpus could not be prepared or served."""


def pin_index_path(name: str = PIN_INDEX_NAME) -> Path:
    return QMD_CACHE / f"{name}.sqlite"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


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
        if not QMD_CLI.exists():
            raise PinError(f"qmd CLI not found at {QMD_CLI}")
        if not port_free(self.port):
            raise PinError(
                f"port {self.port} is already in use — a stale pinned daemon? "
                "Refusing to compare against something I did not start.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.provenance = snapshot(self.name)

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "QMD_LLAMA_GPU": "cuda",
        }
        log = self.workdir / "qmd-pin.log"
        self._log_fh = open(log, "wb")
        # Deliberately NOT `start_new_session=True`. The promoter spawns that
        # way because it must outlive the process that started it; a pinned
        # corpus is the opposite — it exists only for this comparison, holds an
        # embedding model, and an orphan would keep answering on :8182 while a
        # later run refuses the port. Its own group so `stop` can take the
        # whole tree, but still a child of this process.
        self.proc = subprocess.Popen(
            ["/usr/bin/node", str(QMD_CLI), "mcp", "--http",
             "--port", str(self.port), "--index", self.name],
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, process_group=0, env=env,
        )

        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise PinError(
                    f"pinned qmd exited immediately (rc={self.proc.returncode}); "
                    f"see {log}")
            if _probe(self.port):
                break
            time.sleep(1.0)
        else:
            self.stop()
            raise PinError(f"pinned qmd never answered on :{self.port}; see {log}")

        self.overlay = write_overlay(self.workdir / "qmd-pin-overlay.yaml", self.port)
        self.provenance["port"] = self.port
        self.provenance["overlay"] = str(self.overlay)
        return self

    def env_for(self, base: dict | None = None, *, code_root=None) -> dict:
        """Environment that points a child process at the pinned corpus.

        `code_root` pins the OTHER half of the document corpus. This retriever
        greps the repository it ships in, so without it each arm of a paired
        comparison searches its own source and any commit that adds prose to
        `app/` or `scripts/` moves the document metrics on its own.
        """
        env = dict(base if base is not None else os.environ)
        env["LLOYD_CONFIG_OVERLAY"] = str(self.overlay)
        if code_root is not None:
            env["LLOYD_CODE_ROOT"] = str(code_root)
        return env

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
