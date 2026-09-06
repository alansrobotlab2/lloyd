"""Boot a candidate build in isolation and prove it actually works.

This is the rung the whole gate exists for. `pytest` imports modules inside
its own process and never starts `server.py` or `agent_mcp.main`, so it stays
green through a broken startup event, a port collision, a bad supervisor
environment block, or an aggregator that fails to register its tools —
precisely the failures that put a service in FATAL.

The canary runs under a throwaway supervisord (own socket, own pidfile, own
group) rather than bare subprocesses, so the rollback drill can exercise the
same control plane the guardian uses in production.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.selfmod import canary_config as cc

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
SUPERVISORD_BIN = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisord"
SUPERVISORCTL_BIN = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisorctl"

# Modules whose tool count depends on an external application being open.
# Excluded from the floor so the gate does not depend on whether the user
# happens to have Thunderbird running.
EXTERNAL_MODULES = {"thunderbird"}
MIN_INTERNAL_TOOLS = 80
REQUIRED_TOOLS = {"Bash", "Read", "Write", "Edit", "Grep", "Glob"}


class CanaryError(RuntimeError):
    pass


def _get(url: str, timeout: float = 5.0) -> tuple[int | None, dict | None, str | None]:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url), timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace")), None
        except Exception:
            return e.code, None, None
    except Exception as e:
        return None, None, str(e)[:200]


def port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Canary:
    def __init__(self, round_dir: Path, worktree: Path, *,
                 python: Path | None = None,
                 backend_port: int = cc.BACKEND_PORT,
                 mcp_port: int = cc.MCP_PORT,
                 autorestart: bool = True):
        self.round_dir = Path(round_dir)
        self.worktree = Path(worktree)
        self.backend_port = backend_port
        self.mcp_port = mcp_port
        self.autorestart = autorestart
        self.python = Path(python) if python else LIVE_ROOT / ".venvs/lloyd/bin/python"
        self.proc: subprocess.Popen | None = None
        self.sock: Path | None = None
        self.conf: Path | None = None
        self.env: dict = {}
        self.backend_health = f"http://127.0.0.1:{backend_port}/health"
        self.mcp_health = f"http://127.0.0.1:{mcp_port}/health"

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self, timeout: float = 120.0) -> None:
        for port in (self.backend_port, self.mcp_port):
            if not port_free(port):
                raise CanaryError(f"port {port} is already in use — stale canary?")

        cc.materialize_home(self.round_dir, self.worktree, live_root=LIVE_ROOT)
        overlay = cc.write_overlay(self.round_dir, backend_port=self.backend_port,
                                   mcp_port=self.mcp_port, live_root=LIVE_ROOT)
        self.env = cc.canary_env(self.round_dir, self.worktree, overlay=overlay,
                                 python=self.python, mcp_port=self.mcp_port)
        self.conf, self.sock = cc.write_supervisord_conf(
            self.round_dir, self.worktree, self.env, self.python,
            autorestart=self.autorestart)

        if not SUPERVISORD_BIN.exists():
            raise CanaryError(f"supervisord not found at {SUPERVISORD_BIN}")

        self.proc = subprocess.Popen(
            [str(SUPERVISORD_BIN), "-n", "-c", str(self.conf)],
            stdout=open(self.round_dir / "logs" / "supervisord.out", "wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,   # own process group, so teardown can killpg
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.sock.exists():
                return
            if self.proc.poll() is not None:
                raise CanaryError(
                    f"canary supervisord exited immediately (rc={self.proc.returncode}); "
                    f"see {self.round_dir / 'logs' / 'supervisord.out'}")
            time.sleep(0.2)
        raise CanaryError("canary supervisord never created its socket")

    def ctl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(SUPERVISORCTL_BIN), "-c", str(self.conf), *args],
            capture_output=True, text=True, timeout=60, check=False)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.ctl("shutdown")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None

    def __enter__(self) -> "Canary":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── probing ────────────────────────────────────────────────────────
    def probe(self, timeout: float = 120.0) -> dict:
        """Wait for both services and assert what they advertise."""
        report: dict = {"mcp": None, "backend": None, "ok": False, "errors": []}

        ok, body = self._wait(self.mcp_health, timeout)
        report["mcp"] = body
        if not ok:
            report["errors"].append(f"mcp /health never returned 200: {body}")
            return report

        discovery = (body or {}).get("discovery") or {}
        internal = sum(e.get("tools", 0) for name, e in discovery.items()
                       if name not in EXTERNAL_MODULES)
        report["internal_tools"] = internal
        if (body or {}).get("degraded_modules"):
            report["errors"].append(f"degraded modules: {body['degraded_modules']}")
        if internal < MIN_INTERNAL_TOOLS:
            report["errors"].append(
                f"only {internal} internal tools advertised (floor {MIN_INTERNAL_TOOLS})")

        ok, body = self._wait(self.backend_health, timeout)
        report["backend"] = body
        if not ok:
            report["errors"].append(f"backend /health never returned 200: {body}")
            return report

        status, deep, err = _get(f"http://127.0.0.1:{self.backend_port}/health/deep", 15.0)
        report["deep"] = deep
        if deep:
            mcp_url = (deep.get("mcp") or {}).get("url", "")
            # THE assertion that proves config-follows-code held for this
            # candidate: a round that breaks service_url() would leave the
            # canary backend pointing at the LIVE aggregator, and everything
            # else would still look green.
            if f":{self.mcp_port}" not in mcp_url:
                report["errors"].append(
                    f"canary backend resolved MCP to {mcp_url!r}, not its own port "
                    f"{self.mcp_port} — config is not following the code dir")
            if (deep.get("mcp") or {}).get("status") != "ok":
                report["errors"].append(f"deep health MCP status: {deep.get('mcp')}")

        status, _, err = _get(f"http://127.0.0.1:{self.backend_port}/openapi.json", 20.0)
        if status != 200:
            report["errors"].append(f"/openapi.json returned {status} ({err}) — "
                                    "a router failed to import or a model is unserializable")

        report["ok"] = not report["errors"]
        return report

    def _wait(self, url: str, timeout: float) -> tuple[bool, dict | None]:
        deadline = time.time() + timeout
        last: dict | None = None
        while time.time() < deadline:
            status, body, err = _get(url, 5.0)
            if status == 200:
                return True, body
            last = body if body is not None else {"status_code": status, "error": err}
            time.sleep(1.0)
        return False, last

    def assert_commit(self, expected: str) -> tuple[bool, str]:
        """Confirm the canary is running the code we think it is."""
        _, body, _ = _get(self.backend_health, 5.0)
        actual = (body or {}).get("commit")
        if actual != expected:
            return False, f"canary reports commit {actual}, expected {expected}"
        return True, "commit matches"

    def smoke(self, timeout: float = 150.0) -> dict:
        """Drive one real turn end to end. See canary_smoke.py."""
        from scripts.selfmod import canary_smoke
        return canary_smoke.run(
            backend=f"http://127.0.0.1:{self.backend_port}",
            sessions_dir=self.worktree / "sessions",
            timeout=timeout,
        )

    def cleanup(self) -> None:
        self.stop()
        for port in (self.backend_port, self.mcp_port):
            if not port_free(port):
                time.sleep(2.0)


def remove_round(round_dir: Path) -> None:
    shutil.rmtree(round_dir, ignore_errors=True)
