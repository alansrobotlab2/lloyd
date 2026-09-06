"""Client for talking to supervisord over its Unix socket.

Provides the XML-RPC transport that speaks the abstract HTTP-over-AF_UNIX
protocol supervisord uses, plus a handful of read-side helpers that
services routes use to project process state into the UI.

Two things here are load-bearing for the self-modification loop and are
easy to get wrong:

  * **Names must be group-qualified.** `lloyd-backend`, `lloyd-frontend` and
    `lloyd-mcp` live in supervisord group `lloyd-mc` (see
    `agent-services/supervisor/conf.d/lloyd-mc.conf`). supervisord's XML-RPC
    name resolution requires `lloyd-mc:lloyd-backend` for those; passing the
    bare name returns `Fault 10 BAD_NAME`. Before 2026-09-06 every write call
    here used bare names, so `POST /api/services/action` had been returning
    HTTP 500 for all three Lloyd services. `qualify()` is the fix, and it
    derives the group from supervisord's own `group` field rather than
    hardcoding it, so a conf.d regrouping cannot silently break it again.

  * **An unreachable supervisord is not an empty supervisord.** `_supervisor_all`
    used to swallow every exception and return `{}`, which made "the socket is
    gone" indistinguishable from "no processes are configured". Those have
    opposite remediations, so it now raises `SupervisordUnreachable` and
    callers that genuinely want the lenient behavior catch it.
"""

import os
import time
import xmlrpc.client as _xmlrpc
import http.client as _http
import socket as _socket


# Overridable so a canary can be pointed at a throwaway supervisord (or at a
# path that does not exist, which is how `_sync_secondary_llm_state` is kept
# from reaching the live socket during a gate run).
_SUPERVISOR_SOCK = os.environ.get("LLOYD_SUPERVISOR_SOCK", "/tmp/agent-supervisor.sock")

# service_id → (display_name, port_or_None)
_INFRA_SERVICES = {
    "agent-llm-primary":    ("LLM Primary",     8096),
    "agent-llm-secondary":  ("LLM Secondary",   8091),
    "agent-qmd-daemon":     ("QMD Daemon",      8181),
    "agent-qmd-watcher":    ("QMD Watcher",     None),
    "agent-tts":            ("TTS",             None),
    "agent-livekit-server": ("LiveKit Server",  7880),
    "lloyd-agent-worker":   ("Voice Worker",    None),
}

_LLOYD_SERVICES = {
    "lloyd-backend":    ("Lloyd Backend",  8080),
    "lloyd-frontend":   ("Lloyd Frontend", 5173),
    "lloyd-mcp":        ("Lloyd MCP",      8500),
}

# Fallback used only when supervisord cannot be reached to ask. Kept in sync
# with conf.d/lloyd-mc.conf by `tests/test_supervisor_client.py`.
_FALLBACK_GROUPS = {
    "lloyd-backend":  "lloyd-mc",
    "lloyd-frontend": "lloyd-mc",
    "lloyd-mcp":      "lloyd-mc",
}


class SupervisordUnreachable(RuntimeError):
    """The supervisord socket could not be reached or spoke gibberish.

    Distinct from "supervisord answered and has no processes" — that returns
    an empty dict. A watchdog must tell these apart: the first is remediated
    by restarting supervisord, the second by starting programs.
    """


class _UnixSocketHTTPConn(_http.HTTPConnection):
    def __init__(self, sock_path: str):
        super().__init__("localhost")
        self._sock_path = sock_path
    def connect(self):
        self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self.sock.connect(self._sock_path)


class _UnixSocketTransport(_xmlrpc.Transport):
    def __init__(self, sock_path: str):
        super().__init__()
        self._sock_path = sock_path
    def make_connection(self, host):
        return _UnixSocketHTTPConn(self._sock_path)


def _supervisor_proxy() -> _xmlrpc.ServerProxy:
    """Return an XML-RPC proxy talking to supervisord over its Unix socket."""
    return _xmlrpc.ServerProxy("http://localhost/RPC2", transport=_UnixSocketTransport(_SUPERVISOR_SOCK))


def _supervisor_all() -> dict:
    """Return {program_name: process_info_dict} from supervisord.

    Raises `SupervisordUnreachable` if the socket cannot be reached. An
    answering supervisord with no programs returns `{}`.
    """
    try:
        procs = _supervisor_proxy().supervisor.getAllProcessInfo()
    except Exception as exc:
        raise SupervisordUnreachable(str(exc)) from exc
    return {p["name"]: p for p in procs}


def _supervisor_all_lenient() -> dict:
    """`_supervisor_all` with the pre-2026-09 behavior: {} on any failure.

    For UI routes, where "cannot reach supervisord" and "nothing configured"
    render the same way anyway.
    """
    try:
        return _supervisor_all()
    except SupervisordUnreachable:
        return {}


def _group_map() -> dict[str, str]:
    """{program_name: group_name} as supervisord itself reports it."""
    try:
        procs = _supervisor_proxy().supervisor.getAllProcessInfo()
    except Exception:
        return dict(_FALLBACK_GROUPS)
    out = {}
    for p in procs:
        name, group = p.get("name"), p.get("group")
        if name and group:
            out[name] = group
    return out or dict(_FALLBACK_GROUPS)


def qualify(name: str) -> str:
    """Return the group-qualified supervisord name for `name`.

    `lloyd-backend` → `lloyd-mc:lloyd-backend`; `agent-llm-secondary`
    (ungrouped, i.e. its group equals its name) is returned unchanged. An
    already-qualified name passes straight through.
    """
    if not name or ":" in name:
        return name
    group = _group_map().get(name) or _FALLBACK_GROUPS.get(name)
    if not group or group == name:
        return name
    return f"{group}:{name}"


def process_info(name: str) -> dict:
    """getProcessInfo for `name`, qualifying it first. Raises on failure."""
    try:
        return _supervisor_proxy().supervisor.getProcessInfo(qualify(name))
    except _xmlrpc.Fault:
        raise
    except Exception as exc:
        raise SupervisordUnreachable(str(exc)) from exc


def _port_open(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        try:
            with _socket.create_connection((host, port), timeout=0.3):
                return True
        except Exception:
            pass
    return False


def _sup_state(proc: dict | None) -> tuple[str, str]:
    """Return (activeState, subState) from a supervisord process dict."""
    if not proc:
        return "unknown", "unknown"
    state = proc.get("statename", "UNKNOWN").lower()
    if state == "running":
        return "active", "running"
    if state in ("stopped", "exited"):
        return "inactive", state
    if state == "fatal":
        return "failed", "fatal"
    return "unknown", state


def _health(active: str, port_healthy: bool | None) -> str:
    """Project (supervisord state, port probe) into a UI health string.

    NOTE: port-open deliberately wins over supervisord state here, because for
    the Services page a process whose port answers is useful regardless of what
    supervisord thinks. That inversion makes this function **unsuitable for a
    watchdog** — a FATAL backend whose port is still held by a zombie reads
    "healthy". `agent-services/guardian/detect.py` implements its own predicate
    and asserts the divergence in `tests/test_guardian_predicates.py`.
    """
    if port_healthy is True:
        return "healthy"
    if active != "active":
        return "stopped"
    if port_healthy is False:
        return "degraded"
    return "healthy"


def _read_log_tail(log_path: str, lines: int = 50) -> list[str]:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-lines:]
    except Exception:
        return []


# supervisord fault codes we want to treat as no-ops (idempotent calls).
# 60 ALREADY_STARTED, 70 NOT_RUNNING, 80 SUCCESS
_FAULT_ALREADY_STARTED = 60
_FAULT_NOT_RUNNING = 70


def start_process(name: str, wait: bool = False) -> tuple[bool, str]:
    """Start a supervisord-managed process by name. Idempotent.

    Returns (ok, message). ok=True if the process is running (or was already).
    """
    try:
        _supervisor_proxy().supervisor.startProcess(qualify(name), wait)
        return True, "started"
    except _xmlrpc.Fault as fault:
        if fault.faultCode == _FAULT_ALREADY_STARTED:
            return True, "already running"
        return False, f"fault {fault.faultCode}: {fault.faultString}"
    except Exception as exc:
        return False, f"error: {exc}"


def stop_process(name: str, wait: bool = False) -> tuple[bool, str]:
    """Stop a supervisord-managed process by name. Idempotent.

    Returns (ok, message). ok=True if the process is stopped (or was already).
    """
    try:
        _supervisor_proxy().supervisor.stopProcess(qualify(name), wait)
        return True, "stopped"
    except _xmlrpc.Fault as fault:
        if fault.faultCode == _FAULT_NOT_RUNNING:
            return True, "already stopped"
        return False, f"fault {fault.faultCode}: {fault.faultString}"
    except Exception as exc:
        return False, f"error: {exc}"


_STOPPED_STATES = {"STOPPED", "EXITED", "FATAL"}


def restart_process(
    name: str,
    *,
    stop_timeout: float = 25.0,
    start_timeout: float = 90.0,
    poll_interval: float = 0.5,
) -> tuple[bool, str]:
    """Stop then start a process, polling rather than blocking on supervisord.

    `wait=False` on both legs is deliberate. `lloyd-backend` may carry a large
    `startsecs`, and a blocking `startProcess` would leave the caller (and any
    watchdog sharing its thread) blind for that whole window. We poll
    `getProcessInfo` instead, which is also how `agent-llm-primary.conf`
    advises handling its 300s `startsecs`.

    If the process has not reached a stopped state by `stop_timeout` we SIGKILL
    its pid directly — `lloyd-backend.conf` sets `stopwaitsecs=15`, so a hung
    shutdown is a real possibility and leaving it half-stopped would wedge the
    restart. Returns (ok, message).
    """
    qualified = qualify(name)

    stop_process(name, wait=False)
    deadline = time.monotonic() + stop_timeout
    while time.monotonic() < deadline:
        try:
            info = process_info(name)
        except Exception:
            break
        if info.get("statename", "").upper() in _STOPPED_STATES:
            break
        time.sleep(poll_interval)
    else:
        # Still not stopped. Kill the pid supervisord reported.
        try:
            info = process_info(name)
            pid = int(info.get("pid") or 0)
            if pid > 0:
                os.kill(pid, 9)
        except Exception:
            pass
        time.sleep(poll_interval)

    ok, msg = start_process(name, wait=False)
    if not ok:
        return False, f"start failed: {msg}"

    deadline = time.monotonic() + start_timeout
    last = "?"
    while time.monotonic() < deadline:
        try:
            info = process_info(name)
        except Exception as exc:
            return False, f"cannot read state after start: {exc}"
        last = info.get("statename", "?").upper()
        if last == "RUNNING":
            return True, f"restarted {qualified}"
        if last == "FATAL":
            return False, f"{qualified} entered FATAL: {info.get('spawnerr') or ''}".strip()
        time.sleep(poll_interval)
    return False, f"{qualified} did not reach RUNNING within {start_timeout}s (last={last})"
