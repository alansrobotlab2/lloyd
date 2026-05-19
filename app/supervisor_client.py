"""Client for talking to supervisord over its Unix socket.

Provides the XML-RPC transport that speaks the abstract HTTP-over-AF_UNIX
protocol supervisord uses, plus a handful of read-side helpers that
services routes use to project process state into the UI.
"""

import xmlrpc.client as _xmlrpc
import http.client as _http
import socket as _socket


_SUPERVISOR_SOCK = "/tmp/agent-supervisor.sock"

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
    """Return {program_name: process_info_dict} from supervisord."""
    try:
        procs = _supervisor_proxy().supervisor.getAllProcessInfo()
        return {p["name"]: p for p in procs}
    except Exception:
        return {}


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
    # Port being open is the strongest signal — trust it over supervisord state
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
        _supervisor_proxy().supervisor.startProcess(name, wait)
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
        _supervisor_proxy().supervisor.stopProcess(name, wait)
        return True, "stopped"
    except _xmlrpc.Fault as fault:
        if fault.faultCode == _FAULT_NOT_RUNNING:
            return True, "already stopped"
        return False, f"fault {fault.faultCode}: {fault.faultString}"
    except Exception as exc:
        return False, f"error: {exc}"
