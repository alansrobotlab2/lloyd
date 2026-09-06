"""Minimal supervisord XML-RPC client over AF_UNIX. Stdlib only.

This duplicates the ~20-line transport in `app/supervisor_client.py`, and the
duplication is **correct**. The guardian must not share a failure domain with
the code it guards: `app.supervisor_client` has no third-party imports today,
but the moment it grows an `app.config` import, a malformed `config.yaml` that
Lloyd wrote would take down the watchdog along with the backend. A watchdog
that dies from the same cause as its patient is not a watchdog.

Two behaviours differ deliberately from the app-side client:

  * **Names are group-qualified from supervisord's own `group` field.**
    `getAllProcessInfo` returns `name` and `group` separately, so the mapping
    is derived rather than hardcoded and a conf.d regrouping cannot silently
    break it.
  * **An unreachable socket raises.** `app.supervisor_client._supervisor_all`
    used to swallow that into `{}`, making "supervisord is gone" look like
    "nothing is configured". Those have opposite remediations — restart the
    unit versus roll back code — so conflating them would make the guardian
    revert Lloyd's Python because supervisord died.
"""

from __future__ import annotations

import http.client
import socket
import xmlrpc.client


class SupervisordUnreachable(RuntimeError):
    """The supervisord socket could not be reached."""


class _UnixSocketHTTPConn(http.client.HTTPConnection):
    def __init__(self, sock_path: str, timeout: float = 5.0):
        super().__init__("localhost", timeout=timeout)
        self._sock_path = sock_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._sock_path)


class _UnixSocketTransport(xmlrpc.client.Transport):
    def __init__(self, sock_path: str, timeout: float = 5.0):
        super().__init__()
        self._sock_path = sock_path
        self._timeout = timeout

    def make_connection(self, host):
        return _UnixSocketHTTPConn(self._sock_path, self._timeout)


class SupervisorClient:
    """Read/write supervisord access. The unit of interchange for tests."""

    def __init__(self, sock_path: str, timeout: float = 5.0):
        self.sock_path = sock_path
        self.timeout = timeout

    def _proxy(self):
        return xmlrpc.client.ServerProxy(
            "http://localhost/RPC2",
            transport=_UnixSocketTransport(self.sock_path, self.timeout),
        )

    def get_state(self) -> dict:
        try:
            return self._proxy().supervisor.getState()
        except Exception as exc:
            raise SupervisordUnreachable(str(exc)) from exc

    def all_process_info(self) -> dict[str, dict]:
        """{qualified_name: info}. Raises SupervisordUnreachable if the socket is down.

        An answering supervisord with no programs returns {} — which is a
        different fact, and the caller must be able to tell them apart.
        """
        try:
            procs = self._proxy().supervisor.getAllProcessInfo()
        except Exception as exc:
            raise SupervisordUnreachable(str(exc)) from exc
        out: dict[str, dict] = {}
        for p in procs:
            name = p.get("name")
            group = p.get("group") or name
            if not name:
                continue
            qualified = name if group == name else f"{group}:{name}"
            info = dict(p)
            info["qualified"] = qualified
            out[qualified] = info
            out.setdefault(name, info)  # convenience alias for lookups
        return out

    def start(self, name: str, wait: bool = False) -> tuple[bool, str]:
        try:
            self._proxy().supervisor.startProcess(name, wait)
            return True, "started"
        except xmlrpc.client.Fault as f:
            if f.faultCode == 60:  # ALREADY_STARTED
                return True, "already running"
            return False, f"fault {f.faultCode}: {f.faultString}"
        except Exception as exc:
            return False, f"error: {exc}"

    def stop(self, name: str, wait: bool = True) -> tuple[bool, str]:
        try:
            self._proxy().supervisor.stopProcess(name, wait)
            return True, "stopped"
        except xmlrpc.client.Fault as f:
            if f.faultCode == 70:  # NOT_RUNNING
                return True, "already stopped"
            return False, f"fault {f.faultCode}: {f.faultString}"
        except Exception as exc:
            return False, f"error: {exc}"
