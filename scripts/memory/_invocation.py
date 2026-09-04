"""Who-ran-this ledger for destructive knowledge-graph tools.

On 2026-09-03 an `entity-resolution-sweep.py --apply` merged 2,034 entity
directories against an empty graph and nothing recorded what invoked it — not
the autonomy run records, not any chat session, not shell history. The apply
report carried the *effects* but not the *actor*. Every tool that mutates the
facts tree now stamps this into its report so the next incident has a suspect.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import os
import sys
from pathlib import Path


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").replace("\0", " ").strip()
    except Exception:
        return ""


def invocation_ledger() -> dict:
    ppid = os.getppid()
    env_hints = {k: v for k, v in os.environ.items()
                 if k.startswith(("LLOYD_", "CLAUDE_", "AUTONOMY_")) or k in ("TERM_PROGRAM", "SUPERVISOR_PROCESS_NAME")}
    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "pid": os.getpid(),
        "ppid": ppid,
        "parent_cmdline": _read(f"/proc/{ppid}/cmdline")[:400],
        "grandparent_cmdline": _read(f"/proc/{_ppid_of(ppid)}/cmdline")[:400] if _ppid_of(ppid) else "",
        "env_hints": env_hints,
    }


def _ppid_of(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except Exception:
        pass
    return None
