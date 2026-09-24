#!/usr/bin/env python3
"""Service Health Check Skill

Bundle multiple service status checks into a single call.
Returns structured status for LLM, MCP, and other Lloyd services.
"""

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SUPervisor_CONF = "/home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf"

SERVICES = {
    # Supervisor services — status via supervisorctl, no HTTP check
    "lloyd-backend": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-backend"], "category": "lloyd"},
    "lloyd-frontend": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-frontend"], "category": "lloyd"},
    "lloyd-mcp": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-mcp"], "category": "lloyd"},

    # LLM inference servers — status via supervisorctl, no HTTP check
    "agent-llm-primary": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-llm-primary"], "category": "supervisor"},
    "agent-llm-secondary": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-llm-secondary"], "category": "supervisor"},
    # djev shares GPU 2 with the secondary, so at most one of these two is ever
    # enabled. Port 8011 is the structured decision API, which start-djev.sh
    # only launches once vLLM on 8010 answers — so an open 8011 means the whole
    # stack is serving, while an open 8010 during a cold boot does not.
    "agent-djev": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-djev"], "category": "supervisor", "port": 8011},

    # QMD retrieval daemon — the path every vault_search/vault_recall needs,
    # and until #406 this check could not see it at all. Port is the one
    # `agent-services/supervisor/conf.d/agent-qmd-daemon.conf` launches it on
    # (`qmd mcp --http --port 8181`); it serves a real /health route, so a
    # supervisor RUNNING line that no longer has a listener behind it — the
    # failure this entry exists to catch — now shows up here instead of in the
    # first failed vault_search of the day. This is also the first SERVICES
    # entry to declare a `port`, i.e. the first thing to ever exercise
    # http_check().
    "agent-qmd-daemon": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-qmd-daemon"], "category": "retrieval", "port": 8181},
}

def _switched_off() -> set:
    """Supervisord programs config.yaml has deliberately switched off.

    The optional LLM slots share GPU 2, so one of them is always stopped on
    purpose, and reporting it as an unhealthy service every time this skill runs
    trains the reader past the one line that would matter. `app/llm_slots.py` is
    the single definition; this is a consumer of it, not a second copy.

    Fails OPEN — an unreadable config reports everything. Under-reporting hides
    a service that really is down, which is the failure this skill exists to
    catch; over-reporting only costs a line.
    """
    try:
        import sys
        sys.path.insert(0, "/home/alansrobotlab/lloyd")
        from app import llm_slots
        return {program for _flag, program, on in llm_slots.slots() if not on}
    except Exception:
        return set()


# The GPU power clamp runs as root, so its unit and script are not symlinked
# from the tree like every other unit: they are deployed by a hand `install`
# to root-owned paths (the unit's own header carries the two commands). Twice
# in one week (`bb0dbca` 09-11, `36ba27e` 09-17) the tracked copy was edited
# and the deployed one re-installed by hand the same minute, and had the second
# step been forgotten nothing would have said so — the next boot would simply
# have clamped the old watts. This is the gate (#1108): each tracked file
# against the copy that actually runs, reported beside the services.
_TREE = Path(__file__).resolve().parent.parent
DEPLOY_CATEGORY = "deploy"
DEPLOYED_COPIES = (
    (_TREE / "agent-services" / "systemd" / "nvidia-power-limit.service",
     Path("/etc/systemd/system/nvidia-power-limit.service")),
    (_TREE / "agent-services" / "bin" / "set-gpu-power-limit.sh",
     Path("/usr/local/sbin/set-gpu-power-limit.sh")),
)

CATEGORIES = {
    "llm": ["agent-llm-primary", "agent-llm-secondary", "agent-djev"],
    "lloyd": ["lloyd-backend", "lloyd-frontend", "lloyd-mcp"],
    "retrieval": ["agent-qmd-daemon"],
    # No supervisor program behind it: `main` answers this category from
    # `check_deployed_copies` instead.
    DEPLOY_CATEGORY: [],
    "all": list(SERVICES.keys()),
}


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check_deployed_copies(pairs=DEPLOYED_COPIES) -> list:
    """Compare each tracked file against its root-owned deployed copy.

    One result per pair, in the shape `check_service` returns, so the two
    formatters and the category summary carry it without a special case. The
    status is `ok` only when both files read and their bytes match, `drift`
    when they differ, and `missing` when either cannot be read — each naming
    the pair. An unreadable deployed copy is never `ok`: a check that read
    nothing verified nothing, and a green line from it would be exactly the
    false comfort the check exists to remove.
    """
    results = []
    for tracked, deployed in pairs:
        tracked, deployed = Path(tracked), Path(deployed)
        want, have = _sha256(tracked), _sha256(deployed)
        if want is None:
            status = f"missing: tracked copy {tracked} is unreadable"
        elif have is None:
            status = (f"missing: {deployed} is unreadable or not installed "
                      f"(tracked copy: {tracked})")
        elif want != have:
            status = (f"drift: {deployed} differs from tracked {tracked} "
                      f"— reinstall it from the tree")
        else:
            status = f"ok: {deployed} matches {tracked}"
        healthy = status.startswith("ok:")
        results.append({
            "name": f"deployed:{deployed.name}",
            "status": status,
            "healthy": healthy,
            "exit_code": 0 if healthy else 1,
            "output": status,
            "category": DEPLOY_CATEGORY,
            "tracked": str(tracked),
            "deployed": str(deployed),
        })
    return results


def _probe_targets(host: str, port: int) -> list:
    """Addresses worth trying for `host:port`, in the order curl would try them.

    `getaddrinfo` on this box returns `::1` *first* for "localhost", which is why
    `curl localhost:8181/health` succeeds against qmd while an IPv4-literal
    probe of the same port is refused. Loopback literals are appended for any
    family resolution did not yield, so a resolver hiccup cannot by itself
    produce an "unhealthy" verdict about a service.
    """
    targets = []
    try:
        for info in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            targets.append((info[0], info[4]))
    except Exception:
        pass
    resolved = {t[1][0] for t in targets}
    for family, literal in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        if literal not in resolved:
            addr = (literal, port, 0, 0) if family == socket.AF_INET6 else (literal, port)
            targets.append((family, addr))
    return targets


def http_check(port: int, host: str = "localhost") -> tuple:
    """Quick TCP connect check against a port. Returns (connected, error).

    Tries every address `host` resolves to rather than assuming the port is on
    IPv4 loopback. qmd is the reason: it binds `[::1]:8181` and nothing else, so
    a `127.0.0.1` probe answers ECONNREFUSED (111) from a daemon that is up and
    serving `/health` — a healthy service reported dead, which is a worse
    failure than not probing at all.
    """
    last_err = "no addresses to try"
    for family, addr in _probe_targets(host, port):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.settimeout(2)
            result = s.connect_ex(addr)
            s.close()
        except Exception as e:
            last_err = str(e)
            continue
        if result == 0:
            return (True, 0)
        last_err = result
    return (False, last_err)


# supervisor prints its status line as `name  STATENAME  description`, so the
# state is field 2 and only field 2 may be read as health. Reading anything
# else mis-verdicts, for two reasons measured on this box with the installed
# supervisor 4.3.0:
#
#   * `supervisorctl status` exits 0 for BACKOFF. `do_status` moves the exit
#     status off SUCCESS only for `states.STOPPED_STATES`
#     (supervisorctl.py:696-698), and states.py:14-22 files BACKOFF under
#     RUNNING_STATES — so a process that is spawned, dies inside `startsecs`
#     and is being retried used to satisfy the old
#     `healthy = result.returncode == 0` fallback and printed `[✓] healthy`.
#   * field 3 is free text, so `agent-tts  FATAL  can't spawn process: RUNNING
#     helper not found` satisfied the old `"RUNNING" in output` test and was
#     reported as `healthy=True, status="RUNNING"`.
#
# The exit code is deliberately not consulted: acting on it is #1029's
# contract, not this script's.
def _supervisor_verdict(output: str) -> tuple:
    """Decide health from supervisorctl's own state field.

    Returns `(status, healthy)`. Healthy only when every printed status line
    names exactly `RUNNING`: a group namespec (or a bare `supervisorctl
    status`) prints one line per process, so one BACKOFF process inside an
    otherwise RUNNING answer is not a healthy service. A line with no state
    field, and an empty answer, are unparsable and therefore unhealthy — the
    raw output is returned as `status` so the printed verdict still carries
    what supervisor actually said.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return ("unknown (empty supervisorctl status output)", False)
    for line in lines:
        fields = line.split()
        if len(fields) < 2 or fields[1] != "RUNNING":
            return (output, False)
    return ("RUNNING", True)


def check_service(name: str, service_def: dict) -> dict:
    """Check a single service status."""
    command = service_def["command"]
    category = service_def["category"]
    port = service_def.get("port")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout.strip() or result.stderr.strip()
        status = output or "unknown"
        healthy = False

        # Special handling for supervisorctl: the state field decides, not the
        # exit code and not a substring of the answer.
        if "supervisorctl" in command[0]:
            status, healthy = _supervisor_verdict(output)

        # If supervisor says running, also check port is reachable
        extra = ""
        if healthy and port:
            connected, err = http_check(port)
            if connected:
                extra = f" (port {port} OK)"
            else:
                extra = f" (port {port} FAIL: {err})"
                healthy = False
                status += extra

        return {
            "name": name,
            "status": status,
            "healthy": healthy,
            "exit_code": result.returncode,
            "output": output,
            "category": category
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "timeout",
            "healthy": False,
            "exit_code": -1,
            "output": "timeout",
            "category": category
        }
    except Exception as e:
        return {
            "name": name,
            "status": "error",
            "healthy": False,
            "exit_code": -1,
            "output": str(e),
            "category": category
        }


def format_text(results: list, summary: dict) -> str:
    """Format results as human-readable text."""
    lines = []
    healthy_count = sum(1 for r in results if r["healthy"])
    total = len(results)

    lines.append("=== Service Health Check ===")
    lines.append(f"Time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Overall: {healthy_count}/{total} services healthy")
    lines.append("")

    for result in results:
        icon = "[✓]" if result["healthy"] else "[✗]"
        lines.append(f"{icon} {result['name']}\u2014 {result['status']}")

    lines.append("")
    lines.append("Categories:")
    for cat, status in summary.items():
        lines.append(f"  {cat.capitalize()}: {status}")

    return "\n".join(lines)


def format_json(results: list, summary: dict) -> str:
    """Format results as JSON."""
    healthy_count = sum(1 for r in results if r["healthy"])
    total = len(results)

    output = {
        "check_time": datetime.now(timezone.utc).isoformat(),
        "total_services": total,
        "healthy": healthy_count,
        "unhealthy": total - healthy_count,
        "services": results,
        "summary": summary
    }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Check status of multiple services")
    parser.add_argument("--services", nargs="+", help="Specific services to check")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), help="Check by category")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()

    # Determine which services to check
    if args.services:
        # An explicit ask is answered whatever the config says: someone naming a
        # slot by hand wants to know about that slot, including that it is off.
        service_names = args.services
    else:
        service_names = CATEGORIES[args.category] if args.category else CATEGORIES["all"]
        off = _switched_off()
        service_names = [n for n in service_names if n not in off]

    # Run checks
    results = []
    for name in service_names:
        if name in SERVICES:
            results.append(check_service(name, SERVICES[name]))
    # The deployed-copy pairs ride with the full check and with their own
    # category; a `--services` ask names supervisor programs and gets only those.
    if not args.services and args.category in (None, DEPLOY_CATEGORY):
        results.extend(check_deployed_copies())

    # Calculate summary
    summary = {}
    for result in results:
        cat = result["category"]
        if cat not in summary:
            healthy_in_cat = sum(1 for r in results if r["category"] == cat and r["healthy"])
            total_in_cat = len([r for r in results if r["category"] == cat])
            if healthy_in_cat == total_in_cat:
                summary[cat] = "healthy"
            elif healthy_in_cat > 0:
                summary[cat] = "degraded"
            else:
                summary[cat] = "unhealthy"

    # Output
    if args.format == "json":
        print(format_json(results, summary))
    else:
        print(format_text(results, summary))


if __name__ == "__main__":
    main()
