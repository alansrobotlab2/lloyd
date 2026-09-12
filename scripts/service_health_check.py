#!/usr/bin/env python3
"""Service Health Check Skill

Bundle multiple service status checks into a single call.
Returns structured status for LLM, MCP, and other Lloyd services.
"""

import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
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

CATEGORIES = {
    "llm": ["agent-llm-primary", "agent-llm-secondary"],
    "lloyd": ["lloyd-backend", "lloyd-frontend", "lloyd-mcp"],
    "retrieval": ["agent-qmd-daemon"],
    "all": list(SERVICES.keys()),
}


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

        # Special handling for supervisorctl
        if "supervisorctl" in command[0]:
            if "RUNNING" in output:
                status = "RUNNING"
                healthy = True
            elif "STOPPED" in output:
                status = "STOPPED"
                healthy = False
            elif "STARTING" in output:
                status = "STARTING"
                healthy = False
            elif "STOPPING" in output:
                status = "STOPPING"
                healthy = False
            else:
                status = output if output else "unknown"
                healthy = result.returncode == 0

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
        service_names = args.services
    elif args.category:
        service_names = CATEGORIES[args.category]
    else:
        # Default: check all services
        service_names = CATEGORIES["all"]

    # Run checks
    results = []
    for name in service_names:
        if name in SERVICES:
            results.append(check_service(name, SERVICES[name]))

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
