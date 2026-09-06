"""Generate the isolated environment a candidate build boots into.

Two mechanisms, and the split matters:

**HOME redirection does the heavy lifting.** Lay the round out so the worktree
is `<round>/home/lloyd` and run every canary process with `HOME=<round>/home`.
Then `Path.home()/"lloyd"` and `app.paths.LLOYD_HOME` are the same directory,
and `LLOYD_HOME.parent/"obsidian"` (what `prompt_builder` uses) and
`$HOME/obsidian` are the same directory. One lever neutralizes the sessions
dir, `autonomy-runs/`, the task registry, `workers.db`, the vault paths, and —
critically — `autonomy.py`'s `AUTONOMY_DIR`, which is a hardcoded
`Path.home()/"obsidian"/"autonomy"` that no config key reaches
(`config.yaml`'s `autonomy.task_dir` is dead config; nothing reads it). With
an empty scratch autonomy dir, `recover_stuck_tasks()` finds nothing to
rewrite.

**A config overlay handles the rest.** Ports, and the handful of side effects
that reach *outside* the filesystem: the worker pool claiming live jobs, the
supervisord socket, the Discord gateway.

The overlay reads `config.yaml` **raw**, never `app.config.CONFIG`, because
CONFIG has already expanded `${VAR}` placeholders — writing it back out would
bake LiveKit and API secrets into a plaintext file on disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

BACKEND_PORT = 18080
MCP_PORT = 18500


def build_overlay(live_raw: dict, *, backend_port: int = BACKEND_PORT,
                  mcp_port: int = MCP_PORT) -> dict:
    """Return the config overlay for a canary. Pure — no I/O.

    Note what is deliberately NOT overridden: `secondary_enabled`. It must
    equal the live value so `_sync_secondary_llm_state` is a no-op by
    construction, in addition to being unreachable via the bogus supervisord
    socket. Overriding it to false would make booting a canary *stop the live
    secondary vLLM*.
    """
    mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
    overlay: dict = {
        # Loopback only: a canary must never be reachable on the tailnet.
        "server": {"host": "127.0.0.1", "port": backend_port},
        "services": {
            "backend": f"http://127.0.0.1:{backend_port}",
            "lloyd_mcp": mcp_url,
            # Never let a canary's startup hook reach the live supervisord.
            "sync_secondary_llm": False,
        },
        "mcp_servers": {"lloyd-mcp": {"url": mcp_url}},
        # THE critical one: workers/queue.py claim_next() does claim-by-UPDATE,
        # so a second pool on the same DB executes real jobs. HOME redirection
        # already gives the canary its own empty workers.db; this is the
        # explicit second layer.
        "workers": {"enabled": False},
        "autonomy": {"enabled": False, "ticker_enabled": False},
        "discord": {"token": ""},
        "knowledge_graph": {"write_enabled": False},
    }
    return overlay


def write_overlay(round_dir: Path, *, backend_port: int = BACKEND_PORT,
                  mcp_port: int = MCP_PORT, live_root: Path | None = None) -> Path:
    root = live_root or LIVE_ROOT
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    overlay = build_overlay(raw, backend_port=backend_port, mcp_port=mcp_port)
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "canary-config.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path


def materialize_home(round_dir: Path, worktree: Path,
                     live_root: Path | None = None) -> Path:
    """Build the scratch HOME the canary runs under. Returns its path.

    The worktree must already exist at `<round>/home/lloyd` — `app/paths.py`
    calls `.resolve()`, so a symlink there would resolve back to the live tree
    and defeat the entire isolation model.
    """
    root = live_root or LIVE_ROOT
    home = round_dir / "home"
    home.mkdir(parents=True, exist_ok=True)

    expected = home / "lloyd"
    if worktree.resolve() != expected.resolve():
        raise RuntimeError(
            f"worktree must be at {expected} for HOME isolation to hold, got {worktree}")

    vault = home / "obsidian"
    (vault / "lloyd").mkdir(parents=True, exist_ok=True)
    # Empty on purpose: this is what makes recover_stuck_tasks() a no-op.
    (vault / "autonomy").mkdir(parents=True, exist_ok=True)
    (vault / "skills").mkdir(parents=True, exist_ok=True)
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    (vault / "backlog").mkdir(parents=True, exist_ok=True)

    # Copy the prompt surface so the canary's system prompt resembles the real
    # one. Copied, not symlinked — a canary must not be able to write to the
    # live vault.
    for name in ("SOUL.md", "MEMORY.md", "USER.md"):
        src = root.parent / "obsidian" / "lloyd" / name
        if src.exists():
            shutil.copy2(src, vault / "lloyd" / name)

    # Shared read-only caches: rebuilding these per round would be pointless.
    for name in (".cache", ".local", ".config", ".bun", ".npm"):
        target = Path.home() / name
        link = home / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target)
            except OSError:
                pass

    # .env is gitignored, so the worktree has none. Without it every ${VAR}
    # placeholder expands to "" and the canary diverges from live.
    env_src = root / ".env"
    env_link = worktree / ".env"
    if env_src.exists() and not env_link.exists():
        try:
            env_link.symlink_to(env_src)
        except OSError:
            shutil.copy2(env_src, env_link)

    return home


def canary_env(round_dir: Path, worktree: Path, *, overlay: Path,
               python: Path, mcp_port: int = MCP_PORT) -> dict:
    """Environment for the canary processes."""
    home = round_dir / "home"
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PYTHONPATH": f"{worktree}:{worktree}/scripts/intel-pipeline",
        "PYTHONUNBUFFERED": "1",
        "LLOYD_CONFIG_OVERLAY": str(overlay),
        "LLOYD_MCP_PORT": str(mcp_port),
        # A path that cannot exist, so a canary's startup hook physically
        # cannot reach the live supervisord and stop the live secondary vLLM.
        "LLOYD_SUPERVISOR_SOCK": str(round_dir / "no-such-supervisor.sock"),
        "LLOYD_SELFMOD_STATE": str(round_dir / "selfmod-state"),
        "LLOYD_GUARDIAN_STATE": str(round_dir / "guardian-state"),
        "PATH": f"{python.parent}:{env.get('PATH', '')}",
    })
    # No DISPLAY/WAYLAND_DISPLAY: a headful Chromium launch should fail fast
    # rather than pop a window onto the user's screen mid-gate.
    for key in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(key, None)
    return env


SUPERVISORD_TEMPLATE = """\
[unix_http_server]
file={sock}

[supervisord]
logfile={logs}/supervisord.log
pidfile={pid}
nodaemon=true
loglevel=info

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{sock}

[group:lloyd-mc]
programs=lloyd-backend,lloyd-mcp

[program:lloyd-mcp]
command={python} -m agent_mcp.main
directory={worktree}
autostart=true
autorestart={autorestart}
startsecs=5
startretries=2
stopwaitsecs=10
stopasgroup=true
killasgroup=true
environment={env}
stdout_logfile={logs}/mcp.log
stderr_logfile={logs}/mcp.err

[program:lloyd-backend]
command={python} server.py
directory={worktree}
autostart=true
autorestart={autorestart}
startsecs=5
startretries=2
stopwaitsecs=10
stopasgroup=true
killasgroup=true
environment={env}
stdout_logfile={logs}/server.log
stderr_logfile={logs}/server.err
"""


def _env_line(env: dict) -> str:
    """supervisord `environment=` line. Values are quoted; %% escapes %."""
    keep = ("HOME", "PYTHONPATH", "PYTHONUNBUFFERED", "LLOYD_CONFIG_OVERLAY",
            "LLOYD_MCP_PORT", "LLOYD_SUPERVISOR_SOCK", "LLOYD_SELFMOD_STATE",
            "LLOYD_GUARDIAN_STATE", "PATH")
    parts = []
    for k in keep:
        if k in env:
            parts.append(f'{k}="{str(env[k]).replace("%", "%%")}"')
    return ",".join(parts)


def write_supervisord_conf(round_dir: Path, worktree: Path, env: dict,
                           python: Path, *, autorestart: bool = True) -> tuple[Path, Path]:
    """Write a throwaway supervisord config. Returns (conf_path, sock_path).

    The canary runs under its own supervisord rather than bare subprocesses so
    the rollback drill exercises the *same* control plane the guardian uses in
    production — group-qualified names, FATAL semantics, XML-RPC and all.
    A drill against bare subprocesses would prove much less.
    """
    logs = round_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sock = round_dir / "supervisor.sock"
    conf = round_dir / "supervisord.conf"
    conf.write_text(SUPERVISORD_TEMPLATE.format(
        sock=sock, pid=round_dir / "supervisord.pid", logs=logs,
        python=python, worktree=worktree, env=_env_line(env),
        autorestart="true" if autorestart else "false",
    ), encoding="utf-8")
    return conf, sock
