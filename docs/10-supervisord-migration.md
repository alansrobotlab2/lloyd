# Supervisord Migration: Move Lloyd Services into Lloyd Repo

## Context

Lloyd currently depends on `agent-services` for its supervisord setup. The lloyd-specific service configs live in `/home/alansrobotlab/agent-services/supervisor/conf.d/` (4 files: lloyd-backend.conf, lloyd-frontend.conf, lloyd-mcp.conf, lloyd-mc.conf) and the supervisord instance is defined in `agent-services/supervisor/supervisord.conf`.

The venv is already self-contained at `lloyd/.venvs/lloyd/` (Python 3.12). The goal is to make lloyd fully self-contained by giving it its own supervisord config and removing the dependency on agent-services for service management.

## What Moves

| Item | Current Location | New Location |
|------|-----------------|--------------|
| lloyd-backend.conf | `agent-services/supervisor/conf.d/` | `lloyd/supervisor/conf.d/` |
| lloyd-frontend.conf | `agent-services/supervisor/conf.d/` | `lloyd/supervisor/conf.d/` |
| lloyd-mcp.conf | `agent-services/supervisor/conf.d/` | `lloyd/supervisor/conf.d/` |
| lloyd-mc.conf | `agent-services/supervisor/conf.d/` | `lloyd/supervisor/conf.d/` |
| supervisord config | (none — used agent-services one) | `lloyd/supervisor/supervisord.conf` (new) |

The venv at `.venvs/lloyd/` is already in the lloyd folder — no change needed.

## Implementation Steps

### 1. Create `lloyd/supervisor/supervisord.conf`

A lloyd-specific supervisord config using distinct socket/pid paths so it doesn't conflict with the agent-services supervisord:

```ini
[unix_http_server]
file=/tmp/lloyd-supervisor.sock

[supervisord]
logfile=/home/alansrobotlab/lloyd/logs/supervisord.log
pidfile=/tmp/lloyd-supervisord.pid
nodaemon=false
user=alansrobotlab

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/lloyd-supervisor.sock

[include]
files = /home/alansrobotlab/lloyd/supervisor/conf.d/*.conf
```

### 2. Create `lloyd/supervisor/conf.d/` with 4 conf files

Copy content from agent-services conf files as-is (paths are already correct).

**lloyd-backend.conf**
```ini
[program:lloyd-backend]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python /home/alansrobotlab/lloyd/server.py
directory=/home/alansrobotlab/lloyd
user=alansrobotlab
environment=HOME="/home/alansrobotlab",PATH="/home/alansrobotlab/.local/bin:/usr/local/bin:/usr/bin:/bin",ANTHROPIC_BASE_URL="http://127.0.0.1:8096",ANTHROPIC_API_KEY="no-key-required",ANTHROPIC_CUSTOM_MODEL_OPTION="Qwen3.5-122B-A10B",ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 122B Local"
autorestart=true
startsecs=5
startretries=3
autostart=true
stopwaitsecs=15
stdout_logfile=/home/alansrobotlab/lloyd/logs/server.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/server.err
stdout_logfile_maxbytes=10MB
stderr_logfile_maxbytes=10MB
```

**lloyd-frontend.conf**
```ini
[program:lloyd-frontend]
command=/usr/bin/npm --prefix /home/alansrobotlab/lloyd/web run dev
directory=/home/alansrobotlab/lloyd/web
user=alansrobotlab
environment=HOME="/home/alansrobotlab",PATH="/home/alansrobotlab/.local/bin:/usr/local/bin:/usr/bin:/bin"
autorestart=true
startsecs=5
startretries=3
autostart=true
stopwaitsecs=15
stdout_logfile=/home/alansrobotlab/lloyd/logs/frontend.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/frontend.err
stdout_logfile_maxbytes=10MB
stderr_logfile_maxbytes=10MB
```

**lloyd-mcp.conf**
```ini
[program:lloyd-mcp]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m mcp_server.main
directory=/home/alansrobotlab/lloyd
autostart=true
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/mcp.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/mcp.err
```

**lloyd-mc.conf**
```ini
[group:lloyd-mc]
programs=lloyd-backend,lloyd-frontend,lloyd-mcp
```

### 3. Update `CLAUDE.md` service management commands

Change the supervisorctl `-c` path from `agent-services/supervisor/supervisord.conf` to `lloyd/supervisor/supervisord.conf` in all three commands.

### 4. Remove `docs/supervisor.conf`

This is superseded by the proper `supervisor/` directory structure (and it had a `.venv` vs `.venvs/lloyd` path bug anyway).

### 5. Manual cleanup in agent-services (separate step)

After lloyd's own supervisord is confirmed working, remove the 4 lloyd conf files from `agent-services/supervisor/conf.d/` and reload agent-services supervisord. This avoids dual-management.

## Service Management Commands (after migration)

```bash
# Start supervisord (first time / after reboot)
distrobox enter lloyd -- supervisord -c /home/alansrobotlab/lloyd/supervisor/supervisord.conf

# Restart backend
distrobox enter lloyd -- supervisorctl -c /home/alansrobotlab/lloyd/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend

# Restart frontend
distrobox enter lloyd -- supervisorctl -c /home/alansrobotlab/lloyd/supervisor/supervisord.conf restart lloyd-mc:lloyd-frontend

# Status
distrobox enter lloyd -- supervisorctl -c /home/alansrobotlab/lloyd/supervisor/supervisord.conf status
```

## Verification

1. Start lloyd's supervisord: `distrobox enter lloyd -- supervisord -c /home/alansrobotlab/lloyd/supervisor/supervisord.conf`
2. Check status — all 3 programs should show RUNNING (lloyd-backend, lloyd-frontend, lloyd-mcp)
3. Confirm backend responds: `curl http://localhost:8080/`
4. Stop lloyd services in agent-services supervisord and confirm lloyd's supervisord is the sole manager
