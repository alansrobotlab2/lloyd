# Lloyd Idle Worker Service

A continuous background process that checks monitoring sources when the local LLM is idle.

## Overview

This daemon:
1. Runs continuously (not cron-based)
2. Checks if the local LLM (vLLM on port 8097) is idle before each check
3. If idle: picks the next due source, executes the check, waits 5 minutes
4. If busy: waits 1-2 minutes and retries
5. Maintains state in `check-scheduler.json`

## Files

- `scheduler-daemon.py` - Main background process
- `start-monitoring.sh` - Start the daemon
- `stop-monitoring.sh` - Stop the daemon
- `status.sh` - Check daemon status
- `check-scheduler.json` - State file tracking last checks
- `background-task.log` - Log file
- `background-task.pid` - PID file

## Usage

### Start
```bash
~/agents/dee/state/monitoring/start-monitoring.sh
```

### Stop
```bash
~/agents/dee/state/monitoring/stop-monitoring.sh
```

### Status
```bash
~/agents/dee/state/monitoring/status.sh
```

### View Logs
```bash
tail -f ~/agents/dee/state/monitoring/background-task.log
```

## How It Works

### Model Idle Detection
- Checks vLLM health endpoint at `http://127.0.0.1:8097/health`
- If active sessions/requests > 0 → model is busy
- If busy: wait 1-2 minutes, retry
- If idle: proceed with check

### Check Loop
```
while True:
    if is_model_idle():
        source = get_next_due_source()
        if source:
            check_source(source)
            update_state(source, last_checked=now)
            wait(5 minutes)
        else:
            wait(10 minutes)  # No sources due
    else:
        wait(1-2 minutes)  # Model busy
```

### Source Priority
1. HIGH priority sources checked first
2. Then MEDIUM
3. Then LOW
4. Respects frequency settings (24h, 48h, 7d, etc.)

### State Management
- Sources read from `~/obsidian/knowledge/monitoring-interests.md`
- State tracked in `check-scheduler.json`
- Each source tracks: `last_checked`, `next_due`, `last_result`

## Environment

The daemon uses these environment variables:
- `NODE_EXTRA_CA_CERTS=/home/alansrobotlab/agents/lloyd/certs/mc.crt`
- OpenClaw CLI: `/home/alansrobotlab/.npm-global/bin/openclaw`
- Gateway URL: `https://127.0.0.1:19789`

## Auto-Restart

The daemon runs with `nohup` and will:
- Continue running after terminal closes
- Log all output to `background-task.log`
- Can be manually restarted if it crashes

## Rate Limiting

- Max 2 checks per hour
- Backs off sources that return 429 for 24h
- Reduces frequency for slow responses (>10s)

## Notes

This is a work-in-progress monitoring system. The actual check implementations (GitHub, HuggingFace, RSS, Search) are simplified placeholders and should be expanded with real API calls.
