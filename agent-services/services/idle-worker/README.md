# Lloyd Idle Worker Service

A systemd-managed background task executor that runs when the local LLM is idle.

## Location

**Service files:** `/home/alansrobotlab/lloyd/agent-services/services/idle-worker/`

**Systemd unit:** `/home/alansrobotlab/lloyd/agent-services/systemd/lloyd-idle-worker.service`

**State:** `/home/alansrobotlab/lloyd/agent-services/services/idle-worker/task-queue.json`

**Logs:** `/home/alansrobotlab/lloyd/agent-services/logs/idle-worker.log`

## Installation

```bash
# 1. Copy unit file to systemd
sudo cp ~/lloyd/agent-services/systemd/lloyd-idle-worker.service /etc/systemd/system/

# 2. Reload systemd
sudo systemctl daemon-reload

# 3. (Optional) Enable for auto-start
sudo systemctl enable lloyd-idle-worker.service

# 4. Start the service
sudo systemctl start lloyd-idle-worker.service
```

## Management

```bash
# Check status
sudo systemctl status lloyd-idle-worker.service

# View logs
journalctl -u lloyd-idle-worker.service -f

# Stop service
sudo systemctl stop lloyd-idle-worker.service

# Restart service
sudo systemctl restart lloyd-idle-worker.service
```

## Configuration

**Task definitions:** `/home/alansrobotlab/obsidian/knowledge/idle-worker-tasks.md`

Edit this file to add/remove background tasks. The daemon parses the table format and automatically picks up changes on the next check cycle.

## Features

- **GPU yield at call boundaries**: Polls model health every 250ms, waits if busy
- **No checkpoint state**: Tasks restart at next call boundary (simpler, no state persistence)
- **Priority-based scheduling**: HIGH → MEDIUM → LOW task ordering
- **Rate limiting**: Max 2 checks/hour, 24h backoff on 429 errors
- **Auto-restart**: systemd restarts on failure (30s delay)
- **Resource limits**: 512MB RAM, 25% CPU max

## Preemption Behavior

The daemon yields control when the local LLM (port 8097) becomes busy:

1. **Before each call** (HTTP request, file I/O): Polls `127.0.0.1:8097/health`
2. **If busy**: Waits 250ms, polls again
3. **When idle**: Proceeds with the call

Tasks can take minutes but will pause if main tasks need the GPU. No checkpoint state is needed — tasks naturally pause/resume at call boundaries.

## Task Types

| Type | Description |
|------|-------------|
| GitHub API | Monitor repository releases |
| HF API | Monitor HuggingFace models |
| RSS | Standard RSS feed parsing |
| Search | Periodic web search queries |
| YouTube | Channel RSS feed monitoring |

## Migration from Old Setup

The old monitoring daemon at `~/agents/dee/state/monitoring/` has been migrated to the agent-services structure as the idle-worker service. The old PID file and manual start/stop scripts are no longer needed when running as a systemd service.

To fully migrate:
1. Install the new systemd service (see above)
2. The state file has been copied to the new location
3. Old manual scripts can be removed after confirming the new service works
