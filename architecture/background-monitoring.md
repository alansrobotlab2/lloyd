---
segment: architecture
type: reference
tags: [architecture]

---

# Background Monitoring Daemon

## Overview

A continuous background monitoring system that checks internet sources for updates when the local LLM is idle. Runs as a daemon process (not cron-based),checking sources every 5 minutes with intelligent model idle detection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Scheduler Daemon                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Model Idle   │  │ Source       │  │ Check        │     │
│  │ Detection    │──│ Selection    │──│ Execution    │     │
│  │ (port 8097)  │  │ (next due)   │  │ + Logging    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Health       │  │ check-       │  │ background-  │     │
│  │ Endpoint     │  │ scheduler    │  │ task.log     │     │
│  │ 127.0.0.1:   │  │ .json        │  │              │     │
│  │ 8097/health  │  │ (state)      │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Main Process

**File:** `~/agents/dee/state/monitoring/scheduler-daemon.py`

**PID:** 2287333 (as of 2026-03-15 17:18)

**State File:** `~/agents/dee/state/monitoring/check-scheduler.json`

**Log File:** `~/agents/dee/state/monitoring/background-task.log`

**PID File:** `~/agents/dee/state/monitoring/background-task.pid`

### Source Configuration

**File:** `~/obsidian/knowledge/monitoring-interests.md`

Defines what sources to monitor with:
- Source name/ID
- Type (GitHub,HuggingFace,RSS,YouTube,Search)
- Priority (HIGH,MEDIUM,LOW)
- Frequency (24h,48h,7d,etc.)

## How It Works

### Model Idle Detection

Before each check,the daemon verifies the local LLM is not busy:

```python
# Health endpoint check
response = requests.get("http://127.0.0.1:8097/health")
active_sessions = response.json().get("active_sessions",0)

if active_sessions > 0:
    # Model is busy,wait 1-2 minutes and retry
    time.sleep(random.uniform(60,120))
    continue
```

**Port:** 8097 (local LLM consolidation model,Qwen3.5-2B)

**Threshold:** 0 active sessions = idle

### Main Loop

```python
while True:
    # Step 1: Check if model is idle
    if is_model_idle():
        # Step 2: Find next due source
        source = get_next_due_source()
        
        if source:
            # Step 3: Execute check
            result = check_source(source)
            
            # Step 4: Update state
            update_state(source,last_checked=now,result=result)
            
            # Step 5: Wait before next check
            time.sleep(300)  # 5 minutes
        else:
            # No sources due,wait longer
            time.sleep(600)  # 10 minutes
    else:
        # Model busy,wait and retry
        time.sleep(random.uniform(60,120))
```

### Source Selection

**Function:** `get_next_due_source()`

Selection criteria:
1. Sources are sorted by `next_due` timestamp (earliest first)
2. Priority order: HIGH → MEDIUM → LOW
3. Respects frequency settings (24h,48h,7d)
4. Skips sources that are:
   - Rate-limited (429) for <24h
   - Currently being checked
   - Past their `next_due` window

### Check Execution

**Function:** `check_source(source)`

Supported source types:
- **GitHub Releases** - GitHub API `/repos/{owner}/{repo}/releases/latest`
- **HuggingFace** - HF Hub API for model updates
- **RSS Feeds** - Standard RSS/Atom feed parsing
- **YouTube Channels** - Channel RSS feed (`videos.xml`)
- **Search Queries** - Periodic DuckDuckGo search results

Each check:
1. Fetches current state from source
2. Compares against stored state in `check-scheduler.json`
3. Logs any changes to `background-task.log`
4. Updates `last_checked` and `next_due` timestamps

### State Management

**File:** `check-scheduler.json`

```json
{
  "sources": {
    "github-openclaw": {
      "type": "github",
      "repo": "openclaw/openclaw",
      "priority": "HIGH",
      "frequency_hours": 24,
      "last_checked": "2026-03-15T17:14:24Z",
      "next_due": "2026-03-16T17:14:24Z",
      "last_result": {
        "tag_name": "v2026.3.7",
        "published_at": "2026-03-10T00:00:00Z"
      }
    },
    "hf-llama-cpp": {
      "type": "huggingface",
      "model": "TheBloke/llama-cpp-python",
      "priority": "MEDIUM",
      "frequency_hours": 48,
      ...
    }
  },
  "last_check": "2026-03-15T17:14:24Z",
  "rate_limits": {
    "github-openclaw": {
      "blocked_until": null,
      "consecutive_errors": 0
    }
  }
}
```

## Operational Details

### Starting the Daemon

```bash
cd ~/agents/dee/state/monitoring
./start