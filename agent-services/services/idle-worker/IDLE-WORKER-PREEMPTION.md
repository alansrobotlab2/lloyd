# Idle-Worker Preemption Architecture

## Overview

The Lloyd Idle Worker Service executes background tasks when the local LLM (port 8097) is idle. It implements **GPU yield at call boundaries** — tasks pause when main tasks need the GPU and resume when idle again.

**Key principle:** No checkpoint state. Tasks naturally pause/resume at call boundaries (HTTP requests, file I/O).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Idle Worker Daemon                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Main Loop   │───▶│  Preemption  │                   │
│  │  (5 min)     │    │  Controller  │                   │
│  └──────────────┘    └──────┬───────┘                   │
│                             │                            │
│              ┌──────────────┼──────────────┐             │
│              │              │              │             │
│              ▼              ▼              ▼             │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│       │  HTTP    │   │  File    │   │  State   │       │
│       │  Calls   │   │  I/O     │   │  Writes  │       │
│       └──────────┘   └──────────┘   └──────────┘       │
│              │              │              │             │
│              └──────────────┼──────────────┘             │
│                             │                            │
│              ┌──────────────▼──────────────┐             │
│              │  wait_if_busy()              │             │
│              │  (250ms polls to 8097/health)│             │
│              └──────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## Preemption Controller

**Location:** `idle-worker.py` (lines 21-48)

```python
class PreemptionController:
    def __init__(self, check_interval=0.25):  # 250ms polls
        self.check_interval = check_interval
        self.health_url = "http://127.0.0.1:8097/health"
    
    def wait_if_busy(self):
        """Poll until model is idle."""
        while self.is_model_busy():
            logger.debug("Model busy, waiting...")
            time.sleep(self.check_interval)
        return True
    
    def is_model_busy(self):
        try:
            resp = urllib.request.urlopen(self.health_url, timeout=1)
            data = json.loads(resp.read())
            return data.get('active_sessions', 0) > 0
        except:
            return False  # Assume idle if can't reach
```

**Behavior:**
- Polls `127.0.0.1:8097/health` every 250ms
- Checks `active_sessions > 0` or `is_busy == true`
- Returns immediately if idle
- Blocks until GPU available if main tasks are running

---

## Call Boundary Wrapping

Every "call" (HTTP request, file I/O, database access) is wrapped:

```python
def check_source(source, controller):
    # Before HTTP call
    controller.wait_if_busy()
    result = fetch_from_api(source['url'])
    
    # Before file write
    controller.wait_if_busy()
    save_to_file(result)
    
    return result
```

**Wrapped functions:**
| Function | Call Type |
|----------|-----------|
| `check_github()` | HTTP (GitHub API) |
| `check_huggingface()` | HTTP (HF API) |
| `check_rss()` | HTTP (RSS feed) |
| `check_search()` | HTTP (Search API) |
| `update_source_state()` | File I/O (state.json) |
| `log_daily_note()` | File I/O (daily notes) |

---

## Task Lifecycle

```
1. Daemon starts
   │
   ▼
2. Check if model idle (8097/health)
   │
   ├── Busy ──▶ Wait 90s, check again
   │
   └── Idle ──▶ Proceed
                │
                ▼
3. Get next due task from task-queue.json
   │
   ├── None due ──▶ Wait 10min, check again
   │
   └── Task found ──▶ Execute task
                      │
                      ├── Before each call: wait_if_busy()
                      │
                      └── Complete ──▶ Update state, wait 5min
```

---

## State Management

**State file:** `task-queue.json`

```json
{
  "sources": {
    "gh-001": {
      "last_checked": "2026-03-18T09:00:00Z",
      "next_due": "2026-03-19T09:00:00Z",
      "last_result": {"status": "checked", "repo": "openclaw/openclaw"}
    }
  },
  "last_run": "2026-03-18T09:05:00Z"
}
```

**No checkpoint state needed** — tasks restart from beginning if preempted mid-execution. Since individual calls are short (< 1s), restarting the whole task is acceptable.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Poll interval | 250ms |
| Max wait before yielding | 250ms |
| CPU usage while waiting | ~0% (sleep-based) |
| GPU yield latency | < 250ms |
| Task restart overhead | None (no checkpoint save/load) |

---

## Tradeoffs

### Advantages
- **Simple:** No checkpoint state, no resume logic
- **Responsive:** GPU available to main tasks within 250ms
- **Low overhead:** No state persistence during task execution
- **Clean:** Tasks are naturally interruptible at call boundaries

### Limitations
- **No progress persistence:** If daemon restarts mid-task, task restarts from beginning
- **Atomic calls not interruptible:** Long-running single calls (e.g., 10s HTTP request) can't be preempted mid-call
- **Task restart acceptable:** Only works for tasks where restarting is cheap

### When to Use This Approach
- ✅ Short individual calls (< 5s)
- ✅ Tasks where restart is acceptable
- ✅ GPU yield is priority over progress preservation
- ✅ Simple implementation desired

### When to Use Full Preemption Instead
- ❌ Long individual calls (> 30s)
- ❌ Expensive operations that can't be redone
- ❌ Streaming/paginated data where progress matters
- ❌ Need progress persistence across daemon restarts

---

## Configuration

**Poll interval:** `PreemptionController(check_interval=0.25)`

Adjust based on needs:
- `0.1` (100ms): Faster yield, more CPU
- `0.25` (250ms): Balanced (default)
- `0.5` (500ms): Slower yield, less overhead

**Health endpoint:** `127.0.0.1:8097/health`

Target the correct port for your LLM service (Qwen3.5-2B consolidation model).

---

## Testing

**Verify preemption works:**
```bash
# Terminal 1: Start idle worker
sudo systemctl start lloyd-idle-worker.service
journalctl -u lloyd-idle-worker.service -f

# Terminal 2: Generate load on port 8097
# (Run LLM inference, watch idle worker pause)

# Check logs for "Model busy, waiting..."
```

**Verify GPU availability:**
```bash
# While idle worker running, check GPU usage
watch -n 1 nvidia-smi

# Main tasks should have priority, idle worker yields
```

---

## Related Files

| File | Purpose |
|------|---------|
| `idle-worker.py` | Main daemon with preemption controller |
| `task-queue.json` | Task state and scheduling |
| `idle-worker-tasks.md` | Task definitions (config) |
| `lloyd-idle-worker.service` | Systemd unit file |
| `README.md` | Installation and usage guide |

---

## Future Enhancements

- **Configurable poll interval:** Per-task or global setting
- **Task-specific yielding:** Some tasks don't need preemption
- **Metrics:** Track preemption frequency and wait times
- **Timeout handling:** Prevent tasks from running forever if GPU always busy
