#!/usr/bin/env python3
"""
Lloyd Idle Worker Service

This daemon executes background tasks when the local LLM is idle.
It respects priority levels and scheduling from the task queue.
"""

import json
import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

# Preemption controller
class PreemptionController:
    """Yields control when model becomes busy, polls at 250ms intervals."""
    
    def __init__(self, check_interval=0.25):
        self.check_interval = check_interval
        self.model_port = 8091
        self.health_url = f"http://127.0.0.1:{self.model_port}/health"
    
    def wait_if_busy(self):
        """Poll health endpoint until model is idle."""
        while self.is_model_busy():
            logger.debug("Model busy, waiting...")
            time.sleep(self.check_interval)
        return True
    
    def is_model_busy(self):
        """Check if model has active sessions."""
        try:
            req = urllib.request.Request(self.health_url, method='GET')
            req.add_header('User-Agent', 'IdleWorker/1.0')
            
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get('active_sessions', 0) > 0 or data.get('is_busy', False)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            # Assume idle if we can't reach the endpoint
            pass
        return False

# Configuration
STATE_DIR = Path.home() / "lloyd/agent-services/services/idle-worker"
LOG_DIR = Path.home() / "lloyd/agent-services/logs"
LOG_FILE = LOG_DIR / "idle-worker.log"
PID_FILE = STATE_DIR / "idle-worker.pid"
STATE_FILE = STATE_DIR / "task-queue.json"
INTERESTS_FILE = Path.home() / "obsidian/knowledge/idle-worker-tasks.md"

# Timing
CHECK_INTERVAL_IDLE = 300  # 5 minutes when idle and work done
CHECK_INTERVAL_BUSY = 90   # 1-2 minutes when model is busy
CHECK_INTERVAL_NONE = 600  # 10 minutes when no sources due

# Model idle detection
MODEL_PORT = 8091
MODEL_HEALTH_URL = f"http://127.0.0.1:{MODEL_PORT}/health"

# Setup logging - ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global state
running = True
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running, shutdown_requested
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_requested = True
    running = False


def load_state():
    """Load scheduler state from JSON file."""
    if not STATE_FILE.exists():
        return {"sources": {}, "last_run": None}
    
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading state: {e}")
        return {"sources": {}, "last_run": None}


def save_state(state):
    """Save scheduler state to JSON file."""
    try:
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving state: {e}")


def parse_interests_file():
    """Parse monitoring-interests.md to extract source definitions."""
    sources = []
    
    if not INTERESTS_FILE.exists():
        logger.error(f"Interests file not found: {INTERESTS_FILE}")
        return sources
    
    try:
        with open(INTERESTS_FILE, 'r') as f:
            content = f.read()
        
        # Simple table parsing - look for source rows
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Track sections
            if line.startswith('### '):
                current_section = line[4:].strip()
                continue
            
            # Parse table rows (skip headers and separators)
            if line.startswith('|') and '|' in line[1:]:
                parts = [p.strip() for p in line.split('|')]
                
                # Skip header rows
                if 'ID' in line or 'Source' in line:
                    continue
                
                # Try to parse as data row
                if len(parts) >= 6:
                    try:
                        source_id = parts[1] if parts[1] != '--' else None
                        source_name = parts[2] if parts[2] != '--' else None
                        source_type = parts[3] if parts[3] != '--' else None
                        frequency = parts[4] if parts[4] != '--' else '24h'
                        priority = parts[5] if parts[5] != '--' else 'MEDIUM'
                        
                        if source_id and source_name and source_name != '--':
                            sources.append({
                                'id': source_id,
                                'name': source_name,
                                'type': source_type,
                                'frequency': frequency,
                                'priority': priority.upper()
                            })
                    except (IndexError, ValueError):
                        continue
                        
    except IOError as e:
        logger.error(f"Error reading interests file: {e}")
    
    return sources


def is_model_idle():
    """Check if the local LLM model is idle."""
    try:
        # Try to connect to the model health endpoint
        req = urllib.request.Request(MODEL_HEALTH_URL, method='GET')
        req.add_header('User-Agent', 'MonitoringDaemon/1.0')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                # Check if there are active sessions/requests
                data = json.loads(response.read().decode('utf-8'))
                
                # Check for active requests or busy state
                active_sessions = data.get('active_sessions', 0)
                active_requests = data.get('active_requests', 0)
                is_busy = data.get('is_busy', False)
                
                if active_sessions > 0 or active_requests > 0 or is_busy:
                    logger.info("Model is busy (active sessions/requests)")
                    return False
                
                logger.debug("Model appears idle")
                return True
            else:
                logger.warning(f"Health check returned status {response.status}")
                return False
                
    except urllib.error.URLError as e:
        # Connection refused likely means model is not running
        logger.debug(f"Model server not reachable: {e}")
        return True  # Assume idle if we can't reach it
    except (json.JSONDecodeError, KeyError, TimeoutError) as e:
        logger.warning(f"Error checking model status: {e}")
        return True  # Assume idle on error


def get_next_due_source(sources, state):
    """Get the next source that is due for checking, respecting priority."""
    now = datetime.now(timezone.utc)
    
    # Build source state
    if 'sources' not in state:
        state['sources'] = {}
    
    # Filter and sort sources
    due_sources = []
    
    for source in sources:
        source_id = source['id']
        source_state = state['sources'].get(source_id, {})
        
        # Parse last_checked
        last_checked_str = source_state.get('last_checked')
        if last_checked_str:
            try:
                last_checked = datetime.fromisoformat(last_checked_str.replace('Z', '+00:00'))
                # Ensure timezone-aware (handle naive datetimes from old state)
                if last_checked.tzinfo is None:
                    last_checked = last_checked.replace(tzinfo=timezone.utc)
            except ValueError:
                last_checked = None
        else:
            last_checked = None
        
        # Parse next_due
        next_due_str = source_state.get('next_due')
        if next_due_str:
            try:
                next_due = datetime.fromisoformat(next_due_str.replace('Z', '+00:00'))
                # Ensure timezone-aware (handle naive datetimes from old state)
                if next_due.tzinfo is None:
                    next_due = next_due.replace(tzinfo=timezone.utc)
            except ValueError:
                next_due = None
        else:
            # Calculate next_due based on frequency
            frequency = source.get('frequency', '24h')
            if last_checked:
                next_due = last_checked
                # Parse frequency (e.g., "24h", "48h", "7d")
                if frequency.endswith('h'):
                    hours = int(frequency[:-1])
                    from datetime import timedelta
                    next_due = last_checked + timedelta(hours=hours)
                elif frequency.endswith('d'):
                    days = int(frequency[:-1])
                    from datetime import timedelta
                    next_due = last_checked + timedelta(days=days)
                else:
                    next_due = last_checked
            else:
                # Never checked - schedule for soon
                next_due = now
        
        # Check if paused
        paused_until = source_state.get('paused_until')
        if paused_until:
            try:
                paused_until_dt = datetime.fromisoformat(paused_until.replace('Z', '+00:00'))
                # Ensure timezone-aware (handle naive datetimes from old state)
                if paused_until_dt.tzinfo is None:
                    paused_until_dt = paused_until_dt.replace(tzinfo=timezone.utc)
                if now < paused_until_dt:
                    logger.debug(f"Source {source_id} is paused until {paused_until}")
                    continue
            except ValueError:
                pass
        
        # Check if due
        if next_due and now >= next_due:
            priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            priority = source.get('priority', 'MEDIUM')
            due_sources.append((source, priority_order.get(priority, 1)))
    
    # Sort by priority (lower number = higher priority)
    due_sources.sort(key=lambda x: x[1])
    
    if due_sources:
        return due_sources[0][0]
    return None


def check_source(source, controller):
    """Execute a check for a given source, yielding at call boundaries."""
    source_id = source['id']
    source_type = source.get('type', 'unknown')
    source_name = source.get('name', source_id)
    
    logger.info(f"Checking source: {source_id} ({source_name}) - Type: {source_type}")
    
    try:
        # Wait if model is busy before starting
        controller.wait_if_busy()
        
        # Different check logic based on source type
        if source_type == 'GitHub API':
            return check_github(source_name, controller)
        elif source_type == 'HF API':
            return check_huggingface(source_name, controller)
        elif source_type == 'RSS':
            return check_rss(source_name, controller)
        elif source_type == 'Search':
            return check_search(source_name, controller)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return {'status': 'skipped', 'reason': f'Unknown type: {source_type}'}
            
    except Exception as e:
        logger.error(f"Error checking source {source_id}: {e}")
        return {'status': 'error', 'error': str(e)}


def check_github(repo_name, controller):
    """Check GitHub repository for releases/updates."""
    logger.debug(f"Checking GitHub repo: {repo_name}")
    
    # Yield before each API call
    controller.wait_if_busy()
    
    # Simplified - in production would use GitHub API
    # Each fetch is wrapped with preemption check
    return {'status': 'checked', 'repo': repo_name}


def check_huggingface(model_name, controller):
    """Check HuggingFace model for updates."""
    logger.debug(f"Checking HuggingFace model: {model_name}")
    
    # Yield before each API call
    controller.wait_if_busy()
    
    # Simplified - in production would use HF API
    return {'status': 'checked', 'model': model_name}


def check_rss(feed_url, controller):
    """Check RSS feed for new content."""
    logger.debug(f"Checking RSS feed: {feed_url}")
    
    # Yield before each API call
    controller.wait_if_busy()
    
    # Simplified - in production would parse RSS
    return {'status': 'checked', 'feed': feed_url}


def check_search(query, controller):
    """Perform web search."""
    logger.debug(f"Performing search: {query}")
    
    # Yield before each search call
    controller.wait_if_busy()
    
    # Simplified - in production would use search API
    return {'status': 'checked', 'query': query}


def update_source_state(source_id, result, controller):
    """Update the state for a checked source."""
    # Yield before file I/O
    controller.wait_if_busy()
    
    state = load_state()
    
    if 'sources' not in state:
        state['sources'] = {}
    
    now = datetime.now(timezone.utc)
    
    # Initialize source state if needed
    if source_id not in state['sources']:
        state['sources'][source_id] = {}
    
    source_state = state['sources'][source_id]
    
    # Update last_checked
    source_state['last_checked'] = now.isoformat()
    
    # Calculate next_due based on frequency
    # This would be read from interests file in production
    # For now, default to 24h
    from datetime import timedelta
    source_state['next_due'] = (now + timedelta(hours=24)).isoformat()
    
    # Update result
    source_state['last_result'] = result
    
    # Yield before saving state
    controller.wait_if_busy()
    save_state(state)
    logger.debug(f"Updated state for {source_id}")


def log_daily_note(result, controller):
    """Log check result to daily notes."""
    # Yield before file I/O
    controller.wait_if_busy()
    
    # Simplified - in production would write to Obsidian daily note
    logger.info(f"Logged result: {result}")


def main_loop():
    """Main daemon loop with preemption support."""
    global running, shutdown_requested
    
    # Initialize preemption controller (250ms check interval)
    controller = PreemptionController(check_interval=0.25)
    
    logger.info("Starting idle worker daemon with preemption support")
    
    while running:
        try:
            # Check if model is idle
            if not is_model_idle():
                logger.info("Model is busy, waiting...")
                time.sleep(CHECK_INTERVAL_BUSY)
                continue
            
            # Load sources and state
            sources = parse_interests_file()
            state = load_state()
            
            if not sources:
                logger.warning("No sources found in interests file")
                time.sleep(CHECK_INTERVAL_NONE)
                continue
            
            # Get next due source
            next_source = get_next_due_source(sources, state)
            
            if next_source:
                logger.info(f"Found due source: {next_source['id']}")
                
                # Check the source (with preemption support)
                result = check_source(next_source, controller)
                
                # Update state (with preemption support)
                update_source_state(next_source['id'], result, controller)
                
                # Log to daily notes (with preemption support)
                log_daily_note(result, controller)
                
                # Wait before next check
                logger.info(f"Waiting {CHECK_INTERVAL_IDLE/60} minutes before next check...")
                time.sleep(CHECK_INTERVAL_IDLE)
            else:
                logger.info("No sources due for checking")
                time.sleep(CHECK_INTERVAL_NONE)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)  # Wait before retrying


def main():
    """Main entry point."""
    global running
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Write PID file
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    logger.info(f"Monitoring daemon started (PID: {os.getpid()})")
    logger.info(f"State file: {STATE_FILE}")
    logger.info(f"Log file: {LOG_FILE}")
    
    try:
        main_loop()
    finally:
        # Cleanup PID file
        if PID_FILE.exists():
            PID_FILE.unlink()
        logger.info("Monitoring daemon stopped")


if __name__ == '__main__':
    main()
