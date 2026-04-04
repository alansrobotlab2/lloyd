#!/usr/bin/env python3
"""
Email + Calendar Monitoring Agent

Periodic monitoring of email inbox and calendar for relevant items,
surfacing important updates to active Mission Control sessions.
Runs every 15 minutes via cron job.
"""
import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import requests
import urllib3

# Suppress InsecureRequestWarning for self-signed cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration paths
CONFIG_PATH = os.path.expanduser("~/.openclaw/monitoring-config.json")
STATE_PATH = os.path.expanduser("~/.openclaw/monitoring-state.json")
METRICS_LOG = os.path.expanduser("~/.openclaw/monitoring-metrics.jsonl")

# Gateway API URLs
GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://127.0.0.1:18789")
MC_API_URL = os.environ.get("MC_API_URL", "https://127.0.0.1:18789/api/mc")

# Gateway and MC auth
GATEWAY_TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", os.environ.get("GATEWAY_HOOKS_TOKEN", ""))

logger = logging.getLogger("monitoring-agent")


def setup_logging():
    """Configure logging for the monitoring agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config(path: str = CONFIG_PATH) -> Dict:
    """Load monitoring configuration from JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        # Set defaults if missing
        config.setdefault("interval_minutes", 15)
        config.setdefault("email_window_hours", 24)
        config.setdefault("calendar_window_hours", 48)
        config.setdefault("relevance_threshold", 50)
        config.setdefault("post_interval_minutes", 30)
        config.setdefault("priority_senders", [])
        config.setdefault("subject_keywords", ["meeting", "urgent", "action", "review", "deadline"])
        config.setdefault("enabled", True)
        config.setdefault("log_level", "info")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config: {e}")
        # Return minimal config
        return {
            "interval_minutes": 15,
            "email_window_hours": 24,
            "calendar_window_hours": 48,
            "relevance_threshold": 50,
            "post_interval_minutes": 30,
            "priority_senders": [],
            "subject_keywords": ["meeting", "urgent", "action", "review", "deadline"],
            "enabled": True,
            "log_level": "info"
        }


def load_state(path: str = STATE_PATH) -> Dict:
    """Load monitoring state from JSON file."""
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        # Set defaults if missing
        state.setdefault("last_email_timestamp", None)
        state.setdefault("last_post_time", None)
        state.setdefault("last_processed_ids", [])
        state.setdefault("last_cycle_start", None)
        state.setdefault("metrics", {
            "total_cycles": 0,
            "total_emails_polled": 0,
            "total_emails_relevant": 0,
            "total_calendar_polled": 0,
            "total_calendar_relevant": 0,
            "total_posts_sent": 0,
            "total_errors": 0
        })
        return state
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"State file not found or corrupted, resetting: {e}")
        return {
            "last_email_timestamp": None,
            "last_post_time": None,
            "last_processed_ids": [],
            "last_cycle_start": None,
            "metrics": {
                "total_cycles": 0,
                "total_emails_polled": 0,
                "total_emails_relevant": 0,
                "total_calendar_polled": 0,
                "total_calendar_relevant": 0,
                "total_posts_sent": 0,
                "total_errors": 0
            }
        }


def save_state(state: Dict, path: str = STATE_PATH):
    """Save monitoring state to JSON file atomically."""
    temp_path = path + ".tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(temp_path, path)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def log_metrics(entry: Dict):
    """Log a metrics entry to JSONL file."""
    try:
        os.makedirs(os.path.dirname(METRICS_LOG), exist_ok=True)
        with open(METRICS_LOG, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def get_headers() -> Dict:
    """Get HTTP headers for API calls."""
    headers = {"Content-Type": "application/json"}
    if GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_TOKEN}"
    return headers


def invoke_tool(tool_name: str, args: Dict) -> Optional[Any]:
    """Invoke a gateway tool via HTTP POST /tools/invoke."""
    url = f"{GATEWAY_URL}/tools/invoke"
    payload = {
        "tool": tool_name,
        "args": args
    }
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), verify=False, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Tool {tool_name} returned {resp.status_code}: {resp.text[:200]}")
            return None
        result = resp.json()
        if result.get("error"):
            logger.error(f"Tool {tool_name} error: {result['error']}")
            return None
        return result.get("result", result)
    except requests.Timeout:
        logger.error(f"Tool {tool_name} timed out")
        return None
    except Exception as e:
        logger.error(f"Tool {tool_name} error: {e}")
        return None


def poll_email(config: Dict, last_timestamp: Optional[str]) -> List[Dict]:
    """
    Poll email inbox for recent messages.
    Returns list of email items with scores.
    """
    logger.info("Polling email inbox...")
    
    # Call email_recent tool
    result = invoke_tool("email_recent", {"limit": 20})
    if result is None:
        logger.error("Failed to poll email")
        return []
    
    emails = []
    now = datetime.now(timezone.utc)
    
    for msg in result.get("messages", []):
        # Extract email metadata
        sender = msg.get("from", msg.get("sender", ""))
        subject = msg.get("subject", "")
        body = msg.get("body", msg.get("text", ""))[:500]
        timestamp = msg.get("timestamp", msg.get("date", ""))
        is_unread = msg.get("unread", msg.get("is_unread", False))
        is_flagged = msg.get("flagged", msg.get("is_flagged", False))
        msg_id = msg.get("id", msg.get("message_id", ""))
        
        # Skip if already processed
        if msg_id in state.get("last_processed_ids", []):
            continue
        
        # Calculate score
        score = 0
        reasons = []
        
        # Unread: +30
        if is_unread:
            score += 30
            reasons.append("unread")
        
        # Priority sender: +25
        if any(p in sender.lower() for p in config.get("priority_senders", [])):
            score += 25
            reasons.append("priority_sender")
        
        # Flagged: +20
        if is_flagged:
            score += 20
            reasons.append("flagged")
        
        # Subject keywords: +15
        subject_lower = subject.lower()
        if any(kw in subject_lower for kw in config.get("subject_keywords", [])):
            score += 15
            reasons.append("subject_keyword")
        
        # Recent (within 1 hour): +10
        try:
            msg_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if (now - msg_dt).total_seconds() < 3600:
                score += 10
                reasons.append("recent")
        except (ValueError, AttributeError):
            pass
        
        emails.append({
            "id": msg_id,
            "sender": sender,
            "subject": subject,
            "body": body,
            "timestamp": timestamp,
            "score": score,
            "reasons": reasons,
            "type": "email"
        })
    
    logger.info(f"Polled {len(emails)} emails, {len([e for e in emails if e['score'] >= config['relevance_threshold']])} relevant")
    return emails


def check_calendar(config: Dict, window_hours: int = 48) -> List[Dict]:
    """
    Check calendar for upcoming events.
    Returns list of calendar items with scores.
    """
    logger.info(f"Checking calendar for next {window_hours} hours...")
    
    # Call calendar_events tool
    result = invoke_tool("calendar_events", {"hours": window_hours})
    if result is None:
        logger.error("Failed to check calendar")
        return []
    
    events = []
    now = datetime.now(timezone.utc)
    
    for event in result.get("events", []):
        # Extract event metadata
        title = event.get("title", event.get("summary", ""))
        start_time = event.get("start", event.get("start_time", ""))
        end_time = event.get("end", event.get("end_time", ""))
        attendees = event.get("attendees", [])
        is_important = event.get("important", event.get("high_priority", False))
        event_id = event.get("id", event.get("event_id", ""))
        
        # Calculate score
        score = 0
        reasons = []
        
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            time_until = (start_dt - now).total_seconds()
            
            # Within 1 hour: +40
            if 0 < time_until < 3600:
                score += 40
                reasons.append("imminent")
            # Within 4 hours: +25
            elif 3600 <= time_until < 14400:
                score += 25
                reasons.append("soon")
            # Within 24 hours: +15
            elif 14400 <= time_until < 86400:
                score += 15
                reasons.append("today")
            
            # Important: +20
            if is_important:
                score += 20
                reasons.append("important")
            
            # Multiple attendees (>5): +10
            if len(attendees) > 5:
                score += 10
                reasons.append("many_attendees")
                
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse event time: {e}")
            pass
        
        events.append({
            "id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "score": score,
            "reasons": reasons,
            "type": "calendar"
        })
    
    logger.info(f"Found {len(events)} calendar events, {len([e for e in events if e['score'] >= config['relevance_threshold']])} relevant")
    return events


def get_active_session() -> Optional[str]:
    """
    Get the most active user session from Mission Control.
    Returns session key or None if no active sessions.
    """
    try:
        url = f"{MC_API_URL}/sessions"
        resp = requests.get(url, headers=get_headers(), verify=False, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Sessions API returned {resp.status_code}")
            return None
        
        data = resp.json()
        sessions = data.get("sessions", [])
        
        # Filter to user sessions (exclude hook, subagent, cron sessions)
        user_sessions = [
            s for s in sessions
            if ":sub:" not in s.get("sessionKey", "") and
               ":cron:" not in s.get("sessionKey", "") and
               ":hook:" not in s.get("sessionKey", "") and
               s.get("sessionKey", "").startswith("agent:main:")
        ]
        
        if not user_sessions:
            logger.info("No active user sessions found")
            return None
        
        # Sort by lastActivity descending
        user_sessions.sort(
            key=lambda s: s.get("lastActivity", ""),
            reverse=True
        )
        
        # Return most recent session
        most_recent = user_sessions[0]
        last_activity = most_recent.get("lastActivity", "")
        
        # Check if session is active (within last 30 minutes)
        try:
            activity_dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
            if (datetime.now(timezone.utc) - activity_dt).total_seconds() > 1800:
                logger.info(f"Most recent session inactive (>30 min): {most_recent.get('sessionKey')}")
                return None
        except (ValueError, AttributeError):
            pass
        
        logger.info(f"Active session detected: {most_recent.get('sessionKey')}")
        return most_recent.get("sessionKey")
        
    except Exception as e:
        logger.error(f"Failed to get active session: {e}")
        return None


def generate_summary(emails: List[Dict], events: List[Dict]) -> str:
    """Generate markdown summary of relevant items."""
    lines = []
    
    if emails:
        lines.append("📧 **New Emails**")
        for email in emails[:5]:  # Limit to 5
            sender = email.get("sender", "Unknown")[:30]
            subject = email.get("subject", "No subject")[:50]
            lines.append(f"- [{sender}] {subject}")
        if len(emails) > 5:
            lines.append(f"*...and {len(emails) - 5} more*")
        lines.append("")
    
    if events:
        lines.append("📅 **Upcoming Events**")
        for event in events[:5]:  # Limit to 5
            title = event.get("title", "Untitled")[:50]
            start = event.get("start_time", "")
            if start:
                try:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    time_str = start_dt.strftime("%I:%M %p")
                    lines.append(f"- {time_str}: {title}")
                except (ValueError, AttributeError):
                    lines.append(f"- {title}")
            else:
                lines.append(f"- {title}")
        if len(events) > 5:
            lines.append(f"*...and {len(events) - 5} more*")
        lines.append("")
    
    lines.append("[Full details in Mission Control → Sessions]")
    
    return "\n".join(lines)


def post_to_session(session_key: str, message: str) -> bool:
    """Post a message to a Mission Control session."""
    # Use send_to_session from gateway
    url = f"{GATEWAY_URL}/tools/invoke"
    payload = {
        "tool": "sessions_send",
        "args": {
            "sessionKey": session_key,
            "message": message,
            "timeoutSeconds": 0  # Fire-and-forget
        }
    }
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), verify=False, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Post to session returned {resp.status_code}: {resp.text[:200]}")
            return False
        result = resp.json()
        if result.get("ok"):
            logger.info(f"Posted to session {session_key}")
            return True
        else:
            logger.error(f"Post failed: {result.get('error', {}).get('message', 'unknown')}")
            return False
    except Exception as e:
        logger.error(f"Post to session error: {e}")
        return False


def can_post(state: Dict, config: Dict) -> bool:
    """Check if we can post (respect post_interval_minutes)."""
    last_post = state.get("last_post_time")
    if not last_post:
        return True
    
    try:
        last_post_dt = datetime.fromisoformat(last_post.replace('Z', '+00:00'))
        elapsed = (datetime.now(timezone.utc) - last_post_dt).total_seconds()
        min_interval = config.get("post_interval_minutes", 30) * 60
        return elapsed >= min_interval
    except (ValueError, AttributeError):
        return True


def run_cycle(config: Dict, state: Dict) -> Dict:
    """Run one monitoring cycle. Returns cycle result dict."""
    cycle_start = datetime.now(timezone.utc).isoformat()
    result = {
        "cycle_start": cycle_start,
        "emails_polled": 0,
        "emails_relevant": 0,
        "calendar_polled": 0,
        "calendar_relevant": 0,
        "post_sent": False,
        "error": None
    }
    
    try:
        # Poll email
        emails = poll_email(config, state.get("last_email_timestamp"))
        result["emails_polled"] = len(emails)
        
        # Filter relevant
        relevant_emails = [e for e in emails if e["score"] >= config["relevance_threshold"]]
        result["emails_relevant"] = len(relevant_emails)
        
        # Check calendar
        events = check_calendar(config, config.get("calendar_window_hours", 48))
        result["calendar_polled"] = len(events)
        
        # Filter relevant
        relevant_events = [e for e in events if e["score"] >= config["relevance_threshold"]]
        result["calendar_relevant"] = len(relevant_events)
        
        # Update state with processed IDs
        processed_ids = state.get("last_processed_ids", [])
        for email in emails:
            if email["id"] not in processed_ids:
                processed_ids.append(email["id"])
        for event in relevant_events:
            if event["id"] not in processed_ids:
                processed_ids.append(event["id"])
        
        # Keep only last 100 IDs to prevent unbounded growth
        state["last_processed_ids"] = processed_ids[-100:]
        state["last_email_timestamp"] = cycle_start
        
        # If no relevant items, skip posting
        if not relevant_emails and not relevant_events:
            logger.info("No relevant items found, skipping post")
            return result
        
        # Get active session
        session_key = get_active_session()
        if not session_key:
            logger.warning("No active session, skipping post")
            return result
        
        # Check if we can post
        if not can_post(state, config):
            logger.info("Post interval not elapsed, skipping post")
            return result
        
        # Generate and post summary
        summary = generate_summary(relevant_emails, relevant_events)
        logger.info(f"Posting summary to {session_key}...")
        logger.info(f"Summary:\n{summary}")
        
        if post_to_session(session_key, summary):
            result["post_sent"] = True
            state["last_post_time"] = cycle_start
            state["metrics"]["total_posts_sent"] += 1
        
        # Update state
        state["last_processed_ids"] = processed_ids
        state["last_cycle_start"] = cycle_start
        
    except Exception as e:
        logger.error(f"Cycle error: {e}", exc_info=True)
        result["error"] = str(e)
        state["metrics"]["total_errors"] += 1
    
    # Update metrics
    state["metrics"]["total_cycles"] += 1
    state["metrics"]["total_emails_polled"] += result["emails_polled"]
    state["metrics"]["total_emails_relevant"] += result["emails_relevant"]
    state["metrics"]["total_calendar_polled"] += result["calendar_polled"]
    state["metrics"]["total_calendar_relevant"] += result["calendar_relevant"]
    
    return result


def main():
    """Main entry point for monitoring agent."""
    setup_logging()
    logger.info("Email + Calendar Monitoring Agent starting...")
    
    # Load config and state
    config = load_config()
    state = load_state()
    
    # Check if enabled
    if not config.get("enabled", True):
        logger.info("Monitoring agent disabled in config, exiting")
        return
    
    # Run single cycle
    result = run_cycle(config, state)
    
    # Save state
    save_state(state)
    
    # Log result
    log_metrics(result)
    
    # Summary
    if result.get("error"):
        logger.error(f"Cycle completed with error: {result['error']}")
    else:
        logger.info(
            f"Cycle complete: {result['emails_polled']} emails, "
            f"{result['emails_relevant']} relevant, "
            f"{result['calendar_polled']} events, "
            f"{result['calendar_relevant']} relevant, "
            f"post_sent={result['post_sent']}"
        )


if __name__ == "__main__":
    main()
