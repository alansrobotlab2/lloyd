#!/usr/bin/env python3
"""
GitHub Release Monitor
Checks specified repositories for new releases and notifies via OpenClaw.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
STATE_FILE = Path.home() / "agents/dee/state/monitoring/github-checks.json"
REPOS = [
    "openclaw/openclaw",
    "NVIDIA/Isaac-GR00T",
    "tobi/qmd"
]

# OpenClaw CLI configuration
OPENCLAW_CERTS = "/home/alansrobotlab/agents/lloyd/certs/mc.crt"
OPENCLAW_CLI = "/home/alansrobotlab/.npm-global/bin/openclaw"
GATEWAY_URL = "https://127.0.0.1:19789"
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

def get_github_release(repo: str) -> dict:
    """Fetch the latest release info from GitHub API."""
    import urllib.request
    import urllib.error
    
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {
        "User-Agent": "GitHub-Release-Monitor",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {
                "tag_name": data.get("tag_name", ""),
                "name": data.get("name", ""),
                "published_at": data.get("published_at", ""),
                "html_url": data.get("html_url", "")
            }
    except urllib.error.HTTPError as e:
        print(f"Error fetching {repo}: {e.code} {e.reason}")
        return None
    except Exception as e:
        print(f"Error fetching {repo}: {e}")
        return None

def load_state() -> dict:
    """Load the current state from JSON file."""
    if not STATE_FILE.exists():
        return {"repos": {}, "last_check": None}
    
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state: dict):
    """Save state to JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def send_openclaw_message(message: str):
    """Send a message via OpenClaw CLI."""
    if not GATEWAY_TOKEN:
        print("WARNING: OPENCLAW_GATEWAY_TOKEN not set, skipping notification")
        return False
    
    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = OPENCLAW_CERTS
    
    cmd = [
        "bun", OPENCLAW_CLI,
        "--url", GATEWAY_URL,
        "--token", GATEWAY_TOKEN,
        "message",
        "--message", message
    ]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"OpenClaw message failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Failed to send OpenClaw message: {e}")
        return False

def check_releases():
    """Check all monitored repos for new releases."""
    state = load_state()
    now = datetime.utcnow().isoformat() + "Z"
    
    notifications = []
    
    for repo in REPOS:
        print(f"Checking {repo}...")
        
        release = get_github_release(repo)
        if not release:
            continue
        
        repo_state = state["repos"].get(repo, {})
        last_tag = repo_state.get("tag_name", "")
        
        if last_tag and release["tag_name"] != last_tag:
            # New release detected!
            notifications.append(f"🆕 New release for {repo}:\n"
                               f"  Tag: {release['tag_name']}\n"
                               f"  Name: {release['name']}\n"
                               f"  Published: {release['published_at']}\n"
                               f"  URL: {release['html_url']}")
            print(f"  NEW: {release['tag_name']} (was {last_tag})")
        elif last_tag:
            print(f"  No change: {release['tag_name']}")
        else:
            print(f"  Initial state: {release['tag_name']}")
        
        # Update state for this repo
        state["repos"][repo] = {
            "tag_name": release["tag_name"],
            "name": release["name"],
            "published_at": release["published_at"],
            "html_url": release["html_url"],
            "last_checked": now
        }
    
    state["last_check"] = now
    
    # Save updated state
    save_state(state)
    
    # Send notification if there are new releases
    if notifications:
        message = "GitHub Release Monitor Alert\n\n" + "\n\n".join(notifications)
        print(f"\nSending notification...\n{message}")
        send_openclaw_message(message)
    else:
        print("\nNo new releases detected.")
    
    return state

def initialize_state():
    """Initialize state file with baseline versions."""
    baseline = {
        "repos": {
            "openclaw/openclaw": {
                "tag_name": "v2026.3.13-1",
                "name": "OpenClaw v2026.3.13-1",
                "published_at": "2026-03-15T00:00:00Z",
                "html_url": "https://github.com/openclaw/openclaw/releases/tag/v2026.3.13-1",
                "last_checked": None
            },
            "NVIDIA/Isaac-GR00T": {
                "tag_name": "n1.5-release",
                "name": "Isaac GR00T n1.5 Release",
                "published_at": "2026-03-15T00:00:00Z",
                "html_url": "https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.5-release",
                "last_checked": None
            },
            "tobi/qmd": {
                "tag_name": "v2.0.1",
                "name": "QMD v2.0.1",
                "published_at": "2026-03-15T00:00:00Z",
                "html_url": "https://github.com/tobi/qmd/releases/tag/v2.0.1",
                "last_checked": None
            }
        },
        "last_check": None
    }
    
    save_state(baseline)
    print("Initialized state file with baseline versions.")
    return baseline

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        initialize_state()
    else:
        check_releases()
