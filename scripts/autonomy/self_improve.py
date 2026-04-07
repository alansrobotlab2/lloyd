#!/usr/bin/env python3
"""
Self-Improvement Loop (Karpathy Cycle)

Implements a tight measure → propose → evaluate → keep/revert loop
for autonomous system prompt optimization.
"""

import json
import logging
import os
import re
import subprocess
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import argparse
import requests

# Add idler directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple logging setup (avoiding config dependency)
import logging
logger = logging.getLogger("self_improve")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
SESSIONS_DIR = os.path.expanduser("~/lloyd/sessions/")
CORRECTIONS_FILE = os.path.expanduser("~/obsidian/memory/corrections.md")
METRICS_DIR = os.path.expanduser("~/lloyd/_pipeline/metrics/")
QUALITY_SCORE_FILE = os.path.join(METRICS_DIR, "quality-score.jsonl")
WATERMARKS_FILE = os.path.expanduser("~/lloyd/_pipeline/autonomy-watermarks.json")
OBSIDIAN_DIR = os.path.expanduser("~/obsidian")
PENDING_IMPROVEMENTS_FILE = os.path.expanduser("~/lloyd/_pipeline/metrics/pending-improvements.jsonl")

# Local LLM config
LLM_ENDPOINT = "http://127.0.0.1:8096/v1/chat/completions"
LLM_MODEL = "Qwen3.5-122B-A10B"
LLM_API_KEY = os.environ.get("LOCAL_LLM_KEY", "dummy")

# Workspace files for context
WORKSPACE_FILES = [
    os.path.expanduser("~/obsidian/lloyd/SOUL.md"),
    os.path.expanduser("~/obsidian/lloyd/MEMORY.md"),
    os.path.expanduser("~/obsidian/lloyd/USER.md"),
    os.path.expanduser("~/obsidian/memory/corrections.md"),
]

# Files to read for judge evaluation
JUDGE_FILES = [
    os.path.expanduser("~/obsidian/lloyd/SOUL.md"),
    os.path.expanduser("~/obsidian/lloyd/MEMORY.md"),
    os.path.expanduser("~/obsidian/lloyd/USER.md"),
]


def call_llm(prompt: str, temperature: float = 0.7) -> str | None:
    """Call local 122B via vLLM API."""
    try:
        response = requests.post(
            LLM_ENDPOINT,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2000,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def get_session_date(session_file: str) -> str | None:
    """
    Get the date from a session file (supports both JSONL and JSON formats).

    - JSONL (.jsonl): reads line-by-line, finds last entry with 'timestamp'
    - JSON (.json): reads as single object, checks messages for timestamps

    Returns date as YYYY-MM-DD or None.
    """
    try:
        with open(session_file) as f:
            content = f.read().strip()

        if not content:
            return None

        timestamps = []

        if session_file.endswith('.json'):
            # Lloyd format: single JSON object with messages array
            try:
                data = json.loads(content)
                # Check top-level timestamps

                for key in ('last_updated', 'session_start', 'created_at', 'started_at', 'timestamp'):
                    ts = data.get(key)
                    if ts and isinstance(ts, str) and 'T' in ts:
                        timestamps.append(ts)
                # Check messages for timestamps (some formats include per-message timestamps)
                for msg in data.get('messages', data.get('conversation', [])):
                    ts = msg.get('timestamp')
                    if ts and isinstance(ts, str) and 'T' in ts:
                        timestamps.append(ts)
            except json.JSONDecodeError:
                return None
        else:
            # Legacy JSONL format: one JSON object per line
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp")
                    if ts and isinstance(ts, str) and 'T' in ts:
                        timestamps.append(ts)
                except json.JSONDecodeError:
                    continue

        if timestamps:
            # Use the last timestamp found
            date_part = timestamps[-1].split("T")[0]
            if re.match(r"\d{4}-\d{2}-\d{2}", date_part):
                return date_part

        return None
    except Exception as e:
        logger.error(f"Error reading session date from {session_file}: {e}")
        return None


def get_recent_sessions(days: int = 7) -> list[str]:
    """Get session files from the last N days (by last interaction timestamp)."""
    if not os.path.exists(SESSIONS_DIR):
        return []

    cutoff_date = (datetime.now() - timedelta(days=days)).date()
    sessions = []

    session_dir = Path(SESSIONS_DIR)
    for f in list(session_dir.glob("*.json")) + list(session_dir.glob("*.jsonl")):
        session_date = get_session_date(str(f))
        if session_date:
            try:
                file_date = datetime.strptime(session_date, "%Y-%m-%d").date()
                if file_date >= cutoff_date:
                    sessions.append((str(f), file_date))
            except ValueError:
                continue

    # Sort by date descending
    sessions.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in sessions]

def get_sessions_for_date(date_str: str) -> list[str]:
    """Get session files whose last interaction date matches the given date."""
    if not os.path.exists(SESSIONS_DIR):
        return []

    sessions = []
    session_dir = Path(SESSIONS_DIR)
    for f in list(session_dir.glob("*.json")) + list(session_dir.glob("*.jsonl")):
        session_date = get_session_date(str(f))
        if session_date == date_str:
            sessions.append(str(f))

    return sessions




def parse_corrections_last_days(days: int = 7) -> int:
    """Count correction entries from the last N days."""
    if not os.path.exists(CORRECTIONS_FILE):
        return 0
    
    with open(CORRECTIONS_FILE) as f:
        content = f.read()
    
    # Find all ## YYYY-MM-DD headers
    pattern = r"^## (\d{4}-\d{2}-\d{2})"
    matches = re.findall(pattern, content, re.MULTILINE)
    
    if not matches:
        return 0
    
    # Count entries within the date range
    cutoff = datetime.now() - timedelta(days=days)
    count = 0
    
    for date_str in matches:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date >= cutoff:
                count += 1
        except ValueError:
            continue
    
    return count

def parse_corrections_for_date(date_str: str) -> int:
    """Count correction entries for a specific date."""
    if not os.path.exists(CORRECTIONS_FILE):
        return 0
    
    with open(CORRECTIONS_FILE) as f:
        content = f.read()
    
    # Find all ## YYYY-MM-DD headers
    pattern = r"^## (\d{4}-\d{2}-\d{2})"
    matches = re.findall(pattern, content, re.MULTILINE)
    
    # Count entries matching the specific date
    count = sum(1 for d in matches if d == date_str)
    return count


def _iter_messages(session_file: str):
    """Yield (role, content_str) from either Hermes JSON or legacy JSONL session files."""
    try:
        with open(session_file) as f:
            content = f.read().strip()
        if not content:
            return
        if session_file.endswith('.json'):
            data = json.loads(content)
            for msg in data.get('messages', []):
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role', '')
                c = msg.get('content', '')
                if isinstance(c, list):
                    c = ' '.join(b.get('text', '') for b in c if isinstance(b, dict))
                yield role, c
        else:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict) or entry.get('type') != 'message':
                        continue
                    msg = entry.get('message', {})
                    role = msg.get('role', '')
                    c = msg.get('content', '')
                    if isinstance(c, list):
                        c = ' '.join(b.get('text', '') for b in c if isinstance(b, dict))
                    yield role, c
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error reading {session_file}: {e}")


def is_real_session(session_file: str) -> bool:
    """
    Check if a session contains at least 2 real user messages.
    
    A user message is considered "real" if it passes ALL these filters:
    - Does NOT contain <daily_notes> or </daily_notes>
    - Does NOT contain <memory_context> (system-injected prefill)
    - Does NOT contain "Session Startup"
    - Does NOT contain "heartbeat" (case-insensitive) or "HEARTBEAT_OK"
    - Does NOT contain <active_mode>
    - Does NOT start with "Sender (untrusted metadata)" without other substantial content
    - Message content (after stripping system patterns) is > 30 characters
    
    Returns True if at least 2 user messages pass all filters.
    """
    real_message_count = 0
    for role, content in _iter_messages(session_file):
        if role != "user":
            continue

        # Strip timestamp prefixes like [Mon 2026-03-23 07:30 PDT]
        content = re.sub(r'\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{3,4}\]\s*', '', content)

        content_lower = content.lower()

        if "<daily_notes>" in content or "</daily_notes>" in content:
            continue
        if content.strip().startswith("<memory_context>"):
            continue
        if "Session Startup" in content:
            continue
        if "heartbeat" in content_lower or "HEARTBEAT_OK" in content:
            continue
        if "<active_mode>" in content:
            continue
        if "Sender (untrusted metadata)" in content:
            content = re.sub(r'Sender \(untrusted metadata\):\s*```json.*?```', '', content, flags=re.DOTALL)
        if content.strip().startswith("[cron:"):
            continue
        if content.strip().startswith("System: ["):
            continue
        if len(content.strip()) <= 30:
            continue

        real_message_count += 1
        if real_message_count >= 2:
            return True

    return False


def compute_metrics(session_files: list, corrections_count: int, date_str: str) -> dict:
    """
    Compute quality metrics from session files and corrections count.
    
    Args:
        session_files: list of session file paths
        corrections_count: number of corrections for the date
        date_str: date string for the output
    
    Returns:
        dict with correction_rate, efficiency, engagement, composite, sessions_analyzed
    """
    if not session_files:
        return {
            "correction_rate": 0,
            "efficiency": 0,
            "engagement": 0,
            "composite": 0,
            "sessions_analyzed": 0,
        }
    
    # Filter session_files through is_real_session() - only count real sessions
    real_sessions = [f for f in session_files if is_real_session(f)]
    
    # Compute correction rate based on REAL sessions only
    correction_rate = corrections_count / len(real_sessions) if real_sessions else 0
    
    # Parse session JSONL files (only real sessions)
    total_user_messages = 0
    total_tool_results = 0
    
    for session_file in real_sessions:
        for role, _content in _iter_messages(session_file):
            if role == "user":
                total_user_messages += 1
            elif role == "toolResult":
                total_tool_results += 1
    
    # Compute efficiency (tool calls per user message)
    efficiency = total_tool_results / total_user_messages if total_user_messages > 0 else 0
    
    # Compute engagement (average user messages per real session)
    engagement = total_user_messages / len(real_sessions) if real_sessions else 0
    
    # Composite score (0-100 scale)
    # Each sub-score is 0.0 to 1.0
    # Correction score: fewer corrections = better
    # 0 corrections/session = 1.0, 0.2+ corrections/session = 0.0
    correction_score = max(0, 1 - correction_rate * 5)
    
    # Efficiency score: fewer tool calls per message = better
    # 0 tools/msg = 1.0, 10+ tools/msg = 0.0
    efficiency_score = max(0, 1 - (efficiency / 10))
    
    # Engagement score: log-scale so diminishing returns
    # 1 msg = 0, ~4 msgs = 0.4, ~8 msgs = 0.6, ~32 msgs = 1.0
    engagement_score = min(1, math.log2(max(engagement, 1)) / 5)
    
    # Weighted composite (0-100)
    composite = (correction_score * 0.40 + efficiency_score * 0.30 + engagement_score * 0.30) * 100
    
    return {
        "correction_rate": round(correction_rate, 3),
        "efficiency": round(efficiency, 2),
        "engagement": round(engagement, 1),
        "composite": round(composite, 1),
        "sessions_analyzed": len(real_sessions),
    }

def measure() -> dict:
    """
    Compute quality metrics from existing data (last 1 day).
    Returns dict with individual metrics + composite score.
    """
    logger.info("Measuring session quality metrics...")
    
    # Get recent sessions
    sessions = get_recent_sessions(1)
    if not sessions:
        logger.warning("No recent sessions found")
        return {
            "correction_rate": 0,
            "efficiency": 0,
            "engagement": 0,
            "composite": 0,
            "sessions_analyzed": 0,
        }
    
    # Count corrections
    correction_count = parse_corrections_last_days(1)
    
    # Use helper to compute metrics
    result = compute_metrics(sessions, correction_count, datetime.now().strftime("%Y-%m-%d"))
    
    # Store result
    os.makedirs(METRICS_DIR, exist_ok=True)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    
    with open(QUALITY_SCORE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Metrics: correction_rate={result['correction_rate']}, "
                f"efficiency={result['efficiency']}, engagement={result['engagement']}, "
                f"composite={result['composite']}")
    
    return result


def load_history() -> list:
    """Load recent experiment results from quality-score.jsonl."""
    if not os.path.exists(QUALITY_SCORE_FILE):
        return []
    
    results = []
    with open(QUALITY_SCORE_FILE) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return results[-30:]  # last 30 entries


def propose(metrics: dict, history: list) -> dict | None:
    """
    Ask local 122B to propose one improvement.
    
    Args:
        metrics: current quality scores from measure()
        history: list of recent experiment results from quality-score.jsonl
    
    Returns:
        {"file": "agents/lloyd/SOUL.md", "diff": "...", "hypothesis": "...", "target_metric": "..."}
        or None if no improvement needed
    """
    logger.info("Asking LLM to propose improvement...")
    
    # Read workspace files
    context_parts = []
    for file_path in WORKSPACE_FILES:
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    content = f.read()
                    context_parts.append(f"--- {file_path} ---\n{content}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
    
    context = "\n\n".join(context_parts)
    
    # Recent corrections summary
    recent_corrections = ""
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE) as f:
                content = f.read()
                # Get last 500 chars
                recent_corrections = content[-500:] if len(content) > 500 else content
        except Exception as e:
            logger.error(f"Error reading corrections: {e}")
    
    # Recent experiment history summary
    history_summary = ""
    if history:
        # Get last 5 experiments
        recent_history = history[-5:]
        history_parts = []
        for h in recent_history:
            if h.get("experiment") and h["experiment"].get("change"):
                change = h["experiment"]["change"]
                outcome = h["experiment"].get("outcome", "unknown")
                history_parts.append(f"- Target: {change.get('target_metric')}, Hypothesis: {change.get('hypothesis', 'N/A')}, Outcome: {outcome}")
        if history_parts:
            history_summary = "\n".join(history_parts)
    
    # Build prompt
    prompt = f"""You are an AI system optimizer. Your task is to suggest ONE specific change to improve system performance.

## Current Quality Metrics
- Correction rate: {metrics.get('correction_rate', 0)} (fewer is better)
- Response efficiency: {metrics.get('efficiency', 0)} tool calls per user message (lower is better)
- Session engagement: {metrics.get('engagement', 0)} user messages per session (higher is better)
- Composite score: {metrics.get('composite', 0)}/100

## Recent Corrections
{recent_corrections or "No recent corrections"}

## Recent Experiment History
{history_summary or "No recent experiments"}

## Current System Files
{context}

## Your Task
Analyze the metrics and system files. Identify the weakest metric and propose ONE specific, targeted change to improve it.

## Constraints
- Propose ONLY ONE change to ONE file
- Change must be a specific text replacement (not vague)
- The original_text MUST be copied EXACTLY from the file content above — character-for-character
- The original_text must be a COMPLETE line or multi-line block — never a partial line or sentence fragment
- The original_text must be at least 40 characters long
- Only propose changes to text you can see IN FULL — never reference "[... middle section omitted" markers or incomplete text
- The replacement_text must be a COMPLETE replacement — it replaces original_text entirely, so include everything needed
- Only modify: SOUL.md, AGENTS.md, or TOOLS.md
- NEVER remove safety rules or boundaries
- NEVER change task routing rules (those need human approval)
- Prefer small, targeted edits over rewrites

## Output Format
Return JSON with these fields:
{{
  "file": "agents/lloyd/SOUL.md",
  "original_text": "exact text to replace",
  "replacement_text": "new text to use",
  "hypothesis": "why this change should improve the metric",
  "target_metric": "correction_rate or efficiency or engagement"
}}

If no improvement is needed, return: {{"reason": "no improvement needed"}}

/no_think"""

    response = call_llm(prompt, temperature=0.7)
    
    if not response:
        logger.error("LLM returned no response")
        return None
    
    # Try to parse JSON
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            if "reason" in data and "no improvement" in data["reason"].lower():
                logger.info("LLM determined no improvement needed")
                return None
            
            # Validate required fields
            required = ["file", "original_text", "replacement_text", "hypothesis", "target_metric"]
            if all(k in data for k in required):
                # Normalize file path
                if data["file"].startswith("lloyd/"):
                    data["file"] = os.path.expanduser(f"~/obsidian/{data['file']}")
                elif data["file"].startswith("agents/lloyd/"):
                    # Remap legacy agents/lloyd/ paths to lloyd/
                    data["file"] = os.path.expanduser(f"~/obsidian/lloyd/{data['file'][len('agents/lloyd/'):]}")
                elif data["file"].startswith("~/"):
                    data["file"] = os.path.expanduser(data["file"])
                elif not data["file"].startswith("/"):
                    data["file"] = os.path.expanduser(f"~/obsidian/lloyd/{data['file']}")
                
                # Verify original_text actually exists in the file on disk
                if os.path.exists(data["file"]):
                    with open(data["file"]) as f:
                        actual_content = f.read()
                    if data["original_text"] not in actual_content:
                        logger.warning(f"Proposal verification failed: original_text not found in actual file {data['file']}")
                        return None
                
                logger.info(f"Proposed change to {data['file']} targeting {data['target_metric']}")
                return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Response: {response}")
    
    logger.warning("LLM did not return valid proposal")
    return None


def get_diverse_user_messages(count: int = 5, session_files: list[str] | None = None) -> list[str]:
    """Get diverse user messages from recent sessions.
    
    Args:
        count: Number of messages to return
        session_files: Optional list of session files to use instead of get_recent_sessions()
    """
    if session_files is None:
        sessions = get_recent_sessions(1)
    else:
        sessions = session_files
    messages = []
    
    # Filter patterns to exclude (same as is_real_session)
    exclude_patterns = [
        r"<daily_notes>",
        r"</daily_notes>",
        r"Session Startup",
        r"heartbeat",
        r"HEARTBEAT_OK",
        r"<active_mode>",
        r"Sender.*untrusted metadata",
    ]
    
    for session_file in sessions:
        try:
            for role, content in _iter_messages(session_file):
                if role != "user":
                    continue

                skip = False
                for pattern in exclude_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        skip = True
                        break

                if skip or len(content.strip()) < 20:
                    continue

                messages.append(content)
                if len(messages) >= count * 2:
                    break

            if len(messages) >= count * 2:
                break
        except Exception as e:
            logger.error(f"Error reading {session_file}: {e}")
            continue
    
    # Return diverse subset
    return messages[:count] if len(messages) >= count else messages


def validate_proposal(change: dict) -> tuple[bool, str]:
    """
    Pre-validate a proposal before evaluation.
    Returns (is_valid, reason).
    """
    file_path = change.get("file", "")
    original_text = change.get("original_text", "")
    replacement_text = change.get("replacement_text", "")
    
    if not os.path.exists(file_path):
        return False, f"file not found: {file_path}"
    
    try:
        with open(file_path) as f:
            file_content = f.read()
    except Exception as e:
        return False, f"error reading file: {e}"
    
    # Check 1: original_text must exist in file
    if original_text not in file_content:
        return False, "original_text not found in file"
    
    # Check 2: original_text must be at least 40 chars
    if len(original_text) < 40:
        return False, f"original_text too short ({len(original_text)} chars, need 40+)"
    
    # Check 3: exactly one occurrence
    count = file_content.count(original_text)
    if count != 1:
        return False, f"original_text appears {count} times (need exactly 1)"
    
    # Check 4: replacement doesn't overlap with surrounding text
    idx = file_content.index(original_text)
    after_original = file_content[idx + len(original_text):idx + len(original_text) + 50]
    if after_original and replacement_text.rstrip().endswith(after_original[:20].rstrip()):
        return False, "replacement text overlaps with text after original"
    
    return True, "ok"


def evaluate(change: dict, session_files: list[str] | None = None, skip_branch: bool = False) -> bool:
    """
    Apply change on branch, replay recent sessions, judge with 122B.
    
    Args:
        change: dict from propose() with file, original_text, replacement_text
    
    Returns:
        True if change should be kept, False to revert
    """
    logger.info("Evaluating proposed change...")
    
    # Create timestamp for branch name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"experiment-{timestamp}"
    
    file_path = change["file"]
    original_text = change["original_text"]
    replacement_text = change["replacement_text"]
    
    # 1. Create git branch (skip if skip_branch=True)
    if not skip_branch:
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=OBSIDIAN_DIR,
                check=True,
                capture_output=True,
            )
            logger.info(f"Created branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    else:
        logger.info(f"Skipping branch creation (skip_branch=True)")
    
    # 2. Read ORIGINAL content BEFORE making any changes
    try:
        with open(file_path) as f:
            original_file_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read original file: {e}")
        subprocess.run(
            ["git", "checkout", "main", "--", file_path],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        return False
    
    # 3. Apply the change
    try:
        if original_text not in original_file_content:
            logger.error(f"Original text not found in {file_path}")
            with GitRevertContext(file_path, branch_name, OBSIDIAN_DIR) as ctx:
                ctx.failed = True
            return False
        
        new_content = original_file_content.replace(original_text, replacement_text, 1)
        
        with open(file_path, "w") as f:
            f.write(new_content)
        
        logger.info(f"Applied change to {file_path}")
    except Exception as e:
        logger.error(f"Failed to apply change: {e}")
        with GitRevertContext(file_path, branch_name, OBSIDIAN_DIR) as ctx:
            ctx.failed = True
        return False
    
    # 4. Get diverse user messages
    user_messages = get_diverse_user_messages(5, session_files=session_files)
    if not user_messages:
        logger.warning("No user messages found for evaluation")
        with GitRevertContext(file_path, branch_name, OBSIDIAN_DIR) as ctx:
            ctx.failed = True
        return False
    
    # 5. Build ORIGINAL and MODIFIED system contexts
    # At this point, the file on disk has the MODIFIED content.
    # original_file_content = content before our change
    # new_content = content after our change (what's on disk now)
    # We build two full contexts by reading all workspace files and swapping
    # the target file's content.
    
    def build_context(files, target_path, target_content):
        """Build system context with a specific version of the target file."""
        parts = []
        for fp in files:
            if os.path.exists(fp):
                try:
                    if os.path.abspath(fp) == os.path.abspath(target_path):
                        content = target_content
                    else:
                        with open(fp) as f:
                            content = f.read()
                    if len(content) > 5000:
                        content = content[:3500] + "\n\n[... middle section omitted ...]\n\n" + content[-1500:]
                    parts.append(f"--- {fp} ---\n{content}")
                except Exception as e:
                    logger.error(f"Error reading {fp}: {e}")
        return "\n\n".join(parts)
    
    original_context = build_context(JUDGE_FILES, file_path, original_file_content)
    modified_context = build_context(JUDGE_FILES, file_path, new_content)
    
    # 6. Generate and judge responses for each user message
    wins = 0
    
    for i, user_msg in enumerate(user_messages):
        # Generate response using ORIGINAL system prompt
        original_response = call_llm(f"""You are an AI assistant with the following system prompt:

{original_context}

User message: {user_msg}

Respond concisely and helpfully.""", temperature=0.7)
        
        # Generate response using MODIFIED system prompt
        modified_response = call_llm(f"""You are an AI assistant with the following system prompt:

{modified_context}

User message: {user_msg}

Respond concisely and helpfully.""", temperature=0.7)
        
        if not original_response or not modified_response:
            logger.error(f"LLM failed to generate responses for message {i+1}")
            with GitRevertContext(file_path, branch_name, OBSIDIAN_DIR) as ctx:
                ctx.failed = True
            return False
        
        # Blind judge: randomly label A/B to avoid position bias
        use_random_order = (i % 2 == 0)  # Alternate order
        if use_random_order:
            response_a, response_b = original_response, modified_response
            modified_label = "B"
        else:
            response_a, response_b = modified_response, original_response
            modified_label = "A"
        
        judge_prompt = f"""Compare these two responses to a user message.

User message: {user_msg}

Response A: {response_a}

Response B: {response_b}

Which response better serves the user? Consider:
- Accuracy (does it follow instructions?)
- Conciseness (user prefers concise responses)
- Helpfulness (does it actually help?)

Reply ONLY "A" or "B". /no_think"""

        judge_response = call_llm(judge_prompt, temperature=0.1)
        
        # Determine winner: did the judge pick the modified version?
        if judge_response and modified_label in judge_response:
            wins += 1
            logger.info(f"Judgment {i+1}: Modified wins")
        else:
            logger.info(f"Judgment {i+1}: Original wins")
    
    # 7. Decide
    keep = wins >= 3
    logger.info(f"Results: {wins}/{len(user_messages)} for modified — {'KEEPING' if keep else 'REVERTING'}")
    
    # Commit change if keeping
    if keep:
        try:
            subprocess.run(
                ["git", "add", file_path],
                cwd=OBSIDIAN_DIR,
                check=True,
                capture_output=True,
            )
            commit_msg = f"self-improve: {change.get('hypothesis', 'experiment')}"
            if skip_branch:
                commit_msg = f"replay: {change.get('hypothesis', 'experiment')}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=OBSIDIAN_DIR,
                check=True,
                capture_output=True,
            )
            if not skip_branch:
                subprocess.run(
                    ["git", "checkout", "main"],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "merge", branch_name],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
                logger.info("Change merged to main")
            else:
                logger.info("Change committed to current branch (skip_branch=True)")
        except subprocess.CalledProcessError as e:
            stdout = e.stdout.decode(errors="replace").strip() if e.stdout else ""
            stderr = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            logger.error(f"Failed to commit/merge: {e}\n  stdout: {stdout}\n  stderr: {stderr}")
            if not skip_branch:
                # Try to recover
                subprocess.run(
                    ["git", "checkout", "main"],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
    else:
        # Revert using context manager
        if skip_branch:
            # Just revert the file, don't touch branches
            try:
                subprocess.run(
                    ["git", "checkout", "HEAD", "--", file_path],
                    cwd=OBSIDIAN_DIR,
                    check=True,
                    capture_output=True,
                )
                logger.info("Changes reverted (skip_branch=True)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to revert file: {e}")
        else:
            with GitRevertContext(file_path, branch_name, OBSIDIAN_DIR) as ctx:
                pass  # Context manager handles revert
    
    return keep


class GitRevertContext:
    """Context manager for git revert operations during evaluation."""
    
    def __init__(self, file_path, branch_name, obsidian_dir):
        self.file_path = file_path
        self.branch_name = branch_name
        self.obsidian_dir = obsidian_dir
        self.failed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Only revert if we failed or had an exception
        if self.failed or exc_type is not None:
            self.revert()
        return False
    
    def revert(self):
        """Revert changes and clean up branch."""
        try:
            subprocess.run(
                ["git", "checkout", "main", "--", self.file_path],
                cwd=self.obsidian_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to revert file: {e}")
        
        try:
            subprocess.run(
                ["git", "branch", "-D", self.branch_name],
                cwd=self.obsidian_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete branch: {e}")
        
        logger.info("Changes reverted")


def load_watermarks() -> dict:
    """Load watermarks file."""
    if not os.path.exists(WATERMARKS_FILE):
        return {}
    
    try:
        with open(WATERMARKS_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading watermarks: {e}")
        return {}


def update_watermark(key: str = "self_improvement"):
    """Update watermark after successful run."""
    watermarks = load_watermarks()
    
    if key not in watermarks:
        watermarks[key] = {}
    
    watermarks[key]["last_run"] = datetime.now(timezone.utc).isoformat()
    
    # Update session mtime
    sessions = get_recent_sessions(1)
    if sessions:
        latest_mtime = os.path.getmtime(sessions[0])
        watermarks[key]["last_session_mtime"] = latest_mtime
    
    try:
        with open(WATERMARKS_FILE, "w") as f:
            json.dump(watermarks, f, indent=2)
        logger.info("Updated watermark")
    except Exception as e:
        logger.error(f"Error updating watermark: {e}")


def check_watermark() -> bool:
    """Check if new sessions since last run. Returns True if should run."""
    watermarks = load_watermarks()
    self_improve = watermarks.get("self_improvement", {})
    
    last_run = self_improve.get("last_run")
    last_session_mtime = self_improve.get("last_session_mtime")
    
    if not last_run:
        logger.info("No previous run found — proceeding")
        return True
    
    # Get newest session mtime
    sessions = get_recent_sessions(1)
    if not sessions:
        logger.warning("No sessions found")
        return False
    
    current_mtime = os.path.getmtime(sessions[0])
    
    # Check if sessions are newer than last run
    if last_session_mtime and current_mtime <= last_session_mtime:
        logger.info("No new sessions since last run — skipping")
        return False
    
    return True


def log_result(metrics: dict, change: dict | None, outcome: str):
    """Append result to quality-score.jsonl."""
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
        "experiment": {
            "change": change,
            "outcome": outcome,
        } if change else None,
    }
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(QUALITY_SCORE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Logged result: {outcome}")


def queue_proposal(change: dict, metrics: dict, eval_score: tuple = None):
    """Queue a proposal for human approval.
    
    Args:
        change: proposal change dict
        metrics: current metrics dict
        eval_score: tuple of (wins, total) from evaluation, or None
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": change["file"],
        "original_text": change["original_text"],
        "replacement_text": change["replacement_text"],
        "hypothesis": change.get("hypothesis", ""),
        "target_metric": change.get("target_metric", ""),
        "eval_score": f"{eval_score[0]}/{eval_score[1]}" if eval_score else "N/A",
        "composite_before": metrics.get("composite", 0),
    }
    
    os.makedirs(os.path.dirname(PENDING_IMPROVEMENTS_FILE), exist_ok=True)
    with open(PENDING_IMPROVEMENTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Queued proposal to {PENDING_IMPROVEMENTS_FILE}")


def apply_pending_proposal():
    """Apply the top pending proposal with git branch + commit."""
    if not os.path.exists(PENDING_IMPROVEMENTS_FILE):
        logger.warning(f"No pending improvements file found: {PENDING_IMPROVEMENTS_FILE}")
        return False
    
    # Read pending proposals
    pending = []
    with open(PENDING_IMPROVEMENTS_FILE) as f:
        for line in f:
            try:
                pending.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not pending:
        logger.info("No pending improvements to apply")
        return False
    
    # Get top proposal (first one)
    proposal = pending[0]
    logger.info(f"Applying pending proposal: {proposal.get('hypothesis', 'N/A')}")
    
    file_path = proposal["file"]
    original_text = proposal["original_text"]
    replacement_text = proposal["replacement_text"]
    
    # Create timestamp for branch name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"pending-{timestamp}"
    
    # Create git branch
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        logger.info(f"Created branch: {branch_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create branch: {e}")
        return False
    
    # Apply the change
    try:
        with open(file_path) as f:
            content = f.read()
        
        if original_text not in content:
            logger.error(f"Original text not found in {file_path}")
            subprocess.run(
                ["git", "checkout", "main", "--", file_path],
                cwd=OBSIDIAN_DIR,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=OBSIDIAN_DIR,
                check=True,
                capture_output=True,
            )
            return False
        
        new_content = content.replace(original_text, replacement_text, 1)
        
        with open(file_path, "w") as f:
            f.write(new_content)
        
        # Commit and merge
        subprocess.run(
            ["git", "add", file_path],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"self-improve: apply-pending {proposal.get('hypothesis', 'N/A')}"],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "merge", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        
        # Remove applied proposal from queue
        remaining = pending[1:]
        with open(PENDING_IMPROVEMENTS_FILE, "w") as f:
            for p in remaining:
                f.write(json.dumps(p) + "\n")
        
        logger.info("Pending proposal applied and removed from queue")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply proposal: {e}")
        subprocess.run(
            ["git", "checkout", "main", "--", file_path],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        return False


def backfill():
    """
    Backfill quality scores for all historical dates.
    Scans SESSIONS_DIR, groups by date, computes metrics per day.
    """
    logger.info("Starting backfill of historical quality scores...")
    
    # Check if SESSIONS_DIR exists
    if not os.path.exists(SESSIONS_DIR):
        logger.warning(f"Sessions directory not found: {SESSIONS_DIR}")
        print("No sessions directory found.")
        return
    
    # Group session files by date (using last interaction timestamp)
    sessions_by_date = {}
    for f in Path(SESSIONS_DIR).glob("*.jsonl"):
        # Skip reset/deleted files
        if '.reset.' in f.name or '.deleted.' in f.name:
            continue
        
        date_str = get_session_date(str(f))
        if date_str is None:
            # Skip sessions without timestamps
            continue
        
        if date_str not in sessions_by_date:
            sessions_by_date[date_str] = []
        sessions_by_date[date_str].append(str(f))
    
    if not sessions_by_date:
        logger.warning("No session files found")
        print("No session files found.")
        return
    
    # Sort dates (oldest first)
    sorted_dates = sorted(sessions_by_date.keys())
    
    # Check existing entries in quality-score.jsonl to avoid duplicates
    existing_dates = set()
    if os.path.exists(QUALITY_SCORE_FILE):
        with open(QUALITY_SCORE_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "date" in entry:
                        existing_dates.add(entry["date"])
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Found {len(sorted_dates)} dates with sessions, {len(existing_dates)} existing entries")
    
    # Process each date
    processed = 0
    skipped = 0
    
    for date_str in sorted_dates:
        # Skip if already has entry
        if date_str in existing_dates:
            logger.info(f"Skipping {date_str} (already exists)")
            skipped += 1
            continue
        
        # Get sessions for this date
        session_files = sessions_by_date[date_str]
        
        # Skip if no sessions (shouldn't happen, but safety check)
        if not session_files:
            continue
        
        # Count corrections for this specific date
        corrections_count = parse_corrections_for_date(date_str)
        
        # Compute metrics using helper
        metrics = compute_metrics(session_files, corrections_count, date_str)
        
        # Skip days with fewer than 3 real sessions during backfill
        if metrics['sessions_analyzed'] < 3:
            print(f"{date_str}: skipped (only {metrics['sessions_analyzed']} session)")
            continue
        
        # Write to quality-score.jsonl
        os.makedirs(METRICS_DIR, exist_ok=True)
        entry = {
            "date": date_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        
        with open(QUALITY_SCORE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Print summary line
        print(f"{date_str}: composite={metrics['composite']} (sessions={metrics['sessions_analyzed']}, corrections={corrections_count}, efficiency={metrics['efficiency']}, engagement={metrics['engagement']})")
        
        processed += 1
    
    logger.info(f"Backfill complete: {processed} processed, {skipped} skipped")
    print(f"\nBackfill complete: {processed} dates processed, {skipped} skipped (already existed)")



def replay():
    """
    Replay historical days from quality-score.jsonl sequentially.
    Each day builds on the previous day's results — kept changes persist.
    """
    logger.info("Starting replay of historical days...")
    
    # 1. Ensure quality-score.jsonl has backfill data
    if not os.path.exists(QUALITY_SCORE_FILE):
        logger.error("quality-score.jsonl not found — run --backfill first")
        print("ERROR: quality-score.jsonl not found. Run --backfill first.")
        return
    
    # Read all dates from quality-score.jsonl
    dates_with_data = []
    all_history = []
    with open(QUALITY_SCORE_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "date" in entry:
                    dates_with_data.append(entry["date"])
                all_history.append(entry)
            except json.JSONDecodeError:
                continue
    
    if not dates_with_data:
        logger.error("quality-score.jsonl is empty — run --backfill first")
        print("ERROR: quality-score.jsonl is empty. Run --backfill first.")
        return
    
    # Sort dates chronologically
    sorted_dates = sorted(set(dates_with_data))
    logger.info(f"Found {len(sorted_dates)} dates to replay")
    print(f"\n=== Starting Replay ===")
    print(f"Dates to process: {len(sorted_dates)}")
    
    # 3. Create git branch
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"experiment-replay-{timestamp}"
    
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
        )
        logger.info(f"Created branch: {branch_name}")
        print(f"Created branch: {branch_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create branch: {e}")
        print(f"ERROR: Failed to create branch: {e}")
        return
    
    # Track results
    results = []
    days_with_proposals = 0
    days_kept = 0
    days_reverted = 0
    days_no_proposal = 0
    
    # 4. For each date
    for date_str in sorted_dates:
        print(f"\n{'='*60}")
        print(f"Processing: {date_str}")
        print(f"{'='*60}")
        
        # a. Load that day's metrics
        day_metrics = None
        for entry in all_history:
            if entry.get("date") == date_str:
                day_metrics = entry
                break
        
        if not day_metrics:
            print(f"  No metrics found for {date_str}, skipping")
            continue
        
        # Filter history to include only entries up to and including this date
        # (for replay, we want to see all previous outcomes including replay results)
        filtered_history = [h for h in all_history if h.get("date") <= date_str]
        
        # c. Get session files for that specific date
        day_sessions = get_sessions_for_date(date_str)
        if not day_sessions:
            print(f"  No sessions found for {date_str}, skipping")
            continue
        
        print(f"  Sessions: {len(day_sessions)}")
        
        # d. Call propose
        print(f"  Proposing improvement...")
        change = propose(day_metrics, filtered_history)
        
        if change is None:
            print(f"  No proposal for {date_str}")
            results.append({
                "date": date_str,
                "outcome": "no_proposal",
                "hypothesis": None,
                "target_metric": None,
                "wins": 0,
                "total": 0
            })
            days_no_proposal += 1
            continue
        
        days_with_proposals += 1
        print(f"  Proposed: {change.get('hypothesis')}")
        print(f"  Target: {change.get('target_metric')}")
        
        # e. Call evaluate with skip_branch=True
        print(f"  Evaluating...")
        kept = evaluate(change, session_files=day_sessions, skip_branch=True)
        
        # f/g. Log result
        if kept:
            days_kept += 1
            outcome = "kept"
            wins = 3  # We don't have the exact count, but it passed
            total = 5
        else:
            days_reverted += 1
            outcome = "reverted"
            wins = 0  # Placeholder
            total = 5
        
        results.append({
            "date": date_str,
            "outcome": outcome,
            "hypothesis": change.get("hypothesis"),
            "target_metric": change.get("target_metric"),
            "wins": wins,
            "total": total
        })
        
        # h. Print summary line
        status = "KEPT" if kept else "REVERTED"
        print(f"  Result: {status}")
    
    # 5. Print summary
    print(f"\n{'='*60}")
    print("=== Replay Summary ===")
    print(f"{'='*60}")
    print(f"Days processed: {len(sorted_dates)}")
    print(f"Proposals: {days_with_proposals} ({days_no_proposal} days had no proposal)")
    print(f"Kept: {days_kept}")
    print(f"Reverted: {days_reverted}")
    print(f"\nDay-by-day:")
    
    for r in results:
        if r["outcome"] == "no_proposal":
            print(f"  {r['date']}: NO PROPOSAL")
        else:
            status = "KEPT" if r["outcome"] == "kept" else "REVERTED"
            wins = r.get("wins", 0)
            total = r.get("total", 5)
            print(f"  {r['date']}: {status} ({wins}/{total}) — {r.get('target_metric', 'N/A')} — \"{r.get('hypothesis', 'N/A')[:50]}...")
    
    # Print cumulative diff
    print(f"\nCumulative diff from main:")
    try:
        diff_result = subprocess.run(
            ["git", "diff", "main...HEAD", "--stat"],
            cwd=OBSIDIAN_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        print(diff_result.stdout)
    except Exception as e:
        print(f"  Could not get diff: {e}")
    
    print(f"\nBranch: {branch_name}")
    print(f"Review with: git diff main...{branch_name}")
    print(f"\nNOTE: Branch is NOT merged. Alan decides whether to merge any of it.")
    
    logger.info(f"Replay complete: {days_kept} kept, {days_reverted} reverted, {days_no_proposal} no proposal")


def main():
    """Run one cycle of the self-improvement loop."""
    parser = argparse.ArgumentParser(description="Self-improvement loop for system prompt optimization")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Run measure and propose only, do not evaluate or modify files")
    parser.add_argument("--apply-pending", action="store_true",
                        help="Apply the top pending proposal from the queue")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill historical quality scores for all dates")
    parser.add_argument("--replay", action="store_true",
                        help="Replay historical days from quality-score.jsonl sequentially")
    args = parser.parse_args()
    
    # Handle mutually exclusive flags
    if args.replay and (args.dry_run or args.apply_pending or args.backfill):
        parser.error("--replay is mutually exclusive with --dry-run, --apply-pending, and --backfill")
    if args.backfill and (args.dry_run or args.apply_pending):
        parser.error("--backfill is mutually exclusive with --dry-run and --apply-pending")
    
    # Handle --replay flag
    if args.replay:
        replay()
        return
    
    # Handle --backfill flag
    if args.backfill:
        backfill()
        return
    
    # Handle --apply-pending flag
    if args.apply_pending:
        success = apply_pending_proposal()
        if success:
            print("Pending proposal applied successfully")
        else:
            print("Failed to apply pending proposal")
        return
    
    logger.info("Starting self-improvement cycle...")
    
    # Watermark check
    if not check_watermark():
        print("No new sessions — skipping")
        return
    
    # Measure
    metrics = measure()
    print(f"Metrics: composite={metrics.get('composite', 0)}/100")
    
    # Load history
    history = load_history()
    
    # Propose
    change = propose(metrics, history)
    if change is None:
        log_result(metrics, None, "no_proposal")
        print("No improvement proposed — skipping evaluation")
        update_watermark()
        return
    
    print(f"\nProposed change:")
    print(f"  File: {change.get('file')}")
    print(f"  Target metric: {change.get('target_metric')}")
    print(f"  Hypothesis: {change.get('hypothesis')}")
    
    # Handle --dry-run flag
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Proposed change (not applied):")
        print(f"  Original text: {change.get('original_text')[:200]}...")
        print(f"  Replacement text: {change.get('replacement_text')[:200]}...")
        print("Updating watermark to prevent re-trigger...")
        update_watermark()
        return
    
    # Pre-validate: verify proposal is safe to apply
    is_valid, reason = validate_proposal(change)
    if not is_valid:
        logger.warning(f"Pre-validation failed: {reason}")
        log_result(metrics, change, "bad_match")
        queue_proposal(change, metrics, eval_score=None)  # Queue for human review
        update_watermark()
        print(f"\nProposal queued for human review — {reason}")
        return

    # Evaluate
    result = evaluate(change)
    if isinstance(result, tuple):
        kept, wins, total = result
    else:
        # Fallback for backwards compatibility
        kept = result
        wins, total = 0, 0
    
    # Log result (only for actual experiments, not dry-run)
    log_result(metrics, change, "kept" if kept else "reverted")
    
    # Queue for human approval instead of auto-apply
    if kept:
        eval_score = (wins, total)
        queue_proposal(change, metrics, eval_score=eval_score)
        print(f"\nChange passed evaluation and queued for approval.")
        print(f"  Use --apply-pending to apply the top queued proposal.")
    else:
        print(f"\nChange failed evaluation and was reverted.")
    
    # Update watermark
    update_watermark()
    
    # Summary
    outcome = "queued" if kept else "reverted"
    print(f"\nSelf-improvement cycle complete:")
    print(f"  Target metric: {change.get('target_metric')}")
    print(f"  Hypothesis: {change.get('hypothesis')}")
    print(f"  Outcome: {outcome}")
    print(f"  Current composite score: {metrics.get('composite', 0)}/100")


if __name__ == "__main__":
    main()
