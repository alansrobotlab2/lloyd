#!/usr/bin/env python3
"""
mine-trajectories.py — Mine trajectory JSONL files for skill candidates.

Reads trajectory files from Phase 1 (extract-trajectories.py) and produces
markdown candidate files for skill patterns (both error and success patterns).

Output: ~/obsidian/skills/candidates/candidate-{pattern-slug}-{YYYYMMDD}.md
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ── Paths ────────────────────────────────────────────────────────────────────

TRAJECTORY_DIR = Path.home() / "obsidian" / "memory" / "_pipeline" / "trajectories"
OUTPUT_DIR = Path.home() / "obsidian" / "memory" / "_pipeline" / "skills" / "candidates"


def set_trajectory_dir(path: Path) -> None:
    """Override trajectory directory for testing."""
    global TRAJECTORY_DIR
    TRAJECTORY_DIR = path


# ── Error categorization ─────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    ("permission", re.compile(r"permission denied|access denied|forbidden|EPERM|EACCES", re.IGNORECASE)),
    ("not_found", re.compile(r"file not found|no such file|not found|404|ENOENT", re.IGNORECASE)),
    ("timeout", re.compile(r"timeout|timed out|ETIMEDOUT|deadline exceeded", re.IGNORECASE)),
    ("network", re.compile(r"connection refused|ECONNREFUSED|DNS|ENOTFOUND|network|EHOSTUNREACH", re.IGNORECASE)),
    ("validation", re.compile(r"invalid|malformed|parse error|syntax error|schema|validation", re.IGNORECASE)),
    ("resource", re.compile(r"out of memory|disk full|quota|ENOMEM|ENOSPC|resource exhausted", re.IGNORECASE)),
]


def categorize_error(text: str) -> str:
    """Categorize an error message into a type."""
    if not text:
        return "logic"
    text_lower = text.lower()
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(text):
            return name
    return "logic"


def categorize_result_summary(result_summary: str) -> str:
    """Categorize based on result_summary field."""
    if not result_summary:
        return "logic"
    result_lower = result_summary.lower()
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(result_lower):
            return name
    return "logic"


# ── Pattern matching ─────────────────────────────────────────────────────────

def normalize_params_signature(params_summary: dict) -> str:
    """
    Create a normalized signature from params_summary for grouping.
    Uses tool name + key parameter patterns.
    """
    if not params_summary:
        return ""
    
    # For run_bash, extract command pattern
    if "command" in params_summary:
        cmd = params_summary["command"]
        if isinstance(cmd, str):
            # Normalize paths and specific values
            cmd = re.sub(r'/home/[^\s]+', '/home/USER', cmd)
            cmd = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', 'DATE', cmd)
            cmd = re.sub(r'[0-9]{2}:[0-9]{2}:[0-9]{2}', 'TIME', cmd)
            # Extract command type
            parts = cmd.split()
            if parts:
                return f"{parts[0]}_signature"
        return "run_bash_signature"
    
    # For file operations, extract operation + pattern
    if "path" in params_summary:
        path = str(params_summary["path"])
        # Extract file extension pattern
        ext_match = re.search(r'\.([a-zA-Z0-9]+)$', path)
        if ext_match:
            return f"file_{ext_match.group(1)}_signature"
        return "file_signature"
    
    if "pattern" in params_summary:
        return f"pattern_{str(params_summary['pattern'])[:20]}_signature"
    
    # Generic signature based on keys
    keys = sorted(params_summary.keys())
    return f"{'_'.join(keys)}_signature"


def slugify(text: str) -> str:
    """Convert text to URL-safe slug (no slashes for filenames)."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)  # Replace everything non-alphanumeric with -
    text = re.sub(r'-+', '-', text)
    text = text.strip('-')
    return text[:50]  # Limit length


# ── Data loading ─────────────────────────────────────────────────────────────

def load_trajectories(days: int = 7, agent_filter: str = "worker") -> list[dict]:
    """Load trajectory JSONL files with optional filters."""
    trajectories = []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    
    if not TRAJECTORY_DIR.exists():
        print(f"Warning: Trajectory directory not found: {TRAJECTORY_DIR}")
        return trajectories
    
    for jsonl_file in sorted(TRAJECTORY_DIR.glob("*.jsonl")):
        # Skip non-date files
        if not re.match(r'^\d{4}-\d{2}-\d{2}\.jsonl$', jsonl_file.name):
            continue
        
        # Check date filter
        try:
            file_date = datetime.strptime(jsonl_file.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                continue
        except ValueError:
            continue
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traj = json.loads(line)
                        # Apply agent filter
                        if agent_filter != "all":
                            if traj.get("agent_id") != agent_filter:
                                continue
                        trajectories.append(traj)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except Exception as e:
            print(f"Warning: Could not read {jsonl_file}: {e}")
            continue
    
    return trajectories


# ── Pattern mining ───────────────────────────────────────────────────────────

def mine_error_patterns(trajectories: list[dict], threshold: int = 2) -> list[dict]:
    """
    Mine error patterns from trajectories.
    Groups errors by (tool_name, error_category, params_signature).
    Returns patterns that appear in >= threshold distinct sessions.
    """
    # Structure: {(tool_name, error_category, params_sig): {session_keys: set, examples: list, dates: set}}
    pattern_data = defaultdict(lambda: {
        "sessions": set(),
        "examples": [],
        "dates": set(),
        "total_calls": 0
    })
    
    for traj in trajectories:
        session_key = traj.get("session_key", "unknown")
        timestamp = traj.get("timestamp", "")
        
        # Extract date from timestamp
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        
        # Process error_tools
        for error_tool in traj.get("error_tools", []):
            tool_name = error_tool.get("name", "unknown")
            error_type = error_tool.get("error_type", "logic")
            params_summary = error_tool.get("params_summary", {})
            params_sig = normalize_params_signature(params_summary) if params_summary else "generic"
            
            key = (tool_name, error_type, params_sig)
            pattern_data[key]["sessions"].add(session_key)
            pattern_data[key]["dates"].add(date_str)
            pattern_data[key]["total_calls"] += 1
            
            # Store example (limit per pattern)
            if len(pattern_data[key]["examples"]) < 5:
                example = {
                    "session_key": session_key,
                    "date": date_str,
                    "tool": tool_name,
                    "error_type": error_type,
                    "params_summary": params_summary,
                    "sequence": error_tool.get("sequence", 0)
                }
                pattern_data[key]["examples"].append(example)
    
    # Filter by threshold
    qualifying_patterns = []
    for key, data in pattern_data.items():
        if len(data["sessions"]) >= threshold:
            tool_name, error_type, params_sig = key
            qualifying_patterns.append({
                "type": "error",
                "tool_name": tool_name,
                "error_type": error_type,
                "params_signature": params_sig,
                "sessions": data["sessions"],
                "examples": data["examples"],
                "dates": data["dates"],
                "total_calls": data["total_calls"],
                "first_seen": min(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                "last_seen": max(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            })
    
    return qualifying_patterns


def mine_success_patterns(trajectories: list[dict], threshold: int = 2) -> list[dict]:
    """
    Mine successful tool sequences that repeat across sessions.
    Groups by (tool_name, params_signature) for non-error calls.
    Returns patterns that appear in >= threshold distinct sessions.
    """
    # Structure: {(tool_name, params_sig): {session_keys: set, examples: list, dates: set, total: int}}
    pattern_data = defaultdict(lambda: {
        "sessions": set(),
        "examples": [],
        "dates": set(),
        "total_calls": 0,
        "error_count": 0
    })
    
    for traj in trajectories:
        session_key = traj.get("session_key", "unknown")
        timestamp = traj.get("timestamp", "")
        
        # Extract date from timestamp
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        
        # Process all tools (both success and error)
        for tool in traj.get("tools", []):
            tool_name = tool.get("name", "unknown")
            is_error = tool.get("is_error", False)
            params_summary = tool.get("params_summary", {})
            params_sig = normalize_params_signature(params_summary) if params_summary else "generic"
            
            key = (tool_name, params_sig)
            pattern_data[key]["sessions"].add(session_key)
            pattern_data[key]["dates"].add(date_str)
            pattern_data[key]["total_calls"] += 1
            
            if is_error:
                pattern_data[key]["error_count"] += 1
            
            # Store example (limit per pattern)
            if len(pattern_data[key]["examples"]) < 3:
                example = {
                    "session_key": session_key,
                    "date": date_str,
                    "tool": tool_name,
                    "params_summary": params_summary,
                    "result_summary": tool.get("result_summary", ""),
                    "is_error": is_error,
                    "sequence": tool.get("sequence", 0)
                }
                pattern_data[key]["examples"].append(example)
    
    # Filter by threshold and focus on high-frequency patterns
    qualifying_patterns = []
    for key, data in pattern_data.items():
        if len(data["sessions"]) >= threshold and data["total_calls"] >= threshold:
            tool_name, params_sig = key
            error_rate = data["error_count"] / data["total_calls"] if data["total_calls"] > 0 else 0
            
            # Only include if it has some success rate (not 100% errors)
            if error_rate < 0.9:
                qualifying_patterns.append({
                    "type": "success",
                    "tool_name": tool_name,
                    "params_signature": params_sig,
                    "sessions": data["sessions"],
                    "examples": data["examples"],
                    "dates": data["dates"],
                    "total_calls": data["total_calls"],
                    "error_count": data["error_count"],
                    "error_rate": error_rate,
                    "first_seen": min(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                    "last_seen": max(data["dates"]) if data["dates"] else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                })
    
    # Sort by frequency
    qualifying_patterns.sort(key=lambda x: -x["total_calls"])
    return qualifying_patterns


# ── Output generation ────────────────────────────────────────────────────────

def generate_mitigation(tool_name: str, error_type: str) -> str:
    """Generate a suggested mitigation based on tool and error type."""
    mitigations = {
        ("run_bash", "permission"): "Check file ownership and permissions before executing commands. Use sudo with explicit path validation when elevated privileges are needed.",
        ("run_bash", "not_found"): "Verify file/directory paths exist before operations. Consider adding existence checks or using absolute paths.",
        ("run_bash", "timeout"): "Review command complexity and consider breaking into smaller operations. Add progress indicators for long-running tasks.",
        ("run_bash", "network"): "Check network connectivity and DNS resolution. Consider adding retry logic with exponential backoff.",
        ("run_bash", "validation"): "Validate input parameters before command execution. Use schema validation for complex arguments.",
        ("run_bash", "resource"): "Monitor system resources (disk, memory) before heavy operations. Consider cleanup strategies.",
        ("file_read", "permission"): "Check file ownership and ensure read permissions. Consider using absolute paths and verifying access before reading.",
        ("file_read", "not_found"): "Verify file existence before attempting to read. Consider graceful fallback for missing files.",
        ("file_write", "permission"): "Check directory write permissions and disk space. Consider using temp files with atomic moves.",
        ("file_write", "not_found"): "Ensure parent directories exist before writing. Use mkdir -p or equivalent for path creation.",
        ("http_fetch", "network"): "Add retry logic with exponential backoff. Consider timeout configurations and connection pooling.",
        ("http_fetch", "timeout"): "Increase timeout values for large responses. Consider streaming for large payloads.",
        ("http_request", "network"): "Verify endpoint availability and network configuration. Add health checks before requests.",
        ("http_request", "timeout"): "Review timeout settings based on expected response times. Consider async operations for long requests.",
    }
    
    key = (tool_name, error_type)
    if key in mitigations:
        return mitigations[key]
    
    # Default mitigations by error type
    default_by_type = {
        "permission": "Check file/directory permissions and ownership. Consider using appropriate privilege escalation when needed.",
        "not_found": "Verify paths and existence before operations. Add defensive checks for missing resources.",
        "timeout": "Review operation complexity and consider breaking into smaller steps. Add progress tracking.",
        "network": "Implement retry logic with exponential backoff. Add connection health checks.",
        "validation": "Add input validation before processing. Use schema validation for complex structures.",
        "resource": "Monitor system resources and implement cleanup strategies. Consider batch processing for large operations.",
        "logic": "Review error handling logic. Add more specific error checking and fallback behavior.",
    }
    
    return default_by_type.get(error_type, "Review error context and add appropriate handling for this scenario.")


def generate_title(pattern: dict) -> str:
    """Generate a descriptive title for a pattern."""
    if pattern["type"] == "error":
        tool_name = pattern["tool_name"]
        error_type = pattern["error_type"]
        return f"Handle {tool_name} {error_type} errors"
    else:
        tool_name = pattern["tool_name"]
        return f"Pattern: {tool_name} usage"


def write_candidate_file(pattern: dict, output_dir: Path) -> str:
    """Write a single candidate markdown file. Returns the file path."""
    # Generate filename
    if pattern["type"] == "error":
        pattern_slug = f"{pattern['tool_name']}/{pattern['error_type']}"
    else:
        pattern_slug = f"{pattern['tool_name']}/{pattern['params_signature']}"
    
    slug = slugify(pattern_slug)
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    filename = f"candidate-{slug}-{today}.md"
    filepath = output_dir / filename
    
    # Generate content
    title = generate_title(pattern)
    
    if pattern["type"] == "error":
        error_rate = pattern["total_calls"] / len(pattern["sessions"]) if pattern["sessions"] else 0
        content = f"""---
candidate: true
pattern: {pattern_slug}
type: error
occurrences: {pattern["total_calls"]}
sessions: {len(pattern["sessions"])}
first_seen: {pattern["first_seen"]}
last_seen: {pattern["last_seen"]}
error_rate: 1.0
status: pending_review
---

# Skill Candidate: {title}

## Pattern Summary
This pattern captures repeated {pattern['error_type']} errors when using the `{pattern['tool_name']}` tool. 
These errors occur across {len(pattern["sessions"])} distinct sessions, indicating a systematic issue worth addressing.

## Error Examples
"""
        for i, example in enumerate(pattern["examples"], 1):
            params = example.get("params_summary", {})
            params_str = str(params) if params else "N/A"
            if len(params_str) > 200:
                params_str = params_str[:200] + "..."
            content += f"""### Example {i} (session: {example["session_key"]}, {example["date"]})
- **Tool:** {example["tool"]}
- **Input:** `{params_str}`
- **Error Type:** {example["error_type"]}

"""
        
        content += f"""## Suggested Mitigation
{generate_mitigation(pattern['tool_name'], pattern['error_type'])}

## Sessions Affected
"""
        for session in sorted(pattern["sessions"]):
            content += f"- {session}\n"
    else:
        content = f"""---
candidate: true
pattern: {pattern_slug}
type: success
occurrences: {pattern["total_calls"]}
sessions: {len(pattern["sessions"])}
first_seen: {pattern["first_seen"]}
last_seen: {pattern["last_seen"]}
error_rate: {pattern["error_rate"]:.2f}
status: pending_review
---

# Skill Candidate: {title}

## Pattern Summary
This pattern represents a reusable procedural pattern: using `{pattern['tool_name']}` with consistent parameters across {len(pattern["sessions"])} distinct sessions.
This represents a candidate for skill encoding to improve efficiency and consistency.

## Usage Examples
"""
        for i, example in enumerate(pattern["examples"], 1):
            params = example.get("params_summary", {})
            params_str = str(params) if params else "N/A"
            if len(params_str) > 200:
                params_str = params_str[:200] + "..."
            status = "SUCCESS" if not example.get("is_error") else "ERROR"
            content += f"""### Example {i} (session: {example["session_key"]}, {example["date"]})
- **Tool:** {example["tool"]}
- **Status:** {status}
- **Input:** `{params_str}`
- **Result:** {example.get("result_summary", "N/A")[:50]}

"""
        
        content += f"""## Suggested Skill Encoding
This pattern should be encoded as a skill with:
- Pre-condition checks for required resources
- Standardized parameter handling
- Error recovery strategies

## Sessions Affected
"""
        for session in sorted(pattern["sessions"]):
            content += f"- {session}\n"
    
    # Write file
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


def write_index(candidates: list[str], output_dir: Path) -> None:
    """Write/update the INDEX.md file."""
    index_path = output_dir / "INDEX.md"
    
    content = """---
type: index
scope: skill-candidates
---

# Skill Candidates Index

This index tracks all skill candidates discovered through trajectory mining.
Generated by `mine-trajectories.py`.

## Summary

"""
    
    error_count = sum(1 for c in candidates if "error" in c)
    success_count = len(candidates) - error_count
    
    content += f"""- **Total candidates:** {len(candidates)}
- **Error patterns:** {error_count}
- **Success patterns:** {success_count}
- **Last updated:** {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Candidates

"""
    
    for candidate in sorted(candidates):
        filepath = output_dir / candidate
        if filepath.exists():
            # Try to extract frontmatter
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_lines = []
                    in_frontmatter = False
                    for line in f:
                        first_lines.append(line)
                        if line.strip() == '---' and len(first_lines) > 1:
                            if not in_frontmatter:
                                in_frontmatter = True
                            else:
                                break
                    # Parse frontmatter
                    frontmatter = ''.join(first_lines)
                    pattern_match = re.search(r'pattern:\s*(\S+)', frontmatter)
                    pattern = pattern_match.group(1) if pattern_match else "unknown"
                    sessions_match = re.search(r'sessions:\s*(\d+)', frontmatter)
                    sessions = sessions_match.group(1) if sessions_match else "?"
                    status_match = re.search(r'status:\s*(\S+)', frontmatter)
                    status = status_match.group(1) if status_match else "?"
                    
                    content += f"- [{candidate}]({candidate}) — Pattern: `{pattern}` — Sessions: {sessions} — Status: {status}\n"
            except Exception:
                content += f"- [{candidate}]({candidate}) — (could not parse)\n"
        else:
            content += f"- [{candidate}]({candidate}) — (file missing)\n"
    
    content += f"""

## Generation Commands

```bash
# Show statistics without writing files
python3 ~/obsidian/scripts/mine-trajectories.py --stats

# Generate candidates for last 7 days (worker agent only)
python3 ~/obsidian/scripts/mine-trajectories.py --days 7 --agent worker --threshold 2

# Generate for all agents
python3 ~/obsidian/scripts/mine-trajectories.py --days 7 --agent all --threshold 2

# Custom output directory
python3 ~/obsidian/scripts/mine-trajectories.py --days 7 --output-dir ~/custom/output/
```

"""
    
    index_path.write_text(content, encoding="utf-8")


# ── Statistics ───────────────────────────────────────────────────────────────

def print_stats(trajectories: list[dict]) -> None:
    """Print summary statistics from trajectory data."""
    if not trajectories:
        print("No trajectories found.")
        return
    
    total_tools = 0
    total_errors = 0
    agent_counts = defaultdict(int)
    error_type_counts = defaultdict(int)
    tool_name_counts = defaultdict(int)
    
    for traj in trajectories:
        total_tools += traj.get("tool_count", 0)
        total_errors += traj.get("error_count", 0)
        agent_counts[traj.get("agent_id", "unknown")] += 1
        
        for et in traj.get("error_tools", []):
            error_type_counts[et.get("error_type", "unknown")] += 1
        
        for tool in traj.get("tools", []):
            tool_name_counts[tool.get("name", "unknown")] += 1
    
    print("=" * 60)
    print("TRAJECTORY MINING STATS")
    print("=" * 60)
    print(f"  Trajectories loaded:  {len(trajectories)}")
    print(f"  Total tool calls:     {total_tools}")
    print(f"  Total errors:         {total_errors}")
    if total_tools > 0:
        print(f"  Error rate:           {total_errors / total_tools:.1%}")
    print()
    print("By agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent:<20} {count}")
    print()
    print("Top tools:")
    for name, count in sorted(tool_name_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<30} {count}")
    print()
    print("Error types:")
    for etype, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<20} {count}")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine trajectory JSONL files for skill candidates."
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Process trajectories from the last N days (default: 7)"
    )
    parser.add_argument(
        "--agent", type=str, default="worker",
        help="Filter by agent: worker, main, or all (default: worker)"
    )
    parser.add_argument(
        "--threshold", type=int, default=2,
        help="Minimum distinct sessions for a pattern to qualify (default: 2)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print summary statistics without writing files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: ~/obsidian/skills/candidates/)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectories
    print(f"Loading trajectories from last {args.days} days...", file=sys.stderr)
    trajectories = load_trajectories(days=args.days, agent_filter=args.agent)
    print(f"  Loaded {len(trajectories)} trajectory(ies)", file=sys.stderr)
    
    if args.stats:
        print_stats(trajectories)
        return
    
    # Mine patterns
    print("Mining error patterns...", file=sys.stderr)
    error_patterns = mine_error_patterns(trajectories, threshold=args.threshold)
    print(f"  Found {len(error_patterns)} qualifying error pattern(s)", file=sys.stderr)
    
    print("Mining success patterns...", file=sys.stderr)
    success_patterns = mine_success_patterns(trajectories, threshold=args.threshold)
    print(f"  Found {len(success_patterns)} qualifying success pattern(s)", file=sys.stderr)
    
    # Write candidates
    all_patterns = error_patterns + success_patterns
    candidate_files = []
    
    for pattern in all_patterns:
        filepath = write_candidate_file(pattern, output_dir)
        candidate_files.append(os.path.basename(filepath))
        print(f"  Written: {filepath}", file=sys.stderr)
    
    # Write index
    write_index(candidate_files, output_dir)
    print(f"  Updated: {output_dir / 'INDEX.md'}", file=sys.stderr)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Trajectories read:    {len(trajectories)}")
    print(f"  Error patterns:       {len(error_patterns)}")
    print(f"  Success patterns:     {len(success_patterns)}")
    print(f"  Candidates written:   {len(candidate_files)}")
    print(f"  Output directory:     {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
