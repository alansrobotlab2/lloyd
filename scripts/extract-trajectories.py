#!/usr/bin/env python3
"""
extract-trajectories.py — Parse session files into structured trajectory logs.

Reads Lloyd session JSON files from ~/lloyd/sessions/*.json

Use --agent worker to process only autonomy sessions (autonomy_*.json)
Use --agent main  to process only interactive sessions (non-autonomy *.json)

Output: ~/lloyd/_pipeline/trajectories/YYYY-MM-DD.jsonl
State:  ~/lloyd/_pipeline/trajectories/.watermark.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

LLOYD_SESSIONS = Path.home() / "lloyd" / "sessions"
OUTPUT_DIR = Path.home() / "lloyd" / "_pipeline" / "trajectories"
WATERMARK_PATH = OUTPUT_DIR / ".watermark.json"

# ── Sensitive data patterns ───────────────────────────────────────────────────

SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|apikey|secret|token|password|passwd|auth)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{36}", re.IGNORECASE),   # GitHub PAT
    re.compile(r"xoxb-[A-Za-z0-9\-]+", re.IGNORECASE),  # Slack bot token
]

MAX_STRING_LEN = 500
MAX_FILE_CONTENT_LEN = 200
MAX_ERROR_LEN = 200

# Keys whose values are likely to contain file content (truncate aggressively)
CONTENT_ARG_KEYS = {"content", "text", "body", "data", "message"}


# ── Error categorization ──────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    ("permission",  re.compile(r"permission denied|access denied|forbidden|EPERM|EACCES", re.IGNORECASE)),
    ("not_found",   re.compile(r"file not found|no such file|not found|404|ENOENT", re.IGNORECASE)),
    ("timeout",     re.compile(r"timeout|timed out|ETIMEDOUT|deadline exceeded", re.IGNORECASE)),
    ("network",     re.compile(r"connection refused|ECONNREFUSED|DNS|ENOTFOUND|network|EHOSTUNREACH", re.IGNORECASE)),
    ("validation",  re.compile(r"invalid|malformed|parse error|syntax error|schema|validation", re.IGNORECASE)),
    ("resource",    re.compile(r"out of memory|disk full|quota|ENOMEM|ENOSPC|resource exhausted", re.IGNORECASE)),
]


# Semantic error patterns — detect errors in result content even when isError=False
SEMANTIC_ERROR_PATTERNS = [
    # Non-zero exit codes from run_bash
    re.compile(r"\bexit\s+(?!0\b)([1-9]\d*)\b", re.IGNORECASE),
    # Python exceptions
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"(?:Error|Exception|Warning):\s+\S", re.IGNORECASE),
    # Shell errors
    re.compile(r"(?:command not found|No such file or directory|Permission denied|Operation not permitted)", re.IGNORECASE),
    re.compile(r"bash: \S+: ", re.IGNORECASE),
    # Node/npm errors
    re.compile(r"(?:npm ERR!|SyntaxError:|ReferenceError:|TypeError:)", re.IGNORECASE),
    # Common failure keywords in result text (conservative — require context)
    re.compile(r"^FAILED\b", re.MULTILINE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
]


def has_semantic_error(content_text: str) -> bool:
    """Return True if result content suggests an error even without isError=True."""
    for pat in SEMANTIC_ERROR_PATTERNS:
        if pat.search(content_text):
            return True
    return False


def categorize_error(text: str) -> str:
    for name, pattern in ERROR_CATEGORIES:
        if pattern.search(text):
            return name
    return "logic"


# ── Data scrubbing ────────────────────────────────────────────────────────────

def mask_sensitive(value: str) -> str:
    """Replace sensitive tokens in a string with [MASKED]."""
    for pat in SENSITIVE_PATTERNS:
        value = pat.sub("[MASKED]", value)
    return value


def scrub_value(key: str, value) -> object:
    """Scrub a single argument value."""
    if not isinstance(value, str):
        return value

    # Mask sensitive patterns first
    value = mask_sensitive(value)

    # File content keys — truncate aggressively
    if key.lower() in CONTENT_ARG_KEYS and len(value) > MAX_FILE_CONTENT_LEN:
        return value[:MAX_FILE_CONTENT_LEN] + f" [truncated: {len(value)} chars]"

    # Large strings — truncate
    if len(value) > MAX_STRING_LEN:
        return f"[truncated: {len(value)} chars]"

    return value


def scrub_params(arguments: dict) -> dict:
    """Produce a sanitized params_summary dict from tool arguments."""
    if not isinstance(arguments, dict):
        return {}
    return {k: scrub_value(k, v) for k, v in arguments.items()}


def extract_result_text(content) -> str:
    """Pull plain text from a toolResult content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def result_summary(content, is_error: bool) -> str:
    text = extract_result_text(content)
    if is_error:
        preview = text[:MAX_ERROR_LEN].replace("\n", " ").strip()
        return f"ERROR: {preview}"
    return f"OK: {len(text)} chars"


# ── Signal extraction ─────────────────────────────────────────────────────────

SIGNAL_RE = re.compile(r"SIGNAL:([A-Z_]+)")


# ── Session parsing ──────────────────────────────────────────────────────────

def parse_session(path: Path) -> dict | None:
    """
    Parse a single Hermes session JSON file.

    Hermes format: single JSON object with top-level messages array.
    Messages use role "user"/"assistant"/"tool" with tool_calls on assistant
    messages and tool_call_id on tool messages.

    Returns a trajectory dict or None if the session has no tool calls.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    session_id = data.get("session_id", path.stem)
    session_ts = data.get("session_start", "") or data.get("created_at", "")
    messages = data.get("messages", [])

    # Detect agent_id from source path
    agent_id = "autonomy" if path.stem.startswith("autonomy_") else "lloyd"

    # Build call_id → tool_call map from assistant messages
    call_map: dict[str, dict] = {}
    call_sequence: dict[str, int] = {}
    sequence_counter = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            call_id = tc.get("id") or tc.get("call_id", "")
            if not call_id:
                continue
            # Normalize to shared structure
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"_raw": raw_args[:MAX_STRING_LEN]}
            call_map[call_id] = {
                "name": func.get("name", "unknown"),
                "arguments": raw_args,
            }
            call_sequence[call_id] = sequence_counter
            sequence_counter += 1

    # Build call_id → result map from tool messages
    result_map: dict[str, dict] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id", "")
        if call_id:
            content = msg.get("content", "")
            # Normalize content to string (lloyd sessions use [{type,text}] blocks)
            content_text = extract_result_text(content)
            is_error = False
            try:
                parsed = json.loads(content_text)
                if isinstance(parsed, dict) and parsed.get("error"):
                    is_error = True
            except (json.JSONDecodeError, TypeError):
                pass
            result_map[call_id] = {
                "content": content_text,
                "isError": is_error,
            }

    if not call_map:
        return None

    # Build unified tool list sorted by sequence
    tools: list[dict] = []
    error_tools: list[dict] = []

    for call_id, call in sorted(call_map.items(), key=lambda x: call_sequence[x[0]]):
        name = call["name"]
        raw_args = call["arguments"]
        params = scrub_params(raw_args) if isinstance(raw_args, dict) else {}

        result = result_map.get(call_id)
        if result is not None:
            protocol_error = bool(result.get("isError", False))
            content_text = extract_result_text(result.get("content"))
            semantic_err = has_semantic_error(content_text) if not protocol_error else False
            is_error = protocol_error or semantic_err
            res_summary = result_summary(result.get("content"), is_error)
            error_source = "protocol" if protocol_error else ("semantic" if semantic_err else None)
        else:
            is_error = False
            res_summary = "OK: no result recorded"
            error_source = None
            content_text = ""

        seq = call_sequence[call_id]
        entry = {
            "name": name,
            "params_summary": params,
            "result_summary": res_summary,
            "is_error": is_error,
            "error_source": error_source,
            "call_id": call_id,
            "sequence": seq,
        }
        tools.append(entry)

        if is_error:
            error_tools.append({
                "name": name,
                "sequence": seq,
                "error_type": categorize_error(content_text),
                "error_source": error_source,
                "params_summary": params,
            })

    error_count = len(error_tools)

    # Use file mtime as fallback timestamp
    if not session_ts:
        mtime = os.path.getmtime(path)
        session_ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    # Extract signals from all message text
    signals: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        content = msg.get("content", "")
        text = extract_result_text(content)
        for match in SIGNAL_RE.finditer(text):
            sig = match.group(1)
            if sig not in seen:
                signals.append(sig)
                seen.add(sig)

    return {
        "session_key": session_id,
        "agent_id": agent_id,
        "timestamp": session_ts,
        "tool_count": len(tools),
        "error_count": error_count,
        "has_errors": error_count > 0,
        "tools": tools,
        "error_tools": error_tools,
        "signals": signals,
    }


# ── Watermark state ───────────────────────────────────────────────────────────

def load_watermark() -> dict:
    if WATERMARK_PATH.exists():
        try:
            return json.loads(WATERMARK_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "last_run": None,
        "sessions_processed": 0,
        "last_session_mtime": None,
    }


def save_watermark(state: dict) -> None:
    WATERMARK_PATH.write_text(json.dumps(state, indent=2))


# ── Session discovery ─────────────────────────────────────────────────────────

def discover_sessions(agent_filter: str | None = None) -> list[Path]:
    """Return sorted list of session JSON paths from ~/lloyd/sessions/.

    agent_filter:
      'worker' / 'autonomy' → only autonomy_*.json files
      'main'   / 'lloyd'    → only non-autonomy *.json files
      None                  → all sessions
    """
    paths: list[Path] = []
    if LLOYD_SESSIONS.exists():
        for p in LLOYD_SESSIONS.glob("*.json"):
            is_autonomy = p.stem.startswith("autonomy_")
            if agent_filter in ("worker", "autonomy"):
                if is_autonomy:
                    paths.append(p)
            elif agent_filter in ("main", "lloyd"):
                if not is_autonomy:
                    paths.append(p)
            else:
                paths.append(p)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def filter_by_mtime(paths: list[Path], since_mtime: float | None) -> list[Path]:
    if since_mtime is None:
        return paths
    return [p for p in paths if p.stat().st_mtime > since_mtime]


def filter_by_days(paths: list[Path], days: int) -> list[Path]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    return [p for p in paths if p.stat().st_mtime >= cutoff_ts]


# ── Output writing ────────────────────────────────────────────────────────────

def trajectory_date_key(traj: dict) -> str:
    """Return YYYY-MM-DD from trajectory timestamp, defaulting to today."""
    ts = traj.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def append_trajectories(trajectories: list[dict]) -> None:
    """Append trajectories to date-bucketed output files."""
    # Group by date
    by_date: dict[str, list[dict]] = {}
    for traj in trajectories:
        date_key = trajectory_date_key(traj)
        by_date.setdefault(date_key, []).append(traj)

    for date_key, items in by_date.items():
        out_path = OUTPUT_DIR / f"{date_key}.jsonl"
        with open(out_path, "a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def rewrite_trajectories(trajectories: list[dict]) -> None:
    """Write trajectories to date-bucketed output files (overwrite mode)."""
    # Group by date
    by_date: dict[str, list[dict]] = {}
    for traj in trajectories:
        date_key = trajectory_date_key(traj)
        by_date.setdefault(date_key, []).append(traj)

    # Remove existing output files that will be rewritten
    existing = list(OUTPUT_DIR.glob("*.jsonl"))
    for p in existing:
        if p.name != ".watermark.json":
            p.unlink(missing_ok=True)

    for date_key, items in by_date.items():
        out_path = OUTPUT_DIR / f"{date_key}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    """Print summary statistics from existing trajectory files."""
    files = sorted(OUTPUT_DIR.glob("*.jsonl"))
    if not files:
        print("No trajectory files found.")
        return

    total_sessions = 0
    total_tools = 0
    total_errors = 0
    agent_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}
    error_source_counts: dict[str, int] = {}
    tool_name_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    traj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                total_sessions += 1
                total_tools += traj.get("tool_count", 0)
                total_errors += traj.get("error_count", 0)

                agent = traj.get("agent_id", "unknown")
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

                for et in traj.get("error_tools", []):
                    etype = et.get("error_type", "unknown")
                    error_type_counts[etype] = error_type_counts.get(etype, 0) + 1
                    source = et.get("error_source")
                    if source:
                        error_source_counts[source] = error_source_counts.get(source, 0) + 1

                for tool in traj.get("tools", []):
                    tname = tool.get("name", "unknown")
                    tool_name_counts[tname] = tool_name_counts.get(tname, 0) + 1
                    source = tool.get("error_source")
                    if source:
                        error_source_counts[source] = error_source_counts.get(source, 0) + 1

                for sig in traj.get("signals", []):
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1

    watermark = load_watermark()

    print("=" * 60)
    print("TRAJECTORY STATS")
    print("=" * 60)
    print(f"  Output files:       {len(files)}")
    print(f"  Total sessions:     {total_sessions}")
    print(f"  Total tool calls:   {total_tools}")
    print(f"  Total errors:       {total_errors}")
    if total_tools > 0:
        print(f"  Error rate:         {total_errors / total_tools:.1%}")
    print()
    print("By agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent:<20} {count}")
    print()
    print("Top tools:")
    for name, count in sorted(tool_name_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<30} {count}")
    print()
    print("Error types:")
    for etype, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<20} {count}")
    print()
    print("Error sources:")
    for src, count in sorted(error_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<20} {count}")
    print()
    print("Signals seen:")
    for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {sig:<30} {count}")
    print()
    print("Watermark:")
    print(f"  last_run:            {watermark.get('last_run', 'never')}")
    print(f"  sessions_processed:  {watermark.get('sessions_processed', 0)}")
    print(f"  last_session_mtime:  {watermark.get('last_session_mtime', 'none')}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trajectory logs from Lloyd session JSON files."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Reprocess all sessions (ignore watermark)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Process only sessions modified in the last N days"
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help="Filter to a specific agent (e.g. worker, main)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print summary statistics from existing trajectory files"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_stats()
        return

    watermark = load_watermark()

    # Discover sessions
    all_sessions = discover_sessions(agent_filter=args.agent)

    # Determine filter
    if args.full:
        sessions_to_process = all_sessions
    elif args.days is not None:
        sessions_to_process = filter_by_days(all_sessions, args.days)
    else:
        # Incremental: only sessions newer than watermark
        last_mtime_str = watermark.get("last_session_mtime")
        if last_mtime_str:
            try:
                last_mtime_dt = datetime.fromisoformat(last_mtime_str.replace("Z", "+00:00"))
                last_mtime_ts = last_mtime_dt.timestamp()
                sessions_to_process = filter_by_mtime(all_sessions, last_mtime_ts)
            except (ValueError, AttributeError):
                sessions_to_process = all_sessions
        else:
            sessions_to_process = all_sessions

    print(f"Processing {len(sessions_to_process)} session(s)...", file=sys.stderr)

    trajectories: list[dict] = []
    max_mtime: float = 0.0
    skipped = 0
    failed = 0

    for path in sessions_to_process:
        try:
            traj = parse_session(path)
            if traj is None:
                skipped += 1
                continue
            trajectories.append(traj)
            mtime = path.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        except Exception as exc:
            print(f"  ERROR parsing {path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"  Parsed: {len(trajectories)}  Skipped (no tools): {skipped}  Failed: {failed}", file=sys.stderr)

    # Write output
    if trajectories:
        if args.full:
            rewrite_trajectories(trajectories)
        else:
            append_trajectories(trajectories)

    # Update watermark
    now = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    new_mtime_str = None
    if max_mtime > 0:
        new_mtime_str = datetime.fromtimestamp(max_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    updated_watermark = {
        "last_run": now,
        "sessions_processed": watermark.get("sessions_processed", 0) + len(trajectories),
        "last_session_mtime": new_mtime_str or watermark.get("last_session_mtime"),
    }
    save_watermark(updated_watermark)

    print(f"  Written to: {OUTPUT_DIR}", file=sys.stderr)
    print(f"  Watermark updated: {now}", file=sys.stderr)


if __name__ == "__main__":
    main()
