#!/usr/bin/env python3
"""
Extract dense session summaries from OpenClaw session JSONL files.
Includes tool calls, results, assistant reasoning, and actions.

Assigns sessions to a date based on the LAST interaction timestamp.

Usage:
    python3 extract-tool-sequences.py --date 2026-03-08
    python3 extract-tool-sequences.py --hours 24
    python3 extract-tool-sequences.py --date 2026-03-08 --outdir ~/obsidian/memory/skill-maintenance

Output: One file per session + orchestrator summary file.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

AGENTS_DIR = Path.home() / ".openclaw" / "state" / "agents"
CC_LOGS_DIR = Path.home() / ".openclaw" / "state" / "logs" / "cc-instances"
PST = ZoneInfo("America/Los_Angeles")
DEFAULT_OUTDIR = Path.home() / "obsidian" / "memory" / "skill-maintenance"


def parse_args():
    hours = None
    target_date = None
    outdir = DEFAULT_OUTDIR
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--hours" and i + 1 < len(args):
            hours = int(args[i + 1]); i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            target_date = args[i + 1]; i += 2
        elif args[i] == "--outdir" and i + 1 < len(args):
            outdir = Path(args[i + 1]); i += 2
        else:
            i += 1
    if not hours and not target_date:
        hours = 24
    return hours, target_date, outdir


def ts_to_unix(ts_str):
    if not ts_str: return 0
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except: return 0


def format_time(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(PST).strftime("%H:%M %Z")
    except: return "??"


def truncate(s, n):
    s = s.strip()
    if len(s) <= n: return s
    return s[:n] + "..."


def extract_text(content):
    if not isinstance(content, list): return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").strip()
            if t: parts.append(t)
    return "\n".join(parts)


def extract_tool_calls(content):
    if not isinstance(content, list): return []
    tools = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            name = block.get("name", "unknown")
            args = block.get("arguments", {})
            # Compact args
            compact = {}
            for k, v in args.items():
                if isinstance(v, str) and len(v) > 150:
                    compact[k] = v[:150] + "..."
                elif isinstance(v, dict):
                    compact[k] = truncate(json.dumps(v, default=str), 150)
                else:
                    compact[k] = v
            tools.append({"name": name, "args": compact})
    return tools


def extract_tool_results(content):
    if not isinstance(content, list): return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            results.append(block.get("text", "").strip())
    return results


def get_last_interaction_ts(filepath):
    """Get the timestamp of the last user or assistant message in a session file."""
    last_ts = 0
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except: continue
                if obj.get("type") == "message":
                    msg = obj.get("message", {})
                    role = msg.get("role", "")
                    if role in ("user", "assistant"):
                        ts = ts_to_unix(obj.get("timestamp", ""))
                        if ts > last_ts:
                            last_ts = ts
    except: pass
    return last_ts


def process_session(filepath):
    """Process a session JSONL into a dense summary."""
    lines_out = []
    session_id = None
    session_ts = None
    interaction_count = 0
    tool_call_count = 0

    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue

                if obj.get("type") == "session":
                    session_id = obj.get("id", "unknown")
                    session_ts = obj.get("timestamp", "")
                    continue

                if obj.get("type") != "message": continue
                msg = obj["message"]
                role = msg.get("role", "")
                content = msg.get("content", [])
                msg_ts = obj.get("timestamp", "")
                time_str = format_time(msg_ts)

                if role == "user":
                    text = extract_text(content)
                    if not text: continue
                    # Skip system/cron injections
                    stripped = text.strip()
                    if stripped.startswith("[cron:") or stripped.startswith("[System"):
                        continue
                    # Strip the sender metadata block but keep the actual message
                    # Find the actual user text after metadata
                    user_msg = text
                    if "```json" in text and "```\n\n" in text:
                        parts = text.split("```\n\n", 1)
                        if len(parts) > 1:
                            user_msg = parts[1].strip()
                    # Also strip daily_notes blocks
                    if "<daily_notes>" in user_msg:
                        idx = user_msg.find("</daily_notes>")
                        if idx >= 0:
                            user_msg = user_msg[idx+14:].strip()
                    # Strip active_mode tags
                    if "<active_mode>" in user_msg:
                        import re
                        user_msg = re.sub(r'<active_mode>.*?</active_mode>\s*', '', user_msg).strip()
                    # Strip remaining metadata headers
                    if user_msg.startswith("Sender (untrusted"):
                        lines = user_msg.split("\n")
                        for j, l in enumerate(lines):
                            if l.startswith("[") and "PDT]" in l or "PST]" in l:
                                user_msg = l.split("]", 1)[1].strip() if "]" in l else ""
                                if j + 1 < len(lines):
                                    user_msg += "\n" + "\n".join(lines[j+1:])
                                break

                    if not user_msg or len(user_msg) < 2: continue
                    interaction_count += 1
                    lines_out.append(f"\n--- [{time_str}] USER ---")
                    lines_out.append(truncate(user_msg, 500))

                elif role == "assistant":
                    tools = extract_tool_calls(content)
                    text = extract_text(content)

                    if tools:
                        tool_call_count += len(tools)
                        for t in tools:
                            args_str = json.dumps(t["args"], default=str)
                            lines_out.append(f"  → {t['name']}({truncate(args_str, 200)})")

                    if text and len(text) > 15:
                        # Skip NO_REPLY and HEARTBEAT_OK
                        if text.strip() in ("NO_REPLY", "HEARTBEAT_OK"):
                            lines_out.append(f"  [{text.strip()}]")
                        else:
                            lines_out.append(f"  ASSISTANT: {truncate(text, 600)}")

                elif role == "toolResult":
                    results = extract_tool_results(content)
                    for r in results:
                        if len(r) > 10:
                            lines_out.append(f"  ← RESULT: {truncate(r, 300)}")

    except (OSError, IOError) as e:
        lines_out.append(f"ERROR reading session: {e}")

    return session_id, session_ts, interaction_count, tool_call_count, lines_out


def process_cc_instances(cutoff_ts, end_ts):
    """Read CC instance summaries from the time window."""
    instances = []
    if not CC_LOGS_DIR.is_dir(): return instances

    for f in sorted(CC_LOGS_DIR.glob("*.summary.json")):
        try:
            mtime = f.stat().st_mtime
            if mtime < cutoff_ts: continue
            if end_ts and mtime > end_ts: continue
            data = json.loads(f.read_text())
            instances.append({
                "id": data.get("id", "?"),
                "type": data.get("type", "?"),
                "task": (data.get("task", "") or "")[:800],
                "result": (data.get("resultPreview", "") or "")[:800],
                "pipeline": data.get("pipeline", "?"),
                "turns": data.get("turns", 0),
                "cost": data.get("costUsd", 0),
                "elapsed_min": round((data.get("elapsedMs", 0) or 0) / 60000, 1),
                "status": data.get("status", "?"),
            })
        except: continue

    return instances


def main():
    hours, target_date, outdir = parse_args()

    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = dt.replace(tzinfo=PST)
        cutoff_ts = start.timestamp()
        end_ts = (start + timedelta(days=1)).timestamp()
        date_label = target_date
    else:
        now = datetime.now(timezone.utc).timestamp()
        cutoff_ts = now - (hours * 3600)
        end_ts = now
        date_label = datetime.fromtimestamp(cutoff_ts, PST).strftime("%Y-%m-%d")

    outdir.mkdir(parents=True, exist_ok=True)
    session_dir = outdir / date_label
    session_dir.mkdir(parents=True, exist_ok=True)

    written = []

    # Process all agent session dirs
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.is_dir(): continue
        agent_name = agent_dir.name

        for filepath in sorted(sessions_dir.glob("*.jsonl")):
            # Assign to date based on LAST interaction
            last_ts = get_last_interaction_ts(filepath)
            if last_ts == 0: continue
            if last_ts < cutoff_ts or last_ts >= end_ts: continue

            sid, sts, interactions, tools, lines = process_session(filepath)
            if interactions == 0: continue

            # Build session file
            header = [
                f"# Session: {agent_name}/{sid}",
                f"# Started: {sts}",
                f"# Agent: {agent_name}",
                f"# Interactions: {interactions}",
                f"# Tool calls: {tools}",
                "",
            ]

            content = "\n".join(header + lines)
            filename = f"{agent_name}--{sid[:12]}.md"
            out_path = session_dir / filename
            out_path.write_text(content, encoding="utf-8")
            written.append((filename, interactions, tools))

    # Orchestrator runs
    instances = process_cc_instances(cutoff_ts, end_ts)
    if instances:
        orch_lines = [
            f"# Orchestrator Runs — {date_label}",
            f"# Total: {len(instances)}",
            f"# Total cost: ${sum(i['cost'] for i in instances):.2f}",
            "",
        ]
        for inst in instances:
            orch_lines.append(f"## [{inst['id']}] {inst['type']} | {inst['pipeline']} | {inst['status']}")
            orch_lines.append(f"- Turns: {inst['turns']} | Cost: ${inst['cost']:.2f} | Time: {inst['elapsed_min']}min")
            orch_lines.append(f"- **Task:** {inst['task'][:600]}")
            if inst['result']:
                orch_lines.append(f"- **Result:** {inst['result'][:600]}")
            orch_lines.append("")

        orch_path = session_dir / "orchestrator-runs.md"
        orch_path.write_text("\n".join(orch_lines), encoding="utf-8")
        written.append(("orchestrator-runs.md", len(instances), 0))

    # Summary index
    index_lines = [
        f"# Session Extracts — {date_label}",
        f"# Sessions: {len([w for w in written if w[0] != 'orchestrator-runs.md'])}",
        f"# Orchestrator runs: {len(instances)}",
        "",
        "| File | Interactions | Tool Calls |",
        "|------|-------------|------------|",
    ]
    for fname, ints, tools in sorted(written):
        index_lines.append(f"| {fname} | {ints} | {tools} |")

    index_path = session_dir / "index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    # Print summary to stdout
    print(f"Wrote {len(written)} files to {session_dir}/")
    for fname, ints, tools in sorted(written):
        print(f"  {fname} ({ints} interactions, {tools} tool calls)")


if __name__ == "__main__":
    main()
