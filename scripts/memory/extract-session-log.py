#!/usr/bin/env python3
"""
Extract session logs — enhanced version with full tool call details.
Captures tool calls, responses, success/failure status.

Usage:
    python3 extract-session-log-enhanced.py --date 2026-03-08
    python3 extract-session-log-enhanced.py --hours 24
"""

import json, re, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

AGENTS_DIR = Path.home() / ".openclaw" / "agents"
PST = ZoneInfo("America/Los_Angeles")
DEFAULT_OUTDIR = Path.home() / "obsidian" / "sessions"


def parse_args():
    hours = None; target_date = None; outdir = DEFAULT_OUTDIR
    args = sys.argv[1:]; i = 0
    while i < len(args):
        if args[i] == "--hours" and i+1 < len(args): hours = int(args[i+1]); i += 2
        elif args[i] == "--date" and i+1 < len(args): target_date = args[i+1]; i += 2
        elif args[i] == "--outdir" and i+1 < len(args): outdir = Path(args[i+1]); i += 2
        else: i += 1
    if not hours and not target_date: hours = 24
    return hours, target_date, outdir


def ts_to_unix(ts_str):
    if not ts_str: return 0
    try: return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except: return 0

def trunc(s, n):
    if not s: return ""
    s = s.strip().replace('\n', ' ').replace('\r', '')
    return s[:n] + "..." if len(s) > n else s

def clean_user_text(text):
    t = text
    for tag in ["<daily_notes>", "<memory_context>"]:
        end_tag = tag.replace("<", "</")
        if tag in t:
            idx = t.find(end_tag)
            if idx >= 0: t = t[idx+len(end_tag):].strip()
    t = re.sub(r'<active_mode>.*?</active_mode>\s*', '', t).strip()
    if "```json" in t and "```\n" in t:
        parts = t.split("```\n", 1)
        if len(parts) > 1: t = parts[1].strip()
    m = re.search(r'\[.*?(?:PDT|PST|UTC)\]\s*(.*)', t, re.DOTALL)
    if m: t = m.group(1).strip()
    for skip in ["System:", "[System", "[cron:", "Execute your Session Startup", "Read HEARTBEAT.md"]:
        if skip in t[:100]: return ""
    return t.strip().replace('\n', ' ')

def extract_text(content):
    if not isinstance(content, list): return ""
    return "\n".join(b.get("text","").strip() for b in content if isinstance(b,dict) and b.get("type")=="text" and b.get("text","").strip())

def get_lloyd_text(text):
    t = re.sub(r'<summary>.*?</summary>\s*', '', text, flags=re.DOTALL).strip()
    if not t:
        m = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
        if m: t = m.group(1).strip()
    return t.replace('\n', ' ')

def process_session(filepath):
    lines = []
    session_id = None
    session_ts = None
    last_was_lloyd = False
    
    # First pass: collect all tool calls and results
    tool_calls = {}  # call_id -> {name, args}
    tool_results = {}  # call_id -> {content, error}
    
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue

                if obj.get("type") == "session":
                    session_id = obj.get("id", "?")
                    session_ts = obj.get("timestamp", "")
                    continue

                if obj.get("type") != "message": continue
                msg = obj["message"]; role = msg.get("role","")
                content = msg.get("content", [])

                if role == "toolCall":
                    call_id = msg.get("id", "")
                    name = msg.get("toolName", "?")
                    args = msg.get("arguments", {})
                    tool_calls[call_id] = {"name": name, "args": args}

                elif role == "toolResult":
                    call_id = msg.get("toolCallId", "")
                    content_data = msg.get("content", [])
                    is_error = msg.get("isError", False)
                    tool_results[call_id] = {"content": content_data, "error": is_error}

    except: pass

    # Second pass: output in order
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue

                if obj.get("type") != "message": continue
                msg = obj["message"]; role = msg.get("role","")
                content = msg.get("content", [])
                msg_ts = obj.get("timestamp", session_ts or "")

                if role == "user":
                    text = clean_user_text(extract_text(content))
                    if not text or len(text) < 2: continue
                    if last_was_lloyd:
                        lines.append("")
                    lines.append(f"user: {trunc(text, 300)}")
                    last_was_lloyd = False

                elif role == "assistant":
                    # Collect tool calls from this message
                    msg_tool_calls = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "toolCall":
                            call_id = b.get("id", "")
                            name = b.get("name", "?")
                            args = b.get("arguments", {})
                            msg_tool_calls.append({"id": call_id, "name": name, "args": args})
                    
                    # Output tool calls with results
                    for tc in msg_tool_calls:
                        call_id = tc["id"]
                        name = tc["name"]
                        args = tc["args"]
                        
                        # Format arguments (full, not truncated)
                        arg_parts = []
                        for k, v in args.items():
                            if isinstance(v, str):
                                arg_parts.append(f"{k}={v[:200] if len(v) > 200 else v}")
                            elif isinstance(v, bool):
                                arg_parts.append(f"{k}={v}")
                            elif isinstance(v, (int, float)):
                                arg_parts.append(f"{k}={v}")
                            elif isinstance(v, dict):
                                arg_parts.append(f"{k}={{...}}")
                            elif isinstance(v, list):
                                arg_parts.append(f"{k}=[...]")
                        
                        tool_line = f"tool_call: {name}({', '.join(arg_parts)})"
                        lines.append(tool_line)
                        
                        # Get result if available
                        if call_id in tool_results:
                            result_data = tool_results[call_id]
                            result_text = extract_text(result_data["content"])
                            is_error = result_data["error"]
                            status = "ERROR" if is_error else "OK"
                            result_preview = trunc(result_text, 500) if result_text else "(empty)"
                            lines.append(f"  → [{status}] {result_preview}")
                        else:
                            lines.append(f"  → [pending]")

                    # Output assistant text
                    text = extract_text(content)
                    if text and len(text) > 10:
                        resp = get_lloyd_text(text)
                        if resp and resp not in ("NO_REPLY", "HEARTBEAT_OK"):
                            lines.append(f"lloyd: {trunc(resp, 400)}")
                            last_was_lloyd = True

    except: pass

    return session_id, session_ts, lines


def get_last_ts(filepath):
    last = 0
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue
                if obj.get("type") == "message" and obj.get("message",{}).get("role","") in ("user","assistant","toolCall","toolResult"):
                    ts = ts_to_unix(obj.get("timestamp",""))
                    if ts > last: last = ts
    except: pass
    return last


def main():
    hours, target_date, outdir = parse_args()
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = dt.replace(tzinfo=PST)
        cutoff_ts = start.timestamp(); end_ts = (start + timedelta(days=1)).timestamp()
        date_label = target_date
    else:
        now = datetime.now(timezone.utc).timestamp()
        cutoff_ts = now - (hours * 3600); end_ts = now
        date_label = datetime.fromtimestamp(cutoff_ts, PST).strftime("%Y-%m-%d")

    outdir.mkdir(parents=True, exist_ok=True)
    session_dir = outdir / date_label
    session_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        sdir = agent_dir / "sessions"
        if not sdir.is_dir(): continue
        agent = agent_dir.name
        if agent == "memory": continue

        for fp in sorted(sdir.glob("*.jsonl")):
            last = get_last_ts(fp)
            if last == 0 or last < cutoff_ts or last >= end_ts: continue
            sid, sts, lines = process_session(fp)
            if not lines: continue

            header = f"# {agent}/{sid}\n# {sts}\n\n"
            content = header + "\n".join(lines) + "\n"
            filename = f"{agent}--{sid[:12]}.md"
            (session_dir / filename).write_text(content, encoding="utf-8")
            written.append((filename, len([l for l in lines if l.startswith("user:")]), len(content)))

    total_size = sum(s for _, _, s in written)
    print(f"Wrote {len(written)} logs to {session_dir}/")
    print(f"Total: {total_size / 1024:.1f} KB")
    for fname, ints, size in sorted(written):
        print(f"  {fname} ({ints} interactions, {size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
