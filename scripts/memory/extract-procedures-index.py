#!/usr/bin/env python3
"""
Compact procedures index with failure→success detection.
Identifies retry chains, corrections, and non-obvious fixes.
Target: <20KB for LLM analysis.

Usage:
    python3 extract-procedures-index.py --date 2026-03-08
    python3 extract-procedures-index.py --hours 24
"""

import json, re, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

AGENTS_DIR = Path.home() / ".openclaw" / "state" / "agents"
CC_LOGS_DIR = Path.home() / ".openclaw" / "state" / "logs" / "cc-instances"
PST = ZoneInfo("America/Los_Angeles")
DEFAULT_OUTDIR = Path.home() / "obsidian" / "memory" / "skill-maintenance"

ACTION_TOOLS = {
    "cc_orchestrate", "cc_spawn", "mem_write", "file_write", "file_edit",
    "file_patch", "cron", "backlog_create_task", "backlog_update_task",
    "message", "tts", "email_send",
}

# Signals that indicate something failed or didn't work
# Signals that Alan is correcting Lloyd's behavior/approach
CORRECTION_PATTERNS_USER = [
    r"you screwed up", r"bad lloyd", r"don't do that",
    r"that's not what", r"that's wrong", r"I wanted.*not",
    r"why did you", r"I said", r"I asked for",
    r"you shouldn't", r"never do that", r"stop doing",
    r"what happened to", r"you broke", r"you deleted",
    r"after you.*(?:broke|deleted|screwed|messed|wiped|removed|accidentally)",
    r"overzealously", r"accidentally",
]

# Signals that Alan wants to preserve/remember a procedure
REMEMBER_PATTERNS_USER = [
    r"remember (?:how|this|that|when)", r"for (?:next|future) time",
    r"make (?:this |it )?a skill", r"write (?:this|that) down",
    r"let's make (?:a )?note", r"make a note",
    r"always (?:do|use|check|run)", r"never (?:do|use|skip)",
    r"standard (?:procedure|process|workflow)",
    r"going forward", r"from now on",
]

FAILURE_PATTERNS_USER = [
    r"\bnope\b", r"\bno\b(?!\w)", r"\bnot working\b", r"\bstill broken\b",
    r"\bdoesn'?t work\b", r"\bdidn'?t work\b", r"\btry again\b",
    r"\bwrong\b", r"\bstill not\b", r"\bnot hearing\b", r"\bnot seeing\b",
    r"\bno audio\b", r"\bno tts\b", r"\bno response\b", 
    r"\bbad lloyd\b", 
]

FAILURE_PATTERNS_RESULT = [
    r"error", r"Error", r"FAIL", r"failed", r"exit 1", r"exit code [^0]",
    r"Permission denied", r"No such file", r"not found", r"timed out",
    r"timeout", r"refused", r"ECONNREFUSED", r"ENOENT", r"cannot",
]

FAILURE_PATTERNS_ASSISTANT = [
    r"that didn'?t work", r"the (?:issue|problem|bug) (?:is|was)",
    r"failed", r"broke", r"wrong", r"the real issue",
    r"the root cause", r"actually caused by", r"workaround",
    r"not the (?:right|correct)", r"let me try a different",
    r"abandoned", r"doesn'?t work", r"can'?t ", r"timed out",
    r"the fix (?:is|was)", r"the actual fix", r"the solution",
    r"the missing piece", r"found it", r"there it is",
    r"that was the", r"here'?s what actually",
]

SUCCESS_PATTERNS_ASSISTANT = [
    r"(?:that |it )?(?:works?|worked)\b", r"\bfixed\b", r"\bresolved\b",
    r"\bdone\b", r"\bcompleted?\b", r"\bverified\b", r"\bconfirmed\b",
    r"all (?:good|set|checks? out)", r"\bclean\b", r"\bsolid\b",
    r"\bready\b", r"✅", r"\bsuccess",
]


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
    s = s.strip(); return s[:n]+"..." if len(s) > n else s

def matches_any(text, patterns):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

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
    return t.strip()

def extract_text(content):
    if not isinstance(content, list): return ""
    return "\n".join(b.get("text","").strip() for b in content if isinstance(b,dict) and b.get("type")=="text" and b.get("text","").strip())

def first_sentence(text):
    t = re.sub(r'<summary>.*?</summary>', '', text, flags=re.DOTALL).strip()
    if not t:
        m = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
        if m: t = m.group(1).strip()
    if not t: return ""
    for end in ['. ', '.\n', '! ', '!\n']:
        idx = t.find(end)
        if 0 < idx < 150: return t[:idx+1]
    return trunc(t, 120)

def get_tools(content):
    if not isinstance(content, list): return []
    tools = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "toolCall":
            name = b.get("name", "?")
            args = b.get("arguments", {})
            is_action = name in ACTION_TOOLS
            if name == "run_bash":
                cmd = str(args.get("command", ""))
                if any(w in cmd for w in ["sed -i", "echo >", "cat >", "tee ", "mkdir", "mv ", "cp ", "rm ",
                                           "git commit", "git merge", "git checkout", "npm run build",
                                           "systemctl", "pip install", "npm install",
                                           "curl -X POST", "curl -X PUT", "fuser -k"]):
                    is_action = True
            tools.append({"name": name, "action": is_action})
    return tools

def get_last_ts(filepath):
    last = 0
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue
                if obj.get("type") == "message" and obj.get("message",{}).get("role","") in ("user","assistant"):
                    ts = ts_to_unix(obj.get("timestamp",""))
                    if ts > last: last = ts
    except: pass
    return last


def extract_procedures(filepath):
    """Extract procedures with failure/success signal detection."""
    procedures = []
    session_id = None

    req = None; tools = []; approach = None; outcome = None
    user_signals = []; result_signals = []; assistant_signals = []
    has_failure = False; has_fix = False; has_success = False; has_correction = False; has_remember = False

    def classify():
        """Determine the signal type for this procedure."""
        if has_correction:
            return "correction"
        if has_remember:
            return "remember"
        if has_failure and has_fix:
            return "failure→fix"
        if has_failure and has_success:
            return "failure→success"
        if has_failure:
            return "failure"
        if has_fix:
            return "fix"
        return "routine"

    def extract_key_findings():
        """Pull out the most informative failure/fix text."""
        findings = []
        for sig in assistant_signals:
            # Look for "the issue/problem/fix/root cause" sentences
            for pattern in [r"the (?:issue|problem|bug|root cause|missing piece|fix|actual fix|solution) (?:is|was)[^.!]*[.!]",
                          r"(?:found it|there it is)[^.!]*[.!]",
                          r"workaround[^.!]*[.!]"]:
                m = re.search(pattern, sig, re.IGNORECASE)
                if m:
                    finding = trunc(m.group(0), 150)
                    if len(finding) > 20:
                        findings.append(finding)
            for pattern in [r"can'?t (?:run )?two [^.!]*[.!]",
                          r"(?:timed out|timeout)[^.!]*[.!]"]:
                m = re.search(pattern, sig, re.IGNORECASE)
                if m:
                    finding = trunc(m.group(0), 150)
                    if len(finding) > 20:
                        findings.append(finding)
        # Deduplicate
        seen = set()
        unique = []
        for f in findings:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique[:3]  # Max 3 findings

    def flush():
        nonlocal has_failure, has_fix, has_success, has_correction, has_remember
        if req and len(tools) >= 3:
            unique = list(dict.fromkeys(t["name"] for t in tools))
            has_actions = any(t["action"] and t["name"] != "run_bash" for t in tools)
            signal = classify()

            # Include if: has action tools OR has interesting signals
            if has_actions or signal in ("failure→fix", "failure→success", "correction", "remember"):
                findings = extract_key_findings()
                procedures.append({
                    "goal": trunc(req, 100),
                    "n": len(tools),
                    "tools": unique,
                    "approach": trunc(approach or "", 100),
                    "outcome": trunc(outcome or "", 100),
                    "signal": signal,
                    "findings": findings,
                    "user_neg": bool(user_signals),
                    "is_correction": has_correction,
                    "is_remember": has_remember,
                })
        has_failure = False; has_fix = False; has_success = False; has_correction = False; has_remember = False; has_correction = False; has_remember = False

    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except: continue
                if obj.get("type") == "session": session_id = obj.get("id","?"); continue
                if obj.get("type") != "message": continue
                msg = obj["message"]; role = msg.get("role","")
                content = msg.get("content",[])

                if role == "user":
                    flush()
                    text = clean_user_text(extract_text(content))
                    if not text or len(text) < 2: continue
                    req = text; tools = []; approach = None; outcome = None
                    user_signals = []; result_signals = []; assistant_signals = []
                    if matches_any(text, FAILURE_PATTERNS_USER):
                        user_signals.append(text[:100])
                        has_failure = True
                    if matches_any(text, CORRECTION_PATTERNS_USER):
                        user_signals.append('CORRECTION: ' + text[:100])
                        has_correction = True
                    if matches_any(text, REMEMBER_PATTERNS_USER):
                        user_signals.append('REMEMBER: ' + text[:100])
                        has_remember = True

                elif role == "assistant":
                    tools.extend(get_tools(content))
                    text = extract_text(content)
                    if text and len(text) > 15:
                        s = first_sentence(text)
                        if not approach and s: approach = s
                        if s: outcome = s
                        # Check for failure/fix signals
                        if matches_any(text, FAILURE_PATTERNS_ASSISTANT):
                            assistant_signals.append(text[:300])
                            if any(re.search(p, text, re.IGNORECASE) for p in [
                                r"the fix", r"the actual fix", r"the solution", r"the missing piece",
                                r"found it", r"there it is", r"workaround", r"here'?s what actually"
                            ]):
                                has_fix = True
                            else:
                                has_failure = True
                        if matches_any(text, SUCCESS_PATTERNS_ASSISTANT):
                            has_success = True

                elif role == "toolResult":
                    text = extract_text(content)
                    if text and matches_any(text, FAILURE_PATTERNS_RESULT):
                        result_signals.append(text[:100])
                        has_failure = True

        flush()
    except: pass
    return session_id, procedures


def group_into_arcs(procedures):
    """Group consecutive procedures that appear to be retries of the same goal."""
    if not procedures:
        return procedures

    arcs = []
    current_arc = [procedures[0]]

    for proc in procedures[1:]:
        prev = current_arc[-1]
        # Same goal text or user expressing failure/retry
        is_retry = proc.get("user_neg", False)
        if is_retry and len(current_arc) < 5:
            current_arc.append(proc)
        else:
            arcs.append(current_arc)
            current_arc = [proc]

    arcs.append(current_arc)
    return arcs


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

    all_by_session = {}
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        sdir = agent_dir / "sessions"
        if not sdir.is_dir(): continue
        agent = agent_dir.name
        if agent == "memory": continue
        for fp in sorted(sdir.glob("*.jsonl")):
            last = get_last_ts(fp)
            if last == 0 or last < cutoff_ts or last >= end_ts: continue
            sid, procs = extract_procedures(fp)
            if not procs: continue
            sf = f"{agent}--{sid[:12]}.md"
            all_by_session[sf] = procs

    # CC instances
    instances = []
    if CC_LOGS_DIR.is_dir():
        for f in sorted(CC_LOGS_DIR.glob("*.summary.json")):
            try:
                mt = f.stat().st_mtime
                if mt < cutoff_ts or (end_ts and mt > end_ts): continue
                d = json.loads(f.read_text())
                cost = d.get('costUsd', 0)
                task = d.get('task', '') or ''
                # Skip echo tests and trivial runs
                if cost < 0.10 or 'echo test' in task.lower():
                    continue
                instances.append(f"[{d.get('id','?')}] {d.get('pipeline','?')} ${cost:.2f}: {trunc(task, 80)}")
            except: continue

    # Count signals
    total_procs = sum(len(v) for v in all_by_session.values())
    signal_counts = {}
    findings_count = 0
    for procs in all_by_session.values():
        for p in procs:
            signal_counts[p["signal"]] = signal_counts.get(p["signal"], 0) + 1
            findings_count += len(p.get("findings", []))

    # Build output — interesting stuff first
    lines = [
        f"# Procedures Index — {date_label}",
        f"",
        f"{total_procs} actionable procedures | {len(instances)} orchestrator runs",
        f"Signals: {', '.join(f'{v} {k}' for k,v in sorted(signal_counts.items(), key=lambda x: -x[1]))}",
        f"Key findings extracted: {findings_count}",
        f"",
        f"For full transcript, read the session file in this directory.",
        f"",
    ]

    # Section 1: Failure→fix and failure→success (the gems)
    has_interesting = False
    for sf, procs in sorted(all_by_session.items()):
        arcs = group_into_arcs(procs)
        interesting_arcs = [a for a in arcs if any(p["signal"] in ("failure→fix", "correction", "remember") for p in a)]
        if not interesting_arcs: continue

        if not has_interesting:
            lines.append("## 🔍 Skill Candidates (failures, corrections, remember-requests)")
            lines.append("")
            has_interesting = True

        lines.append(f"### {sf}")
        for arc in interesting_arcs:
            if len(arc) == 1:
                p = arc[0]
                emoji = {"failure→fix": "🔧", "correction": "⚠️", "remember": "📌"}.get(p["signal"], "🔍")
                lines.append(f"{emoji} [{p['signal']}] [{p['n']} calls] {p['goal']}")
                lines.append(f"  Tools: {', '.join(p['tools'])}")
                if p["approach"]: lines.append(f"  → {p['approach']}")
                if p["outcome"] and p["outcome"] != p["approach"]: lines.append(f"  ✓ {p['outcome']}")
                for f in p.get("findings", []):
                    lines.append(f"  💡 {f}")
            else:
                # Multi-step arc
                lines.append(f"[RETRY CHAIN: {len(arc)} attempts]")
                for i, p in enumerate(arc):
                    label = "FINAL" if i == len(arc)-1 and p["signal"] in ("failure→fix","failure→success") else f"attempt {i+1}"
                    lines.append(f"  {label} ({p['signal']}): {p['goal']}")
                    lines.append(f"    Tools: {', '.join(p['tools'])}")
                    if p["approach"]: lines.append(f"    → {p['approach']}")
                    if p["outcome"] and p["outcome"] != p["approach"]: lines.append(f"    ✓ {p['outcome']}")
                    for f in p.get("findings", []):
                        lines.append(f"    💡 {f}")
        lines.append("")

    if not has_interesting:
        lines.append("## 🔍 Interesting Patterns")
        lines.append("(none detected today)")
        lines.append("")

    # Section 2: Routine procedures (condensed to one line each)
    lines.append("## Routine Procedures")
    lines.append("")
    for sf, procs in sorted(all_by_session.items()):
        routine = [p for p in procs if p["signal"] in ("routine", "failure→success", "failure")]
        if not routine: continue
        for p in routine:
            tools = ', '.join(p['tools'][:4])
            lines.append(f"- ({sf}) [{p['n']} calls] {p['goal'][:80]} | {tools}")
    lines.append("")

    # Section 3: Orchestrator runs (one-liners)
    if instances:
        lines.append("## Orchestrator Runs")
        for inst in instances: lines.append(f"- {inst}")
        lines.append("")

    content = "\n".join(lines)
    idx_path = session_dir / "procedures-index.md"
    idx_path.write_text(content, encoding="utf-8")

    size_kb = len(content.encode()) / 1024
    print(f"procedures-index.md: {size_kb:.1f} KB")
    print(f"  {total_procs} procedures ({', '.join(f'{v} {k}' for k,v in sorted(signal_counts.items(), key=lambda x: -x[1]))})")
    print(f"  {findings_count} key findings extracted")
    print(f"  {len(instances)} orchestrator runs")


if __name__ == "__main__":
    main()
