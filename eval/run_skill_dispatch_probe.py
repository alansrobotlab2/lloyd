#!/usr/bin/env python3
"""Dispatch-level probe harness for #536 (dispatch-time skill delivery).

Replays drafted tool calls harvested from real session transcripts against the
rule set in `app/harness/skill_dispatch.py`, without running a model or a tool.
Two numbers per protocol, and they answer different questions:

1. **Trigger study** (measured here, deterministic): of the dispatches replayed,
   how many would be held, at what injected-token cost. `spurious_rate` is the
   acceptance metric verbatim — triggering dispatches over total dispatches.
2. **Protocol compliance before / after** — `before` is measured from the logs
   (what fraction of the protocol's dispatches actually took the step the
   SKILL.md prescribes); `after` is the *predicted* rate if every non-compliant
   dispatch had been held and re-issued informed. It is labelled predicted
   because the real delta needs the flag on and traffic through it: a rule that
   fires proves the protocol reached the model, not that the model obeyed.

Usage:
    .venvs/lloyd/bin/python eval/run_skill_dispatch_probe.py
    .venvs/lloyd/bin/python eval/run_skill_dispatch_probe.py --sessions 200 --json /tmp/probe.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.harness import skill_dispatch as sd  # noqa: E402

# NOT `app.paths.SESSIONS_DIR`: that resolves relative to the package, so from a
# self-mod worktree it points at the worktree's empty `sessions/` and the probe
# silently reports zero dispatches. Transcripts are a fact about the live box.
DEFAULT_SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
CHARS_PER_TOKEN = 4.0

# What the protocol requires, expressed as the argument shape that shows the
# step was taken. Each entry is measured over the dispatches its rule triggers
# on, so `before` is a rate among calls the deliverer would have held.
PROTOCOL_STEPS: dict[str, dict[str, re.Pattern[str]]] = {
    "youtube-transcript": {
        # HARD guardrail: never yt-dlp on this box (no Node runtime).
        "required": re.compile(r"youtube[-_]transcript|transcriptExtractor|\.vtt"),
        "violating": re.compile(r"\byt[-_]dlp\b"),
    },
    "restart-lloyd": {
        # supervisorctl against the process group, with the conf.
        "required": re.compile(r"supervisorctl[^\n]*lloyd-mc:"),
        "violating": re.compile(r"\bsystemctl\s+--user\s+(?:restart|stop|start)\s+\S*lloyd-mc\S*"),
    },
    "voice-mode": {
        # The service, never the script.
        "required": re.compile(r"\b(?:lloyd-voice-mode\.service|agent-livekit-server|agent-tts)\b"),
        "violating": re.compile(r"\bpython\S*[^\n]*\bvoice_mode\.py\b"),
    },
}


def harvest(sessions_dir: Path, limit_sessions: int) -> tuple[list[dict], int]:
    """Every tool call in the newest N session transcripts.

    Reads the persisted shape (`{function: {name, arguments: "<json>"}}`, what
    `compaction`/session IO writes), so the replay is over calls the model
    actually drafted with the arguments it actually sent.
    """
    files = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    scanned = 0
    calls: list[dict] = []
    for path in files[:limit_sessions]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        scanned += 1
        for msg in data.get("messages") or []:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or ""
                args = fn.get("arguments") or tc.get("args_dict") or tc.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"__raw__": args}
                if isinstance(args, dict):
                    calls.append({"session": path.name, "name": name, "args": args})
    return calls, scanned


def score(calls: list[dict], *, scanned_sessions: int) -> dict:
    per: dict[str, dict] = defaultdict(lambda: {
        "dispatched": 0, "triggered": 0, "compliant": 0, "violating": 0,
        "unlabelled": 0, "injected_chars": 0, "examples": [],
    })
    total_dispatched = 0
    total_triggered = 0
    for call in calls:
        total_dispatched += 1
        rule = sd.match_rule(call["name"], call["args"], rules=sd.DISPATCH_RULES)
        if rule is None:
            continue
        total_triggered += 1
        row = per[rule.skill]
        row["dispatched"] += 1
        row["triggered"] += 1
        step = PROTOCOL_STEPS.get(rule.skill, {})
        text = "\n".join(v for v in call["args"].values() if isinstance(v, str))
        viol, req = step.get("violating"), step.get("required")
        if viol and viol.search(text):
            row["violating"] += 1
        elif req and req.search(text):
            row["compliant"] += 1
        else:
            row["unlabelled"] += 1
        if len(row["examples"]) < 4:
            row["examples"].append(text.replace("\n", " ")[:160])
        body = sd.skill_body(rule.skill)
        row["injected_chars"] += min(len(body), sd.MAX_DELIVERY_CHARS)

    out: dict[str, dict] = {}
    for skill, row in per.items():
        labelled = row["compliant"] + row["violating"]
        before = round(100.0 * row["compliant"] / labelled, 1) if labelled else None
        # Predicted: every violating dispatch is held and re-issued informed.
        after = round(100.0 * (row["compliant"] + row["violating"]) / labelled, 1) if labelled else None
        out[skill] = {
            **{k: v for k, v in row.items() if k != "examples"},
            "examples": row["examples"],
            "compliance_before_pct": before,
            "compliance_after_predicted_pct": after,
            "avg_injected_tokens_per_event": round(row["injected_chars"] / max(1, row["triggered"]) / CHARS_PER_TOKEN, 1),
        }
    for skill, spec in PROTOCOL_STEPS.items():
        out.setdefault(skill, {
            "dispatched": 0, "triggered": 0, "compliant": 0, "violating": 0,
            "unlabelled": 0, "injected_chars": 0, "examples": [],
            "compliance_before_pct": None, "compliance_after_predicted_pct": None,
            "avg_injected_tokens_per_event": 0,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions_scanned": scanned_sessions,
        "dispatches_replayed": total_dispatched,
        "dispatches_triggered": total_triggered,
        "spurious_rate_pct": round(100.0 * total_triggered / max(1, total_dispatched), 2),
        "per_protocol": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=400, help="Newest N session transcripts")
    ap.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR),
                    help="Directory of session JSON transcripts (default: the live box's ~/lloyd/sessions)")
    ap.add_argument("--json", default="", help="Also write the report here")
    args = ap.parse_args()

    calls, scanned = harvest(Path(args.sessions_dir).expanduser(), args.sessions)
    report = score(calls, scanned_sessions=scanned)

    print(f"Sessions scanned:        {report['sessions_scanned']}")
    print(f"Dispatches replayed:     {report['dispatches_replayed']}")
    print(f"Dispatches triggered:    {report['dispatches_triggered']}")
    print(f"Spurious rate (trigger / all dispatches) = {report['spurious_rate_pct']} %   <-- guard, target < 10 %")
    print()
    hdr = f"{'protocol':<20} {'held':>5} {'ok':>4} {'viol':>5} {'?':>4} {'before%':>8} {'after*%':>8} {'tok/evt':>8}"
    print(hdr)
    print("-" * len(hdr))
    for skill, row in sorted(report["per_protocol"].items()):
        before = row["compliance_before_pct"]
        after = row["compliance_after_predicted_pct"]
        cells = [
            f"{skill:<20}",
            f"{row['triggered']:>5}",
            f"{row['compliant']:>4}",
            f"{row['violating']:>5}",
            f"{row['unlabelled']:>4}",
            f"{'n/a' if before is None else before:>8}",
            f"{'n/a' if after is None else after:>8}",
            f"{row['avg_injected_tokens_per_event']:>8}",
        ]
        print(" ".join(cells))
    print("\n* after is PREDICTED (every held-and-non-compliant dispatch re-issued "
          "informed). The measured delta needs the flag on; see the item's soak.")
    for skill, row in report["per_protocol"].items():
        if row["examples"]:
            print(f"\n{skill} examples:")
            for ex in row["examples"]:
                print(f"  - {ex}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\n[written] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
