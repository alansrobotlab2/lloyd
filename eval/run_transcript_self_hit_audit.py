#!/usr/bin/env python3
"""#1511 — how often does a prompt retrieve its own transcript, and does the fix hold?

`_export_session_markdown` writes every chat into qmd's `sessions` collection,
and prefetch's vault leg searches it. A prompt that has been sent before
therefore retrieves the transcript holding it — prompt, tool results and
answer — and a re-run "passes" from its own last answer. Two modes:

``audit`` (no model; the item's clause 3, the denominator)
    The FIRST user message of every conversation session in ``--sessions-dir``
    (newest first, up to ``--limit``) goes through the vault leg as deployed on a
    first turn — lex ladder, hybrid, `_fuse_fresh_vault` — twice: without the
    #1511 filter and with it (`prefetch._drop_self_transcripts`, the session's
    own id passed, so its own export counts as ``own_session``). Per prompt it
    records whether the top hit is a `sessions/` transcript at all, whether it
    is a self-hit, and whether ANY self-hit survived. Probe prompts ("E2E harness
    check") are counted apart from the rest, which is the denominator the item
    asked for. Replaying a historical prompt is the worst case on purpose: its
    own transcript is in the index, exactly as a regression re-run's would be.

``probe`` (primary model, `flock -s` on primary.lock; clauses 1 and 2)
    Each distinct probe prompt is rendered through `prefetch_context` (no
    session: an independent first turn) and asserted to carry no transcript that
    echoes it, then run to its FIRST tool call on the production system prompt
    and tool surface — the call is not dispatched (see
    `run_tool_choice_eval._first_tool_call`). The pass condition is the tool
    call — a `Read` (or any file tool) naming the file the prompt asks for —
    never the answer text, which a self-hit can supply.

Usage:
    .venvs/lloyd/bin/python eval/run_transcript_self_hit_audit.py audit \\
        --sessions-dir ~/lloyd-data/sessions --limit 300
    flock -s -w 7200 ~/.local/state/lloyd-automod/primary.lock \\
        .venvs/lloyd/bin/python eval/run_transcript_self_hit_audit.py probe \\
        --sessions-dir ~/lloyd-data/sessions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

from run_prefetch_eval import _quiet_logging  # noqa: E402

with _quiet_logging():
    import prefetch  # noqa: E402
    from agent_mcp.transcript_self_hit import is_transcript, self_hit_reason  # noqa: E402

PROBE_MARK = "E2E harness check"
_INJECTED = ("<context>", "<system-reminder>", "<memory>", "<daily_notes>",
             "[cron:", "[System Message]", "[autonomy:")


def _msg_text(msg: dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return c if isinstance(c, str) else ""


def first_prompts(sessions_dir: Path, limit: int, all_turns: bool = False) -> list[dict]:
    """(session_id, user message) for the newest conversation sessions: the first
    user message of each, or with `all_turns` every one of them (up to `limit`)."""
    from app.sessions_io import is_conversation_session

    out: list[dict] = []
    for f in sorted(sessions_dir.glob("*.json"), reverse=True):
        if len(out) >= limit:
            break
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not is_conversation_session(f.stem, data):
            continue
        for m in data.get("messages", []):
            if m.get("role") != "user":
                continue
            t = _msg_text(m).strip()
            if not t or t.startswith(_INJECTED) or len(t) < prefetch.MIN_MESSAGE_LEN:
                continue
            out.append({"session_id": data.get("session_id") or f.stem, "prompt": t})
            if not all_turns or len(out) >= limit:
                break
    return out


def vault_hits(prompt: str) -> list[dict]:
    """The deployed first-turn vault leg: lex ladder + hybrid, fused."""
    focus = prefetch.SessionFocus()
    focus.update(prompt)
    lex = prefetch._search_vault_lex(prompt, focus, deadline=None)
    hyb = prefetch._search_vault_hybrid_and_stash(prompt, focus)
    return prefetch._fuse_fresh_vault(lex, hyb)


def classify(prompt: str, hits: list[dict], session_id: str | None) -> dict:
    top = hits[0] if hits else None
    reasons = [self_hit_reason(prompt, h, session_id) for h in hits]
    return {
        "n_hits": len(hits),
        "top_file": (top or {}).get("file"),
        "top_is_transcript": bool(top and is_transcript(str(top.get("file") or ""))),
        "top_self_hit": reasons[0] if reasons else None,
        "any_self_hit": any(reasons),
        "self_hits": [r for r in reasons if r],
    }


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    from stats import wilson_ci
    lo, hi = wilson_ci(k, n)
    return [round(lo, 3), round(hi, 3)]


def _rate(rows: list[dict], arm: str, key: str) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r[arm][key])
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": _wilson(k, n)}


def audit(args) -> dict:
    prompts = first_prompts(Path(args.sessions_dir).expanduser(), args.limit,
                            all_turns=args.all_turns)
    rows = []
    t0 = time.monotonic()
    for p in prompts:
        hits = vault_hits(p["prompt"])
        filtered = prefetch._drop_self_transcripts(p["prompt"], hits, p["session_id"])
        rows.append({
            **p, "prompt": p["prompt"][:300],
            "probe": PROBE_MARK in p["prompt"],
            "before": classify(p["prompt"], hits, p["session_id"]),
            "after": classify(p["prompt"], filtered, p["session_id"]),
        })
    groups = {"all": rows, "probe": [r for r in rows if r["probe"]],
              "non_probe": [r for r in rows if not r["probe"]]}
    summary = {}
    for g, rs in groups.items():
        summary[g] = {arm: {k: _rate(rs, arm, k) for k in
                            ("top_is_transcript", "top_self_hit", "any_self_hit")}
                      for arm in ("before", "after")}
        own = sum(1 for r in rs if r["before"]["top_self_hit"] == "own_session")
        verb = sum(1 for r in rs if r["before"]["top_self_hit"] == "verbatim")
        summary[g]["before_top_self_hit_by_reason"] = {"own_session": own, "verbatim": verb}
    return {"mode": "audit", "elapsed_s": round(time.monotonic() - t0, 1),
            "summary": summary, "records": rows}


_PATH_RE = re.compile(r"(/[\w./-]+\.\w+)")


async def probe(args) -> dict:
    import run_tool_choice_eval as tce
    import yaml as _yaml
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from prompt_builder import build_system_prompt

    # Every probe session, repeats included: the item asks for each existing
    # probe run to be re-run, and identical prompts are the A/A of the check.
    prompts = [p for p in first_prompts(Path(args.sessions_dir).expanduser(), 10_000)
               if PROBE_MARK in p["prompt"]]
    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    alias = (config.get("model") or {}).get("default", "primary")
    mcfg = (config.get("models") or {}).get(alias) or {}
    base_url = (mcfg.get("env") or {}).get("ANTHROPIC_BASE_URL") or mcfg.get("base_url")
    system_prompt = build_system_prompt()
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    rows = []
    for p in prompts:
        prompt = p["prompt"]
        rendered = {}
        for arm, on in (("before", False), ("after", True)):
            rendered[arm] = prefetch_render(prompt, on)
        options = RunOptions(model=alias, base_url=base_url, system_prompt=system_prompt,
                             max_turns=1, mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
                             disallowed_tools=_get_disallowed_tools(), priority=3,
                             hooks=hooks, **_get_harness_kwargs())
        outcome = await tce._first_tool_call(rendered["after"], options, args.timeout)
        first = outcome["first_tool"] or {}
        wanted = _PATH_RE.findall(prompt)
        named = json.dumps(first.get("args") or {})
        rows.append({
            "session_id": p["session_id"], "prompt": prompt[:300],
            "echo_before": _echoes(prompt, rendered["before"]),
            "echo_after": _echoes(prompt, rendered["after"]),
            "first_tool": first.get("name"),
            "tool_names_target": bool(wanted and any(w in named for w in wanted)),
            "pass": bool(first.get("name")) and (not wanted or any(w in named for w in wanted)),
            "error": outcome["error"],
        })
        print(f"  {'PASS' if rows[-1]['pass'] else 'FAIL'} echo {rows[-1]['echo_before']}->"
              f"{rows[-1]['echo_after']} first_tool={rows[-1]['first_tool']}")
    return {"mode": "probe", "n": len(rows),
            "clean_after": sum(1 for r in rows if not r["echo_after"]),
            "echo_before": sum(1 for r in rows if r["echo_before"]),
            "passed": sum(1 for r in rows if r["pass"]), "records": rows}


def prefetch_render(prompt: str, exclude: bool) -> str:
    """`prefetch_context` for an independent first turn, with the filter on or off."""
    real = prefetch.exclude_self_transcripts_enabled
    prefetch.exclude_self_transcripts_enabled = lambda: exclude
    try:
        return prefetch.prefetch_context(prompt, session_id=None, plan_mode=False)
    finally:
        prefetch.exclude_self_transcripts_enabled = real


def _echoes(prompt: str, rendered: str) -> bool:
    """Does the rendered `<vault-context>` carry a transcript that echoes `prompt`?"""
    m = re.search(r"<vault-context>(.*?)</vault-context>", rendered or "", re.S)
    if not m:
        return False
    for line in m.group(1).splitlines():
        f = re.search(r"file: (sessions/[^,)\s]+)", line)
        if f and self_hit_reason(prompt, {"file": f.group(1), "snippet": line}, None):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("audit", "probe"))
    ap.add_argument("--sessions-dir", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--all-turns", action="store_true",
                    help="audit every user message, not only each session's first")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with _quiet_logging():
        result = audit(args) if args.mode == "audit" else asyncio.run(probe(args))
    result["ran_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(text)
        print(f"[info] wrote {args.out}")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
